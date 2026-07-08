const std = @import("std");
const series_mod = @import("series.zig");
const tensor_mod = @import("tensor.zig");

pub const DataError = series_mod.DataError;
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

    pub fn toTensor(self: DataFrame, comptime T: type, names: []const []const u8) (DataError || tensor_mod.TensorError)!tensor_mod.Tensor(T) {
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
        return tensor_mod.Tensor(T).fromSlice(self.allocator, values, &.{ self.rows, names.len });
    }

    pub fn readCsv(allocator: std.mem.Allocator, text: []const u8, has_header: bool) DataError!DataFrame {
        var lines = std.mem.splitScalar(u8, text, '\n');
        const first_raw = lines.next() orelse return error.InvalidCsv;
        const first = std.mem.trim(u8, first_raw, "\r ");
        if (first.len == 0) return error.InvalidCsv;
        var headers_list: std.ArrayList([]const u8) = .empty;
        defer {
            for (headers_list.items) |h| allocator.free(h);
            headers_list.deinit(allocator);
        }

        var first_values: ?[][]const u8 = null;
        defer if (first_values) |vals| {
            for (vals) |v| allocator.free(v);
            allocator.free(vals);
        };

        if (has_header) {
            try splitCsvLineOwned(allocator, first, &headers_list);
        } else {
            var vals_list: std.ArrayList([]const u8) = .empty;
            defer vals_list.deinit(allocator);
            try splitCsvLineOwned(allocator, first, &vals_list);
            first_values = try vals_list.toOwnedSlice(allocator);
            for (0..first_values.?.len) |i| {
                var aw: std.Io.Writer.Allocating = .init(allocator);
                defer aw.deinit();
                try aw.writer.print("col{}", .{i});
                try headers_list.append(allocator, try aw.toOwnedSlice());
            }
        }

        const cols = headers_list.items.len;
        var string_cols = try allocator.alloc(std.ArrayList([]const u8), cols);
        defer allocator.free(string_cols);
        for (string_cols) |*list| list.* = .empty;
        defer {
            for (string_cols) |*list| {
                for (list.items) |cell| allocator.free(cell);
                list.deinit(allocator);
            }
        }

        if (first_values) |vals| {
            if (vals.len != cols) return error.InvalidCsv;
            for (vals, 0..) |v, i| try string_cols[i].append(allocator, try allocator.dupe(u8, v));
        }
        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, "\r ");
            if (line.len == 0) continue;
            var vals_list: std.ArrayList([]const u8) = .empty;
            defer {
                for (vals_list.items) |v| allocator.free(v);
                vals_list.deinit(allocator);
            }
            try splitCsvLineOwned(allocator, line, &vals_list);
            if (vals_list.items.len != cols) return error.InvalidCsv;
            for (vals_list.items, 0..) |v, i| try string_cols[i].append(allocator, try allocator.dupe(u8, v));
        }

        var defs = try allocator.alloc(ColumnDef, cols);
        defer allocator.free(defs);
        for (headers_list.items, string_cols, 0..) |header, cells, i| {
            defs[i] = .{ .name = header, .data = try inferColumn(allocator, cells.items) };
        }
        defer {
            for (defs) |def| freeColumn(allocator, def.data);
        }
        return DataFrame.init(allocator, defs);
    }

    pub fn writeCsv(self: DataFrame, allocator: std.mem.Allocator) DataError![]u8 {
        var aw: std.Io.Writer.Allocating = .init(allocator);
        errdefer aw.deinit();
        for (self.names, 0..) |name, i| {
            if (i != 0) try aw.writer.print(",", .{});
            try aw.writer.print("{s}", .{name});
        }
        try aw.writer.print("\n", .{});
        for (0..self.rows) |r| {
            for (self.columns, 0..) |col, c| {
                if (c != 0) try aw.writer.print(",", .{});
                try printCell(&aw.writer, col, r);
            }
            try aw.writer.print("\n", .{});
        }
        return aw.toOwnedSlice();
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
        if (self.rows > limit) try writer.print("...\n", .{});
    }
};

fn cloneColumn(allocator: std.mem.Allocator, col: Column) DataError!Column {
    return switch (col) {
        .f64 => |v| .{ .f64 = try allocator.dupe(f64, v) },
        .i64 => |v| .{ .i64 = try allocator.dupe(i64, v) },
        .bool => |v| .{ .bool = try allocator.dupe(bool, v) },
        .string => |v| blk: {
            var out = try allocator.alloc([]const u8, v.len);
            errdefer allocator.free(out);
            var initialized: usize = 0;
            errdefer {
                for (out[0..initialized]) |s| allocator.free(s);
            }
            for (v, 0..) |s, i| {
                out[i] = try allocator.dupe(u8, s);
                initialized += 1;
            }
            break :blk .{ .string = out };
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

fn filterColumn(allocator: std.mem.Allocator, col: Column, mask: []const bool) DataError!Column {
    var count: usize = 0;
    for (mask) |keep| {
        if (keep) count += 1;
    }
    return switch (col) {
        .f64 => |v| blk: {
            var out = try allocator.alloc(f64, count);
            var w: usize = 0;
            for (v, mask) |x, keep| if (keep) {
                out[w] = x;
                w += 1;
            };
            break :blk .{ .f64 = out };
        },
        .i64 => |v| blk: {
            var out = try allocator.alloc(i64, count);
            var w: usize = 0;
            for (v, mask) |x, keep| if (keep) {
                out[w] = x;
                w += 1;
            };
            break :blk .{ .i64 = out };
        },
        .bool => |v| blk: {
            var out = try allocator.alloc(bool, count);
            var w: usize = 0;
            for (v, mask) |x, keep| if (keep) {
                out[w] = x;
                w += 1;
            };
            break :blk .{ .bool = out };
        },
        .string => |v| blk: {
            var out = try allocator.alloc([]const u8, count);
            errdefer allocator.free(out);
            var w: usize = 0;
            errdefer {
                for (out[0..w]) |s| allocator.free(s);
            }
            for (v, mask) |x, keep| if (keep) {
                out[w] = try allocator.dupe(u8, x);
                w += 1;
            };
            break :blk .{ .string = out };
        },
    };
}

fn sliceColumn(col: Column, start: usize, stop: usize) Column {
    return switch (col) {
        .f64 => |v| .{ .f64 = v[start..stop] },
        .i64 => |v| .{ .i64 = v[start..stop] },
        .bool => |v| .{ .bool = v[start..stop] },
        .string => |v| .{ .string = v[start..stop] },
    };
}

fn takeColumn(allocator: std.mem.Allocator, col: Column, indices: []const usize) DataError!Column {
    return switch (col) {
        .f64 => |v| blk: {
            const out = try allocator.alloc(f64, indices.len);
            for (indices, out) |idx, *slot| slot.* = v[idx];
            break :blk .{ .f64 = out };
        },
        .i64 => |v| blk: {
            const out = try allocator.alloc(i64, indices.len);
            for (indices, out) |idx, *slot| slot.* = v[idx];
            break :blk .{ .i64 = out };
        },
        .bool => |v| blk: {
            const out = try allocator.alloc(bool, indices.len);
            for (indices, out) |idx, *slot| slot.* = v[idx];
            break :blk .{ .bool = out };
        },
        .string => |v| blk: {
            var out = try allocator.alloc([]const u8, indices.len);
            errdefer allocator.free(out);
            var initialized: usize = 0;
            errdefer {
                for (out[0..initialized]) |s| allocator.free(s);
            }
            for (indices, out) |idx, *slot| {
                slot.* = try allocator.dupe(u8, v[idx]);
                initialized += 1;
            }
            break :blk .{ .string = out };
        },
    };
}

fn describeF64(allocator: std.mem.Allocator, v: []const f64) DataError![]f64 {
    var out = try allocator.alloc(f64, 4);
    if (v.len == 0) {
        @memset(out, std.math.nan(f64));
        out[0] = 0;
        return out;
    }
    var total: f64 = 0;
    var min_v = v[0];
    var max_v = v[0];
    for (v) |x| {
        total += x;
        if (x < min_v) min_v = x;
        if (x > max_v) max_v = x;
    }
    out[0] = @floatFromInt(v.len);
    out[1] = total / @as(f64, @floatFromInt(v.len));
    out[2] = min_v;
    out[3] = max_v;
    return out;
}

fn describeI64(allocator: std.mem.Allocator, v: []const i64) DataError![]f64 {
    const tmp = try allocator.alloc(f64, v.len);
    defer allocator.free(tmp);
    for (v, tmp) |x, *slot| slot.* = @floatFromInt(x);
    return describeF64(allocator, tmp);
}

fn splitCsvLineOwned(allocator: std.mem.Allocator, line: []const u8, out: *std.ArrayList([]const u8)) DataError!void {
    var it = std.mem.splitScalar(u8, line, ',');
    while (it.next()) |cell| {
        try out.append(allocator, try allocator.dupe(u8, std.mem.trim(u8, cell, " \t\r\"")));
    }
}

fn inferColumn(allocator: std.mem.Allocator, cells: []const []const u8) DataError!Column {
    const i64_values = try allocator.alloc(i64, cells.len);
    var all_i64 = true;
    for (cells, i64_values) |cell, *slot| {
        slot.* = std.fmt.parseInt(i64, cell, 10) catch {
            all_i64 = false;
            break;
        };
    }
    if (all_i64) return .{ .i64 = i64_values };
    allocator.free(i64_values);

    const f64_values = try allocator.alloc(f64, cells.len);
    var all_f64 = true;
    for (cells, f64_values) |cell, *slot| {
        slot.* = std.fmt.parseFloat(f64, cell) catch {
            all_f64 = false;
            break;
        };
    }
    if (all_f64) return .{ .f64 = f64_values };
    allocator.free(f64_values);

    const bool_values = try allocator.alloc(bool, cells.len);
    var all_bool = true;
    for (cells, bool_values) |cell, *slot| {
        if (std.ascii.eqlIgnoreCase(cell, "true")) {
            slot.* = true;
        } else if (std.ascii.eqlIgnoreCase(cell, "false")) {
            slot.* = false;
        } else {
            all_bool = false;
            break;
        }
    }
    if (all_bool) return .{ .bool = bool_values };
    allocator.free(bool_values);

    var strings = try allocator.alloc([]const u8, cells.len);
    errdefer allocator.free(strings);
    var initialized: usize = 0;
    errdefer {
        for (strings[0..initialized]) |s| allocator.free(s);
    }
    for (cells, strings) |cell, *slot| {
        slot.* = try allocator.dupe(u8, cell);
        initialized += 1;
    }
    return .{ .string = strings };
}

fn printCell(writer: *std.Io.Writer, col: Column, row: usize) std.Io.Writer.Error!void {
    switch (col) {
        .f64 => |v| try writer.print("{}", .{v[row]}),
        .i64 => |v| try writer.print("{}", .{v[row]}),
        .bool => |v| try writer.print("{}", .{v[row]}),
        .string => |v| try writer.print("{s}", .{v[row]}),
    }
}

pub fn dataframe(allocator: std.mem.Allocator, defs: []const ColumnDef) DataError!DataFrame {
    return DataFrame.init(allocator, defs);
}

test "dataframe select filter groupby and csv" {
    const gpa = std.testing.allocator;
    var df = try DataFrame.init(gpa, &.{
        .{ .name = "city", .data = .{ .string = &.{ "hz", "bj", "hz" } } },
        .{ .name = "sales", .data = .{ .f64 = &.{ 2.0, 3.0, 5.0 } } },
        .{ .name = "units", .data = .{ .i64 = &.{ 1, 2, 3 } } },
    });
    defer df.deinit();
    try std.testing.expectEqual(@as(usize, 3), df.height());
    var filtered = try df.filter(&.{ true, false, true });
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered.height());
    var grouped = try df.groupBySum("city", "sales");
    defer grouped.deinit();
    try std.testing.expectEqual(@as(usize, 2), grouped.height());
    var desc = try df.describe();
    defer desc.deinit();
    try std.testing.expectEqual(@as(usize, 4), desc.height());
    const csv = try df.writeCsv(gpa);
    defer gpa.free(csv);
    var parsed = try DataFrame.readCsv(gpa, csv, true);
    defer parsed.deinit();
    try std.testing.expectEqual(df.height(), parsed.height());
}
