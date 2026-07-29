//! CSV and textual I/O helpers for the small host-side dataframe.
//!
//! These routines are generic over the concrete `DataFrame`/`Column` types so
//! `dataframe_host.zig` can keep owning the public host facade without growing
//! a large I/O implementation body.

const std = @import("std");
const series_mod = @import("series.zig");
const dataframe_column_mod = @import("dataframe_column.zig");
const csv_mod = @import("dataframe_csv.zig");

pub const DataError = series_mod.DataError;
const freeColumn = dataframe_column_mod.freeColumn;
const splitCsvLineOwned = csv_mod.splitCsvLineOwned;
const printCell = csv_mod.printCell;

pub fn readCsv(
    comptime DataFrame: type,
    comptime Column: type,
    comptime ColumnDef: type,
    allocator: std.mem.Allocator,
    text: []const u8,
    has_header: bool,
) DataError!DataFrame {
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
        defs[i] = .{ .name = header, .data = try csv_mod.inferColumn(Column, allocator, cells.items) };
    }
    defer {
        for (defs) |def| freeColumn(allocator, def.data);
    }
    return DataFrame.init(allocator, defs);
}

pub fn writeCsv(frame: anytype, allocator: std.mem.Allocator) DataError![]u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    for (frame.names, 0..) |name, i| {
        if (i != 0) try aw.writer.print(",", .{});
        try aw.writer.print("{s}", .{name});
    }
    try aw.writer.print("\n", .{});
    for (0..frame.rows) |r| {
        for (frame.columns, 0..) |col, c| {
            if (c != 0) try aw.writer.print(",", .{});
            try printCell(&aw.writer, col, r);
        }
        try aw.writer.print("\n", .{});
    }
    return aw.toOwnedSlice();
}

pub fn print(frame: anytype, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("DataFrame(shape=({}, {}))\n", .{ frame.rows, frame.columns.len });
    for (frame.names, 0..) |name, i| {
        if (i != 0) try writer.print("\t", .{});
        try writer.print("{s}", .{name});
    }
    try writer.print("\n", .{});
    const limit = @min(frame.rows, 12);
    for (0..limit) |r| {
        for (frame.columns, 0..) |col, c| {
            if (c != 0) try writer.print("\t", .{});
            try printCell(writer, col, r);
        }
        try writer.print("\n", .{});
    }
    if (frame.rows > limit) try writer.print("...\n", .{});
}
