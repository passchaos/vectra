const std = @import("std");

pub fn splitCsvLineOwned(allocator: std.mem.Allocator, line: []const u8, out: *std.ArrayList([]const u8)) std.mem.Allocator.Error!void {
    var it = std.mem.splitScalar(u8, line, ',');
    while (it.next()) |cell| {
        try out.append(allocator, try allocator.dupe(u8, std.mem.trim(u8, cell, " \t\r\"")));
    }
}

pub fn printCell(writer: *std.Io.Writer, col: anytype, row: usize) std.Io.Writer.Error!void {
    switch (col) {
        .f64 => |v| try writer.print("{}", .{v[row]}),
        .i64 => |v| try writer.print("{}", .{v[row]}),
        .bool => |v| try writer.print("{}", .{v[row]}),
        .string => |v| try writer.print("{s}", .{v[row]}),
    }
}
