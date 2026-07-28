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

pub fn inferColumn(comptime Column: type, allocator: std.mem.Allocator, cells: []const []const u8) std.mem.Allocator.Error!Column {
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
