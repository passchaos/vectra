const std = @import("std");

pub fn cloneColumn(allocator: std.mem.Allocator, col: anytype) std.mem.Allocator.Error!@TypeOf(col) {
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

pub fn freeColumn(allocator: std.mem.Allocator, col: anytype) void {
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

pub fn filterColumn(allocator: std.mem.Allocator, col: anytype, mask: []const bool) std.mem.Allocator.Error!@TypeOf(col) {
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

pub fn sliceColumn(col: anytype, start: usize, stop: usize) @TypeOf(col) {
    return switch (col) {
        .f64 => |v| .{ .f64 = v[start..stop] },
        .i64 => |v| .{ .i64 = v[start..stop] },
        .bool => |v| .{ .bool = v[start..stop] },
        .string => |v| .{ .string = v[start..stop] },
    };
}

pub fn takeColumn(allocator: std.mem.Allocator, col: anytype, indices: []const usize) std.mem.Allocator.Error!@TypeOf(col) {
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
