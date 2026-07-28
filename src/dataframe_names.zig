const std = @import("std");

pub fn cloneNameList(allocator: std.mem.Allocator, names: []const []const u8) std.mem.Allocator.Error![][]const u8 {
    const owned = try allocator.alloc([]const u8, names.len);
    errdefer allocator.free(owned);
    var initialized: usize = 0;
    errdefer {
        for (owned[0..initialized]) |name| allocator.free(name);
    }
    for (names, owned) |name, *slot| {
        slot.* = try allocator.dupe(u8, name);
        initialized += 1;
    }
    return owned;
}

pub fn freeNameList(allocator: std.mem.Allocator, names: [][]const u8) void {
    for (names) |name| allocator.free(name);
    allocator.free(names);
}

pub fn appendOwnedNameUnique(allocator: std.mem.Allocator, names: *std.ArrayList([]const u8), name: []const u8) std.mem.Allocator.Error!void {
    for (names.items) |existing| {
        if (std.mem.eql(u8, existing, name)) return;
    }
    const owned = try allocator.dupe(u8, name);
    errdefer allocator.free(owned);
    try names.append(allocator, owned);
}

pub fn appendBorrowedNameUnique(allocator: std.mem.Allocator, names: *std.ArrayList([]const u8), name: []const u8) std.mem.Allocator.Error!void {
    if (nameInBorrowedList(name, names.items)) return;
    try names.append(allocator, name);
}

pub fn nameInBorrowedList(name: []const u8, names: []const []const u8) bool {
    for (names) |existing| {
        if (std.mem.eql(u8, existing, name)) return true;
    }
    return false;
}

pub fn freeOwnedNameItems(allocator: std.mem.Allocator, names: []const []const u8) void {
    for (names) |name| allocator.free(name);
}

pub fn allNamesIn(names: []const []const u8, allowed: []const []const u8) bool {
    for (names) |name| {
        var found = false;
        for (allowed) |candidate| {
            if (std.mem.eql(u8, name, candidate)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

pub fn statsOutputNames(allocator: std.mem.Allocator, key_name: []const u8, prefix: []const u8) std.mem.Allocator.Error![]const []const u8 {
    const names = try allocator.alloc([]const u8, 6);
    errdefer allocator.free(names);
    names[0] = key_name;
    names[1] = try std.fmt.allocPrint(allocator, "{s}_count", .{prefix});
    errdefer allocator.free(names[1]);
    names[2] = try std.fmt.allocPrint(allocator, "{s}_sum", .{prefix});
    errdefer allocator.free(names[2]);
    names[3] = try std.fmt.allocPrint(allocator, "{s}_min", .{prefix});
    errdefer allocator.free(names[3]);
    names[4] = try std.fmt.allocPrint(allocator, "{s}_max", .{prefix});
    errdefer allocator.free(names[4]);
    names[5] = try std.fmt.allocPrint(allocator, "{s}_mean", .{prefix});
    return names;
}

pub fn freeStatsOutputNames(allocator: std.mem.Allocator, names: []const []const u8) void {
    for (names[1..]) |name| allocator.free(name);
    allocator.free(names);
}

pub fn profileOutputNames(allocator: std.mem.Allocator, key_names: []const []const u8, prefix: []const u8) std.mem.Allocator.Error![]const []const u8 {
    const names = try allocator.alloc([]const u8, key_names.len + 7);
    errdefer allocator.free(names);
    for (key_names, 0..) |key_name, i| names[i] = key_name;
    var initialized: usize = 0;
    errdefer {
        for (names[key_names.len .. key_names.len + initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "count", "sum", "mean", "variance", "stddev", "skewness", "kurtosis" };
    for (suffixes, 0..) |suffix, i| {
        names[key_names.len + i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub fn freeProfileOutputNames(allocator: std.mem.Allocator, names: []const []const u8, key_count: usize) void {
    for (names[key_count..]) |name| allocator.free(name);
    allocator.free(names);
}
