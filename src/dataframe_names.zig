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
