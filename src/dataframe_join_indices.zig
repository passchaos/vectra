//! Row-index builders used by eager dataframe joins.

const std = @import("std");
const keys_mod = @import("dataframe_keys.zig");

pub const JoinRowIndexPair = struct {
    allocator: std.mem.Allocator,
    left: []?usize,
    right: []?usize,

    pub fn deinit(self: *JoinRowIndexPair) void {
        self.allocator.free(self.left);
        self.allocator.free(self.right);
        self.* = undefined;
    }
};

pub fn innerJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.innerJoinRowIndices(JoinRowIndexPair, allocator, left, right);
}

pub fn innerJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.innerJoinRowIndicesMulti(JoinRowIndexPair, allocator, left, right, left_key_names, right_key_names);
}

pub fn leftJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.leftJoinRowIndices(JoinRowIndexPair, allocator, left, right);
}

pub fn leftJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.leftJoinRowIndicesMulti(JoinRowIndexPair, allocator, left, right, left_key_names, right_key_names);
}

pub fn fullJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.fullJoinRowIndices(JoinRowIndexPair, allocator, left, right);
}

pub fn fullJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.fullJoinRowIndicesMulti(JoinRowIndexPair, allocator, left, right, left_key_names, right_key_names);
}

pub fn semiAntiJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype, keep_matches: bool) keys_mod.KeyMatchError![]usize {
    return keys_mod.semiAntiJoinRowIndices(allocator, left, right, keep_matches);
}

pub fn semiAntiJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    keep_matches: bool,
) keys_mod.KeyMatchError![]usize {
    return keys_mod.semiAntiJoinRowIndicesMulti(allocator, left, right, left_key_names, right_key_names, keep_matches);
}
