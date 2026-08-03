//! Shared deinitialization helpers for profile-style lazy operation payloads.

const std = @import("std");

/// Profile payloads own their duplicated column/prefix strings but only borrow
/// or store by value their options/scalars. Keeping the string-freeing invariant
/// in one place avoids subtle drift between clone and deinit switch arms.
pub fn freeNameOutput(allocator: std.mem.Allocator, payload: anytype) void {
    allocator.free(payload.name);
    allocator.free(payload.output_prefix);
}

pub fn freePairOutput(
    allocator: std.mem.Allocator,
    payload: anytype,
    comptime lhs_field_name: []const u8,
    comptime rhs_field_name: []const u8,
) void {
    allocator.free(@field(payload, lhs_field_name));
    allocator.free(@field(payload, rhs_field_name));
    allocator.free(payload.output_prefix);
}
