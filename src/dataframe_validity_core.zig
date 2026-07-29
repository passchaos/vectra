//! Core validity-mask helpers shared by typed columns and profile operations.
//!
//! The full `dataframe_validity.zig` module builds profile output columns and
//! therefore imports the tagged `DeviceColumn` union. Low-level storage and
//! conversion code only needs these tiny mask primitives, so they live here to
//! keep the dependency graph acyclic and the larger facade files small.

const std = @import("std");
const array_mod = @import("array.zig");

pub fn countNulls(validity_values: []const bool) usize {
    var nulls: usize = 0;
    for (validity_values) |valid| {
        if (!valid) nulls += 1;
    }
    return nulls;
}

pub fn countNullsInArray(mask: array_mod.Array(bool)) array_mod.ArrayError!usize {
    const values = try mask.toOwnedSlice(mask.allocator);
    defer mask.allocator.free(values);
    return countNulls(values);
}

pub fn validityValues(column: anytype, allocator: std.mem.Allocator) array_mod.ArrayError!?[]bool {
    const mask = column.validity orelse return null;
    return try mask.toOwnedSlice(allocator);
}
