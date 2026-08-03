//! Null-predicate helpers for Parquet scan pushdown.
//!
//! The planner can only pass one nullable-column predicate to Boltha's simple
//! table reader today, so this module owns the small state machine for installing
//! or clearing that predicate while leaving dependency discovery in the main
//! planner.

const std = @import("std");
const options_mod = @import("../../../dataframe_options.zig");

const DeviceParquetNullFilter = options_mod.DeviceParquetNullFilter;

pub fn setNullPredicate(
    allocator: std.mem.Allocator,
    current: *?DeviceParquetNullFilter,
    column_name: []const u8,
    want_nulls: bool,
) std.mem.Allocator.Error!void {
    if (current.*) |existing| {
        if (existing.want_nulls == want_nulls and std.mem.eql(u8, existing.column, column_name)) return;
        return;
    }
    current.* = .{
        .column = try allocator.dupe(u8, column_name),
        .want_nulls = want_nulls,
    };
}

pub fn clearNullPredicate(allocator: std.mem.Allocator, current: *?DeviceParquetNullFilter) void {
    if (current.*) |predicate| {
        allocator.free(predicate.column);
        current.* = null;
    }
}
