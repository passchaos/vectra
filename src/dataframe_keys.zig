const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const validity_mod = @import("dataframe_validity.zig");

const validityValues = validity_mod.validityValues;
const columnsRowsEqual = dataframe_array_mod.columnsRowsEqual;

pub const KeyMatchError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
    IndexOutOfBounds,
};

pub fn distinctRowIndices(allocator: std.mem.Allocator, frame: anytype, key_names: []const []const u8) KeyMatchError![]usize {
    if (key_names.len == 0) return error.LengthMismatch;
    for (key_names) |name| _ = try frame.column(name);

    var representatives: std.ArrayList(usize) = .empty;
    errdefer representatives.deinit(allocator);

    // Preserve first-seen row order, matching the common stable
    // `drop_duplicates(keep=first)` dataframe behavior. The current
    // implementation deliberately routes through the same row-comparison helper
    // used by multi-key joins/grouping so null-key rows are skipped and future
    // Axiom hash-distinct lowering has a single API seam to replace.
    for (0..frame.rows) |row| {
        if (!try rowHasValidKeys(allocator, frame, key_names, row)) continue;
        const maybe_seen = try findMultiKeyGroupIndex(allocator, frame, key_names, representatives.items, row);
        if (maybe_seen == null) try representatives.append(allocator, row);
    }

    return representatives.toOwnedSlice(allocator);
}

pub fn rowHasValidKeys(allocator: std.mem.Allocator, frame: anytype, key_names: []const []const u8, row: usize) KeyMatchError!bool {
    for (key_names) |key_name| {
        const key = try frame.column(key_name);
        if (!try columnRowValid(allocator, key.*, row)) return false;
    }
    return true;
}

pub fn columnRowValid(allocator: std.mem.Allocator, column: anytype, row: usize) KeyMatchError!bool {
    return switch (column) {
        inline else => |typed| blk: {
            if (row >= typed.len()) return error.IndexOutOfBounds;
            const maybe_validity = try validityValues(typed, allocator);
            defer if (maybe_validity) |validity| allocator.free(validity);
            break :blk if (maybe_validity) |validity| validity[row] else true;
        },
    };
}

pub fn findMultiKeyGroupIndex(allocator: std.mem.Allocator, frame: anytype, key_names: []const []const u8, representatives: []const usize, row: usize) KeyMatchError!?usize {
    for (representatives, 0..) |representative, i| {
        if (try rowsMatchAllKeys(allocator, frame, frame, key_names, key_names, representative, row)) return i;
    }
    return null;
}

pub fn rowsMatchAllKeys(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    left_i: usize,
    right_i: usize,
) KeyMatchError!bool {
    for (left_key_names, right_key_names) |left_name, right_name| {
        const left_key = try left.column(left_name);
        const right_key = try right.column(right_name);
        if (!try columnsRowsEqual(allocator, left_key.*, right_key.*, left_i, right_i)) return false;
    }
    return true;
}
