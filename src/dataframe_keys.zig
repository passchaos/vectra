const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const validity_mod = @import("dataframe_validity.zig");
const single_join_mod = @import("dataframe_keys_single_join.zig");

const validityValues = validity_mod.validityValues;
const columnsRowsEqual = dataframe_array_mod.columnsRowsEqual;
const groupKeyEqual = numeric_mod.groupKeyEqual;
const compareSortValues = numeric_mod.compareSortValues;
const asofDistance = numeric_mod.asofDistance;

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

pub const asofRightRowIndices = single_join_mod.asofRightRowIndices;
pub const asofRightRowIndicesTyped = single_join_mod.asofRightRowIndicesTyped;
pub const innerJoinRowIndicesTyped = single_join_mod.innerJoinRowIndicesTyped;
pub const leftJoinRowIndicesTyped = single_join_mod.leftJoinRowIndicesTyped;
pub const fullJoinRowIndicesTyped = single_join_mod.fullJoinRowIndicesTyped;
pub const innerJoinRowIndices = single_join_mod.innerJoinRowIndices;
pub const leftJoinRowIndices = single_join_mod.leftJoinRowIndices;
pub const fullJoinRowIndices = single_join_mod.fullJoinRowIndices;
pub const semiAntiJoinRowIndices = single_join_mod.semiAntiJoinRowIndices;
pub const semiAntiJoinRowIndicesTyped = single_join_mod.semiAntiJoinRowIndicesTyped;

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

pub fn innerJoinRowIndicesMulti(
    comptime JoinRowIndexPair: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) KeyMatchError!JoinRowIndexPair {
    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    // This is intentionally expressed as a row-pair builder, mirroring the
    // single-key join path and cuDF's hash-join output shape. The current
    // implementation materializes key columns through `Array.toOwnedSlice()` in
    // `columnsRowsEqual`; the API boundary is what future Axiom lowering will
    // replace with a multi-key hash table/probe kernel.
    for (0..left.rows) |left_i| {
        for (0..right.rows) |right_i| {
            if (try rowsMatchAllKeys(allocator, left, right, left_key_names, right_key_names, left_i, right_i)) {
                try left_indices.append(allocator, left_i);
                try right_indices.append(allocator, right_i);
            }
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

pub fn leftJoinRowIndicesMulti(
    comptime JoinRowIndexPair: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) KeyMatchError!JoinRowIndexPair {
    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    for (0..left.rows) |left_i| {
        var matched = false;
        for (0..right.rows) |right_i| {
            if (try rowsMatchAllKeys(allocator, left, right, left_key_names, right_key_names, left_i, right_i)) {
                try left_indices.append(allocator, left_i);
                try right_indices.append(allocator, right_i);
                matched = true;
            }
        }
        if (!matched) {
            try left_indices.append(allocator, left_i);
            try right_indices.append(allocator, null);
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

pub fn fullJoinRowIndicesMulti(
    comptime JoinRowIndexPair: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) KeyMatchError!JoinRowIndexPair {
    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);
    const right_matched = try allocator.alloc(bool, right.rows);
    defer allocator.free(right_matched);
    @memset(right_matched, false);

    for (0..left.rows) |left_i| {
        var matched = false;
        for (0..right.rows) |right_i| {
            if (try rowsMatchAllKeys(allocator, left, right, left_key_names, right_key_names, left_i, right_i)) {
                try left_indices.append(allocator, left_i);
                try right_indices.append(allocator, right_i);
                right_matched[right_i] = true;
                matched = true;
            }
        }
        if (!matched) {
            try left_indices.append(allocator, left_i);
            try right_indices.append(allocator, null);
        }
    }

    for (0..right.rows) |right_i| {
        if (!right_matched[right_i]) {
            try left_indices.append(allocator, null);
            try right_indices.append(allocator, right_i);
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

pub fn semiAntiJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    keep_matches: bool,
) KeyMatchError![]usize {
    var indices: std.ArrayList(usize) = .empty;
    errdefer indices.deinit(allocator);

    for (0..left.rows) |left_i| {
        var matched = false;
        for (0..right.rows) |right_i| {
            if (try rowsMatchAllKeys(allocator, left, right, left_key_names, right_key_names, left_i, right_i)) {
                matched = true;
                break;
            }
        }
        if (matched == keep_matches) try indices.append(allocator, left_i);
    }

    return indices.toOwnedSlice(allocator);
}

pub fn distinctRows(comptime DeviceDataFrame: type, frame: DeviceDataFrame) KeyMatchError!DeviceDataFrame {
    return distinctOn(DeviceDataFrame, frame, frame.names);
}

pub fn distinctOn(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_names: []const []const u8,
) KeyMatchError!DeviceDataFrame {
    const indices = try distinctRowIndices(frame.allocator, frame, key_names);
    defer frame.allocator.free(indices);
    return frame.take(indices);
}
