const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const validity_mod = @import("dataframe_validity.zig");

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

pub fn asofRightRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype, strategy: anytype) KeyMatchError![]?usize {
    return switch (left) {
        .i8 => |typed| asofRightRowIndicesTyped(i8, allocator, typed, right.i8, strategy),
        .i16 => |typed| asofRightRowIndicesTyped(i16, allocator, typed, right.i16, strategy),
        .i32 => |typed| asofRightRowIndicesTyped(i32, allocator, typed, right.i32, strategy),
        .i64 => |typed| asofRightRowIndicesTyped(i64, allocator, typed, right.i64, strategy),
        .u8 => |typed| asofRightRowIndicesTyped(u8, allocator, typed, right.u8, strategy),
        .u16 => |typed| asofRightRowIndicesTyped(u16, allocator, typed, right.u16, strategy),
        .u32 => |typed| asofRightRowIndicesTyped(u32, allocator, typed, right.u32, strategy),
        .u64 => |typed| asofRightRowIndicesTyped(u64, allocator, typed, right.u64, strategy),
        .usize => |typed| asofRightRowIndicesTyped(usize, allocator, typed, right.usize, strategy),
        .isize => |typed| asofRightRowIndicesTyped(isize, allocator, typed, right.isize, strategy),
        .f16 => |typed| asofRightRowIndicesTyped(f16, allocator, typed, right.f16, strategy),
        .f32 => |typed| asofRightRowIndicesTyped(f32, allocator, typed, right.f32, strategy),
        .f64 => |typed| asofRightRowIndicesTyped(f64, allocator, typed, right.f64, strategy),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn asofRightRowIndicesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    strategy: anytype,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{ InvalidDevice, TypeUnsupported })![]?usize {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    const indices = try allocator.alloc(?usize, left_values.len);
    for (left_values, indices, 0..) |left_value, *slot, left_i| {
        slot.* = null;
        if (maybe_left_validity) |validity| {
            if (!validity[left_i]) continue;
        }
        var best: ?usize = null;
        for (right_values, 0..) |right_value, right_i| {
            if (maybe_right_validity) |validity| {
                if (!validity[right_i]) continue;
            }
            switch (strategy) {
                .previous => {
                    if (compareSortValues(T, right_value, left_value) <= 0 and (best == null or compareSortValues(T, right_value, right_values[best.?]) > 0)) best = right_i;
                },
                .next => {
                    if (compareSortValues(T, right_value, left_value) >= 0 and (best == null or compareSortValues(T, right_value, right_values[best.?]) < 0)) best = right_i;
                },
                .nearest => {
                    if (best == null or asofDistance(T, left_value, right_value) < asofDistance(T, left_value, right_values[best.?])) best = right_i;
                },
            }
        }
        slot.* = best;
    }
    return indices;
}

pub fn innerJoinRowIndicesTyped(
    comptime T: type,
    comptime JoinRowIndexPair: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{InvalidDevice})!JoinRowIndexPair {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    for (left_values, 0..) |left_value, left_i| {
        if (maybe_left_validity) |validity| {
            if (!validity[left_i]) continue;
        }
        for (right_values, 0..) |right_value, right_i| {
            if (maybe_right_validity) |validity| {
                if (!validity[right_i]) continue;
            }
            if (groupKeyEqual(T, left_value, right_value)) {
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

pub fn leftJoinRowIndicesTyped(
    comptime T: type,
    comptime JoinRowIndexPair: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{InvalidDevice})!JoinRowIndexPair {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    for (left_values, 0..) |left_value, left_i| {
        var matched = false;
        const left_valid = if (maybe_left_validity) |validity| validity[left_i] else true;
        if (left_valid) {
            for (right_values, 0..) |right_value, right_i| {
                if (maybe_right_validity) |validity| {
                    if (!validity[right_i]) continue;
                }
                if (groupKeyEqual(T, left_value, right_value)) {
                    try left_indices.append(allocator, left_i);
                    try right_indices.append(allocator, right_i);
                    matched = true;
                }
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

pub fn fullJoinRowIndicesTyped(
    comptime T: type,
    comptime JoinRowIndexPair: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{InvalidDevice})!JoinRowIndexPair {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);
    const right_matched = try allocator.alloc(bool, right_values.len);
    defer allocator.free(right_matched);
    @memset(right_matched, false);

    for (left_values, 0..) |left_value, left_i| {
        var matched = false;
        const left_valid = if (maybe_left_validity) |validity| validity[left_i] else true;
        if (left_valid) {
            for (right_values, 0..) |right_value, right_i| {
                if (maybe_right_validity) |validity| {
                    if (!validity[right_i]) continue;
                }
                if (groupKeyEqual(T, left_value, right_value)) {
                    try left_indices.append(allocator, left_i);
                    try right_indices.append(allocator, right_i);
                    right_matched[right_i] = true;
                    matched = true;
                }
            }
        }
        if (!matched) {
            try left_indices.append(allocator, left_i);
            try right_indices.append(allocator, null);
        }
    }

    for (right_values, 0..) |_, right_i| {
        if (maybe_right_validity) |validity| {
            if (!validity[right_i]) {
                try left_indices.append(allocator, null);
                try right_indices.append(allocator, right_i);
                continue;
            }
        }
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

pub fn semiAntiJoinRowIndicesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    keep_matches: bool,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{InvalidDevice})![]usize {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var indices: std.ArrayList(usize) = .empty;
    errdefer indices.deinit(allocator);
    for (left_values, 0..) |left_value, left_i| {
        const left_valid = if (maybe_left_validity) |validity| validity[left_i] else true;
        var matched = false;
        if (left_valid) {
            for (right_values, 0..) |right_value, right_i| {
                if (maybe_right_validity) |validity| {
                    if (!validity[right_i]) continue;
                }
                if (groupKeyEqual(T, left_value, right_value)) {
                    matched = true;
                    break;
                }
            }
        }
        if (matched == keep_matches) try indices.append(allocator, left_i);
    }
    return indices.toOwnedSlice(allocator);
}
