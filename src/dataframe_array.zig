const std = @import("std");
const alea = @import("alea");
const array_mod = @import("array.zig");
const array_helpers_mod = @import("dataframe_array_helpers.zig");
const names_mod = @import("dataframe_names.zig");
const options_mod = @import("dataframe_options.zig");
const validity_core_mod = @import("dataframe_validity_core.zig");

const DeviceFrameArrayError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    InvalidCsv,
    EmptyDataFrame,
    UnsupportedType,
    InvalidDevice,
};

pub const requireCompatibleColumnArrays = array_helpers_mod.requireCompatibleColumnArrays;
pub const combineValidityMasks = array_helpers_mod.combineValidityMasks;
pub const zeroValue = array_helpers_mod.zeroValue;
pub const rowIndicesFromMask = array_helpers_mod.rowIndicesFromMask;
pub const sliceArray1d = array_helpers_mod.sliceArray1d;
pub const takeArray1d = array_helpers_mod.takeArray1d;
pub const concatTypedColumns = array_helpers_mod.concatTypedColumns;
pub const coalesceTypedColumns = array_helpers_mod.coalesceTypedJoinKeys;
pub const coalesceTypedJoinKeys = array_helpers_mod.coalesceTypedJoinKeys;
pub const concatDeviceColumns = array_helpers_mod.concatDeviceColumns;
pub const coalesceJoinKeys = array_helpers_mod.coalesceJoinKeys;
pub const initDeviceDataFrameFromOwnedColumns = array_helpers_mod.initDeviceDataFrameFromOwnedColumns;
pub const concatDeviceDataFramesRows = array_helpers_mod.concatDeviceDataFramesRows;
pub const concatDeviceDataFramesColumns = array_helpers_mod.concatDeviceDataFramesColumns;
pub const takeOptionalRows = array_helpers_mod.takeOptionalRows;
pub const columnsRowsEqual = array_helpers_mod.columnsRowsEqual;
pub const columnsRowsEqualTyped = array_helpers_mod.columnsRowsEqualTyped;
const nameInBorrowedList = names_mod.nameInBorrowedList;
const DeviceScalar = options_mod.DeviceScalar;
const validityValues = validity_core_mod.validityValues;
const countNulls = validity_core_mod.countNulls;

pub fn select(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    wanted_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    if (wanted_names.len == 0) return DeviceDataFrame.initEmpty(input.allocator, input.rows, input.device);
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, wanted_names.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (wanted_names, 0..) |name, i| {
        const source = try input.column(name);
        columns[i] = try source.clone();
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, wanted_names, columns, input.rows, input.device);
}

pub fn selectByColumnIndices(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    indices: []const usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const names = try input.allocator.alloc([]const u8, indices.len);
    defer input.allocator.free(names);
    for (indices, names) |index, *slot| {
        if (index >= input.names.len) return error.IndexOutOfBounds;
        slot.* = input.names[index];
    }
    return select(DeviceDataFrame, input, names);
}

pub fn selectColumnRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    start: usize,
    stop: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const end = @min(stop, input.names.len);
    const begin = @min(start, end);
    return select(DeviceDataFrame, input, input.names[begin..end]);
}

pub fn dropByColumnIndices(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    indices: []const usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const drop_names = try input.allocator.alloc([]const u8, indices.len);
    defer input.allocator.free(drop_names);
    for (indices, drop_names) |index, *slot| {
        if (index >= input.names.len) return error.IndexOutOfBounds;
        slot.* = input.names[index];
    }
    return dropColumns(DeviceDataFrame, input, drop_names);
}

pub fn dropColumnRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    start: usize,
    stop: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const end = @min(stop, input.names.len);
    const begin = @min(start, end);
    return dropColumns(DeviceDataFrame, input, input.names[begin..end]);
}

fn selectByNamePredicate(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    predicate: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    var selected_names: std.ArrayList([]const u8) = .empty;
    defer selected_names.deinit(input.allocator);
    for (input.names) |name| {
        if (predicate.matches(name)) try selected_names.append(input.allocator, name);
    }
    return select(DeviceDataFrame, input, selected_names.items);
}

fn dropByNamePredicate(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    predicate: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    var kept_names: std.ArrayList([]const u8) = .empty;
    defer kept_names.deinit(input.allocator);
    for (input.names) |name| {
        if (!predicate.matches(name)) try kept_names.append(input.allocator, name);
    }
    return select(DeviceDataFrame, input, kept_names.items);
}

const NamePrefixPredicate = struct {
    pattern: []const u8,

    fn matches(self: @This(), name: []const u8) bool {
        return std.mem.startsWith(u8, name, self.pattern);
    }
};

const NameSuffixPredicate = struct {
    pattern: []const u8,

    fn matches(self: @This(), name: []const u8) bool {
        return std.mem.endsWith(u8, name, self.pattern);
    }
};

const NameContainsPredicate = struct {
    pattern: []const u8,

    fn matches(self: @This(), name: []const u8) bool {
        return std.mem.indexOf(u8, name, self.pattern) != null;
    }
};

fn globMatches(pattern: []const u8, text: []const u8) bool {
    var pattern_index: usize = 0;
    var text_index: usize = 0;
    var star_index: ?usize = null;
    var retry_text_index: usize = 0;

    while (text_index < text.len) {
        if (pattern_index < pattern.len and (pattern[pattern_index] == '?' or pattern[pattern_index] == text[text_index])) {
            pattern_index += 1;
            text_index += 1;
        } else if (pattern_index < pattern.len and pattern[pattern_index] == '*') {
            star_index = pattern_index;
            pattern_index += 1;
            retry_text_index = text_index;
        } else if (star_index) |star| {
            pattern_index = star + 1;
            retry_text_index += 1;
            text_index = retry_text_index;
        } else {
            return false;
        }
    }

    while (pattern_index < pattern.len and pattern[pattern_index] == '*') pattern_index += 1;
    return pattern_index == pattern.len;
}

const NameGlobPredicate = struct {
    pattern: []const u8,

    fn matches(self: @This(), name: []const u8) bool {
        return globMatches(self.pattern, name);
    }
};

pub fn selectByNamePrefix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    prefix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByNamePredicate(DeviceDataFrame, input, NamePrefixPredicate{ .pattern = prefix });
}

pub fn selectByNameSuffix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    suffix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByNamePredicate(DeviceDataFrame, input, NameSuffixPredicate{ .pattern = suffix });
}

pub fn selectByNameContains(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    needle: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByNamePredicate(DeviceDataFrame, input, NameContainsPredicate{ .pattern = needle });
}

pub fn selectByNameGlob(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    pattern: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByNamePredicate(DeviceDataFrame, input, NameGlobPredicate{ .pattern = pattern });
}

pub fn dropByNamePrefix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    prefix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByNamePredicate(DeviceDataFrame, input, NamePrefixPredicate{ .pattern = prefix });
}

pub fn dropByNameSuffix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    suffix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByNamePredicate(DeviceDataFrame, input, NameSuffixPredicate{ .pattern = suffix });
}

pub fn dropByNameContains(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    needle: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByNamePredicate(DeviceDataFrame, input, NameContainsPredicate{ .pattern = needle });
}

pub fn dropByNameGlob(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    pattern: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByNamePredicate(DeviceDataFrame, input, NameGlobPredicate{ .pattern = pattern });
}

fn selectByDTypePredicate(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    predicate: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    var selected_names: std.ArrayList([]const u8) = .empty;
    defer selected_names.deinit(input.allocator);
    for (input.names, input.columns) |name, column| {
        if (predicate.matches(column.dtype())) try selected_names.append(input.allocator, name);
    }
    return select(DeviceDataFrame, input, selected_names.items);
}

fn dropByDTypePredicate(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    predicate: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    var kept_names: std.ArrayList([]const u8) = .empty;
    defer kept_names.deinit(input.allocator);
    for (input.names, input.columns) |name, column| {
        if (!predicate.matches(column.dtype())) try kept_names.append(input.allocator, name);
    }
    return select(DeviceDataFrame, input, kept_names.items);
}

const DTypeListPredicate = struct {
    wanted: []const array_mod.DType,

    fn matches(self: @This(), dtype: array_mod.DType) bool {
        for (self.wanted) |candidate| {
            if (candidate == dtype) return true;
        }
        return false;
    }
};

pub fn selectByDTypes(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    dtypes: []const array_mod.DType,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByDTypePredicate(DeviceDataFrame, input, DTypeListPredicate{ .wanted = dtypes });
}

pub fn selectByDTypeClass(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    class: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByDTypePredicate(DeviceDataFrame, input, class);
}

pub fn dropByDTypes(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    dtypes: []const array_mod.DType,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByDTypePredicate(DeviceDataFrame, input, DTypeListPredicate{ .wanted = dtypes });
}

pub fn dropByDTypeClass(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    class: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByDTypePredicate(DeviceDataFrame, input, class);
}

pub fn reverseColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    for (source_names, 0..) |*slot, i| {
        slot.* = input.names[input.names.len - 1 - i];
    }
    return select(DeviceDataFrame, input, source_names);
}

pub fn sortColumnsByName(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    descending: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    const order = try input.allocator.alloc(usize, input.names.len);
    defer input.allocator.free(order);
    for (order, 0..) |*slot, i| slot.* = i;

    const Ctx = struct {
        names: []const []const u8,
        descending: bool,

        fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            if (ctx.descending) return std.mem.lessThan(u8, ctx.names[b], ctx.names[a]);
            return std.mem.lessThan(u8, ctx.names[a], ctx.names[b]);
        }
    };
    std.sort.insertion(usize, order, Ctx{ .names = input.names, .descending = descending }, Ctx.lessThan);

    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    for (order, source_names) |index, *slot| slot.* = input.names[index];
    return select(DeviceDataFrame, input, source_names);
}

fn selectByColumnPredicate(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    predicate: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    var selected_names: std.ArrayList([]const u8) = .empty;
    defer selected_names.deinit(input.allocator);
    for (input.names, input.columns) |name, column| {
        if (predicate.matches(column)) try selected_names.append(input.allocator, name);
    }
    return select(DeviceDataFrame, input, selected_names.items);
}

fn dropByColumnPredicate(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    predicate: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    var kept_names: std.ArrayList([]const u8) = .empty;
    defer kept_names.deinit(input.allocator);
    for (input.names, input.columns) |name, column| {
        if (!predicate.matches(column)) try kept_names.append(input.allocator, name);
    }
    return select(DeviceDataFrame, input, kept_names.items);
}

const NullableColumnPredicate = struct {
    wanted: bool,

    fn matches(self: @This(), column: anytype) bool {
        return column.nullable() == self.wanted;
    }
};

const HasNullsColumnPredicate = struct {
    wanted: bool,

    fn matches(self: @This(), column: anytype) bool {
        return column.hasNulls() == self.wanted;
    }
};

fn columnHasNumericPredicate(column: anytype, allocator: std.mem.Allocator, comptime predicate: RowNumericPredicate) DeviceFrameArrayError!bool {
    return switch (column) {
        inline else => |typed| {
            const host_values = try typed.toOwnedSlice(allocator);
            defer allocator.free(host_values);
            const maybe_validity = try validityValues(typed, allocator);
            defer if (maybe_validity) |validity| allocator.free(validity);
            for (host_values, 0..) |value, row| {
                const valid = if (maybe_validity) |validity| validity[row] else true;
                if (valid and rowNumericPredicateMatches(@TypeOf(value), value, predicate)) return true;
            }
            return false;
        },
    };
}

fn selectColumnsByNumericPredicatePresence(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    wanted: bool,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    var selected_names: std.ArrayList([]const u8) = .empty;
    defer selected_names.deinit(input.allocator);
    for (input.names, input.columns) |name, column| {
        if ((try columnHasNumericPredicate(column, input.allocator, predicate)) == wanted) try selected_names.append(input.allocator, name);
    }
    return select(DeviceDataFrame, input, selected_names.items);
}

fn dropColumnsByNumericPredicatePresence(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    wanted: bool,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    var kept_names: std.ArrayList([]const u8) = .empty;
    defer kept_names.deinit(input.allocator);
    for (input.names, input.columns) |name, column| {
        if ((try columnHasNumericPredicate(column, input.allocator, predicate)) != wanted) try kept_names.append(input.allocator, name);
    }
    return select(DeviceDataFrame, input, kept_names.items);
}

pub fn selectNullableColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByColumnPredicate(DeviceDataFrame, input, NullableColumnPredicate{ .wanted = true });
}

pub fn selectNonNullableColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByColumnPredicate(DeviceDataFrame, input, NullableColumnPredicate{ .wanted = false });
}

pub fn selectColumnsWithNulls(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByColumnPredicate(DeviceDataFrame, input, HasNullsColumnPredicate{ .wanted = true });
}

pub fn selectColumnsWithoutNulls(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectByColumnPredicate(DeviceDataFrame, input, HasNullsColumnPredicate{ .wanted = false });
}

pub fn dropNullableColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByColumnPredicate(DeviceDataFrame, input, NullableColumnPredicate{ .wanted = true });
}

pub fn dropNonNullableColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByColumnPredicate(DeviceDataFrame, input, NullableColumnPredicate{ .wanted = false });
}

pub fn dropColumnsWithNulls(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByColumnPredicate(DeviceDataFrame, input, HasNullsColumnPredicate{ .wanted = true });
}

pub fn dropColumnsWithoutNulls(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropByColumnPredicate(DeviceDataFrame, input, HasNullsColumnPredicate{ .wanted = false });
}

pub fn selectColumnsWithNaNs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .nan);
}

pub fn selectColumnsWithoutNaNs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .nan);
}

pub fn dropColumnsWithNaNs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .nan);
}

pub fn dropColumnsWithoutNaNs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .nan);
}

pub fn selectColumnsWithInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .inf);
}

pub fn selectColumnsWithoutInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .inf);
}

pub fn dropColumnsWithInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .inf);
}

pub fn dropColumnsWithoutInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .inf);
}

pub fn selectColumnsWithPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .positive_inf);
}

pub fn selectColumnsWithoutPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .positive_inf);
}

pub fn dropColumnsWithPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .positive_inf);
}

pub fn dropColumnsWithoutPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .positive_inf);
}

pub fn selectColumnsWithNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .negative_inf);
}

pub fn selectColumnsWithoutNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .negative_inf);
}

pub fn dropColumnsWithNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .negative_inf);
}

pub fn dropColumnsWithoutNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .negative_inf);
}

pub fn selectColumnsWithZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .zero);
}

pub fn selectColumnsWithoutZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .zero);
}

pub fn dropColumnsWithZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .zero);
}

pub fn dropColumnsWithoutZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .zero);
}

pub fn selectColumnsWithPositiveZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .positive_zero);
}

pub fn selectColumnsWithoutPositiveZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .positive_zero);
}

pub fn dropColumnsWithPositiveZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .positive_zero);
}

pub fn dropColumnsWithoutPositiveZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .positive_zero);
}

pub fn selectColumnsWithNegativeZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .negative_zero);
}

pub fn selectColumnsWithoutNegativeZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .negative_zero);
}

pub fn dropColumnsWithNegativeZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .negative_zero);
}

pub fn dropColumnsWithoutNegativeZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .negative_zero);
}

pub fn selectColumnsWithNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .non_zero);
}

pub fn selectColumnsWithoutNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .non_zero);
}

pub fn dropColumnsWithNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .non_zero);
}

pub fn dropColumnsWithoutNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .non_zero);
}

pub fn selectColumnsWithPositives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .positive);
}

pub fn selectColumnsWithoutPositives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .positive);
}

pub fn dropColumnsWithPositives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .positive);
}

pub fn dropColumnsWithoutPositives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .positive);
}

pub fn selectColumnsWithSignBits(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .signbit);
}

pub fn selectColumnsWithoutSignBits(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .signbit);
}

pub fn dropColumnsWithSignBits(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .signbit);
}

pub fn dropColumnsWithoutSignBits(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .signbit);
}

pub fn selectColumnsWithNegatives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .negative);
}

pub fn selectColumnsWithoutNegatives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .negative);
}

pub fn dropColumnsWithNegatives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .negative);
}

pub fn dropColumnsWithoutNegatives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .negative);
}

pub fn selectColumnsWithFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .finite);
}

pub fn selectColumnsWithoutFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .finite);
}

pub fn dropColumnsWithFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .finite);
}

pub fn dropColumnsWithoutFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .finite);
}

pub fn selectColumnsWithNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .normal);
}

pub fn selectColumnsWithoutNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .normal);
}

pub fn dropColumnsWithNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .normal);
}

pub fn dropColumnsWithoutNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .normal);
}

pub fn selectColumnsWithSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .subnormal);
}

pub fn selectColumnsWithoutSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .subnormal);
}

pub fn dropColumnsWithSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .subnormal);
}

pub fn dropColumnsWithoutSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .subnormal);
}

pub fn selectColumnsWithNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .non_finite);
}

pub fn selectColumnsWithoutNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .non_finite);
}

pub fn dropColumnsWithNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, true, .non_finite);
}

pub fn dropColumnsWithoutNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsByNumericPredicatePresence(DeviceDataFrame, input, false, .non_finite);
}

pub fn withColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    data: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    const target_index = input.columnIndex(name) orelse input.columns.len;
    return withColumnAt(DeviceDataFrame, input, name, data, target_index);
}

pub fn withColumnAt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    data: anytype,
    target_index: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    if (data.len() != input.rows) return error.LengthMismatch;
    if (!data.device().sameDevice(input.device)) return error.InvalidDevice;
    const maybe_replace_index = input.columnIndex(name);
    const output_len = input.columns.len + @as(usize, if (maybe_replace_index == null) 1 else 0);
    if (target_index >= output_len) return error.IndexOutOfBounds;

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var source_names = try input.allocator.alloc([]const u8, output_len);
    defer input.allocator.free(source_names);

    var columns = try input.allocator.alloc(DeviceColumn, output_len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }

    // The target index is expressed in the final output schema.  For
    // replacements, stream all source columns except the replaced one and inject
    // the new column at the requested final position; for insertions, stream all
    // existing columns around the insertion point.
    var source_scan: usize = 0;
    for (0..output_len) |output_index| {
        if (output_index == target_index) {
            source_names[output_index] = name;
            columns[output_index] = try data.clone();
            initialized += 1;
            continue;
        }

        if (maybe_replace_index) |replace_index| {
            while (source_scan == replace_index) source_scan += 1;
        }
        source_names[output_index] = input.names[source_scan];
        columns[output_index] = try input.columns[source_scan].clone();
        initialized += 1;
        source_scan += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, source_names, columns, input.rows, input.device);
}

pub fn withColumnBefore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    data: anytype,
    before_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const maybe_replace_index = input.columnIndex(name);
    const anchor_index = input.columnIndex(before_name) orelse return error.ColumnNotFound;
    if (maybe_replace_index) |replace_index| {
        if (replace_index == anchor_index) return withColumnAt(DeviceDataFrame, input, name, data, replace_index);
    }

    const target_index = if (maybe_replace_index != null and maybe_replace_index.? < anchor_index) anchor_index - 1 else anchor_index;
    return withColumnAt(DeviceDataFrame, input, name, data, target_index);
}

pub fn withColumnAfter(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    data: anytype,
    after_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const maybe_replace_index = input.columnIndex(name);
    const anchor_index = input.columnIndex(after_name) orelse return error.ColumnNotFound;
    if (maybe_replace_index) |replace_index| {
        if (replace_index == anchor_index) return withColumnAt(DeviceDataFrame, input, name, data, replace_index);
    }

    const target_index = if (maybe_replace_index != null and maybe_replace_index.? < anchor_index) anchor_index else anchor_index + 1;
    return withColumnAt(DeviceDataFrame, input, name, data, target_index);
}

pub fn copyColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    source_name: []const u8,
    new_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(source_name);
    return withColumn(DeviceDataFrame, input, new_name, source.*);
}

pub fn copyColumnAt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    source_name: []const u8,
    new_name: []const u8,
    target_index: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(source_name);
    return withColumnAt(DeviceDataFrame, input, new_name, source.*, target_index);
}

pub fn copyColumnBefore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    source_name: []const u8,
    new_name: []const u8,
    before_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(source_name);
    return withColumnBefore(DeviceDataFrame, input, new_name, source.*, before_name);
}

pub fn copyColumnAfter(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    source_name: []const u8,
    new_name: []const u8,
    after_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(source_name);
    return withColumnAfter(DeviceDataFrame, input, new_name, source.*, after_name);
}

pub fn castColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    dtype_value: array_mod.DType,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(name);
    var casted = try source.castToDType(dtype_value);
    defer casted.deinit();
    return withColumn(DeviceDataFrame, input, name, casted);
}

pub fn fillNullColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(name);
    var filled = try source.fillNullWithScalar(scalar);
    defer filled.deinit();
    return withColumn(DeviceDataFrame, input, name, filled);
}

pub fn withColumnFillNull(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(input_name);
    var filled = try source.fillNullWithScalar(scalar);
    defer filled.deinit();
    return withColumn(DeviceDataFrame, input, output_name, filled);
}

pub fn withColumnFillNullScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNull(DeviceDataFrame, input, output_name, input_name, scalar);
}

const FillNullDirection = enum {
    forward,
    backward,
};

fn fillNullDirectionalTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: anytype,
    direction: FillNullDirection,
) array_mod.ArrayError!@TypeOf(column) {
    const maybe_validity = try validityValues(column, allocator);
    const existing_validity = maybe_validity orelse return column.clone();
    defer allocator.free(existing_validity);

    const values = try column.toOwnedSlice(allocator);
    defer allocator.free(values);
    const validity = try allocator.dupe(bool, existing_validity);
    defer allocator.free(validity);

    switch (direction) {
        .forward => {
            var last_valid: ?T = null;
            for (values, validity) |*value, *valid| {
                if (valid.*) {
                    last_valid = value.*;
                } else if (last_valid) |replacement| {
                    value.* = replacement;
                    valid.* = true;
                }
            }
        },
        .backward => {
            var next_valid: ?T = null;
            var index = values.len;
            while (index > 0) {
                index -= 1;
                if (validity[index]) {
                    next_valid = values[index];
                } else if (next_valid) |replacement| {
                    values[index] = replacement;
                    validity[index] = true;
                }
            }
        },
    }

    if (countNulls(validity) == 0) return @TypeOf(column).fromSlice(allocator, values, column.device());
    return @TypeOf(column).fromSliceWithValidity(allocator, values, validity, column.device());
}

fn fillNullDirectionalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    direction: FillNullDirection,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(input_name);
    return switch (source.*) {
        inline else => |typed, tag| blk: {
            const T = tag.Type();
            const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
            var filled: DeviceColumn = @unionInit(
                DeviceColumn,
                @tagName(tag),
                try fillNullDirectionalTyped(T, input.allocator, typed, direction),
            );
            defer filled.deinit();
            break :blk try withColumn(DeviceDataFrame, input, output_name, filled);
        },
    };
}

pub fn fillNullForwardColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNullDirectionalColumn(DeviceDataFrame, input, name, name, .forward);
}

pub fn fillNullBackwardColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNullDirectionalColumn(DeviceDataFrame, input, name, name, .backward);
}

pub fn withColumnFillNullForward(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNullDirectionalColumn(DeviceDataFrame, input, output_name, input_name, .forward);
}

pub fn withColumnFillNullBackward(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNullDirectionalColumn(DeviceDataFrame, input, output_name, input_name, .backward);
}

fn nullIfScalarTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: anytype,
    target: T,
) array_mod.ArrayError!@TypeOf(column) {
    const ColumnType = @TypeOf(column);
    const values = try column.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const validity = try allocator.alloc(bool, values.len);
    defer allocator.free(validity);
    if (maybe_validity) |existing| {
        @memcpy(validity, existing);
    } else {
        @memset(validity, true);
    }

    for (values, validity) |value, *valid| {
        if (valid.* and std.meta.eql(value, target)) valid.* = false;
    }

    if (countNulls(validity) == 0) return ColumnType.fromSlice(allocator, values, column.device());
    return ColumnType.fromSliceWithValidity(allocator, values, validity, column.device());
}

fn nullIfValuesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: anytype,
    targets: []const T,
) array_mod.ArrayError!@TypeOf(column) {
    const ColumnType = @TypeOf(column);
    const values = try column.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const validity = try allocator.alloc(bool, values.len);
    defer allocator.free(validity);
    if (maybe_validity) |existing| {
        @memcpy(validity, existing);
    } else {
        @memset(validity, true);
    }

    for (values, validity) |value, *valid| {
        if (!valid.*) continue;
        for (targets) |target| {
            if (std.meta.eql(value, target)) {
                valid.* = false;
                break;
            }
        }
    }

    if (countNulls(validity) == 0) return ColumnType.fromSlice(allocator, values, column.device());
    return ColumnType.fromSliceWithValidity(allocator, values, validity, column.device());
}

fn nullIfColumnCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(input_name);
    return switch (scalar) {
        inline else => |target, tag| blk: {
            if (source.dtype() != tag) return error.TypeUnsupported;
            const T = @TypeOf(target);
            const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
            var nullified: DeviceColumn = @unionInit(
                DeviceColumn,
                @tagName(tag),
                try nullIfScalarTyped(T, input.allocator, @field(source.*, @tagName(tag)), target),
            );
            defer nullified.deinit();
            break :blk try withColumn(DeviceDataFrame, input, output_name, nullified);
        },
    };
}

fn nullIfValuesColumnCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    test_values: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(input_name);
    if (source.dtype() != test_values.dtype()) return error.TypeUnsupported;
    if (!source.device().sameDevice(test_values.device())) return error.InvalidDevice;
    return switch (source.*) {
        inline else => |typed, tag| blk: {
            const T = tag.Type();
            const targets = try @field(test_values, @tagName(tag)).toOwnedSlice(input.allocator);
            defer input.allocator.free(targets);
            const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
            var nullified: DeviceColumn = @unionInit(
                DeviceColumn,
                @tagName(tag),
                try nullIfValuesTyped(T, input.allocator, typed, targets),
            );
            defer nullified.deinit();
            break :blk try withColumn(DeviceDataFrame, input, output_name, nullified);
        },
    };
}

pub fn nullIfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfColumnCore(DeviceDataFrame, input, name, name, scalar);
}

pub fn nullIfColumnScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfColumn(DeviceDataFrame, input, name, scalar);
}

pub fn withColumnNullIf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfColumnCore(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnNullIfScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnNullIf(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn nullIfValuesColumnWithDeviceColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    test_values: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfValuesColumnCore(DeviceDataFrame, input, name, name, test_values);
}

pub fn nullIfValuesColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    comptime T: type,
    values: []const T,
) DeviceFrameArrayError!DeviceDataFrame {
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var test_values = try DeviceColumn.fromSlice(T, input.allocator, values, input.device);
    defer test_values.deinit();
    return nullIfValuesColumnWithDeviceColumn(DeviceDataFrame, input, name, test_values);
}

pub fn withColumnNullIfValuesWithDeviceColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    test_values: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfValuesColumnCore(DeviceDataFrame, input, output_name, input_name, test_values);
}

pub fn withColumnNullIfValues(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    comptime T: type,
    values: []const T,
) DeviceFrameArrayError!DeviceDataFrame {
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var test_values = try DeviceColumn.fromSlice(T, input.allocator, values, input.device);
    defer test_values.deinit();
    return withColumnNullIfValuesWithDeviceColumn(DeviceDataFrame, input, output_name, input_name, test_values);
}

fn nullIfNumericPredicateTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: anytype,
    comptime predicate: RowNumericPredicate,
) array_mod.ArrayError!@TypeOf(column) {
    const ColumnType = @TypeOf(column);
    const values = try column.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const validity = try allocator.alloc(bool, values.len);
    defer allocator.free(validity);
    if (maybe_validity) |existing| {
        @memcpy(validity, existing);
    } else {
        @memset(validity, true);
    }

    for (values, validity) |value, *valid| {
        if (valid.* and rowNumericPredicateMatches(T, value, predicate)) valid.* = false;
    }

    if (countNulls(validity) == 0) return ColumnType.fromSlice(allocator, values, column.device());
    return ColumnType.fromSliceWithValidity(allocator, values, validity, column.device());
}

fn nullIfNumericPredicateColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(input_name);
    return switch (source.*) {
        inline else => |typed, tag| blk: {
            const T = @TypeOf(typed).Scalar;
            const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
            var nullified: DeviceColumn = @unionInit(
                DeviceColumn,
                @tagName(tag),
                try nullIfNumericPredicateTyped(T, input.allocator, typed, predicate),
            );
            defer nullified.deinit();
            break :blk try withColumn(DeviceDataFrame, input, output_name, nullified);
        },
    };
}

pub fn nullIfNaNColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .nan);
}

pub fn withColumnNullIfNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .nan);
}

pub fn nullIfInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .inf);
}

pub fn withColumnNullIfInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .inf);
}

pub fn nullIfPositiveInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .positive_inf);
}

pub fn withColumnNullIfPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .positive_inf);
}

pub fn nullIfNegativeInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .negative_inf);
}

pub fn withColumnNullIfNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .negative_inf);
}

pub fn nullIfZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .zero);
}

pub fn withColumnNullIfZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .zero);
}

pub fn nullIfPositiveZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .positive_zero);
}

pub fn withColumnNullIfPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .positive_zero);
}

pub fn nullIfNegativeZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .negative_zero);
}

pub fn withColumnNullIfNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .negative_zero);
}

pub fn nullIfNonZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .non_zero);
}

pub fn withColumnNullIfNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .non_zero);
}

pub fn nullIfPositiveColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .positive);
}

pub fn withColumnNullIfPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .positive);
}

pub fn nullIfSignBitColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .signbit);
}

pub fn withColumnNullIfSignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .signbit);
}

pub fn nullIfNegativeColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .negative);
}

pub fn withColumnNullIfNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .negative);
}

pub fn nullIfFiniteColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .finite);
}

pub fn withColumnNullIfFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .finite);
}

pub fn nullIfNormalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .normal);
}

pub fn withColumnNullIfNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .normal);
}

pub fn nullIfSubnormalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .subnormal);
}

pub fn withColumnNullIfSubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .subnormal);
}

pub fn nullIfNonFiniteColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, name, name, .non_finite);
}

pub fn withColumnNullIfNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return nullIfNumericPredicateColumn(DeviceDataFrame, input, output_name, input_name, .non_finite);
}

fn fillNumericPredicateTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: anytype,
    replacement: T,
    comptime predicate: RowNumericPredicate,
) array_mod.ArrayError!@TypeOf(column) {
    const ColumnType = @TypeOf(column);
    const values = try column.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    for (values, 0..) |*slot, row| {
        const valid = if (maybe_validity) |validity| validity[row] else true;
        if (valid and rowNumericPredicateMatches(T, slot.*, predicate)) slot.* = replacement;
    }

    if (maybe_validity) |validity| {
        return ColumnType.fromSliceWithValidity(allocator, values, validity, column.device());
    }
    return ColumnType.fromSlice(allocator, values, column.device());
}

fn fillNumericPredicateColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(name);
    return switch (scalar) {
        inline else => |replacement, tag| blk: {
            if (source.dtype() != tag) return error.TypeUnsupported;
            const T = @TypeOf(replacement);
            const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
            var filled: DeviceColumn = @unionInit(
                DeviceColumn,
                @tagName(tag),
                try fillNumericPredicateTyped(T, input.allocator, @field(source.*, @tagName(tag)), replacement, predicate),
            );
            defer filled.deinit();
            break :blk try withColumn(DeviceDataFrame, input, name, filled);
        },
    };
}

fn withColumnFillNumericPredicate(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(input_name);
    return switch (scalar) {
        inline else => |replacement, tag| blk: {
            if (source.dtype() != tag) return error.TypeUnsupported;
            const T = @TypeOf(replacement);
            const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
            var filled: DeviceColumn = @unionInit(
                DeviceColumn,
                @tagName(tag),
                try fillNumericPredicateTyped(T, input.allocator, @field(source.*, @tagName(tag)), replacement, predicate),
            );
            defer filled.deinit();
            break :blk try withColumn(DeviceDataFrame, input, output_name, filled);
        },
    };
}

pub fn withColumnFillNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .nan);
}

pub fn withColumnFillNaNScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNaN(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .inf);
}

pub fn withColumnFillInfScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillInf(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .positive_inf);
}

pub fn withColumnFillPositiveInfScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillPositiveInf(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .negative_inf);
}

pub fn withColumnFillNegativeInfScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNegativeInf(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .zero);
}

pub fn withColumnFillZeroScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillZero(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .positive_zero);
}

pub fn withColumnFillPositiveZeroScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillPositiveZero(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .negative_zero);
}

pub fn withColumnFillNegativeZeroScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNegativeZero(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .non_zero);
}

pub fn withColumnFillNonZeroScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNonZero(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .positive);
}

pub fn withColumnFillPositiveScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillPositive(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillSignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .signbit);
}

pub fn withColumnFillSignBitScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillSignBit(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .negative);
}

pub fn withColumnFillNegativeScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNegative(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .finite);
}

pub fn withColumnFillFiniteScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillFinite(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .normal);
}

pub fn withColumnFillNormalScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNormal(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillSubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .subnormal);
}

pub fn withColumnFillSubnormalScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillSubnormal(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn withColumnFillNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNumericPredicate(DeviceDataFrame, input, output_name, input_name, scalar, .non_finite);
}

pub fn withColumnFillNonFiniteScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_name: []const u8,
    input_name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return withColumnFillNonFinite(DeviceDataFrame, input, output_name, input_name, scalar);
}

pub fn fillNaNColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .nan);
}

pub fn fillInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .inf);
}

pub fn fillPositiveInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .positive_inf);
}

pub fn fillNegativeInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .negative_inf);
}

pub fn fillZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .zero);
}

pub fn fillPositiveZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .positive_zero);
}

pub fn fillNegativeZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .negative_zero);
}

pub fn fillNonZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .non_zero);
}

pub fn fillPositiveColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .positive);
}

pub fn fillSignBitColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .signbit);
}

pub fn fillNegativeColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .negative);
}

pub fn fillFiniteColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .finite);
}

pub fn fillNormalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .normal);
}

pub fn fillSubnormalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .subnormal);
}

pub fn fillNonFiniteColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillNumericPredicateColumn(DeviceDataFrame, input, name, scalar, .non_finite);
}

pub fn coalesceColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    primary_name: []const u8,
    fallback_name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const primary = try input.column(primary_name);
    const fallback = try input.column(fallback_name);
    if (primary.dtype() != fallback.dtype()) return error.TypeMismatch;
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var coalesced: DeviceColumn = switch (primary.*) {
        .bool => |typed| @unionInit(DeviceColumn, "bool", try coalesceTypedColumns(bool, typed, fallback.bool)),
        .i8 => |typed| @unionInit(DeviceColumn, "i8", try coalesceTypedColumns(i8, typed, fallback.i8)),
        .i16 => |typed| @unionInit(DeviceColumn, "i16", try coalesceTypedColumns(i16, typed, fallback.i16)),
        .i32 => |typed| @unionInit(DeviceColumn, "i32", try coalesceTypedColumns(i32, typed, fallback.i32)),
        .i64 => |typed| @unionInit(DeviceColumn, "i64", try coalesceTypedColumns(i64, typed, fallback.i64)),
        .u8 => |typed| @unionInit(DeviceColumn, "u8", try coalesceTypedColumns(u8, typed, fallback.u8)),
        .u16 => |typed| @unionInit(DeviceColumn, "u16", try coalesceTypedColumns(u16, typed, fallback.u16)),
        .u32 => |typed| @unionInit(DeviceColumn, "u32", try coalesceTypedColumns(u32, typed, fallback.u32)),
        .u64 => |typed| @unionInit(DeviceColumn, "u64", try coalesceTypedColumns(u64, typed, fallback.u64)),
        .usize => |typed| @unionInit(DeviceColumn, "usize", try coalesceTypedColumns(usize, typed, fallback.usize)),
        .isize => |typed| @unionInit(DeviceColumn, "isize", try coalesceTypedColumns(isize, typed, fallback.isize)),
        .f16 => |typed| @unionInit(DeviceColumn, "f16", try coalesceTypedColumns(f16, typed, fallback.f16)),
        .f32 => |typed| @unionInit(DeviceColumn, "f32", try coalesceTypedColumns(f32, typed, fallback.f32)),
        .f64 => |typed| @unionInit(DeviceColumn, "f64", try coalesceTypedColumns(f64, typed, fallback.f64)),
        .bf16 => |typed| @unionInit(DeviceColumn, "bf16", try coalesceTypedColumns(array_mod.BFloat16, typed, fallback.bf16)),
        .c64 => |typed| @unionInit(DeviceColumn, "c64", try coalesceTypedColumns(array_mod.Complex64, typed, fallback.c64)),
        .c128 => |typed| @unionInit(DeviceColumn, "c128", try coalesceTypedColumns(array_mod.Complex128, typed, fallback.c128)),
    };
    defer coalesced.deinit();
    return withColumn(DeviceDataFrame, input, output_name, coalesced);
}

pub fn coalesceColumnsMany(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    if (names.len == 0) return error.LengthMismatch;

    const first = try input.column(names[0]);
    var coalesced = try first.clone();
    errdefer coalesced.deinit();

    for (names[1..]) |name| {
        const fallback = try input.column(name);
        if (coalesced.dtype() != fallback.dtype()) return error.TypeMismatch;

        const next = try coalesceJoinKeys(coalesced, fallback.*);
        coalesced.deinit();
        coalesced = next;
    }

    defer coalesced.deinit();
    return withColumn(DeviceDataFrame, input, output_name, coalesced);
}

pub fn coalesceManyColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return coalesceColumnsMany(DeviceDataFrame, input, names, output_name);
}

pub fn coalesceFirstValidColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return coalesceColumnsMany(DeviceDataFrame, input, names, output_name);
}

pub fn isNullColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(name);
    const values = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(values);
    @memset(values, false);

    switch (source.*) {
        inline else => |typed| {
            const maybe_validity = try validityValues(typed, input.allocator);
            defer if (maybe_validity) |validity| input.allocator.free(validity);
            if (maybe_validity) |validity| {
                for (values, validity) |*slot, valid| slot.* = !valid;
            }
        },
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(bool, input.allocator, values, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn isValidColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(name);
    const values = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(values);
    @memset(values, true);

    switch (source.*) {
        inline else => |typed| {
            const maybe_validity = try validityValues(typed, input.allocator);
            defer if (maybe_validity) |validity| input.allocator.free(validity);
            if (maybe_validity) |validity| @memcpy(values, validity);
        },
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(bool, input.allocator, values, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

fn isNanValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return std.math.isNan(value.toF32());
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isNan(value.re) or std.math.isNan(value.im);
    return switch (@typeInfo(T)) {
        .float => std.math.isNan(value),
        else => false,
    };
}

fn isInfValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return std.math.isInf(value.toF32());
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isInf(value.re) or std.math.isInf(value.im);
    return switch (@typeInfo(T)) {
        .float => std.math.isInf(value),
        else => false,
    };
}

// Match the Array signed-Inf predicates for non-native scalar layouts: BF16 is
// widened to f32 for classification, and complex values are flagged when either
// component carries the requested signed infinity.  Row validity is handled by
// withNumericPredicateColumn so null rows always emit false.
fn isPositiveInfValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return std.math.isPositiveInf(value.toF32());
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isPositiveInf(value.re) or std.math.isPositiveInf(value.im);
    return switch (@typeInfo(T)) {
        .float => std.math.isPositiveInf(value),
        else => false,
    };
}

fn isNegativeInfValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return std.math.isNegativeInf(value.toF32());
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isNegativeInf(value.re) or std.math.isNegativeInf(value.im);
    return switch (@typeInfo(T)) {
        .float => std.math.isNegativeInf(value),
        else => false,
    };
}

fn isFiniteValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return std.math.isFinite(value.toF32());
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isFinite(value.re) and std.math.isFinite(value.im);
    return switch (@typeInfo(T)) {
        .float => std.math.isFinite(value),
        .int, .comptime_int, .bool => true,
        else => true,
    };
}

fn isZeroValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return value.toF32() == 0;
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return value.re == 0 and value.im == 0;
    return switch (@typeInfo(T)) {
        .float, .comptime_float, .int, .comptime_int => value == 0,
        .bool => !value,
        else => false,
    };
}

fn isPositiveZeroValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) {
        const widened = value.toF32();
        return widened == 0 and !std.math.signbit(widened);
    }
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) {
        // A complex value has no single sign bit.  Treat it as positive zero
        // only when the whole value is zero and neither component carries the
        // IEEE negative-zero bit; `isNegativeZeroValue` flags the mixed-sign
        // zero cases so they remain visible to data-quality checks.
        return value.re == 0 and value.im == 0 and !std.math.signbit(value.re) and !std.math.signbit(value.im);
    }
    return switch (@typeInfo(T)) {
        .float, .comptime_float => value == 0 and !std.math.signbit(value),
        else => false,
    };
}

fn isNegativeZeroValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) {
        const widened = value.toF32();
        return widened == 0 and std.math.signbit(widened);
    }
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) {
        // Flag complex zeros when either component explicitly carries a
        // negative-zero sign.  Non-zero complex values are excluded even if one
        // component happens to be -0, matching the scalar "signed zero" intent.
        return value.re == 0 and value.im == 0 and (std.math.signbit(value.re) or std.math.signbit(value.im));
    }
    return switch (@typeInfo(T)) {
        .float, .comptime_float => value == 0 and std.math.signbit(value),
        else => false,
    };
}

fn isNonZeroValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return value.toF32() != 0;
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return value.re != 0 or value.im != 0;
    return switch (@typeInfo(T)) {
        .float, .comptime_float, .int, .comptime_int => value != 0,
        .bool => value,
        else => false,
    };
}

fn isPositiveValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return value.toF32() > 0;
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return false;
    return switch (@typeInfo(T)) {
        .float, .comptime_float, .int, .comptime_int => value > 0,
        .bool => false,
        else => false,
    };
}

fn isSignBitValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return std.math.signbit(value.toF32());
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return false;
    return switch (@typeInfo(T)) {
        .float, .comptime_float, .int, .comptime_int => std.math.signbit(value),
        .bool => false,
        else => false,
    };
}

fn isNegativeValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return value.toF32() < 0;
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return false;
    return switch (@typeInfo(T)) {
        .float, .comptime_float, .int, .comptime_int => value < 0,
        .bool => false,
        else => false,
    };
}

fn isNormalValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return std.math.isNormal(value.toF32());
    // Treat complex values as normal only when both components are normal IEEE
    // floats. This keeps the predicate stricter than finite and makes zero or
    // subnormal components visible to data-quality checks.
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return std.math.isNormal(value.re) and std.math.isNormal(value.im);
    return switch (@typeInfo(T)) {
        .float => std.math.isNormal(value),
        else => false,
    };
}

fn isSubnormalFloat(comptime T: type, value: T) bool {
    return std.math.isFinite(value) and !std.math.isNormal(value) and value != 0;
}

fn isSubnormalValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return isSubnormalFloat(f32, value.toF32());
    // Subnormal is a presence-style predicate for complex values: flag the row
    // when either component is a finite non-zero subnormal. Zeros, NaNs, and
    // infinities are deliberately excluded.
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return isSubnormalFloat(@TypeOf(value.re), value.re) or isSubnormalFloat(@TypeOf(value.im), value.im);
    return switch (@typeInfo(T)) {
        .float => isSubnormalFloat(T, value),
        else => false,
    };
}

fn withNumericPredicateColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
    comptime predicate: enum { nan, inf, positive_inf, negative_inf, zero, positive_zero, negative_zero, non_zero, positive, signbit, negative, finite, normal, subnormal, non_finite },
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(name);
    const values = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(values);

    switch (source.*) {
        inline else => |typed| {
            const host_values = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(host_values);
            const maybe_validity = try validityValues(typed, input.allocator);
            defer if (maybe_validity) |validity| input.allocator.free(validity);
            for (values, host_values, 0..) |*slot, value, row| {
                if (maybe_validity) |validity| {
                    if (!validity[row]) {
                        slot.* = false;
                        continue;
                    }
                }
                slot.* = switch (predicate) {
                    .nan => isNanValue(@TypeOf(value), value),
                    .inf => isInfValue(@TypeOf(value), value),
                    .positive_inf => isPositiveInfValue(@TypeOf(value), value),
                    .negative_inf => isNegativeInfValue(@TypeOf(value), value),
                    .zero => isZeroValue(@TypeOf(value), value),
                    .positive_zero => isPositiveZeroValue(@TypeOf(value), value),
                    .negative_zero => isNegativeZeroValue(@TypeOf(value), value),
                    .non_zero => isNonZeroValue(@TypeOf(value), value),
                    .positive => isPositiveValue(@TypeOf(value), value),
                    .signbit => isSignBitValue(@TypeOf(value), value),
                    .negative => isNegativeValue(@TypeOf(value), value),
                    .finite => isFiniteValue(@TypeOf(value), value),
                    .normal => isNormalValue(@TypeOf(value), value),
                    .subnormal => isSubnormalValue(@TypeOf(value), value),
                    .non_finite => !isFiniteValue(@TypeOf(value), value),
                };
            }
        },
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(bool, input.allocator, values, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn isNanColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .nan);
}

pub fn isZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .zero);
}

pub fn isPositiveZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .positive_zero);
}

pub fn isNegativeZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .negative_zero);
}

pub fn isNonZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .non_zero);
}

pub fn isPositiveColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .positive);
}

pub fn isSignBitColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .signbit);
}

pub fn isNegativeColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .negative);
}

pub fn isFiniteColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .finite);
}

pub fn isNormalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .normal);
}

pub fn isSubnormalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .subnormal);
}

pub fn isNonFiniteColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .non_finite);
}

pub fn isInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .inf);
}

pub fn isPositiveInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .positive_inf);
}

pub fn isNegativeInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .negative_inf);
}

fn withRowValidityCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime count_valid: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (counts, validity) |*slot, valid| {
                        if (valid == count_valid) slot.* += 1;
                    }
                } else if (count_valid) {
                    for (counts) |*slot| slot.* += 1;
                }
            },
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(i64, input.allocator, counts, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowNullCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityCount(DeviceDataFrame, input, names, output_name, false);
}

pub fn withRowValidCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityCount(DeviceDataFrame, input, names, output_name, true);
}

const RowValidityReduction = enum { any_null, all_null, any_valid, all_valid };

fn rowValidityReductionSeed(comptime reduction: RowValidityReduction) bool {
    return switch (reduction) {
        .any_null, .any_valid => false,
        .all_null, .all_valid => true,
    };
}

fn rowValidityReductionTargetValid(comptime reduction: RowValidityReduction) bool {
    return switch (reduction) {
        .any_null, .all_null => false,
        .any_valid, .all_valid => true,
    };
}

fn rowValidityReductionIsAny(comptime reduction: RowValidityReduction) bool {
    return switch (reduction) {
        .any_null, .any_valid => true,
        .all_null, .all_valid => false,
    };
}

fn withRowValidityReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowValidityReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const values = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(values);
    @memset(values, rowValidityReductionSeed(reduction));

    const target_valid = rowValidityReductionTargetValid(reduction);
    const is_any = rowValidityReductionIsAny(reduction);
    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (values, validity) |*slot, valid| {
                        const matches = valid == target_valid;
                        slot.* = if (is_any) slot.* or matches else slot.* and matches;
                    }
                } else {
                    const matches = target_valid;
                    for (values) |*slot| {
                        slot.* = if (is_any) slot.* or matches else slot.* and matches;
                    }
                }
            },
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(bool, input.allocator, values, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowAnyNull(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityReduction(DeviceDataFrame, input, names, output_name, .any_null);
}

pub fn withRowAllNull(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityReduction(DeviceDataFrame, input, names, output_name, .all_null);
}

pub fn withRowAnyValid(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityReduction(DeviceDataFrame, input, names, output_name, .any_valid);
}

pub fn withRowAllValid(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityReduction(DeviceDataFrame, input, names, output_name, .all_valid);
}

fn withRowCumulativeValidityReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowValidityReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    try validateRowCumulativeOutputNames(output_names, check_names.len);

    const running = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(running);
    @memset(running, rowValidityReductionSeed(reduction));

    const target_valid = rowValidityReductionTargetValid(reduction);
    const is_any = rowValidityReductionIsAny(reduction);
    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (running, validity) |*slot, valid| {
                        const matches = valid == target_valid;
                        slot.* = if (is_any) slot.* or matches else slot.* and matches;
                    }
                } else {
                    const matches = target_valid;
                    for (running) |*slot| {
                        slot.* = if (is_any) slot.* or matches else slot.* and matches;
                    }
                }
            },
        }

        var column = try DeviceColumn.fromSlice(bool, input.allocator, running, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeAnyNull(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityReduction(DeviceDataFrame, input, names, output_names, .any_null);
}

pub fn withRowCumAnyNull(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNull(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyNull(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNull(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllNull(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityReduction(DeviceDataFrame, input, names, output_names, .all_null);
}

pub fn withRowCumAllNull(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNull(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllNull(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNull(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAnyValid(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityReduction(DeviceDataFrame, input, names, output_names, .any_valid);
}

pub fn withRowCumAnyValid(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyValid(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyValid(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyValid(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllValid(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityReduction(DeviceDataFrame, input, names, output_names, .all_valid);
}

pub fn withRowCumAllValid(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllValid(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllValid(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllValid(DeviceDataFrame, input, names, output_names);
}

fn withRowCumulativeValidityCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime count_valid: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (counts, validity) |*slot, valid| {
                        if (valid == count_valid) slot.* += 1;
                    }
                } else if (count_valid) {
                    for (counts) |*slot| slot.* += 1;
                }
            },
        }

        var column = try DeviceColumn.fromSlice(i64, input.allocator, counts, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeNullCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityCount(DeviceDataFrame, input, names, output_names, false);
}

pub fn withRowCumNullCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNullCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNullCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNullCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeValidCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityCount(DeviceDataFrame, input, names, output_names, true);
}

pub fn withRowCumValidCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixValidCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidCount(DeviceDataFrame, input, names, output_names);
}

fn withRowCumulativeValidityRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime count_valid: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);
    const ratios = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(ratios);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |name, output_name, col_index| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (counts, validity) |*slot, valid| {
                        if (valid == count_valid) slot.* += 1;
                    }
                } else if (count_valid) {
                    for (counts) |*slot| slot.* += 1;
                }
            },
        }

        const denominator: f64 = @floatFromInt(col_index + 1);
        for (ratios, counts) |*ratio, count| {
            ratio.* = @as(f64, @floatFromInt(count)) / denominator;
        }

        var column = try DeviceColumn.fromSlice(f64, input.allocator, ratios, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeNullRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityRatio(DeviceDataFrame, input, names, output_names, false);
}

pub fn withRowCumNullRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNullRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNullRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNullRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeValidRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityRatio(DeviceDataFrame, input, names, output_names, true);
}

pub fn withRowCumValidRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixValidRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidRatio(DeviceDataFrame, input, names, output_names);
}

fn withRowValidityRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime count_valid: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (counts, validity) |*slot, valid| {
                        if (valid == count_valid) slot.* += 1;
                    }
                } else if (count_valid) {
                    for (counts) |*slot| slot.* += 1;
                }
            },
        }
    }

    const ratios = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(ratios);
    const denominator: f64 = @floatFromInt(check_names.len);
    for (ratios, counts) |*ratio, count| {
        ratio.* = if (check_names.len == 0) std.math.nan(f64) else @as(f64, @floatFromInt(count)) / denominator;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(f64, input.allocator, ratios, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowNullRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityRatio(DeviceDataFrame, input, names, output_name, false);
}

pub fn withRowValidRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityRatio(DeviceDataFrame, input, names, output_name, true);
}

pub fn withRowPairCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    if (lhs_names.len == 0 or lhs_names.len != rhs_names.len) return error.LengthMismatch;

    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    for (lhs_names, rhs_names) |lhs_name, rhs_name| {
        const lhs_source = try input.column(lhs_name);
        const rhs_source = try input.column(rhs_name);

        switch (lhs_source.*) {
            inline else => |lhs_typed| {
                const maybe_lhs_validity = try validityValues(lhs_typed, input.allocator);
                defer if (maybe_lhs_validity) |mask| input.allocator.free(mask);

                switch (rhs_source.*) {
                    inline else => |rhs_typed| {
                        const maybe_rhs_validity = try validityValues(rhs_typed, input.allocator);
                        defer if (maybe_rhs_validity) |mask| input.allocator.free(mask);

                        for (counts, 0..) |*count, row| {
                            const lhs_valid = if (maybe_lhs_validity) |mask| mask[row] else true;
                            const rhs_valid = if (maybe_rhs_validity) |mask| mask[row] else true;
                            if (lhs_valid and rhs_valid) count.* += 1;
                        }
                    },
                }
            },
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(i64, input.allocator, counts, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

const RowWeightedPairSupportReduction = enum { weight_sum, positive_count, effective_n };

fn withRowWeightedPairSupport(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowWeightedPairSupportReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    if (lhs_names.len == 0 or lhs_names.len != rhs_names.len or lhs_names.len != weight_names.len) return error.LengthMismatch;

    const weight_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weight_sums);
    const weight_square_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weight_square_sums);
    const positive_counts = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(positive_counts);
    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(weight_sums, 0.0);
    @memset(weight_square_sums, 0.0);
    @memset(positive_counts, 0.0);
    @memset(values, 0.0);
    @memset(validity, false);

    for (lhs_names, rhs_names, weight_names) |lhs_name, rhs_name, weight_name| {
        const lhs_source = try input.column(lhs_name);
        const rhs_source = try input.column(rhs_name);
        const weight_source = try input.column(weight_name);

        var lhs_column = try ownedRealF64Column(input.allocator, lhs_source);
        defer lhs_column.deinit();
        var rhs_column = try ownedRealF64Column(input.allocator, rhs_source);
        defer rhs_column.deinit();
        var weight_column = try ownedRealF64Column(input.allocator, weight_source);
        defer weight_column.deinit();

        for (lhs_column.values, rhs_column.values, weight_column.values, 0..) |_, _, weight, row| {
            const lhs_valid = if (lhs_column.validity) |mask| mask[row] else true;
            const rhs_valid = if (rhs_column.validity) |mask| mask[row] else true;
            const weight_valid = if (weight_column.validity) |mask| mask[row] else true;
            if (!lhs_valid or !rhs_valid or !weight_valid) continue;
            if (weight < 0.0) return error.InvalidShape;
            validity[row] = true;
            weight_sums[row] += weight;
            if (weight > 0.0) {
                weight_square_sums[row] += weight * weight;
                positive_counts[row] += 1.0;
            }
        }
    }

    for (values, weight_sums, weight_square_sums, positive_counts, validity) |*value, weight_sum, weight_square_sum, positive_count, valid| {
        if (!valid) continue;
        value.* = switch (reduction) {
            .weight_sum => weight_sum,
            .positive_count => positive_count,
            .effective_n => rowWeightedEffectiveN(weight_sum, weight_square_sum),
        };
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedPairWeightSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairSupport(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, .weight_sum);
}

pub fn withRowWeightedPairPositiveCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairSupport(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, .positive_count);
}

pub fn withRowWeightedPairEffectiveN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairSupport(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, .effective_n);
}

pub const withRowWeightedPairEffectiveCount = withRowWeightedPairEffectiveN;

const RowCumulativeWeightedPairSupportReduction = enum { weight_sum, positive_count, effective_n };

fn withRowCumulativeWeightedPairSupport(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowCumulativeWeightedPairSupportReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    if (lhs_names.len == 0 or lhs_names.len != rhs_names.len or lhs_names.len != weight_names.len) return error.LengthMismatch;
    try validateRowCumulativeWeightedOutputs(output_names, lhs_names.len);

    const running_weight_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_weight_sums);
    const running_weight_square_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_weight_square_sums);
    const running_positive_counts = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_positive_counts);
    const cumulative = try input.allocator.alloc(f64, input.rows * lhs_names.len);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, input.rows * lhs_names.len);
    defer input.allocator.free(cumulative_validity);
    @memset(running_weight_sums, 0.0);
    @memset(running_weight_square_sums, 0.0);
    @memset(running_positive_counts, 0.0);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (lhs_names, rhs_names, weight_names, 0..) |lhs_name, rhs_name, weight_name, col_index| {
        const lhs_source = try input.column(lhs_name);
        const rhs_source = try input.column(rhs_name);
        const weight_source = try input.column(weight_name);

        var lhs_column = try ownedRealF64Column(input.allocator, lhs_source);
        defer lhs_column.deinit();
        var rhs_column = try ownedRealF64Column(input.allocator, rhs_source);
        defer rhs_column.deinit();
        var weight_column = try ownedRealF64Column(input.allocator, weight_source);
        defer weight_column.deinit();

        for (lhs_column.values, rhs_column.values, weight_column.values, 0..) |_, _, weight, row| {
            const lhs_valid = if (lhs_column.validity) |mask| mask[row] else true;
            const rhs_valid = if (rhs_column.validity) |mask| mask[row] else true;
            const weight_valid = if (weight_column.validity) |mask| mask[row] else true;
            if (!lhs_valid or !rhs_valid or !weight_valid) continue;
            if (weight < 0.0) return error.InvalidShape;
            running_weight_sums[row] += weight;
            if (weight > 0.0) {
                running_weight_square_sums[row] += weight * weight;
                running_positive_counts[row] += 1.0;
            }
            const offset = row * lhs_names.len + col_index;
            cumulative[offset] = switch (reduction) {
                .weight_sum => running_weight_sums[row],
                .positive_count => running_positive_counts[row],
                .effective_n => rowWeightedEffectiveN(running_weight_sums[row], running_weight_square_sums[row]),
            };
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, input.rows, lhs_names.len, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedPairWeightSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairSupport(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .weight_sum);
}

pub fn withRowCumulativeWeightedPairPositiveCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairSupport(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .positive_count);
}

pub fn withRowCumulativeWeightedPairEffectiveN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairSupport(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .effective_n);
}

pub const withRowCumulativeWeightedPairEffectiveCount = withRowCumulativeWeightedPairEffectiveN;
pub const withRowCumWeightedPairWeightSum = withRowCumulativeWeightedPairWeightSum;
pub const withRowPrefixWeightedPairWeightSum = withRowCumulativeWeightedPairWeightSum;
pub const withRowCumWeightedPairPositiveCount = withRowCumulativeWeightedPairPositiveCount;
pub const withRowPrefixWeightedPairPositiveCount = withRowCumulativeWeightedPairPositiveCount;
pub const withRowCumWeightedPairEffectiveN = withRowCumulativeWeightedPairEffectiveN;
pub const withRowCumWeightedPairEffectiveCount = withRowCumulativeWeightedPairEffectiveN;
pub const withRowPrefixWeightedPairEffectiveN = withRowCumulativeWeightedPairEffectiveN;
pub const withRowPrefixWeightedPairEffectiveCount = withRowCumulativeWeightedPairEffectiveN;

pub fn withRowCumulativeWeightedDot(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    if (lhs_names.len == 0 or lhs_names.len != rhs_names.len or lhs_names.len != weight_names.len) return error.LengthMismatch;
    try validateRowCumulativeWeightedOutputs(output_names, lhs_names.len);

    const running_weight_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_weight_sums);
    const running_cross_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_cross_sums);
    const cumulative = try input.allocator.alloc(f64, input.rows * lhs_names.len);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, input.rows * lhs_names.len);
    defer input.allocator.free(cumulative_validity);
    @memset(running_weight_sums, 0.0);
    @memset(running_cross_sums, 0.0);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (lhs_names, rhs_names, weight_names, 0..) |lhs_name, rhs_name, weight_name, col_index| {
        const lhs_source = try input.column(lhs_name);
        const rhs_source = try input.column(rhs_name);
        const weight_source = try input.column(weight_name);

        var lhs_column = try ownedRealF64Column(input.allocator, lhs_source);
        defer lhs_column.deinit();
        var rhs_column = try ownedRealF64Column(input.allocator, rhs_source);
        defer rhs_column.deinit();
        var weight_column = try ownedRealF64Column(input.allocator, weight_source);
        defer weight_column.deinit();

        for (lhs_column.values, rhs_column.values, weight_column.values, 0..) |lhs, rhs, weight, row| {
            const lhs_valid = if (lhs_column.validity) |mask| mask[row] else true;
            const rhs_valid = if (rhs_column.validity) |mask| mask[row] else true;
            const weight_valid = if (weight_column.validity) |mask| mask[row] else true;
            if (!lhs_valid or !rhs_valid or !weight_valid) continue;
            if (weight < 0.0) return error.InvalidShape;
            if (weight > 0.0) {
                running_weight_sums[row] += weight;
                running_cross_sums[row] += weight * lhs * rhs;
            }
            if (!(running_weight_sums[row] > 0.0)) continue;
            const offset = row * lhs_names.len + col_index;
            cumulative[offset] = running_cross_sums[row];
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, input.rows, lhs_names.len, cumulative, cumulative_validity);
}

pub const withRowCumWeightedDot = withRowCumulativeWeightedDot;
pub const withRowPrefixWeightedDot = withRowCumulativeWeightedDot;

pub fn withRowCumulativeWeightedCosineSimilarity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    if (lhs_names.len == 0 or lhs_names.len != rhs_names.len or lhs_names.len != weight_names.len) return error.LengthMismatch;
    try validateRowCumulativeWeightedOutputs(output_names, lhs_names.len);

    const running_weight_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_weight_sums);
    const running_cross_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_cross_sums);
    const running_lhs_square_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_lhs_square_sums);
    const running_rhs_square_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_rhs_square_sums);
    const cumulative = try input.allocator.alloc(f64, input.rows * lhs_names.len);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, input.rows * lhs_names.len);
    defer input.allocator.free(cumulative_validity);
    @memset(running_weight_sums, 0.0);
    @memset(running_cross_sums, 0.0);
    @memset(running_lhs_square_sums, 0.0);
    @memset(running_rhs_square_sums, 0.0);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (lhs_names, rhs_names, weight_names, 0..) |lhs_name, rhs_name, weight_name, col_index| {
        const lhs_source = try input.column(lhs_name);
        const rhs_source = try input.column(rhs_name);
        const weight_source = try input.column(weight_name);

        var lhs_column = try ownedRealF64Column(input.allocator, lhs_source);
        defer lhs_column.deinit();
        var rhs_column = try ownedRealF64Column(input.allocator, rhs_source);
        defer rhs_column.deinit();
        var weight_column = try ownedRealF64Column(input.allocator, weight_source);
        defer weight_column.deinit();

        for (lhs_column.values, rhs_column.values, weight_column.values, 0..) |lhs, rhs, weight, row| {
            const lhs_valid = if (lhs_column.validity) |mask| mask[row] else true;
            const rhs_valid = if (rhs_column.validity) |mask| mask[row] else true;
            const weight_valid = if (weight_column.validity) |mask| mask[row] else true;
            if (!lhs_valid or !rhs_valid or !weight_valid) continue;
            if (weight < 0.0) return error.InvalidShape;
            if (weight > 0.0) {
                running_weight_sums[row] += weight;
                running_cross_sums[row] += weight * lhs * rhs;
                running_lhs_square_sums[row] += weight * lhs * lhs;
                running_rhs_square_sums[row] += weight * rhs * rhs;
            }
            if (!(running_weight_sums[row] > 0.0)) continue;
            const offset = row * lhs_names.len + col_index;
            cumulative[offset] = if (running_lhs_square_sums[row] == 0.0 or running_rhs_square_sums[row] == 0.0)
                quietNanF64()
            else
                running_cross_sums[row] / (std.math.sqrt(running_lhs_square_sums[row]) * std.math.sqrt(running_rhs_square_sums[row]));
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, input.rows, lhs_names.len, cumulative, cumulative_validity);
}

pub const withRowCumulativeWeightedCosine = withRowCumulativeWeightedCosineSimilarity;
pub const withRowCumWeightedCosineSimilarity = withRowCumulativeWeightedCosineSimilarity;
pub const withRowCumWeightedCosine = withRowCumulativeWeightedCosineSimilarity;
pub const withRowPrefixWeightedCosineSimilarity = withRowCumulativeWeightedCosineSimilarity;
pub const withRowPrefixWeightedCosine = withRowCumulativeWeightedCosineSimilarity;

const RowCumulativeWeightedPairMetricReduction = enum { squared_euclidean, euclidean, manhattan, chebyshev, canberra, bray_curtis, mean_error, mae, mse, rmse };

// Keep the cumulative pair-metric family in one engine so all variants share
// the same null/current-position validity and positive-weight prefix contract.
fn withRowCumulativeWeightedPairMetric(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowCumulativeWeightedPairMetricReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    if (lhs_names.len == 0 or lhs_names.len != rhs_names.len or lhs_names.len != weight_names.len) return error.LengthMismatch;
    try validateRowCumulativeWeightedOutputs(output_names, lhs_names.len);

    const running_weight_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(running_weight_sums);
    const needs_quadratic_state = reduction == .squared_euclidean or reduction == .euclidean or reduction == .mse or reduction == .rmse;
    const needs_abs_error_sum_state = reduction == .manhattan or reduction == .bray_curtis or reduction == .mae;
    const needs_chebyshev_state = reduction == .chebyshev;
    const needs_canberra_state = reduction == .canberra;
    const needs_bray_curtis_state = reduction == .bray_curtis;
    const needs_signed_sum_state = reduction == .mean_error;

    const running_cross_sums = try input.allocator.alloc(f64, if (needs_quadratic_state) input.rows else 0);
    defer input.allocator.free(running_cross_sums);
    const running_lhs_sums = try input.allocator.alloc(f64, if (needs_signed_sum_state) input.rows else 0);
    defer input.allocator.free(running_lhs_sums);
    const running_rhs_sums = try input.allocator.alloc(f64, if (needs_signed_sum_state) input.rows else 0);
    defer input.allocator.free(running_rhs_sums);
    const running_lhs_square_sums = try input.allocator.alloc(f64, if (needs_quadratic_state) input.rows else 0);
    defer input.allocator.free(running_lhs_square_sums);
    const running_rhs_square_sums = try input.allocator.alloc(f64, if (needs_quadratic_state) input.rows else 0);
    defer input.allocator.free(running_rhs_square_sums);
    const running_abs_error_sums = try input.allocator.alloc(f64, if (needs_abs_error_sum_state) input.rows else 0);
    defer input.allocator.free(running_abs_error_sums);
    const running_chebyshev_values = try input.allocator.alloc(f64, if (needs_chebyshev_state) input.rows else 0);
    defer input.allocator.free(running_chebyshev_values);
    const running_canberra_sums = try input.allocator.alloc(f64, if (needs_canberra_state) input.rows else 0);
    defer input.allocator.free(running_canberra_sums);
    const running_bray_denominators = try input.allocator.alloc(f64, if (needs_bray_curtis_state) input.rows else 0);
    defer input.allocator.free(running_bray_denominators);
    const cumulative = try input.allocator.alloc(f64, input.rows * lhs_names.len);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, input.rows * lhs_names.len);
    defer input.allocator.free(cumulative_validity);
    @memset(running_weight_sums, 0.0);
    @memset(running_cross_sums, 0.0);
    @memset(running_lhs_sums, 0.0);
    @memset(running_rhs_sums, 0.0);
    @memset(running_lhs_square_sums, 0.0);
    @memset(running_rhs_square_sums, 0.0);
    @memset(running_abs_error_sums, 0.0);
    @memset(running_chebyshev_values, 0.0);
    @memset(running_canberra_sums, 0.0);
    @memset(running_bray_denominators, 0.0);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (lhs_names, rhs_names, weight_names, 0..) |lhs_name, rhs_name, weight_name, col_index| {
        const lhs_source = try input.column(lhs_name);
        const rhs_source = try input.column(rhs_name);
        const weight_source = try input.column(weight_name);

        var lhs_column = try ownedRealF64Column(input.allocator, lhs_source);
        defer lhs_column.deinit();
        var rhs_column = try ownedRealF64Column(input.allocator, rhs_source);
        defer rhs_column.deinit();
        var weight_column = try ownedRealF64Column(input.allocator, weight_source);
        defer weight_column.deinit();

        for (lhs_column.values, rhs_column.values, weight_column.values, 0..) |lhs, rhs, weight, row| {
            const lhs_valid = if (lhs_column.validity) |mask| mask[row] else true;
            const rhs_valid = if (rhs_column.validity) |mask| mask[row] else true;
            const weight_valid = if (weight_column.validity) |mask| mask[row] else true;
            if (!lhs_valid or !rhs_valid or !weight_valid) continue;
            if (weight < 0.0) return error.InvalidShape;
            if (weight > 0.0) {
                running_weight_sums[row] += weight;
                switch (reduction) {
                    .squared_euclidean, .euclidean, .mse, .rmse => {
                        running_cross_sums[row] += weight * lhs * rhs;
                        running_lhs_square_sums[row] += weight * lhs * lhs;
                        running_rhs_square_sums[row] += weight * rhs * rhs;
                    },
                    .manhattan, .mae => {
                        running_abs_error_sums[row] += weight * @abs(lhs - rhs);
                    },
                    .chebyshev => {
                        running_chebyshev_values[row] = @max(running_chebyshev_values[row], @abs(lhs - rhs));
                    },
                    .canberra => {
                        const abs_sum = @abs(lhs) + @abs(rhs);
                        running_canberra_sums[row] += if (abs_sum == 0.0) 0.0 else weight * @abs(lhs - rhs) / abs_sum;
                    },
                    .bray_curtis => {
                        running_abs_error_sums[row] += weight * @abs(lhs - rhs);
                        running_bray_denominators[row] += weight * (@abs(lhs) + @abs(rhs));
                    },
                    .mean_error => {
                        running_lhs_sums[row] += weight * lhs;
                        running_rhs_sums[row] += weight * rhs;
                    },
                }
            }
            if (!(running_weight_sums[row] > 0.0)) continue;
            const offset = row * lhs_names.len + col_index;
            cumulative[offset] = switch (reduction) {
                .squared_euclidean => running_lhs_square_sums[row] + running_rhs_square_sums[row] - 2.0 * running_cross_sums[row],
                .euclidean => std.math.sqrt(running_lhs_square_sums[row] + running_rhs_square_sums[row] - 2.0 * running_cross_sums[row]),
                .mse => (running_lhs_square_sums[row] + running_rhs_square_sums[row] - 2.0 * running_cross_sums[row]) / running_weight_sums[row],
                .rmse => std.math.sqrt((running_lhs_square_sums[row] + running_rhs_square_sums[row] - 2.0 * running_cross_sums[row]) / running_weight_sums[row]),
                .manhattan => running_abs_error_sums[row],
                .chebyshev => running_chebyshev_values[row],
                .canberra => running_canberra_sums[row],
                .bray_curtis => if (running_bray_denominators[row] == 0.0) quietNanF64() else running_abs_error_sums[row] / running_bray_denominators[row],
                .mean_error => (running_lhs_sums[row] - running_rhs_sums[row]) / running_weight_sums[row],
                .mae => running_abs_error_sums[row] / running_weight_sums[row],
            };
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, input.rows, lhs_names.len, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedSquaredEuclideanDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .squared_euclidean);
}

pub const withRowCumulativeWeightedSquaredDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowCumulativeWeightedSqEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowCumWeightedSquaredEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowCumWeightedSquaredDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowCumWeightedSqEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowPrefixWeightedSquaredEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowPrefixWeightedSquaredDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowPrefixWeightedSqEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;

pub fn withRowCumulativeWeightedEuclideanDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .euclidean);
}

pub const withRowCumulativeWeightedL2Distance = withRowCumulativeWeightedEuclideanDistance;
pub const withRowCumWeightedEuclideanDistance = withRowCumulativeWeightedEuclideanDistance;
pub const withRowCumWeightedL2Distance = withRowCumulativeWeightedEuclideanDistance;
pub const withRowPrefixWeightedEuclideanDistance = withRowCumulativeWeightedEuclideanDistance;
pub const withRowPrefixWeightedL2Distance = withRowCumulativeWeightedEuclideanDistance;

pub fn withRowCumulativeWeightedManhattanDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .manhattan);
}

pub const withRowCumulativeWeightedL1Distance = withRowCumulativeWeightedManhattanDistance;
pub const withRowCumWeightedManhattanDistance = withRowCumulativeWeightedManhattanDistance;
pub const withRowCumWeightedL1Distance = withRowCumulativeWeightedManhattanDistance;
pub const withRowPrefixWeightedManhattanDistance = withRowCumulativeWeightedManhattanDistance;
pub const withRowPrefixWeightedL1Distance = withRowCumulativeWeightedManhattanDistance;

pub fn withRowCumulativeWeightedChebyshevDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .chebyshev);
}

pub const withRowCumWeightedChebyshevDistance = withRowCumulativeWeightedChebyshevDistance;
pub const withRowPrefixWeightedChebyshevDistance = withRowCumulativeWeightedChebyshevDistance;

pub fn withRowCumulativeWeightedCanberraDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .canberra);
}

pub const withRowCumWeightedCanberraDistance = withRowCumulativeWeightedCanberraDistance;
pub const withRowPrefixWeightedCanberraDistance = withRowCumulativeWeightedCanberraDistance;

pub fn withRowCumulativeWeightedBrayCurtisDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .bray_curtis);
}

pub const withRowCumWeightedBrayCurtisDistance = withRowCumulativeWeightedBrayCurtisDistance;
pub const withRowPrefixWeightedBrayCurtisDistance = withRowCumulativeWeightedBrayCurtisDistance;

pub fn withRowCumulativeWeightedMeanError(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .mean_error);
}

pub const withRowCumulativeWeightedBias = withRowCumulativeWeightedMeanError;
pub const withRowCumWeightedMeanError = withRowCumulativeWeightedMeanError;
pub const withRowCumWeightedBias = withRowCumulativeWeightedMeanError;
pub const withRowPrefixWeightedMeanError = withRowCumulativeWeightedMeanError;
pub const withRowPrefixWeightedBias = withRowCumulativeWeightedMeanError;

pub fn withRowCumulativeWeightedMae(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .mae);
}

pub const withRowCumulativeWeightedMAE = withRowCumulativeWeightedMae;
pub const withRowCumWeightedMae = withRowCumulativeWeightedMae;
pub const withRowCumWeightedMAE = withRowCumulativeWeightedMae;
pub const withRowPrefixWeightedMae = withRowCumulativeWeightedMae;
pub const withRowPrefixWeightedMAE = withRowCumulativeWeightedMae;

pub fn withRowCumulativeWeightedMse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .mse);
}

pub const withRowCumulativeWeightedMSE = withRowCumulativeWeightedMse;
pub const withRowCumWeightedMse = withRowCumulativeWeightedMse;
pub const withRowCumWeightedMSE = withRowCumulativeWeightedMse;
pub const withRowPrefixWeightedMse = withRowCumulativeWeightedMse;
pub const withRowPrefixWeightedMSE = withRowCumulativeWeightedMse;

pub fn withRowCumulativeWeightedRmse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPairMetric(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_names, .rmse);
}

pub const withRowCumulativeWeightedRMSE = withRowCumulativeWeightedRmse;
pub const withRowCumWeightedRmse = withRowCumulativeWeightedRmse;
pub const withRowCumWeightedRMSE = withRowCumulativeWeightedRmse;
pub const withRowPrefixWeightedRmse = withRowCumulativeWeightedRmse;
pub const withRowPrefixWeightedRMSE = withRowCumulativeWeightedRmse;

const RowValidityMatchIndex = enum { first_valid, last_valid, first_null, last_null };

fn withRowValidityMatchIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime search: RowValidityMatchIndex,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const indices = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(indices);
    const output_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(output_validity);
    @memset(indices, 0);
    @memset(output_validity, false);

    const target_valid = switch (search) {
        .first_valid, .last_valid => true,
        .first_null, .last_null => false,
    };

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        const output_index = std.math.cast(i64, col_index) orelse return error.InvalidShape;
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (validity, 0..) |valid, row| {
                        if (valid != target_valid) continue;
                        switch (search) {
                            .first_valid, .first_null => if (!output_validity[row]) {
                                indices[row] = output_index;
                                output_validity[row] = true;
                            },
                            .last_valid, .last_null => {
                                indices[row] = output_index;
                                output_validity[row] = true;
                            },
                        }
                    }
                } else if (target_valid) {
                    for (output_validity, 0..) |*found, row| {
                        switch (search) {
                            .first_valid => if (!found.*) {
                                indices[row] = output_index;
                                found.* = true;
                            },
                            .last_valid => {
                                indices[row] = output_index;
                                found.* = true;
                            },
                            .first_null, .last_null => unreachable,
                        }
                    }
                }
            },
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, indices, output_validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowFirstValidIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityMatchIndex(DeviceDataFrame, input, names, output_name, .first_valid);
}

pub fn withRowLastValidIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityMatchIndex(DeviceDataFrame, input, names, output_name, .last_valid);
}

pub fn withRowFirstNullIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityMatchIndex(DeviceDataFrame, input, names, output_name, .first_null);
}

pub fn withRowLastNullIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowValidityMatchIndex(DeviceDataFrame, input, names, output_name, .last_null);
}

fn withRowCumulativeValidityMatchIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime search: RowValidityMatchIndex,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const indices = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(indices);
    const output_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(output_validity);
    @memset(indices, 0);
    @memset(output_validity, false);

    const target_valid = switch (search) {
        .first_valid, .last_valid => true,
        .first_null, .last_null => false,
    };

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |name, output_name, col_index| {
        const source = try input.column(name);
        const output_index = std.math.cast(i64, col_index) orelse return error.InvalidShape;
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (validity, 0..) |valid, row| {
                        if (valid != target_valid) continue;
                        switch (search) {
                            .first_valid, .first_null => if (!output_validity[row]) {
                                indices[row] = output_index;
                                output_validity[row] = true;
                            },
                            .last_valid, .last_null => {
                                indices[row] = output_index;
                                output_validity[row] = true;
                            },
                        }
                    }
                } else if (target_valid) {
                    for (output_validity, 0..) |*found, row| {
                        switch (search) {
                            .first_valid => if (!found.*) {
                                indices[row] = output_index;
                                found.* = true;
                            },
                            .last_valid => {
                                indices[row] = output_index;
                                found.* = true;
                            },
                            .first_null, .last_null => unreachable,
                        }
                    }
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, indices, output_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeFirstValidIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityMatchIndex(DeviceDataFrame, input, names, output_names, .first_valid);
}

pub fn withRowPrefixFirstValidIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstValidIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastValidIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityMatchIndex(DeviceDataFrame, input, names, output_names, .last_valid);
}

pub fn withRowPrefixLastValidIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastValidIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstNullIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityMatchIndex(DeviceDataFrame, input, names, output_names, .first_null);
}

pub fn withRowPrefixFirstNullIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstNullIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastNullIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeValidityMatchIndex(DeviceDataFrame, input, names, output_names, .last_null);
}

pub fn withRowPrefixLastNullIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastNullIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowWeightedMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    if (value_names.len == 0 or value_names.len != weight_names.len) return error.LengthMismatch;

    const numerators = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(numerators);
    const denominators = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(denominators);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(numerators, 0.0);
    @memset(denominators, 0.0);
    @memset(validity, false);

    for (value_names, weight_names) |value_name, weight_name| {
        const value_source = try input.column(value_name);
        const weight_source = try input.column(weight_name);
        if (!value_source.dtype().isReal() or !weight_source.dtype().isReal()) return error.TypeMismatch;

        switch (value_source.*) {
            inline else => |value_typed| {
                const value_values = try value_typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(value_values);
                const maybe_value_validity = try validityValues(value_typed, input.allocator);
                defer if (maybe_value_validity) |mask| input.allocator.free(mask);

                switch (weight_source.*) {
                    inline else => |weight_typed| {
                        const weight_values = try weight_typed.toOwnedSlice(input.allocator);
                        defer input.allocator.free(weight_values);
                        const maybe_weight_validity = try validityValues(weight_typed, input.allocator);
                        defer if (maybe_weight_validity) |mask| input.allocator.free(mask);

                        for (value_values, weight_values, 0..) |raw_value, raw_weight, row| {
                            const value_valid = if (maybe_value_validity) |mask| mask[row] else true;
                            const weight_valid = if (maybe_weight_validity) |mask| mask[row] else true;
                            if (!value_valid or !weight_valid) continue;
                            const weight = realValueAsF64(@TypeOf(raw_weight), raw_weight);
                            numerators[row] += realValueAsF64(@TypeOf(raw_value), raw_value) * weight;
                            denominators[row] += weight;
                        }
                    },
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    for (values, validity, numerators, denominators) |*value, *valid, numerator, denominator| {
        valid.* = denominator != 0.0;
        value.* = if (denominator == 0.0) 0.0 else numerator / denominator;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..flat.rows) |row| {
        var weighted_sum: f64 = 0.0;
        var weight_sum: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (!(weight > 0.0)) continue;
            weighted_sum += flat.values[offset] * weight;
            weight_sum += weight;
        }
        if (!(weight_sum > 0.0)) continue;
        values[row] = weighted_sum;
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowCumulativeWeightedSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        var running: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            running += flat.values[offset] * flat.weights[offset];
            cumulative[offset] = running;
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub const withRowCumWeightedSum = withRowCumulativeWeightedSum;
pub const withRowPrefixWeightedSum = withRowCumulativeWeightedSum;

fn validateRowCumulativeWeightedOutputs(output_names: []const []const u8, width: usize) DeviceFrameArrayError!void {
    if (output_names.len != width) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }
}

fn withRowCumulativeWeightedOutputColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    output_names: []const []const u8,
    rows: usize,
    width: usize,
    cumulative: []const f64,
    cumulative_validity: []const bool,
) DeviceFrameArrayError!DeviceDataFrame {
    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (output_names, 0..) |output_name, col_index| {
        const values = try input.allocator.alloc(f64, rows);
        defer input.allocator.free(values);
        const validity = try input.allocator.alloc(bool, rows);
        defer input.allocator.free(validity);
        @memset(values, 0.0);
        @memset(validity, false);
        for (0..rows) |row| {
            const offset = row * width + col_index;
            if (!cumulative_validity[offset]) continue;
            values[row] = cumulative[offset];
            validity[row] = true;
        }
        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeWeightedMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        var numerator: f64 = 0.0;
        var denominator: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (weight > 0.0) {
                numerator += flat.values[offset] * weight;
                denominator += weight;
            }
            // A valid zero-weight pair should preserve the current prefix
            // state.  If no positive weight has appeared yet, the weighted
            // mean is still undefined, so the current output remains null.
            if (!(denominator > 0.0)) continue;
            cumulative[offset] = numerator / denominator;
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub const withRowCumWeightedMean = withRowCumulativeWeightedMean;
pub const withRowPrefixWeightedMean = withRowCumulativeWeightedMean;
pub const withRowCumulativeWeightedAverage = withRowCumulativeWeightedMean;
pub const withRowCumulativeWeightedAvg = withRowCumulativeWeightedMean;
pub const withRowCumWeightedAverage = withRowCumulativeWeightedMean;
pub const withRowCumWeightedAvg = withRowCumulativeWeightedMean;
pub const withRowPrefixWeightedAverage = withRowCumulativeWeightedMean;
pub const withRowPrefixWeightedAvg = withRowCumulativeWeightedMean;

const RowCumulativeWeightedMomentReduction = enum { mean_square, rms, mean_abs, l1_norm, l2_norm };

fn withRowCumulativeWeightedMoment(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowCumulativeWeightedMomentReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        var weight_sum: f64 = 0.0;
        var weighted_square_sum: f64 = 0.0;
        var weighted_abs_sum: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (weight > 0.0) {
                const value = flat.values[offset];
                weight_sum += weight;
                weighted_square_sum += value * value * weight;
                weighted_abs_sum += @abs(value) * weight;
            }
            if (!(weight_sum > 0.0)) continue;
            cumulative[offset] = switch (reduction) {
                .mean_square => weighted_square_sum / weight_sum,
                .rms => std.math.sqrt(weighted_square_sum / weight_sum),
                .mean_abs => weighted_abs_sum / weight_sum,
                .l1_norm => weighted_abs_sum,
                .l2_norm => std.math.sqrt(weighted_square_sum),
            };
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedMeanSquare(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_names, .mean_square);
}

pub fn withRowCumulativeWeightedRms(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_names, .rms);
}

pub fn withRowCumulativeWeightedMeanAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_names, .mean_abs);
}

pub fn withRowCumulativeWeightedL1Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_names, .l1_norm);
}

pub fn withRowCumulativeWeightedL2Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_names, .l2_norm);
}

pub const withRowCumulativeWeightedMeanSquared = withRowCumulativeWeightedMeanSquare;
pub const withRowCumulativeWeightedMeanSq = withRowCumulativeWeightedMeanSquare;
pub const withRowCumWeightedMeanSquare = withRowCumulativeWeightedMeanSquare;
pub const withRowCumWeightedMeanSquared = withRowCumulativeWeightedMeanSquare;
pub const withRowCumWeightedMeanSq = withRowCumulativeWeightedMeanSquare;
pub const withRowPrefixWeightedMeanSquare = withRowCumulativeWeightedMeanSquare;
pub const withRowPrefixWeightedMeanSquared = withRowCumulativeWeightedMeanSquare;
pub const withRowPrefixWeightedMeanSq = withRowCumulativeWeightedMeanSquare;
pub const withRowCumulativeWeightedRMS = withRowCumulativeWeightedRms;
pub const withRowCumWeightedRms = withRowCumulativeWeightedRms;
pub const withRowCumWeightedRMS = withRowCumulativeWeightedRms;
pub const withRowPrefixWeightedRms = withRowCumulativeWeightedRms;
pub const withRowPrefixWeightedRMS = withRowCumulativeWeightedRms;
pub const withRowCumWeightedMeanAbs = withRowCumulativeWeightedMeanAbs;
pub const withRowPrefixWeightedMeanAbs = withRowCumulativeWeightedMeanAbs;
pub const withRowCumulativeWeightedL1 = withRowCumulativeWeightedL1Norm;
pub const withRowCumWeightedL1Norm = withRowCumulativeWeightedL1Norm;
pub const withRowCumWeightedL1 = withRowCumulativeWeightedL1Norm;
pub const withRowPrefixWeightedL1Norm = withRowCumulativeWeightedL1Norm;
pub const withRowPrefixWeightedL1 = withRowCumulativeWeightedL1Norm;
pub const withRowCumulativeWeightedL2 = withRowCumulativeWeightedL2Norm;
pub const withRowCumWeightedL2Norm = withRowCumulativeWeightedL2Norm;
pub const withRowCumWeightedL2 = withRowCumulativeWeightedL2Norm;
pub const withRowPrefixWeightedL2Norm = withRowCumulativeWeightedL2Norm;
pub const withRowPrefixWeightedL2 = withRowCumulativeWeightedL2Norm;

const RowCumulativeWeightedSupportReduction = enum { weight_sum, positive_count, effective_n };

fn withRowCumulativeWeightedSupport(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowCumulativeWeightedSupportReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        var weight_sum: f64 = 0.0;
        var weight_square_sum: f64 = 0.0;
        var positive_count: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            weight_sum += weight;
            if (weight > 0.0) {
                weight_square_sum += weight * weight;
                positive_count += 1.0;
            }
            cumulative[offset] = switch (reduction) {
                .weight_sum => weight_sum,
                .positive_count => positive_count,
                .effective_n => rowWeightedEffectiveN(weight_sum, weight_square_sum),
            };
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedWeightSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedSupport(DeviceDataFrame, input, value_names, weight_names, output_names, .weight_sum);
}

pub fn withRowCumulativeWeightedPositiveCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedSupport(DeviceDataFrame, input, value_names, weight_names, output_names, .positive_count);
}

pub fn withRowCumulativeWeightedEffectiveN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedSupport(DeviceDataFrame, input, value_names, weight_names, output_names, .effective_n);
}

pub const withRowCumWeightedWeightSum = withRowCumulativeWeightedWeightSum;
pub const withRowPrefixWeightedWeightSum = withRowCumulativeWeightedWeightSum;
pub const withRowCumWeightedPositiveCount = withRowCumulativeWeightedPositiveCount;
pub const withRowPrefixWeightedPositiveCount = withRowCumulativeWeightedPositiveCount;
pub const withRowCumWeightedEffectiveN = withRowCumulativeWeightedEffectiveN;
pub const withRowPrefixWeightedEffectiveN = withRowCumulativeWeightedEffectiveN;
pub const withRowCumulativeWeightedEffectiveCount = withRowCumulativeWeightedEffectiveN;
pub const withRowCumWeightedEffectiveCount = withRowCumulativeWeightedEffectiveN;
pub const withRowPrefixWeightedEffectiveCount = withRowCumulativeWeightedEffectiveN;

const RowWeightedSupportReduction = enum { weight_sum, positive_count, effective_n };

fn rowWeightedEffectiveN(weight_sum: f64, weight_square_sum: f64) f64 {
    if (!(weight_sum > 0.0) or !(weight_square_sum > 0.0)) return quietNanF64();
    return weight_sum * weight_sum / weight_square_sum;
}

fn withRowWeightedSupport(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowWeightedSupportReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..flat.rows) |row| {
        var weight_sum: f64 = 0.0;
        var weight_square_sum: f64 = 0.0;
        var positive_count: usize = 0;
        var pair_count: usize = 0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            weight_sum += weight;
            pair_count += 1;
            if (weight > 0.0) {
                weight_square_sum += weight * weight;
                positive_count += 1;
            }
        }
        if (pair_count == 0) continue;

        values[row] = switch (reduction) {
            .weight_sum => weight_sum,
            .positive_count => @as(f64, @floatFromInt(positive_count)),
            .effective_n => rowWeightedEffectiveN(weight_sum, weight_square_sum),
        };
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedWeightSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedSupport(DeviceDataFrame, input, value_names, weight_names, output_name, .weight_sum);
}

pub fn withRowWeightedPositiveCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedSupport(DeviceDataFrame, input, value_names, weight_names, output_name, .positive_count);
}

pub fn withRowWeightedEffectiveN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedSupport(DeviceDataFrame, input, value_names, weight_names, output_name, .effective_n);
}

pub const withRowWeightedEffectiveCount = withRowWeightedEffectiveN;

const RowWeightedMomentReduction = enum { mean_square, rms, mean_abs, l1_norm, l2_norm };

fn withRowWeightedMoment(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowWeightedMomentReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..flat.rows) |row| {
        var weight_sum: f64 = 0.0;
        var weighted_square_sum: f64 = 0.0;
        var weighted_abs_sum: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (!(weight > 0.0)) continue;
            const value = flat.values[offset];
            weight_sum += weight;
            weighted_square_sum += value * value * weight;
            weighted_abs_sum += @abs(value) * weight;
        }
        if (!(weight_sum > 0.0)) continue;

        values[row] = switch (reduction) {
            .mean_square => weighted_square_sum / weight_sum,
            .rms => std.math.sqrt(weighted_square_sum / weight_sum),
            .mean_abs => weighted_abs_sum / weight_sum,
            .l1_norm => weighted_abs_sum,
            .l2_norm => std.math.sqrt(weighted_square_sum),
        };
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedMeanSquare(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_name, .mean_square);
}

pub fn withRowWeightedRms(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_name, .rms);
}

pub fn withRowWeightedMeanAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_name, .mean_abs);
}

pub fn withRowWeightedL1Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_name, .l1_norm);
}

pub fn withRowWeightedL2Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedMoment(DeviceDataFrame, input, value_names, weight_names, output_name, .l2_norm);
}

pub const withRowWeightedMeanSquared = withRowWeightedMeanSquare;
pub const withRowWeightedMeanSq = withRowWeightedMeanSquare;
pub const withRowWeightedRMS = withRowWeightedRms;
pub const withRowWeightedL1 = withRowWeightedL1Norm;
pub const withRowWeightedL2 = withRowWeightedL2Norm;

const RowWeightedExtremaReduction = enum { min, max, max_abs, min_abs, range, midrange, range_coeff };

fn finishRowWeightedRange(min_value: f64, max_value: f64, positive_weight_count: usize, comptime reduction: RowWeightedExtremaReduction) f64 {
    if (positive_weight_count == 0) return quietNanF64();
    const range = max_value - min_value;
    return switch (reduction) {
        .range => range,
        .midrange => (min_value + max_value) / 2.0,
        .range_coeff => blk: {
            const denominator = min_value + max_value;
            break :blk if (denominator == 0.0) quietNanF64() else range / denominator;
        },
        else => unreachable,
    };
}

fn withRowCumulativeWeightedExtrema(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowWeightedExtremaReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        var min_value: f64 = 0.0;
        var max_value: f64 = 0.0;
        var min_abs_value: f64 = 0.0;
        var max_abs_value: f64 = 0.0;
        var positive_count: usize = 0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (weight > 0.0) {
                const value = flat.values[offset];
                const abs_value = @abs(value);
                if (positive_count == 0 or std.math.isNan(value) or (!std.math.isNan(min_value) and value < min_value)) {
                    min_value = value;
                }
                if (positive_count == 0 or std.math.isNan(value) or (!std.math.isNan(max_value) and value > max_value)) {
                    max_value = value;
                }
                if (positive_count == 0 or std.math.isNan(abs_value) or (!std.math.isNan(min_abs_value) and abs_value < min_abs_value)) {
                    min_abs_value = abs_value;
                }
                if (positive_count == 0 or std.math.isNan(abs_value) or (!std.math.isNan(max_abs_value) and abs_value > max_abs_value)) {
                    max_abs_value = abs_value;
                }
                positive_count += 1;
            }
            if (positive_count == 0) continue;
            cumulative[offset] = switch (reduction) {
                .min => min_value,
                .max => max_value,
                .max_abs => max_abs_value,
                .min_abs => min_abs_value,
                .range, .midrange, .range_coeff => finishRowWeightedRange(min_value, max_value, positive_count, reduction),
            };
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_names, .min);
}

pub fn withRowCumulativeWeightedMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_names, .max);
}

pub fn withRowCumulativeWeightedMaxAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_names, .max_abs);
}

pub fn withRowCumulativeWeightedMinAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_names, .min_abs);
}

pub fn withRowCumulativeWeightedRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_names, .range);
}

pub fn withRowCumulativeWeightedMidrange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_names, .midrange);
}

pub fn withRowCumulativeWeightedRangeCoeff(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_names, .range_coeff);
}

pub const withRowCumulativeWeightedMinimum = withRowCumulativeWeightedMin;
pub const withRowCumulativeWeightedMaximum = withRowCumulativeWeightedMax;
pub const withRowCumulativeWeightedMaximumAbs = withRowCumulativeWeightedMaxAbs;
pub const withRowCumulativeWeightedMaxAbsolute = withRowCumulativeWeightedMaxAbs;
pub const withRowCumulativeWeightedMinimumAbs = withRowCumulativeWeightedMinAbs;
pub const withRowCumulativeWeightedMinAbsolute = withRowCumulativeWeightedMinAbs;
pub const withRowCumulativeWeightedRangeCoefficient = withRowCumulativeWeightedRangeCoeff;
pub const withRowCumWeightedMin = withRowCumulativeWeightedMin;
pub const withRowCumWeightedMinimum = withRowCumulativeWeightedMin;
pub const withRowCumWeightedMax = withRowCumulativeWeightedMax;
pub const withRowCumWeightedMaximum = withRowCumulativeWeightedMax;
pub const withRowCumWeightedMaxAbs = withRowCumulativeWeightedMaxAbs;
pub const withRowCumWeightedMinAbs = withRowCumulativeWeightedMinAbs;
pub const withRowCumWeightedRange = withRowCumulativeWeightedRange;
pub const withRowCumWeightedMidrange = withRowCumulativeWeightedMidrange;
pub const withRowCumWeightedRangeCoeff = withRowCumulativeWeightedRangeCoeff;
pub const withRowCumWeightedRangeCoefficient = withRowCumulativeWeightedRangeCoeff;
pub const withRowPrefixWeightedMin = withRowCumulativeWeightedMin;
pub const withRowPrefixWeightedMinimum = withRowCumulativeWeightedMin;
pub const withRowPrefixWeightedMax = withRowCumulativeWeightedMax;
pub const withRowPrefixWeightedMaximum = withRowCumulativeWeightedMax;
pub const withRowPrefixWeightedMaxAbs = withRowCumulativeWeightedMaxAbs;
pub const withRowPrefixWeightedMinAbs = withRowCumulativeWeightedMinAbs;
pub const withRowPrefixWeightedRange = withRowCumulativeWeightedRange;
pub const withRowPrefixWeightedMidrange = withRowCumulativeWeightedMidrange;
pub const withRowPrefixWeightedRangeCoeff = withRowCumulativeWeightedRangeCoeff;
pub const withRowPrefixWeightedRangeCoefficient = withRowCumulativeWeightedRangeCoeff;

fn withRowWeightedExtrema(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowWeightedExtremaReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..flat.rows) |row| {
        var min_value: f64 = 0.0;
        var max_value: f64 = 0.0;
        var min_abs_value: f64 = 0.0;
        var max_abs_value: f64 = 0.0;
        var positive_count: usize = 0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (!(weight > 0.0)) continue;
            const value = flat.values[offset];
            const abs_value = @abs(value);
            if (positive_count == 0 or std.math.isNan(value) or (!std.math.isNan(min_value) and value < min_value)) {
                min_value = value;
            }
            if (positive_count == 0 or std.math.isNan(value) or (!std.math.isNan(max_value) and value > max_value)) {
                max_value = value;
            }
            if (positive_count == 0 or std.math.isNan(abs_value) or (!std.math.isNan(min_abs_value) and abs_value < min_abs_value)) {
                min_abs_value = abs_value;
            }
            if (positive_count == 0 or std.math.isNan(abs_value) or (!std.math.isNan(max_abs_value) and abs_value > max_abs_value)) {
                max_abs_value = abs_value;
            }
            positive_count += 1;
        }
        if (positive_count == 0) continue;

        values[row] = switch (reduction) {
            .min => min_value,
            .max => max_value,
            .max_abs => max_abs_value,
            .min_abs => min_abs_value,
            .range, .midrange, .range_coeff => finishRowWeightedRange(min_value, max_value, positive_count, reduction),
        };
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_name, .min);
}

pub fn withRowWeightedMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_name, .max);
}

pub fn withRowWeightedMaxAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_name, .max_abs);
}

pub fn withRowWeightedMinAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_name, .min_abs);
}

pub fn withRowWeightedRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_name, .range);
}

pub fn withRowWeightedMidrange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_name, .midrange);
}

pub fn withRowWeightedRangeCoeff(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedExtrema(DeviceDataFrame, input, value_names, weight_names, output_name, .range_coeff);
}

pub const withRowWeightedMinimum = withRowWeightedMin;
pub const withRowWeightedMaximum = withRowWeightedMax;
pub const withRowWeightedMaximumAbs = withRowWeightedMaxAbs;
pub const withRowWeightedMaxAbsolute = withRowWeightedMaxAbs;
pub const withRowWeightedMinimumAbs = withRowWeightedMinAbs;
pub const withRowWeightedMinAbsolute = withRowWeightedMinAbs;
pub const withRowWeightedRangeCoefficient = withRowWeightedRangeCoeff;

const RowWeightedLogProductReduction = enum { product, geometric_mean, harmonic_mean, logsumexp, logmeanexp };

const RowWeightedLogExpState = struct {
    max_value: f64 = 0.0,
    scaled_sum: f64 = 0.0,
    seen_positive_weight: bool = false,

    fn update(self: *RowWeightedLogExpState, value: f64, weight: f64) void {
        if (!(weight > 0.0)) return;
        self.seen_positive_weight = true;
        if (std.math.isNan(value) or std.math.isNan(self.max_value)) {
            self.max_value = quietNanF64();
            self.scaled_sum = quietNanF64();
            return;
        }
        if (self.scaled_sum == 0.0) {
            self.max_value = value;
            self.scaled_sum = weight;
            return;
        }
        if (std.math.isPositiveInf(self.max_value)) {
            if (std.math.isPositiveInf(value)) self.scaled_sum += weight;
            return;
        }
        if (std.math.isPositiveInf(value)) {
            self.max_value = value;
            self.scaled_sum = weight;
            return;
        }
        if (value > self.max_value) {
            self.scaled_sum = self.scaled_sum * std.math.exp(self.max_value - value) + weight;
            self.max_value = value;
        } else if (std.math.isNegativeInf(self.max_value) and std.math.isNegativeInf(value)) {
            self.scaled_sum += weight;
        } else {
            self.scaled_sum += weight * std.math.exp(value - self.max_value);
        }
    }

    fn finish(self: RowWeightedLogExpState, weight_sum: f64, comptime normalize_by_weight: bool) f64 {
        if (!(weight_sum > 0.0) or !self.seen_positive_weight) return quietNanF64();
        if (std.math.isNan(self.max_value) or std.math.isNan(self.scaled_sum)) return quietNanF64();
        if (std.math.isPositiveInf(self.max_value) or std.math.isNegativeInf(self.max_value)) return self.max_value;
        if (!(self.scaled_sum > 0.0)) return -std.math.inf(f64);
        var result = self.max_value + std.math.log(f64, std.math.e, self.scaled_sum);
        if (normalize_by_weight) result -= std.math.log(f64, std.math.e, weight_sum);
        return result;
    }
};

const RowWeightedProductState = struct {
    signed_log_abs_sum: f64 = 0.0,
    negative_factor_count: usize = 0,
    zero_seen: bool = false,

    fn update(self: *RowWeightedProductState, value: f64, weight: f64) void {
        if (!(weight > 0.0)) return;
        if (std.math.isNan(value) or std.math.isNan(weight)) {
            self.signed_log_abs_sum = quietNanF64();
            return;
        }
        if (value == 0.0) {
            self.zero_seen = true;
            return;
        }
        if (value < 0.0) self.negative_factor_count += 1;
        self.signed_log_abs_sum += weight * std.math.log(f64, std.math.e, @abs(value));
    }

    fn finish(self: RowWeightedProductState, weight_sum: f64) f64 {
        if (!(weight_sum > 0.0)) return quietNanF64();
        if (std.math.isNan(self.signed_log_abs_sum)) return quietNanF64();
        if (self.zero_seen) return 0.0;
        const magnitude = std.math.exp(self.signed_log_abs_sum);
        return if (self.negative_factor_count % 2 == 0) magnitude else -magnitude;
    }
};

fn withRowWeightedLogProduct(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowWeightedLogProductReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..flat.rows) |row| {
        var weight_sum: f64 = 0.0;
        var weighted_log_sum: f64 = 0.0;
        var weighted_reciprocal_sum: f64 = 0.0;
        var weighted_zero_seen = false;
        var log_exp_state: RowWeightedLogExpState = .{};
        var product_state: RowWeightedProductState = .{};
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (!(weight > 0.0)) continue;
            const value = flat.values[offset];
            weight_sum += weight;
            if (value < 0.0) {
                weighted_log_sum = quietNanF64();
            } else if (value == 0.0 and !std.math.isNan(weighted_log_sum)) {
                weighted_zero_seen = true;
            } else if (!weighted_zero_seen and !std.math.isNan(weighted_log_sum)) {
                weighted_log_sum += weight * std.math.log(f64, std.math.e, value);
            }
            if (value == 0.0 and !std.math.isNan(weighted_reciprocal_sum)) {
                weighted_reciprocal_sum = std.math.inf(f64);
            } else if (!std.math.isInf(weighted_reciprocal_sum)) {
                weighted_reciprocal_sum += weight / value;
            }
            log_exp_state.update(value, weight);
            product_state.update(value, weight);
        }
        if (!(weight_sum > 0.0)) continue;

        values[row] = switch (reduction) {
            .product => product_state.finish(weight_sum),
            .geometric_mean => if (std.math.isNan(weighted_log_sum)) quietNanF64() else if (weighted_zero_seen) 0.0 else std.math.exp(weighted_log_sum / weight_sum),
            .harmonic_mean => if (std.math.isInf(weighted_reciprocal_sum)) 0.0 else weight_sum / weighted_reciprocal_sum,
            .logsumexp => log_exp_state.finish(weight_sum, false),
            .logmeanexp => log_exp_state.finish(weight_sum, true),
        };
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedProduct(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_name, .product);
}

pub fn withRowWeightedGeometricMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_name, .geometric_mean);
}

pub fn withRowWeightedHarmonicMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_name, .harmonic_mean);
}

pub fn withRowWeightedLogSumExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_name, .logsumexp);
}

pub fn withRowWeightedLogMeanExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_name, .logmeanexp);
}

pub const withRowWeightedProd = withRowWeightedProduct;
pub const withRowWeightedGeoMean = withRowWeightedGeometricMean;
pub const withRowWeightedHarmMean = withRowWeightedHarmonicMean;
pub const withRowWeightedLogsumexp = withRowWeightedLogSumExp;
pub const withRowWeightedLogmeanexp = withRowWeightedLogMeanExp;

fn withRowCumulativeWeightedLogProduct(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowWeightedLogProductReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        var weight_sum: f64 = 0.0;
        var weighted_log_sum: f64 = 0.0;
        var weighted_reciprocal_sum: f64 = 0.0;
        var weighted_zero_seen = false;
        var log_exp_state: RowWeightedLogExpState = .{};
        var product_state: RowWeightedProductState = .{};
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (weight > 0.0) {
                const value = flat.values[offset];
                weight_sum += weight;
                if (value < 0.0) {
                    weighted_log_sum = quietNanF64();
                } else if (value == 0.0 and !std.math.isNan(weighted_log_sum)) {
                    weighted_zero_seen = true;
                } else if (!weighted_zero_seen and !std.math.isNan(weighted_log_sum)) {
                    weighted_log_sum += weight * std.math.log(f64, std.math.e, value);
                }
                if (value == 0.0 and !std.math.isNan(weighted_reciprocal_sum)) {
                    weighted_reciprocal_sum = std.math.inf(f64);
                } else if (!std.math.isInf(weighted_reciprocal_sum)) {
                    weighted_reciprocal_sum += weight / value;
                }
                log_exp_state.update(value, weight);
                product_state.update(value, weight);
            }
            if (!(weight_sum > 0.0)) continue;
            cumulative[offset] = switch (reduction) {
                .product => product_state.finish(weight_sum),
                .geometric_mean => if (std.math.isNan(weighted_log_sum)) quietNanF64() else if (weighted_zero_seen) 0.0 else std.math.exp(weighted_log_sum / weight_sum),
                .harmonic_mean => if (std.math.isInf(weighted_reciprocal_sum)) 0.0 else weight_sum / weighted_reciprocal_sum,
                .logsumexp => log_exp_state.finish(weight_sum, false),
                .logmeanexp => log_exp_state.finish(weight_sum, true),
            };
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedProduct(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_names, .product);
}

pub fn withRowCumulativeWeightedGeometricMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_names, .geometric_mean);
}

pub fn withRowCumulativeWeightedHarmonicMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_names, .harmonic_mean);
}

pub fn withRowCumulativeWeightedLogSumExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_names, .logsumexp);
}

pub fn withRowCumulativeWeightedLogMeanExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedLogProduct(DeviceDataFrame, input, value_names, weight_names, output_names, .logmeanexp);
}

pub const withRowCumulativeWeightedProd = withRowCumulativeWeightedProduct;
pub const withRowCumulativeWeightedGeoMean = withRowCumulativeWeightedGeometricMean;
pub const withRowCumulativeWeightedHarmMean = withRowCumulativeWeightedHarmonicMean;
pub const withRowCumulativeWeightedLogsumexp = withRowCumulativeWeightedLogSumExp;
pub const withRowCumulativeWeightedLogmeanexp = withRowCumulativeWeightedLogMeanExp;
pub const withRowCumWeightedProduct = withRowCumulativeWeightedProduct;
pub const withRowCumWeightedProd = withRowCumulativeWeightedProduct;
pub const withRowCumWeightedGeometricMean = withRowCumulativeWeightedGeometricMean;
pub const withRowCumWeightedGeoMean = withRowCumulativeWeightedGeometricMean;
pub const withRowCumWeightedHarmonicMean = withRowCumulativeWeightedHarmonicMean;
pub const withRowCumWeightedHarmMean = withRowCumulativeWeightedHarmonicMean;
pub const withRowCumWeightedLogSumExp = withRowCumulativeWeightedLogSumExp;
pub const withRowCumWeightedLogsumexp = withRowCumulativeWeightedLogSumExp;
pub const withRowCumWeightedLogMeanExp = withRowCumulativeWeightedLogMeanExp;
pub const withRowCumWeightedLogmeanexp = withRowCumulativeWeightedLogMeanExp;
pub const withRowPrefixWeightedProduct = withRowCumulativeWeightedProduct;
pub const withRowPrefixWeightedProd = withRowCumulativeWeightedProduct;
pub const withRowPrefixWeightedGeometricMean = withRowCumulativeWeightedGeometricMean;
pub const withRowPrefixWeightedGeoMean = withRowCumulativeWeightedGeometricMean;
pub const withRowPrefixWeightedHarmonicMean = withRowCumulativeWeightedHarmonicMean;
pub const withRowPrefixWeightedHarmMean = withRowCumulativeWeightedHarmonicMean;
pub const withRowPrefixWeightedLogSumExp = withRowCumulativeWeightedLogSumExp;
pub const withRowPrefixWeightedLogsumexp = withRowCumulativeWeightedLogSumExp;
pub const withRowPrefixWeightedLogMeanExp = withRowCumulativeWeightedLogMeanExp;
pub const withRowPrefixWeightedLogmeanexp = withRowCumulativeWeightedLogMeanExp;

const RowWeightedDispersion = enum { variance, stddev, sem, cv, fano };

fn finishRowWeightedDispersion(weighted_sum: f64, weighted_square_sum: f64, weight_sum: f64, correction: f64, comptime reduction: RowWeightedDispersion) f64 {
    const denominator = weight_sum - correction;
    var centered_square_sum = weighted_square_sum - weighted_sum * weighted_sum / weight_sum;
    if (centered_square_sum < 0.0 and centered_square_sum > -1e-12) centered_square_sum = 0.0;
    const variance = if (denominator <= 0.0) quietNanF64() else centered_square_sum / denominator;
    const stddev = std.math.sqrt(variance);
    const mean = weighted_sum / weight_sum;
    return switch (reduction) {
        .variance => variance,
        .stddev => stddev,
        .sem => if (denominator <= 0.0) quietNanF64() else std.math.sqrt(variance / weight_sum),
        .cv => if (mean == 0.0) quietNanF64() else stddev / mean,
        .fano => if (mean == 0.0) quietNanF64() else variance / mean,
    };
}

fn withRowCumulativeWeightedDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
    comptime reduction: RowWeightedDispersion,
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(correction) or correction < 0.0) return error.InvalidShape;
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        var weighted_sum: f64 = 0.0;
        var weighted_square_sum: f64 = 0.0;
        var weight_sum: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (weight > 0.0) {
                const value = flat.values[offset];
                weighted_sum += value * weight;
                weighted_square_sum += value * value * weight;
                weight_sum += weight;
            }
            if (!(weight_sum > 0.0)) continue;
            cumulative[offset] = finishRowWeightedDispersion(weighted_sum, weighted_square_sum, weight_sum, correction, reduction);
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedVariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_names, correction, .variance);
}

pub fn withRowCumulativeWeightedVar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedVariance(DeviceDataFrame, input, value_names, weight_names, output_names, correction);
}

pub fn withRowCumulativeWeightedStddev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_names, correction, .stddev);
}

pub fn withRowCumulativeWeightedStd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedStddev(DeviceDataFrame, input, value_names, weight_names, output_names, correction);
}

pub fn withRowCumulativeWeightedSem(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_names, correction, .sem);
}

pub fn withRowCumulativeWeightedCv(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_names, correction, .cv);
}

pub fn withRowCumulativeWeightedFano(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_names, correction, .fano);
}

pub const withRowCumulativeWeightedSEM = withRowCumulativeWeightedSem;
pub const withRowCumulativeWeightedCV = withRowCumulativeWeightedCv;
pub const withRowCumWeightedVariance = withRowCumulativeWeightedVariance;
pub const withRowCumWeightedVar = withRowCumulativeWeightedVariance;
pub const withRowCumWeightedStddev = withRowCumulativeWeightedStddev;
pub const withRowCumWeightedStd = withRowCumulativeWeightedStddev;
pub const withRowCumWeightedSem = withRowCumulativeWeightedSem;
pub const withRowCumWeightedSEM = withRowCumulativeWeightedSem;
pub const withRowCumWeightedCv = withRowCumulativeWeightedCv;
pub const withRowCumWeightedCV = withRowCumulativeWeightedCv;
pub const withRowCumWeightedFano = withRowCumulativeWeightedFano;
pub const withRowPrefixWeightedVariance = withRowCumulativeWeightedVariance;
pub const withRowPrefixWeightedVar = withRowCumulativeWeightedVariance;
pub const withRowPrefixWeightedStddev = withRowCumulativeWeightedStddev;
pub const withRowPrefixWeightedStd = withRowCumulativeWeightedStddev;
pub const withRowPrefixWeightedSem = withRowCumulativeWeightedSem;
pub const withRowPrefixWeightedSEM = withRowCumulativeWeightedSem;
pub const withRowPrefixWeightedCv = withRowCumulativeWeightedCv;
pub const withRowPrefixWeightedCV = withRowCumulativeWeightedCv;
pub const withRowPrefixWeightedFano = withRowCumulativeWeightedFano;

fn withRowWeightedDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
    comptime reduction: RowWeightedDispersion,
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(correction) or correction < 0.0) return error.InvalidShape;
    if (value_names.len == 0 or value_names.len != weight_names.len) return error.LengthMismatch;

    const weighted_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weighted_sums);
    const weighted_square_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weighted_square_sums);
    const weight_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weight_sums);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(weighted_sums, 0.0);
    @memset(weighted_square_sums, 0.0);
    @memset(weight_sums, 0.0);
    @memset(validity, false);

    for (value_names, weight_names) |value_name, weight_name| {
        const value_source = try input.column(value_name);
        const weight_source = try input.column(weight_name);
        if (!value_source.dtype().isReal() or !weight_source.dtype().isReal()) return error.TypeMismatch;

        switch (value_source.*) {
            inline else => |value_typed| {
                const value_values = try value_typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(value_values);
                const maybe_value_validity = try validityValues(value_typed, input.allocator);
                defer if (maybe_value_validity) |mask| input.allocator.free(mask);

                switch (weight_source.*) {
                    inline else => |weight_typed| {
                        const weight_values = try weight_typed.toOwnedSlice(input.allocator);
                        defer input.allocator.free(weight_values);
                        const maybe_weight_validity = try validityValues(weight_typed, input.allocator);
                        defer if (maybe_weight_validity) |mask| input.allocator.free(mask);

                        for (value_values, weight_values, 0..) |raw_value, raw_weight, row| {
                            const value_valid = if (maybe_value_validity) |mask| mask[row] else true;
                            const weight_valid = if (maybe_weight_validity) |mask| mask[row] else true;
                            if (!value_valid or !weight_valid) continue;
                            const weight = realValueAsF64(@TypeOf(raw_weight), raw_weight);
                            if (weight < 0.0) return error.InvalidShape;
                            const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                            weighted_sums[row] += value * weight;
                            weighted_square_sums[row] += value * value * weight;
                            weight_sums[row] += weight;
                        }
                    },
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    for (values, validity, weighted_sums, weighted_square_sums, weight_sums) |*value, *valid, weighted_sum, weighted_square_sum, weight_sum| {
        valid.* = weight_sum > 0.0;
        if (!valid.*) {
            value.* = 0.0;
            continue;
        }
        value.* = finishRowWeightedDispersion(weighted_sum, weighted_square_sum, weight_sum, correction, reduction);
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedVariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_name, correction, .variance);
}

pub fn withRowWeightedVar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedVariance(DeviceDataFrame, input, value_names, weight_names, output_name, correction);
}

pub fn withRowWeightedStddev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_name, correction, .stddev);
}

pub fn withRowWeightedStd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedStddev(DeviceDataFrame, input, value_names, weight_names, output_name, correction);
}

pub fn withRowWeightedSem(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_name, correction, .sem);
}

pub fn withRowWeightedCv(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_name, correction, .cv);
}

pub fn withRowWeightedFano(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDispersion(DeviceDataFrame, input, value_names, weight_names, output_name, correction, .fano);
}

pub const withRowWeightedSEM = withRowWeightedSem;
pub const withRowWeightedCV = withRowWeightedCv;

const RowWeightedShapeReduction = enum { skewness, kurtosis };

fn finishRowWeightedShape(weight_sum: f64, centered2_raw: f64, centered3: f64, centered4: f64, comptime reduction: RowWeightedShapeReduction) f64 {
    var centered2 = centered2_raw;
    if (centered2 < 0.0 and centered2 > -1e-12) centered2 = 0.0;
    if (centered2 == 0.0) return quietNanF64();
    return switch (reduction) {
        .skewness => std.math.sqrt(weight_sum) * centered3 / std.math.pow(f64, centered2, 1.5),
        .kurtosis => weight_sum * centered4 / (centered2 * centered2) - 3.0,
    };
}

fn finishRowWeightedShapeFromRaw(weight_sum: f64, sum1: f64, sum2: f64, sum3: f64, sum4: f64, comptime reduction: RowWeightedShapeReduction) f64 {
    const mean = sum1 / weight_sum;
    const mean2 = mean * mean;
    const centered2 = sum2 - sum1 * sum1 / weight_sum;
    const centered3 = sum3 - 3.0 * mean * sum2 + 3.0 * mean2 * sum1 - mean2 * mean * weight_sum;
    const centered4 = sum4 - 4.0 * mean * sum3 + 6.0 * mean2 * sum2 - 4.0 * mean2 * mean * sum1 + mean2 * mean2 * weight_sum;
    return finishRowWeightedShape(weight_sum, centered2, centered3, centered4, reduction);
}

fn withRowCumulativeWeightedShape(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowWeightedShapeReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        var weight_sum: f64 = 0.0;
        var sum1: f64 = 0.0;
        var sum2: f64 = 0.0;
        var sum3: f64 = 0.0;
        var sum4: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (weight > 0.0) {
                const value = flat.values[offset];
                const value2 = value * value;
                weight_sum += weight;
                sum1 += value * weight;
                sum2 += value2 * weight;
                sum3 += value2 * value * weight;
                sum4 += value2 * value2 * weight;
            }
            if (!(weight_sum > 0.0)) continue;
            cumulative[offset] = finishRowWeightedShapeFromRaw(weight_sum, sum1, sum2, sum3, sum4, reduction);
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedSkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedShape(DeviceDataFrame, input, value_names, weight_names, output_names, .skewness);
}

pub fn withRowCumulativeWeightedKurtosis(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedShape(DeviceDataFrame, input, value_names, weight_names, output_names, .kurtosis);
}

pub const withRowCumulativeWeightedSkew = withRowCumulativeWeightedSkewness;
pub const withRowCumulativeWeightedKurt = withRowCumulativeWeightedKurtosis;
pub const withRowCumWeightedSkewness = withRowCumulativeWeightedSkewness;
pub const withRowCumWeightedSkew = withRowCumulativeWeightedSkewness;
pub const withRowCumWeightedKurtosis = withRowCumulativeWeightedKurtosis;
pub const withRowCumWeightedKurt = withRowCumulativeWeightedKurtosis;
pub const withRowPrefixWeightedSkewness = withRowCumulativeWeightedSkewness;
pub const withRowPrefixWeightedSkew = withRowCumulativeWeightedSkewness;
pub const withRowPrefixWeightedKurtosis = withRowCumulativeWeightedKurtosis;
pub const withRowPrefixWeightedKurt = withRowCumulativeWeightedKurtosis;

fn withRowWeightedShape(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowWeightedShapeReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..flat.rows) |row| {
        var weight_sum: f64 = 0.0;
        var weighted_sum: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (!(weight > 0.0)) continue;
            weight_sum += weight;
            weighted_sum += flat.values[offset] * weight;
        }
        if (!(weight_sum > 0.0)) continue;
        const mean = weighted_sum / weight_sum;

        var centered2: f64 = 0.0;
        var centered3: f64 = 0.0;
        var centered4: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (!(weight > 0.0)) continue;
            const centered = flat.values[offset] - mean;
            const centered_sq = centered * centered;
            centered2 += weight * centered_sq;
            centered3 += weight * centered_sq * centered;
            centered4 += weight * centered_sq * centered_sq;
        }

        values[row] = finishRowWeightedShape(weight_sum, centered2, centered3, centered4, reduction);
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedSkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedShape(DeviceDataFrame, input, value_names, weight_names, output_name, .skewness);
}

pub fn withRowWeightedKurtosis(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedShape(DeviceDataFrame, input, value_names, weight_names, output_name, .kurtosis);
}

pub const withRowWeightedSkew = withRowWeightedSkewness;
pub const withRowWeightedKurt = withRowWeightedKurtosis;

const OwnedRealF64Column = struct {
    allocator: std.mem.Allocator,
    values: []f64,
    validity: ?[]bool,

    fn deinit(self: *OwnedRealF64Column) void {
        self.allocator.free(self.values);
        if (self.validity) |mask| self.allocator.free(mask);
        self.* = undefined;
    }
};

fn ownedRealF64Column(allocator: std.mem.Allocator, source: anytype) DeviceFrameArrayError!OwnedRealF64Column {
    if (!source.dtype().isReal()) return error.TypeMismatch;

    switch (source.*) {
        inline else => |typed| {
            const raw_values = try typed.toOwnedSlice(allocator);
            defer allocator.free(raw_values);
            const values = try allocator.alloc(f64, raw_values.len);
            errdefer allocator.free(values);
            for (raw_values, values) |raw, *out| {
                out.* = realValueAsF64(@TypeOf(raw), raw);
            }
            const maybe_validity = try validityValues(typed, allocator);
            errdefer if (maybe_validity) |mask| allocator.free(mask);
            return .{
                .allocator = allocator,
                .values = values,
                .validity = maybe_validity,
            };
        },
    }
}

const RowWeightedPairReduction = enum { dot, cosine, squared_euclidean, euclidean, manhattan, chebyshev, canberra, bray_curtis, mean_error, mae, mse, rmse, mape, smape, covariance, correlation, beta };

fn withRowWeightedPairReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
    comptime reduction: RowWeightedPairReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(correction) or correction < 0.0) return error.InvalidShape;
    if (lhs_names.len == 0 or lhs_names.len != rhs_names.len or lhs_names.len != weight_names.len) return error.LengthMismatch;

    const weight_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weight_sums);
    const lhs_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(lhs_sums);
    const rhs_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(rhs_sums);
    const lhs_square_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(lhs_square_sums);
    const rhs_square_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(rhs_square_sums);
    const cross_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(cross_sums);
    const weighted_abs_error_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weighted_abs_error_sums);
    const chebyshev_values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(chebyshev_values);
    const weighted_canberra_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weighted_canberra_sums);
    const weighted_bray_denominators = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weighted_bray_denominators);
    const weighted_mape_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weighted_mape_sums);
    const weighted_smape_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(weighted_smape_sums);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(weight_sums, 0.0);
    @memset(lhs_sums, 0.0);
    @memset(rhs_sums, 0.0);
    @memset(lhs_square_sums, 0.0);
    @memset(rhs_square_sums, 0.0);
    @memset(cross_sums, 0.0);
    @memset(weighted_abs_error_sums, 0.0);
    @memset(chebyshev_values, 0.0);
    @memset(weighted_canberra_sums, 0.0);
    @memset(weighted_bray_denominators, 0.0);
    @memset(weighted_mape_sums, 0.0);
    @memset(weighted_smape_sums, 0.0);
    @memset(validity, false);

    for (lhs_names, rhs_names, weight_names) |lhs_name, rhs_name, weight_name| {
        const lhs_source = try input.column(lhs_name);
        const rhs_source = try input.column(rhs_name);
        const weight_source = try input.column(weight_name);

        var lhs_column = try ownedRealF64Column(input.allocator, lhs_source);
        defer lhs_column.deinit();
        var rhs_column = try ownedRealF64Column(input.allocator, rhs_source);
        defer rhs_column.deinit();
        var weight_column = try ownedRealF64Column(input.allocator, weight_source);
        defer weight_column.deinit();

        for (lhs_column.values, rhs_column.values, weight_column.values, 0..) |lhs, rhs, weight, row| {
            const lhs_valid = if (lhs_column.validity) |mask| mask[row] else true;
            const rhs_valid = if (rhs_column.validity) |mask| mask[row] else true;
            const weight_valid = if (weight_column.validity) |mask| mask[row] else true;
            if (!lhs_valid or !rhs_valid or !weight_valid) continue;
            if (weight < 0.0) return error.InvalidShape;
            if (!(weight > 0.0)) continue;
            weight_sums[row] += weight;
            lhs_sums[row] += weight * lhs;
            rhs_sums[row] += weight * rhs;
            lhs_square_sums[row] += weight * lhs * lhs;
            rhs_square_sums[row] += weight * rhs * rhs;
            cross_sums[row] += weight * lhs * rhs;
            const abs_error = @abs(lhs - rhs);
            const abs_lhs = @abs(lhs);
            const abs_rhs = @abs(rhs);
            const abs_sum = abs_lhs + abs_rhs;
            weighted_abs_error_sums[row] += weight * abs_error;
            chebyshev_values[row] = @max(chebyshev_values[row], abs_error);
            weighted_canberra_sums[row] += if (abs_sum == 0.0) 0.0 else weight * abs_error / abs_sum;
            weighted_bray_denominators[row] += weight * abs_sum;
            weighted_mape_sums[row] += if (lhs == 0.0) quietNanF64() else weight * abs_error / abs_lhs;
            weighted_smape_sums[row] += if (abs_sum == 0.0) quietNanF64() else weight * 2.0 * abs_error / abs_sum;
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    for (values, validity, weight_sums, lhs_sums, rhs_sums, lhs_square_sums, rhs_square_sums, cross_sums, weighted_abs_error_sums, chebyshev_values, weighted_canberra_sums, weighted_bray_denominators, weighted_mape_sums, weighted_smape_sums) |*value, *valid, weight_sum, lhs_sum, rhs_sum, lhs_square_sum, rhs_square_sum, cross_sum, weighted_abs_error_sum, chebyshev_value, weighted_canberra_sum, weighted_bray_denominator, weighted_mape_sum, weighted_smape_sum| {
        valid.* = weight_sum > 0.0;
        if (!valid.*) {
            value.* = 0.0;
            continue;
        }
        const denominator = weight_sum - correction;
        var lhs_centered = lhs_square_sum - lhs_sum * lhs_sum / weight_sum;
        var rhs_centered = rhs_square_sum - rhs_sum * rhs_sum / weight_sum;
        const cross_centered = cross_sum - lhs_sum * rhs_sum / weight_sum;
        if (lhs_centered < 0.0 and lhs_centered > -1e-12) lhs_centered = 0.0;
        if (rhs_centered < 0.0 and rhs_centered > -1e-12) rhs_centered = 0.0;

        const covariance = if (denominator <= 0.0) quietNanF64() else cross_centered / denominator;
        const squared_distance = lhs_square_sum + rhs_square_sum - 2.0 * cross_sum;
        value.* = switch (reduction) {
            .dot => cross_sum,
            .cosine => if (lhs_square_sum == 0.0 or rhs_square_sum == 0.0) quietNanF64() else cross_sum / (std.math.sqrt(lhs_square_sum) * std.math.sqrt(rhs_square_sum)),
            .squared_euclidean => squared_distance,
            .euclidean => std.math.sqrt(squared_distance),
            .manhattan => weighted_abs_error_sum,
            .chebyshev => chebyshev_value,
            .canberra => weighted_canberra_sum,
            .bray_curtis => if (weighted_bray_denominator == 0.0) quietNanF64() else weighted_abs_error_sum / weighted_bray_denominator,
            .mean_error => (lhs_sum - rhs_sum) / weight_sum,
            .mae => weighted_abs_error_sum / weight_sum,
            .mse => squared_distance / weight_sum,
            .rmse => std.math.sqrt(squared_distance / weight_sum),
            .mape => weighted_mape_sum / weight_sum,
            .smape => weighted_smape_sum / weight_sum,
            .covariance => covariance,
            .correlation => if (denominator <= 0.0 or lhs_centered == 0.0 or rhs_centered == 0.0) quietNanF64() else cross_centered / std.math.sqrt(lhs_centered * rhs_centered),
            .beta => if (denominator <= 0.0 or lhs_centered == 0.0) quietNanF64() else cross_centered / lhs_centered,
        };
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedDot(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .dot);
}

pub fn withRowWeightedCosineSimilarity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .cosine);
}

pub fn withRowWeightedSquaredEuclideanDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .squared_euclidean);
}

pub fn withRowWeightedEuclideanDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .euclidean);
}

pub fn withRowWeightedManhattanDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .manhattan);
}

pub const withRowWeightedCosine = withRowWeightedCosineSimilarity;
pub const withRowWeightedSquaredDistance = withRowWeightedSquaredEuclideanDistance;
pub const withRowWeightedSqEuclideanDistance = withRowWeightedSquaredEuclideanDistance;
pub const withRowWeightedL2Distance = withRowWeightedEuclideanDistance;
pub const withRowWeightedL1Distance = withRowWeightedManhattanDistance;

pub fn withRowWeightedChebyshevDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .chebyshev);
}

pub fn withRowWeightedCanberraDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .canberra);
}

pub fn withRowWeightedBrayCurtisDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .bray_curtis);
}

pub fn withRowWeightedMeanError(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .mean_error);
}

pub fn withRowWeightedMae(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .mae);
}

pub fn withRowWeightedMse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .mse);
}

pub fn withRowWeightedRmse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .rmse);
}

pub fn withRowWeightedMape(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .mape);
}

pub fn withRowWeightedSmape(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, 0.0, .smape);
}

pub const withRowWeightedBias = withRowWeightedMeanError;
pub const withRowWeightedMAE = withRowWeightedMae;
pub const withRowWeightedMSE = withRowWeightedMse;
pub const withRowWeightedRMSE = withRowWeightedRmse;
pub const withRowWeightedMAPE = withRowWeightedMape;
pub const withRowWeightedSMAPE = withRowWeightedSmape;

pub fn withRowWeightedCovariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, correction, .covariance);
}

pub fn withRowWeightedCorrelation(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, correction, .correlation);
}

pub const withRowWeightedCov = withRowWeightedCovariance;
pub const withRowWeightedCorr = withRowWeightedCorrelation;

pub fn withRowWeightedBeta(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPairReduction(DeviceDataFrame, input, lhs_names, rhs_names, weight_names, output_name, correction, .beta);
}

const RowWeightedValue = struct {
    value: f64,
    weight: f64,
};

fn rowWeightedValueLess(_: void, lhs: RowWeightedValue, rhs: RowWeightedValue) bool {
    return rowQuantileLess({}, lhs.value, rhs.value);
}

fn rowWeightedQuantileFromSorted(sorted: []const RowWeightedValue, q: f64, total_weight: f64) f64 {
    const threshold = q * total_weight;
    var cumulative: f64 = 0.0;
    for (sorted) |item| {
        cumulative += item.weight;
        if (cumulative >= threshold) return item.value;
    }
    return sorted[sorted.len - 1].value;
}

const RowWeightedFlat = struct {
    allocator: std.mem.Allocator,
    values: []f64,
    weights: []f64,
    validity: []bool,
    rows: usize,
    width: usize,

    fn deinit(self: *RowWeightedFlat) void {
        self.allocator.free(self.values);
        self.allocator.free(self.weights);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

fn rowWeightedFlat(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
) DeviceFrameArrayError!RowWeightedFlat {
    if (value_names.len == 0 or value_names.len != weight_names.len) return error.LengthMismatch;

    const total_slots = std.math.mul(usize, input.rows, value_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    errdefer input.allocator.free(flat_values);
    const flat_weights = try input.allocator.alloc(f64, total_slots);
    errdefer input.allocator.free(flat_weights);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    errdefer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_weights, 0.0);
    @memset(flat_validity, false);

    for (value_names, weight_names, 0..) |value_name, weight_name, col_index| {
        const value_source = try input.column(value_name);
        const weight_source = try input.column(weight_name);
        var value_column = try ownedRealF64Column(input.allocator, value_source);
        defer value_column.deinit();
        var weight_column = try ownedRealF64Column(input.allocator, weight_source);
        defer weight_column.deinit();

        for (value_column.values, weight_column.values, 0..) |value, weight, row| {
            const value_valid = if (value_column.validity) |mask| mask[row] else true;
            const weight_valid = if (weight_column.validity) |mask| mask[row] else true;
            if (!value_valid or !weight_valid) continue;
            if (weight < 0.0) return error.InvalidShape;
            const offset = row * value_names.len + col_index;
            flat_values[offset] = value;
            flat_weights[offset] = weight;
            flat_validity[offset] = true;
        }
    }

    return .{
        .allocator = input.allocator,
        .values = flat_values,
        .weights = flat_weights,
        .validity = flat_validity,
        .rows = input.rows,
        .width = value_names.len,
    };
}

const RowWeightedQuantileOutput = struct {
    values: []f64,
    validity: []bool,
};

fn withRowWeightedQuantileValues(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    q: f64,
    subtract_q: ?f64,
) DeviceFrameArrayError!RowWeightedQuantileOutput {
    if (std.math.isNan(q) or q < 0.0 or q > 1.0) return error.InvalidShape;
    if (subtract_q) |lo_q| {
        if (std.math.isNan(lo_q) or lo_q < 0.0 or lo_q > 1.0) return error.InvalidShape;
    }

    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    errdefer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    errdefer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            scratch[count] = .{ .value = flat.values[offset], .weight = flat.weights[offset] };
            total_weight += flat.weights[offset];
            count += 1;
        }
        if (count == 0 or !(total_weight > 0.0)) continue;

        std.sort.insertion(RowWeightedValue, scratch[0..count], {}, rowWeightedValueLess);
        const hi = rowWeightedQuantileFromSorted(scratch[0..count], q, total_weight);
        values[row] = if (subtract_q) |lo_q| hi - rowWeightedQuantileFromSorted(scratch[0..count], lo_q, total_weight) else hi;
        validity[row] = true;
    }

    return .{ .values = values, .validity = validity };
}

pub fn withRowWeightedQuantile(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    q: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    const output = try withRowWeightedQuantileValues(DeviceDataFrame, input, value_names, weight_names, q, null);
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedMedian(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedQuantile(DeviceDataFrame, input, value_names, weight_names, output_name, 0.5);
}

pub fn withRowWeightedIqr(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const output = try withRowWeightedQuantileValues(DeviceDataFrame, input, value_names, weight_names, 0.75, 0.25);
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedMad(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            scratch[count] = .{ .value = flat.values[offset], .weight = flat.weights[offset] };
            total_weight += flat.weights[offset];
            count += 1;
        }
        if (count == 0 or !(total_weight > 0.0)) continue;

        std.sort.insertion(RowWeightedValue, scratch[0..count], {}, rowWeightedValueLess);
        const center = rowWeightedQuantileFromSorted(scratch[0..count], 0.5, total_weight);
        for (scratch[0..count]) |*item| item.value = @abs(item.value - center);
        std.sort.insertion(RowWeightedValue, scratch[0..count], {}, rowWeightedValueLess);
        values[row] = rowWeightedQuantileFromSorted(scratch[0..count], 0.5, total_weight);
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

fn rowWeightedTrimmedMeanFromSorted(sorted: []const RowWeightedValue, total_weight: f64, trim_fraction: f64) f64 {
    if (!(total_weight > 0.0)) return quietNanF64();
    const lower_cut = trim_fraction * total_weight;
    const upper_cut = (1.0 - trim_fraction) * total_weight;
    if (!(upper_cut > lower_cut)) return quietNanF64();

    var cumulative: f64 = 0.0;
    var kept_sum: f64 = 0.0;
    var kept_weight: f64 = 0.0;
    for (sorted) |item| {
        if (!(item.weight > 0.0)) continue;
        const start = cumulative;
        const end = cumulative + item.weight;
        const kept = @max(@as(f64, 0.0), @min(end, upper_cut) - @max(start, lower_cut));
        if (kept > 0.0) {
            kept_sum += kept * item.value;
            kept_weight += kept;
        }
        cumulative = end;
    }
    return if (kept_weight > 0.0) kept_sum / kept_weight else quietNanF64();
}

fn rowWeightedWinsorizedMeanFromSorted(sorted: []const RowWeightedValue, total_weight: f64, winsor_fraction: f64) f64 {
    if (!(total_weight > 0.0)) return quietNanF64();
    const lower = rowWeightedQuantileFromSorted(sorted, winsor_fraction, total_weight);
    const upper = rowWeightedQuantileFromSorted(sorted, 1.0 - winsor_fraction, total_weight);

    var total: f64 = 0.0;
    for (sorted) |item| {
        if (!(item.weight > 0.0)) continue;
        total += item.weight * @min(@max(item.value, lower), upper);
    }
    return total / total_weight;
}

fn withRowWeightedRobustMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    fraction: f64,
    comptime op: enum { trimmed_mean, winsorized_mean },
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(fraction) or fraction < 0.0 or fraction >= 0.5) return error.InvalidShape;

    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (!(weight > 0.0)) continue;
            scratch[count] = .{ .value = flat.values[offset], .weight = weight };
            total_weight += weight;
            count += 1;
        }
        if (count == 0 or !(total_weight > 0.0)) continue;

        const active = scratch[0..count];
        std.sort.insertion(RowWeightedValue, active, {}, rowWeightedValueLess);
        values[row] = switch (op) {
            .trimmed_mean => rowWeightedTrimmedMeanFromSorted(active, total_weight, fraction),
            .winsorized_mean => rowWeightedWinsorizedMeanFromSorted(active, total_weight, fraction),
        };
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedTrimmedMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    trim_fraction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedRobustMean(DeviceDataFrame, input, value_names, weight_names, output_name, trim_fraction, .trimmed_mean);
}

pub fn withRowWeightedWinsorizedMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    winsor_fraction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedRobustMean(DeviceDataFrame, input, value_names, weight_names, output_name, winsor_fraction, .winsorized_mean);
}

fn withRowCumulativeWeightedRobustMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    fraction: f64,
    comptime op: enum { trimmed_mean, winsorized_mean },
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(fraction) or fraction < 0.0 or fraction >= 0.5) return error.InvalidShape;

    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (weight > 0.0) {
                scratch[count] = .{ .value = flat.values[offset], .weight = weight };
                total_weight += weight;
                count += 1;
            }
            if (!(total_weight > 0.0)) continue;

            const active = scratch[0..count];
            std.sort.insertion(RowWeightedValue, active, {}, rowWeightedValueLess);
            cumulative[offset] = switch (op) {
                .trimmed_mean => rowWeightedTrimmedMeanFromSorted(active, total_weight, fraction),
                .winsorized_mean => rowWeightedWinsorizedMeanFromSorted(active, total_weight, fraction),
            };
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedTrimmedMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    trim_fraction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedRobustMean(DeviceDataFrame, input, value_names, weight_names, output_names, trim_fraction, .trimmed_mean);
}

pub fn withRowCumulativeWeightedWinsorizedMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    winsor_fraction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedRobustMean(DeviceDataFrame, input, value_names, weight_names, output_names, winsor_fraction, .winsorized_mean);
}

pub const withRowCumWeightedTrimmedMean = withRowCumulativeWeightedTrimmedMean;
pub const withRowPrefixWeightedTrimmedMean = withRowCumulativeWeightedTrimmedMean;
pub const withRowCumWeightedWinsorizedMean = withRowCumulativeWeightedWinsorizedMean;
pub const withRowPrefixWeightedWinsorizedMean = withRowCumulativeWeightedWinsorizedMean;

const RowWeightedPercentileShapeOp = enum { interdecile_range, midhinge, trimean, bowley_skewness, quartile_coeff_dispersion, kelley_skewness };

fn finishRowWeightedPercentileShape(q10: f64, q25: f64, q50: f64, q75: f64, q90: f64, comptime op: RowWeightedPercentileShapeOp) f64 {
    return switch (op) {
        .interdecile_range => q90 - q10,
        .midhinge => (q25 + q75) / 2.0,
        .trimean => (q25 + 2.0 * q50 + q75) / 4.0,
        .bowley_skewness => blk: {
            const denominator = q75 - q25;
            break :blk if (denominator == 0.0) quietNanF64() else (q75 + q25 - 2.0 * q50) / denominator;
        },
        .quartile_coeff_dispersion => blk: {
            const denominator = q75 + q25;
            break :blk if (denominator == 0.0) quietNanF64() else (q75 - q25) / denominator;
        },
        .kelley_skewness => blk: {
            const denominator = q90 - q10;
            break :blk if (denominator == 0.0) quietNanF64() else (q90 + q10 - 2.0 * q50) / denominator;
        },
    };
}

fn withRowCumulativeWeightedPercentileShape(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime op: RowWeightedPercentileShapeOp,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (weight > 0.0) {
                scratch[count] = .{ .value = flat.values[offset], .weight = weight };
                total_weight += weight;
                count += 1;
            }
            if (!(total_weight > 0.0)) continue;

            const active = scratch[0..count];
            std.sort.insertion(RowWeightedValue, active, {}, rowWeightedValueLess);
            const q10 = rowWeightedQuantileFromSorted(active, 0.10, total_weight);
            const q25 = rowWeightedQuantileFromSorted(active, 0.25, total_weight);
            const q50 = rowWeightedQuantileFromSorted(active, 0.50, total_weight);
            const q75 = rowWeightedQuantileFromSorted(active, 0.75, total_weight);
            const q90 = rowWeightedQuantileFromSorted(active, 0.90, total_weight);
            cumulative[offset] = finishRowWeightedPercentileShape(q10, q25, q50, q75, q90, op);
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedInterdecileRange(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_names, .interdecile_range);
}

pub fn withRowCumulativeWeightedMidhinge(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_names, .midhinge);
}

pub fn withRowCumulativeWeightedTrimean(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_names, .trimean);
}

pub fn withRowCumulativeWeightedBowleySkewness(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_names, .bowley_skewness);
}

pub fn withRowCumulativeWeightedQuartileCoeffDispersion(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_names, .quartile_coeff_dispersion);
}

pub fn withRowCumulativeWeightedKelleySkewness(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_names, .kelley_skewness);
}

pub const withRowCumulativeWeightedIdr = withRowCumulativeWeightedInterdecileRange;
pub const withRowCumulativeWeightedIDR = withRowCumulativeWeightedInterdecileRange;
pub const withRowCumWeightedInterdecileRange = withRowCumulativeWeightedInterdecileRange;
pub const withRowCumWeightedIdr = withRowCumulativeWeightedInterdecileRange;
pub const withRowCumWeightedIDR = withRowCumulativeWeightedInterdecileRange;
pub const withRowPrefixWeightedInterdecileRange = withRowCumulativeWeightedInterdecileRange;
pub const withRowPrefixWeightedIdr = withRowCumulativeWeightedInterdecileRange;
pub const withRowPrefixWeightedIDR = withRowCumulativeWeightedInterdecileRange;
pub const withRowCumWeightedMidhinge = withRowCumulativeWeightedMidhinge;
pub const withRowPrefixWeightedMidhinge = withRowCumulativeWeightedMidhinge;
pub const withRowCumWeightedTrimean = withRowCumulativeWeightedTrimean;
pub const withRowPrefixWeightedTrimean = withRowCumulativeWeightedTrimean;
pub const withRowCumulativeWeightedBowleySkew = withRowCumulativeWeightedBowleySkewness;
pub const withRowCumWeightedBowleySkewness = withRowCumulativeWeightedBowleySkewness;
pub const withRowCumWeightedBowleySkew = withRowCumulativeWeightedBowleySkewness;
pub const withRowPrefixWeightedBowleySkewness = withRowCumulativeWeightedBowleySkewness;
pub const withRowPrefixWeightedBowleySkew = withRowCumulativeWeightedBowleySkewness;
pub const withRowCumulativeWeightedQcd = withRowCumulativeWeightedQuartileCoeffDispersion;
pub const withRowCumulativeWeightedQCD = withRowCumulativeWeightedQuartileCoeffDispersion;
pub const withRowCumWeightedQuartileCoeffDispersion = withRowCumulativeWeightedQuartileCoeffDispersion;
pub const withRowCumWeightedQcd = withRowCumulativeWeightedQuartileCoeffDispersion;
pub const withRowCumWeightedQCD = withRowCumulativeWeightedQuartileCoeffDispersion;
pub const withRowPrefixWeightedQuartileCoeffDispersion = withRowCumulativeWeightedQuartileCoeffDispersion;
pub const withRowPrefixWeightedQcd = withRowCumulativeWeightedQuartileCoeffDispersion;
pub const withRowPrefixWeightedQCD = withRowCumulativeWeightedQuartileCoeffDispersion;
pub const withRowCumulativeWeightedKelleySkew = withRowCumulativeWeightedKelleySkewness;
pub const withRowCumWeightedKelleySkewness = withRowCumulativeWeightedKelleySkewness;
pub const withRowCumWeightedKelleySkew = withRowCumulativeWeightedKelleySkewness;
pub const withRowPrefixWeightedKelleySkewness = withRowCumulativeWeightedKelleySkewness;
pub const withRowPrefixWeightedKelleySkew = withRowCumulativeWeightedKelleySkewness;

fn withRowWeightedPercentileShape(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime op: RowWeightedPercentileShapeOp,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (!(weight > 0.0)) continue;
            scratch[count] = .{ .value = flat.values[offset], .weight = weight };
            total_weight += weight;
            count += 1;
        }
        if (count == 0 or !(total_weight > 0.0)) continue;

        const active = scratch[0..count];
        std.sort.insertion(RowWeightedValue, active, {}, rowWeightedValueLess);
        const q10 = rowWeightedQuantileFromSorted(active, 0.10, total_weight);
        const q25 = rowWeightedQuantileFromSorted(active, 0.25, total_weight);
        const q50 = rowWeightedQuantileFromSorted(active, 0.50, total_weight);
        const q75 = rowWeightedQuantileFromSorted(active, 0.75, total_weight);
        const q90 = rowWeightedQuantileFromSorted(active, 0.90, total_weight);
        values[row] = finishRowWeightedPercentileShape(q10, q25, q50, q75, q90, op);
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedInterdecileRange(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_name, .interdecile_range);
}

pub fn withRowWeightedMidhinge(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_name, .midhinge);
}

pub fn withRowWeightedTrimean(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_name, .trimean);
}

pub fn withRowWeightedBowleySkewness(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_name, .bowley_skewness);
}

pub fn withRowWeightedQuartileCoeffDispersion(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_name, .quartile_coeff_dispersion);
}

pub fn withRowWeightedKelleySkewness(comptime DeviceDataFrame: type, input: DeviceDataFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedPercentileShape(DeviceDataFrame, input, value_names, weight_names, output_name, .kelley_skewness);
}

pub const withRowWeightedIdr = withRowWeightedInterdecileRange;
pub const withRowWeightedIDR = withRowWeightedInterdecileRange;
pub const withRowWeightedIQR = withRowWeightedIqr;
pub const withRowWeightedMAD = withRowWeightedMad;
pub const withRowWeightedBowleySkew = withRowWeightedBowleySkewness;
pub const withRowWeightedQcd = withRowWeightedQuartileCoeffDispersion;
pub const withRowWeightedQCD = withRowWeightedQuartileCoeffDispersion;
pub const withRowWeightedKelleySkew = withRowWeightedKelleySkewness;

fn withRowCumulativeWeightedQuantileCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    q: f64,
    subtract_q: ?f64,
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(q) or q < 0.0 or q > 1.0) return error.InvalidShape;
    if (subtract_q) |lo_q| {
        if (std.math.isNan(lo_q) or lo_q < 0.0 or lo_q > 1.0) return error.InvalidShape;
    }

    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            scratch[count] = .{ .value = flat.values[offset], .weight = flat.weights[offset] };
            total_weight += flat.weights[offset];
            count += 1;
            if (!(total_weight > 0.0)) continue;

            const active = scratch[0..count];
            std.sort.insertion(RowWeightedValue, active, {}, rowWeightedValueLess);
            const hi = rowWeightedQuantileFromSorted(active, q, total_weight);
            cumulative[offset] = if (subtract_q) |lo_q| hi - rowWeightedQuantileFromSorted(active, lo_q, total_weight) else hi;
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedQuantile(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    q: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedQuantileCore(DeviceDataFrame, input, value_names, weight_names, output_names, q, null);
}

pub fn withRowCumulativeWeightedMedian(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedQuantile(DeviceDataFrame, input, value_names, weight_names, output_names, 0.5);
}

pub fn withRowCumulativeWeightedIqr(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedQuantileCore(DeviceDataFrame, input, value_names, weight_names, output_names, 0.75, 0.25);
}

pub fn withRowCumulativeWeightedMad(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    const deviations = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(deviations);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            scratch[count] = .{ .value = flat.values[offset], .weight = flat.weights[offset] };
            total_weight += flat.weights[offset];
            count += 1;
            if (!(total_weight > 0.0)) continue;

            const active = scratch[0..count];
            std.sort.insertion(RowWeightedValue, active, {}, rowWeightedValueLess);
            const center = rowWeightedQuantileFromSorted(active, 0.5, total_weight);
            for (active, deviations[0..count]) |item, *deviation| {
                deviation.* = .{ .value = @abs(item.value - center), .weight = item.weight };
            }
            const active_deviations = deviations[0..count];
            std.sort.insertion(RowWeightedValue, active_deviations, {}, rowWeightedValueLess);
            cumulative[offset] = rowWeightedQuantileFromSorted(active_deviations, 0.5, total_weight);
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub const withRowCumWeightedQuantile = withRowCumulativeWeightedQuantile;
pub const withRowPrefixWeightedQuantile = withRowCumulativeWeightedQuantile;
pub const withRowCumWeightedMedian = withRowCumulativeWeightedMedian;
pub const withRowPrefixWeightedMedian = withRowCumulativeWeightedMedian;
pub const withRowCumulativeWeightedIQR = withRowCumulativeWeightedIqr;
pub const withRowCumWeightedIqr = withRowCumulativeWeightedIqr;
pub const withRowCumWeightedIQR = withRowCumulativeWeightedIqr;
pub const withRowPrefixWeightedIqr = withRowCumulativeWeightedIqr;
pub const withRowPrefixWeightedIQR = withRowCumulativeWeightedIqr;
pub const withRowCumulativeWeightedMAD = withRowCumulativeWeightedMad;
pub const withRowCumWeightedMad = withRowCumulativeWeightedMad;
pub const withRowCumWeightedMAD = withRowCumulativeWeightedMad;
pub const withRowPrefixWeightedMad = withRowCumulativeWeightedMad;
pub const withRowPrefixWeightedMAD = withRowCumulativeWeightedMad;

pub fn withRowCumulativeWeightedMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;

            var found = false;
            var best_value: f64 = 0.0;
            var best_weight: f64 = 0.0;
            var row_weight: f64 = 0.0;
            for (0..col_index + 1) |candidate_index| {
                const candidate_offset = row * flat.width + candidate_index;
                if (!flat.validity[candidate_offset]) continue;
                row_weight += flat.weights[candidate_offset];
                const candidate = flat.values[candidate_offset];

                var seen = false;
                for (0..candidate_index) |previous_index| {
                    const previous_offset = row * flat.width + previous_index;
                    if (!flat.validity[previous_offset]) continue;
                    if (rowModeValueEqual(flat.values[previous_offset], candidate)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;

                var candidate_weight: f64 = 0.0;
                for (candidate_index..col_index + 1) |match_index| {
                    const match_offset = row * flat.width + match_index;
                    if (!flat.validity[match_offset]) continue;
                    if (rowModeValueEqual(candidate, flat.values[match_offset])) candidate_weight += flat.weights[match_offset];
                }

                if (!found or candidate_weight > best_weight) {
                    best_value = candidate;
                    best_weight = candidate_weight;
                    found = true;
                }
            }
            if (!found or !(row_weight > 0.0)) continue;
            cumulative[offset] = best_value;
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

const RowCumulativeWeightedModeDiagnostic = enum { weight, ratio, margin, margin_ratio };

fn withRowCumulativeWeightedModeDiagnostic(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowCumulativeWeightedModeDiagnostic,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;

            var found = false;
            var best_weight: f64 = 0.0;
            var second_weight: f64 = 0.0;
            var row_weight: f64 = 0.0;
            for (0..col_index + 1) |candidate_index| {
                const candidate_offset = row * flat.width + candidate_index;
                if (!flat.validity[candidate_offset]) continue;
                row_weight += flat.weights[candidate_offset];
                const candidate = flat.values[candidate_offset];

                var seen = false;
                for (0..candidate_index) |previous_index| {
                    const previous_offset = row * flat.width + previous_index;
                    if (!flat.validity[previous_offset]) continue;
                    if (rowModeValueEqual(flat.values[previous_offset], candidate)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;

                var candidate_weight: f64 = 0.0;
                for (candidate_index..col_index + 1) |match_index| {
                    const match_offset = row * flat.width + match_index;
                    if (!flat.validity[match_offset]) continue;
                    if (rowModeValueEqual(candidate, flat.values[match_offset])) candidate_weight += flat.weights[match_offset];
                }

                if (!found or candidate_weight > best_weight) {
                    second_weight = best_weight;
                    best_weight = candidate_weight;
                    found = true;
                } else if (candidate_weight > second_weight) {
                    second_weight = candidate_weight;
                }
            }
            if (!found or !(row_weight > 0.0)) continue;
            cumulative[offset] = switch (reduction) {
                .weight => best_weight,
                .ratio => best_weight / row_weight,
                .margin => best_weight - second_weight,
                .margin_ratio => (best_weight - second_weight) / row_weight,
            };
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedModeWeight(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedModeDiagnostic(DeviceDataFrame, input, value_names, weight_names, output_names, .weight);
}

pub fn withRowCumulativeWeightedModeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedModeDiagnostic(DeviceDataFrame, input, value_names, weight_names, output_names, .ratio);
}

pub fn withRowCumulativeWeightedModeMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedModeDiagnostic(DeviceDataFrame, input, value_names, weight_names, output_names, .margin);
}

pub fn withRowCumulativeWeightedModeMarginRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedModeDiagnostic(DeviceDataFrame, input, value_names, weight_names, output_names, .margin_ratio);
}

pub const withRowCumWeightedMode = withRowCumulativeWeightedMode;
pub const withRowPrefixWeightedMode = withRowCumulativeWeightedMode;
pub const withRowCumWeightedModeWeight = withRowCumulativeWeightedModeWeight;
pub const withRowPrefixWeightedModeWeight = withRowCumulativeWeightedModeWeight;
pub const withRowCumWeightedModeRatio = withRowCumulativeWeightedModeRatio;
pub const withRowPrefixWeightedModeRatio = withRowCumulativeWeightedModeRatio;
pub const withRowCumWeightedModeMargin = withRowCumulativeWeightedModeMargin;
pub const withRowPrefixWeightedModeMargin = withRowCumulativeWeightedModeMargin;
pub const withRowCumWeightedModeMarginRatio = withRowCumulativeWeightedModeMarginRatio;
pub const withRowPrefixWeightedModeMarginRatio = withRowCumulativeWeightedModeMarginRatio;

pub fn withRowWeightedMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..flat.rows) |row| {
        var found = false;
        var best_value: f64 = 0.0;
        var best_weight: f64 = 0.0;
        var row_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            row_weight += flat.weights[offset];
            const candidate = flat.values[offset];

            var seen = false;
            for (0..col_index) |previous_index| {
                const previous_offset = row * flat.width + previous_index;
                if (!flat.validity[previous_offset]) continue;
                if (rowModeValueEqual(flat.values[previous_offset], candidate)) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;

            var candidate_weight: f64 = 0.0;
            for (col_index..flat.width) |candidate_index| {
                const candidate_offset = row * flat.width + candidate_index;
                if (!flat.validity[candidate_offset]) continue;
                if (rowModeValueEqual(candidate, flat.values[candidate_offset])) candidate_weight += flat.weights[candidate_offset];
            }

            // Preserve row mode's stable tie-break: the first distinct valid
            // value wins when weighted frequencies are equal.
            if (!found or candidate_weight > best_weight) {
                best_value = candidate;
                best_weight = candidate_weight;
                found = true;
            }
        }
        if (found and row_weight > 0.0) {
            values[row] = best_value;
            validity[row] = true;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

const RowWeightedModeDiagnostic = enum { weight, ratio, margin_ratio };

fn withRowWeightedModeDiagnostic(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowWeightedModeDiagnostic,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..flat.rows) |row| {
        var found = false;
        var best_weight: f64 = 0.0;
        var second: f64 = 0.0;
        var row_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            row_weight += flat.weights[offset];
            const candidate = flat.values[offset];

            var seen = false;
            for (0..col_index) |previous_index| {
                const previous_offset = row * flat.width + previous_index;
                if (!flat.validity[previous_offset]) continue;
                if (rowModeValueEqual(flat.values[previous_offset], candidate)) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;

            var candidate_weight: f64 = 0.0;
            for (col_index..flat.width) |candidate_index| {
                const candidate_offset = row * flat.width + candidate_index;
                if (!flat.validity[candidate_offset]) continue;
                if (rowModeValueEqual(candidate, flat.values[candidate_offset])) candidate_weight += flat.weights[candidate_offset];
            }
            if (!found or candidate_weight > best_weight) {
                second = best_weight;
                best_weight = candidate_weight;
                found = true;
            } else if (candidate_weight > second) {
                second = candidate_weight;
            }
        }
        if (found and row_weight > 0.0) {
            values[row] = switch (reduction) {
                .weight => best_weight,
                .ratio => best_weight / row_weight,
                .margin_ratio => (best_weight - second) / row_weight,
            };
            validity[row] = true;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedModeWeight(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedModeDiagnostic(DeviceDataFrame, input, value_names, weight_names, output_name, .weight);
}

pub fn withRowWeightedModeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedModeDiagnostic(DeviceDataFrame, input, value_names, weight_names, output_name, .ratio);
}

pub fn withRowWeightedModeMarginRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedModeDiagnostic(DeviceDataFrame, input, value_names, weight_names, output_name, .margin_ratio);
}

pub fn withRowWeightedModeMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);
    for (0..flat.rows) |row| {
        var best: f64 = 0.0;
        var second: f64 = 0.0;
        var row_weight: f64 = 0.0;
        var found = false;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            row_weight += flat.weights[offset];
            const candidate = flat.values[offset];
            var seen = false;
            for (0..col_index) |previous_index| {
                const previous_offset = row * flat.width + previous_index;
                if (!flat.validity[previous_offset]) continue;
                if (rowModeValueEqual(flat.values[previous_offset], candidate)) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;
            var candidate_weight: f64 = 0.0;
            for (col_index..flat.width) |candidate_index| {
                const candidate_offset = row * flat.width + candidate_index;
                if (!flat.validity[candidate_offset]) continue;
                if (rowModeValueEqual(candidate, flat.values[candidate_offset])) candidate_weight += flat.weights[candidate_offset];
            }
            found = true;
            if (candidate_weight > best) {
                second = best;
                best = candidate_weight;
            } else if (candidate_weight > second) {
                second = candidate_weight;
            }
        }
        if (found and row_weight > 0.0) {
            values[row] = best - second;
            validity[row] = true;
        }
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

const RowWeightedDistributionReduction = enum { entropy, gini_impurity, perplexity, inverse_simpson, simpson_concentration, evenness };

fn rowWeightedDistributionFromPrefix(
    flat: RowWeightedFlat,
    row: usize,
    limit: usize,
    comptime reduction: RowWeightedDistributionReduction,
) ?f64 {
    var total_weight: f64 = 0.0;
    for (0..limit) |col_index| {
        const offset = row * flat.width + col_index;
        if (flat.validity[offset]) total_weight += flat.weights[offset];
    }
    if (!(total_weight > 0.0)) return null;

    var entropy: f64 = 0.0;
    var sum_prob_sq: f64 = 0.0;
    var distinct_count: usize = 0;
    for (0..limit) |col_index| {
        const offset = row * flat.width + col_index;
        if (!flat.validity[offset]) continue;
        const candidate = flat.values[offset];

        var seen = false;
        for (0..col_index) |previous_index| {
            const previous_offset = row * flat.width + previous_index;
            if (!flat.validity[previous_offset]) continue;
            if (rowModeValueEqual(flat.values[previous_offset], candidate)) {
                seen = true;
                break;
            }
        }
        if (seen) continue;

        var candidate_weight: f64 = 0.0;
        for (col_index..limit) |candidate_index| {
            const candidate_offset = row * flat.width + candidate_index;
            if (!flat.validity[candidate_offset]) continue;
            if (rowModeValueEqual(candidate, flat.values[candidate_offset])) candidate_weight += flat.weights[candidate_offset];
        }
        if (!(candidate_weight > 0.0)) continue;
        const probability = candidate_weight / total_weight;
        entropy -= probability * std.math.log(f64, std.math.e, probability);
        sum_prob_sq += probability * probability;
        distinct_count += 1;
    }

    return switch (reduction) {
        .entropy => entropy,
        .gini_impurity => 1.0 - sum_prob_sq,
        .perplexity => std.math.exp(entropy),
        .inverse_simpson => if (sum_prob_sq == 0.0) quietNanF64() else 1.0 / sum_prob_sq,
        .simpson_concentration => sum_prob_sq,
        .evenness => if (distinct_count <= 1) 1.0 else entropy / std.math.log(f64, std.math.e, @as(f64, @floatFromInt(distinct_count))),
    };
}

fn withRowCumulativeWeightedDistributionReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowWeightedDistributionReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    for (0..flat.rows) |row| {
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            if (rowWeightedDistributionFromPrefix(flat, row, col_index + 1, reduction)) |value| {
                cumulative[offset] = value;
                cumulative_validity[offset] = true;
            }
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedEntropy(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_names, .entropy);
}

pub fn withRowCumulativeWeightedGiniImpurity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_names, .gini_impurity);
}

pub const withRowCumulativeWeightedGini = withRowCumulativeWeightedGiniImpurity;

pub fn withRowCumulativeWeightedPerplexity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_names, .perplexity);
}

pub fn withRowCumulativeWeightedInverseSimpson(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_names, .inverse_simpson);
}

pub fn withRowCumulativeWeightedSimpsonConcentration(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_names, .simpson_concentration);
}

pub const withRowCumulativeWeightedConcentration = withRowCumulativeWeightedSimpsonConcentration;

pub fn withRowCumulativeWeightedEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_names, .evenness);
}

pub const withRowCumWeightedEntropy = withRowCumulativeWeightedEntropy;
pub const withRowPrefixWeightedEntropy = withRowCumulativeWeightedEntropy;
pub const withRowCumWeightedGiniImpurity = withRowCumulativeWeightedGiniImpurity;
pub const withRowCumWeightedGini = withRowCumulativeWeightedGiniImpurity;
pub const withRowPrefixWeightedGiniImpurity = withRowCumulativeWeightedGiniImpurity;
pub const withRowPrefixWeightedGini = withRowCumulativeWeightedGiniImpurity;
pub const withRowCumWeightedPerplexity = withRowCumulativeWeightedPerplexity;
pub const withRowPrefixWeightedPerplexity = withRowCumulativeWeightedPerplexity;
pub const withRowCumWeightedInverseSimpson = withRowCumulativeWeightedInverseSimpson;
pub const withRowPrefixWeightedInverseSimpson = withRowCumulativeWeightedInverseSimpson;
pub const withRowCumWeightedSimpsonConcentration = withRowCumulativeWeightedSimpsonConcentration;
pub const withRowCumWeightedConcentration = withRowCumulativeWeightedSimpsonConcentration;
pub const withRowPrefixWeightedSimpsonConcentration = withRowCumulativeWeightedSimpsonConcentration;
pub const withRowPrefixWeightedConcentration = withRowCumulativeWeightedSimpsonConcentration;
pub const withRowCumWeightedEvenness = withRowCumulativeWeightedEvenness;
pub const withRowPrefixWeightedEvenness = withRowCumulativeWeightedEvenness;

fn withRowWeightedDistributionReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowWeightedDistributionReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..flat.rows) |row| {
        if (rowWeightedDistributionFromPrefix(flat, row, flat.width, reduction)) |value| {
            values[row] = value;
            validity[row] = true;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedEntropy(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_name, .entropy);
}

pub fn withRowWeightedGiniImpurity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_name, .gini_impurity);
}

pub const withRowWeightedGini = withRowWeightedGiniImpurity;

pub fn withRowWeightedPerplexity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_name, .perplexity);
}

pub fn withRowWeightedInverseSimpson(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_name, .inverse_simpson);
}

pub fn withRowWeightedSimpsonConcentration(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_name, .simpson_concentration);
}

pub const withRowWeightedConcentration = withRowWeightedSimpsonConcentration;

pub fn withRowWeightedEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_name, .evenness);
}

const RowWeightedInequalityReduction = enum { mean_abs_dev, mean_abs_dev_ratio, gini_mean_diff, gini_coefficient };

const RowWeightedInequalityStats = struct {
    mean: f64,
    mean_abs_dev: f64,
    mean_diff: f64,
};

fn rowWeightedInequalityStats(active: []const RowWeightedValue, total_weight: f64) RowWeightedInequalityStats {
    if (!(total_weight > 0.0)) return .{
        .mean = quietNanF64(),
        .mean_abs_dev = quietNanF64(),
        .mean_diff = quietNanF64(),
    };

    var weighted_sum: f64 = 0.0;
    for (active) |item| {
        if (!(item.weight > 0.0)) continue;
        weighted_sum += item.value * item.weight;
    }
    const mean = weighted_sum / total_weight;

    var deviation_sum: f64 = 0.0;
    for (active) |item| {
        if (!(item.weight > 0.0)) continue;
        deviation_sum += item.weight * @abs(item.value - mean);
    }

    var pair_weight_sum: f64 = 0.0;
    var pair_diff_sum: f64 = 0.0;
    for (active, 0..) |lhs, lhs_index| {
        if (!(lhs.weight > 0.0)) continue;
        for (active[lhs_index + 1 ..]) |rhs| {
            if (!(rhs.weight > 0.0)) continue;
            // Match the grouped weighted inequality contract: unordered
            // distinct row pairs are averaged with product weights as support.
            const pair_weight = lhs.weight * rhs.weight;
            pair_weight_sum += pair_weight;
            pair_diff_sum += pair_weight * @abs(lhs.value - rhs.value);
        }
    }

    return .{
        .mean = mean,
        .mean_abs_dev = deviation_sum / total_weight,
        .mean_diff = if (pair_weight_sum > 0.0) pair_diff_sum / pair_weight_sum else 0.0,
    };
}

fn finishRowWeightedInequality(stats: RowWeightedInequalityStats, comptime reduction: RowWeightedInequalityReduction) f64 {
    return switch (reduction) {
        .mean_abs_dev => stats.mean_abs_dev,
        .mean_abs_dev_ratio => if (stats.mean == 0.0) quietNanF64() else stats.mean_abs_dev / @abs(stats.mean),
        .gini_mean_diff => stats.mean_diff,
        .gini_coefficient => if (stats.mean == 0.0) quietNanF64() else stats.mean_diff / (2.0 * @abs(stats.mean)),
    };
}

fn withRowCumulativeWeightedInequality(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowWeightedInequalityReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();
    try validateRowCumulativeWeightedOutputs(output_names, flat.width);

    const cumulative = try input.allocator.alloc(f64, flat.rows * flat.width);
    defer input.allocator.free(cumulative);
    const cumulative_validity = try input.allocator.alloc(bool, flat.rows * flat.width);
    defer input.allocator.free(cumulative_validity);
    @memset(cumulative, 0.0);
    @memset(cumulative_validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (weight > 0.0) {
                scratch[count] = .{ .value = flat.values[offset], .weight = weight };
                total_weight += weight;
                count += 1;
            }
            if (!(total_weight > 0.0)) continue;
            const stats = rowWeightedInequalityStats(scratch[0..count], total_weight);
            cumulative[offset] = finishRowWeightedInequality(stats, reduction);
            cumulative_validity[offset] = true;
        }
    }

    return withRowCumulativeWeightedOutputColumns(DeviceDataFrame, input, output_names, flat.rows, flat.width, cumulative, cumulative_validity);
}

pub fn withRowCumulativeWeightedMeanAbsDev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedInequality(DeviceDataFrame, input, value_names, weight_names, output_names, .mean_abs_dev);
}

pub fn withRowCumulativeWeightedMeanAbsDevRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedInequality(DeviceDataFrame, input, value_names, weight_names, output_names, .mean_abs_dev_ratio);
}

pub fn withRowCumulativeWeightedGiniMeanDiff(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedInequality(DeviceDataFrame, input, value_names, weight_names, output_names, .gini_mean_diff);
}

pub fn withRowCumulativeWeightedGiniCoefficient(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeWeightedInequality(DeviceDataFrame, input, value_names, weight_names, output_names, .gini_coefficient);
}

pub const withRowCumulativeWeightedMeanAbsoluteDeviation = withRowCumulativeWeightedMeanAbsDev;
pub const withRowCumulativeWeightedMadRatio = withRowCumulativeWeightedMeanAbsDevRatio;
pub const withRowCumulativeWeightedGiniCoeff = withRowCumulativeWeightedGiniCoefficient;
pub const withRowCumWeightedMeanAbsDev = withRowCumulativeWeightedMeanAbsDev;
pub const withRowCumWeightedMeanAbsDevRatio = withRowCumulativeWeightedMeanAbsDevRatio;
pub const withRowCumWeightedMeanAbsoluteDeviation = withRowCumulativeWeightedMeanAbsDev;
pub const withRowCumWeightedMadRatio = withRowCumulativeWeightedMeanAbsDevRatio;
pub const withRowCumWeightedGiniMeanDiff = withRowCumulativeWeightedGiniMeanDiff;
pub const withRowCumWeightedGiniCoefficient = withRowCumulativeWeightedGiniCoefficient;
pub const withRowCumWeightedGiniCoeff = withRowCumulativeWeightedGiniCoefficient;
pub const withRowPrefixWeightedMeanAbsDev = withRowCumulativeWeightedMeanAbsDev;
pub const withRowPrefixWeightedMeanAbsDevRatio = withRowCumulativeWeightedMeanAbsDevRatio;
pub const withRowPrefixWeightedMeanAbsoluteDeviation = withRowCumulativeWeightedMeanAbsDev;
pub const withRowPrefixWeightedMadRatio = withRowCumulativeWeightedMeanAbsDevRatio;
pub const withRowPrefixWeightedGiniMeanDiff = withRowCumulativeWeightedGiniMeanDiff;
pub const withRowPrefixWeightedGiniCoefficient = withRowCumulativeWeightedGiniCoefficient;
pub const withRowPrefixWeightedGiniCoeff = withRowCumulativeWeightedGiniCoefficient;

fn withRowWeightedInequality(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowWeightedInequalityReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    var flat = try rowWeightedFlat(DeviceDataFrame, input, value_names, weight_names);
    defer flat.deinit();

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    const scratch = try input.allocator.alloc(RowWeightedValue, flat.width);
    defer input.allocator.free(scratch);
    for (0..flat.rows) |row| {
        var count: usize = 0;
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (!flat.validity[offset]) continue;
            const weight = flat.weights[offset];
            if (!(weight > 0.0)) continue;
            scratch[count] = .{ .value = flat.values[offset], .weight = weight };
            total_weight += weight;
            count += 1;
        }
        if (count == 0 or !(total_weight > 0.0)) continue;

        const stats = rowWeightedInequalityStats(scratch[0..count], total_weight);
        values[row] = finishRowWeightedInequality(stats, reduction);
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWeightedMeanAbsDev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedInequality(DeviceDataFrame, input, value_names, weight_names, output_name, .mean_abs_dev);
}

pub fn withRowWeightedMeanAbsDevRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedInequality(DeviceDataFrame, input, value_names, weight_names, output_name, .mean_abs_dev_ratio);
}

pub fn withRowWeightedGiniMeanDiff(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedInequality(DeviceDataFrame, input, value_names, weight_names, output_name, .gini_mean_diff);
}

pub fn withRowWeightedGiniCoefficient(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedInequality(DeviceDataFrame, input, value_names, weight_names, output_name, .gini_coefficient);
}

pub const withRowWeightedMeanAbsoluteDeviation = withRowWeightedMeanAbsDev;
pub const withRowWeightedMadRatio = withRowWeightedMeanAbsDevRatio;
pub const withRowWeightedGiniCoeff = withRowWeightedGiniCoefficient;

const RowPairedNumericReduction = enum { dot, cosine, squared_euclidean, euclidean, manhattan, chebyshev, canberra, bray_curtis, mean_error, mae, mse, rmse, mape, smape, covariance, correlation, beta };

fn quietNanF64() f64 {
    return @bitCast(@as(u64, 0x7ff8_0000_0000_0000));
}

fn withRowPairedNumericReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowPairedNumericReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    if (lhs_names.len == 0 or lhs_names.len != rhs_names.len) return error.LengthMismatch;

    const dots = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(dots);
    const lhs_norm2 = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(lhs_norm2);
    const rhs_norm2 = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(rhs_norm2);
    const lhs_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(lhs_sums);
    const rhs_sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(rhs_sums);
    const manhattan = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(manhattan);
    const chebyshev = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(chebyshev);
    const canberra = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(canberra);
    const bray_curtis_denominator = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(bray_curtis_denominator);
    const mape_sum = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(mape_sum);
    const smape_sum = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(smape_sum);
    const signed_error_sum = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(signed_error_sum);
    const pair_counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(pair_counts);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(dots, 0.0);
    @memset(lhs_norm2, 0.0);
    @memset(rhs_norm2, 0.0);
    @memset(lhs_sums, 0.0);
    @memset(rhs_sums, 0.0);
    @memset(manhattan, 0.0);
    @memset(chebyshev, 0.0);
    @memset(canberra, 0.0);
    @memset(bray_curtis_denominator, 0.0);
    @memset(mape_sum, 0.0);
    @memset(smape_sum, 0.0);
    @memset(signed_error_sum, 0.0);
    @memset(pair_counts, 0);
    @memset(validity, false);

    for (lhs_names, rhs_names) |lhs_name, rhs_name| {
        const lhs_source = try input.column(lhs_name);
        const rhs_source = try input.column(rhs_name);
        if (!lhs_source.dtype().isReal() or !rhs_source.dtype().isReal()) return error.TypeMismatch;

        switch (lhs_source.*) {
            inline else => |lhs_typed| {
                const lhs_values = try lhs_typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(lhs_values);
                const maybe_lhs_validity = try validityValues(lhs_typed, input.allocator);
                defer if (maybe_lhs_validity) |mask| input.allocator.free(mask);

                switch (rhs_source.*) {
                    inline else => |rhs_typed| {
                        const rhs_values = try rhs_typed.toOwnedSlice(input.allocator);
                        defer input.allocator.free(rhs_values);
                        const maybe_rhs_validity = try validityValues(rhs_typed, input.allocator);
                        defer if (maybe_rhs_validity) |mask| input.allocator.free(mask);

                        for (lhs_values, rhs_values, 0..) |raw_lhs, raw_rhs, row| {
                            const lhs_valid = if (maybe_lhs_validity) |mask| mask[row] else true;
                            const rhs_valid = if (maybe_rhs_validity) |mask| mask[row] else true;
                            if (!lhs_valid or !rhs_valid) continue;
                            const lhs = realValueAsF64(@TypeOf(raw_lhs), raw_lhs);
                            const rhs = realValueAsF64(@TypeOf(raw_rhs), raw_rhs);
                            const signed_error = lhs - rhs;
                            lhs_sums[row] += lhs;
                            rhs_sums[row] += rhs;
                            dots[row] += lhs * rhs;
                            lhs_norm2[row] += lhs * lhs;
                            rhs_norm2[row] += rhs * rhs;
                            signed_error_sum[row] += signed_error;
                            const abs_error = @abs(signed_error);
                            const abs_lhs = @abs(lhs);
                            const abs_rhs = @abs(rhs);
                            const abs_sum = abs_lhs + abs_rhs;
                            manhattan[row] += abs_error;
                            chebyshev[row] = @max(chebyshev[row], abs_error);
                            // Canberra treats a zero/zero coordinate as no contribution, while
                            // Bray-Curtis keeps the zero total denominator so the whole-row result
                            // can surface NaN for an all-zero paired row.
                            canberra[row] += if (abs_sum == 0.0) 0.0 else abs_error / abs_sum;
                            bray_curtis_denominator[row] += abs_sum;
                            mape_sum[row] += if (lhs == 0.0) quietNanF64() else abs_error / abs_lhs;
                            smape_sum[row] += if (abs_sum == 0.0) quietNanF64() else 2.0 * abs_error / abs_sum;
                            pair_counts[row] += 1;
                            validity[row] = true;
                        }
                    },
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    for (values, validity, dots, lhs_norm2, rhs_norm2, lhs_sums, rhs_sums, manhattan, chebyshev, canberra, bray_curtis_denominator, mape_sum, smape_sum, signed_error_sum, pair_counts) |*value, valid, dot, lhs2, rhs2, lhs_sum, rhs_sum, l1, linf, canberra_distance, bray_denominator, ape_sum, symmetric_ape_sum, signed_error, pair_count| {
        if (!valid) {
            value.* = 0.0;
        } else {
            const squared_distance = lhs2 + rhs2 - 2.0 * dot;
            const count_f64: f64 = @floatFromInt(pair_count);
            const mean_lhs = lhs_sum / count_f64;
            const mean_rhs = rhs_sum / count_f64;
            const covariance = dot / count_f64 - mean_lhs * mean_rhs;
            const lhs_variance_raw = lhs2 / count_f64 - mean_lhs * mean_lhs;
            const rhs_variance_raw = rhs2 / count_f64 - mean_rhs * mean_rhs;
            const lhs_variance = if (lhs_variance_raw < 0.0) 0.0 else lhs_variance_raw;
            const rhs_variance = if (rhs_variance_raw < 0.0) 0.0 else rhs_variance_raw;
            value.* = switch (reduction) {
                .dot => dot,
                .cosine => if (lhs2 == 0.0 or rhs2 == 0.0) std.math.nan(f64) else dot / (std.math.sqrt(lhs2) * std.math.sqrt(rhs2)),
                .squared_euclidean => squared_distance,
                .euclidean => std.math.sqrt(squared_distance),
                .manhattan => l1,
                .chebyshev => linf,
                .canberra => canberra_distance,
                .bray_curtis => if (bray_denominator == 0.0) quietNanF64() else l1 / bray_denominator,
                .mean_error => signed_error / count_f64,
                .mae => l1 / count_f64,
                .mse => squared_distance / count_f64,
                .rmse => std.math.sqrt(squared_distance / count_f64),
                .mape => ape_sum / count_f64,
                .smape => symmetric_ape_sum / count_f64,
                .covariance => covariance,
                .correlation => if (lhs_variance == 0.0 or rhs_variance == 0.0) quietNanF64() else covariance / std.math.sqrt(lhs_variance * rhs_variance),
                .beta => if (lhs_variance == 0.0) quietNanF64() else covariance / lhs_variance,
            };
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowDot(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .dot);
}

pub fn withRowCosineSimilarity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .cosine);
}

pub fn withRowCosine(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCosineSimilarity(DeviceDataFrame, input, lhs_names, rhs_names, output_name);
}

pub fn withRowSquaredEuclideanDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .squared_euclidean);
}

pub fn withRowEuclideanDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .euclidean);
}

pub fn withRowManhattanDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .manhattan);
}

pub fn withRowChebyshevDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .chebyshev);
}

pub fn withRowCanberraDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .canberra);
}

pub fn withRowBrayCurtisDistance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .bray_curtis);
}

pub fn withRowMeanError(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    actual_names: []const []const u8,
    predicted_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, actual_names, predicted_names, output_name, .mean_error);
}

pub fn withRowBias(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    actual_names: []const []const u8,
    predicted_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMeanError(DeviceDataFrame, input, actual_names, predicted_names, output_name);
}

pub fn withRowMae(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .mae);
}

pub fn withRowMse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .mse);
}

pub fn withRowRmse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .rmse);
}

pub fn withRowMape(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    actual_names: []const []const u8,
    predicted_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, actual_names, predicted_names, output_name, .mape);
}

pub fn withRowSmape(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    actual_names: []const []const u8,
    predicted_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, actual_names, predicted_names, output_name, .smape);
}

pub fn withRowCovariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .covariance);
}

pub fn withRowCorrelation(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .correlation);
}

pub fn withRowBeta(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPairedNumericReduction(DeviceDataFrame, input, lhs_names, rhs_names, output_name, .beta);
}

const RowNumericArgReduction = enum { argmin, argmax };

const RowNumericReduction = enum { sum, prod, mean, logsumexp, logmeanexp, geometric_mean, magnitude_geometric_mean, harmonic_mean, min, max, ptp, magnitude_ptp, midrange, magnitude_midrange, range_coeff, magnitude_range_coeff, mean_abs, hhi, magnitude_normalized_hhi, magnitude_sparsity, magnitude_inverse_simpson, magnitude_simpson_evenness, magnitude_dominance, magnitude_dominance_margin, magnitude_entropy, magnitude_perplexity, magnitude_evenness, rms, l1_norm, l2_norm };

fn realValueAsF64(comptime T: type, value: T) f64 {
    if (comptime T == array_mod.BFloat16) return value.toF64();
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return 0.0;
    return switch (@typeInfo(T)) {
        .float, .comptime_float => @floatCast(value),
        .int, .comptime_int => @floatFromInt(value),
        .bool => if (value) 1.0 else 0.0,
        else => 0.0,
    };
}

fn withRowNumericArgReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowNumericArgReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const indices = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(indices);
    const best_values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(best_values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(indices, 0);
    @memset(best_values, 0.0);
    @memset(validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        const output_index = std.math.cast(i64, col_index) orelse return error.InvalidShape;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    if (!validity[row]) {
                        best_values[row] = value;
                        indices[row] = output_index;
                        validity[row] = true;
                        continue;
                    }

                    const replace = switch (reduction) {
                        .argmin => std.math.isNan(value) or (!std.math.isNan(best_values[row]) and value < best_values[row]),
                        .argmax => std.math.isNan(value) or (!std.math.isNan(best_values[row]) and value > best_values[row]),
                    };
                    if (replace) {
                        best_values[row] = value;
                        indices[row] = output_index;
                    }
                }
            },
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, indices, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowArgMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericArgReduction(DeviceDataFrame, input, names, output_name, .argmin);
}

pub fn withRowArgMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericArgReduction(DeviceDataFrame, input, names, output_name, .argmax);
}

fn withRowCumulativeNumericArgReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowNumericArgReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const indices = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(indices);
    const best_values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(best_values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(indices, 0);
    @memset(best_values, 0.0);
    @memset(validity, false);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |name, output_name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        const output_index = std.math.cast(i64, col_index) orelse return error.InvalidShape;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    if (!validity[row]) {
                        best_values[row] = value;
                        indices[row] = output_index;
                        validity[row] = true;
                        continue;
                    }
                    const replace = switch (reduction) {
                        .argmin => std.math.isNan(value) or (!std.math.isNan(best_values[row]) and value < best_values[row]),
                        .argmax => std.math.isNan(value) or (!std.math.isNan(best_values[row]) and value > best_values[row]),
                    };
                    if (replace) {
                        best_values[row] = value;
                        indices[row] = output_index;
                    }
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, indices, validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeArgMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericArgReduction(DeviceDataFrame, input, names, output_names, .argmin);
}

pub fn withRowCumArgMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeArgMin(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixArgMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeArgMin(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeArgMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericArgReduction(DeviceDataFrame, input, names, output_names, .argmax);
}

pub fn withRowCumArgMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeArgMax(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixArgMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeArgMax(DeviceDataFrame, input, names, output_names);
}

fn withRowNumericReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowNumericReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    const counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(counts);
    const maxima = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(maxima);
    const secondaries = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(secondaries);
    @memset(values, 0.0);
    @memset(validity, false);
    @memset(counts, 0);
    @memset(maxima, 0.0);
    @memset(secondaries, 0.0);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    switch (reduction) {
                        .sum, .mean => values[row] += value,
                        .mean_abs, .l1_norm => values[row] += @abs(value),
                        .hhi, .magnitude_normalized_hhi, .magnitude_sparsity, .magnitude_inverse_simpson, .magnitude_simpson_evenness => {
                            const magnitude = @abs(value);
                            values[row] += magnitude;
                            maxima[row] += magnitude * magnitude;
                        },
                        .magnitude_dominance => {
                            const magnitude = @abs(value);
                            values[row] += magnitude;
                            maxima[row] = if (validity[row]) @max(maxima[row], magnitude) else magnitude;
                        },
                        .magnitude_dominance_margin => {
                            const magnitude = @abs(value);
                            values[row] += magnitude;
                            if (!validity[row]) {
                                maxima[row] = magnitude;
                            } else if (magnitude > maxima[row]) {
                                secondaries[row] = maxima[row];
                                maxima[row] = magnitude;
                            } else if (magnitude > secondaries[row]) {
                                secondaries[row] = magnitude;
                            }
                        },
                        .magnitude_entropy, .magnitude_perplexity, .magnitude_evenness => {
                            const magnitude = @abs(value);
                            values[row] += magnitude;
                            if (magnitude > 0.0) maxima[row] += magnitude * std.math.log(f64, std.math.e, magnitude);
                        },
                        .rms, .l2_norm => values[row] += value * value,
                        .logsumexp, .logmeanexp => {
                            // Maintain a numerically stable log-sum-exp-style state:
                            // `maxima` is the current row max and `values` is
                            // sum(exp(x - maxima)).  This avoids overflow for
                            // large logits while preserving NaN/+Inf/-Inf
                            // semantics explicitly instead of depending on
                            // undefined inf-inf intermediate cases.
                            if (std.math.isNan(value)) {
                                values[row] = std.math.nan(f64);
                                maxima[row] = std.math.nan(f64);
                            } else if (!validity[row]) {
                                maxima[row] = value;
                                values[row] = 1.0;
                            } else if (!std.math.isNan(values[row])) {
                                if (std.math.isPositiveInf(maxima[row])) {
                                    values[row] = 1.0;
                                } else if (std.math.isPositiveInf(value)) {
                                    maxima[row] = value;
                                    values[row] = 1.0;
                                } else if (value > maxima[row]) {
                                    values[row] = values[row] * std.math.exp(maxima[row] - value) + 1.0;
                                    maxima[row] = value;
                                } else if (!(std.math.isNegativeInf(maxima[row]) and std.math.isNegativeInf(value))) {
                                    values[row] += std.math.exp(value - maxima[row]);
                                }
                            }
                        },
                        .geometric_mean => {
                            if (value < 0.0) {
                                values[row] = std.math.nan(f64);
                            } else if (value == 0.0 and !std.math.isNan(values[row])) {
                                // `values` stores the running log-sum, where a
                                // legitimate product of ones also has value 0.
                                // Use the auxiliary `maxima` slot as a
                                // row-local zero-seen flag for geometric mean
                                // so finalization can distinguish log_sum=0
                                // from a true zero product.
                                maxima[row] = 1.0;
                                values[row] = 0.0;
                            } else if (!validity[row] and maxima[row] == 0.0) {
                                values[row] = std.math.log(f64, std.math.e, value);
                            } else if (!std.math.isNan(values[row]) and maxima[row] == 0.0) {
                                values[row] += std.math.log(f64, std.math.e, value);
                            }
                        },
                        .magnitude_geometric_mean => {
                            const magnitude = @abs(value);
                            if (magnitude == 0.0 and !std.math.isNan(values[row])) {
                                maxima[row] = 1.0;
                                values[row] = 0.0;
                            } else if (!validity[row] and maxima[row] == 0.0) {
                                values[row] = std.math.log(f64, std.math.e, magnitude);
                            } else if (!std.math.isNan(values[row]) and maxima[row] == 0.0) {
                                values[row] += std.math.log(f64, std.math.e, magnitude);
                            }
                        },
                        .harmonic_mean => {
                            if (value == 0.0 and !std.math.isNan(values[row])) {
                                values[row] = std.math.inf(f64);
                            } else if (!validity[row]) {
                                values[row] = 1.0 / value;
                            } else if (!std.math.isInf(values[row])) {
                                values[row] += 1.0 / value;
                            }
                        },
                        .prod => {
                            values[row] = if (validity[row]) values[row] * value else value;
                        },
                        .min => {
                            if (!validity[row] or std.math.isNan(value) or (!std.math.isNan(values[row]) and value < values[row])) {
                                values[row] = value;
                            }
                        },
                        .max => {
                            if (!validity[row] or std.math.isNan(value) or (!std.math.isNan(values[row]) and value > values[row])) {
                                values[row] = value;
                            }
                        },
                        .ptp, .midrange, .range_coeff => {
                            if (!validity[row]) {
                                values[row] = value;
                                maxima[row] = value;
                            } else if (std.math.isNan(value)) {
                                // Preserve NaN evidence for range diagnostics:
                                // a row containing NaN has an undefined
                                // peak-to-peak span even if other values are
                                // finite.
                                values[row] = value;
                                maxima[row] = value;
                            } else if (!std.math.isNan(values[row])) {
                                if (value < values[row]) values[row] = value;
                                if (value > maxima[row]) maxima[row] = value;
                            }
                        },
                        .magnitude_ptp, .magnitude_midrange, .magnitude_range_coeff => {
                            const magnitude = @abs(value);
                            if (!validity[row]) {
                                values[row] = magnitude;
                                maxima[row] = magnitude;
                            } else if (std.math.isNan(magnitude)) {
                                values[row] = magnitude;
                                maxima[row] = magnitude;
                            } else if (!std.math.isNan(values[row])) {
                                if (magnitude < values[row]) values[row] = magnitude;
                                if (magnitude > maxima[row]) maxima[row] = magnitude;
                            }
                        },
                    }
                    counts[row] += 1;
                    validity[row] = true;
                }
            },
        }
    }

    for (values, validity, counts, maxima, secondaries) |*value, valid, count, aux_value, secondary_value| {
        if (!valid) {
            value.* = 0.0;
        } else if (reduction == .mean) {
            value.* /= @floatFromInt(count);
        } else if (reduction == .logsumexp or reduction == .logmeanexp) {
            if (std.math.isNan(value.*) or std.math.isNan(aux_value)) {
                value.* = std.math.nan(f64);
            } else if (std.math.isPositiveInf(aux_value) or std.math.isNegativeInf(aux_value)) {
                value.* = aux_value;
            } else {
                value.* = aux_value + std.math.log(f64, std.math.e, value.*);
                if (reduction == .logmeanexp) value.* -= std.math.log(f64, std.math.e, @as(f64, @floatFromInt(count)));
            }
        } else if (reduction == .geometric_mean or reduction == .magnitude_geometric_mean) {
            if (!std.math.isNan(value.*)) {
                value.* = if (aux_value != 0.0) 0.0 else std.math.exp(value.* / @as(f64, @floatFromInt(count)));
            }
        } else if (reduction == .harmonic_mean) {
            value.* = if (std.math.isInf(value.*)) 0.0 else @as(f64, @floatFromInt(count)) / value.*;
        } else if (reduction == .mean_abs) {
            value.* /= @floatFromInt(count);
        } else if (reduction == .hhi) {
            value.* = if (value.* == 0.0) std.math.nan(f64) else aux_value / (value.* * value.*);
        } else if (reduction == .magnitude_normalized_hhi) {
            if (value.* == 0.0) {
                value.* = std.math.nan(f64);
            } else if (count <= 1) {
                value.* = 1.0;
            } else {
                const concentration = aux_value / (value.* * value.*);
                const uniform_floor = 1.0 / @as(f64, @floatFromInt(count));
                value.* = (concentration - uniform_floor) / (1.0 - uniform_floor);
            }
        } else if (reduction == .magnitude_sparsity) {
            if (value.* == 0.0 or aux_value == 0.0) {
                value.* = std.math.nan(f64);
            } else if (count <= 1) {
                value.* = 1.0;
            } else {
                const sqrt_count = std.math.sqrt(@as(f64, @floatFromInt(count)));
                const l1_over_l2 = value.* / std.math.sqrt(aux_value);
                value.* = (sqrt_count - l1_over_l2) / (sqrt_count - 1.0);
            }
        } else if (reduction == .magnitude_inverse_simpson) {
            value.* = if (value.* == 0.0 or aux_value == 0.0) std.math.nan(f64) else (value.* * value.*) / aux_value;
        } else if (reduction == .magnitude_simpson_evenness) {
            value.* = if (value.* == 0.0 or aux_value == 0.0) std.math.nan(f64) else (value.* * value.*) / (aux_value * @as(f64, @floatFromInt(count)));
        } else if (reduction == .magnitude_dominance) {
            value.* = if (value.* == 0.0) std.math.nan(f64) else aux_value / value.*;
        } else if (reduction == .magnitude_dominance_margin) {
            value.* = if (value.* == 0.0) std.math.nan(f64) else (aux_value - secondary_value) / value.*;
        } else if (reduction == .magnitude_entropy) {
            value.* = if (value.* == 0.0) std.math.nan(f64) else std.math.log(f64, std.math.e, value.*) - aux_value / value.*;
        } else if (reduction == .magnitude_perplexity) {
            value.* = if (value.* == 0.0) std.math.nan(f64) else std.math.exp(std.math.log(f64, std.math.e, value.*) - aux_value / value.*);
        } else if (reduction == .magnitude_evenness) {
            value.* = if (count <= 1) 1.0 else if (value.* == 0.0) std.math.nan(f64) else (std.math.log(f64, std.math.e, value.*) - aux_value / value.*) / std.math.log(f64, std.math.e, @as(f64, @floatFromInt(count)));
        } else if (reduction == .rms) {
            value.* = std.math.sqrt(value.* / @as(f64, @floatFromInt(count)));
        } else if (reduction == .l2_norm) {
            value.* = std.math.sqrt(value.*);
        } else if (reduction == .ptp or reduction == .magnitude_ptp) {
            value.* = aux_value - value.*;
        } else if (reduction == .midrange or reduction == .magnitude_midrange) {
            value.* = (value.* + aux_value) / 2.0;
        } else if (reduction == .range_coeff or reduction == .magnitude_range_coeff) {
            const denominator = aux_value + value.*;
            value.* = if (denominator == 0.0) std.math.nan(f64) else (aux_value - value.*) / denominator;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .sum);
}

pub fn withRowMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .mean);
}

pub fn withRowLogSumExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .logsumexp);
}

pub fn withRowLogsumexp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowLogSumExp(DeviceDataFrame, input, names, output_name);
}

pub fn withRowLogMeanExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .logmeanexp);
}

pub fn withRowLogmeanexp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowLogMeanExp(DeviceDataFrame, input, names, output_name);
}

pub fn withRowCentered(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const means = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(means);
    const counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(counts);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(means, 0.0);
    @memset(counts, 0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    means[row] += realValueAsF64(@TypeOf(raw_value), raw_value);
                    counts[row] += 1;
                    row_validity[row] = true;
                }
            },
        }
    }
    for (means, counts, row_validity) |*mean, count, valid| {
        if (valid) mean.* /= @as(f64, @floatFromInt(count));
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var centered = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(centered);
        const centered_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(centered_validity);
        @memset(centered, 0.0);
        @memset(centered_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    centered_validity[row] = true;
                    centered[row] = realValueAsF64(@TypeOf(raw_value), raw_value) - means[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, centered, centered_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowDemean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCentered(DeviceDataFrame, input, names, output_names);
}

pub fn withRowZScore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const means = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(means);
    const sum_squares = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(sum_squares);
    const counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(counts);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(means, 0.0);
    @memset(sum_squares, 0.0);
    @memset(counts, 0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    means[row] += value;
                    sum_squares[row] += value * value;
                    counts[row] += 1;
                    row_validity[row] = true;
                }
            },
        }
    }
    for (means, sum_squares, counts, row_validity) |*mean, *sum_square, count, valid| {
        if (!valid) continue;
        const count_f: f64 = @floatFromInt(count);
        mean.* /= count_f;
        const variance = sum_square.* / count_f - mean.* * mean.*;
        sum_square.* = if (variance <= 0.0) 0.0 else std.math.sqrt(variance);
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var zscores = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(zscores);
        const zscore_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(zscore_validity);
        @memset(zscores, 0.0);
        @memset(zscore_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    zscore_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    zscores[row] = if (sum_squares[row] == 0.0) std.math.nan(f64) else (value - means[row]) / sum_squares[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, zscores, zscore_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowZscore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowZScore(DeviceDataFrame, input, names, output_names);
}

pub fn withRowStandardize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowZScore(DeviceDataFrame, input, names, output_names);
}

pub fn withRowRobustZScore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const medians = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(medians);
    const mads = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(mads);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(medians, 0.0);
    @memset(mads, 0.0);
    @memset(row_validity, false);

    const scratch = try input.allocator.alloc(f64, check_names.len);
    defer input.allocator.free(scratch);
    const deviations = try input.allocator.alloc(f64, check_names.len);
    defer input.allocator.free(deviations);
    for (0..input.rows) |row| {
        var count: usize = 0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            scratch[count] = flat_values[offset];
            count += 1;
        }
        if (count == 0) continue;

        std.sort.insertion(f64, scratch[0..count], {}, rowQuantileLess);
        const median = rowQuantileFromSorted(scratch[0..count], 0.5);
        for (scratch[0..count], deviations[0..count]) |value, *deviation| {
            deviation.* = @abs(value - median);
        }
        std.sort.insertion(f64, deviations[0..count], {}, rowQuantileLess);
        medians[row] = median;
        mads[row] = rowQuantileFromSorted(deviations[0..count], 0.5);
        row_validity[row] = true;
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    const normal_consistency = 0.6744897501960817;
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var robust_zscores = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(robust_zscores);
        const robust_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(robust_validity);
        @memset(robust_zscores, 0.0);
        @memset(robust_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    robust_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    robust_zscores[row] = if (mads[row] == 0.0) std.math.nan(f64) else normal_consistency * (value - medians[row]) / mads[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, robust_zscores, robust_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowRobustZscore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowRobustZScore(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMadZScore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowRobustZScore(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMadZscore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowRobustZScore(DeviceDataFrame, input, names, output_names);
}

const RowRankStorage = struct {
    flat_values: []f64,
    flat_validity: []bool,
    flat_ordinal_ranks: []i64,
    flat_average_ranks: []f64,
    flat_dense_ranks: []i64,
    flat_competition_ranks: []i64,
    flat_cume_counts: []usize,
    counts: []usize,

    fn deinit(self: @This(), allocator: std.mem.Allocator) void {
        allocator.free(self.flat_values);
        allocator.free(self.flat_validity);
        allocator.free(self.flat_ordinal_ranks);
        allocator.free(self.flat_average_ranks);
        allocator.free(self.flat_dense_ranks);
        allocator.free(self.flat_competition_ranks);
        allocator.free(self.flat_cume_counts);
        allocator.free(self.counts);
    }
};

fn computeRowRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    check_names: []const []const u8,
) DeviceFrameArrayError!RowRankStorage {
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    errdefer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    errdefer input.allocator.free(flat_validity);
    const flat_ordinal_ranks = try input.allocator.alloc(i64, total_slots);
    errdefer input.allocator.free(flat_ordinal_ranks);
    const flat_average_ranks = try input.allocator.alloc(f64, total_slots);
    errdefer input.allocator.free(flat_average_ranks);
    const flat_dense_ranks = try input.allocator.alloc(i64, total_slots);
    errdefer input.allocator.free(flat_dense_ranks);
    const flat_competition_ranks = try input.allocator.alloc(i64, total_slots);
    errdefer input.allocator.free(flat_competition_ranks);
    const flat_cume_counts = try input.allocator.alloc(usize, total_slots);
    errdefer input.allocator.free(flat_cume_counts);
    const counts = try input.allocator.alloc(usize, input.rows);
    errdefer input.allocator.free(counts);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);
    @memset(flat_ordinal_ranks, 0);
    @memset(flat_average_ranks, 0.0);
    @memset(flat_dense_ranks, 0);
    @memset(flat_competition_ranks, 0);
    @memset(flat_cume_counts, 0);
    @memset(counts, 0);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const order = try input.allocator.alloc(usize, check_names.len);
    defer input.allocator.free(order);
    const OrderCtx = struct {
        values: []const f64,

        fn less(ctx: @This(), lhs: usize, rhs: usize) bool {
            return rowQuantileLess({}, ctx.values[lhs], ctx.values[rhs]);
        }
    };
    for (0..input.rows) |row| {
        var count: usize = 0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            order[count] = offset;
            count += 1;
        }
        counts[row] = count;
        if (count == 0) continue;

        std.sort.insertion(usize, order[0..count], OrderCtx{ .values = flat_values }, OrderCtx.less);
        for (order[0..count], 1..) |offset, rank| {
            flat_ordinal_ranks[offset] = @intCast(rank);
        }
        var group_start: usize = 0;
        var dense_rank: i64 = 0;
        while (group_start < count) {
            var group_end = group_start + 1;
            while (group_end < count and rowDenseRankValueEqual(flat_values[order[group_start]], flat_values[order[group_end]])) {
                group_end += 1;
            }

            dense_rank += 1;
            const competition_rank: i64 = @intCast(group_start + 1);
            const cume_count = group_end;
            const average_rank = (@as(f64, @floatFromInt(group_start + 1)) + @as(f64, @floatFromInt(group_end))) / 2.0;
            for (order[group_start..group_end]) |offset| {
                flat_dense_ranks[offset] = dense_rank;
                flat_competition_ranks[offset] = competition_rank;
                flat_average_ranks[offset] = average_rank;
                flat_cume_counts[offset] = cume_count;
            }
            group_start = group_end;
        }
    }

    return .{
        .flat_values = flat_values,
        .flat_validity = flat_validity,
        .flat_ordinal_ranks = flat_ordinal_ranks,
        .flat_average_ranks = flat_average_ranks,
        .flat_dense_ranks = flat_dense_ranks,
        .flat_competition_ranks = flat_competition_ranks,
        .flat_cume_counts = flat_cume_counts,
        .counts = counts,
    };
}

pub fn withRowAverageRank(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const ranks_storage = try computeRowRanks(DeviceDataFrame, input, check_names);
    defer ranks_storage.deinit(input.allocator);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        var ranks = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(ranks);
        const rank_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(rank_validity);
        @memset(ranks, 0.0);
        @memset(rank_validity, false);

        for (0..input.rows) |row| {
            const offset = row * check_names.len + col_index;
            if (!ranks_storage.flat_validity[offset]) continue;
            ranks[row] = ranks_storage.flat_average_ranks[offset];
            rank_validity[row] = true;
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, ranks, rank_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowAverageRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowAverageRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAvgRank(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowAverageRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAvgRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowAverageRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowFractionalRank(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowAverageRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowFractionalRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowAverageRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowOrdinalRank(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const ranks_storage = try computeRowRanks(DeviceDataFrame, input, check_names);
    defer ranks_storage.deinit(input.allocator);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        var ranks = try input.allocator.alloc(i64, input.rows);
        defer input.allocator.free(ranks);
        const rank_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(rank_validity);
        @memset(ranks, 0);
        @memset(rank_validity, false);

        for (0..input.rows) |row| {
            const offset = row * check_names.len + col_index;
            if (!ranks_storage.flat_validity[offset]) continue;
            ranks[row] = ranks_storage.flat_ordinal_ranks[offset];
            rank_validity[row] = true;
        }

        var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, ranks, rank_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowOrdinalRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowOrdinalRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowDenseRank(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const ranks_storage = try computeRowRanks(DeviceDataFrame, input, check_names);
    defer ranks_storage.deinit(input.allocator);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        var ranks = try input.allocator.alloc(i64, input.rows);
        defer input.allocator.free(ranks);
        const rank_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(rank_validity);
        @memset(ranks, 0);
        @memset(rank_validity, false);

        for (0..input.rows) |row| {
            const offset = row * check_names.len + col_index;
            if (!ranks_storage.flat_validity[offset]) continue;
            ranks[row] = ranks_storage.flat_dense_ranks[offset];
            rank_validity[row] = true;
        }

        var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, ranks, rank_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowDenseRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowDenseRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCompetitionRank(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const ranks_storage = try computeRowRanks(DeviceDataFrame, input, check_names);
    defer ranks_storage.deinit(input.allocator);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        var ranks = try input.allocator.alloc(i64, input.rows);
        defer input.allocator.free(ranks);
        const rank_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(rank_validity);
        @memset(ranks, 0);
        @memset(rank_validity, false);

        for (0..input.rows) |row| {
            const offset = row * check_names.len + col_index;
            if (!ranks_storage.flat_validity[offset]) continue;
            ranks[row] = ranks_storage.flat_competition_ranks[offset];
            rank_validity[row] = true;
        }

        var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, ranks, rank_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCompetitionRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCompetitionRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMinRank(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCompetitionRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMinRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCompetitionRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPercentRank(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const ranks_storage = try computeRowRanks(DeviceDataFrame, input, check_names);
    defer ranks_storage.deinit(input.allocator);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        var percent_ranks = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(percent_ranks);
        const percent_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(percent_validity);
        @memset(percent_ranks, 0.0);
        @memset(percent_validity, false);

        for (0..input.rows) |row| {
            const offset = row * check_names.len + col_index;
            if (!ranks_storage.flat_validity[offset]) continue;
            const count = ranks_storage.counts[row];
            percent_ranks[row] = if (count <= 1) 0.0 else @as(f64, @floatFromInt(ranks_storage.flat_competition_ranks[offset] - 1)) / @as(f64, @floatFromInt(count - 1));
            percent_validity[row] = true;
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, percent_ranks, percent_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowPercentRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPercentRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPercentileRank(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPercentRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPercentileRanks(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPercentRank(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumeDist(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const ranks_storage = try computeRowRanks(DeviceDataFrame, input, check_names);
    defer ranks_storage.deinit(input.allocator);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        var cume_dist = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(cume_dist);
        const cume_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(cume_validity);
        @memset(cume_dist, 0.0);
        @memset(cume_validity, false);

        for (0..input.rows) |row| {
            const offset = row * check_names.len + col_index;
            if (!ranks_storage.flat_validity[offset]) continue;
            cume_dist[row] = @as(f64, @floatFromInt(ranks_storage.flat_cume_counts[offset])) / @as(f64, @floatFromInt(ranks_storage.counts[row]));
            cume_validity[row] = true;
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, cume_dist, cume_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumeDistribution(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumeDist(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeDistribution(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumeDist(DeviceDataFrame, input, names, output_names);
}

const RowCumulativeReduction = enum { sum, product, mean, logsumexp, logmeanexp, geometric_mean, harmonic_mean, variance, stddev, sem, cv, fano, skewness, kurtosis, rms, mean_abs, mean_square, l1_norm, l2_norm, max_abs, min_abs, max, min, range };

fn withRowCumulativeRealColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
    comptime reduction: RowCumulativeReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    if (std.math.isNan(correction) or correction < 0.0) return error.InvalidShape;
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const cumulative = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(cumulative);
    @memset(cumulative, 0.0);
    for (0..input.rows) |row| {
        var running: f64 = switch (reduction) {
            .product => 1.0,
            .sum, .mean, .logsumexp, .logmeanexp, .geometric_mean, .harmonic_mean, .variance, .stddev, .sem, .cv, .fano, .skewness, .kurtosis, .rms, .mean_abs, .mean_square, .l1_norm, .l2_norm, .max_abs, .min_abs, .max, .min, .range => 0.0,
        };
        var running_mean: f64 = 0.0;
        var running_m2: f64 = 0.0;
        var running_m3: f64 = 0.0;
        var running_m4: f64 = 0.0;
        var running_min: f64 = 0.0;
        var running_max: f64 = 0.0;
        var log_exp_sum: f64 = 0.0;
        var log_exp_max: f64 = 0.0;
        var geometric_zero = false;
        var running_count: usize = 0;
        var running_valid = false;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            const value = flat_values[offset];
            switch (reduction) {
                .sum => running += value,
                .product => running *= value,
                .mean => {
                    running += value;
                    running_count += 1;
                },
                .logsumexp, .logmeanexp => {
                    running_count += 1;
                    if (std.math.isNan(value)) {
                        log_exp_sum = std.math.nan(f64);
                        log_exp_max = std.math.nan(f64);
                    } else if (!running_valid) {
                        log_exp_max = value;
                        log_exp_sum = 1.0;
                    } else if (!std.math.isNan(log_exp_sum)) {
                        if (std.math.isPositiveInf(log_exp_max)) {
                            log_exp_sum = 1.0;
                        } else if (std.math.isPositiveInf(value)) {
                            log_exp_max = value;
                            log_exp_sum = 1.0;
                        } else if (value > log_exp_max) {
                            log_exp_sum = log_exp_sum * std.math.exp(log_exp_max - value) + 1.0;
                            log_exp_max = value;
                        } else if (!(std.math.isNegativeInf(log_exp_max) and std.math.isNegativeInf(value))) {
                            log_exp_sum += std.math.exp(value - log_exp_max);
                        }
                    }
                    running_valid = true;
                    if (std.math.isNan(log_exp_sum) or std.math.isNan(log_exp_max)) {
                        running = std.math.nan(f64);
                    } else if (std.math.isPositiveInf(log_exp_max) or std.math.isNegativeInf(log_exp_max)) {
                        running = log_exp_max;
                    } else {
                        running = log_exp_max + std.math.log(f64, std.math.e, log_exp_sum);
                        if (reduction == .logmeanexp) running -= std.math.log(f64, std.math.e, @as(f64, @floatFromInt(running_count)));
                    }
                },
                .geometric_mean => {
                    running_count += 1;
                    if (value < 0.0) {
                        running = std.math.nan(f64);
                    } else if (!std.math.isNan(running)) {
                        if (value == 0.0) {
                            geometric_zero = true;
                        } else if (!geometric_zero) {
                            running += std.math.log(f64, std.math.e, value);
                        }
                    }
                },
                .harmonic_mean => {
                    running_count += 1;
                    if (value == 0.0 and !std.math.isNan(running)) {
                        running = std.math.inf(f64);
                    } else if (!std.math.isInf(running)) {
                        running += 1.0 / value;
                    }
                },
                .variance, .stddev, .sem, .cv, .fano, .skewness, .kurtosis => {
                    const previous_count = running_count;
                    running_count += 1;
                    const n: f64 = @floatFromInt(running_count);
                    const previous_n: f64 = @floatFromInt(previous_count);
                    const delta = value - running_mean;
                    running_mean += delta / n;
                    const delta_n = delta / n;
                    const delta_n2 = delta_n * delta_n;
                    const term1 = delta * delta_n * previous_n;
                    const previous_m2 = running_m2;
                    const previous_m3 = running_m3;
                    running_m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * previous_m2 - 4.0 * delta_n * previous_m3;
                    running_m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * previous_m2;
                    running_m2 += term1;
                    const denominator = n - correction;
                    const variance = if (denominator <= 0.0) std.math.nan(f64) else running_m2 / denominator;
                    const stddev_value = std.math.sqrt(variance);
                    running = switch (reduction) {
                        .variance => variance,
                        .stddev => stddev_value,
                        .sem => stddev_value / std.math.sqrt(n),
                        .cv => if (running_mean == 0.0) std.math.nan(f64) else stddev_value / running_mean,
                        .fano => if (running_mean == 0.0) std.math.nan(f64) else variance / running_mean,
                        .skewness => if (running_count < 2 or running_m2 == 0.0) std.math.nan(f64) else std.math.sqrt(n) * running_m3 / std.math.pow(f64, running_m2, 1.5),
                        .kurtosis => if (running_count < 2 or running_m2 == 0.0) std.math.nan(f64) else n * running_m4 / (running_m2 * running_m2) - 3.0,
                        else => unreachable,
                    };
                },
                .rms => {
                    running += value * value;
                    running_count += 1;
                },
                .mean_abs => {
                    running += @abs(value);
                    running_count += 1;
                },
                .mean_square => {
                    running += value * value;
                    running_count += 1;
                },
                .l1_norm => running += @abs(value),
                .l2_norm => running += value * value,
                .max_abs => {
                    const magnitude = @abs(value);
                    if (!running_valid or std.math.isNan(magnitude) or (!std.math.isNan(running) and magnitude > running)) {
                        running = magnitude;
                    }
                    running_valid = true;
                },
                .min_abs => {
                    const magnitude = @abs(value);
                    if (!running_valid or std.math.isNan(magnitude) or (!std.math.isNan(running) and magnitude < running)) {
                        running = magnitude;
                    }
                    running_valid = true;
                },
                .max => {
                    if (!running_valid or std.math.isNan(value) or (!std.math.isNan(running) and value > running)) {
                        running = value;
                    }
                    running_valid = true;
                },
                .min => {
                    if (!running_valid or std.math.isNan(value) or (!std.math.isNan(running) and value < running)) {
                        running = value;
                    }
                    running_valid = true;
                },
                .range => {
                    if (!running_valid or std.math.isNan(value)) {
                        running_min = value;
                        running_max = value;
                    } else if (!std.math.isNan(running_min)) {
                        if (value < running_min) running_min = value;
                        if (value > running_max) running_max = value;
                    }
                    running_valid = true;
                    running = running_max - running_min;
                },
            }
            cumulative[offset] = switch (reduction) {
                .mean => running / @as(f64, @floatFromInt(running_count)),
                .geometric_mean => if (std.math.isNan(running)) std.math.nan(f64) else if (geometric_zero) 0.0 else std.math.exp(running / @as(f64, @floatFromInt(running_count))),
                .harmonic_mean => if (std.math.isInf(running)) 0.0 else @as(f64, @floatFromInt(running_count)) / running,
                .rms => std.math.sqrt(running / @as(f64, @floatFromInt(running_count))),
                .mean_abs, .mean_square => running / @as(f64, @floatFromInt(running_count)),
                .l2_norm => std.math.sqrt(running),
                else => running,
            };
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        var values = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(values);
        const value_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(value_validity);
        @memset(values, 0.0);
        @memset(value_validity, false);

        for (0..input.rows) |row| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            values[row] = cumulative[offset];
            value_validity[row] = true;
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, value_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .sum);
}

pub fn withRowCumsum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSum(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSum(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixSum(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSum(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .mean);
}

pub fn withRowCummean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAverage(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumAverage(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumAvg(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAverage(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAvg(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLogSumExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .logsumexp);
}

pub fn withRowCumulativeLogsumexp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogSumExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumLogSumExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogSumExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumLogsumexp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogSumExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixLogSumExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogSumExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixLogsumexp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogSumExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLogMeanExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .logmeanexp);
}

pub fn withRowCumulativeLogmeanexp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogMeanExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumLogMeanExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogMeanExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumLogmeanexp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogMeanExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixLogMeanExp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogMeanExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixLogmeanexp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLogMeanExp(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeGeometricMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .geometric_mean);
}

pub fn withRowCumulativeGeoMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeGeometricMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumGeometricMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeGeometricMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumGeoMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeGeometricMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixGeometricMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeGeometricMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixGeoMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeGeometricMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeHarmonicMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .harmonic_mean);
}

pub fn withRowCumulativeHarmMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeHarmonicMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumHarmonicMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeHarmonicMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumHarmMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeHarmonicMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixHarmonicMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeHarmonicMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixHarmMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeHarmonicMean(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeVariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, correction, .variance);
}

pub fn withRowCumulativeVar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeVariance(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumVariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeVariance(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumVar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeVariance(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowPrefixVariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeVariance(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowPrefixVar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeVariance(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumulativeStddev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, correction, .stddev);
}

pub fn withRowCumulativeStd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeStddev(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumStddev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeStddev(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumStd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeStddev(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowPrefixStddev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeStddev(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowPrefixStd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeStddev(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumulativeSem(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, correction, .sem);
}

pub fn withRowCumSem(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSem(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowPrefixSem(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSem(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumulativeCv(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, correction, .cv);
}

pub fn withRowCumCv(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeCv(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowPrefixCv(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeCv(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumulativeFano(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, correction, .fano);
}

pub fn withRowCumFano(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFano(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowPrefixFano(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFano(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumulativeIndexOfDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFano(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumIndexOfDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFano(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowPrefixIndexOfDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFano(DeviceDataFrame, input, names, output_names, correction);
}

pub fn withRowCumulativeSkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .skewness);
}

pub fn withRowCumulativeSkew(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSkewness(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumSkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSkewness(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumSkew(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSkewness(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixSkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSkewness(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixSkew(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSkewness(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeKurtosis(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .kurtosis);
}

pub fn withRowCumulativeKurt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeKurtosis(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumKurtosis(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeKurtosis(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumKurt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeKurtosis(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixKurtosis(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeKurtosis(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixKurt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeKurtosis(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeRms(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .rms);
}

pub fn withRowCumRms(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRms(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixRms(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRms(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeMeanAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .mean_abs);
}

pub fn withRowCumulativeMeanAbsolute(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMeanAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMeanAbsolute(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMeanAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMeanAbsolute(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeMeanSquare(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .mean_square);
}

pub fn withRowCumulativeMeanSquared(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanSquare(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMeanSquare(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanSquare(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMeanSquared(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanSquare(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMeanSquare(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanSquare(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMeanSquared(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMeanSquare(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeMaxAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .max_abs);
}

pub fn withRowCumulativeMaxAbsolute(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLInfNorm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLinfNorm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMaxAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMaxAbsolute(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumLInfNorm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumLinfNorm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMaxAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMaxAbsolute(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixLInfNorm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixLinfNorm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMaxAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeMinAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .min_abs);
}

pub fn withRowCumulativeMinAbsolute(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMinAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMinAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMinAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMinAbsolute(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMinAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMinAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMinAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMinAbsolute(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMinAbs(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeL1Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .l1_norm);
}

pub fn withRowCumL1Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeL1Norm(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixL1Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeL1Norm(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeL2Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .l2_norm);
}

pub fn withRowCumL2Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeL2Norm(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixL2Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeL2Norm(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeProduct(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .product);
}

pub fn withRowCumprod(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeProduct(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumProd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeProduct(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixProduct(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeProduct(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .max);
}

pub fn withRowCummax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMax(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMax(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMax(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .min);
}

pub fn withRowCummin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMin(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMin(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMin(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRealColumns(DeviceDataFrame, input, names, output_names, 0.0, .range);
}

pub fn withRowCumRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRange(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRange(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativePtp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRange(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumPtp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRange(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixPtp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeRange(DeviceDataFrame, input, names, output_names);
}

const RowTukeyFences = struct {
    lower: []f64,
    upper: []f64,
    validity: []bool,
};

fn freeRowTukeyFences(allocator: std.mem.Allocator, fences: RowTukeyFences) void {
    allocator.free(fences.lower);
    allocator.free(fences.upper);
    allocator.free(fences.validity);
}

fn computeRowTukeyFences(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    check_names: []const []const u8,
) DeviceFrameArrayError!RowTukeyFences {
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const lower_fences = try input.allocator.alloc(f64, input.rows);
    errdefer input.allocator.free(lower_fences);
    const upper_fences = try input.allocator.alloc(f64, input.rows);
    errdefer input.allocator.free(upper_fences);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    errdefer input.allocator.free(row_validity);
    @memset(lower_fences, 0.0);
    @memset(upper_fences, 0.0);
    @memset(row_validity, false);

    const scratch = try input.allocator.alloc(f64, check_names.len);
    defer input.allocator.free(scratch);
    const tukey_multiplier = 1.5;
    for (0..input.rows) |row| {
        var count: usize = 0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            scratch[count] = flat_values[offset];
            count += 1;
        }
        if (count == 0) continue;

        std.sort.insertion(f64, scratch[0..count], {}, rowQuantileLess);
        const q1 = rowQuantileFromSorted(scratch[0..count], 0.25);
        const q3 = rowQuantileFromSorted(scratch[0..count], 0.75);
        const iqr = q3 - q1;
        lower_fences[row] = q1 - tukey_multiplier * iqr;
        upper_fences[row] = q3 + tukey_multiplier * iqr;
        row_validity[row] = true;
    }

    return .{ .lower = lower_fences, .upper = upper_fences, .validity = row_validity };
}

pub fn withRowIqrOutlier(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const fences = try computeRowTukeyFences(DeviceDataFrame, input, check_names);
    defer freeRowTukeyFences(input.allocator, fences);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var flags = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(flags);
        const flag_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(flag_validity);
        @memset(flags, false);
        @memset(flag_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !fences.validity[row]) continue;
                    flag_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flags[row] = value < fences.lower[row] or value > fences.upper[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(bool, input.allocator, flags, flag_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowIqrOutliers(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowIqrOutlier(DeviceDataFrame, input, names, output_names);
}

pub fn withRowTukeyOutlier(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowIqrOutlier(DeviceDataFrame, input, names, output_names);
}

pub fn withRowTukeyOutliers(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowIqrOutlier(DeviceDataFrame, input, names, output_names);
}

pub fn withRowTukeyWinsorize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const fences = try computeRowTukeyFences(DeviceDataFrame, input, check_names);
    defer freeRowTukeyFences(input.allocator, fences);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var winsorized = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(winsorized);
        const winsorized_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(winsorized_validity);
        @memset(winsorized, 0.0);
        @memset(winsorized_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !fences.validity[row]) continue;
                    winsorized_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    winsorized[row] = @min(@max(value, fences.lower[row]), fences.upper[row]);
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, winsorized, winsorized_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowTukeyWinsorized(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowTukeyWinsorize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowIqrWinsorize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowTukeyWinsorize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowIqrWinsorized(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowTukeyWinsorize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMaxIndicator(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const maxima = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(maxima);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(maxima, 0.0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    if (!row_validity[row] or std.math.isNan(value) or (!std.math.isNan(maxima[row]) and value > maxima[row])) {
                        maxima[row] = value;
                    }
                    row_validity[row] = true;
                }
            },
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var indicators = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(indicators);
        const indicator_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(indicator_validity);
        @memset(indicators, false);
        @memset(indicator_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    indicator_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    indicators[row] = (std.math.isNan(value) and std.math.isNan(maxima[row])) or value == maxima[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(bool, input.allocator, indicators, indicator_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowMaxIndicators(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMaxIndicator(DeviceDataFrame, input, names, output_names);
}

pub fn withRowIsMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMaxIndicator(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMaxMask(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMaxIndicator(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMinIndicator(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const minima = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(minima);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(minima, 0.0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    if (!row_validity[row] or std.math.isNan(value) or (!std.math.isNan(minima[row]) and value < minima[row])) {
                        minima[row] = value;
                    }
                    row_validity[row] = true;
                }
            },
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var indicators = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(indicators);
        const indicator_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(indicator_validity);
        @memset(indicators, false);
        @memset(indicator_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    indicator_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    indicators[row] = (std.math.isNan(value) and std.math.isNan(minima[row])) or value == minima[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(bool, input.allocator, indicators, indicator_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowMinIndicators(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMinIndicator(DeviceDataFrame, input, names, output_names);
}

pub fn withRowIsMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMinIndicator(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMinMask(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMinIndicator(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMinMaxScale(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const minima = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(minima);
    const maxima = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(maxima);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(minima, 0.0);
    @memset(maxima, 0.0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    if (!row_validity[row]) {
                        minima[row] = value;
                        maxima[row] = value;
                        row_validity[row] = true;
                    } else if (std.math.isNan(value)) {
                        minima[row] = value;
                        maxima[row] = value;
                    } else if (!std.math.isNan(minima[row])) {
                        if (value < minima[row]) minima[row] = value;
                        if (value > maxima[row]) maxima[row] = value;
                    }
                }
            },
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var scaled = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(scaled);
        const scaled_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(scaled_validity);
        @memset(scaled, 0.0);
        @memset(scaled_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    scaled_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    const range = maxima[row] - minima[row];
                    scaled[row] = if (range == 0.0 or std.math.isNan(range)) std.math.nan(f64) else (value - minima[row]) / range;
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, scaled, scaled_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowMinmaxScale(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMinMaxScale(DeviceDataFrame, input, names, output_names);
}

pub fn withRowL2Normalize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const norms = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(norms);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(norms, 0.0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    norms[row] += value * value;
                    row_validity[row] = true;
                }
            },
        }
    }
    for (norms, row_validity) |*norm, valid| {
        if (valid) norm.* = std.math.sqrt(norm.*);
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var normalized = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(normalized);
        const normalized_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(normalized_validity);
        @memset(normalized, 0.0);
        @memset(normalized_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    normalized_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    normalized[row] = if (norms[row] == 0.0) std.math.nan(f64) else value / norms[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, normalized, normalized_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowL2Normalized(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowL2Normalize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowL1Normalize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const norms = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(norms);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(norms, 0.0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    norms[row] += @abs(realValueAsF64(@TypeOf(raw_value), raw_value));
                    row_validity[row] = true;
                }
            },
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var normalized = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(normalized);
        const normalized_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(normalized_validity);
        @memset(normalized, 0.0);
        @memset(normalized_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    normalized_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    normalized[row] = if (norms[row] == 0.0) std.math.nan(f64) else value / norms[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, normalized, normalized_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowL1Normalized(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowL1Normalize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowSumNormalize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const sums = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(sums);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(sums, 0.0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    sums[row] += realValueAsF64(@TypeOf(raw_value), raw_value);
                    row_validity[row] = true;
                }
            },
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var shares = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(shares);
        const share_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(share_validity);
        @memset(shares, 0.0);
        @memset(share_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    share_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    shares[row] = if (sums[row] == 0.0) std.math.nan(f64) else value / sums[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, shares, share_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowProportion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSumNormalize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowShare(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSumNormalize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMeanNormalize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const means = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(means);
    const counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(counts);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(means, 0.0);
    @memset(counts, 0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    means[row] += realValueAsF64(@TypeOf(raw_value), raw_value);
                    counts[row] += 1;
                    row_validity[row] = true;
                }
            },
        }
    }
    for (means, counts, row_validity) |*mean, count, valid| {
        if (valid) mean.* /= @as(f64, @floatFromInt(count));
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var ratios = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(ratios);
        const ratio_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(ratio_validity);
        @memset(ratios, 0.0);
        @memset(ratio_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    ratio_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    ratios[row] = if (means[row] == 0.0) std.math.nan(f64) else value / means[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, ratios, ratio_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowMeanNormalized(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMeanNormalize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMeanRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMeanNormalize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowMaxAbsNormalize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const scales = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(scales);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(scales, 0.0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const magnitude = @abs(realValueAsF64(@TypeOf(raw_value), raw_value));
                    row_validity[row] = true;
                    if (std.math.isNan(magnitude)) {
                        scales[row] = std.math.nan(f64);
                    } else if (!std.math.isNan(scales[row]) and magnitude > scales[row]) {
                        scales[row] = magnitude;
                    }
                }
            },
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var normalized = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(normalized);
        const normalized_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(normalized_validity);
        @memset(normalized, 0.0);
        @memset(normalized_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row]) continue;
                    normalized_validity[row] = true;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    normalized[row] = if (scales[row] == 0.0 or std.math.isNan(scales[row])) std.math.nan(f64) else value / scales[row];
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, normalized, normalized_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowMaxabsNormalize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMaxAbsNormalize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowLInfNormalize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMaxAbsNormalize(DeviceDataFrame, input, names, output_names);
}

pub fn withRowLinfNormalize(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMaxAbsNormalize(DeviceDataFrame, input, names, output_names);
}

const RowSoftmaxOutput = enum { probability, log_probability };
const RowSoftmaxDirection = enum { max, min };

fn withRowSoftmaxColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime output: RowSoftmaxOutput,
    comptime direction: RowSoftmaxDirection,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const maxima = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(maxima);
    const secondaries = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(secondaries);
    const denom = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(denom);
    const valid_counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(valid_counts);
    const pos_inf_counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(pos_inf_counts);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(maxima, -std.math.inf(f64));
    @memset(secondaries, -std.math.inf(f64));
    @memset(denom, 0.0);
    @memset(valid_counts, 0);
    @memset(pos_inf_counts, 0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    const score = if (direction == .min) -value else value;
                    row_validity[row] = true;
                    valid_counts[row] += 1;
                    if (std.math.isNan(score)) {
                        maxima[row] = std.math.nan(f64);
                    } else if (!std.math.isNan(maxima[row]) and score > maxima[row]) {
                        maxima[row] = score;
                    }
                }
            },
        }
    }

    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row] or std.math.isNan(maxima[row])) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    const score = if (direction == .min) -value else value;
                    if (std.math.isPositiveInf(maxima[row])) {
                        if (std.math.isPositiveInf(score)) pos_inf_counts[row] += 1;
                    } else if (!std.math.isNegativeInf(maxima[row])) {
                        denom[row] += std.math.exp(score - maxima[row]);
                    }
                }
            },
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        var probabilities = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(probabilities);
        const probability_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(probability_validity);
        @memset(probabilities, 0.0);
        @memset(probability_validity, false);

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    probability_validity[row] = row_validity[row];
                    if (!row_validity[row]) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    const score = if (direction == .min) -value else value;
                    if (std.math.isNan(maxima[row]) or std.math.isNan(score)) {
                        probabilities[row] = std.math.nan(f64);
                    } else if (std.math.isPositiveInf(maxima[row])) {
                        const probability = if (std.math.isPositiveInf(score)) 1.0 / @as(f64, @floatFromInt(pos_inf_counts[row])) else 0.0;
                        probabilities[row] = if (output == .log_probability) std.math.log(f64, std.math.e, probability) else probability;
                    } else if (std.math.isNegativeInf(maxima[row])) {
                        const probability = 1.0 / @as(f64, @floatFromInt(valid_counts[row]));
                        probabilities[row] = if (output == .log_probability) std.math.log(f64, std.math.e, probability) else probability;
                    } else if (output == .log_probability) {
                        probabilities[row] = score - maxima[row] - std.math.log(f64, std.math.e, denom[row]);
                    } else {
                        probabilities[row] = std.math.exp(score - maxima[row]) / denom[row];
                    }
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, probabilities, probability_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowSoftmax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxColumns(DeviceDataFrame, input, names, output_names, .probability, .max);
}

pub fn withRowLogSoftmax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxColumns(DeviceDataFrame, input, names, output_names, .log_probability, .max);
}

pub fn withRowLogsoftmax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowLogSoftmax(DeviceDataFrame, input, names, output_names);
}

pub fn withRowSoftmin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxColumns(DeviceDataFrame, input, names, output_names, .probability, .min);
}

pub fn withRowLogSoftmin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxColumns(DeviceDataFrame, input, names, output_names, .log_probability, .min);
}

pub fn withRowLogsoftmin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowLogSoftmin(DeviceDataFrame, input, names, output_names);
}

const RowSoftmaxSummary = enum { entropy, perplexity, confidence, margin, evenness, concentration, normalized_concentration, gini_impurity, inverse_simpson, simpson_evenness, logit_margin };

fn withRowSoftmaxSummary(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime summary: RowSoftmaxSummary,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    const maxima = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(maxima);
    const secondaries = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(secondaries);
    const denom = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(denom);
    const shifted_weighted_sum = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(shifted_weighted_sum);
    const squared_weight_sum = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(squared_weight_sum);
    const valid_counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(valid_counts);
    const pos_inf_counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(pos_inf_counts);
    const row_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(row_validity);
    @memset(maxima, -std.math.inf(f64));
    @memset(secondaries, -std.math.inf(f64));
    @memset(denom, 0.0);
    @memset(shifted_weighted_sum, 0.0);
    @memset(squared_weight_sum, 0.0);
    @memset(valid_counts, 0);
    @memset(pos_inf_counts, 0);
    @memset(row_validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    row_validity[row] = true;
                    valid_counts[row] += 1;
                    if (std.math.isNan(value)) {
                        maxima[row] = std.math.nan(f64);
                        secondaries[row] = std.math.nan(f64);
                    } else if (!std.math.isNan(maxima[row])) {
                        if (value >= maxima[row]) {
                            secondaries[row] = maxima[row];
                            maxima[row] = value;
                        } else if (value > secondaries[row]) {
                            secondaries[row] = value;
                        }
                    }
                }
            },
        }
    }

    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !row_validity[row] or std.math.isNan(maxima[row])) continue;
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    if (std.math.isPositiveInf(maxima[row])) {
                        if (std.math.isPositiveInf(value)) pos_inf_counts[row] += 1;
                    } else if (!std.math.isNegativeInf(maxima[row])) {
                        const shifted = value - maxima[row];
                        const weight = std.math.exp(shifted);
                        denom[row] += weight;
                        shifted_weighted_sum[row] += weight * shifted;
                        squared_weight_sum[row] += weight * weight;
                    }
                }
            },
        }
    }

    const entropies = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(entropies);
    for (entropies, row_validity, maxima, secondaries, denom, shifted_weighted_sum, squared_weight_sum, valid_counts, pos_inf_counts) |*entropy, valid, max_value, second_value, denominator, shifted_sum, squared_sum, valid_count, pos_inf_count| {
        if (!valid) {
            entropy.* = 0.0;
        } else if (std.math.isNan(max_value)) {
            entropy.* = std.math.nan(f64);
        } else if (summary == .concentration or summary == .normalized_concentration or summary == .gini_impurity or summary == .inverse_simpson or summary == .simpson_evenness) {
            const concentration = if (std.math.isPositiveInf(max_value))
                1.0 / @as(f64, @floatFromInt(pos_inf_count))
            else if (std.math.isNegativeInf(max_value))
                1.0 / @as(f64, @floatFromInt(valid_count))
            else
                squared_sum / (denominator * denominator);
            entropy.* = if (summary == .gini_impurity)
                1.0 - concentration
            else if (summary == .normalized_concentration)
                if (valid_count <= 1) 1.0 else (concentration - 1.0 / @as(f64, @floatFromInt(valid_count))) / (1.0 - 1.0 / @as(f64, @floatFromInt(valid_count)))
            else if (summary == .inverse_simpson)
                1.0 / concentration
            else if (summary == .simpson_evenness)
                1.0 / (concentration * @as(f64, @floatFromInt(valid_count)))
            else
                concentration;
        } else if (summary == .logit_margin) {
            entropy.* = if (valid_count <= 1)
                std.math.inf(f64)
            else if (std.math.isPositiveInf(max_value))
                if (std.math.isPositiveInf(second_value)) 0.0 else std.math.inf(f64)
            else if (std.math.isNegativeInf(max_value))
                0.0
            else
                max_value - second_value;
        } else if (summary == .confidence or summary == .margin) {
            const top_probability = if (std.math.isPositiveInf(max_value))
                1.0 / @as(f64, @floatFromInt(pos_inf_count))
            else if (std.math.isNegativeInf(max_value))
                1.0 / @as(f64, @floatFromInt(valid_count))
            else
                1.0 / denominator;
            if (summary == .confidence) {
                entropy.* = top_probability;
            } else if (std.math.isPositiveInf(max_value)) {
                entropy.* = if (pos_inf_count > 1) 0.0 else top_probability;
            } else if (valid_count <= 1) {
                entropy.* = top_probability;
            } else if (std.math.isNegativeInf(max_value)) {
                entropy.* = 0.0;
            } else {
                entropy.* = top_probability - std.math.exp(second_value - max_value) / denominator;
            }
        } else if (std.math.isPositiveInf(max_value)) {
            entropy.* = std.math.log(f64, std.math.e, @as(f64, @floatFromInt(pos_inf_count)));
        } else if (std.math.isNegativeInf(max_value)) {
            entropy.* = std.math.log(f64, std.math.e, @as(f64, @floatFromInt(valid_count)));
        } else {
            entropy.* = std.math.log(f64, std.math.e, denominator) - shifted_sum / denominator;
        }
        if (!std.math.isNan(entropy.*)) {
            if (summary == .perplexity) {
                entropy.* = std.math.exp(entropy.*);
            } else if (summary == .evenness) {
                entropy.* = if (valid_count <= 1) 1.0 else entropy.* / std.math.log(f64, std.math.e, @as(f64, @floatFromInt(valid_count)));
            }
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, entropies, row_validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowSoftmaxEntropy(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .entropy);
}

pub fn withRowSoftmaxPerplexity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .perplexity);
}

pub fn withRowSoftmaxConfidence(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .confidence);
}

pub fn withRowSoftmaxMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .margin);
}

pub fn withRowSoftmaxEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .evenness);
}

pub fn withRowSoftmaxNormalizedEntropy(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxEvenness(DeviceDataFrame, input, names, output_name);
}

pub fn withRowSoftmaxConcentration(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .concentration);
}

pub fn withRowSoftmaxNormalizedHhi(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .normalized_concentration);
}

pub fn withRowSoftmaxNormalizedHHI(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxNormalizedHhi(DeviceDataFrame, input, names, output_name);
}

pub fn withRowSoftmaxNhhi(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxNormalizedHhi(DeviceDataFrame, input, names, output_name);
}

pub fn withRowSoftmaxGiniImpurity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .gini_impurity);
}

pub fn withRowSoftmaxGini(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxGiniImpurity(DeviceDataFrame, input, names, output_name);
}

pub fn withRowSoftmaxInverseSimpson(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .inverse_simpson);
}

pub fn withRowSoftmaxSimpsonEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .simpson_evenness);
}

pub fn withRowSoftmaxSimpsonEven(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSimpsonEvenness(DeviceDataFrame, input, names, output_name);
}

pub fn withRowLogitMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSoftmaxSummary(DeviceDataFrame, input, names, output_name, .logit_margin);
}

pub fn withRowGeometricMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .geometric_mean);
}

pub fn withRowGeoMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowGeometricMean(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeGeometricMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_geometric_mean);
}

pub fn withRowAbsGeometricMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeGeometricMean(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeGeoMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeGeometricMean(DeviceDataFrame, input, names, output_name);
}

pub fn withRowAbsGeoMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeGeometricMean(DeviceDataFrame, input, names, output_name);
}

pub fn withRowHarmonicMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .harmonic_mean);
}

pub fn withRowHarmMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowHarmonicMean(DeviceDataFrame, input, names, output_name);
}

pub fn withRowProd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .prod);
}

pub fn withRowMin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .min);
}

pub fn withRowMax(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .max);
}

pub fn withRowPtp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .ptp);
}

pub fn withRowMagnitudePtp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_ptp);
}

pub fn withRowAbsPtp(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudePtp(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudePeakToPeak(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudePtp(DeviceDataFrame, input, names, output_name);
}

pub fn withRowAbsPeakToPeak(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudePtp(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMidrange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .midrange);
}

pub fn withRowMagnitudeMidrange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_midrange);
}

pub fn withRowAbsMidrange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeMidrange(DeviceDataFrame, input, names, output_name);
}

pub fn withRowRangeCoeff(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .range_coeff);
}

pub fn withRowRangeCoefficient(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowRangeCoeff(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeRangeCoeff(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_range_coeff);
}

pub fn withRowAbsRangeCoeff(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeRangeCoeff(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeRangeCoefficient(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeRangeCoeff(DeviceDataFrame, input, names, output_name);
}

pub fn withRowAbsRangeCoefficient(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeRangeCoeff(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMeanAbs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .mean_abs);
}

pub fn withRowHhi(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .hhi);
}

pub fn withRowHerfindahl(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowHhi(DeviceDataFrame, input, names, output_name);
}

pub fn withRowHerfindahlHirschman(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowHhi(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeNormalizedHhi(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_normalized_hhi);
}

pub fn withRowAbsNormalizedHhi(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeNormalizedHhi(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeSparsity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_sparsity);
}

pub fn withRowAbsSparsity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeSparsity(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeInverseSimpson(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_inverse_simpson);
}

pub fn withRowAbsInverseSimpson(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeInverseSimpson(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeSimpsonEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_simpson_evenness);
}

pub fn withRowAbsSimpsonEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeSimpsonEvenness(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeDominance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_dominance);
}

pub fn withRowAbsDominance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeDominance(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeDominanceMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_dominance_margin);
}

pub fn withRowAbsDominanceMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeDominanceMargin(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeEntropy(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_entropy);
}

pub fn withRowAbsEntropy(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeEntropy(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudePerplexity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_perplexity);
}

pub fn withRowAbsPerplexity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudePerplexity(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .magnitude_evenness);
}

pub fn withRowAbsEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeEvenness(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMeanAbsDev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..input.rows) |row| {
        var count: usize = 0;
        var mean: f64 = 0.0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            mean += flat_values[offset];
            count += 1;
        }
        if (count == 0) continue;
        mean /= @as(f64, @floatFromInt(count));

        var deviation_sum: f64 = 0.0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            deviation_sum += @abs(flat_values[offset] - mean);
        }
        values[row] = deviation_sum / @as(f64, @floatFromInt(count));
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowGiniMeanDiff(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..input.rows) |row| {
        var pair_sum: f64 = 0.0;
        var pair_count: usize = 0;
        var valid_count: usize = 0;
        for (0..check_names.len) |lhs_index| {
            const lhs_offset = row * check_names.len + lhs_index;
            if (!flat_validity[lhs_offset]) continue;
            valid_count += 1;
            for (lhs_index + 1..check_names.len) |rhs_index| {
                const rhs_offset = row * check_names.len + rhs_index;
                if (!flat_validity[rhs_offset]) continue;
                pair_sum += @abs(flat_values[lhs_offset] - flat_values[rhs_offset]);
                pair_count += 1;
            }
        }
        if (valid_count == 0) continue;
        values[row] = if (pair_count == 0) 0.0 else pair_sum / @as(f64, @floatFromInt(pair_count));
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowGiniCoefficient(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..input.rows) |row| {
        var pair_sum: f64 = 0.0;
        var pair_count: usize = 0;
        var valid_count: usize = 0;
        var mean: f64 = 0.0;
        for (0..check_names.len) |lhs_index| {
            const lhs_offset = row * check_names.len + lhs_index;
            if (!flat_validity[lhs_offset]) continue;
            valid_count += 1;
            mean += flat_values[lhs_offset];
            for (lhs_index + 1..check_names.len) |rhs_index| {
                const rhs_offset = row * check_names.len + rhs_index;
                if (!flat_validity[rhs_offset]) continue;
                pair_sum += @abs(flat_values[lhs_offset] - flat_values[rhs_offset]);
                pair_count += 1;
            }
        }
        if (valid_count == 0) continue;
        mean /= @as(f64, @floatFromInt(valid_count));
        const gmd = if (pair_count == 0) 0.0 else pair_sum / @as(f64, @floatFromInt(pair_count));
        values[row] = if (mean == 0.0) std.math.nan(f64) else gmd / (2.0 * @abs(mean));
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowGiniCoeff(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowGiniCoefficient(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMeanAbsDevRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..input.rows) |row| {
        var count: usize = 0;
        var mean: f64 = 0.0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            mean += flat_values[offset];
            count += 1;
        }
        if (count == 0) continue;
        mean /= @as(f64, @floatFromInt(count));

        var deviation_sum: f64 = 0.0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            deviation_sum += @abs(flat_values[offset] - mean);
        }
        const mad = deviation_sum / @as(f64, @floatFromInt(count));
        values[row] = if (mean == 0.0) std.math.nan(f64) else mad / @abs(mean);
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowRms(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .rms);
}

pub fn withRowL1Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .l1_norm);
}

pub fn withRowL2Norm(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .l2_norm);
}

fn rowQuantileLess(_: void, lhs: f64, rhs: f64) bool {
    const lhs_nan = std.math.isNan(lhs);
    const rhs_nan = std.math.isNan(rhs);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (lhs_nan and rhs_nan) return false;
    return lhs < rhs;
}

fn rowDenseRankValueEqual(lhs: f64, rhs: f64) bool {
    return (std.math.isNan(lhs) and std.math.isNan(rhs)) or lhs == rhs;
}

fn rowQuantileFromSorted(sorted_values: []const f64, q: f64) f64 {
    const max_index = sorted_values.len - 1;
    const position = q * @as(f64, @floatFromInt(max_index));
    const lower_float = @floor(position);
    const lower: usize = @intFromFloat(lower_float);
    const upper = @min(lower + 1, max_index);
    const weight = position - lower_float;
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
}

const RowQuantileOutput = struct {
    values: []f64,
    validity: []bool,
};

const RowQuantileMeasure = union(enum) {
    quantile: f64,
    difference: struct { hi: f64, lo: f64 },
    trimmed_mean: f64,
    winsorized_mean: f64,
    midhinge,
    trimean,
    bowley_skewness,
    quartile_coeff_dispersion,
    kelley_skewness,
};

fn validateRowQuantile(q: f64) DeviceFrameArrayError!void {
    if (std.math.isNan(q) or q < 0.0 or q > 1.0) return error.InvalidShape;
}

fn withRowQuantileValues(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    check_names: []const []const u8,
    measure: RowQuantileMeasure,
) DeviceFrameArrayError!RowQuantileOutput {
    switch (measure) {
        .quantile => |q| try validateRowQuantile(q),
        .difference => |qs| {
            try validateRowQuantile(qs.hi);
            try validateRowQuantile(qs.lo);
        },
        .trimmed_mean => |trim_fraction| {
            if (std.math.isNan(trim_fraction) or trim_fraction < 0.0 or trim_fraction >= 0.5) return error.InvalidShape;
        },
        .winsorized_mean => |winsor_fraction| {
            if (std.math.isNan(winsor_fraction) or winsor_fraction < 0.0 or winsor_fraction >= 0.5) return error.InvalidShape;
        },
        .midhinge, .trimean, .bowley_skewness, .quartile_coeff_dispersion, .kelley_skewness => {},
    }

    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    errdefer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    errdefer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    const scratch = try input.allocator.alloc(f64, check_names.len);
    defer input.allocator.free(scratch);
    for (0..input.rows) |row| {
        var count: usize = 0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            scratch[count] = flat_values[offset];
            count += 1;
        }
        if (count == 0) continue;

        // Sort each row's valid values once; derived quartile summaries can
        // then reuse the same interpolation and NaN placement semantics as
        // the public scalar quantile operation.
        std.sort.insertion(f64, scratch[0..count], {}, rowQuantileLess);
        values[row] = switch (measure) {
            .quantile => |q| rowQuantileFromSorted(scratch[0..count], q),
            .difference => |qs| rowQuantileFromSorted(scratch[0..count], qs.hi) - rowQuantileFromSorted(scratch[0..count], qs.lo),
            .trimmed_mean => |trim_fraction| blk: {
                const trim_count: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(count)) * trim_fraction));
                const trimmed = scratch[trim_count .. count - trim_count];
                var total: f64 = 0.0;
                for (trimmed) |value| total += value;
                break :blk total / @as(f64, @floatFromInt(trimmed.len));
            },
            .winsorized_mean => |winsor_fraction| blk: {
                const winsor_count: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(count)) * winsor_fraction));
                const lower = scratch[winsor_count];
                const upper = scratch[count - winsor_count - 1];
                var total: f64 = 0.0;
                for (scratch[0..count]) |value| total += @min(@max(value, lower), upper);
                break :blk total / @as(f64, @floatFromInt(count));
            },
            .midhinge => (rowQuantileFromSorted(scratch[0..count], 0.25) + rowQuantileFromSorted(scratch[0..count], 0.75)) / 2.0,
            .trimean => (rowQuantileFromSorted(scratch[0..count], 0.25) + 2.0 * rowQuantileFromSorted(scratch[0..count], 0.5) + rowQuantileFromSorted(scratch[0..count], 0.75)) / 4.0,
            .bowley_skewness => blk: {
                const q1 = rowQuantileFromSorted(scratch[0..count], 0.25);
                const median = rowQuantileFromSorted(scratch[0..count], 0.5);
                const q3 = rowQuantileFromSorted(scratch[0..count], 0.75);
                const iqr = q3 - q1;
                break :blk if (iqr == 0.0) std.math.nan(f64) else (q3 + q1 - 2.0 * median) / iqr;
            },
            .quartile_coeff_dispersion => blk: {
                const q1 = rowQuantileFromSorted(scratch[0..count], 0.25);
                const q3 = rowQuantileFromSorted(scratch[0..count], 0.75);
                const denominator = q3 + q1;
                break :blk if (denominator == 0.0) std.math.nan(f64) else (q3 - q1) / denominator;
            },
            .kelley_skewness => blk: {
                const p10 = rowQuantileFromSorted(scratch[0..count], 0.10);
                const median = rowQuantileFromSorted(scratch[0..count], 0.50);
                const p90 = rowQuantileFromSorted(scratch[0..count], 0.90);
                const spread = p90 - p10;
                break :blk if (spread == 0.0) std.math.nan(f64) else (p90 + p10 - 2.0 * median) / spread;
            },
        };
        validity[row] = true;
    }

    return .{ .values = values, .validity = validity };
}

fn withRowQuantileCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    q: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(q) or q < 0.0 or q > 1.0) return error.InvalidShape;

    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .{ .quantile = q });
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowQuantile(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    q: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowQuantileCore(DeviceDataFrame, input, names, output_name, q);
}

pub fn withRowQuantileRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    low_q: f64,
    high_q: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(low_q) or low_q < 0.0 or low_q > 1.0) return error.InvalidShape;
    if (std.math.isNan(high_q) or high_q < 0.0 or high_q > 1.0) return error.InvalidShape;
    if (high_q < low_q) return error.InvalidShape;

    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .{ .difference = .{ .hi = high_q, .lo = low_q } });
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowTrimmedMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    trim_fraction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(trim_fraction) or trim_fraction < 0.0 or trim_fraction >= 0.5) return error.InvalidShape;

    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .{ .trimmed_mean = trim_fraction });
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowWinsorizedMean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    winsor_fraction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(winsor_fraction) or winsor_fraction < 0.0 or winsor_fraction >= 0.5) return error.InvalidShape;

    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .{ .winsorized_mean = winsor_fraction });
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowMedian(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowQuantileCore(DeviceDataFrame, input, names, output_name, 0.5);
}

fn withRowIqrCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .{ .difference = .{ .hi = 0.75, .lo = 0.25 } });
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowIqr(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowIqrCore(DeviceDataFrame, input, names, output_name);
}

pub fn withRowInterdecileRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .{ .difference = .{ .hi = 0.90, .lo = 0.10 } });
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowIdr(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowInterdecileRange(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMidhinge(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .midhinge);
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowTrimean(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .trimean);
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowBowleySkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .bowley_skewness);
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowBowleySkew(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBowleySkewness(DeviceDataFrame, input, names, output_name);
}

pub fn withRowQuartileCoeffDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .quartile_coeff_dispersion);
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowQcd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowQuartileCoeffDispersion(DeviceDataFrame, input, names, output_name);
}

pub fn withRowKelleySkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const output = try withRowQuantileValues(DeviceDataFrame, input, check_names, .kelley_skewness);
    defer {
        input.allocator.free(output.values);
        input.allocator.free(output.validity);
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, output.values, output.validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowKelleySkew(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowKelleySkewness(DeviceDataFrame, input, names, output_name);
}

fn withRowMadCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    const scratch = try input.allocator.alloc(f64, check_names.len);
    defer input.allocator.free(scratch);
    for (0..input.rows) |row| {
        var count: usize = 0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            scratch[count] = flat_values[offset];
            count += 1;
        }
        if (count == 0) continue;

        std.sort.insertion(f64, scratch[0..count], {}, rowQuantileLess);
        const center = rowQuantileFromSorted(scratch[0..count], 0.5);
        for (scratch[0..count]) |*slot| slot.* = @abs(slot.* - center);
        std.sort.insertion(f64, scratch[0..count], {}, rowQuantileLess);
        values[row] = rowQuantileFromSorted(scratch[0..count], 0.5);
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowMad(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMadCore(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMedianAbsDev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMad(DeviceDataFrame, input, names, output_name);
}

fn rowModeValueEqual(lhs: f64, rhs: f64) bool {
    return (std.math.isNan(lhs) and std.math.isNan(rhs)) or lhs == rhs;
}

const RowDistributionReduction = enum { entropy, gini_impurity, perplexity, inverse_simpson, simpson_concentration, evenness };

fn withRowDistributionReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowDistributionReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..input.rows) |row| {
        var count: usize = 0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (flat_validity[offset]) count += 1;
        }
        if (count == 0) continue;

        var entropy: f64 = 0.0;
        var sum_prob_sq: f64 = 0.0;
        var distinct_count: usize = 0;
        const count_f: f64 = @floatFromInt(count);
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            const candidate = flat_values[offset];

            var seen = false;
            for (0..col_index) |previous_index| {
                const previous_offset = row * check_names.len + previous_index;
                if (!flat_validity[previous_offset]) continue;
                if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;

            var frequency: usize = 0;
            for (col_index..check_names.len) |candidate_index| {
                const candidate_offset = row * check_names.len + candidate_index;
                if (!flat_validity[candidate_offset]) continue;
                if (rowModeValueEqual(candidate, flat_values[candidate_offset])) frequency += 1;
            }
            const probability = @as(f64, @floatFromInt(frequency)) / count_f;
            entropy -= probability * std.math.log(f64, std.math.e, probability);
            sum_prob_sq += probability * probability;
            distinct_count += 1;
        }

        values[row] = switch (reduction) {
            .entropy => entropy,
            .gini_impurity => 1.0 - sum_prob_sq,
            .perplexity => std.math.exp(entropy),
            .inverse_simpson => if (sum_prob_sq == 0.0) quietNanF64() else 1.0 / sum_prob_sq,
            .simpson_concentration => sum_prob_sq,
            .evenness => if (distinct_count <= 1) 1.0 else entropy / std.math.log(f64, std.math.e, @as(f64, @floatFromInt(distinct_count))),
        };
        validity[row] = true;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowEntropy(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowDistributionReduction(DeviceDataFrame, input, names, output_name, .entropy);
}

pub fn withRowGiniImpurity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowDistributionReduction(DeviceDataFrame, input, names, output_name, .gini_impurity);
}

pub fn withRowPerplexity(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowDistributionReduction(DeviceDataFrame, input, names, output_name, .perplexity);
}

pub fn withRowInverseSimpson(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowDistributionReduction(DeviceDataFrame, input, names, output_name, .inverse_simpson);
}

pub fn withRowSimpsonConcentration(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowDistributionReduction(DeviceDataFrame, input, names, output_name, .simpson_concentration);
}

pub fn withRowEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowDistributionReduction(DeviceDataFrame, input, names, output_name, .evenness);
}

const RowModeFrequency = enum { count, ratio };

fn withRowModeFrequency(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowModeFrequency,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(validity, false);

    switch (reduction) {
        .count => {
            const values = try input.allocator.alloc(i64, input.rows);
            defer input.allocator.free(values);
            @memset(values, 0);
            for (0..input.rows) |row| {
                var valid_count: usize = 0;
                var best_count: usize = 0;
                for (0..check_names.len) |col_index| {
                    const offset = row * check_names.len + col_index;
                    if (!flat_validity[offset]) continue;
                    valid_count += 1;
                    const candidate = flat_values[offset];
                    var seen = false;
                    for (0..col_index) |previous_index| {
                        const previous_offset = row * check_names.len + previous_index;
                        if (!flat_validity[previous_offset]) continue;
                        if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                            seen = true;
                            break;
                        }
                    }
                    if (seen) continue;
                    var candidate_count: usize = 0;
                    for (col_index..check_names.len) |candidate_index| {
                        const candidate_offset = row * check_names.len + candidate_index;
                        if (!flat_validity[candidate_offset]) continue;
                        if (rowModeValueEqual(candidate, flat_values[candidate_offset])) candidate_count += 1;
                    }
                    if (candidate_count > best_count) best_count = candidate_count;
                }
                if (valid_count == 0) continue;
                values[row] = @intCast(best_count);
                validity[row] = true;
            }
            const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
            var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, values, validity, input.device);
            defer column.deinit();
            return withColumn(DeviceDataFrame, input, output_name, column);
        },
        .ratio => {
            const values = try input.allocator.alloc(f64, input.rows);
            defer input.allocator.free(values);
            @memset(values, 0.0);
            for (0..input.rows) |row| {
                var valid_count: usize = 0;
                var best_count: usize = 0;
                for (0..check_names.len) |col_index| {
                    const offset = row * check_names.len + col_index;
                    if (!flat_validity[offset]) continue;
                    valid_count += 1;
                    const candidate = flat_values[offset];
                    var seen = false;
                    for (0..col_index) |previous_index| {
                        const previous_offset = row * check_names.len + previous_index;
                        if (!flat_validity[previous_offset]) continue;
                        if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                            seen = true;
                            break;
                        }
                    }
                    if (seen) continue;
                    var candidate_count: usize = 0;
                    for (col_index..check_names.len) |candidate_index| {
                        const candidate_offset = row * check_names.len + candidate_index;
                        if (!flat_validity[candidate_offset]) continue;
                        if (rowModeValueEqual(candidate, flat_values[candidate_offset])) candidate_count += 1;
                    }
                    if (candidate_count > best_count) best_count = candidate_count;
                }
                if (valid_count == 0) continue;
                values[row] = @as(f64, @floatFromInt(best_count)) / @as(f64, @floatFromInt(valid_count));
                validity[row] = true;
            }
            const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
            var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
            defer column.deinit();
            return withColumn(DeviceDataFrame, input, output_name, column);
        },
    }
}

pub fn withRowModeCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowModeFrequency(DeviceDataFrame, input, names, output_name, .count);
}

pub fn withRowModeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowModeFrequency(DeviceDataFrame, input, names, output_name, .ratio);
}

pub fn withRowModeMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0);
    @memset(validity, false);

    for (0..input.rows) |row| {
        var best: usize = 0;
        var second: usize = 0;
        var found = false;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            const candidate = flat_values[offset];
            var seen = false;
            for (0..col_index) |previous_index| {
                const previous_offset = row * check_names.len + previous_index;
                if (!flat_validity[previous_offset]) continue;
                if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;
            var count: usize = 0;
            for (col_index..check_names.len) |candidate_index| {
                const candidate_offset = row * check_names.len + candidate_index;
                if (!flat_validity[candidate_offset]) continue;
                if (rowModeValueEqual(candidate, flat_values[candidate_offset])) count += 1;
            }
            found = true;
            if (count > best) {
                second = best;
                best = count;
            } else if (count > second) {
                second = count;
            }
        }
        if (found) {
            values[row] = @intCast(best - second);
            validity[row] = true;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowModeMarginRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var margins = try withRowModeMargin(DeviceDataFrame, input, names, output_name);
    defer margins.deinit();
    const check_names = if (names.len == 0) input.names else names;
    var valid_counts_df = try withRowValidCount(DeviceDataFrame, input, check_names, "__row_valid_count_tmp");
    defer valid_counts_df.deinit();
    const margin_col = try margins.column(output_name);
    const count_col = try valid_counts_df.column("__row_valid_count_tmp");
    const margin_values = try margin_col.i64.toOwnedSlice(input.allocator);
    defer input.allocator.free(margin_values);
    const count_values = try count_col.i64.toOwnedSlice(input.allocator);
    defer input.allocator.free(count_values);
    const maybe_validity = try validityValues(margin_col.i64, input.allocator);
    defer if (maybe_validity) |mask| input.allocator.free(mask);
    const ratios = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(ratios);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    for (ratios, validity, margin_values, count_values, 0..) |*ratio, *valid, margin, count, row| {
        valid.* = (if (maybe_validity) |mask| mask[row] else true) and count > 0;
        ratio.* = if (valid.*) @as(f64, @floatFromInt(margin)) / @as(f64, @floatFromInt(count)) else 0.0;
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, ratios, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

fn withRowModeCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, 0.0);
    @memset(validity, false);

    for (0..input.rows) |row| {
        var found = false;
        var best_value: f64 = 0.0;
        var best_count: usize = 0;
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            const candidate = flat_values[offset];

            var seen = false;
            for (0..col_index) |previous_index| {
                const previous_offset = row * check_names.len + previous_index;
                if (!flat_validity[previous_offset]) continue;
                if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;

            var count: usize = 0;
            for (col_index..check_names.len) |candidate_index| {
                const candidate_offset = row * check_names.len + candidate_index;
                if (!flat_validity[candidate_offset]) continue;
                if (rowModeValueEqual(candidate, flat_values[candidate_offset])) count += 1;
            }

            // Stable tie-breaking mirrors column `mode`: first distinct valid
            // row-wise value wins when frequencies are equal.
            if (!found or count > best_count) {
                best_value = candidate;
                best_count = count;
                found = true;
            }
        }
        if (found) {
            values[row] = best_value;
            validity[row] = true;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowModeCore(DeviceDataFrame, input, names, output_name);
}

fn withRowCumulativeModeCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const mode_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(mode_values);
    const mode_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(mode_validity);
    @memset(mode_values, 0.0);
    @memset(mode_validity, false);

    for (0..input.rows) |row| {
        for (0..check_names.len) |prefix_end| {
            var found = false;
            var best_value: f64 = 0.0;
            var best_count: usize = 0;
            for (0..prefix_end + 1) |col_index| {
                const offset = row * check_names.len + col_index;
                if (!flat_validity[offset]) continue;
                const candidate = flat_values[offset];

                var seen = false;
                for (0..col_index) |previous_index| {
                    const previous_offset = row * check_names.len + previous_index;
                    if (!flat_validity[previous_offset]) continue;
                    if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;

                var count: usize = 0;
                for (col_index..prefix_end + 1) |candidate_index| {
                    const candidate_offset = row * check_names.len + candidate_index;
                    if (!flat_validity[candidate_offset]) continue;
                    if (rowModeValueEqual(candidate, flat_values[candidate_offset])) count += 1;
                }
                if (!found or count > best_count) {
                    best_value = candidate;
                    best_count = count;
                    found = true;
                }
            }
            if (found) {
                const prefix_offset = row * check_names.len + prefix_end;
                mode_values[prefix_offset] = best_value;
                mode_validity[prefix_offset] = true;
            }
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        var values = try input.allocator.alloc(f64, input.rows);
        defer input.allocator.free(values);
        const validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(validity);
        @memset(values, 0.0);
        @memset(validity, false);

        for (0..input.rows) |row| {
            const offset = row * check_names.len + col_index;
            if (!mode_validity[offset]) continue;
            values[row] = mode_values[offset];
            validity[row] = true;
        }

        var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeCore(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMode(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeMode(DeviceDataFrame, input, names, output_names);
}

fn withRowCumulativeModeFrequency(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime output: RowModeFrequency,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const counts = try input.allocator.alloc(i64, total_slots);
    defer input.allocator.free(counts);
    const ratios = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(ratios);
    const out_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(out_validity);
    @memset(counts, 0);
    @memset(ratios, 0.0);
    @memset(out_validity, false);

    for (0..input.rows) |row| {
        for (0..check_names.len) |prefix_end| {
            var found = false;
            var best_count: usize = 0;
            var valid_count: usize = 0;
            for (0..prefix_end + 1) |col_index| {
                const offset = row * check_names.len + col_index;
                if (!flat_validity[offset]) continue;
                valid_count += 1;
                const candidate = flat_values[offset];

                var seen = false;
                for (0..col_index) |previous_index| {
                    const previous_offset = row * check_names.len + previous_index;
                    if (!flat_validity[previous_offset]) continue;
                    if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;

                var count: usize = 0;
                for (col_index..prefix_end + 1) |candidate_index| {
                    const candidate_offset = row * check_names.len + candidate_index;
                    if (!flat_validity[candidate_offset]) continue;
                    if (rowModeValueEqual(candidate, flat_values[candidate_offset])) count += 1;
                }
                if (!found or count > best_count) {
                    best_count = count;
                    found = true;
                }
            }
            if (found) {
                const prefix_offset = row * check_names.len + prefix_end;
                counts[prefix_offset] = @intCast(best_count);
                ratios[prefix_offset] = @as(f64, @floatFromInt(best_count)) / @as(f64, @floatFromInt(valid_count));
                out_validity[prefix_offset] = true;
            }
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        const validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(validity);
        @memset(validity, false);

        switch (output) {
            .count => {
                var values = try input.allocator.alloc(i64, input.rows);
                defer input.allocator.free(values);
                @memset(values, 0);
                for (0..input.rows) |row| {
                    const offset = row * check_names.len + col_index;
                    if (!out_validity[offset]) continue;
                    values[row] = counts[offset];
                    validity[row] = true;
                }
                var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, values, validity, input.device);
                defer column.deinit();
                const next = try withColumn(DeviceDataFrame, result, output_name, column);
                result.deinit();
                result = next;
            },
            .ratio => {
                var values = try input.allocator.alloc(f64, input.rows);
                defer input.allocator.free(values);
                @memset(values, 0.0);
                for (0..input.rows) |row| {
                    const offset = row * check_names.len + col_index;
                    if (!out_validity[offset]) continue;
                    values[row] = ratios[offset];
                    validity[row] = true;
                }
                var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
                defer column.deinit();
                const next = try withColumn(DeviceDataFrame, result, output_name, column);
                result.deinit();
                result = next;
            },
        }
    }
    return result;
}

pub fn withRowCumulativeModeCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeFrequency(DeviceDataFrame, input, names, output_names, .count);
}

pub fn withRowCumModeCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixModeCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeModeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeFrequency(DeviceDataFrame, input, names, output_names, .ratio);
}

pub fn withRowCumModeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixModeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeRatio(DeviceDataFrame, input, names, output_names);
}

fn withRowCumulativeModeMarginCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime ratio: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const margins = try input.allocator.alloc(i64, total_slots);
    defer input.allocator.free(margins);
    const margin_ratios = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(margin_ratios);
    const out_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(out_validity);
    @memset(margins, 0);
    @memset(margin_ratios, 0.0);
    @memset(out_validity, false);

    for (0..input.rows) |row| {
        for (0..check_names.len) |prefix_end| {
            var best: usize = 0;
            var second: usize = 0;
            var valid_count: usize = 0;
            var found = false;
            for (0..prefix_end + 1) |col_index| {
                const offset = row * check_names.len + col_index;
                if (!flat_validity[offset]) continue;
                valid_count += 1;
                const candidate = flat_values[offset];
                var seen = false;
                for (0..col_index) |previous_index| {
                    const previous_offset = row * check_names.len + previous_index;
                    if (!flat_validity[previous_offset]) continue;
                    if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;
                var count: usize = 0;
                for (col_index..prefix_end + 1) |candidate_index| {
                    const candidate_offset = row * check_names.len + candidate_index;
                    if (!flat_validity[candidate_offset]) continue;
                    if (rowModeValueEqual(candidate, flat_values[candidate_offset])) count += 1;
                }
                found = true;
                if (count > best) {
                    second = best;
                    best = count;
                } else if (count > second) {
                    second = count;
                }
            }
            if (found) {
                const prefix_offset = row * check_names.len + prefix_end;
                const margin = best - second;
                margins[prefix_offset] = @intCast(margin);
                margin_ratios[prefix_offset] = @as(f64, @floatFromInt(margin)) / @as(f64, @floatFromInt(valid_count));
                out_validity[prefix_offset] = true;
            }
        }
    }

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        const validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(validity);
        @memset(validity, false);
        if (ratio) {
            var values = try input.allocator.alloc(f64, input.rows);
            defer input.allocator.free(values);
            @memset(values, 0.0);
            for (0..input.rows) |row| {
                const offset = row * check_names.len + col_index;
                if (!out_validity[offset]) continue;
                values[row] = margin_ratios[offset];
                validity[row] = true;
            }
            var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
            defer column.deinit();
            const next = try withColumn(DeviceDataFrame, result, output_name, column);
            result.deinit();
            result = next;
        } else {
            var values = try input.allocator.alloc(i64, input.rows);
            defer input.allocator.free(values);
            @memset(values, 0);
            for (0..input.rows) |row| {
                const offset = row * check_names.len + col_index;
                if (!out_validity[offset]) continue;
                values[row] = margins[offset];
                validity[row] = true;
            }
            var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, values, validity, input.device);
            defer column.deinit();
            const next = try withColumn(DeviceDataFrame, result, output_name, column);
            result.deinit();
            result = next;
        }
    }
    return result;
}

pub fn withRowCumulativeModeMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeMarginCore(DeviceDataFrame, input, names, output_names, false);
}

pub fn withRowCumModeMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeMargin(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixModeMargin(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeMargin(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeModeMarginRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeMarginCore(DeviceDataFrame, input, names, output_names, true);
}

pub fn withRowCumModeMarginRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeMarginRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixModeMarginRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeModeMarginRatio(DeviceDataFrame, input, names, output_names);
}

fn withRowDistinctCountCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    for (0..input.rows) |row| {
        for (0..check_names.len) |col_index| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            const candidate = flat_values[offset];

            var seen = false;
            for (0..col_index) |previous_index| {
                const previous_offset = row * check_names.len + previous_index;
                if (!flat_validity[previous_offset]) continue;
                if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                    seen = true;
                    break;
                }
            }
            if (!seen) counts[row] += 1;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(i64, input.allocator, counts, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowCountDistinct(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowDistinctCountCore(DeviceDataFrame, input, names, output_name);
}

pub fn withRowNUnique(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowDistinctCountCore(DeviceDataFrame, input, names, output_name);
}

fn withRowCumulativeDistinctCountCore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const total_slots = std.math.mul(usize, input.rows, check_names.len) catch return error.InvalidShape;
    const flat_values = try input.allocator.alloc(f64, total_slots);
    defer input.allocator.free(flat_values);
    const flat_validity = try input.allocator.alloc(bool, total_slots);
    defer input.allocator.free(flat_validity);
    @memset(flat_values, 0.0);
    @memset(flat_validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const offset = row * check_names.len + col_index;
                    flat_values[offset] = realValueAsF64(@TypeOf(raw_value), raw_value);
                    flat_validity[offset] = true;
                }
            },
        }
    }

    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |_, output_name, col_index| {
        for (0..input.rows) |row| {
            const offset = row * check_names.len + col_index;
            if (!flat_validity[offset]) continue;
            const candidate = flat_values[offset];

            var seen = false;
            for (0..col_index) |previous_index| {
                const previous_offset = row * check_names.len + previous_index;
                if (!flat_validity[previous_offset]) continue;
                if (rowModeValueEqual(flat_values[previous_offset], candidate)) {
                    seen = true;
                    break;
                }
            }
            if (!seen) counts[row] += 1;
        }

        var column = try DeviceColumn.fromSlice(i64, input.allocator, counts, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeDistinctCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeDistinctCountCore(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumDistinctCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeDistinctCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixDistinctCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeDistinctCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNUnique(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeDistinctCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNUnique(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeDistinctCount(DeviceDataFrame, input, names, output_names);
}

const RowNumericDispersion = enum { variance, magnitude_variance, stddev, magnitude_stddev, sem, magnitude_sem, cv, magnitude_cv, fano, magnitude_fano, skewness, magnitude_skewness, kurtosis, magnitude_kurtosis };

fn withRowNumericDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
    comptime reduction: RowNumericDispersion,
) DeviceFrameArrayError!DeviceDataFrame {
    if (std.math.isNan(correction) or correction < 0.0) return error.InvalidShape;

    const check_names = if (names.len == 0) input.names else names;
    const means = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(means);
    const m2s = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(m2s);
    const m3s = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(m3s);
    const m4s = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(m4s);
    const counts = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(counts);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(means, 0.0);
    @memset(m2s, 0.0);
    @memset(m3s, 0.0);
    @memset(m4s, 0.0);
    @memset(counts, 0);
    @memset(validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        if (!source.dtype().isReal()) return error.TypeMismatch;

        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);

                for (host_values, 0..) |raw_value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;

                    // Keep row-wise dispersion numerically aligned with the
                    // scalar column moment implementation: online central
                    // moment updates give stable variance/skew/kurtosis while
                    // skipping nulls without materializing a dense row matrix.
                    const previous_count = counts[row];
                    const real_value = realValueAsF64(@TypeOf(raw_value), raw_value);
                    const value = if (reduction == .magnitude_variance or reduction == .magnitude_stddev or reduction == .magnitude_sem or reduction == .magnitude_cv or reduction == .magnitude_fano or reduction == .magnitude_skewness or reduction == .magnitude_kurtosis) @abs(real_value) else real_value;
                    counts[row] += 1;
                    const n: f64 = @floatFromInt(counts[row]);
                    const previous_n: f64 = @floatFromInt(previous_count);
                    const delta = value - means[row];
                    means[row] += delta / n;
                    const delta_n = delta / n;
                    const delta_n2 = delta_n * delta_n;
                    const term1 = delta * delta_n * previous_n;
                    const previous_m2 = m2s[row];
                    const previous_m3 = m3s[row];
                    m4s[row] += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * previous_m2 - 4.0 * delta_n * previous_m3;
                    m3s[row] += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * previous_m2;
                    m2s[row] += term1;
                    validity[row] = true;
                }
            },
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    for (values, validity, counts, m2s, m3s, m4s, means) |*value, valid, count, m2, m3, m4, mean| {
        if (!valid) {
            value.* = 0.0;
            continue;
        }
        const denominator = @as(f64, @floatFromInt(count)) - correction;
        const variance = if (denominator <= 0.0) std.math.nan(f64) else m2 / denominator;
        const stddev_value = std.math.sqrt(variance);
        value.* = switch (reduction) {
            .variance, .magnitude_variance => variance,
            .stddev, .magnitude_stddev => stddev_value,
            .sem, .magnitude_sem => stddev_value / std.math.sqrt(@as(f64, @floatFromInt(count))),
            .cv, .magnitude_cv => if (mean == 0.0) std.math.nan(f64) else stddev_value / mean,
            .fano, .magnitude_fano => if (mean == 0.0) std.math.nan(f64) else variance / mean,
            .skewness, .magnitude_skewness => if (count < 2 or m2 == 0.0) std.math.nan(f64) else std.math.sqrt(@as(f64, @floatFromInt(count))) * m3 / std.math.pow(f64, m2, 1.5),
            .kurtosis, .magnitude_kurtosis => if (count < 2 or m2 == 0.0) std.math.nan(f64) else @as(f64, @floatFromInt(count)) * m4 / (m2 * m2) - 3.0,
        };
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowVariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .variance);
}

pub fn withRowMagnitudeVariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .magnitude_variance);
}

pub fn withRowAbsVariance(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeVariance(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowMagnitudeVar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeVariance(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowAbsVar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeVariance(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowStddev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .stddev);
}

pub fn withRowMagnitudeStddev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .magnitude_stddev);
}

pub fn withRowAbsStddev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeStddev(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowMagnitudeStd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeStddev(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowAbsStd(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeStddev(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowSem(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .sem);
}

pub fn withRowMagnitudeSem(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .magnitude_sem);
}

pub fn withRowAbsSem(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeSem(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowCv(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .cv);
}

pub fn withRowMagnitudeCv(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .magnitude_cv);
}

pub fn withRowAbsCv(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeCv(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowMagnitudeFano(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .magnitude_fano);
}

pub fn withRowAbsFano(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeFano(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowMagnitudeIndexOfDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeFano(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowAbsIndexOfDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeFano(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowFano(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .fano);
}

pub fn withRowIndexOfDispersion(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowFano(DeviceDataFrame, input, names, output_name, correction);
}

pub fn withRowMagnitudeSkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, 0.0, .magnitude_skewness);
}

pub fn withRowAbsSkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeSkewness(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeSkew(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeSkewness(DeviceDataFrame, input, names, output_name);
}

pub fn withRowAbsSkew(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeSkewness(DeviceDataFrame, input, names, output_name);
}

pub fn withRowSkewness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, 0.0, .skewness);
}

pub fn withRowSkew(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowSkewness(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeKurtosis(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, 0.0, .magnitude_kurtosis);
}

pub fn withRowAbsKurtosis(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeKurtosis(DeviceDataFrame, input, names, output_name);
}

pub fn withRowMagnitudeKurt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeKurtosis(DeviceDataFrame, input, names, output_name);
}

pub fn withRowAbsKurt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowMagnitudeKurtosis(DeviceDataFrame, input, names, output_name);
}

pub fn withRowKurtosis(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, 0.0, .kurtosis);
}

pub fn withRowKurt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowKurtosis(DeviceDataFrame, input, names, output_name);
}

fn withRowBoolPredicateCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime target: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    for (check_names) |name| {
        const source = try input.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;

        const host_values = try source.bool.toOwnedSlice(input.allocator);
        defer input.allocator.free(host_values);
        const maybe_validity = try validityValues(source.bool, input.allocator);
        defer if (maybe_validity) |validity| input.allocator.free(validity);

        for (counts, host_values, 0..) |*slot, value, row| {
            const valid = if (maybe_validity) |validity| validity[row] else true;
            if (valid and value == target) slot.* += 1;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(i64, input.allocator, counts, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowTrueCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolPredicateCount(DeviceDataFrame, input, names, output_name, true);
}

pub fn withRowFalseCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolPredicateCount(DeviceDataFrame, input, names, output_name, false);
}

fn withRowCumulativeBoolPredicateCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime target: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;

        const host_values = try source.bool.toOwnedSlice(input.allocator);
        defer input.allocator.free(host_values);
        const maybe_validity = try validityValues(source.bool, input.allocator);
        defer if (maybe_validity) |validity| input.allocator.free(validity);

        for (counts, host_values, 0..) |*slot, value, row| {
            const valid = if (maybe_validity) |validity| validity[row] else true;
            if (valid and value == target) slot.* += 1;
        }

        var column = try DeviceColumn.fromSlice(i64, input.allocator, counts, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeTrueCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolPredicateCount(DeviceDataFrame, input, names, output_names, true);
}

pub fn withRowCumTrueCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeTrueCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixTrueCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeTrueCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFalseCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolPredicateCount(DeviceDataFrame, input, names, output_names, false);
}

pub fn withRowCumFalseCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFalseCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixFalseCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFalseCount(DeviceDataFrame, input, names, output_names);
}

fn withRowCumulativeBoolPredicateRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime target: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);
    const ratios = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(ratios);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |name, output_name, col_index| {
        const source = try input.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;

        const host_values = try source.bool.toOwnedSlice(input.allocator);
        defer input.allocator.free(host_values);
        const maybe_validity = try validityValues(source.bool, input.allocator);
        defer if (maybe_validity) |validity| input.allocator.free(validity);

        for (counts, host_values, 0..) |*slot, value, row| {
            const valid = if (maybe_validity) |validity| validity[row] else true;
            if (valid and value == target) slot.* += 1;
        }

        const denominator: f64 = @floatFromInt(col_index + 1);
        for (ratios, counts) |*ratio, count| {
            ratio.* = @as(f64, @floatFromInt(count)) / denominator;
        }

        var column = try DeviceColumn.fromSlice(f64, input.allocator, ratios, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeTrueRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolPredicateRatio(DeviceDataFrame, input, names, output_names, true);
}

pub fn withRowCumTrueRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeTrueRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixTrueRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeTrueRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFalseRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolPredicateRatio(DeviceDataFrame, input, names, output_names, false);
}

pub fn withRowCumFalseRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFalseRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixFalseRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFalseRatio(DeviceDataFrame, input, names, output_names);
}

const RowBoolReduction = enum { any_true, all_true, any_false, all_false };

fn withRowBoolReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowBoolReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const values = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(validity, false);
    @memset(values, switch (reduction) {
        .any_true, .any_false => false,
        .all_true, .all_false => true,
    });

    for (check_names) |name| {
        const source = try input.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;

        const host_values = try source.bool.toOwnedSlice(input.allocator);
        defer input.allocator.free(host_values);
        const maybe_validity = try validityValues(source.bool, input.allocator);
        defer if (maybe_validity) |mask| input.allocator.free(mask);

        for (host_values, 0..) |value, row| {
            const valid = if (maybe_validity) |mask| mask[row] else true;
            if (!valid) continue;
            validity[row] = true;
            switch (reduction) {
                .any_true => values[row] = values[row] or value,
                .all_true => values[row] = values[row] and value,
                .any_false => values[row] = values[row] or !value,
                .all_false => values[row] = values[row] and !value,
            }
        }
    }

    for (values, validity) |*value, valid| {
        if (!valid) value.* = false;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(bool, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowAnyTrue(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolReduction(DeviceDataFrame, input, names, output_name, .any_true);
}

pub fn withRowAllTrue(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolReduction(DeviceDataFrame, input, names, output_name, .all_true);
}

pub fn withRowAnyFalse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolReduction(DeviceDataFrame, input, names, output_name, .any_false);
}

pub fn withRowAllFalse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolReduction(DeviceDataFrame, input, names, output_name, .all_false);
}

fn withRowCumulativeBoolReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: RowBoolReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const running = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(running);
    const seen = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(seen);
    @memset(running, switch (reduction) {
        .any_true, .any_false => false,
        .all_true, .all_false => true,
    });
    @memset(seen, false);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;

        const host_values = try source.bool.toOwnedSlice(input.allocator);
        defer input.allocator.free(host_values);
        const maybe_validity = try validityValues(source.bool, input.allocator);
        defer if (maybe_validity) |validity| input.allocator.free(validity);
        var output_values = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(output_values);
        const output_validity = try input.allocator.alloc(bool, input.rows);
        defer input.allocator.free(output_validity);
        @memset(output_values, false);
        @memset(output_validity, false);

        for (host_values, 0..) |value, row| {
            const valid = if (maybe_validity) |validity| validity[row] else true;
            if (!valid) continue;
            const candidate = switch (reduction) {
                .any_true, .all_true => value,
                .any_false, .all_false => !value,
            };
            running[row] = if (!seen[row]) candidate else switch (reduction) {
                .any_true, .any_false => running[row] or candidate,
                .all_true, .all_false => running[row] and candidate,
            };
            seen[row] = true;
            output_values[row] = running[row];
            output_validity[row] = true;
        }

        var column = try DeviceColumn.fromSliceWithValidity(bool, input.allocator, output_values, output_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeAnyTrue(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolReduction(DeviceDataFrame, input, names, output_names, .any_true);
}

pub fn withRowCumAnyTrue(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyTrue(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyTrue(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyTrue(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllTrue(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolReduction(DeviceDataFrame, input, names, output_names, .all_true);
}

pub fn withRowCumAllTrue(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllTrue(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllTrue(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllTrue(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAnyFalse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolReduction(DeviceDataFrame, input, names, output_names, .any_false);
}

pub fn withRowCumAnyFalse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyFalse(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyFalse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyFalse(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllFalse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolReduction(DeviceDataFrame, input, names, output_names, .all_false);
}

pub fn withRowCumAllFalse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllFalse(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllFalse(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllFalse(DeviceDataFrame, input, names, output_names);
}

const RowBoolMatchIndex = enum { first_true, last_true, first_false, last_false };

fn withRowBoolMatchIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime search: RowBoolMatchIndex,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const indices = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(indices);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(indices, 0);
    @memset(validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;

        const host_values = try source.bool.toOwnedSlice(input.allocator);
        defer input.allocator.free(host_values);
        const maybe_validity = try validityValues(source.bool, input.allocator);
        defer if (maybe_validity) |mask| input.allocator.free(mask);
        const output_index = std.math.cast(i64, col_index) orelse return error.InvalidShape;

        for (host_values, 0..) |value, row| {
            const valid = if (maybe_validity) |mask| mask[row] else true;
            if (!valid) continue;
            const matches = switch (search) {
                .first_true, .last_true => value,
                .first_false, .last_false => !value,
            };
            if (!matches) continue;

            // Missing matches are represented by the validity mask rather than
            // a sentinel, so column position 0 remains a legitimate result.
            switch (search) {
                .first_true, .first_false => if (!validity[row]) {
                    indices[row] = output_index;
                    validity[row] = true;
                },
                .last_true, .last_false => {
                    indices[row] = output_index;
                    validity[row] = true;
                },
            }
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, indices, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowFirstTrueIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolMatchIndex(DeviceDataFrame, input, names, output_name, .first_true);
}

pub fn withRowLastTrueIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolMatchIndex(DeviceDataFrame, input, names, output_name, .last_true);
}

pub fn withRowFirstFalseIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolMatchIndex(DeviceDataFrame, input, names, output_name, .first_false);
}

pub fn withRowLastFalseIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolMatchIndex(DeviceDataFrame, input, names, output_name, .last_false);
}

fn withRowCumulativeBoolMatchIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime search: RowBoolMatchIndex,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    if (output_names.len != check_names.len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }

    const indices = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(indices);
    const output_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(output_validity);
    @memset(indices, 0);
    @memset(output_validity, false);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |name, output_name, col_index| {
        const source = try input.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;

        const host_values = try source.bool.toOwnedSlice(input.allocator);
        defer input.allocator.free(host_values);
        const maybe_validity = try validityValues(source.bool, input.allocator);
        defer if (maybe_validity) |mask| input.allocator.free(mask);
        const output_index = std.math.cast(i64, col_index) orelse return error.InvalidShape;

        for (host_values, 0..) |value, row| {
            const valid = if (maybe_validity) |mask| mask[row] else true;
            if (!valid) continue;
            const matches = switch (search) {
                .first_true, .last_true => value,
                .first_false, .last_false => !value,
            };
            if (!matches) continue;

            switch (search) {
                .first_true, .first_false => if (!output_validity[row]) {
                    indices[row] = output_index;
                    output_validity[row] = true;
                },
                .last_true, .last_false => {
                    indices[row] = output_index;
                    output_validity[row] = true;
                },
            }
        }

        var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, indices, output_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeFirstTrueIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolMatchIndex(DeviceDataFrame, input, names, output_names, .first_true);
}

pub fn withRowPrefixFirstTrueIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstTrueIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastTrueIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolMatchIndex(DeviceDataFrame, input, names, output_names, .last_true);
}

pub fn withRowPrefixLastTrueIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastTrueIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstFalseIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolMatchIndex(DeviceDataFrame, input, names, output_names, .first_false);
}

pub fn withRowPrefixFirstFalseIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstFalseIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastFalseIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeBoolMatchIndex(DeviceDataFrame, input, names, output_names, .last_false);
}

pub fn withRowPrefixLastFalseIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastFalseIndex(DeviceDataFrame, input, names, output_names);
}

fn withRowBoolPredicateRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime target: bool,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const numerators = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(numerators);
    const denominators = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(denominators);
    @memset(numerators, 0);
    @memset(denominators, 0);

    for (check_names) |name| {
        const source = try input.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;

        const host_values = try source.bool.toOwnedSlice(input.allocator);
        defer input.allocator.free(host_values);
        const maybe_validity = try validityValues(source.bool, input.allocator);
        defer if (maybe_validity) |validity| input.allocator.free(validity);

        for (host_values, 0..) |value, row| {
            const valid = if (maybe_validity) |validity| validity[row] else true;
            if (!valid) continue;
            denominators[row] += 1;
            if (value == target) numerators[row] += 1;
        }
    }

    const ratios = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(ratios);
    const ratio_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(ratio_validity);
    for (ratios, ratio_validity, numerators, denominators) |*ratio, *valid, numerator, denominator| {
        // Row-wise ratios can represent an all-null denominator precisely via
        // the output validity mask. This avoids overloading NaN with null
        // semantics while keeping scalar ratio APIs free to return NaN when no
        // scalar validity channel exists.
        valid.* = denominator != 0;
        ratio.* = if (denominator == 0) 0.0 else @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator));
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, ratios, ratio_validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowTrueRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolPredicateRatio(DeviceDataFrame, input, names, output_name, true);
}

pub fn withRowFalseRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowBoolPredicateRatio(DeviceDataFrame, input, names, output_name, false);
}

const RowNumericPredicate = enum { nan, inf, positive_inf, negative_inf, zero, positive_zero, negative_zero, non_zero, positive, signbit, negative, finite, normal, subnormal, non_finite };

fn rowNumericPredicateMatches(comptime T: type, value: T, comptime predicate: RowNumericPredicate) bool {
    return switch (predicate) {
        .nan => isNanValue(T, value),
        .inf => isInfValue(T, value),
        .positive_inf => isPositiveInfValue(T, value),
        .negative_inf => isNegativeInfValue(T, value),
        .zero => isZeroValue(T, value),
        .positive_zero => isPositiveZeroValue(T, value),
        .negative_zero => isNegativeZeroValue(T, value),
        .non_zero => isNonZeroValue(T, value),
        .positive => isPositiveValue(T, value),
        .signbit => isSignBitValue(T, value),
        .negative => isNegativeValue(T, value),
        .finite => isFiniteValue(T, value),
        .normal => isNormalValue(T, value),
        .subnormal => isSubnormalValue(T, value),
        .non_finite => !isFiniteValue(T, value),
    };
}

fn withRowNumericPredicateCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                for (counts, host_values, 0..) |*slot, value, row| {
                    const valid = if (maybe_validity) |validity| validity[row] else true;
                    if (valid and rowNumericPredicateMatches(@TypeOf(value), value, predicate)) slot.* += 1;
                }
            },
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(i64, input.allocator, counts, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowNaNCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .nan);
}

pub fn withRowInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .inf);
}

pub fn withRowPositiveInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .positive_inf);
}

pub fn withRowNegativeInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .negative_inf);
}

pub fn withRowZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .zero);
}

pub fn withRowPositiveZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .positive_zero);
}

pub fn withRowNegativeZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .negative_zero);
}

pub fn withRowNonZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .non_zero);
}

pub fn withRowPositiveCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .positive);
}

pub fn withRowSignBitCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .signbit);
}

pub fn withRowNegativeCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .negative);
}

pub fn withRowFiniteCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .finite);
}

pub fn withRowNormalCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .normal);
}

pub fn withRowSubnormalCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .subnormal);
}

pub fn withRowNonFiniteCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .non_finite);
}

const RowNumericPredicateReduction = enum { any, all };

fn withRowNumericPredicateReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime predicate: RowNumericPredicate,
    comptime reduction: RowNumericPredicateReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const values = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(values);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(values, switch (reduction) {
        .any => false,
        .all => true,
    });
    @memset(validity, false);

    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const matches = rowNumericPredicateMatches(@TypeOf(value), value, predicate);
                    values[row] = if (!validity[row]) matches else switch (reduction) {
                        .any => values[row] or matches,
                        .all => values[row] and matches,
                    };
                    validity[row] = true;
                }
            },
        }
    }

    for (values, validity) |*value, valid| {
        if (!valid) value.* = false;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(bool, input.allocator, values, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

fn withRowCumulativeNumericPredicateReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime predicate: RowNumericPredicate,
    comptime reduction: RowNumericPredicateReduction,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    try validateRowCumulativeOutputNames(output_names, check_names.len);

    const running = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(running);
    const seen = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(seen);
    @memset(running, switch (reduction) {
        .any => false,
        .all => true,
    });
    @memset(seen, false);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid) continue;
                    const matches = rowNumericPredicateMatches(@TypeOf(value), value, predicate);
                    running[row] = if (!seen[row]) matches else switch (reduction) {
                        .any => running[row] or matches,
                        .all => running[row] and matches,
                    };
                    seen[row] = true;
                }
            },
        }

        for (running, seen) |*value, valid| {
            if (!valid) value.* = false;
        }

        var column = try DeviceColumn.fromSliceWithValidity(bool, input.allocator, running, seen, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowAnyNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .nan, .any);
}

pub fn withRowAllNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .nan, .all);
}

pub fn withRowCumulativeAnyNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .nan, .any);
}

pub fn withRowCumAnyNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNaN(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNaN(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .nan, .all);
}

pub fn withRowCumAllNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNaN(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllNaN(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNaN(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .inf, .any);
}

pub fn withRowAllInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .inf, .all);
}

pub fn withRowCumulativeAnyInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .inf, .any);
}

pub fn withRowCumAnyInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .inf, .all);
}

pub fn withRowCumAllInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .positive_inf, .any);
}

pub fn withRowAllPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .positive_inf, .all);
}

pub fn withRowCumulativeAnyPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .positive_inf, .any);
}

pub fn withRowCumAnyPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyPositiveInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyPositiveInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .positive_inf, .all);
}

pub fn withRowCumAllPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllPositiveInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllPositiveInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllPositiveInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .negative_inf, .any);
}

pub fn withRowAllNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .negative_inf, .all);
}

pub fn withRowCumulativeAnyNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .negative_inf, .any);
}

pub fn withRowCumAnyNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNegativeInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNegativeInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .negative_inf, .all);
}

pub fn withRowCumAllNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNegativeInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllNegativeInf(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNegativeInf(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .finite, .any);
}

pub fn withRowAllFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .finite, .all);
}

pub fn withRowCumulativeAnyFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .finite, .any);
}

pub fn withRowCumAnyFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyFinite(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyFinite(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .finite, .all);
}

pub fn withRowCumAllFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllFinite(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllFinite(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .normal, .any);
}

pub fn withRowAllNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .normal, .all);
}

pub fn withRowCumulativeAnyNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .normal, .any);
}

pub fn withRowCumAnyNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNormal(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNormal(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .normal, .all);
}

pub fn withRowCumAllNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNormal(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllNormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNormal(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnySubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .subnormal, .any);
}

pub fn withRowAllSubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .subnormal, .all);
}

pub fn withRowCumulativeAnySubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .subnormal, .any);
}

pub fn withRowCumAnySubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnySubnormal(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnySubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnySubnormal(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllSubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .subnormal, .all);
}

pub fn withRowCumAllSubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllSubnormal(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllSubnormal(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllSubnormal(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .non_finite, .any);
}

pub fn withRowAllNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .non_finite, .all);
}

pub fn withRowCumulativeAnyNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .non_finite, .any);
}

pub fn withRowCumAnyNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNonFinite(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNonFinite(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .non_finite, .all);
}

pub fn withRowCumAllNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNonFinite(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllNonFinite(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNonFinite(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .zero, .any);
}

pub fn withRowAllZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .zero, .all);
}

pub fn withRowCumulativeAnyZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .zero, .any);
}

pub fn withRowCumAnyZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .zero, .all);
}

pub fn withRowCumAllZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .non_zero, .any);
}

pub fn withRowAllNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .non_zero, .all);
}

pub fn withRowCumulativeAnyNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .non_zero, .any);
}

pub fn withRowCumAnyNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNonZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNonZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .non_zero, .all);
}

pub fn withRowCumAllNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNonZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllNonZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNonZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .positive_zero, .any);
}

pub fn withRowAllPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .positive_zero, .all);
}

pub fn withRowCumulativeAnyPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .positive_zero, .any);
}

pub fn withRowCumAnyPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyPositiveZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyPositiveZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .positive_zero, .all);
}

pub fn withRowCumAllPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllPositiveZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllPositiveZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllPositiveZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .negative_zero, .any);
}

pub fn withRowAllNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .negative_zero, .all);
}

pub fn withRowCumulativeAnyNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .negative_zero, .any);
}

pub fn withRowCumAnyNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNegativeZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNegativeZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .negative_zero, .all);
}

pub fn withRowCumAllNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNegativeZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllNegativeZero(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNegativeZero(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .positive, .any);
}

pub fn withRowAllPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .positive, .all);
}

pub fn withRowCumulativeAnyPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .positive, .any);
}

pub fn withRowCumAnyPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyPositive(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyPositive(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .positive, .all);
}

pub fn withRowCumAllPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllPositive(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllPositive(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllPositive(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnySignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .signbit, .any);
}

pub fn withRowAllSignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .signbit, .all);
}

pub fn withRowCumulativeAnySignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .signbit, .any);
}

pub fn withRowCumAnySignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnySignBit(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnySignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnySignBit(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllSignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .signbit, .all);
}

pub fn withRowCumAllSignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllSignBit(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllSignBit(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllSignBit(DeviceDataFrame, input, names, output_names);
}

pub fn withRowAnyNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .negative, .any);
}

pub fn withRowAllNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateReduction(DeviceDataFrame, input, names, output_name, .negative, .all);
}

pub fn withRowCumulativeAnyNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .negative, .any);
}

pub fn withRowCumAnyNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNegative(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAnyNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAnyNegative(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeAllNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateReduction(DeviceDataFrame, input, names, output_names, .negative, .all);
}

pub fn withRowCumAllNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNegative(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixAllNegative(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeAllNegative(DeviceDataFrame, input, names, output_names);
}

const RowNumericPredicateIndexSearch = enum { first, last };

fn withRowNumericPredicateIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime predicate: RowNumericPredicate,
    comptime search: RowNumericPredicateIndexSearch,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const indices = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(indices);
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(indices, 0);
    @memset(validity, false);

    for (check_names, 0..) |name, col_index| {
        const source = try input.column(name);
        const output_index = std.math.cast(i64, col_index) orelse return error.InvalidShape;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |mask| input.allocator.free(mask);
                for (host_values, 0..) |value, row| {
                    const valid = if (maybe_validity) |mask| mask[row] else true;
                    if (!valid or !rowNumericPredicateMatches(@TypeOf(value), value, predicate)) continue;
                    switch (search) {
                        .first => if (!validity[row]) {
                            indices[row] = output_index;
                            validity[row] = true;
                        },
                        .last => {
                            indices[row] = output_index;
                            validity[row] = true;
                        },
                    }
                }
            },
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, indices, validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowFirstNaNIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .nan, .first);
}

pub fn withRowFirstNanIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowFirstNaNIndex(DeviceDataFrame, input, names, output_name);
}

pub fn withRowLastNaNIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .nan, .last);
}

pub fn withRowLastNanIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowLastNaNIndex(DeviceDataFrame, input, names, output_name);
}

pub fn withRowFirstInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .inf, .first);
}

pub fn withRowLastInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .inf, .last);
}

pub fn withRowFirstPositiveInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .positive_inf, .first);
}

pub fn withRowLastPositiveInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .positive_inf, .last);
}

pub fn withRowFirstNegativeInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .negative_inf, .first);
}

pub fn withRowLastNegativeInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .negative_inf, .last);
}

pub fn withRowFirstPositiveZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .positive_zero, .first);
}

pub fn withRowLastPositiveZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .positive_zero, .last);
}

pub fn withRowFirstNegativeZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .negative_zero, .first);
}

pub fn withRowLastNegativeZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .negative_zero, .last);
}

pub fn withRowFirstSignBitIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .signbit, .first);
}

pub fn withRowLastSignBitIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .signbit, .last);
}

pub fn withRowFirstFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .finite, .first);
}

pub fn withRowLastFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .finite, .last);
}

pub fn withRowFirstNormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .normal, .first);
}

pub fn withRowLastNormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .normal, .last);
}

pub fn withRowFirstSubnormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .subnormal, .first);
}

pub fn withRowLastSubnormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .subnormal, .last);
}

pub fn withRowFirstNonFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .non_finite, .first);
}

pub fn withRowFirstNonfiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowFirstNonFiniteIndex(DeviceDataFrame, input, names, output_name);
}

pub fn withRowLastNonFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .non_finite, .last);
}

pub fn withRowLastNonfiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowLastNonFiniteIndex(DeviceDataFrame, input, names, output_name);
}

pub fn withRowFirstZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .zero, .first);
}

pub fn withRowLastZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .zero, .last);
}

pub fn withRowFirstNonZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .non_zero, .first);
}

pub fn withRowFirstNonzeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowFirstNonZeroIndex(DeviceDataFrame, input, names, output_name);
}

pub fn withRowLastNonZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .non_zero, .last);
}

pub fn withRowLastNonzeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowLastNonZeroIndex(DeviceDataFrame, input, names, output_name);
}

pub fn withRowFirstPositiveIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .positive, .first);
}

pub fn withRowLastPositiveIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .positive, .last);
}

pub fn withRowFirstNegativeIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .negative, .first);
}

pub fn withRowLastNegativeIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateIndex(DeviceDataFrame, input, names, output_name, .negative, .last);
}

fn validateRowCumulativeOutputNames(
    output_names: []const []const u8,
    expected_len: usize,
) DeviceFrameArrayError!void {
    if (output_names.len != expected_len) return error.LengthMismatch;
    for (output_names, 0..) |output_name, index| {
        for (output_names[0..index]) |previous| {
            if (std.mem.eql(u8, output_name, previous)) return error.InvalidShape;
        }
    }
}

fn withRowCumulativeNumericPredicateIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime predicate: RowNumericPredicate,
    comptime search: RowNumericPredicateIndexSearch,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    try validateRowCumulativeOutputNames(output_names, check_names.len);

    const indices = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(indices);
    const output_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(output_validity);
    @memset(indices, 0);
    @memset(output_validity, false);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |name, output_name, col_index| {
        const source = try input.column(name);
        const output_index = std.math.cast(i64, col_index) orelse return error.InvalidShape;
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                for (host_values, 0..) |value, row| {
                    const valid = if (maybe_validity) |validity| validity[row] else true;
                    if (!valid or !rowNumericPredicateMatches(@TypeOf(value), value, predicate)) continue;
                    switch (search) {
                        .first => if (!output_validity[row]) {
                            indices[row] = output_index;
                            output_validity[row] = true;
                        },
                        .last => {
                            indices[row] = output_index;
                            output_validity[row] = true;
                        },
                    }
                }
            },
        }

        var column = try DeviceColumn.fromSliceWithValidity(i64, input.allocator, indices, output_validity, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeFirstNaNIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .nan, .first);
}

pub fn withRowPrefixFirstNaNIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstNaNIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastNaNIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .nan, .last);
}

pub fn withRowPrefixLastNaNIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastNaNIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .inf, .first);
}

pub fn withRowPrefixFirstInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstInfIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .inf, .last);
}

pub fn withRowPrefixLastInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastInfIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstPositiveInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .positive_inf, .first);
}

pub fn withRowPrefixFirstPositiveInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstPositiveInfIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastPositiveInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .positive_inf, .last);
}

pub fn withRowPrefixLastPositiveInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastPositiveInfIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstNegativeInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .negative_inf, .first);
}

pub fn withRowPrefixFirstNegativeInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstNegativeInfIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastNegativeInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .negative_inf, .last);
}

pub fn withRowPrefixLastNegativeInfIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastNegativeInfIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .finite, .first);
}

pub fn withRowPrefixFirstFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstFiniteIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .finite, .last);
}

pub fn withRowPrefixLastFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastFiniteIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstNormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .normal, .first);
}

pub fn withRowPrefixFirstNormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstNormalIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastNormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .normal, .last);
}

pub fn withRowPrefixLastNormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastNormalIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstSubnormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .subnormal, .first);
}

pub fn withRowPrefixFirstSubnormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstSubnormalIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastSubnormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .subnormal, .last);
}

pub fn withRowPrefixLastSubnormalIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastSubnormalIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstNonFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .non_finite, .first);
}

pub fn withRowPrefixFirstNonFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstNonFiniteIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastNonFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .non_finite, .last);
}

pub fn withRowPrefixLastNonFiniteIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastNonFiniteIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .zero, .first);
}

pub fn withRowPrefixFirstZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .zero, .last);
}

pub fn withRowPrefixLastZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstPositiveZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .positive_zero, .first);
}

pub fn withRowPrefixFirstPositiveZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstPositiveZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastPositiveZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .positive_zero, .last);
}

pub fn withRowPrefixLastPositiveZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastPositiveZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstNegativeZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .negative_zero, .first);
}

pub fn withRowPrefixFirstNegativeZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstNegativeZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastNegativeZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .negative_zero, .last);
}

pub fn withRowPrefixLastNegativeZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastNegativeZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstNonZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .non_zero, .first);
}

pub fn withRowCumulativeFirstNonzeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstNonZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixFirstNonZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstNonZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixFirstNonzeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPrefixFirstNonZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastNonZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .non_zero, .last);
}

pub fn withRowCumulativeLastNonzeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastNonZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixLastNonZeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastNonZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixLastNonzeroIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowPrefixLastNonZeroIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstPositiveIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .positive, .first);
}

pub fn withRowPrefixFirstPositiveIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstPositiveIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastPositiveIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .positive, .last);
}

pub fn withRowPrefixLastPositiveIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastPositiveIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstSignBitIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .signbit, .first);
}

pub fn withRowPrefixFirstSignBitIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstSignBitIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastSignBitIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .signbit, .last);
}

pub fn withRowPrefixLastSignBitIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastSignBitIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFirstNegativeIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .negative, .first);
}

pub fn withRowPrefixFirstNegativeIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFirstNegativeIndex(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeLastNegativeIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateIndex(DeviceDataFrame, input, names, output_names, .negative, .last);
}

pub fn withRowPrefixLastNegativeIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeLastNegativeIndex(DeviceDataFrame, input, names, output_names);
}

fn withRowCumulativeNumericPredicateCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    try validateRowCumulativeOutputNames(output_names, check_names.len);

    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names) |name, output_name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                for (counts, host_values, 0..) |*slot, value, row| {
                    const valid = if (maybe_validity) |validity| validity[row] else true;
                    if (valid and rowNumericPredicateMatches(@TypeOf(value), value, predicate)) slot.* += 1;
                }
            },
        }

        var column = try DeviceColumn.fromSlice(i64, input.allocator, counts, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

fn withRowCumulativeNumericPredicateRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    @setEvalBranchQuota(2000);
    const check_names = if (names.len == 0) input.names else names;
    try validateRowCumulativeOutputNames(output_names, check_names.len);

    const counts = try input.allocator.alloc(i64, input.rows);
    defer input.allocator.free(counts);
    @memset(counts, 0);
    const ratios = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(ratios);

    var result = try input.clone();
    errdefer result.deinit();
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    for (check_names, output_names, 0..) |name, output_name, col_index| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                for (counts, host_values, 0..) |*slot, value, row| {
                    const valid = if (maybe_validity) |validity| validity[row] else true;
                    if (valid and rowNumericPredicateMatches(@TypeOf(value), value, predicate)) slot.* += 1;
                }
            },
        }

        const denominator: f64 = @floatFromInt(col_index + 1);
        for (ratios, counts) |*ratio, count| {
            ratio.* = @as(f64, @floatFromInt(count)) / denominator;
        }

        var column = try DeviceColumn.fromSlice(f64, input.allocator, ratios, input.device);
        defer column.deinit();
        const next = try withColumn(DeviceDataFrame, result, output_name, column);
        result.deinit();
        result = next;
    }
    return result;
}

pub fn withRowCumulativeNaNCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .nan);
}

pub fn withRowCumNaNCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNaNCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNaNCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNaNCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .inf);
}

pub fn withRowCumInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeInfCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeInfCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativePositiveInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .positive_inf);
}

pub fn withRowCumPositiveInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveInfCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixPositiveInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveInfCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNegativeInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .negative_inf);
}

pub fn withRowCumNegativeInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeInfCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNegativeInfCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeInfCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFiniteCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .finite);
}

pub fn withRowCumFiniteCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFiniteCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixFiniteCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFiniteCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNonFiniteCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .non_finite);
}

pub fn withRowCumNonFiniteCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNonFiniteCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNonFiniteCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNonFiniteCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNormalCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .normal);
}

pub fn withRowCumNormalCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNormalCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNormalCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNormalCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeSubnormalCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .subnormal);
}

pub fn withRowCumSubnormalCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSubnormalCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixSubnormalCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSubnormalCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativePositiveZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .positive_zero);
}

pub fn withRowCumPositiveZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveZeroCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixPositiveZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveZeroCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNegativeZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .negative_zero);
}

pub fn withRowCumNegativeZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeZeroCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNegativeZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeZeroCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeSignBitCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .signbit);
}

pub fn withRowCumSignBitCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSignBitCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixSignBitCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSignBitCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .zero);
}

pub fn withRowCumZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeZeroCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeZeroCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNonZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .non_zero);
}

pub fn withRowCumNonZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNonZeroCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNonZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNonZeroCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativePositiveCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .positive);
}

pub fn withRowCumPositiveCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixPositiveCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNegativeCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateCount(DeviceDataFrame, input, names, output_names, .negative);
}

pub fn withRowCumNegativeCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeCount(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNegativeCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeCount(DeviceDataFrame, input, names, output_names);
}

fn withRowNumericPredicateRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const numerators = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(numerators);
    const denominators = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(denominators);
    @memset(numerators, 0);
    @memset(denominators, 0);

    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                for (host_values, 0..) |value, row| {
                    const valid = if (maybe_validity) |validity| validity[row] else true;
                    if (!valid) continue;
                    denominators[row] += 1;
                    if (rowNumericPredicateMatches(@TypeOf(value), value, predicate)) numerators[row] += 1;
                }
            },
        }
    }

    const ratios = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(ratios);
    const ratio_validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(ratio_validity);
    for (ratios, ratio_validity, numerators, denominators) |*ratio, *valid, numerator, denominator| {
        valid.* = denominator != 0;
        ratio.* = if (denominator == 0) 0.0 else @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator));
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSliceWithValidity(f64, input.allocator, ratios, ratio_validity, input.device);
    defer column.deinit();
    return withColumn(DeviceDataFrame, input, output_name, column);
}

pub fn withRowNaNRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .nan);
}

pub fn withRowNanRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNaNRatio(DeviceDataFrame, input, names, output_name);
}

pub fn withRowInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .inf);
}

pub fn withRowPositiveInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .positive_inf);
}

pub fn withRowNegativeInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .negative_inf);
}

pub fn withRowZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .zero);
}

pub fn withRowPositiveZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .positive_zero);
}

pub fn withRowNegativeZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .negative_zero);
}

pub fn withRowNonZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .non_zero);
}

pub fn withRowPositiveRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .positive);
}

pub fn withRowSignBitRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .signbit);
}

pub fn withRowNegativeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .negative);
}

pub fn withRowFiniteRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .finite);
}

pub fn withRowNormalRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .normal);
}

pub fn withRowSubnormalRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .subnormal);
}

pub fn withRowNonFiniteRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateRatio(DeviceDataFrame, input, names, output_name, .non_finite);
}

pub fn withRowCumulativeNaNRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .nan);
}

pub fn withRowCumNaNRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNaNRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNaNRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNaNRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .inf);
}

pub fn withRowCumInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeInfRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeInfRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativePositiveInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .positive_inf);
}

pub fn withRowCumPositiveInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveInfRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixPositiveInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveInfRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNegativeInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .negative_inf);
}

pub fn withRowCumNegativeInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeInfRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNegativeInfRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeInfRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeFiniteRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .finite);
}

pub fn withRowCumFiniteRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFiniteRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixFiniteRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeFiniteRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNonFiniteRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .non_finite);
}

pub fn withRowCumNonFiniteRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNonFiniteRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNonFiniteRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNonFiniteRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNormalRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .normal);
}

pub fn withRowCumNormalRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNormalRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNormalRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNormalRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeSubnormalRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .subnormal);
}

pub fn withRowCumSubnormalRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSubnormalRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixSubnormalRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSubnormalRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativePositiveZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .positive_zero);
}

pub fn withRowCumPositiveZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveZeroRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixPositiveZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveZeroRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNegativeZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .negative_zero);
}

pub fn withRowCumNegativeZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeZeroRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNegativeZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeZeroRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeSignBitRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .signbit);
}

pub fn withRowCumSignBitRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSignBitRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixSignBitRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeSignBitRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .zero);
}

pub fn withRowCumZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeZeroRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeZeroRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNonZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .non_zero);
}

pub fn withRowCumNonZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNonZeroRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNonZeroRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNonZeroRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativePositiveRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .positive);
}

pub fn withRowCumPositiveRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixPositiveRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativePositiveRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowCumulativeNegativeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNumericPredicateRatio(DeviceDataFrame, input, names, output_names, .negative);
}

pub fn withRowCumNegativeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeRatio(DeviceDataFrame, input, names, output_names);
}

pub fn withRowPrefixNegativeRatio(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowCumulativeNegativeRatio(DeviceDataFrame, input, names, output_names);
}

fn literalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    comptime T: type,
    value: T,
) DeviceFrameArrayError!std.meta.Elem(@TypeOf(input.columns)) {
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    const values = try input.allocator.alloc(T, input.rows);
    defer input.allocator.free(values);
    @memset(values, value);
    return DeviceColumn.fromSlice(T, input.allocator, values, input.device);
}

pub fn withColumnLiteral(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    comptime T: type,
    value: T,
) DeviceFrameArrayError!DeviceDataFrame {
    var literal_column = try literalColumn(DeviceDataFrame, input, T, value);
    defer literal_column.deinit();
    return withColumn(DeviceDataFrame, input, name, literal_column);
}

pub fn withColumnLiteralAt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    comptime T: type,
    value: T,
    target_index: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    var literal_column = try literalColumn(DeviceDataFrame, input, T, value);
    defer literal_column.deinit();
    return withColumnAt(DeviceDataFrame, input, name, literal_column, target_index);
}

pub fn withColumnLiteralBefore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    comptime T: type,
    value: T,
    before_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var literal_column = try literalColumn(DeviceDataFrame, input, T, value);
    defer literal_column.deinit();
    return withColumnBefore(DeviceDataFrame, input, name, literal_column, before_name);
}

pub fn withColumnLiteralAfter(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    comptime T: type,
    value: T,
    after_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    var literal_column = try literalColumn(DeviceDataFrame, input, T, value);
    defer literal_column.deinit();
    return withColumnAfter(DeviceDataFrame, input, name, literal_column, after_name);
}

pub fn withColumnLiteralScalar(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return switch (scalar) {
        inline else => |value| withColumnLiteral(DeviceDataFrame, input, name, @TypeOf(value), value),
    };
}

pub fn withColumnLiteralScalarAt(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
    target_index: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    return switch (scalar) {
        inline else => |value| withColumnLiteralAt(DeviceDataFrame, input, name, @TypeOf(value), value, target_index),
    };
}

pub fn withColumnLiteralScalarBefore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
    before_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return switch (scalar) {
        inline else => |value| withColumnLiteralBefore(DeviceDataFrame, input, name, @TypeOf(value), value, before_name),
    };
}

pub fn withColumnLiteralScalarAfter(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
    after_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return switch (scalar) {
        inline else => |value| withColumnLiteralAfter(DeviceDataFrame, input, name, @TypeOf(value), value, after_name),
    };
}

pub fn withRowIndex(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    offset: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    if (input.columnIndex(name) != null) return error.InvalidShape;
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    const values = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(values);
    for (values, 0..) |*slot, row| {
        slot.* = std.math.add(usize, offset, row) catch return error.InvalidShape;
    }

    var index_column = try DeviceColumn.fromSlice(usize, input.allocator, values, input.device);
    defer index_column.deinit();
    return withColumn(DeviceDataFrame, input, name, index_column);
}

pub fn renameColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    old_name: []const u8,
    new_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const rename_index = input.columnIndex(old_name) orelse return error.ColumnNotFound;
    if (input.columnIndex(new_name)) |existing_index| {
        if (existing_index != rename_index) return error.InvalidShape;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    for (input.names, source_names, 0..) |name, *slot, i| {
        slot.* = if (i == rename_index) new_name else name;
    }

    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, columns) |col, *slot| {
        slot.* = try col.clone();
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, source_names, columns, input.rows, input.device);
}

pub fn renameColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    old_names: []const []const u8,
    new_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    if (old_names.len != new_names.len) return error.LengthMismatch;

    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    for (input.names, source_names) |name, *slot| slot.* = name;

    for (old_names, new_names) |old_name, new_name| {
        const rename_index = input.columnIndex(old_name) orelse return error.ColumnNotFound;
        source_names[rename_index] = new_name;
    }

    for (source_names, 0..) |name, i| {
        for (source_names[i + 1 ..]) |candidate| {
            if (std.mem.eql(u8, name, candidate)) return error.InvalidShape;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, columns) |col, *slot| {
        slot.* = try col.clone();
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, source_names, columns, input.rows, input.device);
}

pub fn addColumnNamePrefix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    prefix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    var initialized: usize = 0;
    errdefer {
        for (source_names[0..initialized]) |name| input.allocator.free(name);
    }
    for (input.names, source_names) |name, *slot| {
        slot.* = try std.fmt.allocPrint(input.allocator, "{s}{s}", .{ prefix, name });
        initialized += 1;
    }
    defer {
        for (source_names) |name| input.allocator.free(name);
    }
    return renameColumns(DeviceDataFrame, input, input.names, source_names);
}

pub fn addColumnNameSuffix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    suffix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    var initialized: usize = 0;
    errdefer {
        for (source_names[0..initialized]) |name| input.allocator.free(name);
    }
    for (input.names, source_names) |name, *slot| {
        slot.* = try std.fmt.allocPrint(input.allocator, "{s}{s}", .{ name, suffix });
        initialized += 1;
    }
    defer {
        for (source_names) |name| input.allocator.free(name);
    }
    return renameColumns(DeviceDataFrame, input, input.names, source_names);
}

pub fn stripColumnNamePrefix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    prefix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    for (input.names, source_names) |name, *slot| {
        slot.* = if (std.mem.startsWith(u8, name, prefix)) name[prefix.len..] else name;
    }
    return renameColumns(DeviceDataFrame, input, input.names, source_names);
}

pub const removeColumnNamePrefix = stripColumnNamePrefix;

pub fn stripColumnNameSuffix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    suffix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    for (input.names, source_names) |name, *slot| {
        slot.* = if (std.mem.endsWith(u8, name, suffix)) name[0 .. name.len - suffix.len] else name;
    }
    return renameColumns(DeviceDataFrame, input, input.names, source_names);
}

pub const removeColumnNameSuffix = stripColumnNameSuffix;

pub fn replaceColumnNamePrefix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    old_prefix: []const u8,
    new_prefix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    var initialized: usize = 0;
    errdefer {
        for (source_names[0..initialized]) |name| input.allocator.free(name);
    }
    for (input.names, source_names) |name, *slot| {
        if (std.mem.startsWith(u8, name, old_prefix)) {
            slot.* = try std.fmt.allocPrint(input.allocator, "{s}{s}", .{ new_prefix, name[old_prefix.len..] });
        } else {
            slot.* = try input.allocator.dupe(u8, name);
        }
        initialized += 1;
    }
    defer {
        for (source_names) |name| input.allocator.free(name);
    }
    return renameColumns(DeviceDataFrame, input, input.names, source_names);
}

pub fn replaceColumnNameSuffix(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    old_suffix: []const u8,
    new_suffix: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);
    var initialized: usize = 0;
    errdefer {
        for (source_names[0..initialized]) |name| input.allocator.free(name);
    }
    for (input.names, source_names) |name, *slot| {
        if (std.mem.endsWith(u8, name, old_suffix)) {
            slot.* = try std.fmt.allocPrint(input.allocator, "{s}{s}", .{ name[0 .. name.len - old_suffix.len], new_suffix });
        } else {
            slot.* = try input.allocator.dupe(u8, name);
        }
        initialized += 1;
    }
    defer {
        for (source_names) |name| input.allocator.free(name);
    }
    return renameColumns(DeviceDataFrame, input, input.names, source_names);
}

pub fn moveColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    target_index: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_index = input.columnIndex(name) orelse return error.ColumnNotFound;
    if (target_index >= input.names.len) return error.IndexOutOfBounds;

    const source_names = try input.allocator.alloc([]const u8, input.names.len);
    defer input.allocator.free(source_names);

    // `target_index` is the column's final output position.  Build the desired
    // name order by injecting the moved name at that position and streaming the
    // remaining names around the source slot; `select` then performs the actual
    // column cloning so ownership stays centralized.
    var source_scan: usize = 0;
    for (source_names, 0..) |*slot, output_index| {
        if (output_index == target_index) {
            slot.* = input.names[source_index];
            continue;
        }

        while (source_scan == source_index) source_scan += 1;
        slot.* = input.names[source_scan];
        source_scan += 1;
    }
    return select(DeviceDataFrame, input, source_names);
}

pub fn moveColumnBefore(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    before_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_index = input.columnIndex(name) orelse return error.ColumnNotFound;
    const anchor_index = input.columnIndex(before_name) orelse return error.ColumnNotFound;
    if (source_index == anchor_index) return moveColumn(DeviceDataFrame, input, name, source_index);

    const target_index = if (source_index < anchor_index) anchor_index - 1 else anchor_index;
    return moveColumn(DeviceDataFrame, input, name, target_index);
}

pub fn moveColumnAfter(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    after_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source_index = input.columnIndex(name) orelse return error.ColumnNotFound;
    const anchor_index = input.columnIndex(after_name) orelse return error.ColumnNotFound;
    if (source_index == anchor_index) return moveColumn(DeviceDataFrame, input, name, source_index);

    const target_index = if (source_index < anchor_index) anchor_index else anchor_index + 1;
    return moveColumn(DeviceDataFrame, input, name, target_index);
}

pub fn dropColumns(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    drop_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    for (drop_names) |name| {
        if (input.columnIndex(name) == null) return error.ColumnNotFound;
    }
    if (drop_names.len == 0) return select(DeviceDataFrame, input, input.names);

    var keep_count: usize = 0;
    for (input.names) |name| {
        if (!nameInBorrowedList(name, drop_names)) keep_count += 1;
    }
    if (keep_count == 0) return DeviceDataFrame.initEmpty(input.allocator, input.rows, input.device);

    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    const source_names = try input.allocator.alloc([]const u8, keep_count);
    defer input.allocator.free(source_names);
    var columns = try input.allocator.alloc(DeviceColumn, keep_count);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }

    for (input.names, input.columns) |name, col| {
        if (nameInBorrowedList(name, drop_names)) continue;
        source_names[initialized] = name;
        columns[initialized] = try col.clone();
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, source_names, columns, input.rows, input.device);
}

pub fn dropColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumns(DeviceDataFrame, input, &.{name});
}

pub fn dropNulls(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    for (check_names) |name| {
        _ = try input.column(name);
    }

    const keep = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(keep);
    @memset(keep, true);
    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (keep, validity) |*slot, valid| slot.* = slot.* and valid;
                }
            },
        }
    }
    return filterRows(DeviceDataFrame, input, keep);
}

fn allNullRowMask(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError![]bool {
    const check_names = if (names.len == 0) input.names else names;
    for (check_names) |name| {
        _ = try input.column(name);
    }

    const all_null = try input.allocator.alloc(bool, input.rows);
    errdefer input.allocator.free(all_null);
    @memset(all_null, true);
    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                if (maybe_validity) |validity| {
                    for (all_null, validity) |*slot, valid| slot.* = slot.* and !valid;
                } else {
                    @memset(all_null, false);
                }
            },
        }
    }
    return all_null;
}

pub fn dropAllNulls(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const all_null = try allNullRowMask(DeviceDataFrame, input, names);
    defer input.allocator.free(all_null);
    const keep = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(keep);
    for (all_null, keep) |is_all_null, *slot| slot.* = !is_all_null;
    return filterRows(DeviceDataFrame, input, keep);
}

pub fn filterAllNulls(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const all_null = try allNullRowMask(DeviceDataFrame, input, names);
    defer input.allocator.free(all_null);
    return filterRows(DeviceDataFrame, input, all_null);
}

pub fn filterNullsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(name);
    const keep = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(keep);
    @memset(keep, false);

    switch (source.*) {
        inline else => |typed| {
            const maybe_validity = try validityValues(typed, input.allocator);
            defer if (maybe_validity) |validity| input.allocator.free(validity);
            if (maybe_validity) |validity| {
                for (keep, validity) |*slot, valid| slot.* = !valid;
            }
        },
    }
    return filterRows(DeviceDataFrame, input, keep);
}

fn dropRowsByNumericPredicate(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    const check_names = if (names.len == 0) input.names else names;
    const keep = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(keep);
    @memset(keep, true);

    for (check_names) |name| {
        const source = try input.column(name);
        switch (source.*) {
            inline else => |typed| {
                const host_values = try typed.toOwnedSlice(input.allocator);
                defer input.allocator.free(host_values);
                const maybe_validity = try validityValues(typed, input.allocator);
                defer if (maybe_validity) |validity| input.allocator.free(validity);
                for (keep, host_values, 0..) |*slot, value, row| {
                    const valid = if (maybe_validity) |validity| validity[row] else true;
                    if (valid and rowNumericPredicateMatches(@TypeOf(value), value, predicate)) slot.* = false;
                }
            },
        }
    }
    return filterRows(DeviceDataFrame, input, keep);
}

pub fn dropNaNs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .nan);
}

pub fn dropInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .inf);
}

pub fn dropPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .positive_inf);
}

pub fn dropNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .negative_inf);
}

pub fn dropZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .zero);
}

pub fn dropPositiveZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .positive_zero);
}

pub fn dropNegativeZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .negative_zero);
}

pub fn dropNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .non_zero);
}

pub fn dropPositives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .positive);
}

pub fn dropSignBits(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .signbit);
}

pub fn dropNegatives(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .negative);
}

pub fn dropFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .finite);
}

pub fn dropNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .normal);
}

pub fn dropSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .subnormal);
}

pub fn dropNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByNumericPredicate(DeviceDataFrame, input, names, .non_finite);
}

fn filterRowsByNumericPredicateColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    comptime predicate: RowNumericPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    const source = try input.column(name);
    const keep = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(keep);
    @memset(keep, false);

    switch (source.*) {
        inline else => |typed| {
            const host_values = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(host_values);
            const maybe_validity = try validityValues(typed, input.allocator);
            defer if (maybe_validity) |validity| input.allocator.free(validity);
            for (keep, host_values, 0..) |*slot, value, row| {
                const valid = if (maybe_validity) |validity| validity[row] else true;
                slot.* = valid and rowNumericPredicateMatches(@TypeOf(value), value, predicate);
            }
        },
    }
    return filterRows(DeviceDataFrame, input, keep);
}

pub fn filterNaNsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .nan);
}

pub fn filterInfsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .inf);
}

pub fn filterPositiveInfsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .positive_inf);
}

pub fn filterNegativeInfsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .negative_inf);
}

pub fn filterZerosColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .zero);
}

pub fn filterPositiveZerosColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .positive_zero);
}

pub fn filterNegativeZerosColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .negative_zero);
}

pub fn filterNonZerosColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .non_zero);
}

pub fn filterPositivesColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .positive);
}

pub fn filterSignBitsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .signbit);
}

pub fn filterNegativesColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .negative);
}

pub fn filterFinitesColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .finite);
}

pub fn filterNormalsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .normal);
}

pub fn filterSubnormalsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .subnormal);
}

pub fn filterNonFinitesColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterRowsByNumericPredicateColumn(DeviceDataFrame, input, name, .non_finite);
}

pub fn view(
    comptime DeviceDataFrameView: type,
    comptime DeviceColumnView: type,
    input: anytype,
) DeviceFrameArrayError!DeviceDataFrameView {
    const columns = try input.allocator.alloc(DeviceColumnView, input.columns.len);
    errdefer input.allocator.free(columns);
    for (input.columns, columns) |col, *slot| slot.* = col.view();
    return .{
        .allocator = input.allocator,
        .names = input.names,
        .columns = columns,
        .rows = input.rows,
        .device = input.device,
    };
}

pub fn sliceRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    start: usize,
    stop: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const end = @min(stop, input.rows);
    const begin = @min(start, end);
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, 0..) |col, i| {
        columns[i] = try col.sliceRows(begin, end);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, input.names, columns, end - begin, input.device);
}

pub fn sliceRowsSigned(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    start: isize,
    length: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const begin = try normalizeSignedRowIndexMode(start, input.rows, .raise);
    const stop = std.math.add(usize, begin, length) catch return error.InvalidShape;
    if (stop > input.rows) return error.IndexOutOfBounds;
    return sliceRows(DeviceDataFrame, input, begin, stop);
}

fn normalizeSignedSliceEndpoint(index: isize, rows: usize) DeviceFrameArrayError!usize {
    const signed_rows = std.math.cast(isize, rows) orelse return error.InvalidShape;
    var normalized = if (index < 0) signed_rows + index else index;
    if (normalized < 0) normalized = 0;
    if (normalized > signed_rows) normalized = signed_rows;
    return @intCast(normalized);
}

pub fn sliceRowsSignedStep(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    start: isize,
    stop: isize,
    step: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    if (step == 0) return error.InvalidShape;
    const begin = try normalizeSignedSliceEndpoint(start, input.rows);
    const end = try normalizeSignedSliceEndpoint(stop, input.rows);
    if (begin >= end) return takeRows(DeviceDataFrame, input, &.{});
    return sliceRowsStep(DeviceDataFrame, input, begin, end, step);
}

pub fn takeRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, 0..) |col, i| {
        columns[i] = try col.take(row_indices);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, input.names, columns, row_indices.len, input.device);
}

fn normalizeRowIndexMode(index: usize, rows: usize, mode: array_mod.IndexMode) DeviceFrameArrayError!usize {
    if (rows == 0) return error.IndexOutOfBounds;
    return switch (mode) {
        .raise => if (index >= rows) error.IndexOutOfBounds else index,
        .wrap => index % rows,
        .clip => @min(index, rows - 1),
    };
}

fn normalizeSignedRowIndexMode(index: isize, rows: usize, mode: array_mod.IndexMode) DeviceFrameArrayError!usize {
    if (rows == 0) return error.IndexOutOfBounds;
    const signed_rows = std.math.cast(isize, rows) orelse return error.InvalidShape;
    const normalized = switch (mode) {
        .raise => if (index < 0) signed_rows + index else index,
        .wrap => @mod(index, signed_rows),
        .clip => @min(@max(index, 0), signed_rows - 1),
    };
    if (normalized < 0 or normalized >= signed_rows) return error.IndexOutOfBounds;
    return @intCast(normalized);
}

pub fn takeRowsMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const usize,
    mode: array_mod.IndexMode,
) DeviceFrameArrayError!DeviceDataFrame {
    const normalized = try input.allocator.alloc(usize, row_indices.len);
    defer input.allocator.free(normalized);
    for (row_indices, normalized) |row_index, *slot| {
        slot.* = try normalizeRowIndexMode(row_index, input.rows, mode);
    }
    return takeRows(DeviceDataFrame, input, normalized);
}

pub fn takeRowsSigned(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const isize,
) DeviceFrameArrayError!DeviceDataFrame {
    return takeRowsSignedMode(DeviceDataFrame, input, row_indices, .raise);
}

pub fn takeRowsSignedMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const isize,
    mode: array_mod.IndexMode,
) DeviceFrameArrayError!DeviceDataFrame {
    const normalized = try input.allocator.alloc(usize, row_indices.len);
    defer input.allocator.free(normalized);
    for (row_indices, normalized) |row_index, *slot| {
        slot.* = try normalizeSignedRowIndexMode(row_index, input.rows, mode);
    }
    return takeRows(DeviceDataFrame, input, normalized);
}

fn normalizeRowIndexValue(comptime T: type, value: T, rows: usize, mode: array_mod.IndexMode) DeviceFrameArrayError!usize {
    return if (comptime @typeInfo(T).int.signedness == .signed) blk: {
        const signed = std.math.cast(isize, value) orelse return error.InvalidShape;
        break :blk try normalizeSignedRowIndexMode(signed, rows, mode);
    } else blk: {
        const unsigned = std.math.cast(usize, value) orelse return error.InvalidShape;
        break :blk try normalizeRowIndexMode(unsigned, rows, mode);
    };
}

fn takeRowsByIndexColumnTyped(
    comptime DeviceDataFrame: type,
    comptime T: type,
    input: DeviceDataFrame,
    index_column: anytype,
    mode: array_mod.IndexMode,
) DeviceFrameArrayError!DeviceDataFrame {
    const values = try index_column.toOwnedSlice(input.allocator);
    defer input.allocator.free(values);
    const maybe_validity = try validityValues(index_column, input.allocator);
    defer if (maybe_validity) |validity| input.allocator.free(validity);

    if (maybe_validity) |validity| {
        const row_indices = try input.allocator.alloc(?usize, values.len);
        defer input.allocator.free(row_indices);
        for (values, validity, row_indices) |value, valid, *slot| {
            // A nullable index column is equivalent to optional row gather:
            // invalid index rows materialize an all-null output row, while
            // valid rows still honor the requested bounds mode.
            slot.* = if (valid) try normalizeRowIndexValue(T, value, input.rows, mode) else null;
        }
        return takeOptionalRows(DeviceDataFrame, input, row_indices);
    }

    const row_indices = try input.allocator.alloc(usize, values.len);
    defer input.allocator.free(row_indices);
    for (values, row_indices) |value, *slot| {
        slot.* = try normalizeRowIndexValue(T, value, input.rows, mode);
    }
    return takeRows(DeviceDataFrame, input, row_indices);
}

pub fn takeRowsByColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    index_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return takeRowsByColumnMode(DeviceDataFrame, input, index_name, .raise);
}

pub fn takeRowsByColumnMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    index_name: []const u8,
    mode: array_mod.IndexMode,
) DeviceFrameArrayError!DeviceDataFrame {
    const index_column = try input.column(index_name);
    return switch (index_column.*) {
        .i8 => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, i8, input, typed, mode),
        .i16 => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, i16, input, typed, mode),
        .i32 => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, i32, input, typed, mode),
        .i64 => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, i64, input, typed, mode),
        .isize => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, isize, input, typed, mode),
        .u8 => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, u8, input, typed, mode),
        .u16 => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, u16, input, typed, mode),
        .u32 => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, u32, input, typed, mode),
        .u64 => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, u64, input, typed, mode),
        .usize => |typed| takeRowsByIndexColumnTyped(DeviceDataFrame, usize, input, typed, mode),
        else => error.TypeMismatch,
    };
}

fn appendDropRowsByIndexColumnTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    row_indices: *std.ArrayList(usize),
    index_column: anytype,
    rows: usize,
    mode: array_mod.IndexMode,
) DeviceFrameArrayError!void {
    const values = try index_column.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(index_column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);
    for (values, 0..) |value, i| {
        if (maybe_validity) |validity| {
            if (!validity[i]) continue;
        }
        try row_indices.append(allocator, try normalizeRowIndexValue(T, value, rows, mode));
    }
}

pub fn dropRowsByColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    index_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsByColumnMode(DeviceDataFrame, input, index_name, .raise);
}

pub fn dropRowsByColumnMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    index_name: []const u8,
    mode: array_mod.IndexMode,
) DeviceFrameArrayError!DeviceDataFrame {
    const index_column = try input.column(index_name);
    var row_indices: std.ArrayList(usize) = .empty;
    defer row_indices.deinit(input.allocator);
    switch (index_column.*) {
        .i8 => |typed| try appendDropRowsByIndexColumnTyped(i8, input.allocator, &row_indices, typed, input.rows, mode),
        .i16 => |typed| try appendDropRowsByIndexColumnTyped(i16, input.allocator, &row_indices, typed, input.rows, mode),
        .i32 => |typed| try appendDropRowsByIndexColumnTyped(i32, input.allocator, &row_indices, typed, input.rows, mode),
        .i64 => |typed| try appendDropRowsByIndexColumnTyped(i64, input.allocator, &row_indices, typed, input.rows, mode),
        .isize => |typed| try appendDropRowsByIndexColumnTyped(isize, input.allocator, &row_indices, typed, input.rows, mode),
        .u8 => |typed| try appendDropRowsByIndexColumnTyped(u8, input.allocator, &row_indices, typed, input.rows, mode),
        .u16 => |typed| try appendDropRowsByIndexColumnTyped(u16, input.allocator, &row_indices, typed, input.rows, mode),
        .u32 => |typed| try appendDropRowsByIndexColumnTyped(u32, input.allocator, &row_indices, typed, input.rows, mode),
        .u64 => |typed| try appendDropRowsByIndexColumnTyped(u64, input.allocator, &row_indices, typed, input.rows, mode),
        .usize => |typed| try appendDropRowsByIndexColumnTyped(usize, input.allocator, &row_indices, typed, input.rows, mode),
        else => return error.TypeMismatch,
    }
    return dropRows(DeviceDataFrame, input, row_indices.items);
}

pub fn repeatRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    repeat_count: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    if (repeat_count == 0 or input.rows == 0) return takeRows(DeviceDataFrame, input, &.{});
    const total_rows = std.math.mul(usize, input.rows, repeat_count) catch return error.InvalidShape;
    var row_indices = try input.allocator.alloc(usize, total_rows);
    defer input.allocator.free(row_indices);
    var write: usize = 0;
    for (0..input.rows) |row_index| {
        for (0..repeat_count) |_| {
            row_indices[write] = row_index;
            write += 1;
        }
    }
    return takeRows(DeviceDataFrame, input, row_indices);
}

pub fn tileRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    tile_count: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    if (tile_count == 0 or input.rows == 0) return takeRows(DeviceDataFrame, input, &.{});
    const total_rows = std.math.mul(usize, input.rows, tile_count) catch return error.InvalidShape;
    const row_indices = try input.allocator.alloc(usize, total_rows);
    defer input.allocator.free(row_indices);
    var write: usize = 0;
    // `repeatRows` repeats each row consecutively. `tileRows` instead repeats
    // the whole row block, mirroring Array.tile semantics for dataframe rows.
    for (0..tile_count) |_| {
        for (0..input.rows) |row_index| {
            row_indices[write] = row_index;
            write += 1;
        }
    }
    return takeRows(DeviceDataFrame, input, row_indices);
}

fn appendRepeatRowsFromCounts(
    allocator: std.mem.Allocator,
    row_indices: *std.ArrayList(usize),
    counts: anytype,
) DeviceFrameArrayError!void {
    for (counts, 0..) |count_value, row_index| {
        const count = switch (@typeInfo(@TypeOf(count_value))) {
            .int => |info| blk: {
                if (info.signedness == .signed and count_value < 0) return error.InvalidShape;
                break :blk std.math.cast(usize, count_value) orelse return error.InvalidShape;
            },
            else => return error.TypeMismatch,
        };
        try row_indices.ensureUnusedCapacity(allocator, count);
        for (0..count) |_| row_indices.appendAssumeCapacity(row_index);
    }
}

pub fn repeatRowsByColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    count_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const count_column = try input.column(count_name);
    if (count_column.hasNulls()) return error.TypeMismatch;

    var row_indices: std.ArrayList(usize) = .empty;
    defer row_indices.deinit(input.allocator);
    switch (count_column.*) {
        .i8 => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        .i16 => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        .i32 => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        .i64 => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        .isize => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        .u8 => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        .u16 => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        .u32 => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        .u64 => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        .usize => |typed| {
            const counts = try typed.toOwnedSlice(input.allocator);
            defer input.allocator.free(counts);
            try appendRepeatRowsFromCounts(input.allocator, &row_indices, counts);
        },
        else => return error.TypeMismatch,
    }
    return takeRows(DeviceDataFrame, input, row_indices.items);
}

pub fn dropRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const keep = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(keep);
    @memset(keep, true);

    // Row drops are position-set based rather than sequential deletions:
    // duplicate indices should not shift later positions or remove additional
    // rows.  Validate every requested position before reusing the shared
    // boolean-mask compaction path.
    for (row_indices) |row_index| {
        if (row_index >= input.rows) return error.IndexOutOfBounds;
        keep[row_index] = false;
    }
    return filterRows(DeviceDataFrame, input, keep);
}

pub fn dropRowsMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const usize,
    mode: array_mod.IndexMode,
) DeviceFrameArrayError!DeviceDataFrame {
    const normalized = try input.allocator.alloc(usize, row_indices.len);
    defer input.allocator.free(normalized);
    for (row_indices, normalized) |row_index, *slot| {
        slot.* = try normalizeRowIndexMode(row_index, input.rows, mode);
    }
    return dropRows(DeviceDataFrame, input, normalized);
}

pub fn dropRowsSigned(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const isize,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropRowsSignedMode(DeviceDataFrame, input, row_indices, .raise);
}

pub fn dropRowsSignedMode(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const isize,
    mode: array_mod.IndexMode,
) DeviceFrameArrayError!DeviceDataFrame {
    const normalized = try input.allocator.alloc(usize, row_indices.len);
    defer input.allocator.free(normalized);
    for (row_indices, normalized) |row_index, *slot| {
        slot.* = try normalizeSignedRowIndexMode(row_index, input.rows, mode);
    }
    return dropRows(DeviceDataFrame, input, normalized);
}

pub fn dropRowRange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    start: usize,
    stop: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const end = @min(stop, input.rows);
    const begin = @min(start, end);
    const keep = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(keep);
    @memset(keep, true);
    for (keep[begin..end]) |*slot| slot.* = false;
    return filterRows(DeviceDataFrame, input, keep);
}

pub fn sampleRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    count: usize,
    seed: u64,
) DeviceFrameArrayError!DeviceDataFrame {
    if (count > input.rows) return error.InvalidShape;
    const row_indices = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(row_indices);
    for (row_indices, 0..) |*slot, i| slot.* = i;

    var engine = alea.ScalarPrng.init(seed);
    const rng = alea.Rng.init(&engine);
    rng.shuffle(usize, row_indices);
    return takeRows(DeviceDataFrame, input, row_indices[0..count]);
}

fn sampleFractionCount(rows: usize, fraction: f64, comptime allow_overflow: bool) DeviceFrameArrayError!usize {
    if (std.math.isNan(fraction) or fraction < 0.0) return error.InvalidShape;
    if (!allow_overflow and fraction > 1.0) return error.InvalidShape;
    const count_float = @floor(fraction * @as(f64, @floatFromInt(rows)));
    if (count_float > @as(f64, @floatFromInt(std.math.maxInt(usize)))) return error.InvalidShape;
    return @intFromFloat(count_float);
}

pub fn sampleRowsFraction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    fraction: f64,
    seed: u64,
) DeviceFrameArrayError!DeviceDataFrame {
    return sampleRows(DeviceDataFrame, input, try sampleFractionCount(input.rows, fraction, false), seed);
}

pub fn sampleRowsWithReplacement(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    count: usize,
    seed: u64,
) DeviceFrameArrayError!DeviceDataFrame {
    if (count == 0) return takeRows(DeviceDataFrame, input, &.{});
    if (input.rows == 0) return error.EmptyDataFrame;

    const row_indices = try input.allocator.alloc(usize, count);
    defer input.allocator.free(row_indices);
    var engine = alea.ScalarPrng.init(seed);
    const rng = alea.Rng.init(&engine);
    for (row_indices) |*slot| {
        slot.* = rng.intRangeLessThan(usize, 0, input.rows);
    }
    return takeRows(DeviceDataFrame, input, row_indices);
}

pub fn sampleRowsFractionWithReplacement(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    fraction: f64,
    seed: u64,
) DeviceFrameArrayError!DeviceDataFrame {
    return sampleRowsWithReplacement(DeviceDataFrame, input, try sampleFractionCount(input.rows, fraction, true), seed);
}

pub fn strideRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    start: usize,
    step: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    if (step == 0) return error.InvalidShape;
    if (start >= input.rows) return takeRows(DeviceDataFrame, input, &.{});

    const count = 1 + (input.rows - 1 - start) / step;
    const row_indices = try input.allocator.alloc(usize, count);
    defer input.allocator.free(row_indices);
    var row = start;
    for (row_indices) |*slot| {
        slot.* = row;
        row += step;
    }
    return takeRows(DeviceDataFrame, input, row_indices);
}

pub fn sliceRowsStep(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    start: usize,
    stop: usize,
    step: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    if (step == 0) return error.InvalidShape;
    const end = @min(stop, input.rows);
    const begin = @min(start, end);
    if (begin >= end) return takeRows(DeviceDataFrame, input, &.{});

    const count = 1 + (end - 1 - begin) / step;
    const row_indices = try input.allocator.alloc(usize, count);
    defer input.allocator.free(row_indices);
    var row = begin;
    for (row_indices) |*slot| {
        slot.* = row;
        row += step;
    }
    return takeRows(DeviceDataFrame, input, row_indices);
}

pub fn reverseRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    const row_indices = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(row_indices);
    for (row_indices, 0..) |*slot, i| {
        slot.* = input.rows - 1 - i;
    }
    return takeRows(DeviceDataFrame, input, row_indices);
}

pub fn rollRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    shift: isize,
) DeviceFrameArrayError!DeviceDataFrame {
    if (input.rows == 0) return takeRows(DeviceDataFrame, input, &.{});
    const signed_rows = std.math.cast(isize, input.rows) orelse return error.InvalidShape;
    const normalized_shift: usize = @intCast(@mod(shift, signed_rows));
    const row_indices = try input.allocator.alloc(usize, input.rows);
    defer input.allocator.free(row_indices);
    // Match Array.roll(axis=0): a positive shift moves later rows toward the
    // front, so output row i reads source row i - shift modulo the row count.
    for (row_indices, 0..) |*slot, out_row| {
        slot.* = (out_row + input.rows - normalized_shift) % input.rows;
    }
    return takeRows(DeviceDataFrame, input, row_indices);
}

pub fn shiftRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    shift: isize,
) DeviceFrameArrayError!DeviceDataFrame {
    if (input.rows == 0) return takeOptionalRows(DeviceDataFrame, input, &.{});
    const signed_rows = std.math.cast(isize, input.rows) orelse return error.InvalidShape;
    const row_indices = try input.allocator.alloc(?usize, input.rows);
    defer input.allocator.free(row_indices);

    if (shift >= signed_rows or shift <= -signed_rows) {
        @memset(row_indices, null);
    } else if (shift > 0) {
        const offset: usize = @intCast(shift);
        for (row_indices, 0..) |*slot, out_row| {
            slot.* = if (out_row < offset) null else out_row - offset;
        }
    } else if (shift < 0) {
        const offset: usize = @intCast(-shift);
        for (row_indices, 0..) |*slot, out_row| {
            slot.* = if (out_row >= input.rows - offset) null else out_row + offset;
        }
    } else {
        for (row_indices, 0..) |*slot, out_row| slot.* = out_row;
    }
    return takeOptionalRows(DeviceDataFrame, input, row_indices);
}

pub fn filterRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    mask: []const bool,
) DeviceFrameArrayError!DeviceDataFrame {
    if (mask.len != input.rows) return error.LengthMismatch;
    const row_indices = try rowIndicesFromMask(input.allocator, mask);
    defer input.allocator.free(row_indices);
    return takeRows(DeviceDataFrame, input, row_indices);
}

pub fn whereIndicesColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    mask_name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    const mask_column = try input.column(mask_name);
    const typed_mask = switch (mask_column.*) {
        .bool => |typed| typed,
        else => return error.TypeMismatch,
    };
    if (!typed_mask.device().sameDevice(input.device)) return error.InvalidDevice;
    if (typed_mask.len() != input.rows) return error.LengthMismatch;
    const values = try typed_mask.values.toOwnedSlice(input.allocator);
    defer input.allocator.free(values);
    const validity = if (typed_mask.validity) |validity_array| try validity_array.toOwnedSlice(input.allocator) else null;
    defer if (validity) |validity_values| input.allocator.free(validity_values);
    var row_indices: std.ArrayList(usize) = .empty;
    defer row_indices.deinit(input.allocator);
    for (values, 0..) |value, row| {
        // Match dataframe filter semantics: null predicate rows behave as
        // false, so they do not appear in where-indices output.
        const selected = if (validity) |validity_values| validity_values[row] and value else value;
        if (selected) try row_indices.append(input.allocator, row);
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var column = try DeviceColumn.fromSlice(usize, input.allocator, row_indices.items, input.device);
    defer column.deinit();
    return DeviceDataFrame.init(input.allocator, &.{.{ .name = output_name, .data = column }});
}

pub fn toDevice(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    device_value: array_mod.Device,
) DeviceFrameArrayError!DeviceDataFrame {
    if (!device_value.isAvailable()) return error.InvalidDevice;
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, 0..) |col, i| {
        columns[i] = try col.to(device_value);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, input.names, columns, input.rows, device_value);
}
