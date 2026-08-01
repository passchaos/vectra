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
pub const takeOptionalRows = array_helpers_mod.takeOptionalRows;
pub const columnsRowsEqual = array_helpers_mod.columnsRowsEqual;
pub const columnsRowsEqualTyped = array_helpers_mod.columnsRowsEqualTyped;
const nameInBorrowedList = names_mod.nameInBorrowedList;
const DeviceScalar = options_mod.DeviceScalar;
const validityValues = validity_core_mod.validityValues;

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

const RowWeightedDispersion = enum { variance, stddev };

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
        const denominator = weight_sum - correction;
        var centered_square_sum = weighted_square_sum - weighted_sum * weighted_sum / weight_sum;
        if (centered_square_sum < 0.0 and centered_square_sum > -1e-12) centered_square_sum = 0.0;
        const variance = if (denominator <= 0.0) quietNanF64() else centered_square_sum / denominator;
        value.* = switch (reduction) {
            .variance => variance,
            .stddev => std.math.sqrt(variance),
        };
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

const RowWeightedPairReduction = enum { covariance, correlation, beta };

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
    const validity = try input.allocator.alloc(bool, input.rows);
    defer input.allocator.free(validity);
    @memset(weight_sums, 0.0);
    @memset(lhs_sums, 0.0);
    @memset(rhs_sums, 0.0);
    @memset(lhs_square_sums, 0.0);
    @memset(rhs_square_sums, 0.0);
    @memset(cross_sums, 0.0);
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
            weight_sums[row] += weight;
            lhs_sums[row] += weight * lhs;
            rhs_sums[row] += weight * rhs;
            lhs_square_sums[row] += weight * lhs * lhs;
            rhs_square_sums[row] += weight * rhs * rhs;
            cross_sums[row] += weight * lhs * rhs;
        }
    }

    const values = try input.allocator.alloc(f64, input.rows);
    defer input.allocator.free(values);
    for (values, validity, weight_sums, lhs_sums, rhs_sums, lhs_square_sums, rhs_square_sums, cross_sums) |*value, *valid, weight_sum, lhs_sum, rhs_sum, lhs_square_sum, rhs_square_sum, cross_sum| {
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
        value.* = switch (reduction) {
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
        var total_weight: f64 = 0.0;
        for (0..flat.width) |col_index| {
            const offset = row * flat.width + col_index;
            if (flat.validity[offset]) total_weight += flat.weights[offset];
        }
        if (!(total_weight > 0.0)) continue;

        var entropy: f64 = 0.0;
        var sum_prob_sq: f64 = 0.0;
        var distinct_count: usize = 0;
        for (0..flat.width) |col_index| {
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
            for (col_index..flat.width) |candidate_index| {
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

pub fn withRowWeightedEvenness(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowWeightedDistributionReduction(DeviceDataFrame, input, value_names, weight_names, output_name, .evenness);
}

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

const RowNumericReduction = enum { sum, prod, mean, geometric_mean, harmonic_mean, min, max, ptp, midrange, range_coeff, mean_abs, hhi, magnitude_inverse_simpson, magnitude_dominance, magnitude_dominance_margin, magnitude_entropy, magnitude_perplexity, magnitude_evenness, rms, l1_norm, l2_norm };

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

fn withRowNumericReduction(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: RowNumericReduction,
) DeviceFrameArrayError!DeviceDataFrame {
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
                        .hhi, .magnitude_inverse_simpson => {
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
        } else if (reduction == .geometric_mean) {
            if (!std.math.isNan(value.*)) {
                value.* = if (aux_value != 0.0) 0.0 else std.math.exp(value.* / @as(f64, @floatFromInt(count)));
            }
        } else if (reduction == .harmonic_mean) {
            value.* = if (std.math.isInf(value.*)) 0.0 else @as(f64, @floatFromInt(count)) / value.*;
        } else if (reduction == .mean_abs) {
            value.* /= @floatFromInt(count);
        } else if (reduction == .hhi) {
            value.* = if (value.* == 0.0) std.math.nan(f64) else aux_value / (value.* * value.*);
        } else if (reduction == .magnitude_inverse_simpson) {
            value.* = if (value.* == 0.0 or aux_value == 0.0) std.math.nan(f64) else (value.* * value.*) / aux_value;
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
        } else if (reduction == .ptp) {
            value.* = aux_value - value.*;
        } else if (reduction == .midrange) {
            value.* = (value.* + aux_value) / 2.0;
        } else if (reduction == .range_coeff) {
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

pub fn withRowMidrange(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericReduction(DeviceDataFrame, input, names, output_name, .midrange);
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

const RowNumericDispersion = enum { variance, stddev, sem, cv, fano, skewness, kurtosis };

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
                    const value = realValueAsF64(@TypeOf(raw_value), raw_value);
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
            .variance => variance,
            .stddev => stddev_value,
            .sem => stddev_value / std.math.sqrt(@as(f64, @floatFromInt(count))),
            .cv => stddev_value / mean,
            .fano => if (mean == 0.0) std.math.nan(f64) else variance / mean,
            .skewness => if (count < 2 or m2 == 0.0) std.math.nan(f64) else std.math.sqrt(@as(f64, @floatFromInt(count))) * m3 / std.math.pow(f64, m2, 1.5),
            .kurtosis => if (count < 2 or m2 == 0.0) std.math.nan(f64) else @as(f64, @floatFromInt(count)) * m4 / (m2 * m2) - 3.0,
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

pub fn withRowStddev(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .stddev);
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

pub fn withRowCv(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericDispersion(DeviceDataFrame, input, names, output_name, correction, .cv);
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
