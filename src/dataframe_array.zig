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

fn columnHasSpecialFloat(column: anytype, allocator: std.mem.Allocator, comptime predicate: SpecialFloatPredicate) DeviceFrameArrayError!bool {
    return switch (column) {
        inline else => |typed| {
            const host_values = try typed.toOwnedSlice(allocator);
            defer allocator.free(host_values);
            const maybe_validity = try validityValues(typed, allocator);
            defer if (maybe_validity) |validity| allocator.free(validity);
            for (host_values, 0..) |value, row| {
                const valid = if (maybe_validity) |validity| validity[row] else true;
                if (valid and specialFloatPredicateMatches(@TypeOf(value), value, predicate)) return true;
            }
            return false;
        },
    };
}

fn selectColumnsBySpecialFloatPresence(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    wanted: bool,
    comptime predicate: SpecialFloatPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    var selected_names: std.ArrayList([]const u8) = .empty;
    defer selected_names.deinit(input.allocator);
    for (input.names, input.columns) |name, column| {
        if ((try columnHasSpecialFloat(column, input.allocator, predicate)) == wanted) try selected_names.append(input.allocator, name);
    }
    return select(DeviceDataFrame, input, selected_names.items);
}

fn dropColumnsBySpecialFloatPresence(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    wanted: bool,
    comptime predicate: SpecialFloatPredicate,
) DeviceFrameArrayError!DeviceDataFrame {
    var kept_names: std.ArrayList([]const u8) = .empty;
    defer kept_names.deinit(input.allocator);
    for (input.names, input.columns) |name, column| {
        if ((try columnHasSpecialFloat(column, input.allocator, predicate)) != wanted) try kept_names.append(input.allocator, name);
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
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .nan);
}

pub fn selectColumnsWithoutNaNs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .nan);
}

pub fn dropColumnsWithNaNs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .nan);
}

pub fn dropColumnsWithoutNaNs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .nan);
}

pub fn selectColumnsWithInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .inf);
}

pub fn selectColumnsWithoutInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .inf);
}

pub fn dropColumnsWithInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .inf);
}

pub fn dropColumnsWithoutInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .inf);
}

pub fn selectColumnsWithPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .positive_inf);
}

pub fn selectColumnsWithoutPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .positive_inf);
}

pub fn dropColumnsWithPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .positive_inf);
}

pub fn dropColumnsWithoutPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .positive_inf);
}

pub fn selectColumnsWithNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .negative_inf);
}

pub fn selectColumnsWithoutNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .negative_inf);
}

pub fn dropColumnsWithNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .negative_inf);
}

pub fn dropColumnsWithoutNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .negative_inf);
}

pub fn selectColumnsWithZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .zero);
}

pub fn selectColumnsWithoutZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .zero);
}

pub fn dropColumnsWithZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .zero);
}

pub fn dropColumnsWithoutZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .zero);
}

pub fn selectColumnsWithNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .non_zero);
}

pub fn selectColumnsWithoutNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .non_zero);
}

pub fn dropColumnsWithNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .non_zero);
}

pub fn dropColumnsWithoutNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .non_zero);
}

pub fn selectColumnsWithFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .finite);
}

pub fn selectColumnsWithoutFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .finite);
}

pub fn dropColumnsWithFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .finite);
}

pub fn dropColumnsWithoutFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .finite);
}

pub fn selectColumnsWithNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .normal);
}

pub fn selectColumnsWithoutNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .normal);
}

pub fn dropColumnsWithNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .normal);
}

pub fn dropColumnsWithoutNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .normal);
}

pub fn selectColumnsWithSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .subnormal);
}

pub fn selectColumnsWithoutSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .subnormal);
}

pub fn dropColumnsWithSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .subnormal);
}

pub fn dropColumnsWithoutSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .subnormal);
}

pub fn selectColumnsWithNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .non_finite);
}

pub fn selectColumnsWithoutNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return selectColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .non_finite);
}

pub fn dropColumnsWithNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, true, .non_finite);
}

pub fn dropColumnsWithoutNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropColumnsBySpecialFloatPresence(DeviceDataFrame, input, false, .non_finite);
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

const SpecialFloatPredicate = enum { nan, inf, positive_inf, negative_inf, zero, non_zero, finite, normal, subnormal, non_finite };

fn specialFloatPredicateMatches(comptime T: type, value: T, comptime predicate: SpecialFloatPredicate) bool {
    return switch (predicate) {
        .nan => isNanValue(T, value),
        .inf => isInfValue(T, value),
        .positive_inf => isPositiveInfValue(T, value),
        .negative_inf => isNegativeInfValue(T, value),
        .zero => isZeroValue(T, value),
        .non_zero => isNonZeroValue(T, value),
        .finite => isFiniteValue(T, value),
        .normal => isNormalValue(T, value),
        .subnormal => isSubnormalValue(T, value),
        .non_finite => !isFiniteValue(T, value),
    };
}

fn fillSpecialFloatsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: anytype,
    replacement: T,
    comptime predicate: SpecialFloatPredicate,
) array_mod.ArrayError!@TypeOf(column) {
    const ColumnType = @TypeOf(column);
    const values = try column.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    for (values, 0..) |*slot, row| {
        const valid = if (maybe_validity) |validity| validity[row] else true;
        if (valid and specialFloatPredicateMatches(T, slot.*, predicate)) slot.* = replacement;
    }

    if (maybe_validity) |validity| {
        return ColumnType.fromSliceWithValidity(allocator, values, validity, column.device());
    }
    return ColumnType.fromSlice(allocator, values, column.device());
}

fn fillSpecialFloatColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
    comptime predicate: SpecialFloatPredicate,
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
                try fillSpecialFloatsTyped(T, input.allocator, @field(source.*, @tagName(tag)), replacement, predicate),
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
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .nan);
}

pub fn fillInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .inf);
}

pub fn fillPositiveInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .positive_inf);
}

pub fn fillNegativeInfColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .negative_inf);
}

pub fn fillZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .zero);
}

pub fn fillNonZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .non_zero);
}

pub fn fillFiniteColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .finite);
}

pub fn fillNormalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .normal);
}

pub fn fillSubnormalColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .subnormal);
}

pub fn fillNonFiniteColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    scalar: DeviceScalar,
) DeviceFrameArrayError!DeviceDataFrame {
    return fillSpecialFloatColumn(DeviceDataFrame, input, name, scalar, .non_finite);
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

fn isNonZeroValue(comptime T: type, value: T) bool {
    if (comptime T == array_mod.BFloat16) return value.toF32() != 0;
    if (comptime T == array_mod.Complex64 or T == array_mod.Complex128) return value.re != 0 or value.im != 0;
    return switch (@typeInfo(T)) {
        .float, .comptime_float, .int, .comptime_int => value != 0,
        .bool => value,
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
    comptime predicate: enum { nan, inf, positive_inf, negative_inf, zero, non_zero, finite, normal, subnormal, non_finite },
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
                    .non_zero => isNonZeroValue(@TypeOf(value), value),
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

pub fn isNonZeroColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withNumericPredicateColumn(DeviceDataFrame, input, name, output_name, .non_zero);
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

const RowNumericPredicate = enum { nan, inf, positive_inf, negative_inf, zero, non_zero, finite, normal, subnormal, non_finite };

fn rowNumericPredicateMatches(comptime T: type, value: T, comptime predicate: RowNumericPredicate) bool {
    return switch (predicate) {
        .nan => isNanValue(T, value),
        .inf => isInfValue(T, value),
        .positive_inf => isPositiveInfValue(T, value),
        .negative_inf => isNegativeInfValue(T, value),
        .zero => isZeroValue(T, value),
        .non_zero => isNonZeroValue(T, value),
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

pub fn withRowNonZeroCount(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    output_name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return withRowNumericPredicateCount(DeviceDataFrame, input, names, output_name, .non_zero);
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

fn dropSpecialFloats(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
    comptime predicate: SpecialFloatPredicate,
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
                    if (valid and specialFloatPredicateMatches(@TypeOf(value), value, predicate)) slot.* = false;
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
    return dropSpecialFloats(DeviceDataFrame, input, names, .nan);
}

pub fn dropInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropSpecialFloats(DeviceDataFrame, input, names, .inf);
}

pub fn dropPositiveInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropSpecialFloats(DeviceDataFrame, input, names, .positive_inf);
}

pub fn dropNegativeInfs(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropSpecialFloats(DeviceDataFrame, input, names, .negative_inf);
}

pub fn dropZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropSpecialFloats(DeviceDataFrame, input, names, .zero);
}

pub fn dropNonZeros(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropSpecialFloats(DeviceDataFrame, input, names, .non_zero);
}

pub fn dropFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropSpecialFloats(DeviceDataFrame, input, names, .finite);
}

pub fn dropNormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropSpecialFloats(DeviceDataFrame, input, names, .normal);
}

pub fn dropSubnormals(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropSpecialFloats(DeviceDataFrame, input, names, .subnormal);
}

pub fn dropNonFinites(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return dropSpecialFloats(DeviceDataFrame, input, names, .non_finite);
}

fn filterSpecialFloatsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    comptime predicate: SpecialFloatPredicate,
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
                slot.* = valid and specialFloatPredicateMatches(@TypeOf(value), value, predicate);
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
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .nan);
}

pub fn filterInfsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .inf);
}

pub fn filterPositiveInfsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .positive_inf);
}

pub fn filterNegativeInfsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .negative_inf);
}

pub fn filterZerosColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .zero);
}

pub fn filterNonZerosColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .non_zero);
}

pub fn filterFinitesColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .finite);
}

pub fn filterNormalsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .normal);
}

pub fn filterSubnormalsColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .subnormal);
}

pub fn filterNonFinitesColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    return filterSpecialFloatsColumn(DeviceDataFrame, input, name, .non_finite);
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
