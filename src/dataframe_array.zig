const std = @import("std");
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
