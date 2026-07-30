//! Lazy expression operation builders for DeviceLazyFrame.

const std = @import("std");
const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub fn select(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try frame.allocator.alloc([]const u8, names.len);
    errdefer frame.allocator.free(owned);
    var initialized: usize = 0;
    errdefer {
        for (owned[0..initialized]) |name| frame.allocator.free(name);
    }
    for (names, owned) |name, *slot| {
        slot.* = try frame.allocator.dupe(u8, name);
        initialized += 1;
    }
    try frame.ops.append(frame.allocator, .{ .select = owned });
}

pub fn selectByNamePrefix(frame: anytype, prefix: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, prefix);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .select_name_prefix = .{ .pattern = owned } });
}

pub fn selectByNameSuffix(frame: anytype, suffix: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, suffix);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .select_name_suffix = .{ .pattern = owned } });
}

pub fn selectByNameContains(frame: anytype, needle: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, needle);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .select_name_contains = .{ .pattern = owned } });
}

pub fn selectByDTypes(frame: anytype, dtypes: []const array_mod.DType) DeviceDataError!void {
    try frame.ops.append(frame.allocator, .{ .select_dtypes = try frame.allocator.dupe(array_mod.DType, dtypes) });
}

pub fn selectByDTypeClass(frame: anytype, class: options_mod.DeviceDTypeClass) DeviceDataError!void {
    try frame.ops.append(frame.allocator, .{ .select_dtype_class = class });
}

pub fn selectNumeric(frame: anytype) DeviceDataError!void {
    return selectByDTypeClass(frame, .numeric);
}

pub fn selectReal(frame: anytype) DeviceDataError!void {
    return selectByDTypeClass(frame, .real);
}

pub fn selectFloat(frame: anytype) DeviceDataError!void {
    return selectByDTypeClass(frame, .float);
}

pub fn selectInteger(frame: anytype) DeviceDataError!void {
    return selectByDTypeClass(frame, .integer);
}

pub fn selectBool(frame: anytype) DeviceDataError!void {
    return selectByDTypeClass(frame, .bool);
}

pub fn renameColumn(frame: anytype, old_name: []const u8, new_name: []const u8) DeviceDataError!void {
    const owned_old = try frame.allocator.dupe(u8, old_name);
    errdefer frame.allocator.free(owned_old);
    const owned_new = try frame.allocator.dupe(u8, new_name);
    errdefer frame.allocator.free(owned_new);
    try frame.ops.append(frame.allocator, .{ .rename_column = .{
        .old_name = owned_old,
        .new_name = owned_new,
    } });
}

pub fn dropColumns(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try frame.allocator.alloc([]const u8, names.len);
    errdefer frame.allocator.free(owned);
    var initialized: usize = 0;
    errdefer {
        for (owned[0..initialized]) |name| frame.allocator.free(name);
    }
    for (names, owned) |name, *slot| {
        slot.* = try frame.allocator.dupe(u8, name);
        initialized += 1;
    }
    try frame.ops.append(frame.allocator, .{ .drop_columns = owned });
}

pub fn dropColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropColumns(frame, &.{name});
}

pub fn dropNulls(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try frame.allocator.alloc([]const u8, names.len);
    errdefer frame.allocator.free(owned);
    var initialized: usize = 0;
    errdefer {
        for (owned[0..initialized]) |name| frame.allocator.free(name);
    }
    for (names, owned) |name, *slot| {
        slot.* = try frame.allocator.dupe(u8, name);
        initialized += 1;
    }
    try frame.ops.append(frame.allocator, .{ .drop_nulls = owned });
}

pub fn dropNullsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropNulls(frame, &.{name});
}

pub fn withRowIndex(frame: anytype, name: []const u8, offset: usize) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .with_row_index = .{
        .name = owned_name,
        .offset = offset,
    } });
}

pub fn filter(frame: anytype, mask: anytype) DeviceDataError!void {
    try frame.ops.append(frame.allocator, .{ .filter_mask = try mask.clone() });
}

pub fn filterColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_column = owned_name });
}

pub fn withColumnBinary(frame: anytype, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnBinaryOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    try frame.ops.append(frame.allocator, .{ .with_column_binary = .{
        .name = owned_name,
        .lhs_name = owned_lhs,
        .rhs_name = owned_rhs,
        .op = op,
    } });
}

pub fn withColumnScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .op = op,
        .scalar = DeviceScalar.init(T, scalar),
    } });
}

pub fn withColumnLiteral(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .with_column_literal = .{
        .name = owned_name,
        .scalar = DeviceScalar.init(T, value),
    } });
}

pub fn withColumnLiteralScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .with_column_literal = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn castColumn(frame: anytype, name: []const u8, dtype_value: array_mod.DType) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .cast_column = .{
        .name = owned_name,
        .dtype = dtype_value,
    } });
}

pub fn fillNullColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillNullColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillNullColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_null_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn withColumnCompare(frame: anytype, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    try frame.ops.append(frame.allocator, .{ .with_column_compare = .{
        .name = owned_name,
        .lhs_name = owned_lhs,
        .rhs_name = owned_rhs,
        .op = op,
    } });
}

pub fn withColumnCompareScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_compare_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .op = op,
        .scalar = DeviceScalar.init(T, scalar),
    } });
}

pub fn filterColumnScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_scalar = .{
        .name = owned_name,
        .op = op,
        .scalar = DeviceScalar.init(T, scalar),
    } });
}
