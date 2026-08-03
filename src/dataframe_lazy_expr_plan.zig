//! Lazy expression operation builders for DeviceLazyFrame.

const std = @import("std");
const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

fn cloneNameList(allocator: std.mem.Allocator, names: []const []const u8) std.mem.Allocator.Error![][]const u8 {
    const owned = try allocator.alloc([]const u8, names.len);
    errdefer allocator.free(owned);
    var initialized: usize = 0;
    errdefer {
        for (owned[0..initialized]) |name| allocator.free(name);
    }
    for (names, owned) |name, *slot| {
        slot.* = try allocator.dupe(u8, name);
        initialized += 1;
    }
    return owned;
}

fn freeNameList(allocator: std.mem.Allocator, names: []const []const u8) void {
    for (names) |name| allocator.free(name);
    allocator.free(names);
}

pub fn select(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
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

pub fn selectByNameGlob(frame: anytype, pattern: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, pattern);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .select_name_glob = .{ .pattern = owned } });
}

pub fn dropByNamePrefix(frame: anytype, prefix: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, prefix);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .drop_name_prefix = .{ .pattern = owned } });
}

pub fn dropByNameSuffix(frame: anytype, suffix: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, suffix);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .drop_name_suffix = .{ .pattern = owned } });
}

pub fn dropByNameContains(frame: anytype, needle: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, needle);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .drop_name_contains = .{ .pattern = owned } });
}

pub fn dropByNameGlob(frame: anytype, pattern: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, pattern);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .drop_name_glob = .{ .pattern = owned } });
}

pub fn selectByDTypes(frame: anytype, dtypes: []const array_mod.DType) DeviceDataError!void {
    try frame.ops.append(frame.allocator, .{ .select_dtypes = try frame.allocator.dupe(array_mod.DType, dtypes) });
}

pub fn selectByDTypeClass(frame: anytype, class: options_mod.DeviceDTypeClass) DeviceDataError!void {
    try frame.ops.append(frame.allocator, .{ .select_dtype_class = class });
}

pub fn dropByDTypes(frame: anytype, dtypes: []const array_mod.DType) DeviceDataError!void {
    try frame.ops.append(frame.allocator, .{ .drop_dtypes = try frame.allocator.dupe(array_mod.DType, dtypes) });
}

pub fn dropByDTypeClass(frame: anytype, class: options_mod.DeviceDTypeClass) DeviceDataError!void {
    try frame.ops.append(frame.allocator, .{ .drop_dtype_class = class });
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

pub fn dropNumeric(frame: anytype) DeviceDataError!void {
    return dropByDTypeClass(frame, .numeric);
}

pub fn dropReal(frame: anytype) DeviceDataError!void {
    return dropByDTypeClass(frame, .real);
}

pub fn dropFloat(frame: anytype) DeviceDataError!void {
    return dropByDTypeClass(frame, .float);
}

pub fn dropInteger(frame: anytype) DeviceDataError!void {
    return dropByDTypeClass(frame, .integer);
}

pub fn dropBool(frame: anytype) DeviceDataError!void {
    return dropByDTypeClass(frame, .bool);
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

pub fn renameColumns(frame: anytype, old_names: []const []const u8, new_names: []const []const u8) DeviceDataError!void {
    if (old_names.len != new_names.len) return error.LengthMismatch;
    const owned_old = try cloneNameList(frame.allocator, old_names);
    errdefer {
        for (owned_old) |name| frame.allocator.free(name);
        frame.allocator.free(owned_old);
    }
    const owned_new = try cloneNameList(frame.allocator, new_names);
    errdefer {
        for (owned_new) |name| frame.allocator.free(name);
        frame.allocator.free(owned_new);
    }
    try frame.ops.append(frame.allocator, .{ .rename_columns = .{
        .old_names = owned_old,
        .new_names = owned_new,
    } });
}

pub fn addColumnNamePrefix(frame: anytype, prefix: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, prefix);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .add_column_name_prefix = .{ .pattern = owned } });
}

pub fn addColumnNameSuffix(frame: anytype, suffix: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, suffix);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .add_column_name_suffix = .{ .pattern = owned } });
}

pub fn stripColumnNamePrefix(frame: anytype, prefix: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, prefix);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .strip_column_name_prefix = .{ .pattern = owned } });
}

pub const removeColumnNamePrefix = stripColumnNamePrefix;

pub fn stripColumnNameSuffix(frame: anytype, suffix: []const u8) DeviceDataError!void {
    const owned = try frame.allocator.dupe(u8, suffix);
    errdefer frame.allocator.free(owned);
    try frame.ops.append(frame.allocator, .{ .strip_column_name_suffix = .{ .pattern = owned } });
}

pub const removeColumnNameSuffix = stripColumnNameSuffix;

pub fn replaceColumnNamePrefix(frame: anytype, old_prefix: []const u8, new_prefix: []const u8) DeviceDataError!void {
    const owned_old = try frame.allocator.dupe(u8, old_prefix);
    errdefer frame.allocator.free(owned_old);
    const owned_new = try frame.allocator.dupe(u8, new_prefix);
    errdefer frame.allocator.free(owned_new);
    try frame.ops.append(frame.allocator, .{ .replace_column_name_prefix = .{
        .old_pattern = owned_old,
        .new_pattern = owned_new,
    } });
}

pub fn replaceColumnNameSuffix(frame: anytype, old_suffix: []const u8, new_suffix: []const u8) DeviceDataError!void {
    const owned_old = try frame.allocator.dupe(u8, old_suffix);
    errdefer frame.allocator.free(owned_old);
    const owned_new = try frame.allocator.dupe(u8, new_suffix);
    errdefer frame.allocator.free(owned_new);
    try frame.ops.append(frame.allocator, .{ .replace_column_name_suffix = .{
        .old_pattern = owned_old,
        .new_pattern = owned_new,
    } });
}

pub fn moveColumn(frame: anytype, name: []const u8, target_index: usize) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .move_column = .{
        .name = owned_name,
        .target_index = target_index,
    } });
}

pub fn moveColumnBefore(frame: anytype, name: []const u8, before_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_anchor = try frame.allocator.dupe(u8, before_name);
    errdefer frame.allocator.free(owned_anchor);
    try frame.ops.append(frame.allocator, .{ .move_column_before = .{
        .name = owned_name,
        .anchor_name = owned_anchor,
    } });
}

pub fn moveColumnAfter(frame: anytype, name: []const u8, after_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_anchor = try frame.allocator.dupe(u8, after_name);
    errdefer frame.allocator.free(owned_anchor);
    try frame.ops.append(frame.allocator, .{ .move_column_after = .{
        .name = owned_name,
        .anchor_name = owned_anchor,
    } });
}

pub fn copyColumn(frame: anytype, source_name: []const u8, new_name: []const u8) DeviceDataError!void {
    const owned_source = try frame.allocator.dupe(u8, source_name);
    errdefer frame.allocator.free(owned_source);
    const owned_new = try frame.allocator.dupe(u8, new_name);
    errdefer frame.allocator.free(owned_new);
    try frame.ops.append(frame.allocator, .{ .copy_column = .{
        .source_name = owned_source,
        .new_name = owned_new,
    } });
}

pub fn copyColumnAt(frame: anytype, source_name: []const u8, new_name: []const u8, target_index: usize) DeviceDataError!void {
    const owned_source = try frame.allocator.dupe(u8, source_name);
    errdefer frame.allocator.free(owned_source);
    const owned_new = try frame.allocator.dupe(u8, new_name);
    errdefer frame.allocator.free(owned_new);
    try frame.ops.append(frame.allocator, .{ .copy_column_at = .{
        .source_name = owned_source,
        .new_name = owned_new,
        .target_index = target_index,
    } });
}

pub fn copyColumnBefore(frame: anytype, source_name: []const u8, new_name: []const u8, before_name: []const u8) DeviceDataError!void {
    const owned_source = try frame.allocator.dupe(u8, source_name);
    errdefer frame.allocator.free(owned_source);
    const owned_new = try frame.allocator.dupe(u8, new_name);
    errdefer frame.allocator.free(owned_new);
    const owned_anchor = try frame.allocator.dupe(u8, before_name);
    errdefer frame.allocator.free(owned_anchor);
    try frame.ops.append(frame.allocator, .{ .copy_column_before = .{
        .source_name = owned_source,
        .new_name = owned_new,
        .anchor_name = owned_anchor,
    } });
}

pub fn copyColumnAfter(frame: anytype, source_name: []const u8, new_name: []const u8, after_name: []const u8) DeviceDataError!void {
    const owned_source = try frame.allocator.dupe(u8, source_name);
    errdefer frame.allocator.free(owned_source);
    const owned_new = try frame.allocator.dupe(u8, new_name);
    errdefer frame.allocator.free(owned_new);
    const owned_anchor = try frame.allocator.dupe(u8, after_name);
    errdefer frame.allocator.free(owned_anchor);
    try frame.ops.append(frame.allocator, .{ .copy_column_after = .{
        .source_name = owned_source,
        .new_name = owned_new,
        .anchor_name = owned_anchor,
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

pub fn dropAllNulls(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    try frame.ops.append(frame.allocator, .{ .drop_all_nulls = owned });
}

pub fn dropAllNullsOn(frame: anytype, names: []const []const u8) DeviceDataError!void {
    return dropAllNulls(frame, names);
}

pub fn filterAllNulls(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    try frame.ops.append(frame.allocator, .{ .filter_all_nulls = owned });
}

pub fn filterAllNullsOn(frame: anytype, names: []const []const u8) DeviceDataError!void {
    return filterAllNulls(frame, names);
}

pub fn filterNullsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_nulls_column = owned_name });
}

pub fn dropNaNs(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_nans = owned });
}

pub fn dropNaNsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropNaNs(frame, &.{name});
}

pub fn filterNaNsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_nans_column = owned_name });
}

pub fn dropInfs(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_infs = owned });
}

pub fn dropInfsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropInfs(frame, &.{name});
}

pub fn filterInfsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_infs_column = owned_name });
}

pub fn dropPositiveInfs(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_positive_infs = owned });
}

pub fn dropPositiveInfsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropPositiveInfs(frame, &.{name});
}

pub fn filterPositiveInfsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_positive_infs_column = owned_name });
}

pub fn dropNegativeInfs(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_negative_infs = owned });
}

pub fn dropNegativeInfsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropNegativeInfs(frame, &.{name});
}

pub fn filterNegativeInfsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_negative_infs_column = owned_name });
}

pub fn dropZeros(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_zeros = owned });
}

pub fn dropZerosColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropZeros(frame, &.{name});
}

pub fn filterZerosColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_zeros_column = owned_name });
}

pub fn dropPositiveZeros(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_positive_zeros = owned });
}

pub fn dropPositiveZerosColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropPositiveZeros(frame, &.{name});
}

pub fn filterPositiveZerosColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_positive_zeros_column = owned_name });
}

pub fn dropNegativeZeros(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_negative_zeros = owned });
}

pub fn dropNegativeZerosColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropNegativeZeros(frame, &.{name});
}

pub fn filterNegativeZerosColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_negative_zeros_column = owned_name });
}

pub fn dropNonZeros(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_non_zeros = owned });
}

pub fn dropNonZerosColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropNonZeros(frame, &.{name});
}

pub fn filterNonZerosColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_non_zeros_column = owned_name });
}

pub fn dropPositives(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_positives = owned });
}

pub fn dropPositivesColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropPositives(frame, &.{name});
}

pub fn filterPositivesColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_positives_column = owned_name });
}

pub fn dropSignBits(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_signbits = owned });
}

pub fn dropSignBitsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropSignBits(frame, &.{name});
}

pub fn filterSignBitsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_signbits_column = owned_name });
}

pub fn dropNegatives(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_negatives = owned });
}

pub fn dropNegativesColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropNegatives(frame, &.{name});
}

pub fn filterNegativesColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_negatives_column = owned_name });
}

pub fn dropFinites(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_finites = owned });
}

pub fn dropFinitesColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropFinites(frame, &.{name});
}

pub fn filterFinitesColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_finites_column = owned_name });
}

pub fn dropNormals(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_normals = owned });
}

pub fn dropNormalsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropNormals(frame, &.{name});
}

pub fn filterNormalsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_normals_column = owned_name });
}

pub fn dropSubnormals(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_subnormals = owned });
}

pub fn dropSubnormalsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropSubnormals(frame, &.{name});
}

pub fn filterSubnormalsColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_subnormals_column = owned_name });
}

pub fn dropNonFinites(frame: anytype, names: []const []const u8) DeviceDataError!void {
    const owned = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned) |name| frame.allocator.free(name);
        frame.allocator.free(owned);
    }
    try frame.ops.append(frame.allocator, .{ .drop_non_finites = owned });
}

pub fn dropNonFinitesColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return dropNonFinites(frame, &.{name});
}

pub fn filterNonFinitesColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_non_finites_column = owned_name });
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

fn filterIsInColumnMode(frame: anytype, input_name: []const u8, test_name: []const u8, invert: bool) DeviceDataError!void {
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_test = try frame.allocator.dupe(u8, test_name);
    errdefer frame.allocator.free(owned_test);
    try frame.ops.append(frame.allocator, .{ .filter_isin_column = .{
        .input_name = owned_input,
        .test_name = owned_test,
        .invert = invert,
    } });
}

fn filterIsInValuesMode(frame: anytype, input_name: []const u8, values: anytype, invert: bool) DeviceDataError!void {
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    var owned_values = try values.clone();
    errdefer owned_values.deinit();
    try frame.ops.append(frame.allocator, .{ .filter_isin_values = .{
        .input_name = owned_input,
        .values = owned_values,
        .invert = invert,
    } });
}

pub fn filterIsInColumn(frame: anytype, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
    return filterIsInColumnMode(frame, input_name, test_name, false);
}

pub fn filterNotInColumn(frame: anytype, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
    return filterIsInColumnMode(frame, input_name, test_name, true);
}

pub const filterIsinColumn = filterIsInColumn;
pub const filterIsInColumnInverted = filterNotInColumn;
pub const filterIsinColumnInverted = filterNotInColumn;

pub fn filterIsInValuesColumn(frame: anytype, input_name: []const u8, values: anytype) DeviceDataError!void {
    return filterIsInValuesMode(frame, input_name, values, false);
}

pub fn filterNotInValuesColumn(frame: anytype, input_name: []const u8, values: anytype) DeviceDataError!void {
    return filterIsInValuesMode(frame, input_name, values, true);
}

pub fn dropIsInColumn(frame: anytype, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
    return filterNotInColumn(frame, input_name, test_name);
}

pub fn dropNotInColumn(frame: anytype, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
    return filterIsInColumn(frame, input_name, test_name);
}

pub const dropIsinColumn = dropIsInColumn;
pub const dropIsInColumnInverted = dropNotInColumn;
pub const dropIsinColumnInverted = dropNotInColumn;

pub fn dropIsInValuesColumn(frame: anytype, input_name: []const u8, values: anytype) DeviceDataError!void {
    return filterNotInValuesColumn(frame, input_name, values);
}

pub fn dropNotInValuesColumn(frame: anytype, input_name: []const u8, values: anytype) DeviceDataError!void {
    return filterIsInValuesColumn(frame, input_name, values);
}

pub fn filterBetweenColumnWithDeviceScalars(frame: anytype, name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_between_column = .{
        .name = owned_name,
        .lower = lower,
        .upper = upper,
        .lower_inclusive = lower_inclusive,
        .upper_inclusive = upper_inclusive,
        .keep_inside = true,
    } });
}

pub fn filterBetweenColumnClosed(frame: anytype, name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
    return filterBetweenColumnWithDeviceScalars(frame, name, DeviceScalar.init(T, lower), DeviceScalar.init(T, upper), lower_inclusive, upper_inclusive);
}

pub fn filterBetweenColumn(frame: anytype, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return filterBetweenColumnClosed(frame, name, T, lower, upper, true, true);
}

pub fn filterOutsideColumnWithDeviceScalars(frame: anytype, name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_between_column = .{
        .name = owned_name,
        .lower = lower,
        .upper = upper,
        .lower_inclusive = lower_inclusive,
        .upper_inclusive = upper_inclusive,
        .keep_inside = false,
    } });
}

pub fn filterOutsideColumnClosed(frame: anytype, name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
    return filterOutsideColumnWithDeviceScalars(frame, name, DeviceScalar.init(T, lower), DeviceScalar.init(T, upper), lower_inclusive, upper_inclusive);
}

pub fn filterOutsideColumn(frame: anytype, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return filterOutsideColumnClosed(frame, name, T, lower, upper, true, true);
}

pub fn dropBetweenColumn(frame: anytype, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return filterOutsideColumn(frame, name, T, lower, upper);
}

pub fn dropOutsideColumn(frame: anytype, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return filterBetweenColumn(frame, name, T, lower, upper);
}

pub fn dropRowsByColumnMask(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .drop_rows_by_mask_column = owned_name });
}

pub fn whereIndicesColumn(frame: anytype, mask_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, mask_name);
    errdefer frame.allocator.free(owned_name);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .where_indices_column = .{
        .name = owned_name,
        .output_name = owned_output,
    } });
}

pub const argwhereColumn = whereIndicesColumn;

pub fn withColumnAbs(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_abs = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnNeg(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_neg = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnNegative(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    return withColumnNeg(frame, name, input_name);
}

pub fn withColumnSquare(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_square = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnReciprocal(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_reciprocal = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnSign(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_sign = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnSqrt(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_sqrt = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnRsqrt(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_rsqrt = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnCbrt(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_cbrt = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnFloor(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_floor = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnCeil(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_ceil = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnRound(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_round = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnTrunc(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_trunc = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnDeg2rad(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_deg2rad = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnRad2deg(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_rad2deg = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnExpit(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_expit = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnLogit(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_logit = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnSoftplus(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_softplus = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnLogsigmoid(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_logsigmoid = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnRelu(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_relu = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnLeakyRelu(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, negative_slope: T) DeviceDataError!void {
    return withColumnLeakyReluWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, negative_slope));
}

pub fn withColumnLeakyReluWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, negative_slope: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_leaky_relu = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = negative_slope,
    } });
}

pub fn withColumnRelu6(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_relu6 = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnPowScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, exponent: T) DeviceDataError!void {
    return withColumnPowWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, exponent));
}

pub fn withColumnPowWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, exponent: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_pow_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = exponent,
    } });
}

pub fn withColumnFloorDivScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnFloorDivWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnFloorDivWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_floor_div_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnModScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnModWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnModWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_mod_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnRemainderScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnRemainderWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnRemainderWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_remainder_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnLogAddExpScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnLogAddExpWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnLogAddExpWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_log_add_exp_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnLogAddExp2Scalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnLogAddExp2WithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnLogAddExp2WithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_log_add_exp2_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnXlogyScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnXlogyWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnXlogyWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_xlogy_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnFmaxScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnFmaxWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnFmaxWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_fmax_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnFminScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnFminWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnFminWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_fmin_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnHypotScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnHypotWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnHypotWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_hypot_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnAtan2Scalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnAtan2WithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnAtan2WithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_atan2_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnNextAfterScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnNextAfterWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnNextAfterWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_next_after_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnCopysignScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnCopysignWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnCopysignWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_copysign_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnHeavisideScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, value_at_zero: T) DeviceDataError!void {
    return withColumnHeavisideWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, value_at_zero));
}

pub fn withColumnHeavisideWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, value_at_zero: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_heaviside_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = value_at_zero,
    } });
}

pub fn withColumnLdexpScalar(frame: anytype, name: []const u8, input_name: []const u8, exponent: i32) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_ldexp_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .exponent = exponent,
    } });
}

pub fn withColumnThreshold(
    frame: anytype,
    name: []const u8,
    input_name: []const u8,
    comptime T: type,
    threshold_value: T,
    replacement_value: T,
) DeviceDataError!void {
    return withColumnThresholdWithDeviceScalars(frame, name, input_name, DeviceScalar.init(T, threshold_value), DeviceScalar.init(T, replacement_value));
}

pub fn withColumnThresholdWithDeviceScalars(
    frame: anytype,
    name: []const u8,
    input_name: []const u8,
    threshold_value: DeviceScalar,
    replacement_value: DeviceScalar,
) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_threshold = .{
        .name = owned_name,
        .input_name = owned_input,
        .lhs_scalar = threshold_value,
        .rhs_scalar = replacement_value,
    } });
}

pub fn withColumnHardtanh(
    frame: anytype,
    name: []const u8,
    input_name: []const u8,
    comptime T: type,
    min_value: T,
    max_value: T,
) DeviceDataError!void {
    return withColumnHardtanhWithDeviceScalars(frame, name, input_name, DeviceScalar.init(T, min_value), DeviceScalar.init(T, max_value));
}

pub fn withColumnHardtanhWithDeviceScalars(
    frame: anytype,
    name: []const u8,
    input_name: []const u8,
    min_value: DeviceScalar,
    max_value: DeviceScalar,
) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_hardtanh = .{
        .name = owned_name,
        .input_name = owned_input,
        .lhs_scalar = min_value,
        .rhs_scalar = max_value,
    } });
}

pub fn withColumnBetween(
    frame: anytype,
    name: []const u8,
    input_name: []const u8,
    comptime T: type,
    lower: T,
    upper: T,
) DeviceDataError!void {
    return withColumnBetweenWithDeviceScalars(frame, name, input_name, DeviceScalar.init(T, lower), DeviceScalar.init(T, upper), true, true);
}

pub fn withColumnIsBetween(
    frame: anytype,
    name: []const u8,
    input_name: []const u8,
    comptime T: type,
    lower: T,
    upper: T,
) DeviceDataError!void {
    return withColumnBetween(frame, name, input_name, T, lower, upper);
}

pub fn withColumnBetweenClosed(
    frame: anytype,
    name: []const u8,
    input_name: []const u8,
    comptime T: type,
    lower: T,
    upper: T,
    lower_inclusive: bool,
    upper_inclusive: bool,
) DeviceDataError!void {
    return withColumnBetweenWithDeviceScalars(frame, name, input_name, DeviceScalar.init(T, lower), DeviceScalar.init(T, upper), lower_inclusive, upper_inclusive);
}

pub fn withColumnBetweenExclusive(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return withColumnBetweenClosed(frame, name, input_name, T, lower, upper, false, false);
}

pub fn withColumnBetweenLeftClosed(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return withColumnBetweenClosed(frame, name, input_name, T, lower, upper, true, false);
}

pub fn withColumnBetweenRightClosed(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return withColumnBetweenClosed(frame, name, input_name, T, lower, upper, false, true);
}

pub fn withColumnBetweenWithDeviceScalars(
    frame: anytype,
    name: []const u8,
    input_name: []const u8,
    lower: DeviceScalar,
    upper: DeviceScalar,
    lower_inclusive: bool,
    upper_inclusive: bool,
) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_between = .{
        .name = owned_name,
        .input_name = owned_input,
        .lower = lower,
        .upper = upper,
        .lower_inclusive = lower_inclusive,
        .upper_inclusive = upper_inclusive,
    } });
}

pub fn withColumnNotBetween(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    try withColumnBetween(frame, name, input_name, T, lower, upper);
    return withColumnLogicalNot(frame, name, name);
}

pub fn withColumnOutside(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return withColumnNotBetween(frame, name, input_name, T, lower, upper);
}

pub fn withColumnNotBetweenClosed(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
    try withColumnBetweenClosed(frame, name, input_name, T, lower, upper, lower_inclusive, upper_inclusive);
    return withColumnLogicalNot(frame, name, name);
}

pub fn withColumnNotBetweenExclusive(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return withColumnNotBetweenClosed(frame, name, input_name, T, lower, upper, false, false);
}

pub fn withColumnNotBetweenLeftClosed(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return withColumnNotBetweenClosed(frame, name, input_name, T, lower, upper, true, false);
}

pub fn withColumnNotBetweenRightClosed(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
    return withColumnNotBetweenClosed(frame, name, input_name, T, lower, upper, false, true);
}

pub fn withColumnNotBetweenWithDeviceScalars(frame: anytype, name: []const u8, input_name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
    try withColumnBetweenWithDeviceScalars(frame, name, input_name, lower, upper, lower_inclusive, upper_inclusive);
    return withColumnLogicalNot(frame, name, name);
}

pub fn withColumnMaximumScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnMaximumWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnMaximumWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_maximum_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnMinimumScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
    return withColumnMinimumWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, scalar));
}

pub fn withColumnMinimumWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_minimum_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
    } });
}

pub fn withColumnClipMin(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, min_value: T) DeviceDataError!void {
    return withColumnClipMinWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, min_value));
}

pub fn withColumnClipMinWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, min_value: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_clip_min = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = min_value,
    } });
}

pub fn withColumnClipMax(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, max_value: T) DeviceDataError!void {
    return withColumnClipMaxWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, max_value));
}

pub fn withColumnClipMaxWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, max_value: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_clip_max = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = max_value,
    } });
}

pub fn withColumnHardshrink(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lambd: T) DeviceDataError!void {
    return withColumnHardshrinkWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, lambd));
}

pub fn withColumnHardshrinkWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, lambd: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_hardshrink = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = lambd,
    } });
}

pub fn withColumnSoftshrink(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, lambd: T) DeviceDataError!void {
    return withColumnSoftshrinkWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, lambd));
}

pub fn withColumnSoftshrinkWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, lambd: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_softshrink = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = lambd,
    } });
}

pub fn withColumnTanhshrink(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_tanhshrink = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnElu(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, alpha: T) DeviceDataError!void {
    return withColumnEluWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, alpha));
}

pub fn withColumnEluWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, alpha: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_elu = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = alpha,
    } });
}

pub fn withColumnCelu(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, alpha: T) DeviceDataError!void {
    return withColumnCeluWithDeviceScalar(frame, name, input_name, DeviceScalar.init(T, alpha));
}

pub fn withColumnCeluWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, alpha: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_celu = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = alpha,
    } });
}

pub fn withColumnSoftsign(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_softsign = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnHardsigmoid(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_hardsigmoid = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnHardswish(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_hardswish = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnSilu(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_silu = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnSwish(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_swish = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnMish(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_mish = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnGelu(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_gelu = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnSelu(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_selu = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnExp(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_exp = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnLog(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_log = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnLog1p(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_log1p = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnLgamma(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_lgamma = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnSinc(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_sinc = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnLog2(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_log2 = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnLog10(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_log10 = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnExp2(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_exp2 = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnExpm1(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_expm1 = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnSin(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_sin = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnCos(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_cos = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnTan(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_tan = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnAsin(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_asin = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnAcos(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_acos = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnAtan(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_atan = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnSinh(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_sinh = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnCosh(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_cosh = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnTanh(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_tanh = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnAsinh(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_asinh = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnAcosh(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_acosh = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
}

pub fn withColumnAtanh(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_atanh = .{
        .name = owned_name,
        .input_name = owned_input,
    } });
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

pub fn withColumnLerpScalar(frame: anytype, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, comptime T: type, weight: T) DeviceDataError!void {
    return withColumnLerpWithDeviceScalar(frame, name, lhs_name, rhs_name, DeviceScalar.init(T, weight));
}

pub fn withColumnLerpWithDeviceScalar(frame: anytype, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    try frame.ops.append(frame.allocator, .{ .with_column_lerp_scalar = .{
        .name = owned_name,
        .lhs_name = owned_lhs,
        .rhs_name = owned_rhs,
        .scalar = weight,
    } });
}

fn appendTernaryParamOp(
    frame: anytype,
    name: []const u8,
    base_name: []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    scalar: DeviceScalar,
    comptime tag: enum { addcmul, addcdiv },
) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_base = try frame.allocator.dupe(u8, base_name);
    errdefer frame.allocator.free(owned_base);
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    try frame.ops.append(frame.allocator, switch (tag) {
        .addcmul => .{ .with_column_addcmul_scalar = .{
            .name = owned_name,
            .base_name = owned_base,
            .lhs_name = owned_lhs,
            .rhs_name = owned_rhs,
            .scalar = scalar,
        } },
        .addcdiv => .{ .with_column_addcdiv_scalar = .{
            .name = owned_name,
            .base_name = owned_base,
            .lhs_name = owned_lhs,
            .rhs_name = owned_rhs,
            .scalar = scalar,
        } },
    });
}

pub fn withColumnAddcmulScalar(frame: anytype, name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnAddcmulWithDeviceScalar(frame, name, base_name, input1_name, input2_name, DeviceScalar.init(T, value));
}

pub fn withColumnAddcmulWithDeviceScalar(frame: anytype, name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!void {
    return appendTernaryParamOp(frame, name, base_name, input1_name, input2_name, value, .addcmul);
}

pub fn withColumnAddcdivScalar(frame: anytype, name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnAddcdivWithDeviceScalar(frame, name, base_name, input1_name, input2_name, DeviceScalar.init(T, value));
}

pub fn withColumnAddcdivWithDeviceScalar(frame: anytype, name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!void {
    return appendTernaryParamOp(frame, name, base_name, input1_name, input2_name, value, .addcdiv);
}

pub fn withColumnClipArray(frame: anytype, name: []const u8, input_name: []const u8, min_name: []const u8, max_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_min = try frame.allocator.dupe(u8, min_name);
    errdefer frame.allocator.free(owned_min);
    const owned_max = try frame.allocator.dupe(u8, max_name);
    errdefer frame.allocator.free(owned_max);
    try frame.ops.append(frame.allocator, .{ .with_column_clip_array = .{
        .name = owned_name,
        .input_name = owned_input,
        .lhs_name = owned_min,
        .rhs_name = owned_max,
    } });
}

pub fn withColumnWhereScalar(frame: anytype, name: []const u8, input_name: []const u8, mask_name: []const u8, comptime T: type, other_value: T) DeviceDataError!void {
    return withColumnWhereWithDeviceScalar(frame, name, input_name, mask_name, DeviceScalar.init(T, other_value));
}

pub fn withColumnWhereWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, mask_name: []const u8, other_value: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_mask = try frame.allocator.dupe(u8, mask_name);
    errdefer frame.allocator.free(owned_mask);
    try frame.ops.append(frame.allocator, .{ .with_column_where_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .mask_name = owned_mask,
        .scalar = other_value,
    } });
}

pub fn withColumnWhere(frame: anytype, name: []const u8, input_name: []const u8, mask_name: []const u8, other_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_mask = try frame.allocator.dupe(u8, mask_name);
    errdefer frame.allocator.free(owned_mask);
    const owned_other = try frame.allocator.dupe(u8, other_name);
    errdefer frame.allocator.free(owned_other);
    try frame.ops.append(frame.allocator, .{ .with_column_where = .{
        .name = owned_name,
        .input_name = owned_input,
        .lhs_name = owned_mask,
        .rhs_name = owned_other,
    } });
}

pub fn withColumnIsIn(frame: anytype, name: []const u8, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
    return withColumnIsInMode(frame, name, input_name, test_name, false);
}

pub fn withColumnIsInInverted(frame: anytype, name: []const u8, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
    return withColumnIsInMode(frame, name, input_name, test_name, true);
}

pub const withColumnIsin = withColumnIsIn;
pub const withColumnIsinInverted = withColumnIsInInverted;

fn withColumnIsInMode(frame: anytype, name: []const u8, input_name: []const u8, test_name: []const u8, invert: bool) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_test = try frame.allocator.dupe(u8, test_name);
    errdefer frame.allocator.free(owned_test);
    try frame.ops.append(frame.allocator, .{ .with_column_isin = .{
        .name = owned_name,
        .input_name = owned_input,
        .test_name = owned_test,
        .invert = invert,
    } });
}

fn withColumnIsInValuesMode(frame: anytype, name: []const u8, input_name: []const u8, values: anytype, invert: bool) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    var owned_values = try values.clone();
    errdefer owned_values.deinit();
    try frame.ops.append(frame.allocator, .{ .with_column_isin_values = .{
        .name = owned_name,
        .input_name = owned_input,
        .values = owned_values,
        .invert = invert,
    } });
}

pub fn withColumnIsInValuesColumn(frame: anytype, name: []const u8, input_name: []const u8, values: anytype) DeviceDataError!void {
    return withColumnIsInValuesMode(frame, name, input_name, values, false);
}

pub fn withColumnIsInValuesInvertedColumn(frame: anytype, name: []const u8, input_name: []const u8, values: anytype) DeviceDataError!void {
    return withColumnIsInValuesMode(frame, name, input_name, values, true);
}

pub fn withColumnMaskedPutScalar(frame: anytype, name: []const u8, input_name: []const u8, mask_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnMaskedPutWithDeviceScalar(frame, name, input_name, mask_name, DeviceScalar.init(T, value));
}

pub fn withColumnMaskedPutWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, mask_name: []const u8, value: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_mask = try frame.allocator.dupe(u8, mask_name);
    errdefer frame.allocator.free(owned_mask);
    try frame.ops.append(frame.allocator, .{ .with_column_masked_put_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .mask_name = owned_mask,
        .scalar = value,
    } });
}

pub fn withColumnPutMaskScalar(frame: anytype, name: []const u8, input_name: []const u8, mask_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnMaskedPutScalar(frame, name, input_name, mask_name, T, value);
}

pub fn withColumnPutMaskWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, mask_name: []const u8, value: DeviceScalar) DeviceDataError!void {
    return withColumnMaskedPutWithDeviceScalar(frame, name, input_name, mask_name, value);
}

pub fn withColumnPutFlatScalar(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T) DeviceDataError!void {
    return withColumnPutFlatWithDeviceScalar(frame, name, input_name, row_indices, DeviceScalar.init(T, value));
}

pub fn withColumnPutFlatWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const usize, value: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_indices = try frame.allocator.dupe(usize, row_indices);
    errdefer frame.allocator.free(owned_indices);
    try frame.ops.append(frame.allocator, .{ .with_column_put_flat_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .row_indices = owned_indices,
        .scalar = value,
    } });
}

pub fn withColumnPutFlat(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const usize, value_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_indices = try frame.allocator.dupe(usize, row_indices);
    errdefer frame.allocator.free(owned_indices);
    const owned_values = try frame.allocator.dupe(u8, value_name);
    errdefer frame.allocator.free(owned_values);
    try frame.ops.append(frame.allocator, .{ .with_column_put_flat = .{
        .name = owned_name,
        .input_name = owned_input,
        .row_indices = owned_indices,
        .value_name = owned_values,
    } });
}

pub fn withColumnIndexPut(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const usize, value_name: []const u8) DeviceDataError!void {
    return withColumnPutFlat(frame, name, input_name, row_indices, value_name);
}

pub fn withColumnPutFlatScalarMode(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T, mode: array_mod.IndexMode) DeviceDataError!void {
    return withColumnPutFlatModeWithDeviceScalar(frame, name, input_name, row_indices, DeviceScalar.init(T, value), mode);
}

pub fn withColumnPutFlatModeWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const usize, value: DeviceScalar, mode: array_mod.IndexMode) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_indices = try frame.allocator.dupe(usize, row_indices);
    errdefer frame.allocator.free(owned_indices);
    try frame.ops.append(frame.allocator, .{ .with_column_put_flat_scalar_mode = .{
        .name = owned_name,
        .input_name = owned_input,
        .row_indices = owned_indices,
        .scalar = value,
        .mode = mode,
    } });
}

pub fn withColumnIndexPutScalar(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T) DeviceDataError!void {
    return withColumnPutFlatScalar(frame, name, input_name, row_indices, T, value);
}

pub fn withColumnIndexPutWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const usize, value: DeviceScalar) DeviceDataError!void {
    return withColumnPutFlatWithDeviceScalar(frame, name, input_name, row_indices, value);
}

pub fn withColumnPutFlatScalarSigned(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const isize, comptime T: type, value: T) DeviceDataError!void {
    return withColumnPutFlatSignedWithDeviceScalar(frame, name, input_name, row_indices, DeviceScalar.init(T, value));
}

pub fn withColumnPutFlatSignedWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const isize, value: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    const owned_indices = try frame.allocator.dupe(isize, row_indices);
    errdefer frame.allocator.free(owned_indices);
    try frame.ops.append(frame.allocator, .{ .with_column_put_flat_scalar_signed = .{
        .name = owned_name,
        .input_name = owned_input,
        .row_indices = owned_indices,
        .scalar = value,
    } });
}

pub fn withColumnIndexPutScalarSigned(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const isize, comptime T: type, value: T) DeviceDataError!void {
    return withColumnPutFlatScalarSigned(frame, name, input_name, row_indices, T, value);
}

pub fn withColumnIndexPutSignedWithDeviceScalar(frame: anytype, name: []const u8, input_name: []const u8, row_indices: []const isize, value: DeviceScalar) DeviceDataError!void {
    return withColumnPutFlatSignedWithDeviceScalar(frame, name, input_name, row_indices, value);
}

pub fn withColumnIscloseScalar(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T) DeviceDataError!void {
    return withColumnIscloseWithDeviceScalarsEqualNan(frame, name, input_name, DeviceScalar.init(T, scalar), DeviceScalar.init(T, rtol), DeviceScalar.init(T, atol), false);
}

pub fn withColumnIscloseScalarEqualNan(frame: anytype, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T, equal_nan: bool) DeviceDataError!void {
    return withColumnIscloseWithDeviceScalarsEqualNan(frame, name, input_name, DeviceScalar.init(T, scalar), DeviceScalar.init(T, rtol), DeviceScalar.init(T, atol), equal_nan);
}

pub fn withColumnIscloseWithDeviceScalars(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar) DeviceDataError!void {
    return withColumnIscloseWithDeviceScalarsEqualNan(frame, name, input_name, scalar, rtol, atol, false);
}

pub fn withColumnIscloseWithDeviceScalarsEqualNan(frame: anytype, name: []const u8, input_name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar, equal_nan: bool) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_isclose_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .scalar = scalar,
        .rtol = rtol,
        .atol = atol,
        .equal_nan = equal_nan,
    } });
}

pub fn withColumnLogicalScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: bool, op: DeviceColumnLogicalOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_input = try frame.allocator.dupe(u8, input_name);
    errdefer frame.allocator.free(owned_input);
    try frame.ops.append(frame.allocator, .{ .with_column_logical_scalar = .{
        .name = owned_name,
        .input_name = owned_input,
        .op = op,
        .scalar = scalar,
    } });
}

pub fn withColumnLogicalAndScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: bool) DeviceDataError!void {
    return withColumnLogicalScalar(frame, name, input_name, scalar, .@"and");
}

pub fn withColumnLogicalOrScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: bool) DeviceDataError!void {
    return withColumnLogicalScalar(frame, name, input_name, scalar, .@"or");
}

pub fn withColumnLogicalXorScalar(frame: anytype, name: []const u8, input_name: []const u8, scalar: bool) DeviceDataError!void {
    return withColumnLogicalScalar(frame, name, input_name, scalar, .xor);
}

pub fn withColumnLogicalNot(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    return withColumnLogicalXorScalar(frame, name, input_name, true);
}

pub fn withColumnNot(frame: anytype, name: []const u8, input_name: []const u8) DeviceDataError!void {
    return withColumnLogicalNot(frame, name, input_name);
}

pub fn withColumnLogical(frame: anytype, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnLogicalOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    try frame.ops.append(frame.allocator, .{ .with_column_logical = .{
        .name = owned_name,
        .lhs_name = owned_lhs,
        .rhs_name = owned_rhs,
        .op = op,
    } });
}

pub fn withColumnLogicalAnd(frame: anytype, name: []const u8, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!void {
    return withColumnLogical(frame, name, lhs_name, rhs_name, .@"and");
}

pub fn withColumnLogicalOr(frame: anytype, name: []const u8, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!void {
    return withColumnLogical(frame, name, lhs_name, rhs_name, .@"or");
}

pub fn withColumnLogicalXor(frame: anytype, name: []const u8, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!void {
    return withColumnLogical(frame, name, lhs_name, rhs_name, .xor);
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

pub fn withColumnLiteralAt(frame: anytype, name: []const u8, comptime T: type, value: T, target_index: usize) DeviceDataError!void {
    return withColumnLiteralScalarAt(frame, name, DeviceScalar.init(T, value), target_index);
}

pub fn withColumnLiteralScalarAt(frame: anytype, name: []const u8, scalar: DeviceScalar, target_index: usize) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .with_column_literal_at = .{
        .name = owned_name,
        .scalar = scalar,
        .target_index = target_index,
    } });
}

pub fn withColumnLiteralBefore(frame: anytype, name: []const u8, comptime T: type, value: T, before_name: []const u8) DeviceDataError!void {
    return withColumnLiteralScalarBefore(frame, name, DeviceScalar.init(T, value), before_name);
}

pub fn withColumnLiteralScalarBefore(frame: anytype, name: []const u8, scalar: DeviceScalar, before_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_anchor = try frame.allocator.dupe(u8, before_name);
    errdefer frame.allocator.free(owned_anchor);
    try frame.ops.append(frame.allocator, .{ .with_column_literal_before = .{
        .name = owned_name,
        .scalar = scalar,
        .anchor_name = owned_anchor,
    } });
}

pub fn withColumnLiteralAfter(frame: anytype, name: []const u8, comptime T: type, value: T, after_name: []const u8) DeviceDataError!void {
    return withColumnLiteralScalarAfter(frame, name, DeviceScalar.init(T, value), after_name);
}

pub fn withColumnLiteralScalarAfter(frame: anytype, name: []const u8, scalar: DeviceScalar, after_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_anchor = try frame.allocator.dupe(u8, after_name);
    errdefer frame.allocator.free(owned_anchor);
    try frame.ops.append(frame.allocator, .{ .with_column_literal_after = .{
        .name = owned_name,
        .scalar = scalar,
        .anchor_name = owned_anchor,
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

pub fn withColumnFillNull(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillNullScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNullScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNullColumnWithScalar(frame, output_name, scalar);
}

fn fillNullDirectionalColumn(frame: anytype, name: []const u8, comptime direction: enum { forward, backward }) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    switch (direction) {
        .forward => try frame.ops.append(frame.allocator, .{ .fill_null_forward_column = owned_name }),
        .backward => try frame.ops.append(frame.allocator, .{ .fill_null_backward_column = owned_name }),
    }
}

pub fn fillNullForwardColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return fillNullDirectionalColumn(frame, name, .forward);
}

pub fn fillNullBackwardColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    return fillNullDirectionalColumn(frame, name, .backward);
}

pub fn withColumnFillNullForward(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNullForwardColumn(frame, output_name);
}

pub fn withColumnFillNullBackward(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNullBackwardColumn(frame, output_name);
}

pub fn nullIfColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return nullIfColumnScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn nullIfColumnScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn nullIfValuesColumnWithDeviceColumn(frame: anytype, name: []const u8, values: anytype) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    var owned_values = try values.clone();
    errdefer owned_values.deinit();
    try frame.ops.append(frame.allocator, .{ .null_if_values_column = .{
        .name = owned_name,
        .values = owned_values,
    } });
}

pub fn withColumnNullIf(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnNullIfScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnNullIfScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfColumnScalar(frame, output_name, scalar);
}

pub fn withColumnNullIfValuesWithDeviceColumn(frame: anytype, output_name: []const u8, input_name: []const u8, values: anytype) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfValuesColumnWithDeviceColumn(frame, output_name, values);
}

pub fn nullIfNaNColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_nan_column = owned_name });
}

pub fn withColumnNullIfNaN(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfNaNColumn(frame, output_name);
}

pub fn nullIfInfColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_inf_column = owned_name });
}

pub fn withColumnNullIfInf(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfInfColumn(frame, output_name);
}

pub fn nullIfPositiveInfColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_positive_inf_column = owned_name });
}

pub fn withColumnNullIfPositiveInf(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfPositiveInfColumn(frame, output_name);
}

pub fn nullIfNegativeInfColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_negative_inf_column = owned_name });
}

pub fn withColumnNullIfNegativeInf(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfNegativeInfColumn(frame, output_name);
}

pub fn nullIfZeroColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_zero_column = owned_name });
}

pub fn withColumnNullIfZero(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfZeroColumn(frame, output_name);
}

pub fn nullIfPositiveZeroColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_positive_zero_column = owned_name });
}

pub fn withColumnNullIfPositiveZero(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfPositiveZeroColumn(frame, output_name);
}

pub fn nullIfNegativeZeroColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_negative_zero_column = owned_name });
}

pub fn withColumnNullIfNegativeZero(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfNegativeZeroColumn(frame, output_name);
}

pub fn nullIfNonZeroColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_non_zero_column = owned_name });
}

pub fn withColumnNullIfNonZero(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfNonZeroColumn(frame, output_name);
}

pub fn nullIfPositiveColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_positive_column = owned_name });
}

pub fn withColumnNullIfPositive(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfPositiveColumn(frame, output_name);
}

pub fn nullIfSignBitColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_signbit_column = owned_name });
}

pub fn withColumnNullIfSignBit(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfSignBitColumn(frame, output_name);
}

pub fn nullIfNegativeColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_negative_column = owned_name });
}

pub fn withColumnNullIfNegative(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfNegativeColumn(frame, output_name);
}

pub fn nullIfFiniteColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_finite_column = owned_name });
}

pub fn withColumnNullIfFinite(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfFiniteColumn(frame, output_name);
}

pub fn nullIfNormalColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_normal_column = owned_name });
}

pub fn withColumnNullIfNormal(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfNormalColumn(frame, output_name);
}

pub fn nullIfSubnormalColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_subnormal_column = owned_name });
}

pub fn withColumnNullIfSubnormal(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfSubnormalColumn(frame, output_name);
}

pub fn nullIfNonFiniteColumn(frame: anytype, name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .null_if_non_finite_column = owned_name });
}

pub fn withColumnNullIfNonFinite(frame: anytype, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return nullIfNonFiniteColumn(frame, output_name);
}

pub fn withColumnFillNaN(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillNaNScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNaNScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNaNColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillInf(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillInfScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillInfScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillInfColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillPositiveInf(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillPositiveInfScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillPositiveInfScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillPositiveInfColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillNegativeInf(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillNegativeInfScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNegativeInfScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNegativeInfColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillZero(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillZeroScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillZeroScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillZeroColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillPositiveZero(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillPositiveZeroScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillPositiveZeroScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillPositiveZeroColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillNegativeZero(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillNegativeZeroScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNegativeZeroScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNegativeZeroColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillNonZero(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillNonZeroScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNonZeroScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNonZeroColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillPositive(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillPositiveScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillPositiveScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillPositiveColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillSignBit(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillSignBitScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillSignBitScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillSignBitColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillNegative(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillNegativeScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNegativeScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNegativeColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillFinite(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillFiniteScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillFiniteScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillFiniteColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillNormal(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillNormalScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNormalScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNormalColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillSubnormal(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillSubnormalScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillSubnormalScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillSubnormalColumnWithScalar(frame, output_name, scalar);
}

pub fn withColumnFillNonFinite(frame: anytype, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return withColumnFillNonFiniteScalar(frame, output_name, input_name, DeviceScalar.init(T, value));
}

pub fn withColumnFillNonFiniteScalar(frame: anytype, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    if (!std.mem.eql(u8, output_name, input_name)) {
        try copyColumn(frame, input_name, output_name);
    }
    return fillNonFiniteColumnWithScalar(frame, output_name, scalar);
}

pub fn fillNaNColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillNaNColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillNaNColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_nan_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillInfColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillInfColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillInfColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_inf_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillPositiveInfColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillPositiveInfColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillPositiveInfColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_positive_inf_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillNegativeInfColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillNegativeInfColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillNegativeInfColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_negative_inf_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillZeroColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillZeroColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillZeroColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_zero_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillPositiveZeroColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillPositiveZeroColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillPositiveZeroColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_positive_zero_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillNegativeZeroColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillNegativeZeroColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillNegativeZeroColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_negative_zero_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillNonZeroColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillNonZeroColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillNonZeroColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_non_zero_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillPositiveColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillPositiveColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillPositiveColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_positive_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillSignBitColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillSignBitColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillSignBitColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_signbit_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillNegativeColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillNegativeColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillNegativeColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_negative_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillFiniteColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillFiniteColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillFiniteColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_finite_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillNormalColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillNormalColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillNormalColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_normal_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillSubnormalColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillSubnormalColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillSubnormalColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_subnormal_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn fillNonFiniteColumn(frame: anytype, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
    return fillNonFiniteColumnWithScalar(frame, name, DeviceScalar.init(T, value));
}

pub fn fillNonFiniteColumnWithScalar(frame: anytype, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .fill_non_finite_column = .{
        .name = owned_name,
        .scalar = scalar,
    } });
}

pub fn coalesceColumns(frame: anytype, primary_name: []const u8, fallback_name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_primary = try frame.allocator.dupe(u8, primary_name);
    errdefer frame.allocator.free(owned_primary);
    const owned_fallback = try frame.allocator.dupe(u8, fallback_name);
    errdefer frame.allocator.free(owned_fallback);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .coalesce_columns = .{
        .primary_name = owned_primary,
        .fallback_name = owned_fallback,
        .output_name = owned_output,
    } });
}

pub fn coalesceColumnsMany(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    if (names.len == 0) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .coalesce_columns_many = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn coalesceManyColumns(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return coalesceColumnsMany(frame, names, output_name);
}

pub fn coalesceFirstValidColumns(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return coalesceColumnsMany(frame, names, output_name);
}

pub fn isNullColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .is_null_column = .{
        .name = owned_name,
        .output_name = owned_output,
    } });
}

pub fn isValidColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .is_valid_column = .{
        .name = owned_name,
        .output_name = owned_output,
    } });
}

fn numericPredicateColumn(frame: anytype, name: []const u8, output_name: []const u8, comptime predicate: enum { nan, zero, positive_zero, negative_zero, non_zero, positive, signbit, negative, finite, normal, subnormal, non_finite, inf, positive_inf, negative_inf }) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (predicate) {
        .nan => try frame.ops.append(frame.allocator, .{ .is_nan_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .zero => try frame.ops.append(frame.allocator, .{ .is_zero_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .positive_zero => try frame.ops.append(frame.allocator, .{ .is_positive_zero_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .negative_zero => try frame.ops.append(frame.allocator, .{ .is_negative_zero_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .non_zero => try frame.ops.append(frame.allocator, .{ .is_non_zero_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .positive => try frame.ops.append(frame.allocator, .{ .is_positive_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .signbit => try frame.ops.append(frame.allocator, .{ .is_signbit_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .negative => try frame.ops.append(frame.allocator, .{ .is_negative_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .finite => try frame.ops.append(frame.allocator, .{ .is_finite_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .normal => try frame.ops.append(frame.allocator, .{ .is_normal_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .subnormal => try frame.ops.append(frame.allocator, .{ .is_subnormal_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .non_finite => try frame.ops.append(frame.allocator, .{ .is_non_finite_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .inf => try frame.ops.append(frame.allocator, .{ .is_inf_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .positive_inf => try frame.ops.append(frame.allocator, .{ .is_positive_inf_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
        .negative_inf => try frame.ops.append(frame.allocator, .{ .is_negative_inf_column = .{
            .name = owned_name,
            .output_name = owned_output,
        } }),
    }
}

pub fn isNanColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .nan);
}

pub fn isZeroColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .zero);
}

pub fn isPositiveZeroColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .positive_zero);
}

pub fn isNegativeZeroColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .negative_zero);
}

pub fn isNonZeroColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .non_zero);
}

pub fn isPositiveColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .positive);
}

pub fn isSignBitColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .signbit);
}

pub fn isNegativeColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .negative);
}

pub fn isFiniteColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .finite);
}

pub fn isNormalColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .normal);
}

pub fn isSubnormalColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .subnormal);
}

pub fn isNonFiniteColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .non_finite);
}

pub fn isInfColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .inf);
}

pub fn isPositiveInfColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .positive_inf);
}

pub fn isNegativeInfColumn(frame: anytype, name: []const u8, output_name: []const u8) DeviceDataError!void {
    return numericPredicateColumn(frame, name, output_name, .negative_inf);
}

fn withRowValidityCount(frame: anytype, names: []const []const u8, output_name: []const u8, comptime count_valid: bool) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    if (count_valid) {
        try frame.ops.append(frame.allocator, .{ .row_valid_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_null_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } });
    }
}

pub fn withRowNullCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityCount(frame, names, output_name, false);
}

pub fn withRowValidCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityCount(frame, names, output_name, true);
}

fn withRowValidityReduction(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: enum { any_null, all_null, any_valid, all_valid },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (reduction) {
        .any_null => try frame.ops.append(frame.allocator, .{ .row_any_null = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .all_null => try frame.ops.append(frame.allocator, .{ .row_all_null = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .any_valid => try frame.ops.append(frame.allocator, .{ .row_any_valid = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .all_valid => try frame.ops.append(frame.allocator, .{ .row_all_valid = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowAnyNull(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityReduction(frame, names, output_name, .any_null);
}

pub fn withRowAllNull(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityReduction(frame, names, output_name, .all_null);
}

pub fn withRowAnyValid(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityReduction(frame, names, output_name, .any_valid);
}

pub fn withRowAllValid(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityReduction(frame, names, output_name, .all_valid);
}

fn withRowCumulativeValidityReduction(
    frame: anytype,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: enum { any_null, all_null, any_valid, all_valid },
) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (reduction) {
        .any_null => try frame.ops.append(frame.allocator, .{ .row_cumulative_any_null = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .all_null => try frame.ops.append(frame.allocator, .{ .row_cumulative_all_null = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .any_valid => try frame.ops.append(frame.allocator, .{ .row_cumulative_any_valid = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .all_valid => try frame.ops.append(frame.allocator, .{ .row_cumulative_all_valid = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
    }
}

pub fn withRowCumulativeAnyNull(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityReduction(frame, names, output_names, .any_null);
}

pub fn withRowCumAnyNull(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNull(frame, names, output_names);
}

pub fn withRowPrefixAnyNull(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNull(frame, names, output_names);
}

pub fn withRowCumulativeAllNull(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityReduction(frame, names, output_names, .all_null);
}

pub fn withRowCumAllNull(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNull(frame, names, output_names);
}

pub fn withRowPrefixAllNull(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNull(frame, names, output_names);
}

pub fn withRowCumulativeAnyValid(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityReduction(frame, names, output_names, .any_valid);
}

pub fn withRowCumAnyValid(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyValid(frame, names, output_names);
}

pub fn withRowPrefixAnyValid(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyValid(frame, names, output_names);
}

pub fn withRowCumulativeAllValid(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityReduction(frame, names, output_names, .all_valid);
}

pub fn withRowCumAllValid(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllValid(frame, names, output_names);
}

pub fn withRowPrefixAllValid(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllValid(frame, names, output_names);
}

fn withRowCumulativeValidityCount(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime count_valid: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (count_valid) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_valid_count = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_null_count = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    }
}

pub fn withRowCumulativeNullCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityCount(frame, names, output_names, false);
}

pub fn withRowCumNullCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNullCount(frame, names, output_names);
}

pub fn withRowPrefixNullCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNullCount(frame, names, output_names);
}

pub fn withRowCumulativeValidCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityCount(frame, names, output_names, true);
}

pub fn withRowCumValidCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidCount(frame, names, output_names);
}

pub fn withRowPrefixValidCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidCount(frame, names, output_names);
}

fn withRowCumulativeValidityRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime count_valid: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (count_valid) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_valid_ratio = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_null_ratio = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    }
}

pub fn withRowCumulativeNullRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityRatio(frame, names, output_names, false);
}

pub fn withRowCumNullRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNullRatio(frame, names, output_names);
}

pub fn withRowPrefixNullRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNullRatio(frame, names, output_names);
}

pub fn withRowCumulativeValidRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityRatio(frame, names, output_names, true);
}

pub fn withRowCumValidRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidRatio(frame, names, output_names);
}

pub fn withRowPrefixValidRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidRatio(frame, names, output_names);
}

fn withRowValidityRatio(frame: anytype, names: []const []const u8, output_name: []const u8, comptime count_valid: bool) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    if (count_valid) {
        try frame.ops.append(frame.allocator, .{ .row_valid_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_null_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } });
    }
}

pub fn withRowNullRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityRatio(frame, names, output_name, false);
}

pub fn withRowValidRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityRatio(frame, names, output_name, true);
}

fn withRowValidityMatchIndex(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime search: enum { first_valid, last_valid, first_null, last_null },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (search) {
        .first_valid => try frame.ops.append(frame.allocator, .{ .row_first_valid_index = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .last_valid => try frame.ops.append(frame.allocator, .{ .row_last_valid_index = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .first_null => try frame.ops.append(frame.allocator, .{ .row_first_null_index = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .last_null => try frame.ops.append(frame.allocator, .{ .row_last_null_index = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowFirstValidIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityMatchIndex(frame, names, output_name, .first_valid);
}

pub fn withRowLastValidIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityMatchIndex(frame, names, output_name, .last_valid);
}

pub fn withRowFirstNullIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityMatchIndex(frame, names, output_name, .first_null);
}

pub fn withRowLastNullIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowValidityMatchIndex(frame, names, output_name, .last_null);
}

fn withRowCumulativeValidityMatchIndex(
    frame: anytype,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime search: enum { first_valid, last_valid, first_null, last_null },
) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (search) {
        .first_valid => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_valid_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_valid => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_valid_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_null => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_null_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_null => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_null_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
    }
}

pub fn withRowCumulativeFirstValidIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityMatchIndex(frame, names, output_names, .first_valid);
}

pub fn withRowPrefixFirstValidIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstValidIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastValidIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityMatchIndex(frame, names, output_names, .last_valid);
}

pub fn withRowPrefixLastValidIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastValidIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstNullIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityMatchIndex(frame, names, output_names, .first_null);
}

pub fn withRowPrefixFirstNullIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstNullIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastNullIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeValidityMatchIndex(frame, names, output_names, .last_null);
}

pub fn withRowPrefixLastNullIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastNullIndex(frame, names, output_names);
}

pub fn withRowPairCount(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, lhs_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, rhs_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_pair_count = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

fn withRowWeightedPairSupport(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, comptime reduction: enum { weight_sum, positive_count, effective_n }) DeviceDataError!void {
    const owned_lhs = try cloneNameList(frame.allocator, lhs_names);
    errdefer freeNameList(frame.allocator, owned_lhs);
    const owned_rhs = try cloneNameList(frame.allocator, rhs_names);
    errdefer freeNameList(frame.allocator, owned_rhs);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .weight_sum => .{ .row_weighted_pair_weight_sum = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_name = owned_output, .correction = 0.0 } },
        .positive_count => .{ .row_weighted_pair_positive_count = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_name = owned_output, .correction = 0.0 } },
        .effective_n => .{ .row_weighted_pair_effective_n = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_name = owned_output, .correction = 0.0 } },
    });
}

pub fn withRowWeightedPairWeightSum(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPairSupport(frame, lhs_names, rhs_names, weight_names, output_name, .weight_sum);
}

pub fn withRowWeightedPairPositiveCount(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPairSupport(frame, lhs_names, rhs_names, weight_names, output_name, .positive_count);
}

pub fn withRowWeightedPairEffectiveN(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPairSupport(frame, lhs_names, rhs_names, weight_names, output_name, .effective_n);
}

pub const withRowWeightedPairEffectiveCount = withRowWeightedPairEffectiveN;

fn withRowCumulativeWeightedPairSupport(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime reduction: enum { weight_sum, positive_count, effective_n }) DeviceDataError!void {
    const owned_lhs = try cloneNameList(frame.allocator, lhs_names);
    errdefer freeNameList(frame.allocator, owned_lhs);
    const owned_rhs = try cloneNameList(frame.allocator, rhs_names);
    errdefer freeNameList(frame.allocator, owned_rhs);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .weight_sum => .{ .row_cumulative_weighted_pair_weight_sum = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_names = owned_outputs } },
        .positive_count => .{ .row_cumulative_weighted_pair_positive_count = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_names = owned_outputs } },
        .effective_n => .{ .row_cumulative_weighted_pair_effective_n = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedPairWeightSum(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPairSupport(frame, lhs_names, rhs_names, weight_names, output_names, .weight_sum);
}

pub fn withRowCumulativeWeightedPairPositiveCount(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPairSupport(frame, lhs_names, rhs_names, weight_names, output_names, .positive_count);
}

pub fn withRowCumulativeWeightedPairEffectiveN(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPairSupport(frame, lhs_names, rhs_names, weight_names, output_names, .effective_n);
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

const RowCumulativeWeightedPairMetric = enum { dot, cosine, squared_euclidean, euclidean, manhattan, chebyshev };

fn withRowCumulativeWeightedPairMetric(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime metric: RowCumulativeWeightedPairMetric) DeviceDataError!void {
    const owned_lhs = try cloneNameList(frame.allocator, lhs_names);
    errdefer freeNameList(frame.allocator, owned_lhs);
    const owned_rhs = try cloneNameList(frame.allocator, rhs_names);
    errdefer freeNameList(frame.allocator, owned_rhs);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (metric) {
        .dot => .{ .row_cumulative_weighted_dot = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_names = owned_outputs } },
        .cosine => .{ .row_cumulative_weighted_cosine_similarity = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_names = owned_outputs } },
        .squared_euclidean => .{ .row_cumulative_weighted_squared_euclidean_distance = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_names = owned_outputs } },
        .euclidean => .{ .row_cumulative_weighted_euclidean_distance = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_names = owned_outputs } },
        .manhattan => .{ .row_cumulative_weighted_manhattan_distance = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_names = owned_outputs } },
        .chebyshev => .{ .row_cumulative_weighted_chebyshev_distance = .{ .lhs_names = owned_lhs, .rhs_names = owned_rhs, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedDot(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPairMetric(frame, lhs_names, rhs_names, weight_names, output_names, .dot);
}

pub const withRowCumWeightedDot = withRowCumulativeWeightedDot;
pub const withRowPrefixWeightedDot = withRowCumulativeWeightedDot;

pub fn withRowCumulativeWeightedCosineSimilarity(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPairMetric(frame, lhs_names, rhs_names, weight_names, output_names, .cosine);
}

pub const withRowCumulativeWeightedCosine = withRowCumulativeWeightedCosineSimilarity;
pub const withRowCumWeightedCosineSimilarity = withRowCumulativeWeightedCosineSimilarity;
pub const withRowCumWeightedCosine = withRowCumulativeWeightedCosineSimilarity;
pub const withRowPrefixWeightedCosineSimilarity = withRowCumulativeWeightedCosineSimilarity;
pub const withRowPrefixWeightedCosine = withRowCumulativeWeightedCosineSimilarity;

pub fn withRowCumulativeWeightedSquaredEuclideanDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPairMetric(frame, lhs_names, rhs_names, weight_names, output_names, .squared_euclidean);
}

pub const withRowCumulativeWeightedSquaredDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowCumulativeWeightedSqEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowCumWeightedSquaredEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowCumWeightedSquaredDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowCumWeightedSqEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowPrefixWeightedSquaredEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowPrefixWeightedSquaredDistance = withRowCumulativeWeightedSquaredEuclideanDistance;
pub const withRowPrefixWeightedSqEuclideanDistance = withRowCumulativeWeightedSquaredEuclideanDistance;

pub fn withRowCumulativeWeightedEuclideanDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPairMetric(frame, lhs_names, rhs_names, weight_names, output_names, .euclidean);
}

pub const withRowCumulativeWeightedL2Distance = withRowCumulativeWeightedEuclideanDistance;
pub const withRowCumWeightedEuclideanDistance = withRowCumulativeWeightedEuclideanDistance;
pub const withRowCumWeightedL2Distance = withRowCumulativeWeightedEuclideanDistance;
pub const withRowPrefixWeightedEuclideanDistance = withRowCumulativeWeightedEuclideanDistance;
pub const withRowPrefixWeightedL2Distance = withRowCumulativeWeightedEuclideanDistance;

pub fn withRowCumulativeWeightedManhattanDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPairMetric(frame, lhs_names, rhs_names, weight_names, output_names, .manhattan);
}

pub const withRowCumulativeWeightedL1Distance = withRowCumulativeWeightedManhattanDistance;
pub const withRowCumWeightedManhattanDistance = withRowCumulativeWeightedManhattanDistance;
pub const withRowCumWeightedL1Distance = withRowCumulativeWeightedManhattanDistance;
pub const withRowPrefixWeightedManhattanDistance = withRowCumulativeWeightedManhattanDistance;
pub const withRowPrefixWeightedL1Distance = withRowCumulativeWeightedManhattanDistance;

pub fn withRowCumulativeWeightedChebyshevDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPairMetric(frame, lhs_names, rhs_names, weight_names, output_names, .chebyshev);
}

pub const withRowCumWeightedChebyshevDistance = withRowCumulativeWeightedChebyshevDistance;
pub const withRowPrefixWeightedChebyshevDistance = withRowCumulativeWeightedChebyshevDistance;

pub fn withRowWeightedMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, value_names, weight_names, output_name, .weighted_mean);
}

pub fn withRowWeightedSum(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_sum = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

fn withRowWeightedSupport(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, comptime reduction: enum { weight_sum, positive_count, effective_n }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .weight_sum => .{ .row_weighted_weight_sum = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .positive_count => .{ .row_weighted_positive_count = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .effective_n => .{ .row_weighted_effective_n = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
    });
}

fn withRowCumulativeWeightedColumns(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime reduction: enum { sum, mean }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .sum => .{ .row_cumulative_weighted_sum = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .mean => .{ .row_cumulative_weighted_mean = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedSum(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedColumns(frame, value_names, weight_names, output_names, .sum);
}

pub const withRowCumWeightedSum = withRowCumulativeWeightedSum;
pub const withRowPrefixWeightedSum = withRowCumulativeWeightedSum;

pub fn withRowCumulativeWeightedMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedColumns(frame, value_names, weight_names, output_names, .mean);
}

pub const withRowCumWeightedMean = withRowCumulativeWeightedMean;
pub const withRowPrefixWeightedMean = withRowCumulativeWeightedMean;
pub const withRowCumulativeWeightedAverage = withRowCumulativeWeightedMean;
pub const withRowCumulativeWeightedAvg = withRowCumulativeWeightedMean;
pub const withRowCumWeightedAverage = withRowCumulativeWeightedMean;
pub const withRowCumWeightedAvg = withRowCumulativeWeightedMean;
pub const withRowPrefixWeightedAverage = withRowCumulativeWeightedMean;
pub const withRowPrefixWeightedAvg = withRowCumulativeWeightedMean;

fn withRowCumulativeWeightedMoment(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime reduction: enum { mean_square, rms, mean_abs, l1_norm, l2_norm }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .mean_square => .{ .row_cumulative_weighted_mean_square = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .rms => .{ .row_cumulative_weighted_rms = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .mean_abs => .{ .row_cumulative_weighted_mean_abs = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .l1_norm => .{ .row_cumulative_weighted_l1_norm = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .l2_norm => .{ .row_cumulative_weighted_l2_norm = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedMeanSquare(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedMoment(frame, value_names, weight_names, output_names, .mean_square);
}

pub fn withRowCumulativeWeightedRms(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedMoment(frame, value_names, weight_names, output_names, .rms);
}

pub fn withRowCumulativeWeightedMeanAbs(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedMoment(frame, value_names, weight_names, output_names, .mean_abs);
}

pub fn withRowCumulativeWeightedL1Norm(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedMoment(frame, value_names, weight_names, output_names, .l1_norm);
}

pub fn withRowCumulativeWeightedL2Norm(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedMoment(frame, value_names, weight_names, output_names, .l2_norm);
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

fn withRowCumulativeWeightedExtrema(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime reduction: enum { min, max, max_abs, min_abs, range, midrange, range_coeff }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .min => .{ .row_cumulative_weighted_min = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .max => .{ .row_cumulative_weighted_max = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .max_abs => .{ .row_cumulative_weighted_max_abs = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .min_abs => .{ .row_cumulative_weighted_min_abs = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .range => .{ .row_cumulative_weighted_range = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .midrange => .{ .row_cumulative_weighted_midrange = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .range_coeff => .{ .row_cumulative_weighted_range_coeff = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedMin(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedExtrema(frame, value_names, weight_names, output_names, .min);
}

pub fn withRowCumulativeWeightedMax(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedExtrema(frame, value_names, weight_names, output_names, .max);
}

pub fn withRowCumulativeWeightedMaxAbs(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedExtrema(frame, value_names, weight_names, output_names, .max_abs);
}

pub fn withRowCumulativeWeightedMinAbs(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedExtrema(frame, value_names, weight_names, output_names, .min_abs);
}

pub fn withRowCumulativeWeightedRange(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedExtrema(frame, value_names, weight_names, output_names, .range);
}

pub fn withRowCumulativeWeightedMidrange(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedExtrema(frame, value_names, weight_names, output_names, .midrange);
}

pub fn withRowCumulativeWeightedRangeCoeff(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedExtrema(frame, value_names, weight_names, output_names, .range_coeff);
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

fn withRowCumulativeWeightedLogProduct(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime reduction: enum { product, geometric_mean, harmonic_mean, logsumexp, logmeanexp }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .product => .{ .row_cumulative_weighted_product = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .geometric_mean => .{ .row_cumulative_weighted_geometric_mean = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .harmonic_mean => .{ .row_cumulative_weighted_harmonic_mean = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .logsumexp => .{ .row_cumulative_weighted_logsumexp = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .logmeanexp => .{ .row_cumulative_weighted_logmeanexp = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedProduct(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedLogProduct(frame, value_names, weight_names, output_names, .product);
}

pub fn withRowCumulativeWeightedGeometricMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedLogProduct(frame, value_names, weight_names, output_names, .geometric_mean);
}

pub fn withRowCumulativeWeightedHarmonicMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedLogProduct(frame, value_names, weight_names, output_names, .harmonic_mean);
}

pub fn withRowCumulativeWeightedLogSumExp(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedLogProduct(frame, value_names, weight_names, output_names, .logsumexp);
}

pub fn withRowCumulativeWeightedLogMeanExp(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedLogProduct(frame, value_names, weight_names, output_names, .logmeanexp);
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

fn withRowCumulativeWeightedDispersion(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, correction: f64, comptime reduction: enum { variance, stddev, sem, cv, fano }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .variance => .{ .row_cumulative_weighted_variance = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs, .correction = correction } },
        .stddev => .{ .row_cumulative_weighted_stddev = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs, .correction = correction } },
        .sem => .{ .row_cumulative_weighted_sem = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs, .correction = correction } },
        .cv => .{ .row_cumulative_weighted_cv = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs, .correction = correction } },
        .fano => .{ .row_cumulative_weighted_fano = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs, .correction = correction } },
    });
}

pub fn withRowCumulativeWeightedVariance(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeWeightedDispersion(frame, value_names, weight_names, output_names, correction, .variance);
}

pub fn withRowCumulativeWeightedVar(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeWeightedVariance(frame, value_names, weight_names, output_names, correction);
}

pub fn withRowCumulativeWeightedStddev(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeWeightedDispersion(frame, value_names, weight_names, output_names, correction, .stddev);
}

pub fn withRowCumulativeWeightedStd(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeWeightedStddev(frame, value_names, weight_names, output_names, correction);
}

pub fn withRowCumulativeWeightedSem(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeWeightedDispersion(frame, value_names, weight_names, output_names, correction, .sem);
}

pub fn withRowCumulativeWeightedCv(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeWeightedDispersion(frame, value_names, weight_names, output_names, correction, .cv);
}

pub fn withRowCumulativeWeightedFano(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeWeightedDispersion(frame, value_names, weight_names, output_names, correction, .fano);
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

fn withRowCumulativeWeightedShape(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime reduction: enum { skewness, kurtosis }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .skewness => .{ .row_cumulative_weighted_skewness = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .kurtosis => .{ .row_cumulative_weighted_kurtosis = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedSkewness(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedShape(frame, value_names, weight_names, output_names, .skewness);
}

pub fn withRowCumulativeWeightedKurtosis(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedShape(frame, value_names, weight_names, output_names, .kurtosis);
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

fn withRowCumulativeWeightedSupport(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime reduction: enum { weight_sum, positive_count, effective_n }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .weight_sum => .{ .row_cumulative_weighted_weight_sum = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .positive_count => .{ .row_cumulative_weighted_positive_count = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .effective_n => .{ .row_cumulative_weighted_effective_n = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedWeightSum(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedSupport(frame, value_names, weight_names, output_names, .weight_sum);
}

pub fn withRowCumulativeWeightedPositiveCount(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedSupport(frame, value_names, weight_names, output_names, .positive_count);
}

pub fn withRowCumulativeWeightedEffectiveN(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedSupport(frame, value_names, weight_names, output_names, .effective_n);
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

pub fn withRowWeightedWeightSum(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedSupport(frame, value_names, weight_names, output_name, .weight_sum);
}

pub fn withRowWeightedPositiveCount(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedSupport(frame, value_names, weight_names, output_name, .positive_count);
}

pub fn withRowWeightedEffectiveN(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedSupport(frame, value_names, weight_names, output_name, .effective_n);
}

pub const withRowWeightedEffectiveCount = withRowWeightedEffectiveN;

fn withRowWeightedMoment(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, comptime reduction: enum { mean_square, rms, mean_abs, l1_norm, l2_norm }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .mean_square => .{ .row_weighted_mean_square = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .rms => .{ .row_weighted_rms = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .mean_abs => .{ .row_weighted_mean_abs = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .l1_norm => .{ .row_weighted_l1_norm = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .l2_norm => .{ .row_weighted_l2_norm = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
    });
}

pub fn withRowWeightedMeanSquare(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedMoment(frame, value_names, weight_names, output_name, .mean_square);
}

pub fn withRowWeightedRms(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedMoment(frame, value_names, weight_names, output_name, .rms);
}

pub fn withRowWeightedMeanAbs(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedMoment(frame, value_names, weight_names, output_name, .mean_abs);
}

pub fn withRowWeightedL1Norm(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedMoment(frame, value_names, weight_names, output_name, .l1_norm);
}

pub fn withRowWeightedL2Norm(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedMoment(frame, value_names, weight_names, output_name, .l2_norm);
}

pub const withRowWeightedMeanSquared = withRowWeightedMeanSquare;
pub const withRowWeightedMeanSq = withRowWeightedMeanSquare;
pub const withRowWeightedRMS = withRowWeightedRms;
pub const withRowWeightedL1 = withRowWeightedL1Norm;
pub const withRowWeightedL2 = withRowWeightedL2Norm;

fn withRowWeightedExtrema(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, comptime reduction: enum { min, max, max_abs, min_abs, range, midrange, range_coeff }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .min => .{ .row_weighted_min = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .max => .{ .row_weighted_max = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .max_abs => .{ .row_weighted_max_abs = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .min_abs => .{ .row_weighted_min_abs = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .range => .{ .row_weighted_range = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .midrange => .{ .row_weighted_midrange = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .range_coeff => .{ .row_weighted_range_coeff = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
    });
}

pub fn withRowWeightedMin(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedExtrema(frame, value_names, weight_names, output_name, .min);
}

pub fn withRowWeightedMax(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedExtrema(frame, value_names, weight_names, output_name, .max);
}

pub fn withRowWeightedMaxAbs(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedExtrema(frame, value_names, weight_names, output_name, .max_abs);
}

pub fn withRowWeightedMinAbs(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedExtrema(frame, value_names, weight_names, output_name, .min_abs);
}

pub fn withRowWeightedRange(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedExtrema(frame, value_names, weight_names, output_name, .range);
}

pub fn withRowWeightedMidrange(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedExtrema(frame, value_names, weight_names, output_name, .midrange);
}

pub fn withRowWeightedRangeCoeff(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedExtrema(frame, value_names, weight_names, output_name, .range_coeff);
}

pub const withRowWeightedMinimum = withRowWeightedMin;
pub const withRowWeightedMaximum = withRowWeightedMax;
pub const withRowWeightedMaximumAbs = withRowWeightedMaxAbs;
pub const withRowWeightedMaxAbsolute = withRowWeightedMaxAbs;
pub const withRowWeightedMinimumAbs = withRowWeightedMinAbs;
pub const withRowWeightedMinAbsolute = withRowWeightedMinAbs;
pub const withRowWeightedRangeCoefficient = withRowWeightedRangeCoeff;

fn withRowWeightedLogProduct(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, comptime reduction: enum { product, geometric_mean, harmonic_mean, logsumexp, logmeanexp }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .product => .{ .row_weighted_product = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .geometric_mean => .{ .row_weighted_geometric_mean = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .harmonic_mean => .{ .row_weighted_harmonic_mean = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .logsumexp => .{ .row_weighted_logsumexp = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .logmeanexp => .{ .row_weighted_logmeanexp = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
    });
}

pub fn withRowWeightedProduct(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedLogProduct(frame, value_names, weight_names, output_name, .product);
}

pub fn withRowWeightedGeometricMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedLogProduct(frame, value_names, weight_names, output_name, .geometric_mean);
}

pub fn withRowWeightedHarmonicMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedLogProduct(frame, value_names, weight_names, output_name, .harmonic_mean);
}

pub fn withRowWeightedLogSumExp(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedLogProduct(frame, value_names, weight_names, output_name, .logsumexp);
}

pub fn withRowWeightedLogMeanExp(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedLogProduct(frame, value_names, weight_names, output_name, .logmeanexp);
}

pub const withRowWeightedProd = withRowWeightedProduct;
pub const withRowWeightedGeoMean = withRowWeightedGeometricMean;
pub const withRowWeightedHarmMean = withRowWeightedHarmonicMean;
pub const withRowWeightedLogsumexp = withRowWeightedLogSumExp;
pub const withRowWeightedLogmeanexp = withRowWeightedLogMeanExp;

fn withRowWeightedDispersion(
    frame: anytype,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
    comptime reduction: enum { variance, stddev, sem, cv, fano },
) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (reduction) {
        .variance => try frame.ops.append(frame.allocator, .{ .row_weighted_variance = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .stddev => try frame.ops.append(frame.allocator, .{ .row_weighted_stddev = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .sem => try frame.ops.append(frame.allocator, .{ .row_weighted_sem = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .cv => try frame.ops.append(frame.allocator, .{ .row_weighted_cv = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .fano => try frame.ops.append(frame.allocator, .{ .row_weighted_fano = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
    }
}

pub fn withRowWeightedVariance(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedDispersion(frame, value_names, weight_names, output_name, correction, .variance);
}

pub fn withRowWeightedVar(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedVariance(frame, value_names, weight_names, output_name, correction);
}

pub fn withRowWeightedStddev(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedDispersion(frame, value_names, weight_names, output_name, correction, .stddev);
}

pub fn withRowWeightedStd(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedStddev(frame, value_names, weight_names, output_name, correction);
}

pub fn withRowWeightedSem(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedDispersion(frame, value_names, weight_names, output_name, correction, .sem);
}

pub fn withRowWeightedCv(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedDispersion(frame, value_names, weight_names, output_name, correction, .cv);
}

pub fn withRowWeightedFano(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedDispersion(frame, value_names, weight_names, output_name, correction, .fano);
}

pub const withRowWeightedSEM = withRowWeightedSem;
pub const withRowWeightedCV = withRowWeightedCv;

fn withRowWeightedShape(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, comptime reduction: enum { skewness, kurtosis }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .skewness => .{ .row_weighted_skewness = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .kurtosis => .{ .row_weighted_kurtosis = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
    });
}

pub fn withRowWeightedSkewness(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedShape(frame, value_names, weight_names, output_name, .skewness);
}

pub fn withRowWeightedKurtosis(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedShape(frame, value_names, weight_names, output_name, .kurtosis);
}

pub const withRowWeightedSkew = withRowWeightedSkewness;
pub const withRowWeightedKurt = withRowWeightedKurtosis;

fn withRowWeightedPair(
    frame: anytype,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
    comptime reduction: enum { dot, cosine, squared_euclidean, euclidean, manhattan, chebyshev, canberra, bray_curtis, mean_error, mae, mse, rmse, mape, smape, covariance, correlation, beta },
) DeviceDataError!void {
    const owned_lhs = try cloneNameList(frame.allocator, lhs_names);
    errdefer {
        for (owned_lhs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_lhs);
    }
    const owned_rhs = try cloneNameList(frame.allocator, rhs_names);
    errdefer {
        for (owned_rhs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_rhs);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (reduction) {
        .dot => try frame.ops.append(frame.allocator, .{ .row_weighted_dot = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .cosine => try frame.ops.append(frame.allocator, .{ .row_weighted_cosine_similarity = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .squared_euclidean => try frame.ops.append(frame.allocator, .{ .row_weighted_squared_euclidean_distance = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .euclidean => try frame.ops.append(frame.allocator, .{ .row_weighted_euclidean_distance = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .manhattan => try frame.ops.append(frame.allocator, .{ .row_weighted_manhattan_distance = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .chebyshev => try frame.ops.append(frame.allocator, .{ .row_weighted_chebyshev_distance = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .canberra => try frame.ops.append(frame.allocator, .{ .row_weighted_canberra_distance = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .bray_curtis => try frame.ops.append(frame.allocator, .{ .row_weighted_bray_curtis_distance = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .mean_error => try frame.ops.append(frame.allocator, .{ .row_weighted_mean_error = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .mae => try frame.ops.append(frame.allocator, .{ .row_weighted_mae = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .mse => try frame.ops.append(frame.allocator, .{ .row_weighted_mse = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .rmse => try frame.ops.append(frame.allocator, .{ .row_weighted_rmse = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .mape => try frame.ops.append(frame.allocator, .{ .row_weighted_mape = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .smape => try frame.ops.append(frame.allocator, .{ .row_weighted_smape = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .covariance => try frame.ops.append(frame.allocator, .{ .row_weighted_covariance = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .correlation => try frame.ops.append(frame.allocator, .{ .row_weighted_correlation = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .beta => try frame.ops.append(frame.allocator, .{ .row_weighted_beta = .{
            .lhs_names = owned_lhs,
            .rhs_names = owned_rhs,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .correction = correction,
        } }),
    }
}

pub fn withRowWeightedDot(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .dot);
}

pub fn withRowWeightedCosineSimilarity(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .cosine);
}

pub fn withRowWeightedSquaredEuclideanDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .squared_euclidean);
}

pub fn withRowWeightedEuclideanDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .euclidean);
}

pub fn withRowWeightedManhattanDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .manhattan);
}

pub const withRowWeightedCosine = withRowWeightedCosineSimilarity;
pub const withRowWeightedSquaredDistance = withRowWeightedSquaredEuclideanDistance;
pub const withRowWeightedSqEuclideanDistance = withRowWeightedSquaredEuclideanDistance;
pub const withRowWeightedL2Distance = withRowWeightedEuclideanDistance;
pub const withRowWeightedL1Distance = withRowWeightedManhattanDistance;

pub fn withRowWeightedChebyshevDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .chebyshev);
}

pub fn withRowWeightedCanberraDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .canberra);
}

pub fn withRowWeightedBrayCurtisDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .bray_curtis);
}

pub fn withRowWeightedMeanError(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .mean_error);
}

pub fn withRowWeightedMae(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .mae);
}

pub fn withRowWeightedMse(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .mse);
}

pub fn withRowWeightedRmse(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .rmse);
}

pub fn withRowWeightedMape(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .mape);
}

pub fn withRowWeightedSmape(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, 0.0, .smape);
}

pub const withRowWeightedBias = withRowWeightedMeanError;
pub const withRowWeightedMAE = withRowWeightedMae;
pub const withRowWeightedMSE = withRowWeightedMse;
pub const withRowWeightedRMSE = withRowWeightedRmse;
pub const withRowWeightedMAPE = withRowWeightedMape;
pub const withRowWeightedSMAPE = withRowWeightedSmape;

pub fn withRowWeightedCovariance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, correction, .covariance);
}

pub fn withRowWeightedCorrelation(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, correction, .correlation);
}

pub const withRowWeightedCov = withRowWeightedCovariance;
pub const withRowWeightedCorr = withRowWeightedCorrelation;

pub fn withRowWeightedBeta(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, correction, .beta);
}

fn withRowCumulativeWeightedQuantileCore(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, q: f64, comptime op: enum { quantile, median, iqr, mad, trimmed_mean, winsorized_mean }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (op) {
        .quantile => .{ .row_cumulative_weighted_quantile = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs, .q = q } },
        .median => .{ .row_cumulative_weighted_median = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .iqr => .{ .row_cumulative_weighted_iqr = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .mad => .{ .row_cumulative_weighted_mad = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .trimmed_mean => .{ .row_cumulative_weighted_trimmed_mean = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs, .q = q } },
        .winsorized_mean => .{ .row_cumulative_weighted_winsorized_mean = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs, .q = q } },
    });
}

pub fn withRowCumulativeWeightedQuantile(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, q: f64) DeviceDataError!void {
    return withRowCumulativeWeightedQuantileCore(frame, value_names, weight_names, output_names, q, .quantile);
}

pub fn withRowCumulativeWeightedMedian(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedQuantileCore(frame, value_names, weight_names, output_names, 0.5, .median);
}

pub fn withRowCumulativeWeightedIqr(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedQuantileCore(frame, value_names, weight_names, output_names, 0.5, .iqr);
}

pub fn withRowCumulativeWeightedMad(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedQuantileCore(frame, value_names, weight_names, output_names, 0.5, .mad);
}

pub fn withRowCumulativeWeightedTrimmedMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, trim_fraction: f64) DeviceDataError!void {
    return withRowCumulativeWeightedQuantileCore(frame, value_names, weight_names, output_names, trim_fraction, .trimmed_mean);
}

pub fn withRowCumulativeWeightedWinsorizedMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, winsor_fraction: f64) DeviceDataError!void {
    return withRowCumulativeWeightedQuantileCore(frame, value_names, weight_names, output_names, winsor_fraction, .winsorized_mean);
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
pub const withRowCumWeightedTrimmedMean = withRowCumulativeWeightedTrimmedMean;
pub const withRowPrefixWeightedTrimmedMean = withRowCumulativeWeightedTrimmedMean;
pub const withRowCumWeightedWinsorizedMean = withRowCumulativeWeightedWinsorizedMean;
pub const withRowPrefixWeightedWinsorizedMean = withRowCumulativeWeightedWinsorizedMean;

pub fn withRowWeightedQuantile(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_quantile = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
        .q = q,
    } });
}

pub fn withRowWeightedMedian(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_median = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub fn withRowWeightedIqr(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_iqr = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub fn withRowWeightedMad(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_mad = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

fn withRowWeightedRobustMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, fraction: f64, comptime op: enum { trimmed_mean, winsorized_mean }) DeviceDataError!void {
    if (std.math.isNan(fraction) or fraction < 0.0 or fraction >= 0.5) return error.InvalidShape;
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (op) {
        .trimmed_mean => try frame.ops.append(frame.allocator, .{ .row_weighted_trimmed_mean = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .q = fraction,
        } }),
        .winsorized_mean => try frame.ops.append(frame.allocator, .{ .row_weighted_winsorized_mean = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
            .q = fraction,
        } }),
    }
}

pub fn withRowWeightedTrimmedMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
    return withRowWeightedRobustMean(frame, value_names, weight_names, output_name, trim_fraction, .trimmed_mean);
}

pub fn withRowWeightedWinsorizedMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
    return withRowWeightedRobustMean(frame, value_names, weight_names, output_name, winsor_fraction, .winsorized_mean);
}

fn withRowCumulativeWeightedPercentileShape(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime op: enum { interdecile_range, midhinge, trimean, bowley_skewness, quartile_coeff_dispersion, kelley_skewness }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (op) {
        .interdecile_range => .{ .row_cumulative_weighted_interdecile_range = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .midhinge => .{ .row_cumulative_weighted_midhinge = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .trimean => .{ .row_cumulative_weighted_trimean = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .bowley_skewness => .{ .row_cumulative_weighted_bowley_skewness = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .quartile_coeff_dispersion => .{ .row_cumulative_weighted_quartile_coeff_dispersion = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .kelley_skewness => .{ .row_cumulative_weighted_kelley_skewness = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedInterdecileRange(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPercentileShape(frame, value_names, weight_names, output_names, .interdecile_range);
}

pub fn withRowCumulativeWeightedMidhinge(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPercentileShape(frame, value_names, weight_names, output_names, .midhinge);
}

pub fn withRowCumulativeWeightedTrimean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPercentileShape(frame, value_names, weight_names, output_names, .trimean);
}

pub fn withRowCumulativeWeightedBowleySkewness(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPercentileShape(frame, value_names, weight_names, output_names, .bowley_skewness);
}

pub fn withRowCumulativeWeightedQuartileCoeffDispersion(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPercentileShape(frame, value_names, weight_names, output_names, .quartile_coeff_dispersion);
}

pub fn withRowCumulativeWeightedKelleySkewness(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedPercentileShape(frame, value_names, weight_names, output_names, .kelley_skewness);
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

pub fn withRowCumulativeWeightedMode(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, .{ .row_cumulative_weighted_mode = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } });
}

pub const withRowCumWeightedMode = withRowCumulativeWeightedMode;
pub const withRowPrefixWeightedMode = withRowCumulativeWeightedMode;

fn withRowCumulativeWeightedModeDiagnostic(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime op: enum { weight, ratio, margin, margin_ratio }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (op) {
        .weight => .{ .row_cumulative_weighted_mode_weight = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .ratio => .{ .row_cumulative_weighted_mode_ratio = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .margin => .{ .row_cumulative_weighted_mode_margin = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .margin_ratio => .{ .row_cumulative_weighted_mode_margin_ratio = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedModeWeight(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedModeDiagnostic(frame, value_names, weight_names, output_names, .weight);
}

pub fn withRowCumulativeWeightedModeRatio(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedModeDiagnostic(frame, value_names, weight_names, output_names, .ratio);
}

pub fn withRowCumulativeWeightedModeMargin(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedModeDiagnostic(frame, value_names, weight_names, output_names, .margin);
}

pub fn withRowCumulativeWeightedModeMarginRatio(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedModeDiagnostic(frame, value_names, weight_names, output_names, .margin_ratio);
}

pub const withRowCumWeightedModeWeight = withRowCumulativeWeightedModeWeight;
pub const withRowPrefixWeightedModeWeight = withRowCumulativeWeightedModeWeight;
pub const withRowCumWeightedModeRatio = withRowCumulativeWeightedModeRatio;
pub const withRowPrefixWeightedModeRatio = withRowCumulativeWeightedModeRatio;
pub const withRowCumWeightedModeMargin = withRowCumulativeWeightedModeMargin;
pub const withRowPrefixWeightedModeMargin = withRowCumulativeWeightedModeMargin;
pub const withRowCumWeightedModeMarginRatio = withRowCumulativeWeightedModeMarginRatio;
pub const withRowPrefixWeightedModeMarginRatio = withRowCumulativeWeightedModeMarginRatio;

fn withRowCumulativeWeightedDistribution(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime op: enum { entropy, gini_impurity, perplexity, inverse_simpson, simpson_concentration, evenness }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (op) {
        .entropy => .{ .row_cumulative_weighted_entropy = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .gini_impurity => .{ .row_cumulative_weighted_gini_impurity = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .perplexity => .{ .row_cumulative_weighted_perplexity = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .inverse_simpson => .{ .row_cumulative_weighted_inverse_simpson = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .simpson_concentration => .{ .row_cumulative_weighted_simpson_concentration = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .evenness => .{ .row_cumulative_weighted_evenness = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedEntropy(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedDistribution(frame, value_names, weight_names, output_names, .entropy);
}

pub fn withRowCumulativeWeightedGiniImpurity(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedDistribution(frame, value_names, weight_names, output_names, .gini_impurity);
}

pub const withRowCumulativeWeightedGini = withRowCumulativeWeightedGiniImpurity;

pub fn withRowCumulativeWeightedPerplexity(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedDistribution(frame, value_names, weight_names, output_names, .perplexity);
}

pub fn withRowCumulativeWeightedInverseSimpson(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedDistribution(frame, value_names, weight_names, output_names, .inverse_simpson);
}

pub fn withRowCumulativeWeightedSimpsonConcentration(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedDistribution(frame, value_names, weight_names, output_names, .simpson_concentration);
}

pub const withRowCumulativeWeightedConcentration = withRowCumulativeWeightedSimpsonConcentration;

pub fn withRowCumulativeWeightedEvenness(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedDistribution(frame, value_names, weight_names, output_names, .evenness);
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

fn withRowCumulativeWeightedInequality(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8, comptime reduction: enum { mean_abs_dev, mean_abs_dev_ratio, gini_mean_diff, gini_coefficient }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer freeNameList(frame.allocator, owned_outputs);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .mean_abs_dev => .{ .row_cumulative_weighted_mean_abs_dev = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .mean_abs_dev_ratio => .{ .row_cumulative_weighted_mean_abs_dev_ratio = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .gini_mean_diff => .{ .row_cumulative_weighted_gini_mean_diff = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
        .gini_coefficient => .{ .row_cumulative_weighted_gini_coefficient = .{ .value_names = owned_values, .weight_names = owned_weights, .output_names = owned_outputs } },
    });
}

pub fn withRowCumulativeWeightedMeanAbsDev(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedInequality(frame, value_names, weight_names, output_names, .mean_abs_dev);
}

pub fn withRowCumulativeWeightedMeanAbsDevRatio(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedInequality(frame, value_names, weight_names, output_names, .mean_abs_dev_ratio);
}

pub fn withRowCumulativeWeightedGiniMeanDiff(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedInequality(frame, value_names, weight_names, output_names, .gini_mean_diff);
}

pub fn withRowCumulativeWeightedGiniCoefficient(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeWeightedInequality(frame, value_names, weight_names, output_names, .gini_coefficient);
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

fn withRowWeightedPercentileShape(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, comptime op: enum { interdecile_range, midhinge, trimean, bowley_skewness, quartile_coeff_dispersion, kelley_skewness }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (op) {
        .interdecile_range => .{ .row_weighted_interdecile_range = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .midhinge => .{ .row_weighted_midhinge = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .trimean => .{ .row_weighted_trimean = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .bowley_skewness => .{ .row_weighted_bowley_skewness = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .quartile_coeff_dispersion => .{ .row_weighted_quartile_coeff_dispersion = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .kelley_skewness => .{ .row_weighted_kelley_skewness = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
    });
}

pub fn withRowWeightedInterdecileRange(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPercentileShape(frame, value_names, weight_names, output_name, .interdecile_range);
}

pub fn withRowWeightedMidhinge(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPercentileShape(frame, value_names, weight_names, output_name, .midhinge);
}

pub fn withRowWeightedTrimean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPercentileShape(frame, value_names, weight_names, output_name, .trimean);
}

pub fn withRowWeightedBowleySkewness(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPercentileShape(frame, value_names, weight_names, output_name, .bowley_skewness);
}

pub fn withRowWeightedQuartileCoeffDispersion(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPercentileShape(frame, value_names, weight_names, output_name, .quartile_coeff_dispersion);
}

pub fn withRowWeightedKelleySkewness(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedPercentileShape(frame, value_names, weight_names, output_name, .kelley_skewness);
}

pub const withRowWeightedIdr = withRowWeightedInterdecileRange;
pub const withRowWeightedIDR = withRowWeightedInterdecileRange;
pub const withRowWeightedIQR = withRowWeightedIqr;
pub const withRowWeightedMAD = withRowWeightedMad;
pub const withRowWeightedBowleySkew = withRowWeightedBowleySkewness;
pub const withRowWeightedQcd = withRowWeightedQuartileCoeffDispersion;
pub const withRowWeightedQCD = withRowWeightedQuartileCoeffDispersion;
pub const withRowWeightedKelleySkew = withRowWeightedKelleySkewness;

pub fn withRowWeightedMode(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_mode = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub fn withRowWeightedModeWeight(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_mode_weight = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub fn withRowWeightedModeRatio(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_mode_ratio = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub fn withRowWeightedModeMargin(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_mode_margin = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } });
}

pub fn withRowWeightedModeMarginRatio(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_mode_margin_ratio = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } });
}

pub fn withRowWeightedEntropy(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_entropy = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub fn withRowWeightedGiniImpurity(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_gini_impurity = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub const withRowWeightedGini = withRowWeightedGiniImpurity;

pub fn withRowWeightedPerplexity(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_perplexity = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub fn withRowWeightedInverseSimpson(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_inverse_simpson = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub fn withRowWeightedSimpsonConcentration(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_simpson_concentration = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

pub const withRowWeightedConcentration = withRowWeightedSimpsonConcentration;

pub fn withRowWeightedEvenness(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_weighted_evenness = .{
        .value_names = owned_values,
        .weight_names = owned_weights,
        .output_name = owned_output,
    } });
}

fn withRowWeightedInequality(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, comptime reduction: enum { mean_abs_dev, mean_abs_dev_ratio, gini_mean_diff, gini_coefficient }) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, value_names);
    errdefer freeNameList(frame.allocator, owned_values);
    const owned_weights = try cloneNameList(frame.allocator, weight_names);
    errdefer freeNameList(frame.allocator, owned_weights);
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, switch (reduction) {
        .mean_abs_dev => .{ .row_weighted_mean_abs_dev = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .mean_abs_dev_ratio => .{ .row_weighted_mean_abs_dev_ratio = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .gini_mean_diff => .{ .row_weighted_gini_mean_diff = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
        .gini_coefficient => .{ .row_weighted_gini_coefficient = .{ .value_names = owned_values, .weight_names = owned_weights, .output_name = owned_output } },
    });
}

pub fn withRowWeightedMeanAbsDev(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedInequality(frame, value_names, weight_names, output_name, .mean_abs_dev);
}

pub fn withRowWeightedMeanAbsDevRatio(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedInequality(frame, value_names, weight_names, output_name, .mean_abs_dev_ratio);
}

pub fn withRowWeightedGiniMeanDiff(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedInequality(frame, value_names, weight_names, output_name, .gini_mean_diff);
}

pub fn withRowWeightedGiniCoefficient(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowWeightedInequality(frame, value_names, weight_names, output_name, .gini_coefficient);
}

pub const withRowWeightedMeanAbsoluteDeviation = withRowWeightedMeanAbsDev;
pub const withRowWeightedMadRatio = withRowWeightedMeanAbsDevRatio;
pub const withRowWeightedGiniCoeff = withRowWeightedGiniCoefficient;

fn withRowPairedNumeric(
    frame: anytype,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    output_name: []const u8,
    comptime reduction: enum { weighted_mean, dot, cosine, squared_euclidean, euclidean, manhattan, chebyshev, canberra, bray_curtis, mean_error, mae, mse, rmse, mape, smape, covariance, correlation, beta },
) DeviceDataError!void {
    const owned_values = try cloneNameList(frame.allocator, lhs_names);
    errdefer {
        for (owned_values) |name| frame.allocator.free(name);
        frame.allocator.free(owned_values);
    }
    const owned_weights = try cloneNameList(frame.allocator, rhs_names);
    errdefer {
        for (owned_weights) |name| frame.allocator.free(name);
        frame.allocator.free(owned_weights);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (reduction) {
        .weighted_mean => try frame.ops.append(frame.allocator, .{ .row_weighted_mean = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .dot => try frame.ops.append(frame.allocator, .{ .row_dot = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .cosine => try frame.ops.append(frame.allocator, .{ .row_cosine_similarity = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .squared_euclidean => try frame.ops.append(frame.allocator, .{ .row_squared_euclidean_distance = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .euclidean => try frame.ops.append(frame.allocator, .{ .row_euclidean_distance = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .manhattan => try frame.ops.append(frame.allocator, .{ .row_manhattan_distance = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .chebyshev => try frame.ops.append(frame.allocator, .{ .row_chebyshev_distance = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .canberra => try frame.ops.append(frame.allocator, .{ .row_canberra_distance = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .bray_curtis => try frame.ops.append(frame.allocator, .{ .row_bray_curtis_distance = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .mean_error => try frame.ops.append(frame.allocator, .{ .row_mean_error = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .mae => try frame.ops.append(frame.allocator, .{ .row_mae = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .mse => try frame.ops.append(frame.allocator, .{ .row_mse = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .rmse => try frame.ops.append(frame.allocator, .{ .row_rmse = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .mape => try frame.ops.append(frame.allocator, .{ .row_mape = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .smape => try frame.ops.append(frame.allocator, .{ .row_smape = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .covariance => try frame.ops.append(frame.allocator, .{ .row_covariance = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .correlation => try frame.ops.append(frame.allocator, .{ .row_correlation = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
        .beta => try frame.ops.append(frame.allocator, .{ .row_beta = .{
            .value_names = owned_values,
            .weight_names = owned_weights,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowDot(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .dot);
}

pub fn withRowCosineSimilarity(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .cosine);
}

pub fn withRowCosine(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowCosineSimilarity(frame, lhs_names, rhs_names, output_name);
}

pub fn withRowSquaredEuclideanDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .squared_euclidean);
}

pub fn withRowEuclideanDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .euclidean);
}

pub fn withRowManhattanDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .manhattan);
}

pub fn withRowChebyshevDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .chebyshev);
}

pub fn withRowCanberraDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .canberra);
}

pub fn withRowBrayCurtisDistance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .bray_curtis);
}

pub fn withRowMeanError(frame: anytype, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, actual_names, predicted_names, output_name, .mean_error);
}

pub fn withRowBias(frame: anytype, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMeanError(frame, actual_names, predicted_names, output_name);
}

pub fn withRowMae(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .mae);
}

pub fn withRowMse(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .mse);
}

pub fn withRowRmse(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .rmse);
}

pub fn withRowMape(frame: anytype, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, actual_names, predicted_names, output_name, .mape);
}

pub fn withRowSmape(frame: anytype, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, actual_names, predicted_names, output_name, .smape);
}

pub fn withRowCovariance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .covariance);
}

pub fn withRowCorrelation(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .correlation);
}

pub fn withRowBeta(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, lhs_names, rhs_names, output_name, .beta);
}

fn withRowNumericArgReduction(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: enum { argmin, argmax },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (reduction) {
        .argmin => try frame.ops.append(frame.allocator, .{ .row_argmin = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .argmax => try frame.ops.append(frame.allocator, .{ .row_argmax = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowArgMin(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericArgReduction(frame, names, output_name, .argmin);
}

pub fn withRowArgMax(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericArgReduction(frame, names, output_name, .argmax);
}

fn withRowCumulativeNumericArgReduction(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime argmax: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (argmax) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_argmax = .{ .names = owned_names, .output_names = owned_outputs } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_argmin = .{ .names = owned_names, .output_names = owned_outputs } });
    }
}

pub fn withRowCumulativeArgMin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericArgReduction(frame, names, output_names, false);
}

pub fn withRowCumArgMin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeArgMin(frame, names, output_names);
}

pub fn withRowPrefixArgMin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeArgMin(frame, names, output_names);
}

pub fn withRowCumulativeArgMax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericArgReduction(frame, names, output_names, true);
}

pub fn withRowCumArgMax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeArgMax(frame, names, output_names);
}

pub fn withRowPrefixArgMax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeArgMax(frame, names, output_names);
}

pub fn withRowQuantile(frame: anytype, names: []const []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_quantile = .{
        .names = owned_names,
        .output_name = owned_output,
        .q = q,
    } });
}

pub fn withRowQuantileRange(frame: anytype, names: []const []const u8, output_name: []const u8, low_q: f64, high_q: f64) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_quantile_range = .{
        .names = owned_names,
        .output_name = owned_output,
        .low_q = low_q,
        .high_q = high_q,
    } });
}

pub fn withRowTrimmedMean(frame: anytype, names: []const []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_trimmed_mean = .{
        .names = owned_names,
        .output_name = owned_output,
        .trim_fraction = trim_fraction,
    } });
}

pub fn withRowWinsorizedMean(frame: anytype, names: []const []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_winsorized_mean = .{
        .names = owned_names,
        .output_name = owned_output,
        .winsor_fraction = winsor_fraction,
    } });
}

fn withRowQuantileAlias(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: enum { median, iqr, interdecile_range, midhinge, trimean, bowley_skewness, quartile_coeff_dispersion, kelley_skewness, mad, mode, count_distinct, n_unique, is_duplicated, is_unique },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (reduction) {
        .median => try frame.ops.append(frame.allocator, .{ .row_median = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .iqr => try frame.ops.append(frame.allocator, .{ .row_iqr = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .interdecile_range => try frame.ops.append(frame.allocator, .{ .row_interdecile_range = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .midhinge => try frame.ops.append(frame.allocator, .{ .row_midhinge = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .trimean => try frame.ops.append(frame.allocator, .{ .row_trimean = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .bowley_skewness => try frame.ops.append(frame.allocator, .{ .row_bowley_skewness = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .quartile_coeff_dispersion => try frame.ops.append(frame.allocator, .{ .row_quartile_coeff_dispersion = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .kelley_skewness => try frame.ops.append(frame.allocator, .{ .row_kelley_skewness = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .mad => try frame.ops.append(frame.allocator, .{ .row_mad = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .mode => try frame.ops.append(frame.allocator, .{ .row_mode = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .count_distinct => try frame.ops.append(frame.allocator, .{ .row_count_distinct = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .n_unique => try frame.ops.append(frame.allocator, .{ .row_n_unique = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .is_duplicated => try frame.ops.append(frame.allocator, .{ .row_is_duplicated = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .is_unique => try frame.ops.append(frame.allocator, .{ .row_is_unique = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowMedian(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .median);
}

pub fn withRowIqr(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .iqr);
}

pub fn withRowInterdecileRange(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .interdecile_range);
}

pub fn withRowIdr(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowInterdecileRange(frame, names, output_name);
}

pub fn withRowMidhinge(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .midhinge);
}

pub fn withRowTrimean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .trimean);
}

pub fn withRowBowleySkewness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .bowley_skewness);
}

pub fn withRowBowleySkew(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBowleySkewness(frame, names, output_name);
}

pub fn withRowQuartileCoeffDispersion(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .quartile_coeff_dispersion);
}

pub fn withRowQcd(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuartileCoeffDispersion(frame, names, output_name);
}

pub fn withRowKelleySkewness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .kelley_skewness);
}

pub fn withRowKelleySkew(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowKelleySkewness(frame, names, output_name);
}

pub fn withRowMad(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .mad);
}

pub fn withRowMedianAbsDev(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMad(frame, names, output_name);
}

pub fn withRowMode(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .mode);
}

pub fn withRowCumulativeMode(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_cumulative_mode = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowCumMode(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMode(frame, names, output_names);
}

pub fn withRowPrefixMode(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMode(frame, names, output_names);
}

fn withRowCumulativeModeFrequency(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime ratio: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (ratio) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_mode_ratio = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_mode_count = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    }
}

pub fn withRowCumulativeModeCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeFrequency(frame, names, output_names, false);
}

pub fn withRowCumModeCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeCount(frame, names, output_names);
}

pub fn withRowPrefixModeCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeCount(frame, names, output_names);
}

pub fn withRowCumulativeModeRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeFrequency(frame, names, output_names, true);
}

pub fn withRowCumModeRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeRatio(frame, names, output_names);
}

pub fn withRowPrefixModeRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeRatio(frame, names, output_names);
}

fn withRowCumulativeModeMarginBuilder(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime ratio: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (ratio) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_mode_margin_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_mode_margin = .{ .names = owned_names, .output_names = owned_outputs } });
    }
}

pub fn withRowCumulativeModeMargin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeMarginBuilder(frame, names, output_names, false);
}

pub fn withRowCumModeMargin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeMargin(frame, names, output_names);
}

pub fn withRowPrefixModeMargin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeMargin(frame, names, output_names);
}

pub fn withRowCumulativeModeMarginRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeMarginBuilder(frame, names, output_names, true);
}

pub fn withRowCumModeMarginRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeMarginRatio(frame, names, output_names);
}

pub fn withRowPrefixModeMarginRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeModeMarginRatio(frame, names, output_names);
}

pub fn withRowEntropy(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_entropy = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn withRowGiniImpurity(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_gini_impurity = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn withRowPerplexity(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_perplexity = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn withRowInverseSimpson(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_inverse_simpson = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn withRowSimpsonConcentration(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_simpson_concentration = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn withRowEvenness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_evenness = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn withRowModeCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_mode_count = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn withRowModeRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_mode_ratio = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn withRowModeMargin(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_mode_margin = .{
        .names = owned_names,
        .output_name = owned_output,
    } });
}

pub fn withRowModeMarginRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    try frame.ops.append(frame.allocator, .{ .row_mode_margin_ratio = .{ .names = owned_names, .output_name = owned_output } });
}

pub fn withRowCountDistinct(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .count_distinct);
}

pub fn withRowNUnique(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .n_unique);
}

pub fn withRowIsDuplicated(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .is_duplicated);
}

pub fn withRowIsUnique(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .is_unique);
}

fn withRowCumulativeDistinctCountAlias(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime n_unique: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (n_unique) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_n_unique = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_distinct_count = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    }
}

pub fn withRowCumulativeDistinctCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeDistinctCountAlias(frame, names, output_names, false);
}

pub fn withRowCumDistinctCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeDistinctCount(frame, names, output_names);
}

pub fn withRowPrefixDistinctCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeDistinctCount(frame, names, output_names);
}

pub fn withRowCumulativeNUnique(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeDistinctCountAlias(frame, names, output_names, true);
}

pub fn withRowPrefixNUnique(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNUnique(frame, names, output_names);
}

fn withRowNumericReduction(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: enum { sum, mean, logsumexp, logmeanexp, softmax_entropy, softmax_perplexity, softmax_confidence, softmax_margin, softmax_evenness, softmax_concentration, softmax_normalized_hhi, softmax_gini_impurity, softmax_inverse_simpson, softmax_simpson_evenness, logit_margin, geometric_mean, magnitude_geometric_mean, harmonic_mean, skewness, magnitude_skewness, kurtosis, magnitude_kurtosis, prod, min, max, ptp, magnitude_ptp, midrange, magnitude_midrange, range_coeff, magnitude_range_coeff, mean_abs, hhi, magnitude_normalized_hhi, magnitude_sparsity, magnitude_inverse_simpson, magnitude_simpson_evenness, magnitude_dominance, magnitude_dominance_margin, magnitude_entropy, magnitude_perplexity, magnitude_evenness, mean_abs_dev, gini_mean_diff, gini_coefficient, mean_abs_dev_ratio, rms, l1_norm, l2_norm },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (reduction) {
        .sum => try frame.ops.append(frame.allocator, .{ .row_sum = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .mean => try frame.ops.append(frame.allocator, .{ .row_mean = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .logsumexp => try frame.ops.append(frame.allocator, .{ .row_logsumexp = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .logmeanexp => try frame.ops.append(frame.allocator, .{ .row_logmeanexp = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_entropy => try frame.ops.append(frame.allocator, .{ .row_softmax_entropy = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_perplexity => try frame.ops.append(frame.allocator, .{ .row_softmax_perplexity = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_confidence => try frame.ops.append(frame.allocator, .{ .row_softmax_confidence = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_margin => try frame.ops.append(frame.allocator, .{ .row_softmax_margin = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_evenness => try frame.ops.append(frame.allocator, .{ .row_softmax_evenness = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_concentration => try frame.ops.append(frame.allocator, .{ .row_softmax_concentration = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_normalized_hhi => try frame.ops.append(frame.allocator, .{ .row_softmax_normalized_hhi = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_gini_impurity => try frame.ops.append(frame.allocator, .{ .row_softmax_gini_impurity = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_inverse_simpson => try frame.ops.append(frame.allocator, .{ .row_softmax_inverse_simpson = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .softmax_simpson_evenness => try frame.ops.append(frame.allocator, .{ .row_softmax_simpson_evenness = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .logit_margin => try frame.ops.append(frame.allocator, .{ .row_logit_margin = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .geometric_mean => try frame.ops.append(frame.allocator, .{ .row_geometric_mean = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_geometric_mean => try frame.ops.append(frame.allocator, .{ .row_magnitude_geometric_mean = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .harmonic_mean => try frame.ops.append(frame.allocator, .{ .row_harmonic_mean = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .skewness => try frame.ops.append(frame.allocator, .{ .row_skewness = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_skewness => try frame.ops.append(frame.allocator, .{ .row_magnitude_skewness = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .kurtosis => try frame.ops.append(frame.allocator, .{ .row_kurtosis = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_kurtosis => try frame.ops.append(frame.allocator, .{ .row_magnitude_kurtosis = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .prod => try frame.ops.append(frame.allocator, .{ .row_prod = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .min => try frame.ops.append(frame.allocator, .{ .row_min = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .max => try frame.ops.append(frame.allocator, .{ .row_max = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .ptp => try frame.ops.append(frame.allocator, .{ .row_ptp = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_ptp => try frame.ops.append(frame.allocator, .{ .row_magnitude_ptp = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .midrange => try frame.ops.append(frame.allocator, .{ .row_midrange = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_midrange => try frame.ops.append(frame.allocator, .{ .row_magnitude_midrange = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .range_coeff => try frame.ops.append(frame.allocator, .{ .row_range_coeff = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_range_coeff => try frame.ops.append(frame.allocator, .{ .row_magnitude_range_coeff = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .mean_abs => try frame.ops.append(frame.allocator, .{ .row_mean_abs = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .hhi => try frame.ops.append(frame.allocator, .{ .row_hhi = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_normalized_hhi => try frame.ops.append(frame.allocator, .{ .row_magnitude_normalized_hhi = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_sparsity => try frame.ops.append(frame.allocator, .{ .row_magnitude_sparsity = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_inverse_simpson => try frame.ops.append(frame.allocator, .{ .row_magnitude_inverse_simpson = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_simpson_evenness => try frame.ops.append(frame.allocator, .{ .row_magnitude_simpson_evenness = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_dominance => try frame.ops.append(frame.allocator, .{ .row_magnitude_dominance = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_dominance_margin => try frame.ops.append(frame.allocator, .{ .row_magnitude_dominance_margin = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_entropy => try frame.ops.append(frame.allocator, .{ .row_magnitude_entropy = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_perplexity => try frame.ops.append(frame.allocator, .{ .row_magnitude_perplexity = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .magnitude_evenness => try frame.ops.append(frame.allocator, .{ .row_magnitude_evenness = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .mean_abs_dev => try frame.ops.append(frame.allocator, .{ .row_mean_abs_dev = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .gini_mean_diff => try frame.ops.append(frame.allocator, .{ .row_gini_mean_diff = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .gini_coefficient => try frame.ops.append(frame.allocator, .{ .row_gini_coefficient = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .mean_abs_dev_ratio => try frame.ops.append(frame.allocator, .{ .row_mean_abs_dev_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .rms => try frame.ops.append(frame.allocator, .{ .row_rms = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .l1_norm => try frame.ops.append(frame.allocator, .{ .row_l1_norm = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .l2_norm => try frame.ops.append(frame.allocator, .{ .row_l2_norm = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowSum(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .sum);
}

pub fn withRowMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .mean);
}

pub fn withRowLogSumExp(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .logsumexp);
}

pub fn withRowLogsumexp(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowLogSumExp(frame, names, output_name);
}

pub fn withRowLogMeanExp(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .logmeanexp);
}

pub fn withRowLogmeanexp(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowLogMeanExp(frame, names, output_name);
}

pub fn withRowCentered(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_centered = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowDemean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCentered(frame, names, output_names);
}

pub fn withRowZScore(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_zscore = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowZscore(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowZScore(frame, names, output_names);
}

pub fn withRowStandardize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowZScore(frame, names, output_names);
}

pub fn withRowRobustZScore(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_robust_zscore = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowRobustZscore(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowRobustZScore(frame, names, output_names);
}

pub fn withRowMadZScore(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowRobustZScore(frame, names, output_names);
}

pub fn withRowMadZscore(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowRobustZScore(frame, names, output_names);
}

pub fn withRowAverageRank(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_average_rank = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowAverageRanks(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowAverageRank(frame, names, output_names);
}

pub fn withRowAvgRank(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowAverageRank(frame, names, output_names);
}

pub fn withRowAvgRanks(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowAverageRank(frame, names, output_names);
}

pub fn withRowFractionalRank(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowAverageRank(frame, names, output_names);
}

pub fn withRowFractionalRanks(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowAverageRank(frame, names, output_names);
}

pub fn withRowOrdinalRank(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_ordinal_rank = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowOrdinalRanks(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowOrdinalRank(frame, names, output_names);
}

pub fn withRowDenseRank(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_dense_rank = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowDenseRanks(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowDenseRank(frame, names, output_names);
}

pub fn withRowCompetitionRank(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_competition_rank = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowCompetitionRanks(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCompetitionRank(frame, names, output_names);
}

pub fn withRowMinRank(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCompetitionRank(frame, names, output_names);
}

pub fn withRowMinRanks(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCompetitionRank(frame, names, output_names);
}

pub fn withRowPercentRank(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_percent_rank = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowPercentRanks(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowPercentRank(frame, names, output_names);
}

pub fn withRowPercentileRank(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowPercentRank(frame, names, output_names);
}

pub fn withRowPercentileRanks(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowPercentRank(frame, names, output_names);
}

pub fn withRowCumeDist(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_cume_dist = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowCumeDistribution(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumeDist(frame, names, output_names);
}

pub fn withRowCumulativeDistribution(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumeDist(frame, names, output_names);
}

pub fn withRowCumulativeSum(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_cumulative_sum = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowCumsum(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSum(frame, names, output_names);
}

pub fn withRowCumSum(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSum(frame, names, output_names);
}

pub fn withRowPrefixSum(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSum(frame, names, output_names);
}

pub fn withRowCumulativeMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_cumulative_mean = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowCummean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMean(frame, names, output_names);
}

pub fn withRowCumMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMean(frame, names, output_names);
}

pub fn withRowPrefixMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMean(frame, names, output_names);
}

pub fn withRowCumulativeAverage(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMean(frame, names, output_names);
}

pub fn withRowCumAverage(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMean(frame, names, output_names);
}

pub fn withRowCumAvg(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMean(frame, names, output_names);
}

pub fn withRowPrefixAverage(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMean(frame, names, output_names);
}

pub fn withRowPrefixAvg(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMean(frame, names, output_names);
}

fn withRowCumulativeLogExp(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime mean: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (mean) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_logmeanexp = .{ .names = owned_names, .output_names = owned_outputs } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_logsumexp = .{ .names = owned_names, .output_names = owned_outputs } });
    }
}

pub fn withRowCumulativeLogSumExp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogExp(frame, names, output_names, false);
}

pub fn withRowCumulativeLogsumexp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogSumExp(frame, names, output_names);
}

pub fn withRowCumLogSumExp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogSumExp(frame, names, output_names);
}

pub fn withRowCumLogsumexp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogSumExp(frame, names, output_names);
}

pub fn withRowPrefixLogSumExp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogSumExp(frame, names, output_names);
}

pub fn withRowPrefixLogsumexp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogSumExp(frame, names, output_names);
}

pub fn withRowCumulativeLogMeanExp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogExp(frame, names, output_names, true);
}

pub fn withRowCumulativeLogmeanexp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogMeanExp(frame, names, output_names);
}

pub fn withRowCumLogMeanExp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogMeanExp(frame, names, output_names);
}

pub fn withRowCumLogmeanexp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogMeanExp(frame, names, output_names);
}

pub fn withRowPrefixLogMeanExp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogMeanExp(frame, names, output_names);
}

pub fn withRowPrefixLogmeanexp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLogMeanExp(frame, names, output_names);
}

fn withRowCumulativePowerMean(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime harmonic: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (harmonic) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_harmonic_mean = .{ .names = owned_names, .output_names = owned_outputs } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_geometric_mean = .{ .names = owned_names, .output_names = owned_outputs } });
    }
}

pub fn withRowCumulativeGeometricMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePowerMean(frame, names, output_names, false);
}

pub fn withRowCumulativeGeoMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeGeometricMean(frame, names, output_names);
}

pub fn withRowCumGeometricMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeGeometricMean(frame, names, output_names);
}

pub fn withRowCumGeoMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeGeometricMean(frame, names, output_names);
}

pub fn withRowPrefixGeometricMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeGeometricMean(frame, names, output_names);
}

pub fn withRowPrefixGeoMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeGeometricMean(frame, names, output_names);
}

pub fn withRowCumulativeHarmonicMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePowerMean(frame, names, output_names, true);
}

pub fn withRowCumulativeHarmMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeHarmonicMean(frame, names, output_names);
}

pub fn withRowCumHarmonicMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeHarmonicMean(frame, names, output_names);
}

pub fn withRowCumHarmMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeHarmonicMean(frame, names, output_names);
}

pub fn withRowPrefixHarmonicMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeHarmonicMean(frame, names, output_names);
}

pub fn withRowPrefixHarmMean(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeHarmonicMean(frame, names, output_names);
}

const RowCumulativeDispersion = enum { variance, stddev, sem, cv, fano };

fn withRowCumulativeDispersion(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64, comptime reduction: RowCumulativeDispersion) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (reduction) {
        .variance => try frame.ops.append(frame.allocator, .{ .row_cumulative_variance = .{
            .names = owned_names,
            .output_names = owned_outputs,
            .correction = correction,
        } }),
        .stddev => try frame.ops.append(frame.allocator, .{ .row_cumulative_stddev = .{
            .names = owned_names,
            .output_names = owned_outputs,
            .correction = correction,
        } }),
        .sem => try frame.ops.append(frame.allocator, .{ .row_cumulative_sem = .{
            .names = owned_names,
            .output_names = owned_outputs,
            .correction = correction,
        } }),
        .cv => try frame.ops.append(frame.allocator, .{ .row_cumulative_cv = .{
            .names = owned_names,
            .output_names = owned_outputs,
            .correction = correction,
        } }),
        .fano => try frame.ops.append(frame.allocator, .{ .row_cumulative_fano = .{
            .names = owned_names,
            .output_names = owned_outputs,
            .correction = correction,
        } }),
    }
}

pub fn withRowCumulativeVariance(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeDispersion(frame, names, output_names, correction, .variance);
}

pub fn withRowCumulativeVar(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeVariance(frame, names, output_names, correction);
}

pub fn withRowCumVariance(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeVariance(frame, names, output_names, correction);
}

pub fn withRowCumVar(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeVariance(frame, names, output_names, correction);
}

pub fn withRowPrefixVariance(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeVariance(frame, names, output_names, correction);
}

pub fn withRowPrefixVar(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeVariance(frame, names, output_names, correction);
}

pub fn withRowCumulativeStddev(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeDispersion(frame, names, output_names, correction, .stddev);
}

pub fn withRowCumulativeStd(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeStddev(frame, names, output_names, correction);
}

pub fn withRowCumStddev(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeStddev(frame, names, output_names, correction);
}

pub fn withRowCumStd(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeStddev(frame, names, output_names, correction);
}

pub fn withRowPrefixStddev(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeStddev(frame, names, output_names, correction);
}

pub fn withRowPrefixStd(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeStddev(frame, names, output_names, correction);
}

pub fn withRowCumulativeSem(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeDispersion(frame, names, output_names, correction, .sem);
}

pub fn withRowCumSem(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeSem(frame, names, output_names, correction);
}

pub fn withRowPrefixSem(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeSem(frame, names, output_names, correction);
}

pub fn withRowCumulativeCv(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeDispersion(frame, names, output_names, correction, .cv);
}

pub fn withRowCumCv(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeCv(frame, names, output_names, correction);
}

pub fn withRowPrefixCv(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeCv(frame, names, output_names, correction);
}

pub fn withRowCumulativeFano(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeDispersion(frame, names, output_names, correction, .fano);
}

pub fn withRowCumFano(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeFano(frame, names, output_names, correction);
}

pub fn withRowPrefixFano(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeFano(frame, names, output_names, correction);
}

pub fn withRowCumulativeIndexOfDispersion(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeFano(frame, names, output_names, correction);
}

pub fn withRowCumIndexOfDispersion(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeFano(frame, names, output_names, correction);
}

pub fn withRowPrefixIndexOfDispersion(frame: anytype, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
    return withRowCumulativeFano(frame, names, output_names, correction);
}

fn withRowCumulativeShape(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime kurtosis: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (kurtosis) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_kurtosis = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_skewness = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    }
}

pub fn withRowCumulativeSkewness(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeShape(frame, names, output_names, false);
}

pub fn withRowCumulativeSkew(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSkewness(frame, names, output_names);
}

pub fn withRowCumSkewness(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSkewness(frame, names, output_names);
}

pub fn withRowCumSkew(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSkewness(frame, names, output_names);
}

pub fn withRowPrefixSkewness(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSkewness(frame, names, output_names);
}

pub fn withRowPrefixSkew(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSkewness(frame, names, output_names);
}

pub fn withRowCumulativeKurtosis(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeShape(frame, names, output_names, true);
}

pub fn withRowCumulativeKurt(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeKurtosis(frame, names, output_names);
}

pub fn withRowCumKurtosis(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeKurtosis(frame, names, output_names);
}

pub fn withRowCumKurt(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeKurtosis(frame, names, output_names);
}

pub fn withRowPrefixKurtosis(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeKurtosis(frame, names, output_names);
}

pub fn withRowPrefixKurt(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeKurtosis(frame, names, output_names);
}

const RowCumulativeNorm = enum { rms, mean_abs, mean_square, max_abs, min_abs, l1_norm, l2_norm };

fn withRowCumulativeNorm(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime norm: RowCumulativeNorm) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (norm) {
        .rms => try frame.ops.append(frame.allocator, .{ .row_cumulative_rms = .{ .names = owned_names, .output_names = owned_outputs } }),
        .mean_abs => try frame.ops.append(frame.allocator, .{ .row_cumulative_mean_abs = .{ .names = owned_names, .output_names = owned_outputs } }),
        .mean_square => try frame.ops.append(frame.allocator, .{ .row_cumulative_mean_square = .{ .names = owned_names, .output_names = owned_outputs } }),
        .max_abs => try frame.ops.append(frame.allocator, .{ .row_cumulative_max_abs = .{ .names = owned_names, .output_names = owned_outputs } }),
        .min_abs => try frame.ops.append(frame.allocator, .{ .row_cumulative_min_abs = .{ .names = owned_names, .output_names = owned_outputs } }),
        .l1_norm => try frame.ops.append(frame.allocator, .{ .row_cumulative_l1_norm = .{ .names = owned_names, .output_names = owned_outputs } }),
        .l2_norm => try frame.ops.append(frame.allocator, .{ .row_cumulative_l2_norm = .{ .names = owned_names, .output_names = owned_outputs } }),
    }
}

pub fn withRowCumulativeRms(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNorm(frame, names, output_names, .rms);
}

pub fn withRowCumRms(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeRms(frame, names, output_names);
}

pub fn withRowPrefixRms(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeRms(frame, names, output_names);
}

pub fn withRowCumulativeMeanAbs(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNorm(frame, names, output_names, .mean_abs);
}

pub fn withRowCumulativeMeanAbsolute(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanAbs(frame, names, output_names);
}

pub fn withRowCumMeanAbs(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanAbs(frame, names, output_names);
}

pub fn withRowCumMeanAbsolute(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanAbs(frame, names, output_names);
}

pub fn withRowPrefixMeanAbs(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanAbs(frame, names, output_names);
}

pub fn withRowPrefixMeanAbsolute(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanAbs(frame, names, output_names);
}

pub fn withRowCumulativeMeanSquare(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNorm(frame, names, output_names, .mean_square);
}

pub fn withRowCumulativeMeanSquared(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanSquare(frame, names, output_names);
}

pub fn withRowCumMeanSquare(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanSquare(frame, names, output_names);
}

pub fn withRowCumMeanSquared(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanSquare(frame, names, output_names);
}

pub fn withRowPrefixMeanSquare(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanSquare(frame, names, output_names);
}

pub fn withRowPrefixMeanSquared(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMeanSquare(frame, names, output_names);
}

pub fn withRowCumulativeMaxAbs(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNorm(frame, names, output_names, .max_abs);
}

pub fn withRowCumulativeMaxAbsolute(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowCumulativeLInfNorm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowCumulativeLinfNorm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowCumMaxAbs(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowCumMaxAbsolute(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowCumLInfNorm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowCumLinfNorm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowPrefixMaxAbs(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowPrefixMaxAbsolute(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowPrefixLInfNorm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowPrefixLinfNorm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMaxAbs(frame, names, output_names);
}

pub fn withRowCumulativeMinAbs(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNorm(frame, names, output_names, .min_abs);
}

pub fn withRowCumulativeMinAbsolute(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMinAbs(frame, names, output_names);
}

pub fn withRowCumMinAbs(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMinAbs(frame, names, output_names);
}

pub fn withRowCumMinAbsolute(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMinAbs(frame, names, output_names);
}

pub fn withRowPrefixMinAbs(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMinAbs(frame, names, output_names);
}

pub fn withRowPrefixMinAbsolute(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMinAbs(frame, names, output_names);
}

pub fn withRowCumulativeL1Norm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNorm(frame, names, output_names, .l1_norm);
}

pub fn withRowCumL1Norm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeL1Norm(frame, names, output_names);
}

pub fn withRowPrefixL1Norm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeL1Norm(frame, names, output_names);
}

pub fn withRowCumulativeL2Norm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNorm(frame, names, output_names, .l2_norm);
}

pub fn withRowCumL2Norm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeL2Norm(frame, names, output_names);
}

pub fn withRowPrefixL2Norm(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeL2Norm(frame, names, output_names);
}

pub fn withRowCumulativeProduct(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_cumulative_product = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowCumprod(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeProduct(frame, names, output_names);
}

pub fn withRowCumProd(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeProduct(frame, names, output_names);
}

pub fn withRowPrefixProduct(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeProduct(frame, names, output_names);
}

pub fn withRowCumulativeMax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_cumulative_max = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowCummax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMax(frame, names, output_names);
}

pub fn withRowCumMax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMax(frame, names, output_names);
}

pub fn withRowPrefixMax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMax(frame, names, output_names);
}

pub fn withRowCumulativeMin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_cumulative_min = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowCummin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMin(frame, names, output_names);
}

pub fn withRowCumMin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMin(frame, names, output_names);
}

pub fn withRowPrefixMin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeMin(frame, names, output_names);
}

pub fn withRowCumulativeRange(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_cumulative_range = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowCumRange(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeRange(frame, names, output_names);
}

pub fn withRowPrefixRange(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeRange(frame, names, output_names);
}

pub fn withRowCumulativePtp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeRange(frame, names, output_names);
}

pub fn withRowCumPtp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeRange(frame, names, output_names);
}

pub fn withRowPrefixPtp(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeRange(frame, names, output_names);
}

pub fn withRowIqrOutlier(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_iqr_outlier = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowIqrOutliers(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowIqrOutlier(frame, names, output_names);
}

pub fn withRowTukeyOutlier(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowIqrOutlier(frame, names, output_names);
}

pub fn withRowTukeyOutliers(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowIqrOutlier(frame, names, output_names);
}

pub fn withRowMaxIndicator(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_max_indicator = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowMaxIndicators(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMaxIndicator(frame, names, output_names);
}

pub fn withRowIsMax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMaxIndicator(frame, names, output_names);
}

pub fn withRowMaxMask(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMaxIndicator(frame, names, output_names);
}

pub fn withRowMinIndicator(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_min_indicator = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowMinIndicators(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMinIndicator(frame, names, output_names);
}

pub fn withRowIsMin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMinIndicator(frame, names, output_names);
}

pub fn withRowMinMask(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMinIndicator(frame, names, output_names);
}

pub fn withRowTukeyWinsorize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_tukey_winsorize = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowTukeyWinsorized(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowTukeyWinsorize(frame, names, output_names);
}

pub fn withRowIqrWinsorize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowTukeyWinsorize(frame, names, output_names);
}

pub fn withRowIqrWinsorized(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowTukeyWinsorize(frame, names, output_names);
}

pub fn withRowMinMaxScale(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_minmax_scale = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowMinmaxScale(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMinMaxScale(frame, names, output_names);
}

pub fn withRowL2Normalize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_l2_normalize = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowL2Normalized(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowL2Normalize(frame, names, output_names);
}

pub fn withRowL1Normalize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_l1_normalize = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowL1Normalized(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowL1Normalize(frame, names, output_names);
}

pub fn withRowSumNormalize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_sum_normalize = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowProportion(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowSumNormalize(frame, names, output_names);
}

pub fn withRowShare(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowSumNormalize(frame, names, output_names);
}

pub fn withRowMeanNormalize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_mean_normalize = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowMeanNormalized(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMeanNormalize(frame, names, output_names);
}

pub fn withRowMeanRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMeanNormalize(frame, names, output_names);
}

pub fn withRowMaxAbsNormalize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    try frame.ops.append(frame.allocator, .{ .row_max_abs_normalize = .{
        .names = owned_names,
        .output_names = owned_outputs,
    } });
}

pub fn withRowMaxabsNormalize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMaxAbsNormalize(frame, names, output_names);
}

pub fn withRowLInfNormalize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMaxAbsNormalize(frame, names, output_names);
}

pub fn withRowLinfNormalize(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowMaxAbsNormalize(frame, names, output_names);
}

fn withRowSoftmaxLike(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime mode: enum { softmax, log_softmax, softmin, log_softmin }) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (mode) {
        .softmax => try frame.ops.append(frame.allocator, .{ .row_softmax = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .log_softmax => try frame.ops.append(frame.allocator, .{ .row_log_softmax = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .softmin => try frame.ops.append(frame.allocator, .{ .row_softmin = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .log_softmin => try frame.ops.append(frame.allocator, .{ .row_log_softmin = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
    }
}

pub fn withRowSoftmax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowSoftmaxLike(frame, names, output_names, .softmax);
}

pub fn withRowLogSoftmax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowSoftmaxLike(frame, names, output_names, .log_softmax);
}

pub fn withRowLogsoftmax(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowLogSoftmax(frame, names, output_names);
}

pub fn withRowSoftmin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowSoftmaxLike(frame, names, output_names, .softmin);
}

pub fn withRowLogSoftmin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowSoftmaxLike(frame, names, output_names, .log_softmin);
}

pub fn withRowLogsoftmin(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowLogSoftmin(frame, names, output_names);
}

pub fn withRowSoftmaxEntropy(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_entropy);
}

pub fn withRowSoftmaxPerplexity(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_perplexity);
}

pub fn withRowSoftmaxConfidence(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_confidence);
}

pub fn withRowSoftmaxMargin(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_margin);
}

pub fn withRowSoftmaxEvenness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_evenness);
}

pub fn withRowSoftmaxNormalizedEntropy(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowSoftmaxEvenness(frame, names, output_name);
}

pub fn withRowSoftmaxConcentration(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_concentration);
}

pub fn withRowSoftmaxNormalizedHhi(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_normalized_hhi);
}

pub fn withRowSoftmaxNormalizedHHI(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowSoftmaxNormalizedHhi(frame, names, output_name);
}

pub fn withRowSoftmaxNhhi(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowSoftmaxNormalizedHhi(frame, names, output_name);
}

pub fn withRowSoftmaxGiniImpurity(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_gini_impurity);
}

pub fn withRowSoftmaxGini(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowSoftmaxGiniImpurity(frame, names, output_name);
}

pub fn withRowSoftmaxInverseSimpson(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_inverse_simpson);
}

pub fn withRowSoftmaxSimpsonEvenness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .softmax_simpson_evenness);
}

pub fn withRowSoftmaxSimpsonEven(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowSoftmaxSimpsonEvenness(frame, names, output_name);
}

pub fn withRowLogitMargin(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .logit_margin);
}

pub fn withRowGeometricMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .geometric_mean);
}

pub fn withRowGeoMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowGeometricMean(frame, names, output_name);
}

pub fn withRowMagnitudeGeometricMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_geometric_mean);
}

pub fn withRowAbsGeometricMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeGeometricMean(frame, names, output_name);
}

pub fn withRowMagnitudeGeoMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeGeometricMean(frame, names, output_name);
}

pub fn withRowAbsGeoMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeGeometricMean(frame, names, output_name);
}

pub fn withRowHarmonicMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .harmonic_mean);
}

pub fn withRowHarmMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowHarmonicMean(frame, names, output_name);
}

pub fn withRowSkewness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .skewness);
}

pub fn withRowSkew(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowSkewness(frame, names, output_name);
}

pub fn withRowMagnitudeSkewness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_skewness);
}

pub fn withRowAbsSkewness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeSkewness(frame, names, output_name);
}

pub fn withRowMagnitudeSkew(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeSkewness(frame, names, output_name);
}

pub fn withRowAbsSkew(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeSkewness(frame, names, output_name);
}

pub fn withRowKurtosis(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .kurtosis);
}

pub fn withRowKurt(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowKurtosis(frame, names, output_name);
}

pub fn withRowMagnitudeKurtosis(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_kurtosis);
}

pub fn withRowAbsKurtosis(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeKurtosis(frame, names, output_name);
}

pub fn withRowMagnitudeKurt(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeKurtosis(frame, names, output_name);
}

pub fn withRowAbsKurt(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeKurtosis(frame, names, output_name);
}

pub fn withRowProd(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .prod);
}

pub fn withRowMin(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .min);
}

pub fn withRowMax(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .max);
}

pub fn withRowPtp(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .ptp);
}

pub fn withRowMagnitudePtp(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_ptp);
}

pub fn withRowAbsPtp(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudePtp(frame, names, output_name);
}

pub fn withRowMagnitudePeakToPeak(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudePtp(frame, names, output_name);
}

pub fn withRowAbsPeakToPeak(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudePtp(frame, names, output_name);
}

pub fn withRowMidrange(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .midrange);
}

pub fn withRowMagnitudeMidrange(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_midrange);
}

pub fn withRowAbsMidrange(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeMidrange(frame, names, output_name);
}

pub fn withRowRangeCoeff(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .range_coeff);
}

pub fn withRowRangeCoefficient(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowRangeCoeff(frame, names, output_name);
}

pub fn withRowMagnitudeRangeCoeff(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_range_coeff);
}

pub fn withRowAbsRangeCoeff(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeRangeCoeff(frame, names, output_name);
}

pub fn withRowMagnitudeRangeCoefficient(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeRangeCoeff(frame, names, output_name);
}

pub fn withRowAbsRangeCoefficient(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeRangeCoeff(frame, names, output_name);
}

pub fn withRowMeanAbs(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .mean_abs);
}

pub fn withRowHhi(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .hhi);
}

pub fn withRowHerfindahl(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowHhi(frame, names, output_name);
}

pub fn withRowHerfindahlHirschman(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowHhi(frame, names, output_name);
}

pub fn withRowMagnitudeNormalizedHhi(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_normalized_hhi);
}

pub fn withRowAbsNormalizedHhi(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeNormalizedHhi(frame, names, output_name);
}

pub fn withRowMagnitudeSparsity(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_sparsity);
}

pub fn withRowAbsSparsity(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeSparsity(frame, names, output_name);
}

pub fn withRowMagnitudeInverseSimpson(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_inverse_simpson);
}

pub fn withRowAbsInverseSimpson(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeInverseSimpson(frame, names, output_name);
}

pub fn withRowMagnitudeSimpsonEvenness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_simpson_evenness);
}

pub fn withRowAbsSimpsonEvenness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeSimpsonEvenness(frame, names, output_name);
}

pub fn withRowMagnitudeDominance(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_dominance);
}

pub fn withRowAbsDominance(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeDominance(frame, names, output_name);
}

pub fn withRowMagnitudeDominanceMargin(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_dominance_margin);
}

pub fn withRowAbsDominanceMargin(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeDominanceMargin(frame, names, output_name);
}

pub fn withRowMagnitudeEntropy(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_entropy);
}

pub fn withRowAbsEntropy(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeEntropy(frame, names, output_name);
}

pub fn withRowMagnitudePerplexity(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_perplexity);
}

pub fn withRowAbsPerplexity(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudePerplexity(frame, names, output_name);
}

pub fn withRowMagnitudeEvenness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .magnitude_evenness);
}

pub fn withRowAbsEvenness(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowMagnitudeEvenness(frame, names, output_name);
}

pub fn withRowMeanAbsDev(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .mean_abs_dev);
}

pub fn withRowGiniMeanDiff(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .gini_mean_diff);
}

pub fn withRowGiniCoefficient(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .gini_coefficient);
}

pub fn withRowGiniCoeff(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowGiniCoefficient(frame, names, output_name);
}

pub fn withRowMeanAbsDevRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .mean_abs_dev_ratio);
}

pub fn withRowRms(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .rms);
}

pub fn withRowL1Norm(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .l1_norm);
}

pub fn withRowL2Norm(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .l2_norm);
}

fn withRowNumericDispersion(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    correction: f64,
    comptime reduction: enum { variance, magnitude_variance, stddev, magnitude_stddev, sem, magnitude_sem, cv, magnitude_cv, fano, magnitude_fano },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (reduction) {
        .variance => try frame.ops.append(frame.allocator, .{ .row_variance = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .magnitude_variance => try frame.ops.append(frame.allocator, .{ .row_magnitude_variance = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .stddev => try frame.ops.append(frame.allocator, .{ .row_stddev = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .magnitude_stddev => try frame.ops.append(frame.allocator, .{ .row_magnitude_stddev = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .sem => try frame.ops.append(frame.allocator, .{ .row_sem = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .magnitude_sem => try frame.ops.append(frame.allocator, .{ .row_magnitude_sem = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .cv => try frame.ops.append(frame.allocator, .{ .row_cv = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .magnitude_cv => try frame.ops.append(frame.allocator, .{ .row_magnitude_cv = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .magnitude_fano => try frame.ops.append(frame.allocator, .{ .row_magnitude_fano = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .fano => try frame.ops.append(frame.allocator, .{ .row_fano = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
    }
}

pub fn withRowVariance(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .variance);
}

pub fn withRowVar(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowVariance(frame, names, output_name, correction);
}

pub fn withRowMagnitudeVariance(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .magnitude_variance);
}

pub fn withRowAbsVariance(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeVariance(frame, names, output_name, correction);
}

pub fn withRowMagnitudeVar(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeVariance(frame, names, output_name, correction);
}

pub fn withRowAbsVar(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeVariance(frame, names, output_name, correction);
}

pub fn withRowStddev(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .stddev);
}

pub fn withRowStd(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowStddev(frame, names, output_name, correction);
}

pub fn withRowMagnitudeStddev(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .magnitude_stddev);
}

pub fn withRowAbsStddev(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeStddev(frame, names, output_name, correction);
}

pub fn withRowMagnitudeStd(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeStddev(frame, names, output_name, correction);
}

pub fn withRowAbsStd(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeStddev(frame, names, output_name, correction);
}

pub fn withRowSem(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .sem);
}

pub fn withRowMagnitudeSem(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .magnitude_sem);
}

pub fn withRowAbsSem(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeSem(frame, names, output_name, correction);
}

pub fn withRowCv(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .cv);
}

pub fn withRowMagnitudeCv(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .magnitude_cv);
}

pub fn withRowAbsCv(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeCv(frame, names, output_name, correction);
}

pub fn withRowMagnitudeFano(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .magnitude_fano);
}

pub fn withRowAbsFano(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeFano(frame, names, output_name, correction);
}

pub fn withRowMagnitudeIndexOfDispersion(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeFano(frame, names, output_name, correction);
}

pub fn withRowAbsIndexOfDispersion(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowMagnitudeFano(frame, names, output_name, correction);
}

pub fn withRowFano(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .fano);
}

pub fn withRowIndexOfDispersion(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowFano(frame, names, output_name, correction);
}

fn withRowBoolPredicateCount(frame: anytype, names: []const []const u8, output_name: []const u8, comptime target: bool) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    if (target) {
        try frame.ops.append(frame.allocator, .{ .row_true_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_false_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } });
    }
}

pub fn withRowTrueCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolPredicateCount(frame, names, output_name, true);
}

pub fn withRowFalseCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolPredicateCount(frame, names, output_name, false);
}

fn withRowCumulativeBoolPredicateCount(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime target: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (target) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_true_count = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_false_count = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    }
}

pub fn withRowCumulativeTrueCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolPredicateCount(frame, names, output_names, true);
}

pub fn withRowCumTrueCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeTrueCount(frame, names, output_names);
}

pub fn withRowPrefixTrueCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeTrueCount(frame, names, output_names);
}

pub fn withRowCumulativeFalseCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolPredicateCount(frame, names, output_names, false);
}

pub fn withRowCumFalseCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFalseCount(frame, names, output_names);
}

pub fn withRowPrefixFalseCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFalseCount(frame, names, output_names);
}

fn withRowCumulativeBoolPredicateRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8, comptime target: bool) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    if (target) {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_true_ratio = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_cumulative_false_ratio = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } });
    }
}

pub fn withRowCumulativeTrueRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolPredicateRatio(frame, names, output_names, true);
}

pub fn withRowCumTrueRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeTrueRatio(frame, names, output_names);
}

pub fn withRowPrefixTrueRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeTrueRatio(frame, names, output_names);
}

pub fn withRowCumulativeFalseRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolPredicateRatio(frame, names, output_names, false);
}

pub fn withRowCumFalseRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFalseRatio(frame, names, output_names);
}

pub fn withRowPrefixFalseRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFalseRatio(frame, names, output_names);
}

fn withRowBoolReduction(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: enum { any_true, all_true, any_false, all_false },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (reduction) {
        .any_true => try frame.ops.append(frame.allocator, .{ .row_any_true = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .all_true => try frame.ops.append(frame.allocator, .{ .row_all_true = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .any_false => try frame.ops.append(frame.allocator, .{ .row_any_false = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .all_false => try frame.ops.append(frame.allocator, .{ .row_all_false = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowAnyTrue(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolReduction(frame, names, output_name, .any_true);
}

pub fn withRowAllTrue(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolReduction(frame, names, output_name, .all_true);
}

pub fn withRowAnyFalse(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolReduction(frame, names, output_name, .any_false);
}

pub fn withRowAllFalse(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolReduction(frame, names, output_name, .all_false);
}

fn withRowCumulativeBoolReduction(
    frame: anytype,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime reduction: enum { any_true, all_true, any_false, all_false },
) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (reduction) {
        .any_true => try frame.ops.append(frame.allocator, .{ .row_cumulative_any_true = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .all_true => try frame.ops.append(frame.allocator, .{ .row_cumulative_all_true = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .any_false => try frame.ops.append(frame.allocator, .{ .row_cumulative_any_false = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .all_false => try frame.ops.append(frame.allocator, .{ .row_cumulative_all_false = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
    }
}

pub fn withRowCumulativeAnyTrue(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolReduction(frame, names, output_names, .any_true);
}

pub fn withRowCumAnyTrue(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyTrue(frame, names, output_names);
}

pub fn withRowPrefixAnyTrue(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyTrue(frame, names, output_names);
}

pub fn withRowCumulativeAllTrue(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolReduction(frame, names, output_names, .all_true);
}

pub fn withRowCumAllTrue(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllTrue(frame, names, output_names);
}

pub fn withRowPrefixAllTrue(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllTrue(frame, names, output_names);
}

pub fn withRowCumulativeAnyFalse(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolReduction(frame, names, output_names, .any_false);
}

pub fn withRowCumAnyFalse(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyFalse(frame, names, output_names);
}

pub fn withRowPrefixAnyFalse(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyFalse(frame, names, output_names);
}

pub fn withRowCumulativeAllFalse(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolReduction(frame, names, output_names, .all_false);
}

pub fn withRowCumAllFalse(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllFalse(frame, names, output_names);
}

pub fn withRowPrefixAllFalse(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllFalse(frame, names, output_names);
}

fn withRowBoolMatchIndex(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime search: enum { first_true, last_true, first_false, last_false },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (search) {
        .first_true => try frame.ops.append(frame.allocator, .{ .row_first_true_index = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .last_true => try frame.ops.append(frame.allocator, .{ .row_last_true_index = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .first_false => try frame.ops.append(frame.allocator, .{ .row_first_false_index = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .last_false => try frame.ops.append(frame.allocator, .{ .row_last_false_index = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowFirstTrueIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolMatchIndex(frame, names, output_name, .first_true);
}

pub fn withRowLastTrueIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolMatchIndex(frame, names, output_name, .last_true);
}

pub fn withRowFirstFalseIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolMatchIndex(frame, names, output_name, .first_false);
}

pub fn withRowLastFalseIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolMatchIndex(frame, names, output_name, .last_false);
}

fn withRowCumulativeBoolMatchIndex(
    frame: anytype,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime search: enum { first_true, last_true, first_false, last_false },
) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (search) {
        .first_true => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_true_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_true => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_true_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_false => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_false_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_false => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_false_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
    }
}

pub fn withRowCumulativeFirstTrueIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolMatchIndex(frame, names, output_names, .first_true);
}

pub fn withRowPrefixFirstTrueIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstTrueIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastTrueIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolMatchIndex(frame, names, output_names, .last_true);
}

pub fn withRowPrefixLastTrueIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastTrueIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstFalseIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolMatchIndex(frame, names, output_names, .first_false);
}

pub fn withRowPrefixFirstFalseIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstFalseIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastFalseIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeBoolMatchIndex(frame, names, output_names, .last_false);
}

pub fn withRowPrefixLastFalseIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastFalseIndex(frame, names, output_names);
}

fn withRowBoolPredicateRatio(frame: anytype, names: []const []const u8, output_name: []const u8, comptime target: bool) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    if (target) {
        try frame.ops.append(frame.allocator, .{ .row_true_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } });
    } else {
        try frame.ops.append(frame.allocator, .{ .row_false_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } });
    }
}

pub fn withRowTrueRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolPredicateRatio(frame, names, output_name, true);
}

pub fn withRowFalseRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowBoolPredicateRatio(frame, names, output_name, false);
}

fn withRowNumericPredicateCount(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime tag_name: enum { nan, inf, positive_inf, negative_inf, zero, positive_zero, negative_zero, non_zero, positive, signbit, negative, finite, normal, subnormal, non_finite },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (tag_name) {
        .nan => try frame.ops.append(frame.allocator, .{ .row_nan_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .inf => try frame.ops.append(frame.allocator, .{ .row_inf_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .positive_inf => try frame.ops.append(frame.allocator, .{ .row_positive_inf_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .negative_inf => try frame.ops.append(frame.allocator, .{ .row_negative_inf_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .zero => try frame.ops.append(frame.allocator, .{ .row_zero_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .positive_zero => try frame.ops.append(frame.allocator, .{ .row_positive_zero_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .negative_zero => try frame.ops.append(frame.allocator, .{ .row_negative_zero_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .non_zero => try frame.ops.append(frame.allocator, .{ .row_non_zero_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .positive => try frame.ops.append(frame.allocator, .{ .row_positive_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .signbit => try frame.ops.append(frame.allocator, .{ .row_signbit_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .negative => try frame.ops.append(frame.allocator, .{ .row_negative_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .finite => try frame.ops.append(frame.allocator, .{ .row_finite_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .normal => try frame.ops.append(frame.allocator, .{ .row_normal_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .subnormal => try frame.ops.append(frame.allocator, .{ .row_subnormal_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .non_finite => try frame.ops.append(frame.allocator, .{ .row_non_finite_count = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowNaNCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .nan);
}

fn withRowNumericPredicateRatio(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime tag_name: enum { nan, inf, positive_inf, negative_inf, zero, positive_zero, negative_zero, non_zero, positive, signbit, negative, finite, normal, subnormal, non_finite },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (tag_name) {
        .nan => try frame.ops.append(frame.allocator, .{ .row_nan_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .inf => try frame.ops.append(frame.allocator, .{ .row_inf_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .positive_inf => try frame.ops.append(frame.allocator, .{ .row_positive_inf_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .negative_inf => try frame.ops.append(frame.allocator, .{ .row_negative_inf_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .zero => try frame.ops.append(frame.allocator, .{ .row_zero_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .positive_zero => try frame.ops.append(frame.allocator, .{ .row_positive_zero_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .negative_zero => try frame.ops.append(frame.allocator, .{ .row_negative_zero_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .non_zero => try frame.ops.append(frame.allocator, .{ .row_non_zero_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .positive => try frame.ops.append(frame.allocator, .{ .row_positive_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .signbit => try frame.ops.append(frame.allocator, .{ .row_signbit_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .negative => try frame.ops.append(frame.allocator, .{ .row_negative_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .finite => try frame.ops.append(frame.allocator, .{ .row_finite_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .normal => try frame.ops.append(frame.allocator, .{ .row_normal_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .subnormal => try frame.ops.append(frame.allocator, .{ .row_subnormal_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
        .non_finite => try frame.ops.append(frame.allocator, .{ .row_non_finite_ratio = .{
            .names = owned_names,
            .output_name = owned_output,
        } }),
    }
}

pub fn withRowNaNRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .nan);
}

pub fn withRowNanRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNaNRatio(frame, names, output_name);
}

pub fn withRowInfCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .inf);
}

pub fn withRowInfRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .inf);
}

pub fn withRowPositiveInfCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .positive_inf);
}

pub fn withRowNegativeInfCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .negative_inf);
}

pub fn withRowPositiveInfRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .positive_inf);
}

pub fn withRowNegativeInfRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .negative_inf);
}

pub fn withRowZeroCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .zero);
}

pub fn withRowZeroRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .zero);
}

pub fn withRowPositiveZeroCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .positive_zero);
}

pub fn withRowNegativeZeroCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .negative_zero);
}

pub fn withRowPositiveZeroRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .positive_zero);
}

pub fn withRowNegativeZeroRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .negative_zero);
}

pub fn withRowNonZeroCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .non_zero);
}

pub fn withRowNonZeroRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .non_zero);
}

fn withRowNumericPredicateReduction(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime tag_name: enum { zero, non_zero, positive_zero, negative_zero, positive, signbit, negative, nan, inf, positive_inf, negative_inf, finite, normal, subnormal, non_finite },
    comptime reduction: enum { any, all },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (tag_name) {
        .zero => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_zero = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_zero = .{ .names = owned_names, .output_name = owned_output } });
        },
        .non_zero => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_non_zero = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_non_zero = .{ .names = owned_names, .output_name = owned_output } });
        },
        .positive_zero => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_positive_zero = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_positive_zero = .{ .names = owned_names, .output_name = owned_output } });
        },
        .negative_zero => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_negative_zero = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_negative_zero = .{ .names = owned_names, .output_name = owned_output } });
        },
        .positive => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_positive = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_positive = .{ .names = owned_names, .output_name = owned_output } });
        },
        .signbit => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_signbit = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_signbit = .{ .names = owned_names, .output_name = owned_output } });
        },
        .negative => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_negative = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_negative = .{ .names = owned_names, .output_name = owned_output } });
        },
        .nan => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_nan = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_nan = .{ .names = owned_names, .output_name = owned_output } });
        },
        .inf => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_inf = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_inf = .{ .names = owned_names, .output_name = owned_output } });
        },
        .positive_inf => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_positive_inf = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_positive_inf = .{ .names = owned_names, .output_name = owned_output } });
        },
        .negative_inf => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_negative_inf = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_negative_inf = .{ .names = owned_names, .output_name = owned_output } });
        },
        .finite => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_finite = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_finite = .{ .names = owned_names, .output_name = owned_output } });
        },
        .normal => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_normal = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_normal = .{ .names = owned_names, .output_name = owned_output } });
        },
        .subnormal => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_subnormal = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_subnormal = .{ .names = owned_names, .output_name = owned_output } });
        },
        .non_finite => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_any_non_finite = .{ .names = owned_names, .output_name = owned_output } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_all_non_finite = .{ .names = owned_names, .output_name = owned_output } });
        },
    }
}

fn withRowCumulativeNumericPredicateReduction(
    frame: anytype,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime tag_name: enum { zero, non_zero, positive_zero, negative_zero, positive, signbit, negative, nan, inf, positive_inf, negative_inf, finite, normal, subnormal, non_finite },
    comptime reduction: enum { any, all },
) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (tag_name) {
        .zero => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_zero = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_zero = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .non_zero => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_non_zero = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_non_zero = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .positive_zero => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_positive_zero = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_positive_zero = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .negative_zero => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_negative_zero = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_negative_zero = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .positive => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_positive = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_positive = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .signbit => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_signbit = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_signbit = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .negative => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_negative = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_negative = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .nan => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_nan = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_nan = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .inf => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_inf = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_inf = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .positive_inf => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_positive_inf = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_positive_inf = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .negative_inf => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_negative_inf = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_negative_inf = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .finite => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_finite = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_finite = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .normal => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_normal = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_normal = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .subnormal => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_subnormal = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_subnormal = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .non_finite => if (reduction == .any) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_any_non_finite = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_all_non_finite = .{ .names = owned_names, .output_names = owned_outputs } });
        },
    }
}

pub fn withRowAnyZero(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .zero, .any);
}

pub fn withRowAllZero(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .zero, .all);
}

pub fn withRowCumulativeAnyZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .zero, .any);
}

pub fn withRowCumAnyZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyZero(frame, names, output_names);
}

pub fn withRowPrefixAnyZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyZero(frame, names, output_names);
}

pub fn withRowCumulativeAllZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .zero, .all);
}

pub fn withRowCumAllZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllZero(frame, names, output_names);
}

pub fn withRowPrefixAllZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllZero(frame, names, output_names);
}

pub fn withRowAnyNonZero(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .non_zero, .any);
}

pub fn withRowAllNonZero(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .non_zero, .all);
}

pub fn withRowCumulativeAnyNonZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .non_zero, .any);
}

pub fn withRowCumAnyNonZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNonZero(frame, names, output_names);
}

pub fn withRowPrefixAnyNonZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNonZero(frame, names, output_names);
}

pub fn withRowCumulativeAllNonZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .non_zero, .all);
}

pub fn withRowCumAllNonZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNonZero(frame, names, output_names);
}

pub fn withRowPrefixAllNonZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNonZero(frame, names, output_names);
}

pub fn withRowAnyPositiveZero(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .positive_zero, .any);
}

pub fn withRowAllPositiveZero(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .positive_zero, .all);
}

pub fn withRowCumulativeAnyPositiveZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .positive_zero, .any);
}

pub fn withRowCumAnyPositiveZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyPositiveZero(frame, names, output_names);
}

pub fn withRowPrefixAnyPositiveZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyPositiveZero(frame, names, output_names);
}

pub fn withRowCumulativeAllPositiveZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .positive_zero, .all);
}

pub fn withRowCumAllPositiveZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllPositiveZero(frame, names, output_names);
}

pub fn withRowPrefixAllPositiveZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllPositiveZero(frame, names, output_names);
}

pub fn withRowAnyNegativeZero(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .negative_zero, .any);
}

pub fn withRowAllNegativeZero(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .negative_zero, .all);
}

pub fn withRowCumulativeAnyNegativeZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .negative_zero, .any);
}

pub fn withRowCumAnyNegativeZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNegativeZero(frame, names, output_names);
}

pub fn withRowPrefixAnyNegativeZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNegativeZero(frame, names, output_names);
}

pub fn withRowCumulativeAllNegativeZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .negative_zero, .all);
}

pub fn withRowCumAllNegativeZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNegativeZero(frame, names, output_names);
}

pub fn withRowPrefixAllNegativeZero(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNegativeZero(frame, names, output_names);
}

pub fn withRowAnyPositive(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .positive, .any);
}

pub fn withRowAllPositive(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .positive, .all);
}

pub fn withRowCumulativeAnyPositive(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .positive, .any);
}

pub fn withRowCumAnyPositive(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyPositive(frame, names, output_names);
}

pub fn withRowPrefixAnyPositive(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyPositive(frame, names, output_names);
}

pub fn withRowCumulativeAllPositive(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .positive, .all);
}

pub fn withRowCumAllPositive(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllPositive(frame, names, output_names);
}

pub fn withRowPrefixAllPositive(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllPositive(frame, names, output_names);
}

pub fn withRowAnySignBit(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .signbit, .any);
}

pub fn withRowAllSignBit(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .signbit, .all);
}

pub fn withRowCumulativeAnySignBit(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .signbit, .any);
}

pub fn withRowCumAnySignBit(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnySignBit(frame, names, output_names);
}

pub fn withRowPrefixAnySignBit(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnySignBit(frame, names, output_names);
}

pub fn withRowCumulativeAllSignBit(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .signbit, .all);
}

pub fn withRowCumAllSignBit(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllSignBit(frame, names, output_names);
}

pub fn withRowPrefixAllSignBit(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllSignBit(frame, names, output_names);
}

pub fn withRowAnyNegative(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .negative, .any);
}

pub fn withRowAllNegative(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .negative, .all);
}

pub fn withRowCumulativeAnyNegative(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .negative, .any);
}

pub fn withRowCumAnyNegative(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNegative(frame, names, output_names);
}

pub fn withRowPrefixAnyNegative(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNegative(frame, names, output_names);
}

pub fn withRowCumulativeAllNegative(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .negative, .all);
}

pub fn withRowCumAllNegative(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNegative(frame, names, output_names);
}

pub fn withRowPrefixAllNegative(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNegative(frame, names, output_names);
}

pub fn withRowAnyNaN(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .nan, .any);
}

pub fn withRowAllNaN(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .nan, .all);
}

pub fn withRowCumulativeAnyNaN(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .nan, .any);
}

pub fn withRowCumAnyNaN(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNaN(frame, names, output_names);
}

pub fn withRowPrefixAnyNaN(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNaN(frame, names, output_names);
}

pub fn withRowCumulativeAllNaN(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .nan, .all);
}

pub fn withRowCumAllNaN(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNaN(frame, names, output_names);
}

pub fn withRowPrefixAllNaN(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNaN(frame, names, output_names);
}

pub fn withRowAnyInf(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .inf, .any);
}

pub fn withRowAllInf(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .inf, .all);
}

pub fn withRowCumulativeAnyInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .inf, .any);
}

pub fn withRowCumAnyInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyInf(frame, names, output_names);
}

pub fn withRowPrefixAnyInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyInf(frame, names, output_names);
}

pub fn withRowCumulativeAllInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .inf, .all);
}

pub fn withRowCumAllInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllInf(frame, names, output_names);
}

pub fn withRowPrefixAllInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllInf(frame, names, output_names);
}

pub fn withRowAnyPositiveInf(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .positive_inf, .any);
}

pub fn withRowAllPositiveInf(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .positive_inf, .all);
}

pub fn withRowCumulativeAnyPositiveInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .positive_inf, .any);
}

pub fn withRowCumAnyPositiveInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyPositiveInf(frame, names, output_names);
}

pub fn withRowPrefixAnyPositiveInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyPositiveInf(frame, names, output_names);
}

pub fn withRowCumulativeAllPositiveInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .positive_inf, .all);
}

pub fn withRowCumAllPositiveInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllPositiveInf(frame, names, output_names);
}

pub fn withRowPrefixAllPositiveInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllPositiveInf(frame, names, output_names);
}

pub fn withRowAnyNegativeInf(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .negative_inf, .any);
}

pub fn withRowAllNegativeInf(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .negative_inf, .all);
}

pub fn withRowCumulativeAnyNegativeInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .negative_inf, .any);
}

pub fn withRowCumAnyNegativeInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNegativeInf(frame, names, output_names);
}

pub fn withRowPrefixAnyNegativeInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNegativeInf(frame, names, output_names);
}

pub fn withRowCumulativeAllNegativeInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .negative_inf, .all);
}

pub fn withRowCumAllNegativeInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNegativeInf(frame, names, output_names);
}

pub fn withRowPrefixAllNegativeInf(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNegativeInf(frame, names, output_names);
}

pub fn withRowAnyFinite(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .finite, .any);
}

pub fn withRowAllFinite(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .finite, .all);
}

pub fn withRowCumulativeAnyFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .finite, .any);
}

pub fn withRowCumAnyFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyFinite(frame, names, output_names);
}

pub fn withRowPrefixAnyFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyFinite(frame, names, output_names);
}

pub fn withRowCumulativeAllFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .finite, .all);
}

pub fn withRowCumAllFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllFinite(frame, names, output_names);
}

pub fn withRowPrefixAllFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllFinite(frame, names, output_names);
}

pub fn withRowAnyNormal(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .normal, .any);
}

pub fn withRowAllNormal(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .normal, .all);
}

pub fn withRowCumulativeAnyNormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .normal, .any);
}

pub fn withRowCumAnyNormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNormal(frame, names, output_names);
}

pub fn withRowPrefixAnyNormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNormal(frame, names, output_names);
}

pub fn withRowCumulativeAllNormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .normal, .all);
}

pub fn withRowCumAllNormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNormal(frame, names, output_names);
}

pub fn withRowPrefixAllNormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNormal(frame, names, output_names);
}

pub fn withRowAnySubnormal(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .subnormal, .any);
}

pub fn withRowAllSubnormal(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .subnormal, .all);
}

pub fn withRowCumulativeAnySubnormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .subnormal, .any);
}

pub fn withRowCumAnySubnormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnySubnormal(frame, names, output_names);
}

pub fn withRowPrefixAnySubnormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnySubnormal(frame, names, output_names);
}

pub fn withRowCumulativeAllSubnormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .subnormal, .all);
}

pub fn withRowCumAllSubnormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllSubnormal(frame, names, output_names);
}

pub fn withRowPrefixAllSubnormal(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllSubnormal(frame, names, output_names);
}

pub fn withRowAnyNonFinite(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .non_finite, .any);
}

pub fn withRowAllNonFinite(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateReduction(frame, names, output_name, .non_finite, .all);
}

pub fn withRowCumulativeAnyNonFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .non_finite, .any);
}

pub fn withRowCumAnyNonFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNonFinite(frame, names, output_names);
}

pub fn withRowPrefixAnyNonFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAnyNonFinite(frame, names, output_names);
}

pub fn withRowCumulativeAllNonFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateReduction(frame, names, output_names, .non_finite, .all);
}

pub fn withRowCumAllNonFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNonFinite(frame, names, output_names);
}

pub fn withRowPrefixAllNonFinite(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeAllNonFinite(frame, names, output_names);
}

fn withRowNumericPredicateIndex(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime search: enum { first_nan, last_nan, first_inf, last_inf, first_positive_inf, last_positive_inf, first_negative_inf, last_negative_inf, first_finite, last_finite, first_normal, last_normal, first_subnormal, last_subnormal, first_non_finite, last_non_finite, first_positive_zero, last_positive_zero, first_negative_zero, last_negative_zero, first_signbit, last_signbit, first_zero, last_zero, first_non_zero, last_non_zero, first_positive, last_positive, first_negative, last_negative },
) DeviceDataError!void {
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_output = try frame.allocator.dupe(u8, output_name);
    errdefer frame.allocator.free(owned_output);
    switch (search) {
        .first_nan => try frame.ops.append(frame.allocator, .{ .row_first_nan_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_nan => try frame.ops.append(frame.allocator, .{ .row_last_nan_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_inf => try frame.ops.append(frame.allocator, .{ .row_first_inf_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_inf => try frame.ops.append(frame.allocator, .{ .row_last_inf_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_positive_inf => try frame.ops.append(frame.allocator, .{ .row_first_positive_inf_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_positive_inf => try frame.ops.append(frame.allocator, .{ .row_last_positive_inf_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_negative_inf => try frame.ops.append(frame.allocator, .{ .row_first_negative_inf_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_negative_inf => try frame.ops.append(frame.allocator, .{ .row_last_negative_inf_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_finite => try frame.ops.append(frame.allocator, .{ .row_first_finite_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_finite => try frame.ops.append(frame.allocator, .{ .row_last_finite_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_normal => try frame.ops.append(frame.allocator, .{ .row_first_normal_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_normal => try frame.ops.append(frame.allocator, .{ .row_last_normal_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_subnormal => try frame.ops.append(frame.allocator, .{ .row_first_subnormal_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_subnormal => try frame.ops.append(frame.allocator, .{ .row_last_subnormal_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_non_finite => try frame.ops.append(frame.allocator, .{ .row_first_non_finite_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_non_finite => try frame.ops.append(frame.allocator, .{ .row_last_non_finite_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_positive_zero => try frame.ops.append(frame.allocator, .{ .row_first_positive_zero_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_positive_zero => try frame.ops.append(frame.allocator, .{ .row_last_positive_zero_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_negative_zero => try frame.ops.append(frame.allocator, .{ .row_first_negative_zero_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_negative_zero => try frame.ops.append(frame.allocator, .{ .row_last_negative_zero_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_signbit => try frame.ops.append(frame.allocator, .{ .row_first_signbit_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_signbit => try frame.ops.append(frame.allocator, .{ .row_last_signbit_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_zero => try frame.ops.append(frame.allocator, .{ .row_first_zero_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_zero => try frame.ops.append(frame.allocator, .{ .row_last_zero_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_non_zero => try frame.ops.append(frame.allocator, .{ .row_first_non_zero_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_non_zero => try frame.ops.append(frame.allocator, .{ .row_last_non_zero_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_positive => try frame.ops.append(frame.allocator, .{ .row_first_positive_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_positive => try frame.ops.append(frame.allocator, .{ .row_last_positive_index = .{ .names = owned_names, .output_name = owned_output } }),
        .first_negative => try frame.ops.append(frame.allocator, .{ .row_first_negative_index = .{ .names = owned_names, .output_name = owned_output } }),
        .last_negative => try frame.ops.append(frame.allocator, .{ .row_last_negative_index = .{ .names = owned_names, .output_name = owned_output } }),
    }
}

pub fn withRowFirstNaNIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_nan);
}

pub fn withRowFirstNanIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowFirstNaNIndex(frame, names, output_name);
}

pub fn withRowLastNaNIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_nan);
}

pub fn withRowLastNanIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowLastNaNIndex(frame, names, output_name);
}

pub fn withRowFirstInfIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_inf);
}

pub fn withRowLastInfIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_inf);
}

pub fn withRowFirstPositiveInfIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_positive_inf);
}

pub fn withRowLastPositiveInfIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_positive_inf);
}

pub fn withRowFirstNegativeInfIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_negative_inf);
}

pub fn withRowLastNegativeInfIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_negative_inf);
}

pub fn withRowFirstPositiveZeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_positive_zero);
}

pub fn withRowLastPositiveZeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_positive_zero);
}

pub fn withRowFirstNegativeZeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_negative_zero);
}

pub fn withRowLastNegativeZeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_negative_zero);
}

pub fn withRowFirstSignBitIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_signbit);
}

pub fn withRowLastSignBitIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_signbit);
}

pub fn withRowFirstFiniteIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_finite);
}

pub fn withRowLastFiniteIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_finite);
}

pub fn withRowFirstNormalIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_normal);
}

pub fn withRowLastNormalIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_normal);
}

pub fn withRowFirstSubnormalIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_subnormal);
}

pub fn withRowLastSubnormalIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_subnormal);
}

pub fn withRowFirstNonFiniteIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_non_finite);
}

pub fn withRowFirstNonfiniteIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowFirstNonFiniteIndex(frame, names, output_name);
}

pub fn withRowLastNonFiniteIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_non_finite);
}

pub fn withRowLastNonfiniteIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowLastNonFiniteIndex(frame, names, output_name);
}

pub fn withRowFirstZeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_zero);
}

pub fn withRowLastZeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_zero);
}

pub fn withRowFirstNonZeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_non_zero);
}

pub fn withRowFirstNonzeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowFirstNonZeroIndex(frame, names, output_name);
}

pub fn withRowLastNonZeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_non_zero);
}

pub fn withRowLastNonzeroIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowLastNonZeroIndex(frame, names, output_name);
}

pub fn withRowFirstPositiveIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_positive);
}

pub fn withRowLastPositiveIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_positive);
}

pub fn withRowFirstNegativeIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .first_negative);
}

pub fn withRowLastNegativeIndex(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateIndex(frame, names, output_name, .last_negative);
}

pub fn withRowPositiveCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .positive);
}

pub fn withRowPositiveRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .positive);
}

pub fn withRowSignBitCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .signbit);
}

pub fn withRowSignBitRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .signbit);
}

pub fn withRowNegativeCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .negative);
}

pub fn withRowNegativeRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .negative);
}

fn withRowCumulativeNumericPredicate(
    frame: anytype,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime tag_name: enum { positive_zero, negative_zero, signbit, nan, inf, positive_inf, negative_inf, finite, normal, subnormal, non_finite, zero, non_zero, positive, negative },
    comptime ratio: bool,
) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (tag_name) {
        .positive_zero => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_positive_zero_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_positive_zero_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .negative_zero => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_negative_zero_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_negative_zero_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .signbit => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_signbit_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_signbit_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .nan => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_nan_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_nan_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .inf => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_inf_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_inf_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .positive_inf => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_positive_inf_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_positive_inf_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .negative_inf => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_negative_inf_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_negative_inf_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .finite => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_finite_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_finite_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .normal => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_normal_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_normal_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .subnormal => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_subnormal_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_subnormal_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .non_finite => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_non_finite_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_non_finite_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .zero => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_zero_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_zero_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .non_zero => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_non_zero_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_non_zero_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .positive => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_positive_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_positive_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
        .negative => if (ratio) {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_negative_ratio = .{ .names = owned_names, .output_names = owned_outputs } });
        } else {
            try frame.ops.append(frame.allocator, .{ .row_cumulative_negative_count = .{ .names = owned_names, .output_names = owned_outputs } });
        },
    }
}

pub fn withRowCumulativePositiveZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .positive_zero, false);
}

pub fn withRowCumPositiveZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveZeroCount(frame, names, output_names);
}

pub fn withRowPrefixPositiveZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveZeroCount(frame, names, output_names);
}

pub fn withRowCumulativeNegativeZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .negative_zero, false);
}

pub fn withRowCumNegativeZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeZeroCount(frame, names, output_names);
}

pub fn withRowPrefixNegativeZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeZeroCount(frame, names, output_names);
}

pub fn withRowCumulativeSignBitCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .signbit, false);
}

pub fn withRowCumSignBitCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSignBitCount(frame, names, output_names);
}

pub fn withRowPrefixSignBitCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSignBitCount(frame, names, output_names);
}

pub fn withRowCumulativeNaNCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .nan, false);
}

pub fn withRowCumNaNCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNaNCount(frame, names, output_names);
}

pub fn withRowPrefixNaNCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNaNCount(frame, names, output_names);
}

pub fn withRowCumulativeInfCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .inf, false);
}

pub fn withRowCumInfCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeInfCount(frame, names, output_names);
}

pub fn withRowPrefixInfCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeInfCount(frame, names, output_names);
}

pub fn withRowCumulativePositiveInfCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .positive_inf, false);
}

pub fn withRowCumPositiveInfCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveInfCount(frame, names, output_names);
}

pub fn withRowPrefixPositiveInfCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveInfCount(frame, names, output_names);
}

pub fn withRowCumulativeNegativeInfCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .negative_inf, false);
}

pub fn withRowCumNegativeInfCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeInfCount(frame, names, output_names);
}

pub fn withRowPrefixNegativeInfCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeInfCount(frame, names, output_names);
}

pub fn withRowCumulativeFiniteCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .finite, false);
}

pub fn withRowCumFiniteCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFiniteCount(frame, names, output_names);
}

pub fn withRowPrefixFiniteCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFiniteCount(frame, names, output_names);
}

pub fn withRowCumulativeNormalCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .normal, false);
}

pub fn withRowCumNormalCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNormalCount(frame, names, output_names);
}

pub fn withRowPrefixNormalCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNormalCount(frame, names, output_names);
}

pub fn withRowCumulativeSubnormalCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .subnormal, false);
}

pub fn withRowCumSubnormalCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSubnormalCount(frame, names, output_names);
}

pub fn withRowPrefixSubnormalCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSubnormalCount(frame, names, output_names);
}

pub fn withRowCumulativeNonFiniteCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .non_finite, false);
}

pub fn withRowCumNonFiniteCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNonFiniteCount(frame, names, output_names);
}

pub fn withRowPrefixNonFiniteCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNonFiniteCount(frame, names, output_names);
}

pub fn withRowCumulativeZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .zero, false);
}

pub fn withRowCumZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeZeroCount(frame, names, output_names);
}

pub fn withRowPrefixZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeZeroCount(frame, names, output_names);
}

fn withRowCumulativeNumericPredicateIndex(
    frame: anytype,
    names: []const []const u8,
    output_names: []const []const u8,
    comptime search: enum { first_nan, last_nan, first_inf, last_inf, first_positive_inf, last_positive_inf, first_negative_inf, last_negative_inf, first_finite, last_finite, first_normal, last_normal, first_subnormal, last_subnormal, first_non_finite, last_non_finite, first_zero, last_zero, first_positive_zero, last_positive_zero, first_negative_zero, last_negative_zero, first_non_zero, last_non_zero, first_positive, last_positive, first_signbit, last_signbit, first_negative, last_negative },
) DeviceDataError!void {
    if (names.len != output_names.len) return error.LengthMismatch;
    const owned_names = try cloneNameList(frame.allocator, names);
    errdefer {
        for (owned_names) |name| frame.allocator.free(name);
        frame.allocator.free(owned_names);
    }
    const owned_outputs = try cloneNameList(frame.allocator, output_names);
    errdefer {
        for (owned_outputs) |name| frame.allocator.free(name);
        frame.allocator.free(owned_outputs);
    }
    switch (search) {
        .first_nan => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_nan_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_nan => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_nan_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_inf => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_inf_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_inf => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_inf_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_positive_inf => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_positive_inf_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_positive_inf => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_positive_inf_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_negative_inf => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_negative_inf_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_negative_inf => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_negative_inf_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_finite => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_finite_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_finite => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_finite_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_normal => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_normal_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_normal => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_normal_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_subnormal => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_subnormal_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_subnormal => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_subnormal_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_non_finite => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_non_finite_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_non_finite => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_non_finite_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_zero => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_zero_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_zero => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_zero_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_positive_zero => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_positive_zero_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_positive_zero => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_positive_zero_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_negative_zero => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_negative_zero_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_negative_zero => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_negative_zero_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_non_zero => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_non_zero_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_non_zero => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_non_zero_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_positive => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_positive_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_positive => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_positive_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_signbit => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_signbit_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_signbit => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_signbit_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .first_negative => try frame.ops.append(frame.allocator, .{ .row_cumulative_first_negative_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
        .last_negative => try frame.ops.append(frame.allocator, .{ .row_cumulative_last_negative_index = .{
            .names = owned_names,
            .output_names = owned_outputs,
        } }),
    }
}

pub fn withRowCumulativeFirstNaNIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_nan);
}

pub fn withRowPrefixFirstNaNIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstNaNIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastNaNIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_nan);
}

pub fn withRowPrefixLastNaNIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastNaNIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_inf);
}

pub fn withRowPrefixFirstInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstInfIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_inf);
}

pub fn withRowPrefixLastInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastInfIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstPositiveInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_positive_inf);
}

pub fn withRowPrefixFirstPositiveInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstPositiveInfIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastPositiveInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_positive_inf);
}

pub fn withRowPrefixLastPositiveInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastPositiveInfIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstNegativeInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_negative_inf);
}

pub fn withRowPrefixFirstNegativeInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstNegativeInfIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastNegativeInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_negative_inf);
}

pub fn withRowPrefixLastNegativeInfIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastNegativeInfIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstFiniteIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_finite);
}

pub fn withRowPrefixFirstFiniteIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstFiniteIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastFiniteIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_finite);
}

pub fn withRowPrefixLastFiniteIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastFiniteIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstNormalIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_normal);
}

pub fn withRowPrefixFirstNormalIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstNormalIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastNormalIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_normal);
}

pub fn withRowPrefixLastNormalIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastNormalIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstSubnormalIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_subnormal);
}

pub fn withRowPrefixFirstSubnormalIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstSubnormalIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastSubnormalIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_subnormal);
}

pub fn withRowPrefixLastSubnormalIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastSubnormalIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstNonFiniteIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_non_finite);
}

pub fn withRowPrefixFirstNonFiniteIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstNonFiniteIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastNonFiniteIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_non_finite);
}

pub fn withRowPrefixLastNonFiniteIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastNonFiniteIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_zero);
}

pub fn withRowPrefixFirstZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstZeroIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_zero);
}

pub fn withRowPrefixLastZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastZeroIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstPositiveZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_positive_zero);
}

pub fn withRowPrefixFirstPositiveZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstPositiveZeroIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastPositiveZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_positive_zero);
}

pub fn withRowPrefixLastPositiveZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastPositiveZeroIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstNegativeZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_negative_zero);
}

pub fn withRowPrefixFirstNegativeZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstNegativeZeroIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastNegativeZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_negative_zero);
}

pub fn withRowPrefixLastNegativeZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastNegativeZeroIndex(frame, names, output_names);
}

pub fn withRowCumulativeNonZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .non_zero, false);
}

pub fn withRowCumNonZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNonZeroCount(frame, names, output_names);
}

pub fn withRowPrefixNonZeroCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNonZeroCount(frame, names, output_names);
}

pub fn withRowCumulativeFirstNonZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_non_zero);
}

pub fn withRowCumulativeFirstNonzeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstNonZeroIndex(frame, names, output_names);
}

pub fn withRowPrefixFirstNonZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstNonZeroIndex(frame, names, output_names);
}

pub fn withRowPrefixFirstNonzeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowPrefixFirstNonZeroIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastNonZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_non_zero);
}

pub fn withRowCumulativeLastNonzeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastNonZeroIndex(frame, names, output_names);
}

pub fn withRowPrefixLastNonZeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastNonZeroIndex(frame, names, output_names);
}

pub fn withRowPrefixLastNonzeroIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowPrefixLastNonZeroIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstPositiveIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_positive);
}

pub fn withRowPrefixFirstPositiveIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstPositiveIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastPositiveIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_positive);
}

pub fn withRowPrefixLastPositiveIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastPositiveIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstSignBitIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_signbit);
}

pub fn withRowPrefixFirstSignBitIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstSignBitIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastSignBitIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_signbit);
}

pub fn withRowPrefixLastSignBitIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastSignBitIndex(frame, names, output_names);
}

pub fn withRowCumulativeFirstNegativeIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .first_negative);
}

pub fn withRowPrefixFirstNegativeIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFirstNegativeIndex(frame, names, output_names);
}

pub fn withRowCumulativeLastNegativeIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicateIndex(frame, names, output_names, .last_negative);
}

pub fn withRowPrefixLastNegativeIndex(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeLastNegativeIndex(frame, names, output_names);
}

pub fn withRowCumulativePositiveCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .positive, false);
}

pub fn withRowCumPositiveCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveCount(frame, names, output_names);
}

pub fn withRowPrefixPositiveCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveCount(frame, names, output_names);
}

pub fn withRowCumulativeNegativeCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .negative, false);
}

pub fn withRowCumNegativeCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeCount(frame, names, output_names);
}

pub fn withRowPrefixNegativeCount(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeCount(frame, names, output_names);
}

pub fn withRowCumulativePositiveZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .positive_zero, true);
}

pub fn withRowCumPositiveZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveZeroRatio(frame, names, output_names);
}

pub fn withRowPrefixPositiveZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveZeroRatio(frame, names, output_names);
}

pub fn withRowCumulativeNegativeZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .negative_zero, true);
}

pub fn withRowCumNegativeZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeZeroRatio(frame, names, output_names);
}

pub fn withRowPrefixNegativeZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeZeroRatio(frame, names, output_names);
}

pub fn withRowCumulativeSignBitRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .signbit, true);
}

pub fn withRowCumSignBitRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSignBitRatio(frame, names, output_names);
}

pub fn withRowPrefixSignBitRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSignBitRatio(frame, names, output_names);
}

pub fn withRowCumulativeNaNRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .nan, true);
}

pub fn withRowCumNaNRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNaNRatio(frame, names, output_names);
}

pub fn withRowPrefixNaNRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNaNRatio(frame, names, output_names);
}

pub fn withRowCumulativeInfRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .inf, true);
}

pub fn withRowCumInfRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeInfRatio(frame, names, output_names);
}

pub fn withRowPrefixInfRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeInfRatio(frame, names, output_names);
}

pub fn withRowCumulativePositiveInfRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .positive_inf, true);
}

pub fn withRowCumPositiveInfRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveInfRatio(frame, names, output_names);
}

pub fn withRowPrefixPositiveInfRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveInfRatio(frame, names, output_names);
}

pub fn withRowCumulativeNegativeInfRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .negative_inf, true);
}

pub fn withRowCumNegativeInfRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeInfRatio(frame, names, output_names);
}

pub fn withRowPrefixNegativeInfRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeInfRatio(frame, names, output_names);
}

pub fn withRowCumulativeFiniteRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .finite, true);
}

pub fn withRowCumFiniteRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFiniteRatio(frame, names, output_names);
}

pub fn withRowPrefixFiniteRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeFiniteRatio(frame, names, output_names);
}

pub fn withRowCumulativeNormalRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .normal, true);
}

pub fn withRowCumNormalRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNormalRatio(frame, names, output_names);
}

pub fn withRowPrefixNormalRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNormalRatio(frame, names, output_names);
}

pub fn withRowCumulativeSubnormalRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .subnormal, true);
}

pub fn withRowCumSubnormalRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSubnormalRatio(frame, names, output_names);
}

pub fn withRowPrefixSubnormalRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeSubnormalRatio(frame, names, output_names);
}

pub fn withRowCumulativeNonFiniteRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .non_finite, true);
}

pub fn withRowCumNonFiniteRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNonFiniteRatio(frame, names, output_names);
}

pub fn withRowPrefixNonFiniteRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNonFiniteRatio(frame, names, output_names);
}

pub fn withRowCumulativeZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .zero, true);
}

pub fn withRowCumZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeZeroRatio(frame, names, output_names);
}

pub fn withRowPrefixZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeZeroRatio(frame, names, output_names);
}

pub fn withRowCumulativeNonZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .non_zero, true);
}

pub fn withRowCumNonZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNonZeroRatio(frame, names, output_names);
}

pub fn withRowPrefixNonZeroRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNonZeroRatio(frame, names, output_names);
}

pub fn withRowCumulativePositiveRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .positive, true);
}

pub fn withRowCumPositiveRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveRatio(frame, names, output_names);
}

pub fn withRowPrefixPositiveRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativePositiveRatio(frame, names, output_names);
}

pub fn withRowCumulativeNegativeRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNumericPredicate(frame, names, output_names, .negative, true);
}

pub fn withRowCumNegativeRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeRatio(frame, names, output_names);
}

pub fn withRowPrefixNegativeRatio(frame: anytype, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
    return withRowCumulativeNegativeRatio(frame, names, output_names);
}

pub fn withRowFiniteCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .finite);
}

pub fn withRowFiniteRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .finite);
}

pub fn withRowNormalCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .normal);
}

pub fn withRowNormalRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .normal);
}

pub fn withRowSubnormalCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .subnormal);
}

pub fn withRowSubnormalRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .subnormal);
}

pub fn withRowNonFiniteCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .non_finite);
}

pub fn withRowNonFiniteRatio(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateRatio(frame, names, output_name, .non_finite);
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

pub fn filterColumnScalarWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_scalar = .{
        .name = owned_name,
        .op = op,
        .scalar = scalar,
        .keep_matches = true,
    } });
}

pub fn filterColumnScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
    return filterColumnScalarWithDeviceScalar(frame, name, DeviceScalar.init(T, scalar), op);
}

pub fn dropColumnScalarWithDeviceScalar(frame: anytype, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_scalar = .{
        .name = owned_name,
        .op = op,
        .scalar = scalar,
        .keep_matches = false,
    } });
}

pub fn dropColumnScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
    return dropColumnScalarWithDeviceScalar(frame, name, DeviceScalar.init(T, scalar), op);
}
