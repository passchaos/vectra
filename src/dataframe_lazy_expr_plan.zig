//! Lazy expression operation builders for DeviceLazyFrame.

const std = @import("std");
const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
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

pub fn withRowInfCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .inf);
}

pub fn withRowPositiveInfCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .positive_inf);
}

pub fn withRowNegativeInfCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .negative_inf);
}

pub fn withRowZeroCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .zero);
}

pub fn withRowPositiveZeroCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .positive_zero);
}

pub fn withRowNegativeZeroCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .negative_zero);
}

pub fn withRowNonZeroCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .non_zero);
}

pub fn withRowPositiveCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .positive);
}

pub fn withRowSignBitCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .signbit);
}

pub fn withRowNegativeCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .negative);
}

pub fn withRowFiniteCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .finite);
}

pub fn withRowNormalCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .normal);
}

pub fn withRowSubnormalCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .subnormal);
}

pub fn withRowNonFiniteCount(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericPredicateCount(frame, names, output_name, .non_finite);
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
