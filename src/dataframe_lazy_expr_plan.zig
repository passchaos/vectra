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

pub fn withRowWeightedMean(frame: anytype, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowPairedNumeric(frame, value_names, weight_names, output_name, .weighted_mean);
}

fn withRowWeightedDispersion(
    frame: anytype,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
    comptime reduction: enum { variance, stddev },
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

fn withRowWeightedPair(
    frame: anytype,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
    correction: f64,
    comptime reduction: enum { covariance, correlation, beta },
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

pub fn withRowWeightedCovariance(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, correction, .covariance);
}

pub fn withRowWeightedCorrelation(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, correction, .correlation);
}

pub fn withRowWeightedBeta(frame: anytype, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowWeightedPair(frame, lhs_names, rhs_names, weight_names, output_name, correction, .beta);
}

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

fn withRowQuantileAlias(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: enum { median, iqr, mad, mode, count_distinct, n_unique },
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
    }
}

pub fn withRowMedian(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .median);
}

pub fn withRowIqr(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .iqr);
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

pub fn withRowCountDistinct(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .count_distinct);
}

pub fn withRowNUnique(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowQuantileAlias(frame, names, output_name, .n_unique);
}

fn withRowNumericReduction(
    frame: anytype,
    names: []const []const u8,
    output_name: []const u8,
    comptime reduction: enum { sum, mean, geometric_mean, harmonic_mean, skewness, kurtosis, prod, min, max, ptp, mean_abs, rms, l1_norm, l2_norm },
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
        .geometric_mean => try frame.ops.append(frame.allocator, .{ .row_geometric_mean = .{
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
        .kurtosis => try frame.ops.append(frame.allocator, .{ .row_kurtosis = .{
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
        .mean_abs => try frame.ops.append(frame.allocator, .{ .row_mean_abs = .{
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

pub fn withRowGeometricMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .geometric_mean);
}

pub fn withRowGeoMean(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowGeometricMean(frame, names, output_name);
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

pub fn withRowKurtosis(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .kurtosis);
}

pub fn withRowKurt(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowKurtosis(frame, names, output_name);
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

pub fn withRowMeanAbs(frame: anytype, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
    return withRowNumericReduction(frame, names, output_name, .mean_abs);
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
    comptime reduction: enum { variance, stddev, sem, cv },
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
        .stddev => try frame.ops.append(frame.allocator, .{ .row_stddev = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .sem => try frame.ops.append(frame.allocator, .{ .row_sem = .{
            .names = owned_names,
            .output_name = owned_output,
            .correction = correction,
        } }),
        .cv => try frame.ops.append(frame.allocator, .{ .row_cv = .{
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

pub fn withRowStddev(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .stddev);
}

pub fn withRowStd(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowStddev(frame, names, output_name, correction);
}

pub fn withRowSem(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .sem);
}

pub fn withRowCv(frame: anytype, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
    return withRowNumericDispersion(frame, names, output_name, correction, .cv);
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

pub fn filterColumnScalar(frame: anytype, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    try frame.ops.append(frame.allocator, .{ .filter_scalar = .{
        .name = owned_name,
        .op = op,
        .scalar = DeviceScalar.init(T, scalar),
    } });
}
