//! Ownership helpers for DeviceLazyOp payloads.
//!
//! Kept out-of-line so the operation-tag union stays compact while the large
//! clone/deinit switch remains close to the lazy operation type contract.

const std = @import("std");
const clone_profile_mod = @import("dataframe_lazy_op_clone_profile.zig");
const deinit_mod = @import("dataframe_lazy_op_deinit.zig");
const array_mod = @import("array.zig");
const names_mod = @import("dataframe_names.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

pub const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;

pub const deinit = deinit_mod.deinit;

fn cloneGroupWeightedShift(
    comptime Self: type,
    allocator: std.mem.Allocator,
    shift: anytype,
    comptime tag_name: []const u8,
) DeviceDataError!Self {
    const names = try cloneNameList(allocator, shift.names);
    errdefer freeNameList(allocator, names);
    const value_name = try allocator.dupe(u8, shift.value_name);
    errdefer allocator.free(value_name);
    const weight_name = try allocator.dupe(u8, shift.weight_name);
    errdefer allocator.free(weight_name);
    const output_name = try allocator.dupe(u8, shift.output_name);
    errdefer allocator.free(output_name);
    return @unionInit(Self, tag_name, .{
        .names = names,
        .value_name = value_name,
        .weight_name = weight_name,
        .output_name = output_name,
    });
}

fn cloneGroupWeightedPairShift(
    comptime Self: type,
    allocator: std.mem.Allocator,
    shift: anytype,
    comptime tag_name: []const u8,
) DeviceDataError!Self {
    const names = try cloneNameList(allocator, shift.names);
    errdefer freeNameList(allocator, names);
    const lhs_name = try allocator.dupe(u8, shift.lhs_name);
    errdefer allocator.free(lhs_name);
    const rhs_name = try allocator.dupe(u8, shift.rhs_name);
    errdefer allocator.free(rhs_name);
    const weight_name = try allocator.dupe(u8, shift.weight_name);
    errdefer allocator.free(weight_name);
    const output_name = try allocator.dupe(u8, shift.output_name);
    errdefer allocator.free(output_name);
    return @unionInit(Self, tag_name, .{
        .names = names,
        .lhs_name = lhs_name,
        .rhs_name = rhs_name,
        .weight_name = weight_name,
        .output_name = output_name,
        .correction = shift.correction,
    });
}

fn cloneRowWeightedMean(
    comptime Self: type,
    allocator: std.mem.Allocator,
    row_weighted: anytype,
    comptime tag_name: []const u8,
) DeviceDataError!Self {
    const value_names = try cloneNameList(allocator, row_weighted.value_names);
    errdefer freeNameList(allocator, value_names);
    const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
    errdefer freeNameList(allocator, weight_names);
    const output_name = try allocator.dupe(u8, row_weighted.output_name);
    errdefer allocator.free(output_name);
    return @unionInit(Self, tag_name, .{
        .value_names = value_names,
        .weight_names = weight_names,
        .output_name = output_name,
    });
}

fn cloneRowWeightedDispersion(
    comptime Self: type,
    allocator: std.mem.Allocator,
    row_weighted: anytype,
    comptime tag_name: []const u8,
) DeviceDataError!Self {
    const value_names = try cloneNameList(allocator, row_weighted.value_names);
    errdefer freeNameList(allocator, value_names);
    const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
    errdefer freeNameList(allocator, weight_names);
    const output_name = try allocator.dupe(u8, row_weighted.output_name);
    errdefer allocator.free(output_name);
    return @unionInit(Self, tag_name, .{
        .value_names = value_names,
        .weight_names = weight_names,
        .output_name = output_name,
        .correction = row_weighted.correction,
    });
}

fn cloneRowWeightedPair(
    comptime Self: type,
    allocator: std.mem.Allocator,
    row_weighted: anytype,
    comptime tag_name: []const u8,
) DeviceDataError!Self {
    const lhs_names = try cloneNameList(allocator, row_weighted.lhs_names);
    errdefer freeNameList(allocator, lhs_names);
    const rhs_names = try cloneNameList(allocator, row_weighted.rhs_names);
    errdefer freeNameList(allocator, rhs_names);
    const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
    errdefer freeNameList(allocator, weight_names);
    const output_name = try allocator.dupe(u8, row_weighted.output_name);
    errdefer allocator.free(output_name);
    return @unionInit(Self, tag_name, .{
        .lhs_names = lhs_names,
        .rhs_names = rhs_names,
        .weight_names = weight_names,
        .output_name = output_name,
        .correction = row_weighted.correction,
    });
}

fn cloneRowWeightedColumnOutputs(
    comptime Self: type,
    allocator: std.mem.Allocator,
    row_weighted: anytype,
    comptime tag_name: []const u8,
) DeviceDataError!Self {
    const value_names = try cloneNameList(allocator, row_weighted.value_names);
    errdefer freeNameList(allocator, value_names);
    const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
    errdefer freeNameList(allocator, weight_names);
    const output_names = try cloneNameList(allocator, row_weighted.output_names);
    errdefer freeNameList(allocator, output_names);
    return @unionInit(Self, tag_name, .{
        .value_names = value_names,
        .weight_names = weight_names,
        .output_names = output_names,
    });
}

fn cloneRowWeightedColumnOutputsDispersion(
    comptime Self: type,
    allocator: std.mem.Allocator,
    row_weighted: anytype,
    comptime tag_name: []const u8,
) DeviceDataError!Self {
    const value_names = try cloneNameList(allocator, row_weighted.value_names);
    errdefer freeNameList(allocator, value_names);
    const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
    errdefer freeNameList(allocator, weight_names);
    const output_names = try cloneNameList(allocator, row_weighted.output_names);
    errdefer freeNameList(allocator, output_names);
    return @unionInit(Self, tag_name, .{
        .value_names = value_names,
        .weight_names = weight_names,
        .output_names = output_names,
        .correction = row_weighted.correction,
    });
}

fn cloneRowWeightedColumnOutputsQuantile(
    comptime Self: type,
    allocator: std.mem.Allocator,
    row_weighted: anytype,
    comptime tag_name: []const u8,
) DeviceDataError!Self {
    const value_names = try cloneNameList(allocator, row_weighted.value_names);
    errdefer freeNameList(allocator, value_names);
    const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
    errdefer freeNameList(allocator, weight_names);
    const output_names = try cloneNameList(allocator, row_weighted.output_names);
    errdefer freeNameList(allocator, output_names);
    return @unionInit(Self, tag_name, .{
        .value_names = value_names,
        .weight_names = weight_names,
        .output_names = output_names,
        .q = row_weighted.q,
    });
}

pub fn clone(comptime Self: type, self: Self, allocator: std.mem.Allocator) DeviceDataError!Self {
    return switch (self) {
        .select => |names| blk: {
            const owned = try cloneNameList(allocator, names);
            break :blk .{ .select = owned };
        },
        .select_column_indices => |indices| .{ .select_column_indices = try allocator.dupe(usize, indices) },
        .select_column_range => |range| .{ .select_column_range = range },
        .select_last_columns => |n| .{ .select_last_columns = n },
        .drop_column_indices => |indices| .{ .drop_column_indices = try allocator.dupe(usize, indices) },
        .drop_column_range => |range| .{ .drop_column_range = range },
        .drop_last_columns => |n| .{ .drop_last_columns = n },
        .reverse_columns => .{ .reverse_columns = {} },
        .sort_columns_by_name => |sort| .{ .sort_columns_by_name = sort },
        .select_name_prefix => |pattern| .{ .select_name_prefix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .select_name_suffix => |pattern| .{ .select_name_suffix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .select_name_contains => |pattern| .{ .select_name_contains = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .select_name_glob => |pattern| .{ .select_name_glob = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .drop_name_prefix => |pattern| .{ .drop_name_prefix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .drop_name_suffix => |pattern| .{ .drop_name_suffix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .drop_name_contains => |pattern| .{ .drop_name_contains = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .drop_name_glob => |pattern| .{ .drop_name_glob = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .select_dtypes => |dtypes| .{ .select_dtypes = try allocator.dupe(array_mod.DType, dtypes) },
        .select_dtype_class => |class| .{ .select_dtype_class = class },
        .drop_dtypes => |dtypes| .{ .drop_dtypes = try allocator.dupe(array_mod.DType, dtypes) },
        .drop_dtype_class => |class| .{ .drop_dtype_class = class },
        .select_nullable_columns => .{ .select_nullable_columns = {} },
        .select_non_nullable_columns => .{ .select_non_nullable_columns = {} },
        .select_columns_with_nulls => .{ .select_columns_with_nulls = {} },
        .select_columns_without_nulls => .{ .select_columns_without_nulls = {} },
        .drop_nullable_columns => .{ .drop_nullable_columns = {} },
        .drop_non_nullable_columns => .{ .drop_non_nullable_columns = {} },
        .drop_columns_with_nulls => .{ .drop_columns_with_nulls = {} },
        .drop_columns_without_nulls => .{ .drop_columns_without_nulls = {} },
        .select_columns_with_nans => .{ .select_columns_with_nans = {} },
        .select_columns_without_nans => .{ .select_columns_without_nans = {} },
        .drop_columns_with_nans => .{ .drop_columns_with_nans = {} },
        .drop_columns_without_nans => .{ .drop_columns_without_nans = {} },
        .select_columns_with_infs => .{ .select_columns_with_infs = {} },
        .select_columns_without_infs => .{ .select_columns_without_infs = {} },
        .drop_columns_with_infs => .{ .drop_columns_with_infs = {} },
        .drop_columns_without_infs => .{ .drop_columns_without_infs = {} },
        .select_columns_with_positive_infs => .{ .select_columns_with_positive_infs = {} },
        .select_columns_without_positive_infs => .{ .select_columns_without_positive_infs = {} },
        .drop_columns_with_positive_infs => .{ .drop_columns_with_positive_infs = {} },
        .drop_columns_without_positive_infs => .{ .drop_columns_without_positive_infs = {} },
        .select_columns_with_negative_infs => .{ .select_columns_with_negative_infs = {} },
        .select_columns_without_negative_infs => .{ .select_columns_without_negative_infs = {} },
        .drop_columns_with_negative_infs => .{ .drop_columns_with_negative_infs = {} },
        .drop_columns_without_negative_infs => .{ .drop_columns_without_negative_infs = {} },
        .select_columns_with_zeros => .{ .select_columns_with_zeros = {} },
        .select_columns_without_zeros => .{ .select_columns_without_zeros = {} },
        .drop_columns_with_zeros => .{ .drop_columns_with_zeros = {} },
        .drop_columns_without_zeros => .{ .drop_columns_without_zeros = {} },
        .select_columns_with_positive_zeros => .{ .select_columns_with_positive_zeros = {} },
        .select_columns_without_positive_zeros => .{ .select_columns_without_positive_zeros = {} },
        .drop_columns_with_positive_zeros => .{ .drop_columns_with_positive_zeros = {} },
        .drop_columns_without_positive_zeros => .{ .drop_columns_without_positive_zeros = {} },
        .select_columns_with_negative_zeros => .{ .select_columns_with_negative_zeros = {} },
        .select_columns_without_negative_zeros => .{ .select_columns_without_negative_zeros = {} },
        .drop_columns_with_negative_zeros => .{ .drop_columns_with_negative_zeros = {} },
        .drop_columns_without_negative_zeros => .{ .drop_columns_without_negative_zeros = {} },
        .select_columns_with_non_zeros => .{ .select_columns_with_non_zeros = {} },
        .select_columns_without_non_zeros => .{ .select_columns_without_non_zeros = {} },
        .drop_columns_with_non_zeros => .{ .drop_columns_with_non_zeros = {} },
        .drop_columns_without_non_zeros => .{ .drop_columns_without_non_zeros = {} },
        .select_columns_with_positives => .{ .select_columns_with_positives = {} },
        .select_columns_without_positives => .{ .select_columns_without_positives = {} },
        .drop_columns_with_positives => .{ .drop_columns_with_positives = {} },
        .drop_columns_without_positives => .{ .drop_columns_without_positives = {} },
        .select_columns_with_signbits => .{ .select_columns_with_signbits = {} },
        .select_columns_without_signbits => .{ .select_columns_without_signbits = {} },
        .drop_columns_with_signbits => .{ .drop_columns_with_signbits = {} },
        .drop_columns_without_signbits => .{ .drop_columns_without_signbits = {} },
        .select_columns_with_negatives => .{ .select_columns_with_negatives = {} },
        .select_columns_without_negatives => .{ .select_columns_without_negatives = {} },
        .drop_columns_with_negatives => .{ .drop_columns_with_negatives = {} },
        .drop_columns_without_negatives => .{ .drop_columns_without_negatives = {} },
        .select_columns_with_finites => .{ .select_columns_with_finites = {} },
        .select_columns_without_finites => .{ .select_columns_without_finites = {} },
        .drop_columns_with_finites => .{ .drop_columns_with_finites = {} },
        .drop_columns_without_finites => .{ .drop_columns_without_finites = {} },
        .select_columns_with_normals => .{ .select_columns_with_normals = {} },
        .select_columns_without_normals => .{ .select_columns_without_normals = {} },
        .drop_columns_with_normals => .{ .drop_columns_with_normals = {} },
        .drop_columns_without_normals => .{ .drop_columns_without_normals = {} },
        .select_columns_with_subnormals => .{ .select_columns_with_subnormals = {} },
        .select_columns_without_subnormals => .{ .select_columns_without_subnormals = {} },
        .drop_columns_with_subnormals => .{ .drop_columns_with_subnormals = {} },
        .drop_columns_without_subnormals => .{ .drop_columns_without_subnormals = {} },
        .select_columns_with_non_finites => .{ .select_columns_with_non_finites = {} },
        .select_columns_without_non_finites => .{ .select_columns_without_non_finites = {} },
        .drop_columns_with_non_finites => .{ .drop_columns_with_non_finites = {} },
        .drop_columns_without_non_finites => .{ .drop_columns_without_non_finites = {} },
        .with_row_index => |row_index| blk: {
            const name = try allocator.dupe(u8, row_index.name);
            break :blk .{ .with_row_index = .{
                .name = name,
                .offset = row_index.offset,
            } };
        },
        .rename_column => |rename| blk: {
            const old_name = try allocator.dupe(u8, rename.old_name);
            errdefer allocator.free(old_name);
            const new_name = try allocator.dupe(u8, rename.new_name);
            errdefer allocator.free(new_name);
            break :blk .{ .rename_column = .{
                .old_name = old_name,
                .new_name = new_name,
            } };
        },
        .rename_columns => |rename| blk: {
            const old_names = try cloneNameList(allocator, rename.old_names);
            errdefer freeNameList(allocator, old_names);
            const new_names = try cloneNameList(allocator, rename.new_names);
            errdefer freeNameList(allocator, new_names);
            break :blk .{ .rename_columns = .{
                .old_names = old_names,
                .new_names = new_names,
            } };
        },
        .add_column_name_prefix => |pattern| .{ .add_column_name_prefix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .add_column_name_suffix => |pattern| .{ .add_column_name_suffix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .strip_column_name_prefix => |pattern| .{ .strip_column_name_prefix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .strip_column_name_suffix => |pattern| .{ .strip_column_name_suffix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .replace_column_name_prefix => |replace| blk: {
            const old_pattern = try allocator.dupe(u8, replace.old_pattern);
            errdefer allocator.free(old_pattern);
            const new_pattern = try allocator.dupe(u8, replace.new_pattern);
            errdefer allocator.free(new_pattern);
            break :blk .{ .replace_column_name_prefix = .{
                .old_pattern = old_pattern,
                .new_pattern = new_pattern,
            } };
        },
        .replace_column_name_suffix => |replace| blk: {
            const old_pattern = try allocator.dupe(u8, replace.old_pattern);
            errdefer allocator.free(old_pattern);
            const new_pattern = try allocator.dupe(u8, replace.new_pattern);
            errdefer allocator.free(new_pattern);
            break :blk .{ .replace_column_name_suffix = .{
                .old_pattern = old_pattern,
                .new_pattern = new_pattern,
            } };
        },
        .move_column => |move| blk: {
            const name = try allocator.dupe(u8, move.name);
            break :blk .{ .move_column = .{
                .name = name,
                .target_index = move.target_index,
            } };
        },
        .move_column_before => |move| blk: {
            const name = try allocator.dupe(u8, move.name);
            errdefer allocator.free(name);
            const anchor_name = try allocator.dupe(u8, move.anchor_name);
            errdefer allocator.free(anchor_name);
            break :blk .{ .move_column_before = .{
                .name = name,
                .anchor_name = anchor_name,
            } };
        },
        .move_column_after => |move| blk: {
            const name = try allocator.dupe(u8, move.name);
            errdefer allocator.free(name);
            const anchor_name = try allocator.dupe(u8, move.anchor_name);
            errdefer allocator.free(anchor_name);
            break :blk .{ .move_column_after = .{
                .name = name,
                .anchor_name = anchor_name,
            } };
        },
        .copy_column => |copy| blk: {
            const source_name = try allocator.dupe(u8, copy.source_name);
            errdefer allocator.free(source_name);
            const new_name = try allocator.dupe(u8, copy.new_name);
            errdefer allocator.free(new_name);
            break :blk .{ .copy_column = .{
                .source_name = source_name,
                .new_name = new_name,
            } };
        },
        .copy_column_at => |copy| blk: {
            const source_name = try allocator.dupe(u8, copy.source_name);
            errdefer allocator.free(source_name);
            const new_name = try allocator.dupe(u8, copy.new_name);
            errdefer allocator.free(new_name);
            break :blk .{ .copy_column_at = .{
                .source_name = source_name,
                .new_name = new_name,
                .target_index = copy.target_index,
            } };
        },
        .copy_column_before => |copy| blk: {
            const source_name = try allocator.dupe(u8, copy.source_name);
            errdefer allocator.free(source_name);
            const new_name = try allocator.dupe(u8, copy.new_name);
            errdefer allocator.free(new_name);
            const anchor_name = try allocator.dupe(u8, copy.anchor_name);
            errdefer allocator.free(anchor_name);
            break :blk .{ .copy_column_before = .{
                .source_name = source_name,
                .new_name = new_name,
                .anchor_name = anchor_name,
            } };
        },
        .copy_column_after => |copy| blk: {
            const source_name = try allocator.dupe(u8, copy.source_name);
            errdefer allocator.free(source_name);
            const new_name = try allocator.dupe(u8, copy.new_name);
            errdefer allocator.free(new_name);
            const anchor_name = try allocator.dupe(u8, copy.anchor_name);
            errdefer allocator.free(anchor_name);
            break :blk .{ .copy_column_after = .{
                .source_name = source_name,
                .new_name = new_name,
                .anchor_name = anchor_name,
            } };
        },
        .drop_columns => |names| .{ .drop_columns = try cloneNameList(allocator, names) },
        .drop_nulls => |names| .{ .drop_nulls = try cloneNameList(allocator, names) },
        .drop_all_nulls => |names| .{ .drop_all_nulls = try cloneNameList(allocator, names) },
        .filter_all_nulls => |names| .{ .filter_all_nulls = try cloneNameList(allocator, names) },
        .filter_nulls_column => |name| .{ .filter_nulls_column = try allocator.dupe(u8, name) },
        .drop_nans => |names| .{ .drop_nans = try cloneNameList(allocator, names) },
        .filter_nans_column => |name| .{ .filter_nans_column = try allocator.dupe(u8, name) },
        .drop_infs => |names| .{ .drop_infs = try cloneNameList(allocator, names) },
        .filter_infs_column => |name| .{ .filter_infs_column = try allocator.dupe(u8, name) },
        .drop_positive_infs => |names| .{ .drop_positive_infs = try cloneNameList(allocator, names) },
        .filter_positive_infs_column => |name| .{ .filter_positive_infs_column = try allocator.dupe(u8, name) },
        .drop_negative_infs => |names| .{ .drop_negative_infs = try cloneNameList(allocator, names) },
        .filter_negative_infs_column => |name| .{ .filter_negative_infs_column = try allocator.dupe(u8, name) },
        .drop_zeros => |names| .{ .drop_zeros = try cloneNameList(allocator, names) },
        .filter_zeros_column => |name| .{ .filter_zeros_column = try allocator.dupe(u8, name) },
        .drop_positive_zeros => |names| .{ .drop_positive_zeros = try cloneNameList(allocator, names) },
        .filter_positive_zeros_column => |name| .{ .filter_positive_zeros_column = try allocator.dupe(u8, name) },
        .drop_negative_zeros => |names| .{ .drop_negative_zeros = try cloneNameList(allocator, names) },
        .filter_negative_zeros_column => |name| .{ .filter_negative_zeros_column = try allocator.dupe(u8, name) },
        .drop_non_zeros => |names| .{ .drop_non_zeros = try cloneNameList(allocator, names) },
        .filter_non_zeros_column => |name| .{ .filter_non_zeros_column = try allocator.dupe(u8, name) },
        .drop_positives => |names| .{ .drop_positives = try cloneNameList(allocator, names) },
        .filter_positives_column => |name| .{ .filter_positives_column = try allocator.dupe(u8, name) },
        .drop_signbits => |names| .{ .drop_signbits = try cloneNameList(allocator, names) },
        .filter_signbits_column => |name| .{ .filter_signbits_column = try allocator.dupe(u8, name) },
        .drop_negatives => |names| .{ .drop_negatives = try cloneNameList(allocator, names) },
        .filter_negatives_column => |name| .{ .filter_negatives_column = try allocator.dupe(u8, name) },
        .drop_finites => |names| .{ .drop_finites = try cloneNameList(allocator, names) },
        .filter_finites_column => |name| .{ .filter_finites_column = try allocator.dupe(u8, name) },
        .drop_normals => |names| .{ .drop_normals = try cloneNameList(allocator, names) },
        .filter_normals_column => |name| .{ .filter_normals_column = try allocator.dupe(u8, name) },
        .drop_subnormals => |names| .{ .drop_subnormals = try cloneNameList(allocator, names) },
        .filter_subnormals_column => |name| .{ .filter_subnormals_column = try allocator.dupe(u8, name) },
        .drop_non_finites => |names| .{ .drop_non_finites = try cloneNameList(allocator, names) },
        .filter_non_finites_column => |name| .{ .filter_non_finites_column = try allocator.dupe(u8, name) },
        .with_column_abs => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_abs = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_neg => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_neg = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_square => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_square = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_reciprocal => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_reciprocal = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_sign => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_sign = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_sqrt => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_sqrt = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_rsqrt => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_rsqrt = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_cbrt => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_cbrt = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_floor => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_floor = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_ceil => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_ceil = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_round => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_round = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_trunc => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_trunc = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_deg2rad => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_deg2rad = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_rad2deg => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_rad2deg = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_expit => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_expit = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_logit => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_logit = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_softplus => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_softplus = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_logsigmoid => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_logsigmoid = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_relu => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_relu = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_leaky_relu => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_leaky_relu = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_relu6 => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_relu6 = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_pow_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_pow_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_floor_div_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_floor_div_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_mod_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_mod_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_remainder_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_remainder_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_log_add_exp_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_log_add_exp_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_log_add_exp2_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_log_add_exp2_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_xlogy_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_xlogy_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_fmax_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_fmax_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_fmin_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_fmin_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_hypot_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_hypot_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_atan2_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_atan2_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_next_after_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_next_after_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_copysign_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_copysign_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_heaviside_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_heaviside_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_ldexp_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_ldexp_scalar = .{
                .name = name,
                .input_name = input_name,
                .exponent = expr.exponent,
            } };
        },
        .with_column_threshold => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_threshold = .{
                .name = name,
                .input_name = input_name,
                .lhs_scalar = expr.lhs_scalar,
                .rhs_scalar = expr.rhs_scalar,
            } };
        },
        .with_column_hardtanh => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_hardtanh = .{
                .name = name,
                .input_name = input_name,
                .lhs_scalar = expr.lhs_scalar,
                .rhs_scalar = expr.rhs_scalar,
            } };
        },
        .with_column_between => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_between = .{
                .name = name,
                .input_name = input_name,
                .lower = expr.lower,
                .upper = expr.upper,
                .lower_inclusive = expr.lower_inclusive,
                .upper_inclusive = expr.upper_inclusive,
            } };
        },
        .with_column_maximum_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_maximum_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_minimum_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_minimum_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_clip_min => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_clip_min = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_clip_max => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_clip_max = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_hardshrink => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_hardshrink = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_softshrink => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_softshrink = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_tanhshrink => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_tanhshrink = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_elu => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_elu = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_celu => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_celu = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_softsign => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_softsign = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_hardsigmoid => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_hardsigmoid = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_hardswish => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_hardswish = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_silu => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_silu = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_swish => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_swish = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_mish => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_mish = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_gelu => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_gelu = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_selu => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_selu = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_exp => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_exp = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_exp2 => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_exp2 = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_expm1 => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_expm1 = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_sin => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_sin = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_cos => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_cos = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_tan => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_tan = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_asin => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_asin = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_acos => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_acos = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_atan => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_atan = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_sinh => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_sinh = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_cosh => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_cosh = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_tanh => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_tanh = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_asinh => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_asinh = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_acosh => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_acosh = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_atanh => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_atanh = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_log => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_log = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_log1p => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_log1p = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_lgamma => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_lgamma = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_sinc => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_sinc = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_log2 => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_log2 = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_log10 => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            break :blk .{ .with_column_log10 = .{
                .name = name,
                .input_name = input_name,
            } };
        },
        .with_column_binary => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const lhs_name = try allocator.dupe(u8, expr.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, expr.rhs_name);
            errdefer allocator.free(rhs_name);
            break :blk .{ .with_column_binary = .{
                .name = name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .op = expr.op,
            } };
        },
        .with_column_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            break :blk .{ .with_column_scalar = .{
                .name = name,
                .input_name = input_name,
                .op = expr.op,
                .scalar = expr.scalar,
            } };
        },
        .with_column_lerp_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const lhs_name = try allocator.dupe(u8, expr.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, expr.rhs_name);
            errdefer allocator.free(rhs_name);
            break :blk .{ .with_column_lerp_scalar = .{
                .name = name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_addcmul_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const base_name = try allocator.dupe(u8, expr.base_name);
            errdefer allocator.free(base_name);
            const lhs_name = try allocator.dupe(u8, expr.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, expr.rhs_name);
            errdefer allocator.free(rhs_name);
            break :blk .{ .with_column_addcmul_scalar = .{
                .name = name,
                .base_name = base_name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_addcdiv_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const base_name = try allocator.dupe(u8, expr.base_name);
            errdefer allocator.free(base_name);
            const lhs_name = try allocator.dupe(u8, expr.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, expr.rhs_name);
            errdefer allocator.free(rhs_name);
            break :blk .{ .with_column_addcdiv_scalar = .{
                .name = name,
                .base_name = base_name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_clip_array => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            const lhs_name = try allocator.dupe(u8, expr.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, expr.rhs_name);
            errdefer allocator.free(rhs_name);
            break :blk .{ .with_column_clip_array = .{
                .name = name,
                .input_name = input_name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
            } };
        },
        .with_column_where => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            const lhs_name = try allocator.dupe(u8, expr.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, expr.rhs_name);
            errdefer allocator.free(rhs_name);
            break :blk .{ .with_column_where = .{
                .name = name,
                .input_name = input_name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
            } };
        },
        .with_column_where_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            const mask_name = try allocator.dupe(u8, expr.mask_name);
            errdefer allocator.free(mask_name);
            break :blk .{ .with_column_where_scalar = .{
                .name = name,
                .input_name = input_name,
                .mask_name = mask_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_isin => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            const test_name = try allocator.dupe(u8, expr.test_name);
            errdefer allocator.free(test_name);
            break :blk .{ .with_column_isin = .{
                .name = name,
                .input_name = input_name,
                .test_name = test_name,
                .invert = expr.invert,
            } };
        },
        .with_column_isin_values => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            var values = try expr.values.clone();
            errdefer values.deinit();
            break :blk .{ .with_column_isin_values = .{
                .name = name,
                .input_name = input_name,
                .values = values,
                .invert = expr.invert,
            } };
        },
        .with_column_masked_put_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            const mask_name = try allocator.dupe(u8, expr.mask_name);
            errdefer allocator.free(mask_name);
            break :blk .{ .with_column_masked_put_scalar = .{
                .name = name,
                .input_name = input_name,
                .mask_name = mask_name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_put_flat_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            const row_indices = try allocator.dupe(usize, expr.row_indices);
            errdefer allocator.free(row_indices);
            break :blk .{ .with_column_put_flat_scalar = .{
                .name = name,
                .input_name = input_name,
                .row_indices = row_indices,
                .scalar = expr.scalar,
            } };
        },
        .with_column_put_flat => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            const row_indices = try allocator.dupe(usize, expr.row_indices);
            errdefer allocator.free(row_indices);
            const value_name = try allocator.dupe(u8, expr.value_name);
            errdefer allocator.free(value_name);
            break :blk .{ .with_column_put_flat = .{
                .name = name,
                .input_name = input_name,
                .row_indices = row_indices,
                .value_name = value_name,
            } };
        },
        .with_column_put_flat_scalar_mode => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            const row_indices = try allocator.dupe(usize, expr.row_indices);
            errdefer allocator.free(row_indices);
            break :blk .{ .with_column_put_flat_scalar_mode = .{
                .name = name,
                .input_name = input_name,
                .row_indices = row_indices,
                .scalar = expr.scalar,
                .mode = expr.mode,
            } };
        },
        .with_column_put_flat_scalar_signed => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            const row_indices = try allocator.dupe(isize, expr.row_indices);
            errdefer allocator.free(row_indices);
            break :blk .{ .with_column_put_flat_scalar_signed = .{
                .name = name,
                .input_name = input_name,
                .row_indices = row_indices,
                .scalar = expr.scalar,
            } };
        },
        .with_column_isclose_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            break :blk .{ .with_column_isclose_scalar = .{
                .name = name,
                .input_name = input_name,
                .scalar = expr.scalar,
                .rtol = expr.rtol,
                .atol = expr.atol,
                .equal_nan = expr.equal_nan,
            } };
        },
        .with_column_logical => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const lhs_name = try allocator.dupe(u8, expr.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, expr.rhs_name);
            errdefer allocator.free(rhs_name);
            break :blk .{ .with_column_logical = .{
                .name = name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .op = expr.op,
            } };
        },
        .with_column_logical_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            break :blk .{ .with_column_logical_scalar = .{
                .name = name,
                .input_name = input_name,
                .op = expr.op,
                .scalar = expr.scalar,
            } };
        },
        .with_column_literal => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            break :blk .{ .with_column_literal = .{
                .name = name,
                .scalar = expr.scalar,
            } };
        },
        .with_column_literal_at => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            break :blk .{ .with_column_literal_at = .{
                .name = name,
                .scalar = expr.scalar,
                .target_index = expr.target_index,
            } };
        },
        .with_column_literal_before => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const anchor_name = try allocator.dupe(u8, expr.anchor_name);
            errdefer allocator.free(anchor_name);
            break :blk .{ .with_column_literal_before = .{
                .name = name,
                .scalar = expr.scalar,
                .anchor_name = anchor_name,
            } };
        },
        .with_column_literal_after => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const anchor_name = try allocator.dupe(u8, expr.anchor_name);
            errdefer allocator.free(anchor_name);
            break :blk .{ .with_column_literal_after = .{
                .name = name,
                .scalar = expr.scalar,
                .anchor_name = anchor_name,
            } };
        },
        .cast_column => |cast| blk: {
            const name = try allocator.dupe(u8, cast.name);
            break :blk .{ .cast_column = .{
                .name = name,
                .dtype = cast.dtype,
            } };
        },
        .fill_null_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_null_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_nan_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_nan_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_inf_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_inf_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_positive_inf_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_positive_inf_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_negative_inf_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_negative_inf_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_zero_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_zero_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_positive_zero_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_positive_zero_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_negative_zero_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_negative_zero_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_non_zero_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_non_zero_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_positive_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_positive_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_signbit_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_signbit_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_negative_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_negative_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_finite_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_finite_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_normal_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_normal_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_subnormal_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_subnormal_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_non_finite_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .fill_non_finite_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .fill_null_forward_column => |name| .{ .fill_null_forward_column = try allocator.dupe(u8, name) },
        .fill_null_backward_column => |name| .{ .fill_null_backward_column = try allocator.dupe(u8, name) },
        .null_if_column => |fill| blk: {
            const name = try allocator.dupe(u8, fill.name);
            break :blk .{ .null_if_column = .{
                .name = name,
                .scalar = fill.scalar,
            } };
        },
        .null_if_values_column => |null_if| blk: {
            const name = try allocator.dupe(u8, null_if.name);
            errdefer allocator.free(name);
            var values = try null_if.values.clone();
            errdefer values.deinit();
            break :blk .{ .null_if_values_column = .{
                .name = name,
                .values = values,
            } };
        },
        .null_if_nan_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_nan_column = name };
        },
        .null_if_inf_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_inf_column = name };
        },
        .null_if_positive_inf_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_positive_inf_column = name };
        },
        .null_if_negative_inf_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_negative_inf_column = name };
        },
        .null_if_zero_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_zero_column = name };
        },
        .null_if_positive_zero_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_positive_zero_column = name };
        },
        .null_if_negative_zero_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_negative_zero_column = name };
        },
        .null_if_non_zero_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_non_zero_column = name };
        },
        .null_if_positive_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_positive_column = name };
        },
        .null_if_signbit_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_signbit_column = name };
        },
        .null_if_negative_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_negative_column = name };
        },
        .null_if_finite_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_finite_column = name };
        },
        .null_if_normal_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_normal_column = name };
        },
        .null_if_subnormal_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_subnormal_column = name };
        },
        .null_if_non_finite_column => |source_name| blk: {
            const name = try allocator.dupe(u8, source_name);
            break :blk .{ .null_if_non_finite_column = name };
        },
        .coalesce_columns => |coalesce| blk: {
            const primary_name = try allocator.dupe(u8, coalesce.primary_name);
            errdefer allocator.free(primary_name);
            const fallback_name = try allocator.dupe(u8, coalesce.fallback_name);
            errdefer allocator.free(fallback_name);
            const output_name = try allocator.dupe(u8, coalesce.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .coalesce_columns = .{
                .primary_name = primary_name,
                .fallback_name = fallback_name,
                .output_name = output_name,
            } };
        },
        .coalesce_columns_many => |coalesce| blk: {
            const names = try cloneNameList(allocator, coalesce.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, coalesce.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .coalesce_columns_many = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .is_null_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_null_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_valid_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_valid_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_nan_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_nan_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_zero_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_zero_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_positive_zero_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_positive_zero_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_negative_zero_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_negative_zero_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_non_zero_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_non_zero_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_positive_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_positive_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_signbit_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_signbit_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_negative_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_negative_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_finite_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_finite_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_normal_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_normal_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_subnormal_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_subnormal_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_non_finite_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_non_finite_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_inf_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_inf_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_positive_inf_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_positive_inf_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .is_negative_inf_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .is_negative_inf_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .row_cumulative_argmin, .row_cumulative_argmax, .row_cumulative_mode, .row_cumulative_mode_count, .row_cumulative_mode_ratio, .row_cumulative_mode_margin, .row_cumulative_mode_margin_ratio, .row_cumulative_distinct_count, .row_cumulative_n_unique, .row_cumulative_first_true_index, .row_cumulative_last_true_index, .row_cumulative_first_false_index, .row_cumulative_last_false_index, .row_cumulative_first_valid_index, .row_cumulative_last_valid_index, .row_cumulative_first_null_index, .row_cumulative_last_null_index, .row_cumulative_null_count, .row_cumulative_valid_count, .row_cumulative_any_null, .row_cumulative_all_null, .row_cumulative_any_valid, .row_cumulative_all_valid, .row_cumulative_null_ratio, .row_cumulative_valid_ratio, .row_cumulative_true_count, .row_cumulative_false_count, .row_cumulative_true_ratio, .row_cumulative_false_ratio, .row_cumulative_positive_zero_count, .row_cumulative_negative_zero_count, .row_cumulative_signbit_count, .row_cumulative_positive_zero_ratio, .row_cumulative_negative_zero_ratio, .row_cumulative_signbit_ratio, .row_cumulative_nan_count, .row_cumulative_inf_count, .row_cumulative_positive_inf_count, .row_cumulative_negative_inf_count, .row_cumulative_finite_count, .row_cumulative_normal_count, .row_cumulative_subnormal_count, .row_cumulative_non_finite_count, .row_cumulative_nan_ratio, .row_cumulative_inf_ratio, .row_cumulative_positive_inf_ratio, .row_cumulative_negative_inf_ratio, .row_cumulative_finite_ratio, .row_cumulative_normal_ratio, .row_cumulative_subnormal_ratio, .row_cumulative_non_finite_ratio, .row_cumulative_any_zero, .row_cumulative_all_zero, .row_cumulative_any_non_zero, .row_cumulative_all_non_zero, .row_cumulative_any_positive_zero, .row_cumulative_all_positive_zero, .row_cumulative_any_negative_zero, .row_cumulative_all_negative_zero, .row_cumulative_any_positive, .row_cumulative_all_positive, .row_cumulative_any_signbit, .row_cumulative_all_signbit, .row_cumulative_any_negative, .row_cumulative_all_negative, .row_cumulative_any_nan, .row_cumulative_all_nan, .row_cumulative_any_inf, .row_cumulative_all_inf, .row_cumulative_any_positive_inf, .row_cumulative_all_positive_inf, .row_cumulative_any_negative_inf, .row_cumulative_all_negative_inf, .row_cumulative_any_finite, .row_cumulative_all_finite, .row_cumulative_any_normal, .row_cumulative_all_normal, .row_cumulative_any_subnormal, .row_cumulative_all_subnormal, .row_cumulative_any_non_finite, .row_cumulative_all_non_finite, .row_cumulative_first_nan_index, .row_cumulative_last_nan_index, .row_cumulative_first_inf_index, .row_cumulative_last_inf_index, .row_cumulative_first_positive_inf_index, .row_cumulative_last_positive_inf_index, .row_cumulative_first_negative_inf_index, .row_cumulative_last_negative_inf_index, .row_cumulative_first_finite_index, .row_cumulative_last_finite_index, .row_cumulative_first_normal_index, .row_cumulative_last_normal_index, .row_cumulative_first_subnormal_index, .row_cumulative_last_subnormal_index, .row_cumulative_first_non_finite_index, .row_cumulative_last_non_finite_index, .row_cumulative_zero_count, .row_cumulative_first_zero_index, .row_cumulative_last_zero_index, .row_cumulative_first_positive_zero_index, .row_cumulative_last_positive_zero_index, .row_cumulative_first_negative_zero_index, .row_cumulative_last_negative_zero_index, .row_cumulative_non_zero_count, .row_cumulative_first_non_zero_index, .row_cumulative_last_non_zero_index, .row_cumulative_first_positive_index, .row_cumulative_last_positive_index, .row_cumulative_first_signbit_index, .row_cumulative_last_signbit_index, .row_cumulative_first_negative_index, .row_cumulative_last_negative_index, .row_cumulative_positive_count, .row_cumulative_negative_count, .row_cumulative_zero_ratio, .row_cumulative_non_zero_ratio, .row_cumulative_positive_ratio, .row_cumulative_negative_ratio, .row_cumulative_any_true, .row_cumulative_all_true, .row_cumulative_any_false, .row_cumulative_all_false, .row_centered, .row_zscore, .row_robust_zscore, .row_average_rank, .row_ordinal_rank, .row_dense_rank, .row_competition_rank, .row_percent_rank, .row_cume_dist, .row_cumulative_sum, .row_cumulative_mean, .row_cumulative_logsumexp, .row_cumulative_logmeanexp, .row_cumulative_geometric_mean, .row_cumulative_harmonic_mean, .row_cumulative_skewness, .row_cumulative_kurtosis, .row_cumulative_rms, .row_cumulative_mean_abs, .row_cumulative_mean_square, .row_cumulative_max_abs, .row_cumulative_min_abs, .row_cumulative_l1_norm, .row_cumulative_l2_norm, .row_cumulative_product, .row_cumulative_max, .row_cumulative_min, .row_cumulative_range, .row_iqr_outlier, .row_tukey_winsorize, .row_max_indicator, .row_min_indicator, .row_minmax_scale, .row_l2_normalize, .row_l1_normalize, .row_sum_normalize, .row_mean_normalize, .row_max_abs_normalize, .row_softmax, .row_log_softmax, .row_softmin, .row_log_softmin => |row_outputs, tag| blk: {
            const names = try cloneNameList(allocator, row_outputs.names);
            errdefer freeNameList(allocator, names);
            const output_names = try cloneNameList(allocator, row_outputs.output_names);
            errdefer freeNameList(allocator, output_names);
            break :blk switch (tag) {
                .row_cumulative_argmin => .{ .row_cumulative_argmin = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_argmax => .{ .row_cumulative_argmax = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_mode => .{ .row_cumulative_mode = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_mode_count => .{ .row_cumulative_mode_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_mode_ratio => .{ .row_cumulative_mode_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_mode_margin => .{ .row_cumulative_mode_margin = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_mode_margin_ratio => .{ .row_cumulative_mode_margin_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_distinct_count => .{ .row_cumulative_distinct_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_n_unique => .{ .row_cumulative_n_unique = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_true_index => .{ .row_cumulative_first_true_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_true_index => .{ .row_cumulative_last_true_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_false_index => .{ .row_cumulative_first_false_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_false_index => .{ .row_cumulative_last_false_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_valid_index => .{ .row_cumulative_first_valid_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_valid_index => .{ .row_cumulative_last_valid_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_null_index => .{ .row_cumulative_first_null_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_null_index => .{ .row_cumulative_last_null_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_null_count => .{ .row_cumulative_null_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_valid_count => .{ .row_cumulative_valid_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_null => .{ .row_cumulative_any_null = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_null => .{ .row_cumulative_all_null = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_valid => .{ .row_cumulative_any_valid = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_valid => .{ .row_cumulative_all_valid = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_null_ratio => .{ .row_cumulative_null_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_valid_ratio => .{ .row_cumulative_valid_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_true_count => .{ .row_cumulative_true_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_false_count => .{ .row_cumulative_false_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_true_ratio => .{ .row_cumulative_true_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_false_ratio => .{ .row_cumulative_false_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_positive_zero_count => .{ .row_cumulative_positive_zero_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_negative_zero_count => .{ .row_cumulative_negative_zero_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_signbit_count => .{ .row_cumulative_signbit_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_positive_zero_ratio => .{ .row_cumulative_positive_zero_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_negative_zero_ratio => .{ .row_cumulative_negative_zero_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_signbit_ratio => .{ .row_cumulative_signbit_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_nan_count => .{ .row_cumulative_nan_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_inf_count => .{ .row_cumulative_inf_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_positive_inf_count => .{ .row_cumulative_positive_inf_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_negative_inf_count => .{ .row_cumulative_negative_inf_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_finite_count => .{ .row_cumulative_finite_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_normal_count => .{ .row_cumulative_normal_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_subnormal_count => .{ .row_cumulative_subnormal_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_non_finite_count => .{ .row_cumulative_non_finite_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_nan_ratio => .{ .row_cumulative_nan_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_inf_ratio => .{ .row_cumulative_inf_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_positive_inf_ratio => .{ .row_cumulative_positive_inf_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_negative_inf_ratio => .{ .row_cumulative_negative_inf_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_finite_ratio => .{ .row_cumulative_finite_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_normal_ratio => .{ .row_cumulative_normal_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_subnormal_ratio => .{ .row_cumulative_subnormal_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_non_finite_ratio => .{ .row_cumulative_non_finite_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_zero => .{ .row_cumulative_any_zero = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_zero => .{ .row_cumulative_all_zero = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_non_zero => .{ .row_cumulative_any_non_zero = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_non_zero => .{ .row_cumulative_all_non_zero = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_positive_zero => .{ .row_cumulative_any_positive_zero = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_positive_zero => .{ .row_cumulative_all_positive_zero = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_negative_zero => .{ .row_cumulative_any_negative_zero = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_negative_zero => .{ .row_cumulative_all_negative_zero = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_positive => .{ .row_cumulative_any_positive = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_positive => .{ .row_cumulative_all_positive = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_signbit => .{ .row_cumulative_any_signbit = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_signbit => .{ .row_cumulative_all_signbit = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_negative => .{ .row_cumulative_any_negative = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_negative => .{ .row_cumulative_all_negative = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_nan => .{ .row_cumulative_any_nan = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_nan => .{ .row_cumulative_all_nan = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_inf => .{ .row_cumulative_any_inf = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_inf => .{ .row_cumulative_all_inf = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_positive_inf => .{ .row_cumulative_any_positive_inf = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_positive_inf => .{ .row_cumulative_all_positive_inf = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_negative_inf => .{ .row_cumulative_any_negative_inf = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_negative_inf => .{ .row_cumulative_all_negative_inf = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_finite => .{ .row_cumulative_any_finite = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_finite => .{ .row_cumulative_all_finite = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_normal => .{ .row_cumulative_any_normal = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_normal => .{ .row_cumulative_all_normal = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_subnormal => .{ .row_cumulative_any_subnormal = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_subnormal => .{ .row_cumulative_all_subnormal = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_non_finite => .{ .row_cumulative_any_non_finite = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_non_finite => .{ .row_cumulative_all_non_finite = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_nan_index => .{ .row_cumulative_first_nan_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_nan_index => .{ .row_cumulative_last_nan_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_inf_index => .{ .row_cumulative_first_inf_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_inf_index => .{ .row_cumulative_last_inf_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_positive_inf_index => .{ .row_cumulative_first_positive_inf_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_positive_inf_index => .{ .row_cumulative_last_positive_inf_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_negative_inf_index => .{ .row_cumulative_first_negative_inf_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_negative_inf_index => .{ .row_cumulative_last_negative_inf_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_finite_index => .{ .row_cumulative_first_finite_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_finite_index => .{ .row_cumulative_last_finite_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_normal_index => .{ .row_cumulative_first_normal_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_normal_index => .{ .row_cumulative_last_normal_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_subnormal_index => .{ .row_cumulative_first_subnormal_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_subnormal_index => .{ .row_cumulative_last_subnormal_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_non_finite_index => .{ .row_cumulative_first_non_finite_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_non_finite_index => .{ .row_cumulative_last_non_finite_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_zero_count => .{ .row_cumulative_zero_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_zero_index => .{ .row_cumulative_first_zero_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_zero_index => .{ .row_cumulative_last_zero_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_positive_zero_index => .{ .row_cumulative_first_positive_zero_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_positive_zero_index => .{ .row_cumulative_last_positive_zero_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_negative_zero_index => .{ .row_cumulative_first_negative_zero_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_negative_zero_index => .{ .row_cumulative_last_negative_zero_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_non_zero_count => .{ .row_cumulative_non_zero_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_non_zero_index => .{ .row_cumulative_first_non_zero_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_non_zero_index => .{ .row_cumulative_last_non_zero_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_positive_index => .{ .row_cumulative_first_positive_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_positive_index => .{ .row_cumulative_last_positive_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_signbit_index => .{ .row_cumulative_first_signbit_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_signbit_index => .{ .row_cumulative_last_signbit_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_first_negative_index => .{ .row_cumulative_first_negative_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_last_negative_index => .{ .row_cumulative_last_negative_index = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_positive_count => .{ .row_cumulative_positive_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_negative_count => .{ .row_cumulative_negative_count = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_zero_ratio => .{ .row_cumulative_zero_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_non_zero_ratio => .{ .row_cumulative_non_zero_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_positive_ratio => .{ .row_cumulative_positive_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_negative_ratio => .{ .row_cumulative_negative_ratio = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_true => .{ .row_cumulative_any_true = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_true => .{ .row_cumulative_all_true = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_any_false => .{ .row_cumulative_any_false = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_all_false => .{ .row_cumulative_all_false = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_centered => .{ .row_centered = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_zscore => .{ .row_zscore = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_robust_zscore => .{ .row_robust_zscore = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_average_rank => .{ .row_average_rank = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_ordinal_rank => .{ .row_ordinal_rank = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_dense_rank => .{ .row_dense_rank = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_competition_rank => .{ .row_competition_rank = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_percent_rank => .{ .row_percent_rank = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cume_dist => .{ .row_cume_dist = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_sum => .{ .row_cumulative_sum = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_mean => .{ .row_cumulative_mean = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_logsumexp => .{ .row_cumulative_logsumexp = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_logmeanexp => .{ .row_cumulative_logmeanexp = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_geometric_mean => .{ .row_cumulative_geometric_mean = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_harmonic_mean => .{ .row_cumulative_harmonic_mean = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_skewness => .{ .row_cumulative_skewness = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_kurtosis => .{ .row_cumulative_kurtosis = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_rms => .{ .row_cumulative_rms = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_mean_abs => .{ .row_cumulative_mean_abs = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_mean_square => .{ .row_cumulative_mean_square = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_max_abs => .{ .row_cumulative_max_abs = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_min_abs => .{ .row_cumulative_min_abs = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_l1_norm => .{ .row_cumulative_l1_norm = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_l2_norm => .{ .row_cumulative_l2_norm = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_product => .{ .row_cumulative_product = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_max => .{ .row_cumulative_max = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_min => .{ .row_cumulative_min = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_cumulative_range => .{ .row_cumulative_range = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_iqr_outlier => .{ .row_iqr_outlier = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_tukey_winsorize => .{ .row_tukey_winsorize = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_max_indicator => .{ .row_max_indicator = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_min_indicator => .{ .row_min_indicator = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_minmax_scale => .{ .row_minmax_scale = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_l2_normalize => .{ .row_l2_normalize = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_l1_normalize => .{ .row_l1_normalize = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_sum_normalize => .{ .row_sum_normalize = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_mean_normalize => .{ .row_mean_normalize = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_max_abs_normalize => .{ .row_max_abs_normalize = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_softmax => .{ .row_softmax = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_log_softmax => .{ .row_log_softmax = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_softmin => .{ .row_softmin = .{
                    .names = names,
                    .output_names = output_names,
                } },
                .row_log_softmin => .{ .row_log_softmin = .{
                    .names = names,
                    .output_names = output_names,
                } },
                else => unreachable,
            };
        },
        .row_cumulative_variance, .row_cumulative_stddev, .row_cumulative_sem, .row_cumulative_cv, .row_cumulative_fano => |row_outputs, tag| blk: {
            const names = try cloneNameList(allocator, row_outputs.names);
            errdefer freeNameList(allocator, names);
            const output_names = try cloneNameList(allocator, row_outputs.output_names);
            errdefer freeNameList(allocator, output_names);
            break :blk switch (tag) {
                .row_cumulative_variance => .{ .row_cumulative_variance = .{
                    .names = names,
                    .output_names = output_names,
                    .correction = row_outputs.correction,
                } },
                .row_cumulative_stddev => .{ .row_cumulative_stddev = .{
                    .names = names,
                    .output_names = output_names,
                    .correction = row_outputs.correction,
                } },
                .row_cumulative_sem => .{ .row_cumulative_sem = .{
                    .names = names,
                    .output_names = output_names,
                    .correction = row_outputs.correction,
                } },
                .row_cumulative_cv => .{ .row_cumulative_cv = .{
                    .names = names,
                    .output_names = output_names,
                    .correction = row_outputs.correction,
                } },
                .row_cumulative_fano => .{ .row_cumulative_fano = .{
                    .names = names,
                    .output_names = output_names,
                    .correction = row_outputs.correction,
                } },
                else => unreachable,
            };
        },
        .row_null_count, .row_valid_count, .row_any_null, .row_all_null, .row_any_valid, .row_all_valid, .row_null_ratio, .row_valid_ratio, .row_first_valid_index, .row_last_valid_index, .row_first_null_index, .row_last_null_index, .row_argmin, .row_argmax, .row_median, .row_iqr, .row_interdecile_range, .row_midhinge, .row_trimean, .row_bowley_skewness, .row_quartile_coeff_dispersion, .row_kelley_skewness, .row_mad, .row_mode, .row_entropy, .row_gini_impurity, .row_perplexity, .row_inverse_simpson, .row_simpson_concentration, .row_evenness, .row_mode_count, .row_mode_ratio, .row_mode_margin, .row_mode_margin_ratio, .row_count_distinct, .row_n_unique, .row_is_duplicated, .row_is_unique, .row_sum, .row_mean, .row_logsumexp, .row_logmeanexp, .row_softmax_entropy, .row_softmax_perplexity, .row_softmax_confidence, .row_softmax_margin, .row_softmax_evenness, .row_softmax_concentration, .row_softmax_normalized_hhi, .row_softmax_gini_impurity, .row_softmax_inverse_simpson, .row_softmax_simpson_evenness, .row_logit_margin, .row_geometric_mean, .row_magnitude_geometric_mean, .row_harmonic_mean, .row_skewness, .row_magnitude_skewness, .row_kurtosis, .row_magnitude_kurtosis, .row_prod, .row_min, .row_max, .row_ptp, .row_magnitude_ptp, .row_midrange, .row_magnitude_midrange, .row_range_coeff, .row_magnitude_range_coeff, .row_mean_abs, .row_hhi, .row_magnitude_normalized_hhi, .row_magnitude_sparsity, .row_magnitude_inverse_simpson, .row_magnitude_simpson_evenness, .row_magnitude_dominance, .row_magnitude_dominance_margin, .row_magnitude_entropy, .row_magnitude_perplexity, .row_magnitude_evenness, .row_mean_abs_dev, .row_gini_mean_diff, .row_gini_coefficient, .row_mean_abs_dev_ratio, .row_rms, .row_l1_norm, .row_l2_norm, .row_true_count, .row_false_count, .row_any_true, .row_all_true, .row_any_false, .row_all_false, .row_first_true_index, .row_last_true_index, .row_first_false_index, .row_last_false_index, .row_true_ratio, .row_false_ratio, .row_any_zero, .row_all_zero, .row_any_non_zero, .row_all_non_zero, .row_any_positive_zero, .row_all_positive_zero, .row_any_negative_zero, .row_all_negative_zero, .row_any_positive, .row_all_positive, .row_any_signbit, .row_all_signbit, .row_any_negative, .row_all_negative, .row_any_nan, .row_all_nan, .row_any_inf, .row_all_inf, .row_any_positive_inf, .row_all_positive_inf, .row_any_negative_inf, .row_all_negative_inf, .row_any_finite, .row_all_finite, .row_any_normal, .row_all_normal, .row_any_subnormal, .row_all_subnormal, .row_any_non_finite, .row_all_non_finite, .row_nan_count, .row_nan_ratio, .row_inf_count, .row_inf_ratio, .row_positive_inf_count, .row_negative_inf_count, .row_positive_inf_ratio, .row_negative_inf_ratio, .row_zero_count, .row_zero_ratio, .row_positive_zero_count, .row_negative_zero_count, .row_positive_zero_ratio, .row_negative_zero_ratio, .row_non_zero_count, .row_non_zero_ratio, .row_first_nan_index, .row_last_nan_index, .row_first_inf_index, .row_last_inf_index, .row_first_positive_inf_index, .row_last_positive_inf_index, .row_first_negative_inf_index, .row_last_negative_inf_index, .row_first_finite_index, .row_last_finite_index, .row_first_normal_index, .row_last_normal_index, .row_first_subnormal_index, .row_last_subnormal_index, .row_first_non_finite_index, .row_last_non_finite_index, .row_first_positive_zero_index, .row_last_positive_zero_index, .row_first_negative_zero_index, .row_last_negative_zero_index, .row_first_signbit_index, .row_last_signbit_index, .row_first_zero_index, .row_last_zero_index, .row_first_non_zero_index, .row_last_non_zero_index, .row_first_positive_index, .row_last_positive_index, .row_first_negative_index, .row_last_negative_index, .row_positive_count, .row_positive_ratio, .row_signbit_count, .row_signbit_ratio, .row_negative_count, .row_negative_ratio, .row_finite_count, .row_finite_ratio, .row_normal_count, .row_normal_ratio, .row_subnormal_count, .row_subnormal_ratio, .row_non_finite_count, .row_non_finite_ratio => |row_count, tag| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk switch (tag) {
                .row_null_count => .{ .row_null_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_valid_count => .{ .row_valid_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_null => .{ .row_any_null = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_null => .{ .row_all_null = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_valid => .{ .row_any_valid = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_valid => .{ .row_all_valid = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_null_ratio => .{ .row_null_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_valid_ratio => .{ .row_valid_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_valid_index => .{ .row_first_valid_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_valid_index => .{ .row_last_valid_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_null_index => .{ .row_first_null_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_null_index => .{ .row_last_null_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_argmin => .{ .row_argmin = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_argmax => .{ .row_argmax = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_median => .{ .row_median = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_iqr => .{ .row_iqr = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_interdecile_range => .{ .row_interdecile_range = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_midhinge => .{ .row_midhinge = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_trimean => .{ .row_trimean = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_bowley_skewness => .{ .row_bowley_skewness = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_quartile_coeff_dispersion => .{ .row_quartile_coeff_dispersion = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_kelley_skewness => .{ .row_kelley_skewness = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mad => .{ .row_mad = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mode => .{ .row_mode = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_entropy => .{ .row_entropy = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_gini_impurity => .{ .row_gini_impurity = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_perplexity => .{ .row_perplexity = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_inverse_simpson => .{ .row_inverse_simpson = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_simpson_concentration => .{ .row_simpson_concentration = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_evenness => .{ .row_evenness = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mode_count => .{ .row_mode_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mode_ratio => .{ .row_mode_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mode_margin => .{ .row_mode_margin = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mode_margin_ratio => .{ .row_mode_margin_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_count_distinct => .{ .row_count_distinct = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_n_unique => .{ .row_n_unique = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_is_duplicated => .{ .row_is_duplicated = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_is_unique => .{ .row_is_unique = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_sum => .{ .row_sum = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mean => .{ .row_mean = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_logsumexp => .{ .row_logsumexp = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_logmeanexp => .{ .row_logmeanexp = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_entropy => .{ .row_softmax_entropy = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_perplexity => .{ .row_softmax_perplexity = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_confidence => .{ .row_softmax_confidence = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_margin => .{ .row_softmax_margin = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_evenness => .{ .row_softmax_evenness = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_concentration => .{ .row_softmax_concentration = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_normalized_hhi => .{ .row_softmax_normalized_hhi = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_gini_impurity => .{ .row_softmax_gini_impurity = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_inverse_simpson => .{ .row_softmax_inverse_simpson = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_softmax_simpson_evenness => .{ .row_softmax_simpson_evenness = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_logit_margin => .{ .row_logit_margin = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_geometric_mean => .{ .row_geometric_mean = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_geometric_mean => .{ .row_magnitude_geometric_mean = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_harmonic_mean => .{ .row_harmonic_mean = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_skewness => .{ .row_skewness = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_skewness => .{ .row_magnitude_skewness = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_kurtosis => .{ .row_kurtosis = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_kurtosis => .{ .row_magnitude_kurtosis = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_prod => .{ .row_prod = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_min => .{ .row_min = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_max => .{ .row_max = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_ptp => .{ .row_ptp = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_ptp => .{ .row_magnitude_ptp = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_midrange => .{ .row_midrange = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_midrange => .{ .row_magnitude_midrange = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_range_coeff => .{ .row_range_coeff = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_range_coeff => .{ .row_magnitude_range_coeff = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mean_abs => .{ .row_mean_abs = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_hhi => .{ .row_hhi = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_normalized_hhi => .{ .row_magnitude_normalized_hhi = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_sparsity => .{ .row_magnitude_sparsity = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_inverse_simpson => .{ .row_magnitude_inverse_simpson = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_simpson_evenness => .{ .row_magnitude_simpson_evenness = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_dominance => .{ .row_magnitude_dominance = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_dominance_margin => .{ .row_magnitude_dominance_margin = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_entropy => .{ .row_magnitude_entropy = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_perplexity => .{ .row_magnitude_perplexity = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_magnitude_evenness => .{ .row_magnitude_evenness = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mean_abs_dev => .{ .row_mean_abs_dev = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_gini_mean_diff => .{ .row_gini_mean_diff = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_gini_coefficient => .{ .row_gini_coefficient = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_mean_abs_dev_ratio => .{ .row_mean_abs_dev_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_rms => .{ .row_rms = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_l1_norm => .{ .row_l1_norm = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_l2_norm => .{ .row_l2_norm = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_true_count => .{ .row_true_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_false_count => .{ .row_false_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_true => .{ .row_any_true = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_true => .{ .row_all_true = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_false => .{ .row_any_false = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_false => .{ .row_all_false = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_true_index => .{ .row_first_true_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_true_index => .{ .row_last_true_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_false_index => .{ .row_first_false_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_false_index => .{ .row_last_false_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_true_ratio => .{ .row_true_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_false_ratio => .{ .row_false_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_zero => .{ .row_any_zero = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_zero => .{ .row_all_zero = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_non_zero => .{ .row_any_non_zero = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_non_zero => .{ .row_all_non_zero = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_positive_zero => .{ .row_any_positive_zero = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_positive_zero => .{ .row_all_positive_zero = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_negative_zero => .{ .row_any_negative_zero = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_negative_zero => .{ .row_all_negative_zero = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_positive => .{ .row_any_positive = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_positive => .{ .row_all_positive = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_signbit => .{ .row_any_signbit = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_signbit => .{ .row_all_signbit = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_negative => .{ .row_any_negative = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_negative => .{ .row_all_negative = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_nan => .{ .row_any_nan = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_nan => .{ .row_all_nan = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_inf => .{ .row_any_inf = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_inf => .{ .row_all_inf = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_positive_inf => .{ .row_any_positive_inf = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_positive_inf => .{ .row_all_positive_inf = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_negative_inf => .{ .row_any_negative_inf = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_negative_inf => .{ .row_all_negative_inf = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_finite => .{ .row_any_finite = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_finite => .{ .row_all_finite = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_normal => .{ .row_any_normal = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_normal => .{ .row_all_normal = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_subnormal => .{ .row_any_subnormal = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_subnormal => .{ .row_all_subnormal = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_any_non_finite => .{ .row_any_non_finite = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_all_non_finite => .{ .row_all_non_finite = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_nan_count => .{ .row_nan_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_nan_ratio => .{ .row_nan_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_inf_count => .{ .row_inf_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_inf_ratio => .{ .row_inf_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_positive_inf_count => .{ .row_positive_inf_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_negative_inf_count => .{ .row_negative_inf_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_positive_inf_ratio => .{ .row_positive_inf_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_negative_inf_ratio => .{ .row_negative_inf_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_zero_count => .{ .row_zero_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_zero_ratio => .{ .row_zero_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_positive_zero_count => .{ .row_positive_zero_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_negative_zero_count => .{ .row_negative_zero_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_positive_zero_ratio => .{ .row_positive_zero_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_negative_zero_ratio => .{ .row_negative_zero_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_non_zero_count => .{ .row_non_zero_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_non_zero_ratio => .{ .row_non_zero_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_nan_index => .{ .row_first_nan_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_nan_index => .{ .row_last_nan_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_inf_index => .{ .row_first_inf_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_inf_index => .{ .row_last_inf_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_positive_inf_index => .{ .row_first_positive_inf_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_positive_inf_index => .{ .row_last_positive_inf_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_negative_inf_index => .{ .row_first_negative_inf_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_negative_inf_index => .{ .row_last_negative_inf_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_finite_index => .{ .row_first_finite_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_finite_index => .{ .row_last_finite_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_normal_index => .{ .row_first_normal_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_normal_index => .{ .row_last_normal_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_subnormal_index => .{ .row_first_subnormal_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_subnormal_index => .{ .row_last_subnormal_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_non_finite_index => .{ .row_first_non_finite_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_non_finite_index => .{ .row_last_non_finite_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_positive_zero_index => .{ .row_first_positive_zero_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_positive_zero_index => .{ .row_last_positive_zero_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_negative_zero_index => .{ .row_first_negative_zero_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_negative_zero_index => .{ .row_last_negative_zero_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_signbit_index => .{ .row_first_signbit_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_signbit_index => .{ .row_last_signbit_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_zero_index => .{ .row_first_zero_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_zero_index => .{ .row_last_zero_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_non_zero_index => .{ .row_first_non_zero_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_non_zero_index => .{ .row_last_non_zero_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_positive_index => .{ .row_first_positive_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_positive_index => .{ .row_last_positive_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_first_negative_index => .{ .row_first_negative_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_last_negative_index => .{ .row_last_negative_index = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_positive_count => .{ .row_positive_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_positive_ratio => .{ .row_positive_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_signbit_count => .{ .row_signbit_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_signbit_ratio => .{ .row_signbit_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_negative_count => .{ .row_negative_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_negative_ratio => .{ .row_negative_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_finite_count => .{ .row_finite_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_finite_ratio => .{ .row_finite_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_normal_count => .{ .row_normal_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_normal_ratio => .{ .row_normal_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_subnormal_count => .{ .row_subnormal_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_subnormal_ratio => .{ .row_subnormal_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_non_finite_count => .{ .row_non_finite_count = .{
                    .names = names,
                    .output_name = output_name,
                } },
                .row_non_finite_ratio => .{ .row_non_finite_ratio = .{
                    .names = names,
                    .output_name = output_name,
                } },
                else => unreachable,
            };
        },
        .row_quantile => |row_quantile| blk: {
            const names = try cloneNameList(allocator, row_quantile.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_quantile.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_quantile = .{
                .names = names,
                .output_name = output_name,
                .q = row_quantile.q,
            } };
        },
        .row_quantile_range => |row_quantile_range| blk: {
            const names = try cloneNameList(allocator, row_quantile_range.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_quantile_range.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_quantile_range = .{
                .names = names,
                .output_name = output_name,
                .low_q = row_quantile_range.low_q,
                .high_q = row_quantile_range.high_q,
            } };
        },
        .row_trimmed_mean => |row_trimmed_mean| blk: {
            const names = try cloneNameList(allocator, row_trimmed_mean.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_trimmed_mean.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_trimmed_mean = .{
                .names = names,
                .output_name = output_name,
                .trim_fraction = row_trimmed_mean.trim_fraction,
            } };
        },
        .row_winsorized_mean => |row_winsorized_mean| blk: {
            const names = try cloneNameList(allocator, row_winsorized_mean.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_winsorized_mean.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_winsorized_mean = .{
                .names = names,
                .output_name = output_name,
                .winsor_fraction = row_winsorized_mean.winsor_fraction,
            } };
        },
        .row_pair_count => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_pair_count = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_weighted_pair_weight_sum => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_pair_weight_sum"),
        .row_weighted_pair_positive_count => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_pair_positive_count"),
        .row_weighted_pair_effective_n => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_pair_effective_n"),
        .row_weighted_mean => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_mean"),
        .row_weighted_sum => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_sum"),
        .row_cumulative_weighted_sum => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_sum"),
        .row_cumulative_weighted_mean => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_mean"),
        .row_cumulative_weighted_mean_square => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_mean_square"),
        .row_cumulative_weighted_rms => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_rms"),
        .row_cumulative_weighted_mean_abs => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_mean_abs"),
        .row_cumulative_weighted_l1_norm => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_l1_norm"),
        .row_cumulative_weighted_l2_norm => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_l2_norm"),
        .row_cumulative_weighted_min => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_min"),
        .row_cumulative_weighted_max => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_max"),
        .row_cumulative_weighted_max_abs => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_max_abs"),
        .row_cumulative_weighted_min_abs => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_min_abs"),
        .row_cumulative_weighted_range => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_range"),
        .row_cumulative_weighted_midrange => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_midrange"),
        .row_cumulative_weighted_range_coeff => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_range_coeff"),
        .row_cumulative_weighted_product => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_product"),
        .row_cumulative_weighted_geometric_mean => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_geometric_mean"),
        .row_cumulative_weighted_harmonic_mean => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_harmonic_mean"),
        .row_cumulative_weighted_logsumexp => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_logsumexp"),
        .row_cumulative_weighted_logmeanexp => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_logmeanexp"),
        .row_cumulative_weighted_variance => |row_weighted| try cloneRowWeightedColumnOutputsDispersion(Self, allocator, row_weighted, "row_cumulative_weighted_variance"),
        .row_cumulative_weighted_stddev => |row_weighted| try cloneRowWeightedColumnOutputsDispersion(Self, allocator, row_weighted, "row_cumulative_weighted_stddev"),
        .row_cumulative_weighted_sem => |row_weighted| try cloneRowWeightedColumnOutputsDispersion(Self, allocator, row_weighted, "row_cumulative_weighted_sem"),
        .row_cumulative_weighted_cv => |row_weighted| try cloneRowWeightedColumnOutputsDispersion(Self, allocator, row_weighted, "row_cumulative_weighted_cv"),
        .row_cumulative_weighted_fano => |row_weighted| try cloneRowWeightedColumnOutputsDispersion(Self, allocator, row_weighted, "row_cumulative_weighted_fano"),
        .row_cumulative_weighted_skewness => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_skewness"),
        .row_cumulative_weighted_kurtosis => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_kurtosis"),
        .row_cumulative_weighted_quantile => |row_weighted| try cloneRowWeightedColumnOutputsQuantile(Self, allocator, row_weighted, "row_cumulative_weighted_quantile"),
        .row_cumulative_weighted_median => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_median"),
        .row_cumulative_weighted_iqr => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_iqr"),
        .row_cumulative_weighted_mad => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_mad"),
        .row_cumulative_weighted_trimmed_mean => |row_weighted| try cloneRowWeightedColumnOutputsQuantile(Self, allocator, row_weighted, "row_cumulative_weighted_trimmed_mean"),
        .row_cumulative_weighted_winsorized_mean => |row_weighted| try cloneRowWeightedColumnOutputsQuantile(Self, allocator, row_weighted, "row_cumulative_weighted_winsorized_mean"),
        .row_cumulative_weighted_interdecile_range => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_interdecile_range"),
        .row_cumulative_weighted_midhinge => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_midhinge"),
        .row_cumulative_weighted_trimean => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_trimean"),
        .row_cumulative_weighted_bowley_skewness => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_bowley_skewness"),
        .row_cumulative_weighted_quartile_coeff_dispersion => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_quartile_coeff_dispersion"),
        .row_cumulative_weighted_kelley_skewness => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_kelley_skewness"),
        .row_cumulative_weighted_mode => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_mode"),
        .row_cumulative_weighted_weight_sum => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_weight_sum"),
        .row_cumulative_weighted_positive_count => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_positive_count"),
        .row_cumulative_weighted_effective_n => |row_weighted| try cloneRowWeightedColumnOutputs(Self, allocator, row_weighted, "row_cumulative_weighted_effective_n"),
        .row_weighted_weight_sum => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_weight_sum"),
        .row_weighted_positive_count => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_positive_count"),
        .row_weighted_effective_n => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_effective_n"),
        .row_weighted_mean_square => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_mean_square"),
        .row_weighted_rms => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_rms"),
        .row_weighted_mean_abs => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_mean_abs"),
        .row_weighted_l1_norm => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_l1_norm"),
        .row_weighted_l2_norm => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_l2_norm"),
        .row_weighted_min => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_min"),
        .row_weighted_max => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_max"),
        .row_weighted_max_abs => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_max_abs"),
        .row_weighted_min_abs => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_min_abs"),
        .row_weighted_range => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_range"),
        .row_weighted_midrange => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_midrange"),
        .row_weighted_range_coeff => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_range_coeff"),
        .row_weighted_product => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_product"),
        .row_weighted_geometric_mean => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_geometric_mean"),
        .row_weighted_harmonic_mean => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_harmonic_mean"),
        .row_weighted_logsumexp => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_logsumexp"),
        .row_weighted_logmeanexp => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_logmeanexp"),
        .row_weighted_median => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_median = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_weighted_iqr => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_iqr = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_weighted_mad => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_mad = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_weighted_interdecile_range => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_interdecile_range = .{ .value_names = value_names, .weight_names = weight_names, .output_name = output_name } };
        },
        .row_weighted_midhinge => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_midhinge = .{ .value_names = value_names, .weight_names = weight_names, .output_name = output_name } };
        },
        .row_weighted_trimean => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_trimean = .{ .value_names = value_names, .weight_names = weight_names, .output_name = output_name } };
        },
        .row_weighted_bowley_skewness => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_bowley_skewness = .{ .value_names = value_names, .weight_names = weight_names, .output_name = output_name } };
        },
        .row_weighted_quartile_coeff_dispersion => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_quartile_coeff_dispersion = .{ .value_names = value_names, .weight_names = weight_names, .output_name = output_name } };
        },
        .row_weighted_kelley_skewness => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_kelley_skewness = .{ .value_names = value_names, .weight_names = weight_names, .output_name = output_name } };
        },
        .row_weighted_mode => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_mode = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_weighted_mode_weight => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_mode_weight = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_weighted_mode_ratio => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_mode_ratio = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_weighted_mode_margin => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_mode_margin = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_weighted_mode_margin_ratio => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_mode_margin_ratio = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_weighted_entropy => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_entropy"),
        .row_weighted_gini_impurity => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_gini_impurity"),
        .row_weighted_perplexity => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_perplexity"),
        .row_weighted_inverse_simpson => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_inverse_simpson"),
        .row_weighted_simpson_concentration => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_simpson_concentration"),
        .row_weighted_evenness => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_evenness"),
        .row_weighted_mean_abs_dev => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_mean_abs_dev"),
        .row_weighted_mean_abs_dev_ratio => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_mean_abs_dev_ratio"),
        .row_weighted_gini_mean_diff => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_gini_mean_diff"),
        .row_weighted_gini_coefficient => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_gini_coefficient"),
        .row_weighted_skewness => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_skewness"),
        .row_weighted_kurtosis => |row_weighted| try cloneRowWeightedMean(Self, allocator, row_weighted, "row_weighted_kurtosis"),
        .row_weighted_variance => |row_weighted| try cloneRowWeightedDispersion(Self, allocator, row_weighted, "row_weighted_variance"),
        .row_weighted_stddev => |row_weighted| try cloneRowWeightedDispersion(Self, allocator, row_weighted, "row_weighted_stddev"),
        .row_weighted_sem => |row_weighted| try cloneRowWeightedDispersion(Self, allocator, row_weighted, "row_weighted_sem"),
        .row_weighted_cv => |row_weighted| try cloneRowWeightedDispersion(Self, allocator, row_weighted, "row_weighted_cv"),
        .row_weighted_fano => |row_weighted| try cloneRowWeightedDispersion(Self, allocator, row_weighted, "row_weighted_fano"),
        .row_weighted_quantile => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_quantile = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
                .q = row_weighted.q,
            } };
        },
        .row_weighted_trimmed_mean => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_trimmed_mean = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
                .q = row_weighted.q,
            } };
        },
        .row_weighted_winsorized_mean => |row_weighted| blk: {
            const value_names = try cloneNameList(allocator, row_weighted.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_weighted.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_weighted.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_weighted_winsorized_mean = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
                .q = row_weighted.q,
            } };
        },
        .row_weighted_dot => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_dot"),
        .row_weighted_cosine_similarity => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_cosine_similarity"),
        .row_weighted_squared_euclidean_distance => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_squared_euclidean_distance"),
        .row_weighted_euclidean_distance => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_euclidean_distance"),
        .row_weighted_manhattan_distance => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_manhattan_distance"),
        .row_weighted_chebyshev_distance => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_chebyshev_distance"),
        .row_weighted_canberra_distance => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_canberra_distance"),
        .row_weighted_bray_curtis_distance => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_bray_curtis_distance"),
        .row_weighted_mean_error => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_mean_error"),
        .row_weighted_mae => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_mae"),
        .row_weighted_mse => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_mse"),
        .row_weighted_rmse => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_rmse"),
        .row_weighted_mape => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_mape"),
        .row_weighted_smape => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_smape"),
        .row_weighted_covariance => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_covariance"),
        .row_weighted_correlation => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_correlation"),
        .row_weighted_beta => |row_weighted| try cloneRowWeightedPair(Self, allocator, row_weighted, "row_weighted_beta"),
        .row_dot => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_dot = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_cosine_similarity => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_cosine_similarity = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_squared_euclidean_distance => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_squared_euclidean_distance = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_euclidean_distance => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_euclidean_distance = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_manhattan_distance => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_manhattan_distance = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_chebyshev_distance => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_chebyshev_distance = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_canberra_distance => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_canberra_distance = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_bray_curtis_distance => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_bray_curtis_distance = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_mean_error => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_mean_error = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_mae => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_mae = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_mse => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_mse = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_rmse => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_rmse = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_mape => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_mape = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_smape => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_smape = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_covariance => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_covariance = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_correlation => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_correlation = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_beta => |row_paired| blk: {
            const value_names = try cloneNameList(allocator, row_paired.value_names);
            errdefer freeNameList(allocator, value_names);
            const weight_names = try cloneNameList(allocator, row_paired.weight_names);
            errdefer freeNameList(allocator, weight_names);
            const output_name = try allocator.dupe(u8, row_paired.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_beta = .{
                .value_names = value_names,
                .weight_names = weight_names,
                .output_name = output_name,
            } };
        },
        .row_variance => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_variance = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .row_magnitude_variance => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_magnitude_variance = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .row_stddev => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_stddev = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .row_magnitude_stddev => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_magnitude_stddev = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .row_sem => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_sem = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .row_magnitude_sem => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_magnitude_sem = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .row_cv => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_cv = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .row_magnitude_cv => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_magnitude_cv = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .row_magnitude_fano => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_magnitude_fano = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .row_fano => |row_dispersion| blk: {
            const names = try cloneNameList(allocator, row_dispersion.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_dispersion.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .row_fano = .{
                .names = names,
                .output_name = output_name,
                .correction = row_dispersion.correction,
            } };
        },
        .with_column_compare => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const lhs_name = try allocator.dupe(u8, expr.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, expr.rhs_name);
            errdefer allocator.free(rhs_name);
            break :blk .{ .with_column_compare = .{
                .name = name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .op = expr.op,
            } };
        },
        .with_column_compare_scalar => |expr| blk: {
            const name = try allocator.dupe(u8, expr.name);
            errdefer allocator.free(name);
            const input_name = try allocator.dupe(u8, expr.input_name);
            errdefer allocator.free(input_name);
            break :blk .{ .with_column_compare_scalar = .{
                .name = name,
                .input_name = input_name,
                .op = expr.op,
                .scalar = expr.scalar,
            } };
        },
        .filter_mask => |mask| .{ .filter_mask = try mask.clone() },
        .filter_column => |name| .{ .filter_column = try allocator.dupe(u8, name) },
        .filter_between_column => |range| blk: {
            const name = try allocator.dupe(u8, range.name);
            break :blk .{ .filter_between_column = .{
                .name = name,
                .lower = range.lower,
                .upper = range.upper,
                .lower_inclusive = range.lower_inclusive,
                .upper_inclusive = range.upper_inclusive,
                .keep_inside = range.keep_inside,
            } };
        },
        .filter_isin_column => |membership| blk: {
            const input_name = try allocator.dupe(u8, membership.input_name);
            errdefer allocator.free(input_name);
            const test_name = try allocator.dupe(u8, membership.test_name);
            errdefer allocator.free(test_name);
            break :blk .{ .filter_isin_column = .{
                .input_name = input_name,
                .test_name = test_name,
                .invert = membership.invert,
            } };
        },
        .filter_isin_values => |membership| blk: {
            const input_name = try allocator.dupe(u8, membership.input_name);
            errdefer allocator.free(input_name);
            var values = try membership.values.clone();
            errdefer values.deinit();
            break :blk .{ .filter_isin_values = .{
                .input_name = input_name,
                .values = values,
                .invert = membership.invert,
            } };
        },
        .drop_rows_by_mask_column => |name| .{ .drop_rows_by_mask_column = try allocator.dupe(u8, name) },
        .where_indices_column => |predicate| blk: {
            const name = try allocator.dupe(u8, predicate.name);
            errdefer allocator.free(name);
            const output_name = try allocator.dupe(u8, predicate.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .where_indices_column = .{
                .name = name,
                .output_name = output_name,
            } };
        },
        .filter_scalar => |filter_op| .{ .filter_scalar = .{
            .name = try allocator.dupe(u8, filter_op.name),
            .op = filter_op.op,
            .scalar = filter_op.scalar,
            .keep_matches = filter_op.keep_matches,
        } },
        .group_id => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_id = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_first_row_index => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_first_row_index = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_last_row_index => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_last_row_index = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_is_first_row => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_is_first_row = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_is_last_row => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_is_last_row = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_is_singleton => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_is_singleton = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_is_duplicated => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_is_duplicated = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_cume_dist => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cume_dist = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_percent_rank => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_percent_rank = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_reverse_cume_dist => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_reverse_cume_dist = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_reverse_percent_rank => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_reverse_percent_rank = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_lag, .group_lead, .group_first_row_value, .group_last_row_value, .group_nth_row_value, .group_first_valid_value, .group_last_valid_value, .group_nth_valid_value, .group_fill_null_forward, .group_fill_null_backward, .group_cumulative_valid_count, .group_cumulative_null_count, .group_cumulative_valid_ratio, .group_cumulative_null_ratio, .group_cumulative_first_valid_index, .group_cumulative_last_valid_index, .group_cumulative_first_null_index, .group_cumulative_last_null_index, .group_cumulative_nan_count, .group_cumulative_nan_ratio, .group_cumulative_inf_count, .group_cumulative_inf_ratio, .group_cumulative_positive_inf_count, .group_cumulative_positive_inf_ratio, .group_cumulative_negative_inf_count, .group_cumulative_negative_inf_ratio, .group_cumulative_finite_count, .group_cumulative_finite_ratio, .group_cumulative_normal_count, .group_cumulative_normal_ratio, .group_cumulative_subnormal_count, .group_cumulative_subnormal_ratio, .group_cumulative_non_finite_count, .group_cumulative_non_finite_ratio, .group_cumulative_zero_count, .group_cumulative_zero_ratio, .group_cumulative_positive_zero_count, .group_cumulative_positive_zero_ratio, .group_cumulative_negative_zero_count, .group_cumulative_negative_zero_ratio, .group_cumulative_non_zero_count, .group_cumulative_non_zero_ratio, .group_cumulative_positive_count, .group_cumulative_positive_ratio, .group_cumulative_signbit_count, .group_cumulative_signbit_ratio, .group_cumulative_negative_count, .group_cumulative_negative_ratio, .group_cumulative_first_nan_index, .group_cumulative_last_nan_index, .group_cumulative_first_inf_index, .group_cumulative_last_inf_index, .group_cumulative_first_positive_inf_index, .group_cumulative_last_positive_inf_index, .group_cumulative_first_negative_inf_index, .group_cumulative_last_negative_inf_index, .group_cumulative_first_finite_index, .group_cumulative_last_finite_index, .group_cumulative_first_normal_index, .group_cumulative_last_normal_index, .group_cumulative_first_subnormal_index, .group_cumulative_last_subnormal_index, .group_cumulative_first_non_finite_index, .group_cumulative_last_non_finite_index, .group_cumulative_first_zero_index, .group_cumulative_last_zero_index, .group_cumulative_first_positive_zero_index, .group_cumulative_last_positive_zero_index, .group_cumulative_first_negative_zero_index, .group_cumulative_last_negative_zero_index, .group_cumulative_first_non_zero_index, .group_cumulative_last_non_zero_index, .group_cumulative_first_positive_index, .group_cumulative_last_positive_index, .group_cumulative_first_signbit_index, .group_cumulative_last_signbit_index, .group_cumulative_first_negative_index, .group_cumulative_last_negative_index, .group_cumulative_distinct_count, .group_cumulative_n_unique, .group_cumulative_mode, .group_cumulative_mode_count, .group_cumulative_mode_ratio, .group_cumulative_mode_margin, .group_cumulative_mode_margin_ratio, .group_cumulative_entropy, .group_cumulative_gini_impurity, .group_cumulative_perplexity, .group_cumulative_inverse_simpson, .group_cumulative_simpson_concentration, .group_cumulative_evenness, .group_cumulative_mean_abs_dev, .group_cumulative_mean_abs_dev_ratio, .group_cumulative_gini_mean_diff, .group_cumulative_gini_coefficient, .group_cumulative_median, .group_cumulative_iqr, .group_cumulative_mad, .group_cumulative_interdecile_range, .group_cumulative_midhinge, .group_cumulative_trimean, .group_cumulative_bowley_skewness, .group_cumulative_quartile_coeff_dispersion, .group_cumulative_kelley_skewness, .group_cumulative_any, .group_cumulative_all, .group_cumulative_true_count, .group_cumulative_false_count, .group_cumulative_true_ratio, .group_cumulative_false_ratio, .group_cumulative_first_true_index, .group_cumulative_last_true_index, .group_cumulative_first_false_index, .group_cumulative_last_false_index, .group_cumulative_sum, .group_cumulative_mean, .group_cumulative_product, .group_cumulative_min, .group_cumulative_max, .group_cumulative_variance, .group_cumulative_stddev, .group_cumulative_sem, .group_cumulative_cv, .group_cumulative_fano, .group_cumulative_skewness, .group_cumulative_kurtosis, .group_cumulative_mean_abs, .group_cumulative_mean_square, .group_cumulative_rms, .group_cumulative_max_abs, .group_cumulative_min_abs, .group_cumulative_l1_norm, .group_cumulative_l2_norm, .group_cumulative_range, .group_cumulative_midrange, .group_cumulative_range_coeff, .group_cumulative_logsumexp, .group_cumulative_logmeanexp, .group_cumulative_geometric_mean, .group_cumulative_harmonic_mean, .group_cumulative_argmin, .group_cumulative_argmax => |shift, tag| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk switch (tag) {
                .group_lag => .{ .group_lag = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_lead => .{ .group_lead = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_first_row_value => .{ .group_first_row_value = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_last_row_value => .{ .group_last_row_value = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_nth_row_value => .{ .group_nth_row_value = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_first_valid_value => .{ .group_first_valid_value = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_last_valid_value => .{ .group_last_valid_value = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_nth_valid_value => .{ .group_nth_valid_value = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_fill_null_forward => .{ .group_fill_null_forward = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_fill_null_backward => .{ .group_fill_null_backward = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_valid_count => .{ .group_cumulative_valid_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_null_count => .{ .group_cumulative_null_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_valid_ratio => .{ .group_cumulative_valid_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_null_ratio => .{ .group_cumulative_null_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_valid_index => .{ .group_cumulative_first_valid_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_valid_index => .{ .group_cumulative_last_valid_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_null_index => .{ .group_cumulative_first_null_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_null_index => .{ .group_cumulative_last_null_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_nan_count => .{ .group_cumulative_nan_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_nan_ratio => .{ .group_cumulative_nan_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_inf_count => .{ .group_cumulative_inf_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_inf_ratio => .{ .group_cumulative_inf_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_positive_inf_count => .{ .group_cumulative_positive_inf_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_positive_inf_ratio => .{ .group_cumulative_positive_inf_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_negative_inf_count => .{ .group_cumulative_negative_inf_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_negative_inf_ratio => .{ .group_cumulative_negative_inf_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_finite_count => .{ .group_cumulative_finite_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_finite_ratio => .{ .group_cumulative_finite_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_normal_count => .{ .group_cumulative_normal_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_normal_ratio => .{ .group_cumulative_normal_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_subnormal_count => .{ .group_cumulative_subnormal_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_subnormal_ratio => .{ .group_cumulative_subnormal_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_non_finite_count => .{ .group_cumulative_non_finite_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_non_finite_ratio => .{ .group_cumulative_non_finite_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_zero_count => .{ .group_cumulative_zero_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_zero_ratio => .{ .group_cumulative_zero_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_positive_zero_count => .{ .group_cumulative_positive_zero_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_positive_zero_ratio => .{ .group_cumulative_positive_zero_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_negative_zero_count => .{ .group_cumulative_negative_zero_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_negative_zero_ratio => .{ .group_cumulative_negative_zero_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_non_zero_count => .{ .group_cumulative_non_zero_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_non_zero_ratio => .{ .group_cumulative_non_zero_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_positive_count => .{ .group_cumulative_positive_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_positive_ratio => .{ .group_cumulative_positive_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_signbit_count => .{ .group_cumulative_signbit_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_signbit_ratio => .{ .group_cumulative_signbit_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_negative_count => .{ .group_cumulative_negative_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_negative_ratio => .{ .group_cumulative_negative_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_nan_index => .{ .group_cumulative_first_nan_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_nan_index => .{ .group_cumulative_last_nan_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_inf_index => .{ .group_cumulative_first_inf_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_inf_index => .{ .group_cumulative_last_inf_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_positive_inf_index => .{ .group_cumulative_first_positive_inf_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_positive_inf_index => .{ .group_cumulative_last_positive_inf_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_negative_inf_index => .{ .group_cumulative_first_negative_inf_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_negative_inf_index => .{ .group_cumulative_last_negative_inf_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_finite_index => .{ .group_cumulative_first_finite_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_finite_index => .{ .group_cumulative_last_finite_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_normal_index => .{ .group_cumulative_first_normal_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_normal_index => .{ .group_cumulative_last_normal_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_subnormal_index => .{ .group_cumulative_first_subnormal_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_subnormal_index => .{ .group_cumulative_last_subnormal_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_non_finite_index => .{ .group_cumulative_first_non_finite_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_non_finite_index => .{ .group_cumulative_last_non_finite_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_zero_index => .{ .group_cumulative_first_zero_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_zero_index => .{ .group_cumulative_last_zero_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_positive_zero_index => .{ .group_cumulative_first_positive_zero_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_positive_zero_index => .{ .group_cumulative_last_positive_zero_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_negative_zero_index => .{ .group_cumulative_first_negative_zero_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_negative_zero_index => .{ .group_cumulative_last_negative_zero_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_non_zero_index => .{ .group_cumulative_first_non_zero_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_non_zero_index => .{ .group_cumulative_last_non_zero_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_positive_index => .{ .group_cumulative_first_positive_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_positive_index => .{ .group_cumulative_last_positive_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_signbit_index => .{ .group_cumulative_first_signbit_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_signbit_index => .{ .group_cumulative_last_signbit_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_negative_index => .{ .group_cumulative_first_negative_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_negative_index => .{ .group_cumulative_last_negative_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_distinct_count => .{ .group_cumulative_distinct_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_n_unique => .{ .group_cumulative_n_unique = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mode => .{ .group_cumulative_mode = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mode_count => .{ .group_cumulative_mode_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mode_ratio => .{ .group_cumulative_mode_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mode_margin => .{ .group_cumulative_mode_margin = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mode_margin_ratio => .{ .group_cumulative_mode_margin_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_entropy => .{ .group_cumulative_entropy = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_gini_impurity => .{ .group_cumulative_gini_impurity = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_perplexity => .{ .group_cumulative_perplexity = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_inverse_simpson => .{ .group_cumulative_inverse_simpson = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_simpson_concentration => .{ .group_cumulative_simpson_concentration = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_evenness => .{ .group_cumulative_evenness = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mean_abs_dev => .{ .group_cumulative_mean_abs_dev = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mean_abs_dev_ratio => .{ .group_cumulative_mean_abs_dev_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_gini_mean_diff => .{ .group_cumulative_gini_mean_diff = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_gini_coefficient => .{ .group_cumulative_gini_coefficient = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_median => .{ .group_cumulative_median = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_iqr => .{ .group_cumulative_iqr = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mad => .{ .group_cumulative_mad = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_interdecile_range => .{ .group_cumulative_interdecile_range = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_midhinge => .{ .group_cumulative_midhinge = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_trimean => .{ .group_cumulative_trimean = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_bowley_skewness => .{ .group_cumulative_bowley_skewness = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_quartile_coeff_dispersion => .{ .group_cumulative_quartile_coeff_dispersion = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_kelley_skewness => .{ .group_cumulative_kelley_skewness = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_any => .{ .group_cumulative_any = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_all => .{ .group_cumulative_all = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_true_count => .{ .group_cumulative_true_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_false_count => .{ .group_cumulative_false_count = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_true_ratio => .{ .group_cumulative_true_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_false_ratio => .{ .group_cumulative_false_ratio = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_true_index => .{ .group_cumulative_first_true_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_true_index => .{ .group_cumulative_last_true_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_first_false_index => .{ .group_cumulative_first_false_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_last_false_index => .{ .group_cumulative_last_false_index = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_sum => .{ .group_cumulative_sum = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mean => .{ .group_cumulative_mean = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_product => .{ .group_cumulative_product = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_min => .{ .group_cumulative_min = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_max => .{ .group_cumulative_max = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_variance => .{ .group_cumulative_variance = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_stddev => .{ .group_cumulative_stddev = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_sem => .{ .group_cumulative_sem = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_cv => .{ .group_cumulative_cv = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_fano => .{ .group_cumulative_fano = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_skewness => .{ .group_cumulative_skewness = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_kurtosis => .{ .group_cumulative_kurtosis = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mean_abs => .{ .group_cumulative_mean_abs = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_mean_square => .{ .group_cumulative_mean_square = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_rms => .{ .group_cumulative_rms = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_max_abs => .{ .group_cumulative_max_abs = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_min_abs => .{ .group_cumulative_min_abs = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_l1_norm => .{ .group_cumulative_l1_norm = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_l2_norm => .{ .group_cumulative_l2_norm = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_range => .{ .group_cumulative_range = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_midrange => .{ .group_cumulative_midrange = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_range_coeff => .{ .group_cumulative_range_coeff = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_logsumexp => .{ .group_cumulative_logsumexp = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_logmeanexp => .{ .group_cumulative_logmeanexp = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_geometric_mean => .{ .group_cumulative_geometric_mean = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_harmonic_mean => .{ .group_cumulative_harmonic_mean = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_argmin => .{ .group_cumulative_argmin = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                .group_cumulative_argmax => .{ .group_cumulative_argmax = .{
                    .names = names,
                    .value_name = value_name,
                    .output_name = output_name,
                    .offset = shift.offset,
                } },
                else => unreachable,
            };
        },
        .group_cumulative_quantile => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_quantile = .{
                .names = names,
                .value_name = value_name,
                .output_name = output_name,
                .quantile = shift.quantile,
            } };
        },
        .group_cumulative_trimmed_mean => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_trimmed_mean = .{
                .names = names,
                .value_name = value_name,
                .output_name = output_name,
                .quantile = shift.quantile,
            } };
        },
        .group_cumulative_winsorized_mean => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_winsorized_mean = .{
                .names = names,
                .value_name = value_name,
                .output_name = output_name,
                .quantile = shift.quantile,
            } };
        },
        .group_cumulative_weighted_mean => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, shift.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_weighted_mean = .{
                .names = names,
                .value_name = value_name,
                .weight_name = weight_name,
                .output_name = output_name,
            } };
        },
        .group_cumulative_weighted_mean_square => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_mean_square"),
        .group_cumulative_weighted_rms => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_rms"),
        .group_cumulative_weighted_median => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, shift.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_weighted_median = .{
                .names = names,
                .value_name = value_name,
                .weight_name = weight_name,
                .output_name = output_name,
            } };
        },
        .group_cumulative_weighted_quantile => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, shift.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_weighted_quantile = .{
                .names = names,
                .value_name = value_name,
                .weight_name = weight_name,
                .output_name = output_name,
                .quantile = shift.quantile,
            } };
        },
        .group_cumulative_weighted_trimmed_mean => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, shift.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_weighted_trimmed_mean = .{
                .names = names,
                .value_name = value_name,
                .weight_name = weight_name,
                .output_name = output_name,
                .quantile = shift.quantile,
            } };
        },
        .group_cumulative_weighted_winsorized_mean => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, shift.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_weighted_winsorized_mean = .{
                .names = names,
                .value_name = value_name,
                .weight_name = weight_name,
                .output_name = output_name,
                .quantile = shift.quantile,
            } };
        },
        .group_cumulative_weighted_interdecile_range => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_interdecile_range"),
        .group_cumulative_weighted_midhinge => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_midhinge"),
        .group_cumulative_weighted_trimean => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_trimean"),
        .group_cumulative_weighted_bowley_skewness => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_bowley_skewness"),
        .group_cumulative_weighted_quartile_coeff_dispersion => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_quartile_coeff_dispersion"),
        .group_cumulative_weighted_kelley_skewness => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_kelley_skewness"),
        .group_cumulative_weighted_iqr => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, shift.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_weighted_iqr = .{
                .names = names,
                .value_name = value_name,
                .weight_name = weight_name,
                .output_name = output_name,
            } };
        },
        .group_cumulative_weighted_mad => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, shift.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_weighted_mad = .{
                .names = names,
                .value_name = value_name,
                .weight_name = weight_name,
                .output_name = output_name,
            } };
        },
        .group_cumulative_weighted_mode => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_mode"),
        .group_cumulative_weighted_mode_weight => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_mode_weight"),
        .group_cumulative_weighted_mode_ratio => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_mode_ratio"),
        .group_cumulative_weighted_mode_margin => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_mode_margin"),
        .group_cumulative_weighted_mode_margin_ratio => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_mode_margin_ratio"),
        .group_cumulative_weighted_entropy => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_entropy"),
        .group_cumulative_weighted_gini_impurity => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_gini_impurity"),
        .group_cumulative_weighted_perplexity => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_perplexity"),
        .group_cumulative_weighted_inverse_simpson => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_inverse_simpson"),
        .group_cumulative_weighted_simpson_concentration => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_simpson_concentration"),
        .group_cumulative_weighted_evenness => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_evenness"),
        .group_cumulative_weighted_mean_abs_dev => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_mean_abs_dev"),
        .group_cumulative_weighted_mean_abs_dev_ratio => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_mean_abs_dev_ratio"),
        .group_cumulative_weighted_gini_mean_diff => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_gini_mean_diff"),
        .group_cumulative_weighted_gini_coefficient => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_gini_coefficient"),
        .group_cumulative_weighted_dot => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_dot"),
        .group_cumulative_weighted_cosine_similarity => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_cosine_similarity"),
        .group_cumulative_weighted_squared_euclidean_distance => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_squared_euclidean_distance"),
        .group_cumulative_weighted_euclidean_distance => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_euclidean_distance"),
        .group_cumulative_weighted_manhattan_distance => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_manhattan_distance"),
        .group_cumulative_weighted_chebyshev_distance => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_chebyshev_distance"),
        .group_cumulative_weighted_canberra_distance => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_canberra_distance"),
        .group_cumulative_weighted_bray_curtis_distance => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_bray_curtis_distance"),
        .group_cumulative_weighted_mean_error => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_mean_error"),
        .group_cumulative_weighted_mae => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_mae"),
        .group_cumulative_weighted_mse => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_mse"),
        .group_cumulative_weighted_rmse => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_rmse"),
        .group_cumulative_weighted_mape => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_mape"),
        .group_cumulative_weighted_smape => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_smape"),
        .group_cumulative_weighted_covariance => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_covariance"),
        .group_cumulative_weighted_correlation => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_correlation"),
        .group_cumulative_weighted_beta => |shift| try cloneGroupWeightedPairShift(Self, allocator, shift, "group_cumulative_weighted_beta"),
        .group_cumulative_weighted_sum => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_sum"),
        .group_cumulative_weighted_product => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_product"),
        .group_cumulative_weighted_weight_sum => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_weight_sum"),
        .group_cumulative_weighted_positive_count => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_positive_count"),
        .group_cumulative_weighted_effective_n => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_effective_n"),
        .group_cumulative_weighted_min => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_min"),
        .group_cumulative_weighted_max => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_max"),
        .group_cumulative_weighted_mean_abs => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_mean_abs"),
        .group_cumulative_weighted_l1_norm => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_l1_norm"),
        .group_cumulative_weighted_l2_norm => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_l2_norm"),
        .group_cumulative_weighted_max_abs => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_max_abs"),
        .group_cumulative_weighted_min_abs => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_min_abs"),
        .group_cumulative_weighted_geometric_mean => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_geometric_mean"),
        .group_cumulative_weighted_harmonic_mean => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_harmonic_mean"),
        .group_cumulative_weighted_logsumexp => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_logsumexp"),
        .group_cumulative_weighted_logmeanexp => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_logmeanexp"),
        .group_cumulative_weighted_range => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_range"),
        .group_cumulative_weighted_midrange => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_midrange"),
        .group_cumulative_weighted_range_coeff => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_range_coeff"),
        .group_cumulative_weighted_variance => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, shift.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_weighted_variance = .{ .names = names, .value_name = value_name, .weight_name = weight_name, .output_name = output_name } };
        },
        .group_cumulative_weighted_stddev => |shift| blk: {
            const names = try cloneNameList(allocator, shift.names);
            errdefer freeNameList(allocator, names);
            const value_name = try allocator.dupe(u8, shift.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, shift.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, shift.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_cumulative_weighted_stddev = .{ .names = names, .value_name = value_name, .weight_name = weight_name, .output_name = output_name } };
        },
        .group_cumulative_weighted_sem => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_sem"),
        .group_cumulative_weighted_cv => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_cv"),
        .group_cumulative_weighted_fano => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_fano"),
        .group_cumulative_weighted_skewness => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_skewness"),
        .group_cumulative_weighted_kurtosis => |shift| try cloneGroupWeightedShift(Self, allocator, shift, "group_cumulative_weighted_kurtosis"),
        .group_row_number => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_row_number = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_size => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_size = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_reverse_row_number => |row_count| blk: {
            const names = try cloneNameList(allocator, row_count.names);
            errdefer freeNameList(allocator, names);
            const output_name = try allocator.dupe(u8, row_count.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_reverse_row_number = .{
                .names = names,
                .output_name = output_name,
            } };
        },
        .group_by_count => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_count = .{
                .key_name = key_name,
                .output_name = output_name,
            } };
        },
        .group_by_count_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_count_on = .{
                .key_names = key_names,
                .output_name = output_name,
            } };
        },
        .group_by_rows => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            break :blk .{ .group_by_rows = .{
                .key_name = key_name,
                .start = group.start,
                .signed_start = group.signed_start,
                .use_signed_start = group.use_signed_start,
                .step = group.step,
                .n = group.n,
                .keep_tail = group.keep_tail,
            } };
        },
        .group_by_rows_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            break :blk .{ .group_by_rows_on = .{
                .key_names = key_names,
                .start = group.start,
                .signed_start = group.signed_start,
                .use_signed_start = group.use_signed_start,
                .step = group.step,
                .n = group.n,
                .keep_tail = group.keep_tail,
            } };
        },
        .group_by_sorted_rows => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            const sort_name = try allocator.dupe(u8, group.sort_name);
            errdefer allocator.free(sort_name);
            break :blk .{ .group_by_sorted_rows = .{
                .key_name = key_name,
                .sort_name = sort_name,
                .n = group.n,
                .options = group.options,
                .keep_bottom = group.keep_bottom,
            } };
        },
        .group_by_sorted_rows_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            const sort_name = try allocator.dupe(u8, group.sort_name);
            errdefer allocator.free(sort_name);
            break :blk .{ .group_by_sorted_rows_on = .{
                .key_names = key_names,
                .sort_name = sort_name,
                .n = group.n,
                .options = group.options,
                .keep_bottom = group.keep_bottom,
            } };
        },
        .group_by_sorted_rows_columns => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            const sort_names = try cloneNameList(allocator, group.sort_names);
            errdefer freeNameList(allocator, sort_names);
            const options = try allocator.dupe(options_mod.DeviceSortOptions, group.options);
            errdefer allocator.free(options);
            break :blk .{ .group_by_sorted_rows_columns = .{
                .key_name = key_name,
                .sort_names = sort_names,
                .n = group.n,
                .options = options,
                .keep_bottom = group.keep_bottom,
            } };
        },
        .group_by_sorted_rows_columns_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            const sort_names = try cloneNameList(allocator, group.sort_names);
            errdefer freeNameList(allocator, sort_names);
            const options = try allocator.dupe(options_mod.DeviceSortOptions, group.options);
            errdefer allocator.free(options);
            break :blk .{ .group_by_sorted_rows_columns_on = .{
                .key_names = key_names,
                .sort_names = sort_names,
                .n = group.n,
                .options = options,
                .keep_bottom = group.keep_bottom,
            } };
        },
        .group_by_value => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            const value_name = try allocator.dupe(u8, group.value_name);
            errdefer allocator.free(value_name);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_value = .{
                .key_name = key_name,
                .value_name = value_name,
                .output_name = output_name,
                .aggregation = group.aggregation,
                .quantile = group.quantile,
                .index = group.index,
            } };
        },
        .group_by_value_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            const value_name = try allocator.dupe(u8, group.value_name);
            errdefer allocator.free(value_name);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_value_on = .{
                .key_names = key_names,
                .value_name = value_name,
                .output_name = output_name,
                .aggregation = group.aggregation,
                .quantile = group.quantile,
                .index = group.index,
            } };
        },
        .group_by_weighted => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            const value_name = try allocator.dupe(u8, group.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, group.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_weighted = .{
                .key_name = key_name,
                .value_name = value_name,
                .weight_name = weight_name,
                .output_name = output_name,
                .aggregation = group.aggregation,
                .quantile = group.quantile,
            } };
        },
        .group_by_weighted_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            const value_name = try allocator.dupe(u8, group.value_name);
            errdefer allocator.free(value_name);
            const weight_name = try allocator.dupe(u8, group.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_weighted_on = .{
                .key_names = key_names,
                .value_name = value_name,
                .weight_name = weight_name,
                .output_name = output_name,
                .aggregation = group.aggregation,
                .quantile = group.quantile,
            } };
        },
        .group_by_pair => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            const lhs_name = try allocator.dupe(u8, group.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, group.rhs_name);
            errdefer allocator.free(rhs_name);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_pair = .{
                .key_name = key_name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .output_name = output_name,
                .aggregation = group.aggregation,
            } };
        },
        .group_by_pair_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            const lhs_name = try allocator.dupe(u8, group.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, group.rhs_name);
            errdefer allocator.free(rhs_name);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_pair_on = .{
                .key_names = key_names,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .output_name = output_name,
                .aggregation = group.aggregation,
            } };
        },
        .group_by_weighted_pair => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            const lhs_name = try allocator.dupe(u8, group.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, group.rhs_name);
            errdefer allocator.free(rhs_name);
            const weight_name = try allocator.dupe(u8, group.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_weighted_pair = .{
                .key_name = key_name,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .weight_name = weight_name,
                .output_name = output_name,
                .aggregation = group.aggregation,
                .correction = group.correction,
            } };
        },
        .group_by_weighted_pair_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            const lhs_name = try allocator.dupe(u8, group.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, group.rhs_name);
            errdefer allocator.free(rhs_name);
            const weight_name = try allocator.dupe(u8, group.weight_name);
            errdefer allocator.free(weight_name);
            const output_name = try allocator.dupe(u8, group.output_name);
            errdefer allocator.free(output_name);
            break :blk .{ .group_by_weighted_pair_on = .{
                .key_names = key_names,
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .weight_name = weight_name,
                .output_name = output_name,
                .aggregation = group.aggregation,
                .correction = group.correction,
            } };
        },
        .group_by_stats => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            const value_name = try allocator.dupe(u8, group.value_name);
            errdefer allocator.free(value_name);
            const output_prefix = try allocator.dupe(u8, group.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .group_by_stats = .{
                .key_name = key_name,
                .value_name = value_name,
                .output_prefix = output_prefix,
            } };
        },
        .group_by_stats_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            const value_name = try allocator.dupe(u8, group.value_name);
            errdefer allocator.free(value_name);
            const output_prefix = try allocator.dupe(u8, group.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .group_by_stats_on = .{
                .key_names = key_names,
                .value_name = value_name,
                .output_prefix = output_prefix,
            } };
        },
        .group_by_profile => |group| blk: {
            const key_name = try allocator.dupe(u8, group.key_name);
            errdefer allocator.free(key_name);
            const value_name = try allocator.dupe(u8, group.value_name);
            errdefer allocator.free(value_name);
            const output_prefix = try allocator.dupe(u8, group.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .group_by_profile = .{
                .key_name = key_name,
                .value_name = value_name,
                .output_prefix = output_prefix,
            } };
        },
        .group_by_profile_on => |group| blk: {
            const key_names = try cloneNameList(allocator, group.key_names);
            errdefer freeNameList(allocator, key_names);
            const value_name = try allocator.dupe(u8, group.value_name);
            errdefer allocator.free(value_name);
            const output_prefix = try allocator.dupe(u8, group.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .group_by_profile_on = .{
                .key_names = key_names,
                .value_name = value_name,
                .output_prefix = output_prefix,
            } };
        },
        .join_on => |join| blk: {
            var right = try join.right.clone();
            errdefer right.deinit();
            const left_key_names = try cloneNameList(allocator, join.left_key_names);
            errdefer freeNameList(allocator, left_key_names);
            const right_key_names = try cloneNameList(allocator, join.right_key_names);
            errdefer freeNameList(allocator, right_key_names);
            const right_suffix = try allocator.dupe(u8, join.options.right_suffix);
            errdefer allocator.free(right_suffix);
            break :blk .{ .join_on = .{
                .kind = join.kind,
                .right = right,
                .left_key_names = left_key_names,
                .right_key_names = right_key_names,
                .options = .{ .right_suffix = right_suffix },
            } };
        },
        .asof_join => |join| blk: {
            var right = try join.right.clone();
            errdefer right.deinit();
            const left_key_name = try allocator.dupe(u8, join.left_key_name);
            errdefer allocator.free(left_key_name);
            const right_key_name = try allocator.dupe(u8, join.right_key_name);
            errdefer allocator.free(right_key_name);
            const right_suffix = try allocator.dupe(u8, join.options.right_suffix);
            errdefer allocator.free(right_suffix);
            break :blk .{ .asof_join = .{
                .right = right,
                .left_key_name = left_key_name,
                .right_key_name = right_key_name,
                .options = .{
                    .strategy = join.options.strategy,
                    .right_suffix = right_suffix,
                },
            } };
        },
        .concat_rows => |right| .{ .concat_rows = try right.clone() },
        .concat_columns => |right| .{ .concat_columns = try right.clone() },
        .distinct_rows => .{ .distinct_rows = {} },
        .distinct_rows_last => .{ .distinct_rows_last = {} },
        .distinct_rows_none => .{ .distinct_rows_none = {} },
        .distinct_on => |names| .{ .distinct_on = try cloneNameList(allocator, names) },
        .distinct_on_last => |names| .{ .distinct_on_last = try cloneNameList(allocator, names) },
        .distinct_on_none => |names| .{ .distinct_on_none = try cloneNameList(allocator, names) },
        .sort_by => |sort| .{ .sort_by = .{
            .name = try allocator.dupe(u8, sort.name),
            .options = sort.options,
        } },
        .sort_by_columns => |sort| blk: {
            const names = try cloneNameList(allocator, sort.names);
            errdefer freeNameList(allocator, names);
            const options = try allocator.dupe(std.meta.Elem(@TypeOf(sort.options)), sort.options);
            errdefer allocator.free(options);
            break :blk .{ .sort_by_columns = .{
                .names = names,
                .options = options,
            } };
        },
        .top_k => |top| .{ .top_k = .{
            .name = try allocator.dupe(u8, top.name),
            .options = top.options,
            .k = top.k,
        } },
        .top_k_columns => |top| blk: {
            const names = try cloneNameList(allocator, top.names);
            errdefer freeNameList(allocator, names);
            const options = try allocator.dupe(std.meta.Elem(@TypeOf(top.options)), top.options);
            errdefer allocator.free(options);
            break :blk .{ .top_k_columns = .{
                .names = names,
                .options = options,
                .k = top.k,
            } };
        },
        .rank_profile_by => |rank| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rank_profile_by", rank),
        .rolling_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_profile", profile),
        .rolling_moment_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_moment_profile", profile),
        .rolling_range_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_range_profile", profile),
        .rolling_normalize_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_normalize_profile", profile),
        .expanding_normalize_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "expanding_normalize_profile", profile),
        .rolling_quantile_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_quantile_profile", profile),
        .expanding_quantile_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "expanding_quantile_profile", profile),
        .rolling_bool_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_bool_profile", profile),
        .rolling_drawdown_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_drawdown_profile", profile),
        .rolling_robust_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_robust_profile", profile),
        .rolling_rank_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_rank_profile", profile),
        .lag_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "lag_profile", profile),
        .lead_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "lead_profile", profile),
        .clip_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "clip_profile", profile),
        .rolling_clip_profile => |profile| try clone_profile_mod.cloneNameOutputExtraOptions(Self, allocator, "rolling_clip_profile", profile, "clip_options"),
        .expanding_clip_profile => |profile| try clone_profile_mod.cloneNameOutputExtraOptions(Self, allocator, "expanding_clip_profile", profile, "clip_options"),
        .threshold_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "threshold_profile", profile),
        .rolling_threshold_profile => |profile| try clone_profile_mod.cloneNameOutputThresholdOptions(Self, allocator, "rolling_threshold_profile", profile),
        .expanding_threshold_profile => |profile| try clone_profile_mod.cloneNameOutputThresholdOptions(Self, allocator, "expanding_threshold_profile", profile),
        .expanding_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "expanding_profile", profile),
        .expanding_bool_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "expanding_bool_profile", profile),
        .expanding_rank_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "expanding_rank_profile", profile),
        .expanding_robust_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "expanding_robust_profile", profile),
        .expanding_moment_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "expanding_moment_profile", profile),
        .standardize_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "standardize_profile", profile),
        .robust_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "robust_profile", profile),
        .drawdown_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "drawdown_profile", profile),
        .extrema_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "extrema_profile", profile),
        .trend_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "trend_profile", profile),
        .rolling_trend_profile => |profile| try clone_profile_mod.cloneNameOutputExtraOptions(Self, allocator, "rolling_trend_profile", profile, "trend_options"),
        .expanding_trend_profile => |profile| try clone_profile_mod.cloneNameOutputExtraOptions(Self, allocator, "expanding_trend_profile", profile, "trend_options"),
        .change_point_profile => |profile| try clone_profile_mod.cloneNameOutputThresholdOptions(Self, allocator, "change_point_profile", profile),
        .rolling_change_point_profile => |profile| try clone_profile_mod.cloneNameOutputThresholdExtraOptions(Self, allocator, "rolling_change_point_profile", profile, "change_options"),
        .expanding_change_point_profile => |profile| try clone_profile_mod.cloneNameOutputThresholdExtraOptions(Self, allocator, "expanding_change_point_profile", profile, "change_options"),
        .sign_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "sign_profile", profile),
        .rolling_sign_profile => |profile| try clone_profile_mod.cloneNameOutputExtraOptions(Self, allocator, "rolling_sign_profile", profile, "sign_options"),
        .expanding_sign_profile => |profile| try clone_profile_mod.cloneNameOutputExtraOptions(Self, allocator, "expanding_sign_profile", profile, "sign_options"),
        .crossover_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "crossover_profile", profile, "lhs_name", "rhs_name"),
        .rolling_crossover_profile => |profile| try clone_profile_mod.clonePairOutputExtraOptions(Self, allocator, "rolling_crossover_profile", profile, "lhs_name", "rhs_name", "cross_options"),
        .expanding_crossover_profile => |profile| try clone_profile_mod.clonePairOutputExtraOptions(Self, allocator, "expanding_crossover_profile", profile, "lhs_name", "rhs_name", "cross_options"),
        .bucket_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "bucket_profile", profile),
        .ema_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "ema_profile", profile),
        .linear_fit_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "linear_fit_profile", profile, "x_name", "y_name"),
        .error_profile => |profile| try clone_profile_mod.clonePairOutput(Self, allocator, "error_profile", profile, "actual_name", "predicted_name"),
        .rolling_error_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "rolling_error_profile", profile, "actual_name", "predicted_name"),
        .expanding_error_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "expanding_error_profile", profile, "actual_name", "predicted_name"),
        .classification_profile => |profile| try clone_profile_mod.clonePairOutput(Self, allocator, "classification_profile", profile, "actual_name", "predicted_name"),
        .rolling_classification_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "rolling_classification_profile", profile, "actual_name", "predicted_name"),
        .expanding_classification_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "expanding_classification_profile", profile, "actual_name", "predicted_name"),
        .bool_transition_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "bool_transition_profile", profile),
        .rolling_bool_transition_profile => |profile| try clone_profile_mod.cloneNameOutputExtraOptions(Self, allocator, "rolling_bool_transition_profile", profile, "transition_options"),
        .expanding_bool_transition_profile => |profile| try clone_profile_mod.cloneNameOutputExtraOptions(Self, allocator, "expanding_bool_transition_profile", profile, "transition_options"),
        .rolling_correlation_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "rolling_correlation_profile", profile, "x_name", "y_name"),
        .expanding_correlation_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "expanding_correlation_profile", profile, "x_name", "y_name"),
        .expanding_linear_fit_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "expanding_linear_fit_profile", profile, "x_name", "y_name"),
        .rolling_linear_fit_profile => |profile| try clone_profile_mod.clonePairOutputOptions(Self, allocator, "rolling_linear_fit_profile", profile, "x_name", "y_name"),
        .validity_profile => |profile| try clone_profile_mod.cloneNameOutput(Self, allocator, "validity_profile", profile),
        .rolling_validity_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "rolling_validity_profile", profile),
        .expanding_validity_profile => |profile| try clone_profile_mod.cloneNameOutputOptions(Self, allocator, "expanding_validity_profile", profile),
        .slice_rows => |slice| .{ .slice_rows = slice },
        .slice_rows_signed => |slice| .{ .slice_rows_signed = slice },
        .drop_rows => |row_indices| .{ .drop_rows = try allocator.dupe(usize, row_indices) },
        .drop_rows_mode => |drop_mode| blk: {
            const row_indices = try allocator.dupe(usize, drop_mode.row_indices);
            errdefer allocator.free(row_indices);
            break :blk .{ .drop_rows_mode = .{
                .row_indices = row_indices,
                .mode = drop_mode.mode,
            } };
        },
        .drop_rows_signed => |row_indices| .{ .drop_rows_signed = try allocator.dupe(isize, row_indices) },
        .drop_rows_signed_mode => |drop_mode| blk: {
            const row_indices = try allocator.dupe(isize, drop_mode.row_indices);
            errdefer allocator.free(row_indices);
            break :blk .{ .drop_rows_signed_mode = .{
                .row_indices = row_indices,
                .mode = drop_mode.mode,
            } };
        },
        .drop_row_range => |range| .{ .drop_row_range = range },
        .drop_last_rows => |n| .{ .drop_last_rows = n },
        .slice_rows_step => |slice| .{ .slice_rows_step = slice },
        .slice_rows_signed_step => |slice| .{ .slice_rows_signed_step = slice },
        .stride_rows => |stride| .{ .stride_rows = stride },
        .take_rows => |row_indices| .{ .take_rows = try allocator.dupe(usize, row_indices) },
        .take_rows_optional => |row_indices| .{ .take_rows_optional = try allocator.dupe(?usize, row_indices) },
        .take_rows_mode => |take_mode| blk: {
            const row_indices = try allocator.dupe(usize, take_mode.row_indices);
            errdefer allocator.free(row_indices);
            break :blk .{ .take_rows_mode = .{
                .row_indices = row_indices,
                .mode = take_mode.mode,
            } };
        },
        .take_rows_signed => |row_indices| .{ .take_rows_signed = try allocator.dupe(isize, row_indices) },
        .take_rows_signed_mode => |take_mode| blk: {
            const row_indices = try allocator.dupe(isize, take_mode.row_indices);
            errdefer allocator.free(row_indices);
            break :blk .{ .take_rows_signed_mode = .{
                .row_indices = row_indices,
                .mode = take_mode.mode,
            } };
        },
        .take_rows_by_column => |name| .{ .take_rows_by_column = try allocator.dupe(u8, name) },
        .take_rows_by_column_mode => |take_mode| blk: {
            const name = try allocator.dupe(u8, take_mode.name);
            errdefer allocator.free(name);
            break :blk .{ .take_rows_by_column_mode = .{
                .name = name,
                .mode = take_mode.mode,
            } };
        },
        .drop_rows_by_column => |name| .{ .drop_rows_by_column = try allocator.dupe(u8, name) },
        .drop_rows_by_column_mode => |take_mode| blk: {
            const name = try allocator.dupe(u8, take_mode.name);
            errdefer allocator.free(name);
            break :blk .{ .drop_rows_by_column_mode = .{
                .name = name,
                .mode = take_mode.mode,
            } };
        },
        .repeat_rows => |repeat_count| .{ .repeat_rows = repeat_count },
        .tile_rows => |tile_count| .{ .tile_rows = tile_count },
        .repeat_rows_by => |count_name| .{ .repeat_rows_by = try allocator.dupe(u8, count_name) },
        .sample_rows => |sample| .{ .sample_rows = sample },
        .sample_rows_fraction => |sample| .{ .sample_rows_fraction = sample },
        .sample_rows_with_replacement => |sample| .{ .sample_rows_with_replacement = sample },
        .sample_rows_fraction_with_replacement => |sample| .{ .sample_rows_fraction_with_replacement = sample },
        .roll_rows => |shift| .{ .roll_rows = shift },
        .shift_rows => |shift| .{ .shift_rows = shift },
        .reverse_rows => .{ .reverse_rows = {} },
        .head => |n| .{ .head = n },
        .tail => |n| .{ .tail = n },
    };
}
