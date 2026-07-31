//! Ownership helpers for DeviceLazyOp payloads.
//!
//! Kept out-of-line so the operation-tag union stays compact while the large
//! clone/deinit switch remains close to the lazy operation type contract.

const std = @import("std");
const clone_profile_mod = @import("dataframe_lazy_op_clone_profile.zig");
const deinit_mod = @import("dataframe_lazy_op_deinit.zig");
const array_mod = @import("array.zig");
const names_mod = @import("dataframe_names.zig");
const series_mod = @import("series.zig");

pub const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;

pub const deinit = deinit_mod.deinit;

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
        .drop_name_prefix => |pattern| .{ .drop_name_prefix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .drop_name_suffix => |pattern| .{ .drop_name_suffix = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
        .drop_name_contains => |pattern| .{ .drop_name_contains = .{ .pattern = try allocator.dupe(u8, pattern.pattern) } },
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
        .row_null_count, .row_valid_count, .row_null_ratio, .row_valid_ratio, .row_first_valid_index, .row_last_valid_index, .row_first_null_index, .row_last_null_index, .row_argmin, .row_argmax, .row_median, .row_iqr, .row_mad, .row_sum, .row_mean, .row_geometric_mean, .row_harmonic_mean, .row_skewness, .row_kurtosis, .row_prod, .row_min, .row_max, .row_ptp, .row_mean_abs, .row_rms, .row_l1_norm, .row_l2_norm, .row_true_count, .row_false_count, .row_any_true, .row_all_true, .row_any_false, .row_all_false, .row_first_true_index, .row_last_true_index, .row_first_false_index, .row_last_false_index, .row_true_ratio, .row_false_ratio, .row_nan_count, .row_nan_ratio, .row_inf_count, .row_inf_ratio, .row_positive_inf_count, .row_negative_inf_count, .row_positive_inf_ratio, .row_negative_inf_ratio, .row_zero_count, .row_zero_ratio, .row_positive_zero_count, .row_negative_zero_count, .row_positive_zero_ratio, .row_negative_zero_ratio, .row_non_zero_count, .row_non_zero_ratio, .row_positive_count, .row_positive_ratio, .row_signbit_count, .row_signbit_ratio, .row_negative_count, .row_negative_ratio, .row_finite_count, .row_finite_ratio, .row_normal_count, .row_normal_ratio, .row_subnormal_count, .row_subnormal_ratio, .row_non_finite_count, .row_non_finite_ratio => |row_count, tag| blk: {
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
                .row_mad => .{ .row_mad = .{
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
                .row_geometric_mean => .{ .row_geometric_mean = .{
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
                .row_kurtosis => .{ .row_kurtosis = .{
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
                .row_mean_abs => .{ .row_mean_abs = .{
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
        } },
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
        .distinct_rows => .{ .distinct_rows = {} },
        .distinct_on => |names| .{ .distinct_on = try cloneNameList(allocator, names) },
        .sort_by => |sort| .{ .sort_by = .{
            .name = try allocator.dupe(u8, sort.name),
            .options = sort.options,
        } },
        .top_k => |top| .{ .top_k = .{
            .name = try allocator.dupe(u8, top.name),
            .options = top.options,
            .k = top.k,
        } },
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
        .sample_rows_with_replacement => |sample| .{ .sample_rows_with_replacement = sample },
        .roll_rows => |shift| .{ .roll_rows = shift },
        .shift_rows => |shift| .{ .shift_rows = shift },
        .reverse_rows => .{ .reverse_rows = {} },
        .head => |n| .{ .head = n },
        .tail => |n| .{ .tail = n },
    };
}
