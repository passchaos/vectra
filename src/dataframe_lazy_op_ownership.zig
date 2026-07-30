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
        .row_null_count, .row_valid_count => |row_count, tag| blk: {
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
                else => unreachable,
            };
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
        .drop_rows => |row_indices| .{ .drop_rows = try allocator.dupe(usize, row_indices) },
        .drop_row_range => |range| .{ .drop_row_range = range },
        .drop_last_rows => |n| .{ .drop_last_rows = n },
        .slice_rows_step => |slice| .{ .slice_rows_step = slice },
        .stride_rows => |stride| .{ .stride_rows = stride },
        .take_rows => |row_indices| .{ .take_rows = try allocator.dupe(usize, row_indices) },
        .sample_rows => |sample| .{ .sample_rows = sample },
        .sample_rows_with_replacement => |sample| .{ .sample_rows_with_replacement = sample },
        .reverse_rows => .{ .reverse_rows = {} },
        .head => |n| .{ .head = n },
        .tail => |n| .{ .tail = n },
    };
}
