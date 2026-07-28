//! Ownership helpers for DeviceLazyOp payloads.
//!
//! Kept out-of-line so the operation-tag union stays compact while the large
//! clone/deinit switch remains close to the lazy operation type contract.

const std = @import("std");
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
            break :blk .{ .select = owned };
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
        .rank_profile_by => |rank| blk: {
            const name = try allocator.dupe(u8, rank.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rank.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rank_profile_by = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rank.options,
            } };
        },
        .rolling_profile => |rolling| blk: {
            const name = try allocator.dupe(u8, rolling.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rolling.options,
            } };
        },
        .rolling_moment_profile => |rolling| blk: {
            const name = try allocator.dupe(u8, rolling.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_moment_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rolling.options,
            } };
        },
        .rolling_range_profile => |rolling| blk: {
            const name = try allocator.dupe(u8, rolling.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_range_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rolling.options,
            } };
        },
        .rolling_normalize_profile => |rolling| blk: {
            const name = try allocator.dupe(u8, rolling.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_normalize_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rolling.options,
            } };
        },
        .expanding_normalize_profile => |expanding| blk: {
            const name = try allocator.dupe(u8, expanding.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, expanding.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_normalize_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = expanding.options,
            } };
        },
        .rolling_quantile_profile => |rolling| blk: {
            const name = try allocator.dupe(u8, rolling.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_quantile_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rolling.options,
            } };
        },
        .expanding_quantile_profile => |expanding| blk: {
            const name = try allocator.dupe(u8, expanding.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, expanding.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_quantile_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = expanding.options,
            } };
        },
        .rolling_bool_profile => |rolling| blk: {
            const name = try allocator.dupe(u8, rolling.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_bool_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rolling.options,
            } };
        },
        .rolling_drawdown_profile => |rolling| blk: {
            const name = try allocator.dupe(u8, rolling.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_drawdown_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rolling.options,
            } };
        },
        .rolling_robust_profile => |rolling| blk: {
            const name = try allocator.dupe(u8, rolling.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_robust_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rolling.options,
            } };
        },
        .rolling_rank_profile => |rolling| blk: {
            const name = try allocator.dupe(u8, rolling.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, rolling.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_rank_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = rolling.options,
            } };
        },
        .lag_profile => |lag| blk: {
            const name = try allocator.dupe(u8, lag.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, lag.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .lag_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = lag.options,
            } };
        },
        .lead_profile => |lead| blk: {
            const name = try allocator.dupe(u8, lead.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, lead.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .lead_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = lead.options,
            } };
        },
        .clip_profile => |clip| blk: {
            const name = try allocator.dupe(u8, clip.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, clip.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .clip_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = clip.options,
            } };
        },
        .rolling_clip_profile => |clip| blk: {
            const name = try allocator.dupe(u8, clip.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, clip.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_clip_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .clip_options = clip.clip_options,
                .options = clip.options,
            } };
        },
        .expanding_clip_profile => |clip| blk: {
            const name = try allocator.dupe(u8, clip.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, clip.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_clip_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .clip_options = clip.clip_options,
                .options = clip.options,
            } };
        },
        .threshold_profile => |threshold| blk: {
            const name = try allocator.dupe(u8, threshold.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, threshold.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .threshold_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = threshold.options,
            } };
        },
        .rolling_threshold_profile => |threshold| blk: {
            const name = try allocator.dupe(u8, threshold.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, threshold.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_threshold_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .threshold = threshold.threshold,
                .options = threshold.options,
            } };
        },
        .expanding_threshold_profile => |threshold| blk: {
            const name = try allocator.dupe(u8, threshold.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, threshold.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_threshold_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .threshold = threshold.threshold,
                .options = threshold.options,
            } };
        },
        .expanding_profile => |expanding| blk: {
            const name = try allocator.dupe(u8, expanding.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, expanding.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = expanding.options,
            } };
        },
        .expanding_bool_profile => |expanding| blk: {
            const name = try allocator.dupe(u8, expanding.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, expanding.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_bool_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = expanding.options,
            } };
        },
        .expanding_rank_profile => |expanding| blk: {
            const name = try allocator.dupe(u8, expanding.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, expanding.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_rank_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = expanding.options,
            } };
        },
        .expanding_robust_profile => |expanding| blk: {
            const name = try allocator.dupe(u8, expanding.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, expanding.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_robust_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = expanding.options,
            } };
        },
        .expanding_moment_profile => |expanding| blk: {
            const name = try allocator.dupe(u8, expanding.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, expanding.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_moment_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = expanding.options,
            } };
        },
        .standardize_profile => |standardize| blk: {
            const name = try allocator.dupe(u8, standardize.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, standardize.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .standardize_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = standardize.options,
            } };
        },
        .robust_profile => |robust| blk: {
            const name = try allocator.dupe(u8, robust.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, robust.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .robust_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = robust.options,
            } };
        },
        .drawdown_profile => |drawdown| blk: {
            const name = try allocator.dupe(u8, drawdown.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, drawdown.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .drawdown_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = drawdown.options,
            } };
        },
        .extrema_profile => |extrema| blk: {
            const name = try allocator.dupe(u8, extrema.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, extrema.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .extrema_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = extrema.options,
            } };
        },
        .trend_profile => |trend| blk: {
            const name = try allocator.dupe(u8, trend.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, trend.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .trend_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = trend.options,
            } };
        },
        .rolling_trend_profile => |trend| blk: {
            const name = try allocator.dupe(u8, trend.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, trend.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_trend_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .trend_options = trend.trend_options,
                .options = trend.options,
            } };
        },
        .expanding_trend_profile => |trend| blk: {
            const name = try allocator.dupe(u8, trend.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, trend.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_trend_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .trend_options = trend.trend_options,
                .options = trend.options,
            } };
        },
        .change_point_profile => |change| blk: {
            const name = try allocator.dupe(u8, change.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, change.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .change_point_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .threshold = change.threshold,
                .options = change.options,
            } };
        },
        .rolling_change_point_profile => |change| blk: {
            const name = try allocator.dupe(u8, change.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, change.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_change_point_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .threshold = change.threshold,
                .change_options = change.change_options,
                .options = change.options,
            } };
        },
        .expanding_change_point_profile => |change| blk: {
            const name = try allocator.dupe(u8, change.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, change.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_change_point_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .threshold = change.threshold,
                .change_options = change.change_options,
                .options = change.options,
            } };
        },
        .sign_profile => |sign| blk: {
            const name = try allocator.dupe(u8, sign.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, sign.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .sign_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = sign.options,
            } };
        },
        .rolling_sign_profile => |sign| blk: {
            const name = try allocator.dupe(u8, sign.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, sign.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_sign_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .sign_options = sign.sign_options,
                .options = sign.options,
            } };
        },
        .expanding_sign_profile => |sign| blk: {
            const name = try allocator.dupe(u8, sign.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, sign.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_sign_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .sign_options = sign.sign_options,
                .options = sign.options,
            } };
        },
        .crossover_profile => |cross| blk: {
            const lhs_name = try allocator.dupe(u8, cross.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, cross.rhs_name);
            errdefer allocator.free(rhs_name);
            const output_prefix = try allocator.dupe(u8, cross.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .crossover_profile = .{
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .output_prefix = output_prefix,
                .options = cross.options,
            } };
        },
        .rolling_crossover_profile => |cross| blk: {
            const lhs_name = try allocator.dupe(u8, cross.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, cross.rhs_name);
            errdefer allocator.free(rhs_name);
            const output_prefix = try allocator.dupe(u8, cross.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_crossover_profile = .{
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .output_prefix = output_prefix,
                .cross_options = cross.cross_options,
                .options = cross.options,
            } };
        },
        .expanding_crossover_profile => |cross| blk: {
            const lhs_name = try allocator.dupe(u8, cross.lhs_name);
            errdefer allocator.free(lhs_name);
            const rhs_name = try allocator.dupe(u8, cross.rhs_name);
            errdefer allocator.free(rhs_name);
            const output_prefix = try allocator.dupe(u8, cross.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_crossover_profile = .{
                .lhs_name = lhs_name,
                .rhs_name = rhs_name,
                .output_prefix = output_prefix,
                .cross_options = cross.cross_options,
                .options = cross.options,
            } };
        },
        .bucket_profile => |bucket| blk: {
            const name = try allocator.dupe(u8, bucket.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, bucket.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .bucket_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = bucket.options,
            } };
        },
        .ema_profile => |ema| blk: {
            const name = try allocator.dupe(u8, ema.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, ema.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .ema_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = ema.options,
            } };
        },
        .linear_fit_profile => |fit| blk: {
            const x_name = try allocator.dupe(u8, fit.x_name);
            errdefer allocator.free(x_name);
            const y_name = try allocator.dupe(u8, fit.y_name);
            errdefer allocator.free(y_name);
            const output_prefix = try allocator.dupe(u8, fit.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .linear_fit_profile = .{
                .x_name = x_name,
                .y_name = y_name,
                .output_prefix = output_prefix,
                .options = fit.options,
            } };
        },
        .error_profile => |err| blk: {
            const actual_name = try allocator.dupe(u8, err.actual_name);
            errdefer allocator.free(actual_name);
            const predicted_name = try allocator.dupe(u8, err.predicted_name);
            errdefer allocator.free(predicted_name);
            const output_prefix = try allocator.dupe(u8, err.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .error_profile = .{
                .actual_name = actual_name,
                .predicted_name = predicted_name,
                .output_prefix = output_prefix,
            } };
        },
        .rolling_error_profile => |err| blk: {
            const actual_name = try allocator.dupe(u8, err.actual_name);
            errdefer allocator.free(actual_name);
            const predicted_name = try allocator.dupe(u8, err.predicted_name);
            errdefer allocator.free(predicted_name);
            const output_prefix = try allocator.dupe(u8, err.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_error_profile = .{
                .actual_name = actual_name,
                .predicted_name = predicted_name,
                .output_prefix = output_prefix,
                .options = err.options,
            } };
        },
        .expanding_error_profile => |err| blk: {
            const actual_name = try allocator.dupe(u8, err.actual_name);
            errdefer allocator.free(actual_name);
            const predicted_name = try allocator.dupe(u8, err.predicted_name);
            errdefer allocator.free(predicted_name);
            const output_prefix = try allocator.dupe(u8, err.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_error_profile = .{
                .actual_name = actual_name,
                .predicted_name = predicted_name,
                .output_prefix = output_prefix,
                .options = err.options,
            } };
        },
        .classification_profile => |class| blk: {
            const actual_name = try allocator.dupe(u8, class.actual_name);
            errdefer allocator.free(actual_name);
            const predicted_name = try allocator.dupe(u8, class.predicted_name);
            errdefer allocator.free(predicted_name);
            const output_prefix = try allocator.dupe(u8, class.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .classification_profile = .{
                .actual_name = actual_name,
                .predicted_name = predicted_name,
                .output_prefix = output_prefix,
            } };
        },
        .rolling_classification_profile => |class| blk: {
            const actual_name = try allocator.dupe(u8, class.actual_name);
            errdefer allocator.free(actual_name);
            const predicted_name = try allocator.dupe(u8, class.predicted_name);
            errdefer allocator.free(predicted_name);
            const output_prefix = try allocator.dupe(u8, class.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_classification_profile = .{
                .actual_name = actual_name,
                .predicted_name = predicted_name,
                .output_prefix = output_prefix,
                .options = class.options,
            } };
        },
        .expanding_classification_profile => |class| blk: {
            const actual_name = try allocator.dupe(u8, class.actual_name);
            errdefer allocator.free(actual_name);
            const predicted_name = try allocator.dupe(u8, class.predicted_name);
            errdefer allocator.free(predicted_name);
            const output_prefix = try allocator.dupe(u8, class.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_classification_profile = .{
                .actual_name = actual_name,
                .predicted_name = predicted_name,
                .output_prefix = output_prefix,
                .options = class.options,
            } };
        },
        .bool_transition_profile => |transition| blk: {
            const name = try allocator.dupe(u8, transition.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, transition.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .bool_transition_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = transition.options,
            } };
        },
        .rolling_bool_transition_profile => |transition| blk: {
            const name = try allocator.dupe(u8, transition.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, transition.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_bool_transition_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .transition_options = transition.transition_options,
                .options = transition.options,
            } };
        },
        .expanding_bool_transition_profile => |transition| blk: {
            const name = try allocator.dupe(u8, transition.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, transition.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_bool_transition_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .transition_options = transition.transition_options,
                .options = transition.options,
            } };
        },
        .rolling_correlation_profile => |corr| blk: {
            const x_name = try allocator.dupe(u8, corr.x_name);
            errdefer allocator.free(x_name);
            const y_name = try allocator.dupe(u8, corr.y_name);
            errdefer allocator.free(y_name);
            const output_prefix = try allocator.dupe(u8, corr.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_correlation_profile = .{
                .x_name = x_name,
                .y_name = y_name,
                .output_prefix = output_prefix,
                .options = corr.options,
            } };
        },
        .expanding_correlation_profile => |corr| blk: {
            const x_name = try allocator.dupe(u8, corr.x_name);
            errdefer allocator.free(x_name);
            const y_name = try allocator.dupe(u8, corr.y_name);
            errdefer allocator.free(y_name);
            const output_prefix = try allocator.dupe(u8, corr.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_correlation_profile = .{
                .x_name = x_name,
                .y_name = y_name,
                .output_prefix = output_prefix,
                .options = corr.options,
            } };
        },
        .expanding_linear_fit_profile => |fit| blk: {
            const x_name = try allocator.dupe(u8, fit.x_name);
            errdefer allocator.free(x_name);
            const y_name = try allocator.dupe(u8, fit.y_name);
            errdefer allocator.free(y_name);
            const output_prefix = try allocator.dupe(u8, fit.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_linear_fit_profile = .{
                .x_name = x_name,
                .y_name = y_name,
                .output_prefix = output_prefix,
                .options = fit.options,
            } };
        },
        .rolling_linear_fit_profile => |fit| blk: {
            const x_name = try allocator.dupe(u8, fit.x_name);
            errdefer allocator.free(x_name);
            const y_name = try allocator.dupe(u8, fit.y_name);
            errdefer allocator.free(y_name);
            const output_prefix = try allocator.dupe(u8, fit.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_linear_fit_profile = .{
                .x_name = x_name,
                .y_name = y_name,
                .output_prefix = output_prefix,
                .options = fit.options,
            } };
        },
        .validity_profile => |validity| blk: {
            const name = try allocator.dupe(u8, validity.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, validity.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .validity_profile = .{
                .name = name,
                .output_prefix = output_prefix,
            } };
        },
        .rolling_validity_profile => |validity| blk: {
            const name = try allocator.dupe(u8, validity.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, validity.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .rolling_validity_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = validity.options,
            } };
        },
        .expanding_validity_profile => |validity| blk: {
            const name = try allocator.dupe(u8, validity.name);
            errdefer allocator.free(name);
            const output_prefix = try allocator.dupe(u8, validity.output_prefix);
            errdefer allocator.free(output_prefix);
            break :blk .{ .expanding_validity_profile = .{
                .name = name,
                .output_prefix = output_prefix,
                .options = validity.options,
            } };
        },
        .head => |n| .{ .head = n },
        .tail => |n| .{ .tail = n },
    };
}
