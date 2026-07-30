//! Execution and optimization helpers for DeviceLazyFrame.
//!
//! This module owns collect/explain/optimization so dataframe_lazy_frame.zig
//! can focus on public lazy-plan construction methods.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_arrow_mod = @import("dataframe_arrow.zig");
const lazy_mod = @import("dataframe_lazy.zig");
const names_mod = @import("dataframe_names.zig");
const series_mod = @import("series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
pub const ParquetInteropError = dataframe_arrow_mod.ParquetInteropError;
const allNamesIn = names_mod.allNamesIn;
const planLazyScanPushdown = lazy_mod.planLazyScanPushdown;
const formatLazyScanPushdown = lazy_mod.formatLazyScanPushdown;
const formatLazyOp = lazy_mod.formatLazyOp;

fn deinitLazyOps(comptime DeviceLazyOp: type, allocator: std.mem.Allocator, ops: *std.ArrayList(DeviceLazyOp)) void {
    for (ops.items) |*op| op.deinit(allocator);
    ops.deinit(allocator);
}

pub fn collect(comptime DeviceDataFrame: type, comptime DeviceLazyOp: type, self: anytype) ParquetInteropError!DeviceDataFrame {
    var optimized = try optimizedOps(DeviceLazyOp, self);
    defer deinitLazyOps(DeviceLazyOp, self.allocator, &optimized);
    var current = try collectSource(DeviceDataFrame, DeviceLazyOp, self, optimized.items);
    errdefer current.deinit();
    for (optimized.items) |op| {
        const next = switch (op) {
            .select => |names| try current.select(names),
            .select_column_indices => |indices| try current.selectByColumnIndices(indices),
            .select_column_range => |range| try current.selectColumnRange(range.start, range.stop),
            .select_last_columns => |n| try current.selectLastColumns(n),
            .drop_column_indices => |indices| try current.dropByColumnIndices(indices),
            .drop_column_range => |range| try current.dropColumnRange(range.start, range.stop),
            .drop_last_columns => |n| try current.dropLastColumns(n),
            .select_name_prefix => |pattern| try current.selectByNamePrefix(pattern.pattern),
            .select_name_suffix => |pattern| try current.selectByNameSuffix(pattern.pattern),
            .select_name_contains => |pattern| try current.selectByNameContains(pattern.pattern),
            .drop_name_prefix => |pattern| try current.dropByNamePrefix(pattern.pattern),
            .drop_name_suffix => |pattern| try current.dropByNameSuffix(pattern.pattern),
            .drop_name_contains => |pattern| try current.dropByNameContains(pattern.pattern),
            .select_dtypes => |dtypes| try current.selectByDTypes(dtypes),
            .select_dtype_class => |class| try current.selectByDTypeClass(class),
            .drop_dtypes => |dtypes| try current.dropByDTypes(dtypes),
            .drop_dtype_class => |class| try current.dropByDTypeClass(class),
            .select_nullable_columns => try current.selectNullableColumns(),
            .select_non_nullable_columns => try current.selectNonNullableColumns(),
            .select_columns_with_nulls => try current.selectColumnsWithNulls(),
            .select_columns_without_nulls => try current.selectColumnsWithoutNulls(),
            .drop_nullable_columns => try current.dropNullableColumns(),
            .drop_non_nullable_columns => try current.dropNonNullableColumns(),
            .drop_columns_with_nulls => try current.dropColumnsWithNulls(),
            .drop_columns_without_nulls => try current.dropColumnsWithoutNulls(),
            .with_row_index => |row_index| try current.withRowIndex(row_index.name, row_index.offset),
            .rename_column => |rename| try current.renameColumn(rename.old_name, rename.new_name),
            .rename_columns => |rename| try current.renameColumns(rename.old_names, rename.new_names),
            .add_column_name_prefix => |pattern| try current.addColumnNamePrefix(pattern.pattern),
            .add_column_name_suffix => |pattern| try current.addColumnNameSuffix(pattern.pattern),
            .move_column => |move| try current.moveColumn(move.name, move.target_index),
            .move_column_before => |move| try current.moveColumnBefore(move.name, move.anchor_name),
            .move_column_after => |move| try current.moveColumnAfter(move.name, move.anchor_name),
            .copy_column => |copy| try current.copyColumn(copy.source_name, copy.new_name),
            .copy_column_at => |copy| try current.copyColumnAt(copy.source_name, copy.new_name, copy.target_index),
            .copy_column_before => |copy| try current.copyColumnBefore(copy.source_name, copy.new_name, copy.anchor_name),
            .copy_column_after => |copy| try current.copyColumnAfter(copy.source_name, copy.new_name, copy.anchor_name),
            .drop_columns => |names| try current.dropColumns(names),
            .drop_nulls => |names| try current.dropNulls(names),
            .filter_nulls_column => |name| try current.filterNullsColumn(name),
            .with_column_binary => |expr| blk: {
                var column_value = try current.binaryColumns(expr.lhs_name, expr.rhs_name, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_scalar => |expr| blk: {
                var column_value = try current.binaryColumnScalarWithDeviceScalar(expr.input_name, expr.scalar, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_literal => |expr| try current.withColumnLiteralScalar(expr.name, expr.scalar),
            .with_column_literal_at => |expr| try current.withColumnLiteralScalarAt(expr.name, expr.scalar, expr.target_index),
            .with_column_literal_before => |expr| try current.withColumnLiteralScalarBefore(expr.name, expr.scalar, expr.anchor_name),
            .with_column_literal_after => |expr| try current.withColumnLiteralScalarAfter(expr.name, expr.scalar, expr.anchor_name),
            .cast_column => |cast| try current.castColumn(cast.name, cast.dtype),
            .fill_null_column => |fill| try current.fillNullColumnWithScalar(fill.name, fill.scalar),
            .coalesce_columns => |coalesce| try current.coalesceColumns(coalesce.primary_name, coalesce.fallback_name, coalesce.output_name),
            .is_null_column => |predicate| try current.isNullColumn(predicate.name, predicate.output_name),
            .is_valid_column => |predicate| try current.isValidColumn(predicate.name, predicate.output_name),
            .row_null_count => |row_count| try current.withRowNullCount(row_count.names, row_count.output_name),
            .row_valid_count => |row_count| try current.withRowValidCount(row_count.names, row_count.output_name),
            .with_column_compare => |expr| blk: {
                var column_value = try current.compareColumns(expr.lhs_name, expr.rhs_name, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_compare_scalar => |expr| blk: {
                var column_value = try current.compareColumnScalarWithDeviceScalar(expr.input_name, expr.scalar, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .filter_mask => |mask| try current.filterColumnMask(mask),
            .filter_column => |name| try current.filterColumn(name),
            .filter_scalar => |filter_op| blk: {
                var mask = try current.compareColumnScalarWithDeviceScalar(filter_op.name, filter_op.scalar, filter_op.op);
                defer mask.deinit();
                break :blk try current.filterColumnMask(mask);
            },
            .group_by_count => |group| try current.groupByCount(group.key_name, group.output_name),
            .group_by_value => |group| switch (group.aggregation) {
                .sum => try current.groupBySum(group.key_name, group.value_name, group.output_name),
                .min => try current.groupByMin(group.key_name, group.value_name, group.output_name),
                .max => try current.groupByMax(group.key_name, group.value_name, group.output_name),
                .mean => try current.groupByMean(group.key_name, group.value_name, group.output_name),
            },
            .group_by_stats => |group| try current.groupByStats(group.key_name, group.value_name, group.output_prefix),
            .group_by_stats_on => |group| try current.groupByStatsOn(group.key_names, group.value_name, group.output_prefix),
            .group_by_profile => |group| try current.groupByProfile(group.key_name, group.value_name, group.output_prefix),
            .group_by_profile_on => |group| try current.groupByProfileOn(group.key_names, group.value_name, group.output_prefix),
            .join_on => |join| switch (join.kind) {
                .inner => try current.innerJoinOn(join.right, join.left_key_names, join.right_key_names, join.options),
                .left => try current.leftJoinOn(join.right, join.left_key_names, join.right_key_names, join.options),
                .full => try current.fullJoinOn(join.right, join.left_key_names, join.right_key_names, join.options),
                .semi => try current.semiJoinOn(join.right, join.left_key_names, join.right_key_names),
                .anti => try current.antiJoinOn(join.right, join.left_key_names, join.right_key_names),
            },
            .asof_join => |join| try current.asofJoin(join.right, join.left_key_name, join.right_key_name, join.options),
            .concat_rows => |right| try current.concatRows(right),
            .distinct_rows => try current.distinctRows(),
            .distinct_on => |names| try current.distinctOn(names),
            .sort_by => |sort| try current.sortBy(sort.name, sort.options),
            .top_k => |top| try current.topKBy(top.name, top.k, top.options),
            .rank_profile_by => |rank| try current.rankProfileBy(rank.name, rank.output_prefix, rank.options),
            .rolling_profile => |rolling| try current.rollingProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_moment_profile => |rolling| try current.rollingMomentProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_range_profile => |rolling| try current.rollingRangeProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_normalize_profile => |rolling| try current.rollingNormalizeProfile(rolling.name, rolling.output_prefix, rolling.options),
            .expanding_normalize_profile => |expanding| try current.expandingNormalizeProfile(expanding.name, expanding.output_prefix, expanding.options),
            .rolling_quantile_profile => |rolling| try current.rollingQuantileProfile(rolling.name, rolling.output_prefix, rolling.options),
            .expanding_quantile_profile => |expanding| try current.expandingQuantileProfile(expanding.name, expanding.output_prefix, expanding.options),
            .rolling_bool_profile => |rolling| try current.rollingBoolProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_drawdown_profile => |rolling| try current.rollingDrawdownProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_robust_profile => |rolling| try current.rollingRobustProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_rank_profile => |rolling| try current.rollingRankProfile(rolling.name, rolling.output_prefix, rolling.options),
            .lag_profile => |lag| try current.lagProfile(lag.name, lag.output_prefix, lag.options),
            .lead_profile => |lead| try current.leadProfile(lead.name, lead.output_prefix, lead.options),
            .clip_profile => |clip| try current.clipProfile(clip.name, clip.output_prefix, clip.options),
            .rolling_clip_profile => |clip| try current.rollingClipProfile(clip.name, clip.output_prefix, clip.clip_options, clip.options),
            .expanding_clip_profile => |clip| try current.expandingClipProfile(clip.name, clip.output_prefix, clip.clip_options, clip.options),
            .threshold_profile => |threshold| try current.thresholdProfile(threshold.name, threshold.output_prefix, threshold.options),
            .rolling_threshold_profile => |threshold| try current.rollingThresholdProfile(threshold.name, threshold.output_prefix, threshold.threshold, threshold.options),
            .expanding_threshold_profile => |threshold| try current.expandingThresholdProfile(threshold.name, threshold.output_prefix, threshold.threshold, threshold.options),
            .expanding_profile => |expanding| try current.expandingProfile(expanding.name, expanding.output_prefix, expanding.options),
            .expanding_bool_profile => |expanding| try current.expandingBoolProfile(expanding.name, expanding.output_prefix, expanding.options),
            .expanding_rank_profile => |expanding| try current.expandingRankProfile(expanding.name, expanding.output_prefix, expanding.options),
            .expanding_robust_profile => |expanding| try current.expandingRobustProfile(expanding.name, expanding.output_prefix, expanding.options),
            .expanding_moment_profile => |expanding| try current.expandingMomentProfile(expanding.name, expanding.output_prefix, expanding.options),
            .standardize_profile => |standardize| try current.standardizeProfile(standardize.name, standardize.output_prefix, standardize.options),
            .robust_profile => |robust| try current.robustProfile(robust.name, robust.output_prefix, robust.options),
            .drawdown_profile => |drawdown| try current.drawdownProfile(drawdown.name, drawdown.output_prefix, drawdown.options),
            .extrema_profile => |extrema| try current.extremaProfile(extrema.name, extrema.output_prefix, extrema.options),
            .trend_profile => |trend| try current.trendProfile(trend.name, trend.output_prefix, trend.options),
            .rolling_trend_profile => |trend| try current.rollingTrendProfile(trend.name, trend.output_prefix, trend.trend_options, trend.options),
            .expanding_trend_profile => |trend| try current.expandingTrendProfile(trend.name, trend.output_prefix, trend.trend_options, trend.options),
            .change_point_profile => |change| try current.changePointProfile(change.name, change.output_prefix, change.threshold, change.options),
            .rolling_change_point_profile => |change| try current.rollingChangePointProfile(change.name, change.output_prefix, change.threshold, change.change_options, change.options),
            .expanding_change_point_profile => |change| try current.expandingChangePointProfile(change.name, change.output_prefix, change.threshold, change.change_options, change.options),
            .sign_profile => |sign| try current.signProfile(sign.name, sign.output_prefix, sign.options),
            .rolling_sign_profile => |sign| try current.rollingSignProfile(sign.name, sign.output_prefix, sign.sign_options, sign.options),
            .expanding_sign_profile => |sign| try current.expandingSignProfile(sign.name, sign.output_prefix, sign.sign_options, sign.options),
            .crossover_profile => |cross| try current.crossoverProfile(cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.options),
            .rolling_crossover_profile => |cross| try current.rollingCrossoverProfile(cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.cross_options, cross.options),
            .expanding_crossover_profile => |cross| try current.expandingCrossoverProfile(cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.cross_options, cross.options),
            .bucket_profile => |bucket| try current.bucketProfile(bucket.name, bucket.output_prefix, bucket.options),
            .ema_profile => |ema| try current.emaProfile(ema.name, ema.output_prefix, ema.options),
            .linear_fit_profile => |fit| try current.linearFitProfile(fit.x_name, fit.y_name, fit.output_prefix, fit.options),
            .error_profile => |err| try current.errorProfile(err.actual_name, err.predicted_name, err.output_prefix),
            .rolling_error_profile => |err| try current.rollingErrorProfile(err.actual_name, err.predicted_name, err.output_prefix, err.options),
            .expanding_error_profile => |err| try current.expandingErrorProfile(err.actual_name, err.predicted_name, err.output_prefix, err.options),
            .classification_profile => |class| try current.classificationProfile(class.actual_name, class.predicted_name, class.output_prefix),
            .rolling_classification_profile => |class| try current.rollingClassificationProfile(class.actual_name, class.predicted_name, class.output_prefix, class.options),
            .expanding_classification_profile => |class| try current.expandingClassificationProfile(class.actual_name, class.predicted_name, class.output_prefix, class.options),
            .bool_transition_profile => |transition| try current.boolTransitionProfile(transition.name, transition.output_prefix, transition.options),
            .rolling_bool_transition_profile => |transition| try current.rollingBoolTransitionProfile(transition.name, transition.output_prefix, transition.transition_options, transition.options),
            .expanding_bool_transition_profile => |transition| try current.expandingBoolTransitionProfile(transition.name, transition.output_prefix, transition.transition_options, transition.options),
            .rolling_correlation_profile => |corr| try current.rollingCorrelationProfile(corr.x_name, corr.y_name, corr.output_prefix, corr.options),
            .expanding_correlation_profile => |corr| try current.expandingCorrelationProfile(corr.x_name, corr.y_name, corr.output_prefix, corr.options),
            .expanding_linear_fit_profile => |fit| try current.expandingLinearFitProfile(fit.x_name, fit.y_name, fit.output_prefix, fit.options),
            .rolling_linear_fit_profile => |fit| try current.rollingLinearFitProfile(fit.x_name, fit.y_name, fit.output_prefix, fit.options),
            .validity_profile => |validity| try current.validityProfile(validity.name, validity.output_prefix),
            .rolling_validity_profile => |validity| try current.rollingValidityProfile(validity.name, validity.output_prefix, validity.options),
            .expanding_validity_profile => |validity| try current.expandingValidityProfile(validity.name, validity.output_prefix, validity.options),
            .slice_rows => |slice| try current.sliceRows(slice.start, slice.stop),
            .drop_rows => |row_indices| try current.dropRows(row_indices),
            .drop_row_range => |range| try current.dropRowRange(range.start, range.stop),
            .drop_last_rows => |n| try current.dropLastRows(n),
            .slice_rows_step => |slice| try current.sliceRowsStep(slice.start, slice.stop, slice.step),
            .stride_rows => |stride| try current.strideRows(stride.start, stride.step),
            .take_rows => |row_indices| try current.take(row_indices),
            .sample_rows => |sample| try current.sampleRows(sample.count, sample.seed),
            .sample_rows_with_replacement => |sample| try current.sampleRowsWithReplacement(sample.count, sample.seed),
            .reverse_rows => try current.reverseRows(),
            .head => |n| try current.head(n),
            .tail => |n| try current.tail(n),
        };
        current.deinit();
        current = next;
    }
    return current;
}

pub fn explain(comptime DeviceLazyOp: type, self: anytype, allocator: std.mem.Allocator) DeviceDataError![]u8 {
    var optimized = try optimizedOps(DeviceLazyOp, self);
    defer deinitLazyOps(DeviceLazyOp, self.allocator, &optimized);
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    try aw.writer.print("DeviceLazyFrame(raw_ops={d}, optimized_ops={d}, source={s})\n", .{ self.ops.items.len, optimized.items.len, self.source.name() });
    if (self.source == .parquet_scan) {
        var pushdown = try planLazyScanPushdown(self.allocator, optimized.items);
        defer pushdown.deinit();
        try aw.writer.print("  scan_pushdown: ", .{});
        try formatLazyScanPushdown(&aw.writer, pushdown);
        try aw.writer.print("\n", .{});
    }
    for (optimized.items, 0..) |op, i| {
        try aw.writer.print("  {d}: ", .{i});
        try formatLazyOp(&aw.writer, op);
        try aw.writer.print("\n", .{});
    }
    return aw.toOwnedSlice();
}

fn optimizedOps(comptime DeviceLazyOp: type, self: anytype) DeviceDataError!std.ArrayList(DeviceLazyOp) {
    var optimized: std.ArrayList(DeviceLazyOp) = .empty;
    errdefer deinitLazyOps(DeviceLazyOp, self.allocator, &optimized);
    for (self.ops.items) |op| {
        switch (op) {
            .select => |names| {
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .select) {
                    const previous = optimized.items[optimized.items.len - 1].select;
                    if (allNamesIn(names, previous)) {
                        optimized.items[optimized.items.len - 1].deinit(self.allocator);
                        var cloned_op = try op.clone(self.allocator);
                        errdefer cloned_op.deinit(self.allocator);
                        optimized.items[optimized.items.len - 1] = cloned_op;
                        continue;
                    }
                }
            },
            .head => |n| {
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .sort_by) {
                    const sort = optimized.items[optimized.items.len - 1].sort_by;
                    const name = try self.allocator.dupe(u8, sort.name);
                    optimized.items[optimized.items.len - 1].deinit(self.allocator);
                    optimized.items[optimized.items.len - 1] = .{ .top_k = .{
                        .name = name,
                        .options = sort.options,
                        .k = n,
                    } };
                    continue;
                }
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .top_k) {
                    const top = optimized.items[optimized.items.len - 1].top_k;
                    optimized.items[optimized.items.len - 1] = .{ .top_k = .{
                        .name = top.name,
                        .options = top.options,
                        .k = @min(top.k, n),
                    } };
                    continue;
                }
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .head) {
                    const prev = optimized.items[optimized.items.len - 1].head;
                    optimized.items[optimized.items.len - 1] = .{ .head = @min(prev, n) };
                    continue;
                }
            },
            .tail => |n| {
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .tail) {
                    const prev = optimized.items[optimized.items.len - 1].tail;
                    optimized.items[optimized.items.len - 1] = .{ .tail = @min(prev, n) };
                    continue;
                }
            },
            else => {},
        }
        var cloned_op = try op.clone(self.allocator);
        errdefer cloned_op.deinit(self.allocator);
        try optimized.append(self.allocator, cloned_op);
    }
    return optimized;
}

fn collectSource(comptime DeviceDataFrame: type, comptime DeviceLazyOp: type, self: anytype, ops: []const DeviceLazyOp) ParquetInteropError!DeviceDataFrame {
    return switch (self.source) {
        .dataframe => |frame| try frame.clone(),
        .parquet_scan => |scan| blk: {
            var scan_plan = try scan.clone();
            defer scan_plan.deinit();

            var pushdown = try planLazyScanPushdown(self.allocator, ops);
            defer pushdown.deinit();
            if (pushdown.range_predicate) |predicate| {
                try scan_plan.whereRange(predicate.column, predicate.predicate);
            }
            if (pushdown.projection) |names| {
                try scan_plan.select(names);
            }

            break :blk try scan_plan.collect();
        },
    };
}
