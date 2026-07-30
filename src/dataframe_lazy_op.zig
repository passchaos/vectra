//! Lazy dataframe operation tags and ownership helpers.
//!
//! `DeviceLazyOp` carries many operation payload variants and therefore used to
//! dominate the public dataframe facade. Keeping the union generic over the
//! concrete dataframe/column types avoids an import cycle while moving the
//! clone/deinit ownership rules into a focused module.

const std = @import("std");
const array_mod = @import("array.zig");
const ownership_mod = @import("dataframe_lazy_op_ownership.zig");
const payloads_mod = @import("dataframe_lazy_op_payloads.zig");
const series_mod = @import("series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub const DeviceLazyGroupByAggregation = payloads_mod.DeviceLazyGroupByAggregation;
pub const DeviceLazyJoinKind = payloads_mod.DeviceLazyJoinKind;

pub fn DeviceLazyOp(comptime DeviceDataFrame: type, comptime DeviceColumn: type) type {
    const Payloads = payloads_mod.DeviceLazyPayloads(DeviceDataFrame, DeviceColumn);
    return union(enum) {
        const Self = @This();
        select: Payloads.Select,
        select_dtypes: Payloads.SelectDTypes,
        select_dtype_class: Payloads.SelectDTypeClass,
        with_row_index: Payloads.RowIndex,
        rename_column: Payloads.RenameColumn,
        drop_columns: Payloads.DropColumns,
        with_column_binary: Payloads.WithColumnBinary,
        with_column_scalar: Payloads.WithColumnScalar,
        with_column_compare: Payloads.WithColumnCompare,
        with_column_compare_scalar: Payloads.WithColumnCompareScalar,
        filter_mask: Payloads.FilterMask,
        filter_column: Payloads.FilterColumn,
        filter_scalar: Payloads.FilterScalar,
        group_by_count: Payloads.GroupByCount,
        group_by_value: Payloads.GroupByValue,
        group_by_stats: Payloads.GroupByOutput,
        group_by_stats_on: Payloads.GroupByOutputOn,
        group_by_profile: Payloads.GroupByOutput,
        group_by_profile_on: Payloads.GroupByOutputOn,
        join_on: Payloads.JoinOn,
        asof_join: Payloads.AsofJoin,
        concat_rows: Payloads.ConcatRows,
        distinct_rows,
        distinct_on: Payloads.DistinctOn,
        sort_by: Payloads.SortBy,
        top_k: Payloads.TopK,
        rank_profile_by: Payloads.RankProfileBy,
        rolling_profile: Payloads.RollingProfile,
        rolling_moment_profile: Payloads.RollingProfile,
        rolling_range_profile: Payloads.RollingProfile,
        rolling_normalize_profile: Payloads.RollingProfile,
        expanding_normalize_profile: Payloads.ExpandingProfile,
        rolling_quantile_profile: Payloads.RollingProfile,
        expanding_quantile_profile: Payloads.ExpandingProfile,
        rolling_bool_profile: Payloads.RollingProfile,
        rolling_drawdown_profile: Payloads.RollingProfile,
        rolling_robust_profile: Payloads.RollingRobustProfile,
        rolling_rank_profile: Payloads.RollingRankProfile,
        lag_profile: Payloads.LagProfile,
        lead_profile: Payloads.LagProfile,
        clip_profile: Payloads.ClipProfile,
        rolling_clip_profile: Payloads.RollingClipProfile,
        expanding_clip_profile: Payloads.ExpandingClipProfile,
        threshold_profile: Payloads.ThresholdProfile,
        rolling_threshold_profile: Payloads.RollingThresholdProfile,
        expanding_threshold_profile: Payloads.ExpandingThresholdProfile,
        expanding_profile: Payloads.ExpandingProfile,
        expanding_bool_profile: Payloads.ExpandingProfile,
        expanding_rank_profile: Payloads.ExpandingRankProfile,
        expanding_robust_profile: Payloads.ExpandingRobustProfile,
        expanding_moment_profile: Payloads.ExpandingProfile,
        standardize_profile: Payloads.StandardizeProfile,
        robust_profile: Payloads.RobustProfile,
        drawdown_profile: Payloads.DrawdownProfile,
        extrema_profile: Payloads.ExtremaProfile,
        trend_profile: Payloads.TrendProfile,
        rolling_trend_profile: Payloads.RollingTrendProfile,
        expanding_trend_profile: Payloads.ExpandingTrendProfile,
        change_point_profile: Payloads.ChangePointProfile,
        rolling_change_point_profile: Payloads.RollingChangePointProfile,
        expanding_change_point_profile: Payloads.ExpandingChangePointProfile,
        sign_profile: Payloads.TrendProfile,
        rolling_sign_profile: Payloads.RollingSignProfile,
        expanding_sign_profile: Payloads.ExpandingSignProfile,
        crossover_profile: Payloads.CrossoverProfile,
        rolling_crossover_profile: Payloads.RollingCrossoverProfile,
        expanding_crossover_profile: Payloads.ExpandingCrossoverProfile,
        bucket_profile: Payloads.BucketProfile,
        ema_profile: Payloads.EmaProfile,
        linear_fit_profile: Payloads.LinearFitProfile,
        error_profile: Payloads.PairOutput,
        rolling_error_profile: Payloads.RollingPairOutput,
        expanding_error_profile: Payloads.ExpandingPairOutput,
        classification_profile: Payloads.PairOutput,
        rolling_classification_profile: Payloads.RollingPairOutput,
        expanding_classification_profile: Payloads.ExpandingPairOutput,
        bool_transition_profile: Payloads.BoolTransitionProfile,
        rolling_bool_transition_profile: Payloads.RollingBoolTransitionProfile,
        expanding_bool_transition_profile: Payloads.ExpandingBoolTransitionProfile,
        rolling_correlation_profile: Payloads.RollingCorrelationProfile,
        expanding_correlation_profile: Payloads.ExpandingXYProfile,
        expanding_linear_fit_profile: Payloads.ExpandingXYProfile,
        rolling_linear_fit_profile: Payloads.RollingLinearFitProfile,
        validity_profile: Payloads.ValidityProfile,
        rolling_validity_profile: Payloads.RollingValidityProfile,
        expanding_validity_profile: Payloads.ExpandingValidityProfile,
        slice_rows: Payloads.RowSlice,
        take_rows: Payloads.RowTake,
        head: Payloads.HeadTail,
        tail: Payloads.HeadTail,

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            ownership_mod.deinit(Self, self, allocator);
        }

        pub fn clone(self: Self, allocator: std.mem.Allocator) DeviceDataError!Self {
            return ownership_mod.clone(Self, self, allocator);
        }
    };
}
