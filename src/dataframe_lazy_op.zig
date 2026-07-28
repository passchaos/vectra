//! Lazy dataframe operation tags and ownership helpers.
//!
//! `DeviceLazyOp` carries many operation payload variants and therefore used to
//! dominate the public dataframe facade. Keeping the union generic over the
//! concrete dataframe/column types avoids an import cycle while moving the
//! clone/deinit ownership rules into a focused module.

const std = @import("std");
const array_mod = @import("array.zig");
const ownership_mod = @import("dataframe_lazy_op_ownership.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceJoinOptions = options_mod.DeviceJoinOptions;
const DeviceAsofOptions = options_mod.DeviceAsofOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceLagOptions = options_mod.DeviceLagOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
const DeviceStandardizeOptions = options_mod.DeviceStandardizeOptions;
const DeviceRobustOptions = options_mod.DeviceRobustOptions;
const DeviceDrawdownOptions = options_mod.DeviceDrawdownOptions;
const DeviceExtremaOptions = options_mod.DeviceExtremaOptions;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
const DeviceBucketOptions = options_mod.DeviceBucketOptions;
const DeviceEmaOptions = options_mod.DeviceEmaOptions;
const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
const DeviceClipOptions = options_mod.DeviceClipOptions;
const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
const DeviceRollingRobustOptions = options_mod.DeviceRollingRobustOptions;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub const DeviceLazyGroupByAggregation = enum {
    sum,
    min,
    max,
    mean,
};

pub const DeviceLazyJoinKind = enum {
    inner,
    left,
    full,
    semi,
    anti,
};

pub fn DeviceLazyOp(comptime DeviceDataFrame: type, comptime DeviceColumn: type) type {
    return union(enum) {
        const Self = @This();
        select: [][]const u8,
        with_column_binary: struct {
            name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            op: DeviceColumnBinaryOp,
        },
        with_column_scalar: struct {
            name: []const u8,
            input_name: []const u8,
            op: DeviceColumnBinaryOp,
            scalar: DeviceScalar,
        },
        with_column_compare: struct {
            name: []const u8,
            lhs_name: []const u8,
            rhs_name: []const u8,
            op: DeviceColumnCompareOp,
        },
        with_column_compare_scalar: struct {
            name: []const u8,
            input_name: []const u8,
            op: DeviceColumnCompareOp,
            scalar: DeviceScalar,
        },
        filter_mask: DeviceColumn,
        filter_scalar: struct {
            name: []const u8,
            op: DeviceColumnCompareOp,
            scalar: DeviceScalar,
        },
        group_by_count: struct {
            key_name: []const u8,
            output_name: []const u8,
        },
        group_by_value: struct {
            key_name: []const u8,
            value_name: []const u8,
            output_name: []const u8,
            aggregation: DeviceLazyGroupByAggregation,
        },
        group_by_stats: struct {
            key_name: []const u8,
            value_name: []const u8,
            output_prefix: []const u8,
        },
        group_by_stats_on: struct {
            key_names: [][]const u8,
            value_name: []const u8,
            output_prefix: []const u8,
        },
        group_by_profile: struct {
            key_name: []const u8,
            value_name: []const u8,
            output_prefix: []const u8,
        },
        group_by_profile_on: struct {
            key_names: [][]const u8,
            value_name: []const u8,
            output_prefix: []const u8,
        },
        join_on: struct {
            kind: DeviceLazyJoinKind,
            right: DeviceDataFrame,
            left_key_names: [][]const u8,
            right_key_names: [][]const u8,
            options: DeviceJoinOptions,
        },
        asof_join: struct {
            right: DeviceDataFrame,
            left_key_name: []const u8,
            right_key_name: []const u8,
            options: DeviceAsofOptions,
        },
        concat_rows: DeviceDataFrame,
        distinct_rows,
        distinct_on: [][]const u8,
        sort_by: struct {
            name: []const u8,
            options: DeviceSortOptions,
        },
        top_k: struct {
            name: []const u8,
            options: DeviceSortOptions,
            k: usize,
        },
        rank_profile_by: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceSortOptions,
        },
        rolling_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        rolling_moment_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        rolling_range_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        rolling_normalize_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        expanding_normalize_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        rolling_quantile_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        expanding_quantile_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        rolling_bool_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        rolling_drawdown_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        rolling_robust_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingRobustOptions,
        },
        rolling_rank_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingRankOptions,
        },
        lag_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceLagOptions,
        },
        lead_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceLagOptions,
        },
        clip_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceClipOptions,
        },
        rolling_clip_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            clip_options: DeviceClipOptions,
            options: DeviceRollingOptions,
        },
        expanding_clip_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            clip_options: DeviceClipOptions,
            options: DeviceExpandingOptions,
        },
        threshold_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceThresholdOptions,
        },
        rolling_threshold_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            threshold: f64,
            options: DeviceRollingOptions,
        },
        expanding_threshold_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            threshold: f64,
            options: DeviceExpandingOptions,
        },
        expanding_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        expanding_bool_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        expanding_rank_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingRankOptions,
        },
        expanding_robust_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRobustOptions,
        },
        expanding_moment_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        standardize_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceStandardizeOptions,
        },
        robust_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRobustOptions,
        },
        drawdown_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceDrawdownOptions,
        },
        extrema_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceExtremaOptions,
        },
        trend_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceTrendOptions,
        },
        rolling_trend_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            trend_options: DeviceTrendOptions,
            options: DeviceRollingOptions,
        },
        expanding_trend_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            trend_options: DeviceTrendOptions,
            options: DeviceExpandingOptions,
        },
        change_point_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            threshold: f64,
            options: DeviceTrendOptions,
        },
        rolling_change_point_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            threshold: f64,
            change_options: DeviceTrendOptions,
            options: DeviceRollingOptions,
        },
        expanding_change_point_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            threshold: f64,
            change_options: DeviceTrendOptions,
            options: DeviceExpandingOptions,
        },
        sign_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceTrendOptions,
        },
        rolling_sign_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            sign_options: DeviceTrendOptions,
            options: DeviceRollingOptions,
        },
        expanding_sign_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            sign_options: DeviceTrendOptions,
            options: DeviceExpandingOptions,
        },
        crossover_profile: struct {
            lhs_name: []const u8,
            rhs_name: []const u8,
            output_prefix: []const u8,
            options: DeviceCrossoverOptions,
        },
        rolling_crossover_profile: struct {
            lhs_name: []const u8,
            rhs_name: []const u8,
            output_prefix: []const u8,
            cross_options: DeviceCrossoverOptions,
            options: DeviceRollingOptions,
        },
        expanding_crossover_profile: struct {
            lhs_name: []const u8,
            rhs_name: []const u8,
            output_prefix: []const u8,
            cross_options: DeviceCrossoverOptions,
            options: DeviceExpandingOptions,
        },
        bucket_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceBucketOptions,
        },
        ema_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceEmaOptions,
        },
        linear_fit_profile: struct {
            x_name: []const u8,
            y_name: []const u8,
            output_prefix: []const u8,
            options: DeviceLinearFitOptions,
        },
        error_profile: struct {
            actual_name: []const u8,
            predicted_name: []const u8,
            output_prefix: []const u8,
        },
        rolling_error_profile: struct {
            actual_name: []const u8,
            predicted_name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        expanding_error_profile: struct {
            actual_name: []const u8,
            predicted_name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        classification_profile: struct {
            actual_name: []const u8,
            predicted_name: []const u8,
            output_prefix: []const u8,
        },
        rolling_classification_profile: struct {
            actual_name: []const u8,
            predicted_name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        expanding_classification_profile: struct {
            actual_name: []const u8,
            predicted_name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        bool_transition_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceTrendOptions,
        },
        rolling_bool_transition_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            transition_options: DeviceTrendOptions,
            options: DeviceRollingOptions,
        },
        expanding_bool_transition_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            transition_options: DeviceTrendOptions,
            options: DeviceExpandingOptions,
        },
        rolling_correlation_profile: struct {
            x_name: []const u8,
            y_name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingCorrelationOptions,
        },
        expanding_correlation_profile: struct {
            x_name: []const u8,
            y_name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        expanding_linear_fit_profile: struct {
            x_name: []const u8,
            y_name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        rolling_linear_fit_profile: struct {
            x_name: []const u8,
            y_name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingCorrelationOptions,
        },
        validity_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
        },
        rolling_validity_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceRollingOptions,
        },
        expanding_validity_profile: struct {
            name: []const u8,
            output_prefix: []const u8,
            options: DeviceExpandingOptions,
        },
        head: usize,
        tail: usize,

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            ownership_mod.deinit(Self, self, allocator);
        }

        pub fn clone(self: Self, allocator: std.mem.Allocator) DeviceDataError!Self {
            return ownership_mod.clone(Self, self, allocator);
        }
    };
}
