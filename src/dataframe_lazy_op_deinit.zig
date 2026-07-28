//! Deinitialization helper for DeviceLazyOp payloads.

const std = @import("std");
const names_mod = @import("dataframe_names.zig");

const freeNameList = names_mod.freeNameList;

pub fn deinit(comptime Self: type, self: *Self, allocator: std.mem.Allocator) void {
    switch (self.*) {
        .select => |names| {
            for (names) |name| allocator.free(name);
            allocator.free(names);
        },
        .with_column_binary => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .with_column_compare => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.lhs_name);
            allocator.free(expr.rhs_name);
        },
        .with_column_compare_scalar => |expr| {
            allocator.free(expr.name);
            allocator.free(expr.input_name);
        },
        .filter_mask => |*mask| mask.deinit(),
        .filter_scalar => |filter_op| allocator.free(filter_op.name),
        .group_by_count => |group| {
            allocator.free(group.key_name);
            allocator.free(group.output_name);
        },
        .group_by_value => |group| {
            allocator.free(group.key_name);
            allocator.free(group.value_name);
            allocator.free(group.output_name);
        },
        .group_by_stats => |group| {
            allocator.free(group.key_name);
            allocator.free(group.value_name);
            allocator.free(group.output_prefix);
        },
        .group_by_stats_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.value_name);
            allocator.free(group.output_prefix);
        },
        .group_by_profile => |group| {
            allocator.free(group.key_name);
            allocator.free(group.value_name);
            allocator.free(group.output_prefix);
        },
        .group_by_profile_on => |group| {
            freeNameList(allocator, group.key_names);
            allocator.free(group.value_name);
            allocator.free(group.output_prefix);
        },
        .join_on => |*join| {
            join.right.deinit();
            freeNameList(allocator, join.left_key_names);
            freeNameList(allocator, join.right_key_names);
            allocator.free(join.options.right_suffix);
        },
        .asof_join => |*join| {
            join.right.deinit();
            allocator.free(join.left_key_name);
            allocator.free(join.right_key_name);
            allocator.free(join.options.right_suffix);
        },
        .concat_rows => |*right| right.deinit(),
        .distinct_on => |names| freeNameList(allocator, names),
        .sort_by => |sort| allocator.free(sort.name),
        .top_k => |top| allocator.free(top.name),
        .rank_profile_by => |rank| {
            allocator.free(rank.name);
            allocator.free(rank.output_prefix);
        },
        .rolling_profile => |rolling| {
            allocator.free(rolling.name);
            allocator.free(rolling.output_prefix);
        },
        .rolling_moment_profile => |rolling| {
            allocator.free(rolling.name);
            allocator.free(rolling.output_prefix);
        },
        .rolling_range_profile => |rolling| {
            allocator.free(rolling.name);
            allocator.free(rolling.output_prefix);
        },
        .rolling_normalize_profile => |rolling| {
            allocator.free(rolling.name);
            allocator.free(rolling.output_prefix);
        },
        .expanding_normalize_profile => |expanding| {
            allocator.free(expanding.name);
            allocator.free(expanding.output_prefix);
        },
        .rolling_quantile_profile => |rolling| {
            allocator.free(rolling.name);
            allocator.free(rolling.output_prefix);
        },
        .expanding_quantile_profile => |expanding| {
            allocator.free(expanding.name);
            allocator.free(expanding.output_prefix);
        },
        .rolling_bool_profile => |rolling| {
            allocator.free(rolling.name);
            allocator.free(rolling.output_prefix);
        },
        .rolling_drawdown_profile => |rolling| {
            allocator.free(rolling.name);
            allocator.free(rolling.output_prefix);
        },
        .rolling_robust_profile => |rolling| {
            allocator.free(rolling.name);
            allocator.free(rolling.output_prefix);
        },
        .rolling_rank_profile => |rolling| {
            allocator.free(rolling.name);
            allocator.free(rolling.output_prefix);
        },
        .lag_profile => |lag| {
            allocator.free(lag.name);
            allocator.free(lag.output_prefix);
        },
        .lead_profile => |lead| {
            allocator.free(lead.name);
            allocator.free(lead.output_prefix);
        },
        .clip_profile => |clip| {
            allocator.free(clip.name);
            allocator.free(clip.output_prefix);
        },
        .rolling_clip_profile => |clip| {
            allocator.free(clip.name);
            allocator.free(clip.output_prefix);
        },
        .expanding_clip_profile => |clip| {
            allocator.free(clip.name);
            allocator.free(clip.output_prefix);
        },
        .threshold_profile => |threshold| {
            allocator.free(threshold.name);
            allocator.free(threshold.output_prefix);
        },
        .rolling_threshold_profile => |threshold| {
            allocator.free(threshold.name);
            allocator.free(threshold.output_prefix);
        },
        .expanding_threshold_profile => |threshold| {
            allocator.free(threshold.name);
            allocator.free(threshold.output_prefix);
        },
        .expanding_profile => |expanding| {
            allocator.free(expanding.name);
            allocator.free(expanding.output_prefix);
        },
        .expanding_bool_profile => |expanding| {
            allocator.free(expanding.name);
            allocator.free(expanding.output_prefix);
        },
        .expanding_rank_profile => |expanding| {
            allocator.free(expanding.name);
            allocator.free(expanding.output_prefix);
        },
        .expanding_robust_profile => |expanding| {
            allocator.free(expanding.name);
            allocator.free(expanding.output_prefix);
        },
        .expanding_moment_profile => |expanding| {
            allocator.free(expanding.name);
            allocator.free(expanding.output_prefix);
        },
        .standardize_profile => |standardize| {
            allocator.free(standardize.name);
            allocator.free(standardize.output_prefix);
        },
        .robust_profile => |robust| {
            allocator.free(robust.name);
            allocator.free(robust.output_prefix);
        },
        .drawdown_profile => |drawdown| {
            allocator.free(drawdown.name);
            allocator.free(drawdown.output_prefix);
        },
        .extrema_profile => |extrema| {
            allocator.free(extrema.name);
            allocator.free(extrema.output_prefix);
        },
        .trend_profile => |trend| {
            allocator.free(trend.name);
            allocator.free(trend.output_prefix);
        },
        .rolling_trend_profile => |trend| {
            allocator.free(trend.name);
            allocator.free(trend.output_prefix);
        },
        .expanding_trend_profile => |trend| {
            allocator.free(trend.name);
            allocator.free(trend.output_prefix);
        },
        .change_point_profile => |change| {
            allocator.free(change.name);
            allocator.free(change.output_prefix);
        },
        .rolling_change_point_profile => |change| {
            allocator.free(change.name);
            allocator.free(change.output_prefix);
        },
        .expanding_change_point_profile => |change| {
            allocator.free(change.name);
            allocator.free(change.output_prefix);
        },
        .sign_profile => |sign| {
            allocator.free(sign.name);
            allocator.free(sign.output_prefix);
        },
        .rolling_sign_profile => |sign| {
            allocator.free(sign.name);
            allocator.free(sign.output_prefix);
        },
        .expanding_sign_profile => |sign| {
            allocator.free(sign.name);
            allocator.free(sign.output_prefix);
        },
        .crossover_profile => |cross| {
            allocator.free(cross.lhs_name);
            allocator.free(cross.rhs_name);
            allocator.free(cross.output_prefix);
        },
        .rolling_crossover_profile => |cross| {
            allocator.free(cross.lhs_name);
            allocator.free(cross.rhs_name);
            allocator.free(cross.output_prefix);
        },
        .expanding_crossover_profile => |cross| {
            allocator.free(cross.lhs_name);
            allocator.free(cross.rhs_name);
            allocator.free(cross.output_prefix);
        },
        .bucket_profile => |bucket| {
            allocator.free(bucket.name);
            allocator.free(bucket.output_prefix);
        },
        .ema_profile => |ema| {
            allocator.free(ema.name);
            allocator.free(ema.output_prefix);
        },
        .linear_fit_profile => |fit| {
            allocator.free(fit.x_name);
            allocator.free(fit.y_name);
            allocator.free(fit.output_prefix);
        },
        .error_profile => |err| {
            allocator.free(err.actual_name);
            allocator.free(err.predicted_name);
            allocator.free(err.output_prefix);
        },
        .rolling_error_profile => |err| {
            allocator.free(err.actual_name);
            allocator.free(err.predicted_name);
            allocator.free(err.output_prefix);
        },
        .expanding_error_profile => |err| {
            allocator.free(err.actual_name);
            allocator.free(err.predicted_name);
            allocator.free(err.output_prefix);
        },
        .classification_profile => |class| {
            allocator.free(class.actual_name);
            allocator.free(class.predicted_name);
            allocator.free(class.output_prefix);
        },
        .rolling_classification_profile => |class| {
            allocator.free(class.actual_name);
            allocator.free(class.predicted_name);
            allocator.free(class.output_prefix);
        },
        .expanding_classification_profile => |class| {
            allocator.free(class.actual_name);
            allocator.free(class.predicted_name);
            allocator.free(class.output_prefix);
        },
        .bool_transition_profile => |transition| {
            allocator.free(transition.name);
            allocator.free(transition.output_prefix);
        },
        .rolling_bool_transition_profile => |transition| {
            allocator.free(transition.name);
            allocator.free(transition.output_prefix);
        },
        .expanding_bool_transition_profile => |transition| {
            allocator.free(transition.name);
            allocator.free(transition.output_prefix);
        },
        .rolling_correlation_profile => |corr| {
            allocator.free(corr.x_name);
            allocator.free(corr.y_name);
            allocator.free(corr.output_prefix);
        },
        .expanding_correlation_profile => |corr| {
            allocator.free(corr.x_name);
            allocator.free(corr.y_name);
            allocator.free(corr.output_prefix);
        },
        .expanding_linear_fit_profile => |fit| {
            allocator.free(fit.x_name);
            allocator.free(fit.y_name);
            allocator.free(fit.output_prefix);
        },
        .rolling_linear_fit_profile => |fit| {
            allocator.free(fit.x_name);
            allocator.free(fit.y_name);
            allocator.free(fit.output_prefix);
        },
        .validity_profile => |validity| {
            allocator.free(validity.name);
            allocator.free(validity.output_prefix);
        },
        .rolling_validity_profile => |validity| {
            allocator.free(validity.name);
            allocator.free(validity.output_prefix);
        },
        .expanding_validity_profile => |validity| {
            allocator.free(validity.name);
            allocator.free(validity.output_prefix);
        },
        .distinct_rows, .head, .tail => {},
    }
    self.* = undefined;
}
