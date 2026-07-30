//! Deinitialization helper for DeviceLazyOp payloads.

const std = @import("std");
const names_mod = @import("dataframe_names.zig");
const profile_deinit_mod = @import("dataframe_lazy_op_deinit_profile.zig");

const freeNameList = names_mod.freeNameList;
const freeNameOutput = profile_deinit_mod.freeNameOutput;
const freePairOutput = profile_deinit_mod.freePairOutput;

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
        .filter_column => |name| allocator.free(name),
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
        .rank_profile_by => |payload| freeNameOutput(allocator, payload),
        .rolling_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_moment_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_range_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_normalize_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_normalize_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_quantile_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_quantile_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_bool_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_drawdown_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_robust_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_rank_profile => |payload| freeNameOutput(allocator, payload),
        .lag_profile => |payload| freeNameOutput(allocator, payload),
        .lead_profile => |payload| freeNameOutput(allocator, payload),
        .clip_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_clip_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_clip_profile => |payload| freeNameOutput(allocator, payload),
        .threshold_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_threshold_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_threshold_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_bool_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_rank_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_robust_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_moment_profile => |payload| freeNameOutput(allocator, payload),
        .standardize_profile => |payload| freeNameOutput(allocator, payload),
        .robust_profile => |payload| freeNameOutput(allocator, payload),
        .drawdown_profile => |payload| freeNameOutput(allocator, payload),
        .extrema_profile => |payload| freeNameOutput(allocator, payload),
        .trend_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_trend_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_trend_profile => |payload| freeNameOutput(allocator, payload),
        .change_point_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_change_point_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_change_point_profile => |payload| freeNameOutput(allocator, payload),
        .sign_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_sign_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_sign_profile => |payload| freeNameOutput(allocator, payload),
        .crossover_profile => |payload| freePairOutput(allocator, payload, "lhs_name", "rhs_name"),
        .rolling_crossover_profile => |payload| freePairOutput(allocator, payload, "lhs_name", "rhs_name"),
        .expanding_crossover_profile => |payload| freePairOutput(allocator, payload, "lhs_name", "rhs_name"),
        .bucket_profile => |payload| freeNameOutput(allocator, payload),
        .ema_profile => |payload| freeNameOutput(allocator, payload),
        .linear_fit_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .error_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .rolling_error_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .expanding_error_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .classification_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .rolling_classification_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .expanding_classification_profile => |payload| freePairOutput(allocator, payload, "actual_name", "predicted_name"),
        .bool_transition_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_bool_transition_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_bool_transition_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_correlation_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .expanding_correlation_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .expanding_linear_fit_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .rolling_linear_fit_profile => |payload| freePairOutput(allocator, payload, "x_name", "y_name"),
        .validity_profile => |payload| freeNameOutput(allocator, payload),
        .rolling_validity_profile => |payload| freeNameOutput(allocator, payload),
        .expanding_validity_profile => |payload| freeNameOutput(allocator, payload),
        .take_rows => |row_indices| allocator.free(row_indices),
        .distinct_rows, .slice_rows, .head, .tail => {},
    }
    self.* = undefined;
}
