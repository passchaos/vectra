//! Projection-blocking profile dependency extraction for lazy scan pushdown.
//!
//! Profile operations append generated columns while preserving source columns.
//! Until the lazy planner tracks source-vs-derived schema explicitly, scan
//! projection cannot safely pass through them.  This helper records the source
//! column dependencies that are still useful for a conservative scan plan while
//! keeping the large profile-tag list out of the main pushdown state machine.

const std = @import("std");
const names_mod = @import("../../dataframe_names.zig");

const appendOwnedNameUnique = names_mod.appendOwnedNameUnique;
const nameInBorrowedList = names_mod.nameInBorrowedList;

fn addNameDependency(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: []const []const u8,
    name: []const u8,
) std.mem.Allocator.Error!void {
    if (!nameInBorrowedList(name, derived_names)) try appendOwnedNameUnique(allocator, required_names, name);
}

pub fn addDependencies(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: []const []const u8,
    op: anytype,
) std.mem.Allocator.Error!void {
    switch (op) {
        inline .rank_profile_by,
        .rolling_profile,
        .rolling_moment_profile,
        .rolling_range_profile,
        .rolling_normalize_profile,
        .expanding_normalize_profile,
        .rolling_quantile_profile,
        .expanding_quantile_profile,
        .rolling_bool_profile,
        .rolling_drawdown_profile,
        .rolling_robust_profile,
        .rolling_rank_profile,
        .lag_profile,
        .lead_profile,
        .clip_profile,
        .rolling_clip_profile,
        .expanding_clip_profile,
        .threshold_profile,
        .rolling_threshold_profile,
        .expanding_threshold_profile,
        .expanding_profile,
        .expanding_bool_profile,
        .expanding_rank_profile,
        .expanding_robust_profile,
        .expanding_moment_profile,
        .standardize_profile,
        .robust_profile,
        .drawdown_profile,
        .extrema_profile,
        .trend_profile,
        .rolling_trend_profile,
        .expanding_trend_profile,
        .change_point_profile,
        .rolling_change_point_profile,
        .expanding_change_point_profile,
        .sign_profile,
        .rolling_sign_profile,
        .expanding_sign_profile,
        .bucket_profile,
        .ema_profile,
        .bool_transition_profile,
        .rolling_bool_transition_profile,
        .expanding_bool_transition_profile,
        .validity_profile,
        .rolling_validity_profile,
        .expanding_validity_profile,
        => |profile| try addNameDependency(allocator, required_names, derived_names, profile.name),

        inline .crossover_profile,
        .rolling_crossover_profile,
        .expanding_crossover_profile,
        => |profile| {
            try addNameDependency(allocator, required_names, derived_names, profile.lhs_name);
            try addNameDependency(allocator, required_names, derived_names, profile.rhs_name);
        },

        inline .linear_fit_profile,
        .expanding_linear_fit_profile,
        .rolling_linear_fit_profile,
        .rolling_correlation_profile,
        .expanding_correlation_profile,
        => |profile| {
            try addNameDependency(allocator, required_names, derived_names, profile.x_name);
            try addNameDependency(allocator, required_names, derived_names, profile.y_name);
        },

        inline .error_profile,
        .rolling_error_profile,
        .expanding_error_profile,
        .classification_profile,
        .rolling_classification_profile,
        .expanding_classification_profile,
        => |profile| {
            try addNameDependency(allocator, required_names, derived_names, profile.actual_name);
            try addNameDependency(allocator, required_names, derived_names, profile.predicted_name);
        },

        else => unreachable,
    }
}
