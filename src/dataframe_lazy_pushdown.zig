//! Parquet scan pushdown planning for lazy dataframe plans.

const std = @import("std");
const names_mod = @import("dataframe_names.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
const DeviceScalar = options_mod.DeviceScalar;
const ParquetRangePredicate = options_mod.ParquetRangePredicate;
const Range = options_mod.Range;
const appendOwnedNameUnique = names_mod.appendOwnedNameUnique;
const appendBorrowedNameUnique = names_mod.appendBorrowedNameUnique;
const nameInBorrowedList = names_mod.nameInBorrowedList;
const freeOwnedNameItems = names_mod.freeOwnedNameItems;
const freeNameList = names_mod.freeNameList;

pub const LazyScanPushdown = struct {
    allocator: std.mem.Allocator,
    projection: ?[][]const u8 = null,
    range_predicate: ?DeviceParquetRangeFilter = null,

    pub fn deinit(self: *LazyScanPushdown) void {
        if (self.projection) |names| freeNameList(self.allocator, names);
        if (self.range_predicate) |predicate| self.allocator.free(predicate.column);
        self.* = undefined;
    }
};

pub fn planLazyScanPushdown(allocator: std.mem.Allocator, ops: anytype) std.mem.Allocator.Error!LazyScanPushdown {
    var required_names: std.ArrayList([]const u8) = .empty;
    errdefer required_names.deinit(allocator);
    errdefer freeOwnedNameItems(allocator, required_names.items);
    var derived_names: std.ArrayList([]const u8) = .empty;
    defer derived_names.deinit(allocator);

    var saw_select = false;
    var projection_blocked = false;
    var range_predicate: ?DeviceParquetRangeFilter = null;
    errdefer if (range_predicate) |predicate| allocator.free(predicate.column);

    op_loop: for (ops) |op| {
        switch (op) {
            .select => |names| {
                saw_select = true;
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .with_column_binary => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
            },
            .with_column_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
            },
            .with_column_compare => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.lhs_name);
                try appendOwnedNameUnique(allocator, &required_names, expr.rhs_name);
            },
            .with_column_compare_scalar => |expr| {
                try appendBorrowedNameUnique(allocator, &derived_names, expr.name);
                try appendOwnedNameUnique(allocator, &required_names, expr.input_name);
            },
            .group_by_count => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_value => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_stats => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_stats_on => |group| {
                for (group.key_names) |key_name| {
                    if (!nameInBorrowedList(key_name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, key_name);
                    }
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_profile => |group| {
                if (!nameInBorrowedList(group.key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.key_name);
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .group_by_profile_on => |group| {
                for (group.key_names) |key_name| {
                    if (!nameInBorrowedList(key_name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, key_name);
                    }
                }
                if (!nameInBorrowedList(group.value_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, group.value_name);
                }
                saw_select = true;
                break :op_loop;
            },
            .join_on => |join| {
                // A join changes the output schema by adding right-side payload
                // columns.  Without source schema metadata at this planning
                // layer, a later select cannot be safely split into left-source
                // columns vs. right payload columns.  Keep row-group predicate
                // pruning, but conservatively disable Parquet projection
                // pushdown for the left source rather than risk dropping a left
                // payload or requesting a right column from the source scan.
                projection_blocked = true;
                for (join.left_key_names) |key_name| {
                    if (!nameInBorrowedList(key_name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, key_name);
                    }
                }
                break :op_loop;
            },
            .asof_join => |join| {
                projection_blocked = true;
                if (!nameInBorrowedList(join.left_key_name, derived_names.items)) {
                    try appendOwnedNameUnique(allocator, &required_names, join.left_key_name);
                }
                break :op_loop;
            },
            .concat_rows => {
                break :op_loop;
            },
            .distinct_rows => {
                projection_blocked = true;
            },
            .distinct_on => |names| {
                for (names) |name| {
                    if (!nameInBorrowedList(name, derived_names.items)) {
                        try appendOwnedNameUnique(allocator, &required_names, name);
                    }
                }
            },
            .filter_scalar => |filter_op| {
                const filter_depends_on_source = !nameInBorrowedList(filter_op.name, derived_names.items);
                if (filter_depends_on_source) try appendOwnedNameUnique(allocator, &required_names, filter_op.name);
                if (filter_depends_on_source and range_predicate == null) {
                    if (parquetRangePredicateFromScalar(filter_op.scalar, filter_op.op)) |predicate| {
                        range_predicate = .{
                            .column = try allocator.dupe(u8, filter_op.name),
                            .predicate = predicate,
                        };
                    }
                }
            },
            .sort_by => |sort| {
                if (!nameInBorrowedList(sort.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sort.name);
            },
            .top_k => |top| {
                if (!nameInBorrowedList(top.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, top.name);
            },
            .rank_profile_by => |rank| {
                // A rank profile appends derived rank/window columns while
                // preserving the rest of the input table.  Without source schema
                // metadata here, a later select cannot be split safely into
                // source columns vs. rank-derived columns, so keep scalar
                // predicate pruning but avoid Parquet projection pushdown.
                projection_blocked = true;
                if (!nameInBorrowedList(rank.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rank.name);
                break :op_loop;
            },
            .rolling_profile => |rolling| {
                // Rolling profiles append several derived columns and preserve
                // the existing table, so projection pushdown needs schema
                // awareness to avoid dropping later-selected source columns.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_moment_profile => |rolling| {
                // Rolling moment profiles append higher-order distribution
                // diagnostics while preserving source columns. Keep predicates
                // but block projection until generated-field schema is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_range_profile => |rolling| {
                // Rolling range profiles append low/high/range/position fields
                // and preserve the input table, so projection pushdown needs
                // schema awareness before it can safely pass this operation.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_normalize_profile => |rolling| {
                // Rolling normalize profiles append window-local scaling fields
                // and preserve the input table, so projection pushdown needs
                // derived-field schema awareness before it can pass through.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .expanding_normalize_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .rolling_quantile_profile => |rolling| {
                // Rolling quantile profiles append window distribution fields and
                // preserve the input table, so projection pushdown needs
                // generated-field schema awareness before it can pass through.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .expanding_quantile_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .rolling_bool_profile => |rolling| {
                // Rolling bool profiles append count/rate/predicate diagnostics
                // while preserving source columns. Keep scan predicates, but do
                // not push projection through generated bool-window fields until
                // the lazy planner tracks source-vs-derived schema explicitly.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_drawdown_profile => |rolling| {
                // Rolling drawdown profiles append window-local risk diagnostics
                // and preserve the source schema. Keep scan predicates but block
                // projection pushdown until generated fields are tracked.
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_robust_profile => |rolling| {
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .rolling_rank_profile => |rolling| {
                projection_blocked = true;
                if (!nameInBorrowedList(rolling.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, rolling.name);
                break :op_loop;
            },
            .lag_profile => |lag| {
                // Lag profiles append multiple derived columns and preserve the
                // input schema.  Like rank/rolling profiles, keep scan predicate
                // pruning but avoid unsafe projection pushdown until the planner
                // has schema-level knowledge of derived vs. source fields.
                projection_blocked = true;
                if (!nameInBorrowedList(lag.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, lag.name);
                break :op_loop;
            },
            .lead_profile => |lead| {
                // Lead profiles append forward-looking derived columns and
                // preserve the input schema. Keep scan predicates, but block
                // projection pushdown across generated lead fields.
                projection_blocked = true;
                if (!nameInBorrowedList(lead.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, lead.name);
                break :op_loop;
            },
            .clip_profile => |clip| {
                // Clip profiles append cleaning diagnostics and preserve the
                // source column, so projection pushdown must wait for generated
                // field schema awareness.
                projection_blocked = true;
                if (!nameInBorrowedList(clip.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, clip.name);
                break :op_loop;
            },
            .rolling_clip_profile => |clip| {
                // Rolling clip profiles append window-level clipping summaries
                // and keep source columns. Predicate pruning is still safe, but
                // projection pushdown needs generated-field schema awareness.
                projection_blocked = true;
                if (!nameInBorrowedList(clip.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, clip.name);
                break :op_loop;
            },
            .expanding_clip_profile => |clip| {
                // Expanding clip profiles append cumulative clipping summaries
                // while preserving the input table. Keep scan predicates, but
                // block projection until generated fields are tracked.
                projection_blocked = true;
                if (!nameInBorrowedList(clip.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, clip.name);
                break :op_loop;
            },
            .threshold_profile => |threshold| {
                projection_blocked = true;
                if (!nameInBorrowedList(threshold.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, threshold.name);
                break :op_loop;
            },
            .rolling_threshold_profile => |threshold| {
                // Rolling threshold profiles append window-level distance/rate
                // diagnostics while preserving the input table. Keep scalar
                // predicate pruning, but block projection pushdown until the
                // planner tracks generated profile fields separately.
                projection_blocked = true;
                if (!nameInBorrowedList(threshold.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, threshold.name);
                break :op_loop;
            },
            .expanding_threshold_profile => |threshold| {
                // Expanding threshold profiles append cumulative distance/rate
                // diagnostics and preserve source fields. Keep scan predicates
                // but block projection pushdown across generated columns until
                // lazy schema tracking can split source vs. derived names.
                projection_blocked = true;
                if (!nameInBorrowedList(threshold.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, threshold.name);
                break :op_loop;
            },
            .expanding_profile => |expanding| {
                // Expanding profiles append cumulative derived columns while
                // preserving source columns.  Keep the source dependency for
                // scans, but do not push projection through this schema-changing
                // operation until planner schema metadata is richer.
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .expanding_bool_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .expanding_rank_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .expanding_robust_profile => |expanding| {
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .expanding_moment_profile => |expanding| {
                // Expanding moment profiles append higher-order cumulative
                // distribution diagnostics while preserving source columns.
                projection_blocked = true;
                if (!nameInBorrowedList(expanding.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, expanding.name);
                break :op_loop;
            },
            .standardize_profile => |standardize| {
                // Standardization adds derived scale columns while retaining the
                // input schema. It depends on the whole source column, so keep
                // predicate pruning but avoid unsafe projection pushdown until
                // derived-field schema metadata is available.
                projection_blocked = true;
                if (!nameInBorrowedList(standardize.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, standardize.name);
                break :op_loop;
            },
            .robust_profile => |robust| {
                // Robust profiles append median/MAD/IQR-derived columns while
                // preserving the input table. Keep predicate pruning but avoid
                // projection pushdown until derived-field schema tracking can
                // distinguish source and generated columns.
                projection_blocked = true;
                if (!nameInBorrowedList(robust.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, robust.name);
                break :op_loop;
            },
            .drawdown_profile => |drawdown| {
                // Drawdown profiles append sequence-derived columns while
                // preserving source fields. Keep scan predicates, but avoid
                // projection pushdown until source-vs-derived schema tracking is
                // rich enough to safely split later selects.
                projection_blocked = true;
                if (!nameInBorrowedList(drawdown.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, drawdown.name);
                break :op_loop;
            },
            .extrema_profile => |extrema| {
                projection_blocked = true;
                if (!nameInBorrowedList(extrema.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, extrema.name);
                break :op_loop;
            },
            .trend_profile => |trend| {
                // Trend profiles are row-order dependent and append several
                // derived columns. Preserve scan predicates, but block Parquet
                // projection pushdown until the planner can reason about
                // generated fields separately from source fields.
                projection_blocked = true;
                if (!nameInBorrowedList(trend.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, trend.name);
                break :op_loop;
            },
            .rolling_trend_profile => |trend| {
                projection_blocked = true;
                if (!nameInBorrowedList(trend.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, trend.name);
                break :op_loop;
            },
            .expanding_trend_profile => |trend| {
                projection_blocked = true;
                if (!nameInBorrowedList(trend.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, trend.name);
                break :op_loop;
            },
            .change_point_profile => |change| {
                // Change-point profiles are row-order dependent and append
                // derived jump diagnostics. Keep predicate pruning but block
                // projection across generated fields until schema tracking is
                // explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(change.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, change.name);
                break :op_loop;
            },
            .rolling_change_point_profile => |change| {
                // Rolling change-point profiles append aggregate jump
                // diagnostics over order-dependent deltas. Keep predicate
                // pruning but block projection across generated fields.
                projection_blocked = true;
                if (!nameInBorrowedList(change.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, change.name);
                break :op_loop;
            },
            .expanding_change_point_profile => |change| {
                projection_blocked = true;
                if (!nameInBorrowedList(change.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, change.name);
                break :op_loop;
            },
            .sign_profile => |sign| {
                projection_blocked = true;
                if (!nameInBorrowedList(sign.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sign.name);
                break :op_loop;
            },
            .rolling_sign_profile => |sign| {
                projection_blocked = true;
                if (!nameInBorrowedList(sign.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sign.name);
                break :op_loop;
            },
            .expanding_sign_profile => |sign| {
                projection_blocked = true;
                if (!nameInBorrowedList(sign.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, sign.name);
                break :op_loop;
            },
            .crossover_profile => |cross| {
                // Crossover profiles depend on two source columns and append
                // several signal columns. Keep scan predicates but block
                // projection pushdown until derived-field schema tracking can
                // safely split source and generated columns.
                projection_blocked = true;
                if (!nameInBorrowedList(cross.lhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.lhs_name);
                if (!nameInBorrowedList(cross.rhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.rhs_name);
                break :op_loop;
            },
            .rolling_crossover_profile => |cross| {
                // Rolling crossover profiles append window-level signal
                // summaries derived from two source columns. Predicate pruning
                // remains safe, but projection pushdown must wait until the
                // lazy planner tracks generated crossover fields explicitly.
                projection_blocked = true;
                if (!nameInBorrowedList(cross.lhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.lhs_name);
                if (!nameInBorrowedList(cross.rhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.rhs_name);
                break :op_loop;
            },
            .expanding_crossover_profile => |cross| {
                projection_blocked = true;
                if (!nameInBorrowedList(cross.lhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.lhs_name);
                if (!nameInBorrowedList(cross.rhs_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, cross.rhs_name);
                break :op_loop;
            },
            .bucket_profile => |bucket| {
                // Bucket profiles depend on the whole source distribution and
                // append several derived fields, so keep predicates but block
                // projection pushdown until generated-field schema metadata is
                // tracked explicitly.
                projection_blocked = true;
                if (!nameInBorrowedList(bucket.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, bucket.name);
                break :op_loop;
            },
            .ema_profile => |ema| {
                // EMA profiles are order-dependent and append derived columns,
                // so keep predicate pruning but block projection pushdown until
                // generated-field schema metadata is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(ema.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, ema.name);
                break :op_loop;
            },
            .linear_fit_profile => |fit| {
                // Linear-fit profiles depend on two source columns and append
                // model diagnostics. Keep predicate pruning but block projection
                // pushdown until generated-field schema metadata is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(fit.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.x_name);
                if (!nameInBorrowedList(fit.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.y_name);
                break :op_loop;
            },
            .error_profile => |err| {
                projection_blocked = true;
                if (!nameInBorrowedList(err.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.actual_name);
                if (!nameInBorrowedList(err.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.predicted_name);
                break :op_loop;
            },
            .rolling_error_profile => |err| {
                projection_blocked = true;
                if (!nameInBorrowedList(err.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.actual_name);
                if (!nameInBorrowedList(err.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.predicted_name);
                break :op_loop;
            },
            .expanding_error_profile => |err| {
                projection_blocked = true;
                if (!nameInBorrowedList(err.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.actual_name);
                if (!nameInBorrowedList(err.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, err.predicted_name);
                break :op_loop;
            },
            .classification_profile => |class| {
                projection_blocked = true;
                if (!nameInBorrowedList(class.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.actual_name);
                if (!nameInBorrowedList(class.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.predicted_name);
                break :op_loop;
            },
            .rolling_classification_profile => |class| {
                projection_blocked = true;
                if (!nameInBorrowedList(class.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.actual_name);
                if (!nameInBorrowedList(class.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.predicted_name);
                break :op_loop;
            },
            .expanding_classification_profile => |class| {
                projection_blocked = true;
                if (!nameInBorrowedList(class.actual_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.actual_name);
                if (!nameInBorrowedList(class.predicted_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, class.predicted_name);
                break :op_loop;
            },
            .bool_transition_profile => |transition| {
                projection_blocked = true;
                if (!nameInBorrowedList(transition.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, transition.name);
                break :op_loop;
            },
            .rolling_bool_transition_profile => |transition| {
                // Rolling transition summaries append window-level event-rate
                // columns and preserve the input table. Keep predicates, but
                // block projection pushdown until generated fields are tracked.
                projection_blocked = true;
                if (!nameInBorrowedList(transition.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, transition.name);
                break :op_loop;
            },
            .expanding_bool_transition_profile => |transition| {
                projection_blocked = true;
                if (!nameInBorrowedList(transition.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, transition.name);
                break :op_loop;
            },
            .rolling_correlation_profile => |corr| {
                // Rolling correlation profiles depend on two source columns and
                // append several window diagnostics. Keep predicate pruning but
                // block projection pushdown until generated-field schema
                // metadata is explicit.
                projection_blocked = true;
                if (!nameInBorrowedList(corr.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.x_name);
                if (!nameInBorrowedList(corr.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.y_name);
                break :op_loop;
            },
            .expanding_correlation_profile => |corr| {
                projection_blocked = true;
                if (!nameInBorrowedList(corr.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.x_name);
                if (!nameInBorrowedList(corr.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, corr.y_name);
                break :op_loop;
            },
            .expanding_linear_fit_profile => |fit| {
                projection_blocked = true;
                if (!nameInBorrowedList(fit.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.x_name);
                if (!nameInBorrowedList(fit.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.y_name);
                break :op_loop;
            },
            .rolling_linear_fit_profile => |fit| {
                // Rolling linear-fit profiles append window-local model fields
                // from two source columns. Preserve scan predicates but do not
                // push projection across generated regression diagnostics until
                // the planner can distinguish source and derived schemas.
                projection_blocked = true;
                if (!nameInBorrowedList(fit.x_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.x_name);
                if (!nameInBorrowedList(fit.y_name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, fit.y_name);
                break :op_loop;
            },
            .validity_profile => |validity| {
                // Validity profiles are schema-changing data-quality diagnostics
                // over one source column. Keep source dependency for scans and
                // avoid projection pushdown across generated validity fields.
                projection_blocked = true;
                if (!nameInBorrowedList(validity.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, validity.name);
                break :op_loop;
            },
            .rolling_validity_profile => |validity| {
                // Rolling validity profiles append data-quality window metrics
                // and preserve source fields. Keep scan predicates but avoid
                // unsafe projection across generated validity diagnostics.
                projection_blocked = true;
                if (!nameInBorrowedList(validity.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, validity.name);
                break :op_loop;
            },
            .expanding_validity_profile => |validity| {
                // Expanding validity profiles append cumulative quality metrics
                // and preserve source fields, so projection pushdown must wait
                // for derived-field schema tracking.
                projection_blocked = true;
                if (!nameInBorrowedList(validity.name, derived_names.items)) try appendOwnedNameUnique(allocator, &required_names, validity.name);
                break :op_loop;
            },
            .filter_mask, .head, .tail => {},
        }
    }

    const projection = if (saw_select and !projection_blocked) blk: {
        const owned = try required_names.toOwnedSlice(allocator);
        required_names = .empty;
        break :blk owned;
    } else null;
    if (projection == null) freeOwnedNameItems(allocator, required_names.items);
    required_names.deinit(allocator);

    const out = LazyScanPushdown{
        .allocator = allocator,
        .projection = projection,
        .range_predicate = range_predicate,
    };
    range_predicate = null;
    return out;
}

fn parquetRangePredicateFromScalar(scalar: DeviceScalar, op: DeviceColumnCompareOp) ?ParquetRangePredicate {
    return switch (scalar) {
        .bool => |value| blk: {
            const exact = switch (op) {
                .eq => value,
                .ne => !value,
                .gt, .ge, .lt, .le => break :blk null,
            };
            break :blk .{ .bool = .{ .min = exact, .max = exact } };
        },
        .i8 => |value| if (rangeFromScalarPredicate(i8, value, op)) |range| .{ .i8 = range } else null,
        .i16 => |value| if (rangeFromScalarPredicate(i16, value, op)) |range| .{ .i16 = range } else null,
        .i32 => |value| if (rangeFromScalarPredicate(i32, value, op)) |range| .{ .i32 = range } else null,
        .i64 => |value| if (rangeFromScalarPredicate(i64, value, op)) |range| .{ .i64 = range } else null,
        .u8 => |value| if (rangeFromScalarPredicate(u8, value, op)) |range| .{ .u8 = range } else null,
        .u16 => |value| if (rangeFromScalarPredicate(u16, value, op)) |range| .{ .u16 = range } else null,
        .u32 => |value| if (rangeFromScalarPredicate(u32, value, op)) |range| .{ .u32 = range } else null,
        .u64 => |value| if (rangeFromScalarPredicate(u64, value, op)) |range| .{ .u64 = range } else null,
        .usize => |value| if (rangeFromScalarPredicate(usize, value, op)) |range| .{ .usize = range } else null,
        .isize => |value| if (rangeFromScalarPredicate(isize, value, op)) |range| .{ .isize = range } else null,
        .f16 => |value| if (rangeFromScalarPredicate(f16, value, op)) |range| .{ .f16 = range } else null,
        .f32 => |value| if (rangeFromScalarPredicate(f32, value, op)) |range| .{ .f32 = range } else null,
        .f64 => |value| if (rangeFromScalarPredicate(f64, value, op)) |range| .{ .f64 = range } else null,
        .bf16, .c64, .c128 => null,
    };
}

fn rangeFromScalarPredicate(comptime T: type, value: T, op: DeviceColumnCompareOp) ?Range(T) {
    if (comptime @typeInfo(T) == .float) {
        if (std.math.isNan(value)) return null;
    }
    return switch (op) {
        .eq => .{ .min = value, .max = value },
        .gt, .ge => .{ .min = value },
        .lt, .le => .{ .max = value },
        .ne => null,
    };
}
