//! Lazy dataframe frame/source execution machinery.
//!
//! `DeviceLazyFrame` is intentionally generic over the concrete eager
//! dataframe/column types so it can live outside `dataframe.zig` without an
//! import cycle. This keeps the public facade small while preserving all lazy
//! dataframe method signatures and ownership semantics.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_arrow_mod = @import("dataframe_arrow.zig");
const lazy_mod = @import("dataframe_lazy.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const names_mod = @import("dataframe_names.zig");
const options_mod = @import("dataframe_options.zig");
const parquet_scan_mod = @import("dataframe_parquet_scan.zig");
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
const DeviceLazyGroupByAggregation = lazy_op_mod.DeviceLazyGroupByAggregation;
const DeviceLazyJoinKind = lazy_op_mod.DeviceLazyJoinKind;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const ParquetInteropError = dataframe_arrow_mod.ParquetInteropError;
const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;
const freeOwnedNameItems = names_mod.freeOwnedNameItems;
const allNamesIn = names_mod.allNamesIn;
const planLazyScanPushdown = lazy_mod.planLazyScanPushdown;
const formatLazyScanPushdown = lazy_mod.formatLazyScanPushdown;
const formatLazyOp = lazy_mod.formatLazyOp;

pub fn DeviceLazyTypes(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
) type {
    const DeviceLazyOp = lazy_op_mod.DeviceLazyOp(DeviceDataFrame, DeviceColumn);

    return struct {
        pub const DeviceParquetScan = parquet_scan_mod.DeviceParquetScan(DeviceDataFrame, DeviceLazyFrame, DeviceColumnDef, DeviceColumn);

        pub const DeviceLazySource = union(enum) {
            dataframe: DeviceDataFrame,
            parquet_scan: DeviceParquetScan,

            fn deinit(self: *DeviceLazySource) void {
                switch (self.*) {
                    .dataframe => |*frame| frame.deinit(),
                    .parquet_scan => |*scan| scan.deinit(),
                }
                self.* = undefined;
            }

            fn clone(self: DeviceLazySource) DeviceDataError!DeviceLazySource {
                return switch (self) {
                    .dataframe => |frame| .{ .dataframe = try frame.clone() },
                    .parquet_scan => |scan| .{ .parquet_scan = try scan.clone() },
                };
            }

            fn name(self: DeviceLazySource) []const u8 {
                return switch (self) {
                    .dataframe => "dataframe",
                    .parquet_scan => "parquet_scan",
                };
            }
        };

        /// A compact eager-backed lazy plan for `DeviceDataFrame`.
        ///
        /// Polars' lazy API is valuable because it gives the planner a concrete list of
        /// projections, filters, and ordering operations before execution.  Vectra keeps
        /// the plan small and still executes through the existing `DeviceDataFrame`
        /// methods in `collect()`, but scan sources are represented explicitly so the
        /// planner can push conservative Parquet row-group pruning and column projection
        /// toward Boltha before materializing CPU/CUDA/MPS columns.  That gives callers a
        /// stable API today and gives Axiom a single future lowering boundary for
        /// fusing/reordering dataframe operations across CPU/CUDA/MPS.
        pub const DeviceLazyFrame = struct {
            allocator: std.mem.Allocator,
            source: DeviceLazySource,
            ops: std.ArrayList(DeviceLazyOp) = .empty,

            pub fn init(allocator: std.mem.Allocator, source: DeviceDataFrame) DeviceDataError!DeviceLazyFrame {
                return .{
                    .allocator = allocator,
                    .source = .{ .dataframe = try source.clone() },
                };
            }

            pub fn initParquetScan(allocator: std.mem.Allocator, scan: DeviceParquetScan) DeviceDataError!DeviceLazyFrame {
                return .{
                    .allocator = allocator,
                    .source = .{ .parquet_scan = try scan.clone() },
                };
            }

            pub fn scanParquetBytes(allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) DeviceDataError!DeviceLazyFrame {
                return .{
                    .allocator = allocator,
                    .source = .{ .parquet_scan = try DeviceParquetScan.init(allocator, bytes, device_value) },
                };
            }

            pub fn clone(self: DeviceLazyFrame) DeviceDataError!DeviceLazyFrame {
                var cloned = DeviceLazyFrame{
                    .allocator = self.allocator,
                    .source = try self.source.clone(),
                };
                errdefer cloned.source.deinit();
                errdefer deinitLazyOps(self.allocator, &cloned.ops);
                for (self.ops.items) |op| {
                    var cloned_op = try op.clone(self.allocator);
                    errdefer cloned_op.deinit(self.allocator);
                    try cloned.ops.append(self.allocator, cloned_op);
                }
                return cloned;
            }

            pub fn deinit(self: *DeviceLazyFrame) void {
                self.source.deinit();
                for (self.ops.items) |*op| op.deinit(self.allocator);
                self.ops.deinit(self.allocator);
                self.* = undefined;
            }

            pub fn select(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                const owned = try self.allocator.alloc([]const u8, names.len);
                errdefer self.allocator.free(owned);
                var initialized: usize = 0;
                errdefer {
                    for (owned[0..initialized]) |name| self.allocator.free(name);
                }
                for (names, owned) |name, *slot| {
                    slot.* = try self.allocator.dupe(u8, name);
                    initialized += 1;
                }
                try self.ops.append(self.allocator, .{ .select = owned });
            }

            pub fn filter(self: *DeviceLazyFrame, mask: DeviceColumn) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .filter_mask = try mask.clone() });
            }

            pub fn withColumnBinary(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnBinaryOp) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_lhs = try self.allocator.dupe(u8, lhs_name);
                errdefer self.allocator.free(owned_lhs);
                const owned_rhs = try self.allocator.dupe(u8, rhs_name);
                errdefer self.allocator.free(owned_rhs);
                try self.ops.append(self.allocator, .{ .with_column_binary = .{
                    .name = owned_name,
                    .lhs_name = owned_lhs,
                    .rhs_name = owned_rhs,
                    .op = op,
                } });
            }

            pub fn withColumnScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_input = try self.allocator.dupe(u8, input_name);
                errdefer self.allocator.free(owned_input);
                try self.ops.append(self.allocator, .{ .with_column_scalar = .{
                    .name = owned_name,
                    .input_name = owned_input,
                    .op = op,
                    .scalar = DeviceScalar.init(T, scalar),
                } });
            }

            pub fn withColumnCompare(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_lhs = try self.allocator.dupe(u8, lhs_name);
                errdefer self.allocator.free(owned_lhs);
                const owned_rhs = try self.allocator.dupe(u8, rhs_name);
                errdefer self.allocator.free(owned_rhs);
                try self.ops.append(self.allocator, .{ .with_column_compare = .{
                    .name = owned_name,
                    .lhs_name = owned_lhs,
                    .rhs_name = owned_rhs,
                    .op = op,
                } });
            }

            pub fn withColumnCompareScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_input = try self.allocator.dupe(u8, input_name);
                errdefer self.allocator.free(owned_input);
                try self.ops.append(self.allocator, .{ .with_column_compare_scalar = .{
                    .name = owned_name,
                    .input_name = owned_input,
                    .op = op,
                    .scalar = DeviceScalar.init(T, scalar),
                } });
            }

            pub fn groupByCount(self: *DeviceLazyFrame, key_name: []const u8, output_name: []const u8) DeviceDataError!void {
                const owned_key = try self.allocator.dupe(u8, key_name);
                errdefer self.allocator.free(owned_key);
                const owned_output = try self.allocator.dupe(u8, output_name);
                errdefer self.allocator.free(owned_output);
                try self.ops.append(self.allocator, .{ .group_by_count = .{
                    .key_name = owned_key,
                    .output_name = owned_output,
                } });
            }

            pub fn groupByValue(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8, aggregation: DeviceLazyGroupByAggregation) DeviceDataError!void {
                const owned_key = try self.allocator.dupe(u8, key_name);
                errdefer self.allocator.free(owned_key);
                const owned_value = try self.allocator.dupe(u8, value_name);
                errdefer self.allocator.free(owned_value);
                const owned_output = try self.allocator.dupe(u8, output_name);
                errdefer self.allocator.free(owned_output);
                try self.ops.append(self.allocator, .{ .group_by_value = .{
                    .key_name = owned_key,
                    .value_name = owned_value,
                    .output_name = owned_output,
                    .aggregation = aggregation,
                } });
            }

            pub fn groupBySum(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
                return self.groupByValue(key_name, value_name, output_name, .sum);
            }

            pub fn groupByMin(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
                return self.groupByValue(key_name, value_name, output_name, .min);
            }

            pub fn groupByMax(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
                return self.groupByValue(key_name, value_name, output_name, .max);
            }

            pub fn groupByMean(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!void {
                return self.groupByValue(key_name, value_name, output_name, .mean);
            }

            pub fn groupByStats(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
                const owned_key = try self.allocator.dupe(u8, key_name);
                errdefer self.allocator.free(owned_key);
                const owned_value = try self.allocator.dupe(u8, value_name);
                errdefer self.allocator.free(owned_value);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .group_by_stats = .{
                    .key_name = owned_key,
                    .value_name = owned_value,
                    .output_prefix = owned_prefix,
                } });
            }

            pub fn groupByStatsOn(self: *DeviceLazyFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
                const owned_keys = try cloneNameList(self.allocator, key_names);
                errdefer freeNameList(self.allocator, owned_keys);
                const owned_value = try self.allocator.dupe(u8, value_name);
                errdefer self.allocator.free(owned_value);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .group_by_stats_on = .{
                    .key_names = owned_keys,
                    .value_name = owned_value,
                    .output_prefix = owned_prefix,
                } });
            }

            pub fn groupByProfile(self: *DeviceLazyFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
                const owned_key = try self.allocator.dupe(u8, key_name);
                errdefer self.allocator.free(owned_key);
                const owned_value = try self.allocator.dupe(u8, value_name);
                errdefer self.allocator.free(owned_value);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .group_by_profile = .{
                    .key_name = owned_key,
                    .value_name = owned_value,
                    .output_prefix = owned_prefix,
                } });
            }

            pub fn groupByProfileOn(self: *DeviceLazyFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!void {
                const owned_keys = try cloneNameList(self.allocator, key_names);
                errdefer freeNameList(self.allocator, owned_keys);
                const owned_value = try self.allocator.dupe(u8, value_name);
                errdefer self.allocator.free(owned_value);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .group_by_profile_on = .{
                    .key_names = owned_keys,
                    .value_name = owned_value,
                    .output_prefix = owned_prefix,
                } });
            }

            pub fn joinOn(
                self: *DeviceLazyFrame,
                right: DeviceDataFrame,
                left_key_names: []const []const u8,
                right_key_names: []const []const u8,
                kind: DeviceLazyJoinKind,
                options_value: DeviceJoinOptions,
            ) DeviceDataError!void {
                if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
                var owned_right = try right.clone();
                errdefer owned_right.deinit();
                const owned_left_keys = try cloneNameList(self.allocator, left_key_names);
                errdefer freeNameList(self.allocator, owned_left_keys);
                const owned_right_keys = try cloneNameList(self.allocator, right_key_names);
                errdefer freeNameList(self.allocator, owned_right_keys);
                const owned_suffix = try self.allocator.dupe(u8, options_value.right_suffix);
                errdefer self.allocator.free(owned_suffix);
                try self.ops.append(self.allocator, .{ .join_on = .{
                    .kind = kind,
                    .right = owned_right,
                    .left_key_names = owned_left_keys,
                    .right_key_names = owned_right_keys,
                    .options = .{ .right_suffix = owned_suffix },
                } });
            }

            pub fn innerJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
                return self.joinOn(right, left_key_names, right_key_names, .inner, options_value);
            }

            pub fn leftJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
                return self.joinOn(right, left_key_names, right_key_names, .left, options_value);
            }

            pub fn fullJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8, options_value: DeviceJoinOptions) DeviceDataError!void {
                return self.joinOn(right, left_key_names, right_key_names, .full, options_value);
            }

            pub fn semiJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8) DeviceDataError!void {
                return self.joinOn(right, left_key_names, right_key_names, .semi, .{});
            }

            pub fn antiJoinOn(self: *DeviceLazyFrame, right: DeviceDataFrame, left_key_names: []const []const u8, right_key_names: []const []const u8) DeviceDataError!void {
                return self.joinOn(right, left_key_names, right_key_names, .anti, .{});
            }

            pub fn asofJoin(
                self: *DeviceLazyFrame,
                right: DeviceDataFrame,
                left_key_name: []const u8,
                right_key_name: []const u8,
                options_value: DeviceAsofOptions,
            ) DeviceDataError!void {
                var owned_right = try right.clone();
                errdefer owned_right.deinit();
                const owned_left_key = try self.allocator.dupe(u8, left_key_name);
                errdefer self.allocator.free(owned_left_key);
                const owned_right_key = try self.allocator.dupe(u8, right_key_name);
                errdefer self.allocator.free(owned_right_key);
                const owned_suffix = try self.allocator.dupe(u8, options_value.right_suffix);
                errdefer self.allocator.free(owned_suffix);
                try self.ops.append(self.allocator, .{ .asof_join = .{
                    .right = owned_right,
                    .left_key_name = owned_left_key,
                    .right_key_name = owned_right_key,
                    .options = .{
                        .strategy = options_value.strategy,
                        .right_suffix = owned_suffix,
                    },
                } });
            }

            pub fn concatRows(self: *DeviceLazyFrame, right: DeviceDataFrame) DeviceDataError!void {
                var owned_right = try right.clone();
                errdefer owned_right.deinit();
                try self.ops.append(self.allocator, .{ .concat_rows = owned_right });
            }

            pub fn appendRows(self: *DeviceLazyFrame, right: DeviceDataFrame) DeviceDataError!void {
                return self.concatRows(right);
            }

            pub fn vstack(self: *DeviceLazyFrame, right: DeviceDataFrame) DeviceDataError!void {
                return self.concatRows(right);
            }

            pub fn distinctRows(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .distinct_rows = {} });
            }

            pub fn distinctOn(self: *DeviceLazyFrame, key_names: []const []const u8) DeviceDataError!void {
                if (key_names.len == 0) return error.LengthMismatch;
                try self.ops.append(self.allocator, .{ .distinct_on = try cloneNameList(self.allocator, key_names) });
            }

            pub fn dropDuplicates(self: *DeviceLazyFrame) DeviceDataError!void {
                return self.distinctRows();
            }

            pub fn dropDuplicatesOn(self: *DeviceLazyFrame, key_names: []const []const u8) DeviceDataError!void {
                return self.distinctOn(key_names);
            }

            pub fn uniqueRows(self: *DeviceLazyFrame) DeviceDataError!void {
                return self.distinctRows();
            }

            pub fn filterColumnScalar(self: *DeviceLazyFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .filter_scalar = .{
                    .name = try self.allocator.dupe(u8, name),
                    .op = op,
                    .scalar = DeviceScalar.init(T, scalar),
                } });
            }

            pub fn sortBy(self: *DeviceLazyFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .sort_by = .{
                    .name = try self.allocator.dupe(u8, name),
                    .options = options_value,
                } });
            }

            pub fn rankProfileBy(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rank_profile_by = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingMomentProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_moment_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingRangeProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_range_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingNormalizeProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_normalize_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingNormalizeProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_normalize_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingQuantileProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_quantile_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingQuantileProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_quantile_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingBoolProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_bool_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingDrawdownProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_drawdown_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingRobustProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingRobustOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_robust_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingRankProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingRankOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_rank_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn lagProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .lag_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn leadProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .lead_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn clipProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceClipOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .clip_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingClipProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, clip_options: DeviceClipOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_clip_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .clip_options = clip_options,
                    .options = options_value,
                } });
            }

            pub fn expandingClipProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, clip_options: DeviceClipOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_clip_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .clip_options = clip_options,
                    .options = options_value,
                } });
            }

            pub fn thresholdProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceThresholdOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .threshold_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingThresholdProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_threshold_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .threshold = threshold,
                    .options = options_value,
                } });
            }

            pub fn expandingThresholdProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_threshold_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .threshold = threshold,
                    .options = options_value,
                } });
            }

            pub fn expandingProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingBoolProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_bool_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingRankProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingRankOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_rank_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingRobustProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_robust_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingMomentProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_moment_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn standardizeProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceStandardizeOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .standardize_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn robustProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .robust_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn drawdownProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceDrawdownOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .drawdown_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn extremaProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExtremaOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .extrema_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn trendProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .trend_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingTrendProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_trend_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .trend_options = trend_options,
                    .options = options_value,
                } });
            }

            pub fn expandingTrendProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_trend_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .trend_options = trend_options,
                    .options = options_value,
                } });
            }

            pub fn changePointProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceTrendOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .change_point_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .threshold = threshold,
                    .options = options_value,
                } });
            }

            pub fn rollingChangePointProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_change_point_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .threshold = threshold,
                    .change_options = change_options,
                    .options = options_value,
                } });
            }

            pub fn expandingChangePointProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_change_point_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .threshold = threshold,
                    .change_options = change_options,
                    .options = options_value,
                } });
            }

            pub fn signProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .sign_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingSignProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_sign_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .sign_options = sign_options,
                    .options = options_value,
                } });
            }

            pub fn expandingSignProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_sign_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .sign_options = sign_options,
                    .options = options_value,
                } });
            }

            pub fn crossoverProfile(
                self: *DeviceLazyFrame,
                lhs_name: []const u8,
                rhs_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceCrossoverOptions,
            ) DeviceDataError!void {
                const owned_lhs = try self.allocator.dupe(u8, lhs_name);
                errdefer self.allocator.free(owned_lhs);
                const owned_rhs = try self.allocator.dupe(u8, rhs_name);
                errdefer self.allocator.free(owned_rhs);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .crossover_profile = .{
                    .lhs_name = owned_lhs,
                    .rhs_name = owned_rhs,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingCrossoverProfile(
                self: *DeviceLazyFrame,
                lhs_name: []const u8,
                rhs_name: []const u8,
                output_prefix: []const u8,
                cross_options: DeviceCrossoverOptions,
                options_value: DeviceRollingOptions,
            ) DeviceDataError!void {
                const owned_lhs = try self.allocator.dupe(u8, lhs_name);
                errdefer self.allocator.free(owned_lhs);
                const owned_rhs = try self.allocator.dupe(u8, rhs_name);
                errdefer self.allocator.free(owned_rhs);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_crossover_profile = .{
                    .lhs_name = owned_lhs,
                    .rhs_name = owned_rhs,
                    .output_prefix = owned_prefix,
                    .cross_options = cross_options,
                    .options = options_value,
                } });
            }

            pub fn expandingCrossoverProfile(
                self: *DeviceLazyFrame,
                lhs_name: []const u8,
                rhs_name: []const u8,
                output_prefix: []const u8,
                cross_options: DeviceCrossoverOptions,
                options_value: DeviceExpandingOptions,
            ) DeviceDataError!void {
                const owned_lhs = try self.allocator.dupe(u8, lhs_name);
                errdefer self.allocator.free(owned_lhs);
                const owned_rhs = try self.allocator.dupe(u8, rhs_name);
                errdefer self.allocator.free(owned_rhs);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_crossover_profile = .{
                    .lhs_name = owned_lhs,
                    .rhs_name = owned_rhs,
                    .output_prefix = owned_prefix,
                    .cross_options = cross_options,
                    .options = options_value,
                } });
            }

            pub fn bucketProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceBucketOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .bucket_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn emaProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceEmaOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .ema_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn linearFitProfile(
                self: *DeviceLazyFrame,
                x_name: []const u8,
                y_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceLinearFitOptions,
            ) DeviceDataError!void {
                const owned_x = try self.allocator.dupe(u8, x_name);
                errdefer self.allocator.free(owned_x);
                const owned_y = try self.allocator.dupe(u8, y_name);
                errdefer self.allocator.free(owned_y);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .linear_fit_profile = .{
                    .x_name = owned_x,
                    .y_name = owned_y,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn errorProfile(
                self: *DeviceLazyFrame,
                actual_name: []const u8,
                predicted_name: []const u8,
                output_prefix: []const u8,
            ) DeviceDataError!void {
                const owned_actual = try self.allocator.dupe(u8, actual_name);
                errdefer self.allocator.free(owned_actual);
                const owned_predicted = try self.allocator.dupe(u8, predicted_name);
                errdefer self.allocator.free(owned_predicted);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .error_profile = .{
                    .actual_name = owned_actual,
                    .predicted_name = owned_predicted,
                    .output_prefix = owned_prefix,
                } });
            }

            pub fn rollingErrorProfile(
                self: *DeviceLazyFrame,
                actual_name: []const u8,
                predicted_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceRollingOptions,
            ) DeviceDataError!void {
                const owned_actual = try self.allocator.dupe(u8, actual_name);
                errdefer self.allocator.free(owned_actual);
                const owned_predicted = try self.allocator.dupe(u8, predicted_name);
                errdefer self.allocator.free(owned_predicted);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_error_profile = .{
                    .actual_name = owned_actual,
                    .predicted_name = owned_predicted,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingErrorProfile(
                self: *DeviceLazyFrame,
                actual_name: []const u8,
                predicted_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceExpandingOptions,
            ) DeviceDataError!void {
                const owned_actual = try self.allocator.dupe(u8, actual_name);
                errdefer self.allocator.free(owned_actual);
                const owned_predicted = try self.allocator.dupe(u8, predicted_name);
                errdefer self.allocator.free(owned_predicted);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_error_profile = .{
                    .actual_name = owned_actual,
                    .predicted_name = owned_predicted,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn classificationProfile(
                self: *DeviceLazyFrame,
                actual_name: []const u8,
                predicted_name: []const u8,
                output_prefix: []const u8,
            ) DeviceDataError!void {
                const owned_actual = try self.allocator.dupe(u8, actual_name);
                errdefer self.allocator.free(owned_actual);
                const owned_predicted = try self.allocator.dupe(u8, predicted_name);
                errdefer self.allocator.free(owned_predicted);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .classification_profile = .{
                    .actual_name = owned_actual,
                    .predicted_name = owned_predicted,
                    .output_prefix = owned_prefix,
                } });
            }

            pub fn rollingClassificationProfile(
                self: *DeviceLazyFrame,
                actual_name: []const u8,
                predicted_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceRollingOptions,
            ) DeviceDataError!void {
                const owned_actual = try self.allocator.dupe(u8, actual_name);
                errdefer self.allocator.free(owned_actual);
                const owned_predicted = try self.allocator.dupe(u8, predicted_name);
                errdefer self.allocator.free(owned_predicted);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_classification_profile = .{
                    .actual_name = owned_actual,
                    .predicted_name = owned_predicted,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingClassificationProfile(
                self: *DeviceLazyFrame,
                actual_name: []const u8,
                predicted_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceExpandingOptions,
            ) DeviceDataError!void {
                const owned_actual = try self.allocator.dupe(u8, actual_name);
                errdefer self.allocator.free(owned_actual);
                const owned_predicted = try self.allocator.dupe(u8, predicted_name);
                errdefer self.allocator.free(owned_predicted);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_classification_profile = .{
                    .actual_name = owned_actual,
                    .predicted_name = owned_predicted,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn boolTransitionProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .bool_transition_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingBoolTransitionProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_bool_transition_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .transition_options = transition_options,
                    .options = options_value,
                } });
            }

            pub fn expandingBoolTransitionProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_bool_transition_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .transition_options = transition_options,
                    .options = options_value,
                } });
            }

            pub fn rollingCorrelationProfile(
                self: *DeviceLazyFrame,
                x_name: []const u8,
                y_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceRollingCorrelationOptions,
            ) DeviceDataError!void {
                const owned_x = try self.allocator.dupe(u8, x_name);
                errdefer self.allocator.free(owned_x);
                const owned_y = try self.allocator.dupe(u8, y_name);
                errdefer self.allocator.free(owned_y);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_correlation_profile = .{
                    .x_name = owned_x,
                    .y_name = owned_y,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingCorrelationProfile(
                self: *DeviceLazyFrame,
                x_name: []const u8,
                y_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceExpandingOptions,
            ) DeviceDataError!void {
                const owned_x = try self.allocator.dupe(u8, x_name);
                errdefer self.allocator.free(owned_x);
                const owned_y = try self.allocator.dupe(u8, y_name);
                errdefer self.allocator.free(owned_y);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_correlation_profile = .{
                    .x_name = owned_x,
                    .y_name = owned_y,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingLinearFitProfile(
                self: *DeviceLazyFrame,
                x_name: []const u8,
                y_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceExpandingOptions,
            ) DeviceDataError!void {
                const owned_x = try self.allocator.dupe(u8, x_name);
                errdefer self.allocator.free(owned_x);
                const owned_y = try self.allocator.dupe(u8, y_name);
                errdefer self.allocator.free(owned_y);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_linear_fit_profile = .{
                    .x_name = owned_x,
                    .y_name = owned_y,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn rollingLinearFitProfile(
                self: *DeviceLazyFrame,
                x_name: []const u8,
                y_name: []const u8,
                output_prefix: []const u8,
                options_value: DeviceRollingCorrelationOptions,
            ) DeviceDataError!void {
                const owned_x = try self.allocator.dupe(u8, x_name);
                errdefer self.allocator.free(owned_x);
                const owned_y = try self.allocator.dupe(u8, y_name);
                errdefer self.allocator.free(owned_y);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_linear_fit_profile = .{
                    .x_name = owned_x,
                    .y_name = owned_y,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn validityProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .validity_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                } });
            }

            pub fn rollingValidityProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .rolling_validity_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn expandingValidityProfile(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!void {
                const owned_name = try self.allocator.dupe(u8, name);
                errdefer self.allocator.free(owned_name);
                const owned_prefix = try self.allocator.dupe(u8, output_prefix);
                errdefer self.allocator.free(owned_prefix);
                try self.ops.append(self.allocator, .{ .expanding_validity_profile = .{
                    .name = owned_name,
                    .output_prefix = owned_prefix,
                    .options = options_value,
                } });
            }

            pub fn head(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .head = n });
            }

            pub fn tail(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .tail = n });
            }

            pub fn collect(self: DeviceLazyFrame) ParquetInteropError!DeviceDataFrame {
                var optimized = try self.optimizedOps();
                defer deinitLazyOps(self.allocator, &optimized);
                var current = try self.collectSource(optimized.items);
                errdefer current.deinit();
                for (optimized.items) |op| {
                    const next = switch (op) {
                        .select => |names| try current.select(names),
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
                        .head => |n| try current.head(n),
                        .tail => |n| try current.tail(n),
                    };
                    current.deinit();
                    current = next;
                }
                return current;
            }

            pub fn explain(self: DeviceLazyFrame, allocator: std.mem.Allocator) DeviceDataError![]u8 {
                var optimized = try self.optimizedOps();
                defer deinitLazyOps(self.allocator, &optimized);
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

            fn optimizedOps(self: DeviceLazyFrame) DeviceDataError!std.ArrayList(DeviceLazyOp) {
                var optimized: std.ArrayList(DeviceLazyOp) = .empty;
                errdefer deinitLazyOps(self.allocator, &optimized);
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

            fn collectSource(self: DeviceLazyFrame, ops: []const DeviceLazyOp) ParquetInteropError!DeviceDataFrame {
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
        };

        fn deinitLazyOps(allocator: std.mem.Allocator, ops: *std.ArrayList(DeviceLazyOp)) void {
            for (ops.items) |*op| op.deinit(allocator);
            ops.deinit(allocator);
        }
    };
}
