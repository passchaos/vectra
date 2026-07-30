//! Lazy dataframe frame/source execution machinery.
//!
//! `DeviceLazyFrame` is intentionally generic over the concrete eager
//! dataframe/column types so it can live outside `dataframe.zig` without an
//! import cycle. This keeps the public facade small while preserving all lazy
//! dataframe method signatures and ownership semantics.

const std = @import("std");
const array_mod = @import("array.zig");
const lazy_exec_mod = @import("dataframe_lazy_frame_exec.zig");
const lazy_expr_mod = @import("dataframe_lazy_expr_plan.zig");
const lazy_relation_methods_mod = @import("dataframe_lazy_relation_methods.zig");
const lazy_profile_methods_mod = @import("dataframe_lazy_profile_methods.zig");
const lazy_sort_mod = @import("dataframe_lazy_sort_plan.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const names_mod = @import("dataframe_names.zig");
const options_mod = @import("dataframe_options.zig");
const parquet_scan_mod = @import("dataframe_parquet_scan.zig");
const series_mod = @import("series.zig");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const ParquetInteropError = lazy_exec_mod.ParquetInteropError;
const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;
const freeOwnedNameItems = names_mod.freeOwnedNameItems;

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

            pub fn name(self: DeviceLazySource) []const u8 {
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
                return lazy_expr_mod.select(self, names);
            }

            pub fn withRowIndex(self: *DeviceLazyFrame, name: []const u8, offset: usize) DeviceDataError!void {
                return lazy_expr_mod.withRowIndex(self, name, offset);
            }

            pub fn renameColumn(self: *DeviceLazyFrame, old_name: []const u8, new_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.renameColumn(self, old_name, new_name);
            }

            pub fn dropColumns(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropColumns(self, names);
            }

            pub fn dropColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropColumn(self, name);
            }

            pub fn filter(self: *DeviceLazyFrame, mask: DeviceColumn) DeviceDataError!void {
                return lazy_expr_mod.filter(self, mask);
            }

            pub fn filterColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterColumn(self, name);
            }

            pub fn withColumnBinary(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnBinaryOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnBinary(self, name, lhs_name, rhs_name, op);
            }

            pub fn withColumnScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnScalar(self, name, input_name, T, scalar, op);
            }

            pub fn withColumnCompare(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnCompare(self, name, lhs_name, rhs_name, op);
            }

            pub fn withColumnCompareScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnCompareScalar(self, name, input_name, T, scalar, op);
            }

            pub const groupByCount = lazy_relation_methods_mod.groupByCount;
            pub const groupByValue = lazy_relation_methods_mod.groupByValue;
            pub const groupBySum = lazy_relation_methods_mod.groupBySum;
            pub const groupByMin = lazy_relation_methods_mod.groupByMin;
            pub const groupByMax = lazy_relation_methods_mod.groupByMax;
            pub const groupByMean = lazy_relation_methods_mod.groupByMean;
            pub const groupByStats = lazy_relation_methods_mod.groupByStats;
            pub const groupByStatsOn = lazy_relation_methods_mod.groupByStatsOn;
            pub const groupByProfile = lazy_relation_methods_mod.groupByProfile;
            pub const groupByProfileOn = lazy_relation_methods_mod.groupByProfileOn;
            pub const joinOn = lazy_relation_methods_mod.joinOn;
            pub const innerJoinOn = lazy_relation_methods_mod.innerJoinOn;
            pub const leftJoinOn = lazy_relation_methods_mod.leftJoinOn;
            pub const fullJoinOn = lazy_relation_methods_mod.fullJoinOn;
            pub const semiJoinOn = lazy_relation_methods_mod.semiJoinOn;
            pub const antiJoinOn = lazy_relation_methods_mod.antiJoinOn;
            pub const asofJoin = lazy_relation_methods_mod.asofJoin;
            pub const concatRows = lazy_relation_methods_mod.concatRows;
            pub const appendRows = lazy_relation_methods_mod.appendRows;
            pub const vstack = lazy_relation_methods_mod.vstack;
            pub const distinctRows = lazy_relation_methods_mod.distinctRows;
            pub const distinctOn = lazy_relation_methods_mod.distinctOn;
            pub const dropDuplicates = lazy_relation_methods_mod.dropDuplicates;
            pub const dropDuplicatesOn = lazy_relation_methods_mod.dropDuplicatesOn;
            pub const uniqueRows = lazy_relation_methods_mod.uniqueRows;
            pub fn filterColumnScalar(self: *DeviceLazyFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.filterColumnScalar(self, name, T, scalar, op);
            }

            pub fn sortBy(self: *DeviceLazyFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
                return lazy_sort_mod.sortBy(self, name, options_value);
            }

            pub fn rankProfileBy(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
                return lazy_sort_mod.rankProfileBy(self, name, output_prefix, options_value);
            }

            pub const rollingProfile = lazy_profile_methods_mod.rollingProfile;
            pub const rollingMomentProfile = lazy_profile_methods_mod.rollingMomentProfile;
            pub const rollingRangeProfile = lazy_profile_methods_mod.rollingRangeProfile;
            pub const rollingNormalizeProfile = lazy_profile_methods_mod.rollingNormalizeProfile;
            pub const expandingNormalizeProfile = lazy_profile_methods_mod.expandingNormalizeProfile;
            pub const rollingQuantileProfile = lazy_profile_methods_mod.rollingQuantileProfile;
            pub const expandingQuantileProfile = lazy_profile_methods_mod.expandingQuantileProfile;
            pub const rollingBoolProfile = lazy_profile_methods_mod.rollingBoolProfile;
            pub const rollingDrawdownProfile = lazy_profile_methods_mod.rollingDrawdownProfile;
            pub const rollingRobustProfile = lazy_profile_methods_mod.rollingRobustProfile;
            pub const rollingRankProfile = lazy_profile_methods_mod.rollingRankProfile;
            pub const lagProfile = lazy_profile_methods_mod.lagProfile;
            pub const leadProfile = lazy_profile_methods_mod.leadProfile;
            pub const clipProfile = lazy_profile_methods_mod.clipProfile;
            pub const rollingClipProfile = lazy_profile_methods_mod.rollingClipProfile;
            pub const expandingClipProfile = lazy_profile_methods_mod.expandingClipProfile;
            pub const thresholdProfile = lazy_profile_methods_mod.thresholdProfile;
            pub const rollingThresholdProfile = lazy_profile_methods_mod.rollingThresholdProfile;
            pub const expandingThresholdProfile = lazy_profile_methods_mod.expandingThresholdProfile;
            pub const expandingProfile = lazy_profile_methods_mod.expandingProfile;
            pub const expandingBoolProfile = lazy_profile_methods_mod.expandingBoolProfile;
            pub const expandingRankProfile = lazy_profile_methods_mod.expandingRankProfile;
            pub const expandingRobustProfile = lazy_profile_methods_mod.expandingRobustProfile;
            pub const expandingMomentProfile = lazy_profile_methods_mod.expandingMomentProfile;
            pub const standardizeProfile = lazy_profile_methods_mod.standardizeProfile;
            pub const robustProfile = lazy_profile_methods_mod.robustProfile;
            pub const drawdownProfile = lazy_profile_methods_mod.drawdownProfile;
            pub const extremaProfile = lazy_profile_methods_mod.extremaProfile;
            pub const trendProfile = lazy_profile_methods_mod.trendProfile;
            pub const rollingTrendProfile = lazy_profile_methods_mod.rollingTrendProfile;
            pub const expandingTrendProfile = lazy_profile_methods_mod.expandingTrendProfile;
            pub const changePointProfile = lazy_profile_methods_mod.changePointProfile;
            pub const rollingChangePointProfile = lazy_profile_methods_mod.rollingChangePointProfile;
            pub const expandingChangePointProfile = lazy_profile_methods_mod.expandingChangePointProfile;
            pub const signProfile = lazy_profile_methods_mod.signProfile;
            pub const rollingSignProfile = lazy_profile_methods_mod.rollingSignProfile;
            pub const expandingSignProfile = lazy_profile_methods_mod.expandingSignProfile;
            pub const crossoverProfile = lazy_profile_methods_mod.crossoverProfile;
            pub const rollingCrossoverProfile = lazy_profile_methods_mod.rollingCrossoverProfile;
            pub const expandingCrossoverProfile = lazy_profile_methods_mod.expandingCrossoverProfile;
            pub const bucketProfile = lazy_profile_methods_mod.bucketProfile;
            pub const emaProfile = lazy_profile_methods_mod.emaProfile;
            pub const linearFitProfile = lazy_profile_methods_mod.linearFitProfile;
            pub const errorProfile = lazy_profile_methods_mod.errorProfile;
            pub const rollingErrorProfile = lazy_profile_methods_mod.rollingErrorProfile;
            pub const expandingErrorProfile = lazy_profile_methods_mod.expandingErrorProfile;
            pub const classificationProfile = lazy_profile_methods_mod.classificationProfile;
            pub const rollingClassificationProfile = lazy_profile_methods_mod.rollingClassificationProfile;
            pub const expandingClassificationProfile = lazy_profile_methods_mod.expandingClassificationProfile;
            pub const boolTransitionProfile = lazy_profile_methods_mod.boolTransitionProfile;
            pub const rollingBoolTransitionProfile = lazy_profile_methods_mod.rollingBoolTransitionProfile;
            pub const expandingBoolTransitionProfile = lazy_profile_methods_mod.expandingBoolTransitionProfile;
            pub const rollingCorrelationProfile = lazy_profile_methods_mod.rollingCorrelationProfile;
            pub const expandingCorrelationProfile = lazy_profile_methods_mod.expandingCorrelationProfile;
            pub const expandingLinearFitProfile = lazy_profile_methods_mod.expandingLinearFitProfile;
            pub const rollingLinearFitProfile = lazy_profile_methods_mod.rollingLinearFitProfile;
            pub const validityProfile = lazy_profile_methods_mod.validityProfile;
            pub const rollingValidityProfile = lazy_profile_methods_mod.rollingValidityProfile;
            pub const expandingValidityProfile = lazy_profile_methods_mod.expandingValidityProfile;
            pub fn sliceRows(self: *DeviceLazyFrame, start: usize, stop: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .slice_rows = .{
                    .start = start,
                    .stop = stop,
                } });
            }

            pub fn slice(self: *DeviceLazyFrame, start: usize, len: usize) DeviceDataError!void {
                const stop = std.math.add(usize, start, len) catch return error.InvalidShape;
                return self.sliceRows(start, stop);
            }

            pub fn take(self: *DeviceLazyFrame, row_indices: []const usize) DeviceDataError!void {
                const owned = try self.allocator.dupe(usize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .take_rows = owned });
            }

            pub fn head(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .head = n });
            }

            pub fn tail(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .tail = n });
            }

            pub fn collect(self: DeviceLazyFrame) ParquetInteropError!DeviceDataFrame {
                return lazy_exec_mod.collect(DeviceDataFrame, DeviceLazyOp, self);
            }

            pub fn explain(self: DeviceLazyFrame, allocator: std.mem.Allocator) DeviceDataError![]u8 {
                return lazy_exec_mod.explain(DeviceLazyOp, self, allocator);
            }
        };

        fn deinitLazyOps(allocator: std.mem.Allocator, ops: *std.ArrayList(DeviceLazyOp)) void {
            for (ops.items) |*op| op.deinit(allocator);
            ops.deinit(allocator);
        }
    };
}
