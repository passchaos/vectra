const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_arrow_mod = @import("dataframe_arrow.zig");
const dataframe_column_mod = @import("dataframe_column.zig");
const dataframe_host_mod = @import("dataframe_host.zig");
const options_mod = @import("dataframe_options.zig");
const dataframe_view_mod = @import("dataframe_view.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const keys_mod = @import("dataframe_keys.zig");
const join_mod = @import("dataframe_join.zig");
const lazy_mod = @import("dataframe_lazy.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const parquet_scan_mod = @import("dataframe_parquet_scan.zig");
const csv_mod = @import("dataframe_csv.zig");
const boltha = @import("boltha");
const bool_transition_mod = @import("dataframe_bool_transition.zig");
const BoolTransitionProfileColumnCount = bool_transition_mod.BoolTransitionProfileColumnCount;
const boolTransitionProfileOutputNames = bool_transition_mod.boolTransitionProfileOutputNames;
const RollingBoolTransitionProfileColumnCount = bool_transition_mod.RollingBoolTransitionProfileColumnCount;
const rollingBoolTransitionProfileOutputNames = bool_transition_mod.rollingBoolTransitionProfileOutputNames;
const ExpandingBoolTransitionProfileColumnCount = bool_transition_mod.ExpandingBoolTransitionProfileColumnCount;
const expandingBoolTransitionProfileOutputNames = bool_transition_mod.expandingBoolTransitionProfileOutputNames;
const boolTransitionProfileColumns = bool_transition_mod.boolTransitionProfileColumns;
const rollingBoolTransitionProfileColumns = bool_transition_mod.rollingBoolTransitionProfileColumns;
const expandingBoolTransitionProfileColumns = bool_transition_mod.expandingBoolTransitionProfileColumns;
const classification_mod = @import("dataframe_classification.zig");
const ClassificationProfileColumnCount = classification_mod.ClassificationProfileColumnCount;
const classificationProfileOutputNames = classification_mod.classificationProfileOutputNames;
const RollingClassificationProfileColumnCount = classification_mod.RollingClassificationProfileColumnCount;
const rollingClassificationProfileOutputNames = classification_mod.rollingClassificationProfileOutputNames;
const ExpandingClassificationProfileColumnCount = classification_mod.ExpandingClassificationProfileColumnCount;
const expandingClassificationProfileOutputNames = classification_mod.expandingClassificationProfileOutputNames;
const classificationProfileColumns = classification_mod.classificationProfileColumns;
const rollingClassificationProfileColumns = classification_mod.rollingClassificationProfileColumns;
const expandingClassificationProfileColumns = classification_mod.expandingClassificationProfileColumns;
const error_mod = @import("dataframe_error.zig");
const ErrorProfileColumnCount = error_mod.ErrorProfileColumnCount;
const errorProfileOutputNames = error_mod.errorProfileOutputNames;
const RollingErrorProfileColumnCount = error_mod.RollingErrorProfileColumnCount;
const rollingErrorProfileOutputNames = error_mod.rollingErrorProfileOutputNames;
const ExpandingErrorProfileColumnCount = error_mod.ExpandingErrorProfileColumnCount;
const expandingErrorProfileOutputNames = error_mod.expandingErrorProfileOutputNames;
const errorProfileColumnsByValue = error_mod.errorProfileColumnsByValue;
const rollingErrorProfileColumnsByValue = error_mod.rollingErrorProfileColumnsByValue;
const expandingErrorProfileColumnsByValue = error_mod.expandingErrorProfileColumnsByValue;
const correlation_mod = @import("dataframe_correlation.zig");
const RollingCorrelationProfileColumnCount = correlation_mod.RollingCorrelationProfileColumnCount;
const rollingCorrelationProfileOutputNames = correlation_mod.rollingCorrelationProfileOutputNames;
const ExpandingCorrelationProfileColumnCount = correlation_mod.ExpandingCorrelationProfileColumnCount;
const expandingCorrelationProfileOutputNames = correlation_mod.expandingCorrelationProfileOutputNames;
const rollingCorrelationProfileColumnsByValue = correlation_mod.rollingCorrelationProfileColumnsByValue;
const expandingCorrelationProfileColumnsByValue = correlation_mod.expandingCorrelationProfileColumnsByValue;
const linear_fit_mod = @import("dataframe_linear_fit.zig");
const LinearFitProfileColumnCount = linear_fit_mod.LinearFitProfileColumnCount;
const linearFitProfileOutputNames = linear_fit_mod.linearFitProfileOutputNames;
const ExpandingLinearFitProfileColumnCount = linear_fit_mod.ExpandingLinearFitProfileColumnCount;
const expandingLinearFitProfileOutputNames = linear_fit_mod.expandingLinearFitProfileOutputNames;
const RollingLinearFitProfileColumnCount = linear_fit_mod.RollingLinearFitProfileColumnCount;
const rollingLinearFitProfileOutputNames = linear_fit_mod.rollingLinearFitProfileOutputNames;
const linearFitProfileColumnsByValue = linear_fit_mod.linearFitProfileColumnsByValue;
const expandingLinearFitProfileColumnsByValue = linear_fit_mod.expandingLinearFitProfileColumnsByValue;
const rollingLinearFitProfileColumnsByValue = linear_fit_mod.rollingLinearFitProfileColumnsByValue;
const crossover_mod = @import("dataframe_crossover.zig");
const CrossoverProfileColumnCount = crossover_mod.CrossoverProfileColumnCount;
const crossoverProfileOutputNames = crossover_mod.crossoverProfileOutputNames;
const RollingCrossoverProfileColumnCount = crossover_mod.RollingCrossoverProfileColumnCount;
const rollingCrossoverProfileOutputNames = crossover_mod.rollingCrossoverProfileOutputNames;
const ExpandingCrossoverProfileColumnCount = crossover_mod.ExpandingCrossoverProfileColumnCount;
const expandingCrossoverProfileOutputNames = crossover_mod.expandingCrossoverProfileOutputNames;
const crossoverProfileColumnsByValue = crossover_mod.crossoverProfileColumnsByValue;
const rollingCrossoverProfileColumnsByValue = crossover_mod.rollingCrossoverProfileColumnsByValue;
const expandingCrossoverProfileColumnsByValue = crossover_mod.expandingCrossoverProfileColumnsByValue;
const threshold_mod = @import("dataframe_threshold.zig");
const ThresholdProfileColumnCount = threshold_mod.ThresholdProfileColumnCount;
const thresholdProfileOutputNames = threshold_mod.thresholdProfileOutputNames;
const RollingThresholdProfileColumnCount = threshold_mod.RollingThresholdProfileColumnCount;
const rollingThresholdProfileOutputNames = threshold_mod.rollingThresholdProfileOutputNames;
const ExpandingThresholdProfileColumnCount = threshold_mod.ExpandingThresholdProfileColumnCount;
const expandingThresholdProfileOutputNames = threshold_mod.expandingThresholdProfileOutputNames;
const thresholdProfileColumnsByValue = threshold_mod.thresholdProfileColumnsByValue;
const rollingThresholdProfileColumnsByValue = threshold_mod.rollingThresholdProfileColumnsByValue;
const expandingThresholdProfileColumnsByValue = threshold_mod.expandingThresholdProfileColumnsByValue;
const validity_mod = @import("dataframe_validity.zig");
const bool_profile_mod = @import("dataframe_bool_profile.zig");
const RollingBoolProfileColumnCount = bool_profile_mod.RollingBoolProfileColumnCount;
const rollingBoolProfileOutputNames = bool_profile_mod.rollingBoolProfileOutputNames;
const ExpandingBoolProfileColumnCount = bool_profile_mod.ExpandingBoolProfileColumnCount;
const expandingBoolProfileOutputNames = bool_profile_mod.expandingBoolProfileOutputNames;
const rollingBoolProfileColumns = bool_profile_mod.rollingBoolProfileColumns;
const expandingBoolProfileColumns = bool_profile_mod.expandingBoolProfileColumns;
const clip_mod = @import("dataframe_clip.zig");
const ClipProfileColumnCount = clip_mod.ClipProfileColumnCount;
const clipProfileOutputNames = clip_mod.clipProfileOutputNames;
const RollingClipProfileColumnCount = clip_mod.RollingClipProfileColumnCount;
const rollingClipProfileOutputNames = clip_mod.rollingClipProfileOutputNames;
const ExpandingClipProfileColumnCount = clip_mod.ExpandingClipProfileColumnCount;
const expandingClipProfileOutputNames = clip_mod.expandingClipProfileOutputNames;
const clipProfileColumnsByValue = clip_mod.clipProfileColumnsByValue;
const rollingClipProfileColumnsByValue = clip_mod.rollingClipProfileColumnsByValue;
const expandingClipProfileColumnsByValue = clip_mod.expandingClipProfileColumnsByValue;
const risk_mod = @import("dataframe_risk.zig");
const DrawdownProfileColumnCount = risk_mod.DrawdownProfileColumnCount;
const drawdownProfileOutputNames = risk_mod.drawdownProfileOutputNames;
const RollingDrawdownProfileColumnCount = risk_mod.RollingDrawdownProfileColumnCount;
const rollingDrawdownProfileOutputNames = risk_mod.rollingDrawdownProfileOutputNames;
const ExtremaProfileColumnCount = risk_mod.ExtremaProfileColumnCount;
const extremaProfileOutputNames = risk_mod.extremaProfileOutputNames;
const rollingDrawdownProfileColumnsByValue = risk_mod.rollingDrawdownProfileColumnsByValue;
const drawdownProfileColumnsByValue = risk_mod.drawdownProfileColumnsByValue;
const extremaProfileColumnsByValue = risk_mod.extremaProfileColumnsByValue;
const standardize_mod = @import("dataframe_standardize.zig");
const StandardizeProfileColumnCount = standardize_mod.StandardizeProfileColumnCount;
const standardizeProfileOutputNames = standardize_mod.standardizeProfileOutputNames;
const standardizeProfileColumnsByValue = standardize_mod.standardizeProfileColumnsByValue;
const robust_mod = @import("dataframe_robust.zig");
const RobustProfileColumnCount = robust_mod.RobustProfileColumnCount;
const robustProfileOutputNames = robust_mod.robustProfileOutputNames;
const RollingRobustProfileColumnCount = robust_mod.RollingRobustProfileColumnCount;
const rollingRobustProfileOutputNames = robust_mod.rollingRobustProfileOutputNames;
const ExpandingRobustProfileColumnCount = robust_mod.ExpandingRobustProfileColumnCount;
const expandingRobustProfileOutputNames = robust_mod.expandingRobustProfileOutputNames;
const rollingRobustProfileColumnsByValue = robust_mod.rollingRobustProfileColumnsByValue;
const expandingRobustProfileColumnsByValue = robust_mod.expandingRobustProfileColumnsByValue;
const robustProfileColumnsByValue = robust_mod.robustProfileColumnsByValue;
const trend_mod = @import("dataframe_trend.zig");
const TrendProfileColumnCount = trend_mod.TrendProfileColumnCount;
const trendProfileOutputNames = trend_mod.trendProfileOutputNames;
const RollingTrendProfileColumnCount = trend_mod.RollingTrendProfileColumnCount;
const rollingTrendProfileOutputNames = trend_mod.rollingTrendProfileOutputNames;
const ExpandingTrendProfileColumnCount = trend_mod.ExpandingTrendProfileColumnCount;
const expandingTrendProfileOutputNames = trend_mod.expandingTrendProfileOutputNames;
const trendProfileColumnsByValue = trend_mod.trendProfileColumnsByValue;
const rollingTrendProfileColumnsByValue = trend_mod.rollingTrendProfileColumnsByValue;
const expandingTrendProfileColumnsByValue = trend_mod.expandingTrendProfileColumnsByValue;
const change_mod = @import("dataframe_change.zig");
const ChangePointProfileColumnCount = change_mod.ChangePointProfileColumnCount;
const changePointProfileOutputNames = change_mod.changePointProfileOutputNames;
const RollingChangePointProfileColumnCount = change_mod.RollingChangePointProfileColumnCount;
const rollingChangePointProfileOutputNames = change_mod.rollingChangePointProfileOutputNames;
const ExpandingChangePointProfileColumnCount = change_mod.ExpandingChangePointProfileColumnCount;
const expandingChangePointProfileOutputNames = change_mod.expandingChangePointProfileOutputNames;
const changePointProfileColumnsByValue = change_mod.changePointProfileColumnsByValue;
const rollingChangePointProfileColumnsByValue = change_mod.rollingChangePointProfileColumnsByValue;
const expandingChangePointProfileColumnsByValue = change_mod.expandingChangePointProfileColumnsByValue;
const sign_mod = @import("dataframe_sign.zig");
const SignProfileColumnCount = sign_mod.SignProfileColumnCount;
const signProfileOutputNames = sign_mod.signProfileOutputNames;
const RollingSignProfileColumnCount = sign_mod.RollingSignProfileColumnCount;
const rollingSignProfileOutputNames = sign_mod.rollingSignProfileOutputNames;
const ExpandingSignProfileColumnCount = sign_mod.ExpandingSignProfileColumnCount;
const expandingSignProfileOutputNames = sign_mod.expandingSignProfileOutputNames;
const signProfileColumnsByValue = sign_mod.signProfileColumnsByValue;
const rollingSignProfileColumnsByValue = sign_mod.rollingSignProfileColumnsByValue;
const expandingSignProfileColumnsByValue = sign_mod.expandingSignProfileColumnsByValue;
const shift_mod = @import("dataframe_shift.zig");
const LagProfileColumnCount = shift_mod.LagProfileColumnCount;
const lagProfileOutputNames = shift_mod.lagProfileOutputNames;
const LeadProfileColumnCount = shift_mod.LeadProfileColumnCount;
const leadProfileOutputNames = shift_mod.leadProfileOutputNames;
const lagProfileColumnsByValue = shift_mod.lagProfileColumnsByValue;
const leadProfileColumnsByValue = shift_mod.leadProfileColumnsByValue;
const ema_mod = @import("dataframe_ema.zig");
const EmaProfileColumnCount = ema_mod.EmaProfileColumnCount;
const emaProfileOutputNames = ema_mod.emaProfileOutputNames;
const emaProfileColumnsByValue = ema_mod.emaProfileColumnsByValue;
const quantile_mod = @import("dataframe_quantile.zig");
const RollingQuantileProfileColumnCount = quantile_mod.RollingQuantileProfileColumnCount;
const rollingQuantileProfileOutputNames = quantile_mod.rollingQuantileProfileOutputNames;
const ExpandingQuantileProfileColumnCount = quantile_mod.ExpandingQuantileProfileColumnCount;
const expandingQuantileProfileOutputNames = quantile_mod.expandingQuantileProfileOutputNames;
const rollingQuantileProfileColumnsByValue = quantile_mod.rollingQuantileProfileColumnsByValue;
const expandingQuantileProfileColumnsByValue = quantile_mod.expandingQuantileProfileColumnsByValue;
const bucket_mod = @import("dataframe_bucket.zig");
const BucketProfileColumnCount = bucket_mod.BucketProfileColumnCount;
const bucketProfileOutputNames = bucket_mod.bucketProfileOutputNames;
const bucketProfileColumnsByValue = bucket_mod.bucketProfileColumnsByValue;
const rank_mod = @import("dataframe_rank.zig");
const RankProfileColumnCount = rank_mod.RankProfileColumnCount;
const rankProfileOutputNames = rank_mod.rankProfileOutputNames;
const RollingRankProfileColumnCount = rank_mod.RollingRankProfileColumnCount;
const rollingRankProfileOutputNames = rank_mod.rollingRankProfileOutputNames;
const ExpandingRankProfileColumnCount = rank_mod.ExpandingRankProfileColumnCount;
const expandingRankProfileOutputNames = rank_mod.expandingRankProfileOutputNames;
const rankProfileColumnsByKey = rank_mod.rankProfileColumnsByKey;
const rollingRankProfileColumnsByValue = rank_mod.rollingRankProfileColumnsByValue;
const expandingRankProfileColumnsByValue = rank_mod.expandingRankProfileColumnsByValue;
const stats_profile_mod = @import("dataframe_stats_profile.zig");
const RollingProfileColumnCount = stats_profile_mod.RollingProfileColumnCount;
const rollingProfileOutputNames = stats_profile_mod.rollingProfileOutputNames;
const ExpandingProfileColumnCount = stats_profile_mod.ExpandingProfileColumnCount;
const expandingProfileOutputNames = stats_profile_mod.expandingProfileOutputNames;
const rollingProfileColumnsByValue = stats_profile_mod.rollingProfileColumnsByValue;
const expandingProfileColumnsByValue = stats_profile_mod.expandingProfileColumnsByValue;
const moment_mod = @import("dataframe_moment.zig");
const RollingMomentProfileColumnCount = moment_mod.RollingMomentProfileColumnCount;
const rollingMomentProfileOutputNames = moment_mod.rollingMomentProfileOutputNames;
const ExpandingMomentProfileColumnCount = moment_mod.ExpandingMomentProfileColumnCount;
const expandingMomentProfileOutputNames = moment_mod.expandingMomentProfileOutputNames;
const rollingMomentProfileColumnsByValue = moment_mod.rollingMomentProfileColumnsByValue;
const expandingMomentProfileColumnsByValue = moment_mod.expandingMomentProfileColumnsByValue;
const normalize_mod = @import("dataframe_normalize.zig");
const RollingNormalizeProfileColumnCount = normalize_mod.RollingNormalizeProfileColumnCount;
const rollingNormalizeProfileOutputNames = normalize_mod.rollingNormalizeProfileOutputNames;
const ExpandingNormalizeProfileColumnCount = normalize_mod.ExpandingNormalizeProfileColumnCount;
const expandingNormalizeProfileOutputNames = normalize_mod.expandingNormalizeProfileOutputNames;
const rollingNormalizeProfileColumnsByValue = normalize_mod.rollingNormalizeProfileColumnsByValue;
const expandingNormalizeProfileColumnsByValue = normalize_mod.expandingNormalizeProfileColumnsByValue;
const range_mod = @import("dataframe_range.zig");
const RollingRangeProfileColumnCount = range_mod.RollingRangeProfileColumnCount;
const rollingRangeProfileOutputNames = range_mod.rollingRangeProfileOutputNames;
const rollingRangeProfileColumnsByValue = range_mod.rollingRangeProfileColumnsByValue;
const group_profile_mod = @import("dataframe_group_profile.zig");
const group_multi_mod = @import("dataframe_group_multi.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const names_mod = @import("dataframe_names.zig");
const freeColumn = dataframe_column_mod.freeColumn;
const validityValues = validity_mod.validityValues;
const rowIndicesFromMask = dataframe_array_mod.rowIndicesFromMask;
const takeOptionalRows = dataframe_array_mod.takeOptionalRows;
const concatDeviceDataFramesRows = dataframe_array_mod.concatDeviceDataFramesRows;
const concatDeviceColumns = dataframe_array_mod.concatDeviceColumns;
const argsortTypedColumn = dataframe_device_column_mod.argsortTypedColumn;
const distinctRowIndices = keys_mod.distinctRowIndices;
const rowsMatchAllKeys = keys_mod.rowsMatchAllKeys;
const asofRightRowIndices = keys_mod.asofRightRowIndices;
const ValidityProfileColumnCount = validity_mod.ValidityProfileColumnCount;
const validityProfileOutputNames = validity_mod.validityProfileOutputNames;
const RollingValidityProfileColumnCount = validity_mod.RollingValidityProfileColumnCount;
const rollingValidityProfileOutputNames = validity_mod.rollingValidityProfileOutputNames;
const ExpandingValidityProfileColumnCount = validity_mod.ExpandingValidityProfileColumnCount;
const expandingValidityProfileOutputNames = validity_mod.expandingValidityProfileOutputNames;
const validityProfileColumnsByValue = validity_mod.validityProfileColumnsByValue;
const rollingValidityProfileColumnsByValue = validity_mod.rollingValidityProfileColumnsByValue;
const expandingValidityProfileColumnsByValue = validity_mod.expandingValidityProfileColumnsByValue;

pub const DataError = series_mod.DataError;
pub const DType = dataframe_host_mod.DType;
pub const Column = dataframe_host_mod.Column;
pub const ColumnDef = dataframe_host_mod.ColumnDef;
pub const DataFrame = dataframe_host_mod.DataFrame;
pub const dataframe = dataframe_host_mod.dataframe;
pub const DeviceDType = array_mod.DType;
pub const DeviceDataError = DataError || array_mod.ArrayError;
pub const ArrowInteropError = DeviceDataError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
pub const ParquetInteropError = ArrowInteropError || boltha.parquet.SimpleError;

/// Vectra's portable validity representation for device dataframe columns.
///
/// cuDF uses Arrow-compatible packed bitmasks.  Vectra starts one abstraction
/// level higher and keeps validity as a `Array(bool)` so the dataframe wrapper
/// can work across CPU, CUDA, and MPS storage immediately.  A future Arrow ABI
/// bridge can add a packed-bitmask view without changing the owning column/table
/// shape introduced here.
pub const DeviceValidityEncoding = options_mod.DeviceValidityEncoding;
pub const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
pub const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
pub const DeviceScalar = options_mod.DeviceScalar;
pub const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;
pub const NullPlacement = options_mod.NullPlacement;
pub const DeviceSortOptions = options_mod.DeviceSortOptions;
pub const DeviceJoinOptions = options_mod.DeviceJoinOptions;
pub const AsofStrategy = options_mod.AsofStrategy;
pub const DeviceAsofOptions = options_mod.DeviceAsofOptions;
pub const DeviceRollingOptions = options_mod.DeviceRollingOptions;
pub const DeviceLagOptions = options_mod.DeviceLagOptions;
pub const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
pub const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
pub const DeviceStandardizeOptions = options_mod.DeviceStandardizeOptions;
pub const DeviceRobustOptions = options_mod.DeviceRobustOptions;
pub const DeviceDrawdownOptions = options_mod.DeviceDrawdownOptions;
pub const DeviceExtremaOptions = options_mod.DeviceExtremaOptions;
pub const DeviceTrendOptions = options_mod.DeviceTrendOptions;
pub const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
pub const DeviceBucketOptions = options_mod.DeviceBucketOptions;
pub const DeviceEmaOptions = options_mod.DeviceEmaOptions;
pub const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
pub const DeviceClipOptions = options_mod.DeviceClipOptions;
pub const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
pub const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
pub const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
pub const DeviceRollingRobustOptions = options_mod.DeviceRollingRobustOptions;
pub const ParquetRangePredicate = options_mod.ParquetRangePredicate;
pub const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
pub const Range = options_mod.Range;

pub const DeviceColumnView = dataframe_view_mod.DeviceColumnView;
pub const DeviceDataFrameView = dataframe_view_mod.DeviceDataFrameView;

pub const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;

pub const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
pub const DeviceColumnDef = dataframe_device_column_mod.DeviceColumnDef;

pub const DeviceLazyGroupByAggregation = lazy_op_mod.DeviceLazyGroupByAggregation;
pub const DeviceLazyJoinKind = lazy_op_mod.DeviceLazyJoinKind;
pub const DeviceLazyOp = lazy_op_mod.DeviceLazyOp(DeviceDataFrame, DeviceColumn);

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

pub const DeviceParquetScan = parquet_scan_mod.DeviceParquetScan(DeviceDataFrame, DeviceLazyFrame, DeviceColumnDef, DeviceColumn);

fn deinitLazyOps(allocator: std.mem.Allocator, ops: *std.ArrayList(DeviceLazyOp)) void {
    for (ops.items) |*op| op.deinit(allocator);
    ops.deinit(allocator);
}

const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;

const freeOwnedNameItems = names_mod.freeOwnedNameItems;

const allNamesIn = names_mod.allNamesIn;
const planLazyScanPushdown = lazy_mod.planLazyScanPushdown;
const formatLazyScanPushdown = lazy_mod.formatLazyScanPushdown;
const formatLazyOp = lazy_mod.formatLazyOp;

/// Owning fixed-width dataframe that can keep every column on the same Vectra
/// device.
///
/// This is intentionally an owning/table wrapper rather than a CUDA-only API:
/// `.cpu`, `.cuda(index)`, and `.mps(index)` use the same metadata and column
/// invariants.  CUDA/MPS row slicing and host-mask filtering currently
/// materialize through host memory because Vectra has not yet grown
/// dataframe-specific gather/compact kernels; preserving the operation behind
/// this API gives those kernels a single integration point later.
pub const DeviceDataFrame = struct {
    allocator: std.mem.Allocator,
    names: [][]const u8,
    columns: []DeviceColumn,
    rows: usize,
    device: array_mod.Device,

    pub fn init(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
        if (defs.len == 0) return DeviceDataFrame.initEmpty(allocator, 0, .cpu);
        const rows = defs[0].data.len();
        const device_value = defs[0].data.device();
        for (defs) |def| {
            if (def.data.len() != rows) return error.LengthMismatch;
            if (!def.data.device().sameDevice(device_value)) return error.InvalidDevice;
        }

        var names = try allocator.alloc([]const u8, defs.len);
        errdefer allocator.free(names);
        var columns = try allocator.alloc(DeviceColumn, defs.len);
        errdefer allocator.free(columns);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |i| {
                allocator.free(names[i]);
                columns[i].deinit();
            }
        }

        for (defs, 0..) |def, i| {
            names[i] = try allocator.dupe(u8, def.name);
            columns[i] = try def.data.clone();
            initialized += 1;
        }

        return .{ .allocator = allocator, .names = names, .columns = columns, .rows = rows, .device = device_value };
    }

    pub fn initEmpty(allocator: std.mem.Allocator, rows: usize, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        if (!device_value.isAvailable()) return error.InvalidDevice;
        return .{ .allocator = allocator, .names = &.{}, .columns = &.{}, .rows = rows, .device = device_value };
    }

    pub fn fromDataFrame(allocator: std.mem.Allocator, frame: DataFrame, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        if (!device_value.isAvailable()) return error.InvalidDevice;
        if (frame.columns.len == 0) return DeviceDataFrame.initEmpty(allocator, frame.rows, device_value);
        var defs = try allocator.alloc(DeviceColumnDef, frame.columns.len);
        defer allocator.free(defs);
        var initialized: usize = 0;
        defer {
            for (defs[0..initialized]) |*def| def.data.deinit();
        }
        for (frame.names, frame.columns, 0..) |name, col, i| {
            defs[i].name = name;
            defs[i].data = switch (col) {
                .f64 => |values| try DeviceColumn.fromSlice(f64, allocator, values, device_value),
                .i64 => |values| try DeviceColumn.fromSlice(i64, allocator, values, device_value),
                .bool => |values| try DeviceColumn.fromSlice(bool, allocator, values, device_value),
                .string => return error.TypeUnsupported,
            };
            initialized += 1;
        }
        return DeviceDataFrame.init(allocator, defs);
    }

    pub fn deinit(self: *DeviceDataFrame) void {
        for (self.names) |name| self.allocator.free(name);
        for (self.columns) |*col| col.deinit();
        if (self.names.len != 0) self.allocator.free(self.names);
        if (self.columns.len != 0) self.allocator.free(self.columns);
        self.* = undefined;
    }

    pub fn clone(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, self.names, columns, self.rows, self.device);
    }

    pub fn height(self: DeviceDataFrame) usize {
        return self.rows;
    }

    pub fn width(self: DeviceDataFrame) usize {
        return self.columns.len;
    }

    pub fn shape(self: DeviceDataFrame) struct { rows: usize, cols: usize } {
        return .{ .rows = self.rows, .cols = self.columns.len };
    }

    pub fn columnIndex(self: DeviceDataFrame, name: []const u8) ?usize {
        for (self.names, 0..) |existing, i| {
            if (std.mem.eql(u8, existing, name)) return i;
        }
        return null;
    }

    pub fn column(self: *const DeviceDataFrame, name: []const u8) DataError!*const DeviceColumn {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return &self.columns[idx];
    }

    pub fn columnDType(self: DeviceDataFrame, name: []const u8) DataError!DeviceDType {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx].dtype();
    }

    pub fn binaryColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnBinaryOp) DeviceDataError!DeviceColumn {
        const lhs = try self.column(lhs_name);
        const rhs = try self.column(rhs_name);
        return lhs.binary(rhs.*, op);
    }

    pub fn addColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!DeviceColumn {
        return self.binaryColumns(lhs_name, rhs_name, .add);
    }

    pub fn subColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!DeviceColumn {
        return self.binaryColumns(lhs_name, rhs_name, .sub);
    }

    pub fn mulColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!DeviceColumn {
        return self.binaryColumns(lhs_name, rhs_name, .mul);
    }

    pub fn divColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!DeviceColumn {
        return self.binaryColumns(lhs_name, rhs_name, .div);
    }

    pub fn binaryColumnScalar(self: DeviceDataFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) DeviceDataError!DeviceColumn {
        const col = try self.column(name);
        return col.binaryScalar(T, scalar, op);
    }

    pub fn binaryColumnScalarWithDeviceScalar(self: DeviceDataFrame, name: []const u8, scalar: DeviceScalar, op: DeviceColumnBinaryOp) DeviceDataError!DeviceColumn {
        const col = try self.column(name);
        return switch (scalar) {
            .i8 => |value| col.binaryScalar(i8, value, op),
            .i16 => |value| col.binaryScalar(i16, value, op),
            .i32 => |value| col.binaryScalar(i32, value, op),
            .i64 => |value| col.binaryScalar(i64, value, op),
            .u8 => |value| col.binaryScalar(u8, value, op),
            .u16 => |value| col.binaryScalar(u16, value, op),
            .u32 => |value| col.binaryScalar(u32, value, op),
            .u64 => |value| col.binaryScalar(u64, value, op),
            .usize => |value| col.binaryScalar(usize, value, op),
            .isize => |value| col.binaryScalar(isize, value, op),
            .f16 => |value| col.binaryScalar(f16, value, op),
            .f32 => |value| col.binaryScalar(f32, value, op),
            .f64 => |value| col.binaryScalar(f64, value, op),
            .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
        };
    }

    pub fn compareColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        const lhs = try self.column(lhs_name);
        const rhs = try self.column(rhs_name);
        return lhs.compare(rhs.*, op);
    }

    pub fn compareColumnScalar(self: DeviceDataFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        const col = try self.column(name);
        return col.compareScalar(T, scalar, op);
    }

    pub fn compareColumnScalarWithDeviceScalar(self: DeviceDataFrame, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        const col = try self.column(name);
        return switch (scalar) {
            .bool => |value| col.compareScalar(bool, value, op),
            .i8 => |value| col.compareScalar(i8, value, op),
            .i16 => |value| col.compareScalar(i16, value, op),
            .i32 => |value| col.compareScalar(i32, value, op),
            .i64 => |value| col.compareScalar(i64, value, op),
            .u8 => |value| col.compareScalar(u8, value, op),
            .u16 => |value| col.compareScalar(u16, value, op),
            .u32 => |value| col.compareScalar(u32, value, op),
            .u64 => |value| col.compareScalar(u64, value, op),
            .usize => |value| col.compareScalar(usize, value, op),
            .isize => |value| col.compareScalar(isize, value, op),
            .f16 => |value| col.compareScalar(f16, value, op),
            .f32 => |value| col.compareScalar(f32, value, op),
            .f64 => |value| col.compareScalar(f64, value, op),
            .bf16, .c64, .c128 => error.TypeUnsupported,
        };
    }

    pub fn filterColumnMask(self: DeviceDataFrame, mask: DeviceColumn) DeviceDataError!DeviceDataFrame {
        const typed_mask = switch (mask) {
            .bool => |typed| typed,
            else => return error.TypeMismatch,
        };
        if (!typed_mask.device().sameDevice(self.device)) return error.InvalidDevice;
        if (typed_mask.len() != self.rows) return error.LengthMismatch;
        const host_values = try typed_mask.values.toOwnedSlice(self.allocator);
        defer self.allocator.free(host_values);
        if (typed_mask.validity) |validity_array| {
            const host_validity = try validity_array.toOwnedSlice(self.allocator);
            defer self.allocator.free(host_validity);
            const host_mask = try self.allocator.alloc(bool, self.rows);
            defer self.allocator.free(host_mask);
            for (host_values, host_validity, host_mask) |value, valid, *slot| {
                // Match dataframe query semantics used by Polars/Arrow engines:
                // a null predicate does not select the row.  Keeping the mask
                // resolution here makes scalar-filter and Parquet statistics
                // pushdown conservative even when predicate columns are nullable.
                slot.* = valid and value;
            }
            return self.filter(host_mask);
        }
        return self.filter(host_values);
    }

    pub fn toArrowSchema(self: DeviceDataFrame, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.Schema {
        return dataframe_arrow_mod.toArrowSchema(self, allocator);
    }

    pub fn toArrowRecordBatch(self: DeviceDataFrame, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.RecordBatch {
        return dataframe_arrow_mod.toArrowRecordBatch(self, allocator);
    }

    pub fn toArrowTable(self: DeviceDataFrame, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.Table {
        return dataframe_arrow_mod.toArrowTable(self, allocator);
    }

    pub fn toParquetBytes(self: DeviceDataFrame, allocator: std.mem.Allocator) ParquetInteropError![]u8 {
        return dataframe_arrow_mod.toParquetBytes(self, allocator);
    }

    pub fn fromParquetBytes(allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) ParquetInteropError!DeviceDataFrame {
        return dataframe_arrow_mod.fromParquetBytes(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, bytes, device_value);
    }

    pub fn fromParquetBytesPruned(
        allocator: std.mem.Allocator,
        bytes: []const u8,
        column_name: []const u8,
        predicate: ParquetRangePredicate,
        device_value: array_mod.Device,
    ) ParquetInteropError!DeviceDataFrame {
        return dataframe_arrow_mod.fromParquetBytesPruned(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, bytes, column_name, predicate, device_value);
    }

    pub fn fromArrowTable(allocator: std.mem.Allocator, table: boltha.arrow.Table, device_value: array_mod.Device) ArrowInteropError!DeviceDataFrame {
        return dataframe_arrow_mod.fromArrowTable(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, table, device_value);
    }

    pub fn fromArrowTableProjection(
        allocator: std.mem.Allocator,
        table: boltha.arrow.Table,
        wanted_names: []const []const u8,
        device_value: array_mod.Device,
    ) ArrowInteropError!DeviceDataFrame {
        return dataframe_arrow_mod.fromArrowTableProjection(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, table, wanted_names, device_value);
    }

    pub fn fromArrowRecordBatch(allocator: std.mem.Allocator, batch: boltha.arrow.RecordBatch, device_value: array_mod.Device) ArrowInteropError!DeviceDataFrame {
        return dataframe_arrow_mod.fromArrowRecordBatch(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, batch, device_value);
    }

    pub fn fromArrowRecordBatchProjection(
        allocator: std.mem.Allocator,
        batch: boltha.arrow.RecordBatch,
        wanted_names: []const []const u8,
        device_value: array_mod.Device,
    ) ArrowInteropError!DeviceDataFrame {
        return dataframe_arrow_mod.fromArrowRecordBatchProjection(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, batch, wanted_names, device_value);
    }

    pub fn view(self: DeviceDataFrame) DeviceDataError!DeviceDataFrameView {
        const columns = try self.allocator.alloc(DeviceColumnView, self.columns.len);
        errdefer self.allocator.free(columns);
        for (self.columns, columns) |col, *slot| slot.* = col.view();
        return .{
            .allocator = self.allocator,
            .names = self.names,
            .columns = columns,
            .rows = self.rows,
            .device = self.device,
        };
    }

    pub fn select(self: DeviceDataFrame, wanted_names: []const []const u8) DeviceDataError!DeviceDataFrame {
        if (wanted_names.len == 0) return DeviceDataFrame.initEmpty(self.allocator, self.rows, self.device);
        var columns = try self.allocator.alloc(DeviceColumn, wanted_names.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (wanted_names, 0..) |name, i| {
            const source = try self.column(name);
            columns[i] = try source.clone();
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, wanted_names, columns, self.rows, self.device);
    }

    pub fn withColumn(self: DeviceDataFrame, name: []const u8, data: DeviceColumn) DeviceDataError!DeviceDataFrame {
        if (data.len() != self.rows) return error.LengthMismatch;
        if (!data.device().sameDevice(self.device)) return error.InvalidDevice;
        var source_names = try self.allocator.alloc([]const u8, self.columns.len + 1);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |existing, i| source_names[i] = existing;
        source_names[self.columns.len] = name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + 1);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        columns[self.columns.len] = try data.clone();
        initialized += 1;
        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn head(self: DeviceDataFrame, n: usize) DeviceDataError!DeviceDataFrame {
        return self.sliceRows(0, @min(n, self.rows));
    }

    pub fn tail(self: DeviceDataFrame, n: usize) DeviceDataError!DeviceDataFrame {
        const count = @min(n, self.rows);
        return self.sliceRows(self.rows - count, self.rows);
    }

    pub fn sliceRows(self: DeviceDataFrame, start: usize, stop: usize) DeviceDataError!DeviceDataFrame {
        const end = @min(stop, self.rows);
        const begin = @min(start, end);
        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.sliceRows(begin, end);
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, self.names, columns, end - begin, self.device);
    }

    pub fn take(self: DeviceDataFrame, row_indices: []const usize) DeviceDataError!DeviceDataFrame {
        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.take(row_indices);
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, self.names, columns, row_indices.len, self.device);
    }

    pub fn concatRows(self: DeviceDataFrame, other: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return concatDeviceDataFramesRows(DeviceDataFrame, self, other);
    }

    pub fn appendRows(self: DeviceDataFrame, other: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.concatRows(other);
    }

    pub fn vstack(self: DeviceDataFrame, other: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.concatRows(other);
    }

    pub fn distinctRows(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.distinctOn(self.names);
    }

    pub fn distinctOn(self: DeviceDataFrame, key_names: []const []const u8) DeviceDataError!DeviceDataFrame {
        const indices = try distinctRowIndices(self.allocator, self, key_names);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn dropDuplicates(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.distinctRows();
    }

    pub fn dropDuplicatesOn(self: DeviceDataFrame, key_names: []const []const u8) DeviceDataError!DeviceDataFrame {
        return self.distinctOn(key_names);
    }

    pub fn uniqueRows(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.distinctRows();
    }

    pub fn argsortBy(self: DeviceDataFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError![]usize {
        const sort_key = try self.column(name);
        return sort_key.argsort(self.allocator, options_value);
    }

    pub fn sortBy(self: DeviceDataFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        const order = try self.argsortBy(name, options_value);
        defer self.allocator.free(order);
        return self.take(order);
    }

    pub fn sortByColumn(self: DeviceDataFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        return self.sortBy(name, options_value);
    }

    pub fn topKBy(self: DeviceDataFrame, name: []const u8, k: usize, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        var sorted = try self.sortBy(name, options_value);
        defer sorted.deinit();
        return sorted.head(k);
    }

    pub fn rankProfileBy(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        const rank_key = try self.column(name);
        var rank_columns = try rankProfileColumnsByKey(self.allocator, rank_key.*, options_value, self.device, self.rows);
        var rank_columns_transferred: usize = 0;
        errdefer {
            for (rank_columns[rank_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rank_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rank_names = try rankProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rank_names[0..]);
        for (rank_names, 0..) |rank_name, i| source_names[self.columns.len + i] = rank_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rank_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rank_columns) |*rank_col| {
            columns[initialized] = rank_col.*;
            initialized += 1;
            rank_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingMomentProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingMomentProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingMomentProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingRangeProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingRangeProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingRangeProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingNormalizeProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingNormalizeProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingNormalizeProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingNormalizeProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const expanding_value = try self.column(name);
        var expanding_columns = try expandingNormalizeProfileColumnsByValue(self.allocator, expanding_value.*, options_value, self.device, self.rows);
        var expanding_columns_transferred: usize = 0;
        errdefer {
            for (expanding_columns[expanding_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + expanding_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var expanding_names = try expandingNormalizeProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, expanding_names[0..]);
        for (expanding_names, 0..) |expanding_name, i| source_names[self.columns.len + i] = expanding_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + expanding_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&expanding_columns) |*expanding_col| {
            columns[initialized] = expanding_col.*;
            initialized += 1;
            expanding_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingQuantileProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingQuantileProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingQuantileProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingQuantileProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const expanding_value = try self.column(name);
        var expanding_columns = try expandingQuantileProfileColumnsByValue(self.allocator, expanding_value.*, options_value, self.device, self.rows);
        var expanding_columns_transferred: usize = 0;
        errdefer {
            for (expanding_columns[expanding_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + expanding_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var expanding_names = try expandingQuantileProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, expanding_names[0..]);
        for (expanding_names, 0..) |expanding_name, i| source_names[self.columns.len + i] = expanding_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + expanding_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&expanding_columns) |*expanding_col| {
            columns[initialized] = expanding_col.*;
            initialized += 1;
            expanding_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingBoolProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const source = try self.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;
        var bool_columns = try rollingBoolProfileColumns(self.allocator, source.bool, options_value, self.device, self.rows);
        var bool_columns_transferred: usize = 0;
        errdefer {
            for (bool_columns[bool_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + bool_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var bool_names = try rollingBoolProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, bool_names[0..]);
        for (bool_names, 0..) |bool_name, i| source_names[self.columns.len + i] = bool_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + bool_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&bool_columns) |*bool_col| {
            columns[initialized] = bool_col.*;
            initialized += 1;
            bool_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingDrawdownProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingDrawdownProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingDrawdownProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingRobustProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingRobustOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingRobustProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingRobustProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingRankProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingRankOptions) DeviceDataError!DeviceDataFrame {
        const rolling_value = try self.column(name);
        var rolling_columns = try rollingRankProfileColumnsByValue(self.allocator, rolling_value.*, options_value, self.device, self.rows);
        var rolling_columns_transferred: usize = 0;
        errdefer {
            for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + rolling_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var rolling_names = try rollingRankProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, rolling_names[0..]);
        for (rolling_names, 0..) |rolling_name, i| source_names[self.columns.len + i] = rolling_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + rolling_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&rolling_columns) |*rolling_col| {
            columns[initialized] = rolling_col.*;
            initialized += 1;
            rolling_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn lagProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!DeviceDataFrame {
        const lag_value = try self.column(name);
        var lag_columns = try lagProfileColumnsByValue(self.allocator, lag_value.*, options_value, self.device, self.rows);
        var lag_columns_transferred: usize = 0;
        errdefer {
            for (lag_columns[lag_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + lag_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var lag_names = try lagProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, lag_names[0..]);
        for (lag_names, 0..) |lag_name, i| source_names[self.columns.len + i] = lag_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + lag_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&lag_columns) |*lag_col| {
            columns[initialized] = lag_col.*;
            initialized += 1;
            lag_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn leadProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceLagOptions) DeviceDataError!DeviceDataFrame {
        const lead_value = try self.column(name);
        var lead_columns = try leadProfileColumnsByValue(self.allocator, lead_value.*, options_value, self.device, self.rows);
        var lead_columns_transferred: usize = 0;
        errdefer {
            for (lead_columns[lead_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + lead_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var lead_names = try leadProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, lead_names[0..]);
        for (lead_names, 0..) |lead_name, i| source_names[self.columns.len + i] = lead_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + lead_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&lead_columns) |*lead_col| {
            columns[initialized] = lead_col.*;
            initialized += 1;
            lead_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn clipProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceClipOptions) DeviceDataError!DeviceDataFrame {
        const clip_value = try self.column(name);
        var clip_columns = try clipProfileColumnsByValue(self.allocator, clip_value.*, options_value, self.device, self.rows);
        var clip_columns_transferred: usize = 0;
        errdefer {
            for (clip_columns[clip_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + clip_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var clip_names = try clipProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, clip_names[0..]);
        for (clip_names, 0..) |clip_name, i| source_names[self.columns.len + i] = clip_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + clip_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&clip_columns) |*clip_col| {
            columns[initialized] = clip_col.*;
            initialized += 1;
            clip_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingClipProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, clip_options: DeviceClipOptions, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const clip_value = try self.column(name);
        var clip_columns = try rollingClipProfileColumnsByValue(self.allocator, clip_value.*, clip_options, options_value, self.device, self.rows);
        var clip_columns_transferred: usize = 0;
        errdefer {
            for (clip_columns[clip_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + clip_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var clip_names = try rollingClipProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, clip_names[0..]);
        for (clip_names, 0..) |clip_name, i| source_names[self.columns.len + i] = clip_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + clip_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&clip_columns) |*clip_col| {
            columns[initialized] = clip_col.*;
            initialized += 1;
            clip_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingClipProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, clip_options: DeviceClipOptions, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const clip_value = try self.column(name);
        var clip_columns = try expandingClipProfileColumnsByValue(self.allocator, clip_value.*, clip_options, options_value, self.device, self.rows);
        var clip_columns_transferred: usize = 0;
        errdefer {
            for (clip_columns[clip_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + clip_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var clip_names = try expandingClipProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, clip_names[0..]);
        for (clip_names, 0..) |clip_name, i| source_names[self.columns.len + i] = clip_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + clip_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&clip_columns) |*clip_col| {
            columns[initialized] = clip_col.*;
            initialized += 1;
            clip_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn thresholdProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceThresholdOptions) DeviceDataError!DeviceDataFrame {
        const threshold_value = try self.column(name);
        var threshold_columns = try thresholdProfileColumnsByValue(self.allocator, threshold_value.*, options_value, self.device, self.rows);
        var threshold_columns_transferred: usize = 0;
        errdefer {
            for (threshold_columns[threshold_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + threshold_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var threshold_names = try thresholdProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, threshold_names[0..]);
        for (threshold_names, 0..) |threshold_name, i| source_names[self.columns.len + i] = threshold_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + threshold_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&threshold_columns) |*threshold_col| {
            columns[initialized] = threshold_col.*;
            initialized += 1;
            threshold_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingThresholdProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const threshold_value = try self.column(name);
        var threshold_columns = try rollingThresholdProfileColumnsByValue(self.allocator, threshold_value.*, threshold, options_value, self.device, self.rows);
        var threshold_columns_transferred: usize = 0;
        errdefer {
            for (threshold_columns[threshold_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + threshold_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var threshold_names = try rollingThresholdProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, threshold_names[0..]);
        for (threshold_names, 0..) |threshold_name, i| source_names[self.columns.len + i] = threshold_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + threshold_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&threshold_columns) |*threshold_col| {
            columns[initialized] = threshold_col.*;
            initialized += 1;
            threshold_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingThresholdProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const threshold_value = try self.column(name);
        var threshold_columns = try expandingThresholdProfileColumnsByValue(self.allocator, threshold_value.*, threshold, options_value, self.device, self.rows);
        var threshold_columns_transferred: usize = 0;
        errdefer {
            for (threshold_columns[threshold_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + threshold_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var threshold_names = try expandingThresholdProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, threshold_names[0..]);
        for (threshold_names, 0..) |threshold_name, i| source_names[self.columns.len + i] = threshold_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + threshold_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&threshold_columns) |*threshold_col| {
            columns[initialized] = threshold_col.*;
            initialized += 1;
            threshold_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const expanding_value = try self.column(name);
        var expanding_columns = try expandingProfileColumnsByValue(self.allocator, expanding_value.*, options_value, self.device, self.rows);
        var expanding_columns_transferred: usize = 0;
        errdefer {
            for (expanding_columns[expanding_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + expanding_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var expanding_names = try expandingProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, expanding_names[0..]);
        for (expanding_names, 0..) |expanding_name, i| source_names[self.columns.len + i] = expanding_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + expanding_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&expanding_columns) |*expanding_col| {
            columns[initialized] = expanding_col.*;
            initialized += 1;
            expanding_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingBoolProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const source = try self.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;
        var bool_columns = try expandingBoolProfileColumns(self.allocator, source.bool, options_value, self.device, self.rows);
        var bool_columns_transferred: usize = 0;
        errdefer {
            for (bool_columns[bool_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + bool_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var bool_names = try expandingBoolProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, bool_names[0..]);
        for (bool_names, 0..) |bool_name, i| source_names[self.columns.len + i] = bool_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + bool_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&bool_columns) |*bool_col| {
            columns[initialized] = bool_col.*;
            initialized += 1;
            bool_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingRankProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingRankOptions) DeviceDataError!DeviceDataFrame {
        const expanding_value = try self.column(name);
        var expanding_columns = try expandingRankProfileColumnsByValue(self.allocator, expanding_value.*, options_value, self.device, self.rows);
        var expanding_columns_transferred: usize = 0;
        errdefer {
            for (expanding_columns[expanding_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + expanding_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var expanding_names = try expandingRankProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, expanding_names[0..]);
        for (expanding_names, 0..) |expanding_name, i| source_names[self.columns.len + i] = expanding_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + expanding_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&expanding_columns) |*expanding_col| {
            columns[initialized] = expanding_col.*;
            initialized += 1;
            expanding_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingRobustProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!DeviceDataFrame {
        const expanding_value = try self.column(name);
        var expanding_columns = try expandingRobustProfileColumnsByValue(self.allocator, expanding_value.*, options_value, self.device, self.rows);
        var expanding_columns_transferred: usize = 0;
        errdefer {
            for (expanding_columns[expanding_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + expanding_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var expanding_names = try expandingRobustProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, expanding_names[0..]);
        for (expanding_names, 0..) |expanding_name, i| source_names[self.columns.len + i] = expanding_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + expanding_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&expanding_columns) |*expanding_col| {
            columns[initialized] = expanding_col.*;
            initialized += 1;
            expanding_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingMomentProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const expanding_value = try self.column(name);
        var expanding_columns = try expandingMomentProfileColumnsByValue(self.allocator, expanding_value.*, options_value, self.device, self.rows);
        var expanding_columns_transferred: usize = 0;
        errdefer {
            for (expanding_columns[expanding_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + expanding_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var expanding_names = try expandingMomentProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, expanding_names[0..]);
        for (expanding_names, 0..) |expanding_name, i| source_names[self.columns.len + i] = expanding_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + expanding_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&expanding_columns) |*expanding_col| {
            columns[initialized] = expanding_col.*;
            initialized += 1;
            expanding_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn standardizeProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceStandardizeOptions) DeviceDataError!DeviceDataFrame {
        const standardize_value = try self.column(name);
        var standardize_columns = try standardizeProfileColumnsByValue(self.allocator, standardize_value.*, options_value, self.device, self.rows);
        var standardize_columns_transferred: usize = 0;
        errdefer {
            for (standardize_columns[standardize_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + standardize_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var standardize_names = try standardizeProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, standardize_names[0..]);
        for (standardize_names, 0..) |standardize_name, i| source_names[self.columns.len + i] = standardize_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + standardize_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&standardize_columns) |*standardize_col| {
            columns[initialized] = standardize_col.*;
            initialized += 1;
            standardize_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn robustProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!DeviceDataFrame {
        const robust_value = try self.column(name);
        var robust_columns = try robustProfileColumnsByValue(self.allocator, robust_value.*, options_value, self.device, self.rows);
        var robust_columns_transferred: usize = 0;
        errdefer {
            for (robust_columns[robust_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + robust_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var robust_names = try robustProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, robust_names[0..]);
        for (robust_names, 0..) |robust_name, i| source_names[self.columns.len + i] = robust_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + robust_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&robust_columns) |*robust_col| {
            columns[initialized] = robust_col.*;
            initialized += 1;
            robust_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn drawdownProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceDrawdownOptions) DeviceDataError!DeviceDataFrame {
        const drawdown_value = try self.column(name);
        var drawdown_columns = try drawdownProfileColumnsByValue(self.allocator, drawdown_value.*, options_value, self.device, self.rows);
        var drawdown_columns_transferred: usize = 0;
        errdefer {
            for (drawdown_columns[drawdown_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + drawdown_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var drawdown_names = try drawdownProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, drawdown_names[0..]);
        for (drawdown_names, 0..) |drawdown_name, i| source_names[self.columns.len + i] = drawdown_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + drawdown_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&drawdown_columns) |*drawdown_col| {
            columns[initialized] = drawdown_col.*;
            initialized += 1;
            drawdown_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn extremaProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExtremaOptions) DeviceDataError!DeviceDataFrame {
        const extrema_value = try self.column(name);
        var extrema_columns = try extremaProfileColumnsByValue(self.allocator, extrema_value.*, options_value, self.device, self.rows);
        var extrema_columns_transferred: usize = 0;
        errdefer {
            for (extrema_columns[extrema_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + extrema_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var extrema_names = try extremaProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, extrema_names[0..]);
        for (extrema_names, 0..) |extrema_name, i| source_names[self.columns.len + i] = extrema_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + extrema_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&extrema_columns) |*extrema_col| {
            columns[initialized] = extrema_col.*;
            initialized += 1;
            extrema_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn trendProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        const trend_value = try self.column(name);
        var trend_columns = try trendProfileColumnsByValue(self.allocator, trend_value.*, options_value, self.device, self.rows);
        var trend_columns_transferred: usize = 0;
        errdefer {
            for (trend_columns[trend_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + trend_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var trend_names = try trendProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, trend_names[0..]);
        for (trend_names, 0..) |trend_name, i| source_names[self.columns.len + i] = trend_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + trend_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&trend_columns) |*trend_col| {
            columns[initialized] = trend_col.*;
            initialized += 1;
            trend_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn changePointProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        const change_value = try self.column(name);
        var change_columns = try changePointProfileColumnsByValue(self.allocator, change_value.*, threshold, options_value, self.device, self.rows);
        var change_columns_transferred: usize = 0;
        errdefer {
            for (change_columns[change_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + change_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var change_names = try changePointProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, change_names[0..]);
        for (change_names, 0..) |change_name, i| source_names[self.columns.len + i] = change_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + change_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&change_columns) |*change_col| {
            columns[initialized] = change_col.*;
            initialized += 1;
            change_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingChangePointProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const change_value = try self.column(name);
        var change_columns = try rollingChangePointProfileColumnsByValue(self.allocator, change_value.*, threshold, change_options, options_value, self.device, self.rows);
        var change_columns_transferred: usize = 0;
        errdefer {
            for (change_columns[change_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + change_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var change_names = try rollingChangePointProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, change_names[0..]);
        for (change_names, 0..) |change_name, i| source_names[self.columns.len + i] = change_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + change_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&change_columns) |*change_col| {
            columns[initialized] = change_col.*;
            initialized += 1;
            change_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingChangePointProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const change_value = try self.column(name);
        var change_columns = try expandingChangePointProfileColumnsByValue(self.allocator, change_value.*, threshold, change_options, options_value, self.device, self.rows);
        var change_columns_transferred: usize = 0;
        errdefer {
            for (change_columns[change_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + change_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var change_names = try expandingChangePointProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, change_names[0..]);
        for (change_names, 0..) |change_name, i| source_names[self.columns.len + i] = change_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + change_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&change_columns) |*change_col| {
            columns[initialized] = change_col.*;
            initialized += 1;
            change_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingTrendProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const trend_value = try self.column(name);
        var trend_columns = try rollingTrendProfileColumnsByValue(self.allocator, trend_value.*, trend_options, options_value, self.device, self.rows);
        var trend_columns_transferred: usize = 0;
        errdefer {
            for (trend_columns[trend_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + trend_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var trend_names = try rollingTrendProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, trend_names[0..]);
        for (trend_names, 0..) |trend_name, i| source_names[self.columns.len + i] = trend_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + trend_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&trend_columns) |*trend_col| {
            columns[initialized] = trend_col.*;
            initialized += 1;
            trend_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingTrendProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const trend_value = try self.column(name);
        var trend_columns = try expandingTrendProfileColumnsByValue(self.allocator, trend_value.*, trend_options, options_value, self.device, self.rows);
        var trend_columns_transferred: usize = 0;
        errdefer {
            for (trend_columns[trend_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + trend_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var trend_names = try expandingTrendProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, trend_names[0..]);
        for (trend_names, 0..) |trend_name, i| source_names[self.columns.len + i] = trend_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + trend_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&trend_columns) |*trend_col| {
            columns[initialized] = trend_col.*;
            initialized += 1;
            trend_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn signProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        const sign_value = try self.column(name);
        var sign_columns = try signProfileColumnsByValue(self.allocator, sign_value.*, options_value, self.device, self.rows);
        var sign_columns_transferred: usize = 0;
        errdefer {
            for (sign_columns[sign_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + sign_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var sign_names = try signProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, sign_names[0..]);
        for (sign_names, 0..) |sign_name, i| source_names[self.columns.len + i] = sign_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + sign_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&sign_columns) |*sign_col| {
            columns[initialized] = sign_col.*;
            initialized += 1;
            sign_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingSignProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const sign_value = try self.column(name);
        var sign_columns = try rollingSignProfileColumnsByValue(self.allocator, sign_value.*, sign_options, options_value, self.device, self.rows);
        var sign_columns_transferred: usize = 0;
        errdefer {
            for (sign_columns[sign_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + sign_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var sign_names = try rollingSignProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, sign_names[0..]);
        for (sign_names, 0..) |sign_name, i| source_names[self.columns.len + i] = sign_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + sign_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&sign_columns) |*sign_col| {
            columns[initialized] = sign_col.*;
            initialized += 1;
            sign_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingSignProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const sign_value = try self.column(name);
        var sign_columns = try expandingSignProfileColumnsByValue(self.allocator, sign_value.*, sign_options, options_value, self.device, self.rows);
        var sign_columns_transferred: usize = 0;
        errdefer {
            for (sign_columns[sign_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + sign_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var sign_names = try expandingSignProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, sign_names[0..]);
        for (sign_names, 0..) |sign_name, i| source_names[self.columns.len + i] = sign_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + sign_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&sign_columns) |*sign_col| {
            columns[initialized] = sign_col.*;
            initialized += 1;
            sign_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn crossoverProfile(
        self: DeviceDataFrame,
        lhs_name: []const u8,
        rhs_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceCrossoverOptions,
    ) DeviceDataError!DeviceDataFrame {
        const lhs = try self.column(lhs_name);
        const rhs = try self.column(rhs_name);
        if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
        var cross_columns = try crossoverProfileColumnsByValue(self.allocator, lhs.*, rhs.*, options_value, self.device, self.rows);
        var cross_columns_transferred: usize = 0;
        errdefer {
            for (cross_columns[cross_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + cross_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var cross_names = try crossoverProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, cross_names[0..]);
        for (cross_names, 0..) |cross_name, i| source_names[self.columns.len + i] = cross_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + cross_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&cross_columns) |*cross_col| {
            columns[initialized] = cross_col.*;
            initialized += 1;
            cross_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingCrossoverProfile(
        self: DeviceDataFrame,
        lhs_name: []const u8,
        rhs_name: []const u8,
        output_prefix: []const u8,
        cross_options: DeviceCrossoverOptions,
        options_value: DeviceRollingOptions,
    ) DeviceDataError!DeviceDataFrame {
        const lhs = try self.column(lhs_name);
        const rhs = try self.column(rhs_name);
        if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
        var cross_columns = try rollingCrossoverProfileColumnsByValue(self.allocator, lhs.*, rhs.*, cross_options, options_value, self.device, self.rows);
        var cross_columns_transferred: usize = 0;
        errdefer {
            for (cross_columns[cross_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + cross_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var cross_names = try rollingCrossoverProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, cross_names[0..]);
        for (cross_names, 0..) |cross_name, i| source_names[self.columns.len + i] = cross_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + cross_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&cross_columns) |*cross_col| {
            columns[initialized] = cross_col.*;
            initialized += 1;
            cross_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingCrossoverProfile(
        self: DeviceDataFrame,
        lhs_name: []const u8,
        rhs_name: []const u8,
        output_prefix: []const u8,
        cross_options: DeviceCrossoverOptions,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        const lhs = try self.column(lhs_name);
        const rhs = try self.column(rhs_name);
        if (lhs.dtype() != rhs.dtype()) return error.TypeMismatch;
        var cross_columns = try expandingCrossoverProfileColumnsByValue(self.allocator, lhs.*, rhs.*, cross_options, options_value, self.device, self.rows);
        var cross_columns_transferred: usize = 0;
        errdefer {
            for (cross_columns[cross_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + cross_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var cross_names = try expandingCrossoverProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, cross_names[0..]);
        for (cross_names, 0..) |cross_name, i| source_names[self.columns.len + i] = cross_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + cross_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&cross_columns) |*cross_col| {
            columns[initialized] = cross_col.*;
            initialized += 1;
            cross_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn bucketProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceBucketOptions) DeviceDataError!DeviceDataFrame {
        const bucket_value = try self.column(name);
        var bucket_columns = try bucketProfileColumnsByValue(self.allocator, bucket_value.*, options_value, self.device, self.rows);
        var bucket_columns_transferred: usize = 0;
        errdefer {
            for (bucket_columns[bucket_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + bucket_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var bucket_names = try bucketProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, bucket_names[0..]);
        for (bucket_names, 0..) |bucket_name, i| source_names[self.columns.len + i] = bucket_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + bucket_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&bucket_columns) |*bucket_col| {
            columns[initialized] = bucket_col.*;
            initialized += 1;
            bucket_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn emaProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceEmaOptions) DeviceDataError!DeviceDataFrame {
        const ema_value = try self.column(name);
        var ema_columns = try emaProfileColumnsByValue(self.allocator, ema_value.*, options_value, self.device, self.rows);
        var ema_columns_transferred: usize = 0;
        errdefer {
            for (ema_columns[ema_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + ema_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var ema_names = try emaProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, ema_names[0..]);
        for (ema_names, 0..) |ema_name, i| source_names[self.columns.len + i] = ema_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + ema_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&ema_columns) |*ema_col| {
            columns[initialized] = ema_col.*;
            initialized += 1;
            ema_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn linearFitProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceLinearFitOptions,
    ) DeviceDataError!DeviceDataFrame {
        const x = try self.column(x_name);
        const y = try self.column(y_name);
        if (x.dtype() != y.dtype()) return error.TypeMismatch;
        var fit_columns = try linearFitProfileColumnsByValue(self.allocator, x.*, y.*, options_value, self.device, self.rows);
        var fit_columns_transferred: usize = 0;
        errdefer {
            for (fit_columns[fit_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + fit_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var fit_names = try linearFitProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, fit_names[0..]);
        for (fit_names, 0..) |fit_name, i| source_names[self.columns.len + i] = fit_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + fit_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&fit_columns) |*fit_col| {
            columns[initialized] = fit_col.*;
            initialized += 1;
            fit_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn errorProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const actual = try self.column(actual_name);
        const predicted = try self.column(predicted_name);
        if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
        var error_columns = try errorProfileColumnsByValue(self.allocator, actual.*, predicted.*, self.device, self.rows);
        var error_columns_transferred: usize = 0;
        errdefer {
            for (error_columns[error_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + error_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var error_names = try errorProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, error_names[0..]);
        for (error_names, 0..) |error_name, i| source_names[self.columns.len + i] = error_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + error_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&error_columns) |*error_col| {
            columns[initialized] = error_col.*;
            initialized += 1;
            error_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingErrorProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingOptions,
    ) DeviceDataError!DeviceDataFrame {
        const actual = try self.column(actual_name);
        const predicted = try self.column(predicted_name);
        if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
        var error_columns = try rollingErrorProfileColumnsByValue(self.allocator, actual.*, predicted.*, options_value, self.device, self.rows);
        var error_columns_transferred: usize = 0;
        errdefer {
            for (error_columns[error_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + error_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var error_names = try rollingErrorProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, error_names[0..]);
        for (error_names, 0..) |error_name, i| source_names[self.columns.len + i] = error_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + error_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&error_columns) |*error_col| {
            columns[initialized] = error_col.*;
            initialized += 1;
            error_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingErrorProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        const actual = try self.column(actual_name);
        const predicted = try self.column(predicted_name);
        if (actual.dtype() != predicted.dtype()) return error.TypeMismatch;
        var error_columns = try expandingErrorProfileColumnsByValue(self.allocator, actual.*, predicted.*, options_value, self.device, self.rows);
        var error_columns_transferred: usize = 0;
        errdefer {
            for (error_columns[error_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + error_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var error_names = try expandingErrorProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, error_names[0..]);
        for (error_names, 0..) |error_name, i| source_names[self.columns.len + i] = error_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + error_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&error_columns) |*error_col| {
            columns[initialized] = error_col.*;
            initialized += 1;
            error_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn classificationProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const actual = try self.column(actual_name);
        const predicted = try self.column(predicted_name);
        if (actual.dtype() != .bool or predicted.dtype() != .bool) return error.TypeMismatch;
        var class_columns = try classificationProfileColumns(self.allocator, actual.bool, predicted.bool, self.device, self.rows);
        var class_columns_transferred: usize = 0;
        errdefer {
            for (class_columns[class_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + class_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var class_names = try classificationProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, class_names[0..]);
        for (class_names, 0..) |class_name, i| source_names[self.columns.len + i] = class_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + class_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&class_columns) |*class_col| {
            columns[initialized] = class_col.*;
            initialized += 1;
            class_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingClassificationProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingOptions,
    ) DeviceDataError!DeviceDataFrame {
        const actual = try self.column(actual_name);
        const predicted = try self.column(predicted_name);
        if (actual.dtype() != .bool or predicted.dtype() != .bool) return error.TypeMismatch;
        var class_columns = try rollingClassificationProfileColumns(self.allocator, actual.bool, predicted.bool, options_value, self.device, self.rows);
        var class_columns_transferred: usize = 0;
        errdefer {
            for (class_columns[class_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + class_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var class_names = try rollingClassificationProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, class_names[0..]);
        for (class_names, 0..) |class_name, i| source_names[self.columns.len + i] = class_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + class_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&class_columns) |*class_col| {
            columns[initialized] = class_col.*;
            initialized += 1;
            class_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingClassificationProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        const actual = try self.column(actual_name);
        const predicted = try self.column(predicted_name);
        if (actual.dtype() != .bool or predicted.dtype() != .bool) return error.TypeMismatch;
        var class_columns = try expandingClassificationProfileColumns(self.allocator, actual.bool, predicted.bool, options_value, self.device, self.rows);
        var class_columns_transferred: usize = 0;
        errdefer {
            for (class_columns[class_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + class_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var class_names = try expandingClassificationProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, class_names[0..]);
        for (class_names, 0..) |class_name, i| source_names[self.columns.len + i] = class_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + class_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&class_columns) |*class_col| {
            columns[initialized] = class_col.*;
            initialized += 1;
            class_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn boolTransitionProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        const source = try self.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;
        var transition_columns = try boolTransitionProfileColumns(self.allocator, source.bool, options_value, self.device, self.rows);
        var transition_columns_transferred: usize = 0;
        errdefer {
            for (transition_columns[transition_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + transition_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var transition_names = try boolTransitionProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, transition_names[0..]);
        for (transition_names, 0..) |transition_name, i| source_names[self.columns.len + i] = transition_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + transition_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&transition_columns) |*transition_col| {
            columns[initialized] = transition_col.*;
            initialized += 1;
            transition_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingBoolTransitionProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const source = try self.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;
        var transition_columns = try rollingBoolTransitionProfileColumns(self.allocator, source.bool, transition_options, options_value, self.device, self.rows);
        var transition_columns_transferred: usize = 0;
        errdefer {
            for (transition_columns[transition_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + transition_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var transition_names = try rollingBoolTransitionProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, transition_names[0..]);
        for (transition_names, 0..) |transition_name, i| source_names[self.columns.len + i] = transition_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + transition_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&transition_columns) |*transition_col| {
            columns[initialized] = transition_col.*;
            initialized += 1;
            transition_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingBoolTransitionProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const source = try self.column(name);
        if (source.dtype() != .bool) return error.TypeMismatch;
        var transition_columns = try expandingBoolTransitionProfileColumns(self.allocator, source.bool, transition_options, options_value, self.device, self.rows);
        var transition_columns_transferred: usize = 0;
        errdefer {
            for (transition_columns[transition_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + transition_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var transition_names = try expandingBoolTransitionProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, transition_names[0..]);
        for (transition_names, 0..) |transition_name, i| source_names[self.columns.len + i] = transition_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + transition_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&transition_columns) |*transition_col| {
            columns[initialized] = transition_col.*;
            initialized += 1;
            transition_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingCorrelationProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingCorrelationOptions,
    ) DeviceDataError!DeviceDataFrame {
        const x = try self.column(x_name);
        const y = try self.column(y_name);
        if (x.dtype() != y.dtype()) return error.TypeMismatch;
        var corr_columns = try rollingCorrelationProfileColumnsByValue(self.allocator, x.*, y.*, options_value, self.device, self.rows);
        var corr_columns_transferred: usize = 0;
        errdefer {
            for (corr_columns[corr_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + corr_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var corr_names = try rollingCorrelationProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, corr_names[0..]);
        for (corr_names, 0..) |corr_name, i| source_names[self.columns.len + i] = corr_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + corr_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&corr_columns) |*corr_col| {
            columns[initialized] = corr_col.*;
            initialized += 1;
            corr_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingCorrelationProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        const x = try self.column(x_name);
        const y = try self.column(y_name);
        if (x.dtype() != y.dtype()) return error.TypeMismatch;
        var corr_columns = try expandingCorrelationProfileColumnsByValue(self.allocator, x.*, y.*, options_value, self.device, self.rows);
        var corr_columns_transferred: usize = 0;
        errdefer {
            for (corr_columns[corr_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + corr_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var corr_names = try expandingCorrelationProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, corr_names[0..]);
        for (corr_names, 0..) |corr_name, i| source_names[self.columns.len + i] = corr_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + corr_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&corr_columns) |*corr_col| {
            columns[initialized] = corr_col.*;
            initialized += 1;
            corr_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingLinearFitProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        const x = try self.column(x_name);
        const y = try self.column(y_name);
        if (x.dtype() != y.dtype()) return error.TypeMismatch;
        var fit_columns = try expandingLinearFitProfileColumnsByValue(self.allocator, x.*, y.*, options_value, self.device, self.rows);
        var fit_columns_transferred: usize = 0;
        errdefer {
            for (fit_columns[fit_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + fit_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var fit_names = try expandingLinearFitProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, fit_names[0..]);
        for (fit_names, 0..) |fit_name, i| source_names[self.columns.len + i] = fit_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + fit_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&fit_columns) |*fit_col| {
            columns[initialized] = fit_col.*;
            initialized += 1;
            fit_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingLinearFitProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingCorrelationOptions,
    ) DeviceDataError!DeviceDataFrame {
        const x = try self.column(x_name);
        const y = try self.column(y_name);
        if (x.dtype() != y.dtype()) return error.TypeMismatch;
        var fit_columns = try rollingLinearFitProfileColumnsByValue(self.allocator, x.*, y.*, options_value, self.device, self.rows);
        var fit_columns_transferred: usize = 0;
        errdefer {
            for (fit_columns[fit_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + fit_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var fit_names = try rollingLinearFitProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, fit_names[0..]);
        for (fit_names, 0..) |fit_name, i| source_names[self.columns.len + i] = fit_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + fit_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&fit_columns) |*fit_col| {
            columns[initialized] = fit_col.*;
            initialized += 1;
            fit_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn validityProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        const source = try self.column(name);
        var validity_columns = try validityProfileColumnsByValue(self.allocator, source.*, self.device, self.rows);
        var validity_columns_transferred: usize = 0;
        errdefer {
            for (validity_columns[validity_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + validity_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var validity_names = try validityProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, validity_names[0..]);
        for (validity_names, 0..) |validity_name, i| source_names[self.columns.len + i] = validity_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + validity_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&validity_columns) |*validity_col| {
            columns[initialized] = validity_col.*;
            initialized += 1;
            validity_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn rollingValidityProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        const source = try self.column(name);
        var validity_columns = try rollingValidityProfileColumnsByValue(self.allocator, source.*, options_value, self.device, self.rows);
        var validity_columns_transferred: usize = 0;
        errdefer {
            for (validity_columns[validity_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + validity_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var validity_names = try rollingValidityProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, validity_names[0..]);
        for (validity_names, 0..) |validity_name, i| source_names[self.columns.len + i] = validity_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + validity_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&validity_columns) |*validity_col| {
            columns[initialized] = validity_col.*;
            initialized += 1;
            validity_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn expandingValidityProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        const source = try self.column(name);
        var validity_columns = try expandingValidityProfileColumnsByValue(self.allocator, source.*, options_value, self.device, self.rows);
        var validity_columns_transferred: usize = 0;
        errdefer {
            for (validity_columns[validity_columns_transferred..]) |*col| col.deinit();
        }

        const source_names = try self.allocator.alloc([]const u8, self.columns.len + validity_columns.len);
        defer self.allocator.free(source_names);
        for (self.names, 0..) |source_name, i| source_names[i] = source_name;

        var validity_names = try expandingValidityProfileOutputNames(self.allocator, output_prefix);
        defer freeOwnedNameItems(self.allocator, validity_names[0..]);
        for (validity_names, 0..) |validity_name, i| source_names[self.columns.len + i] = validity_name;

        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len + validity_columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.clone();
            initialized += 1;
        }
        for (&validity_columns) |*validity_col| {
            columns[initialized] = validity_col.*;
            initialized += 1;
            validity_columns_transferred += 1;
        }

        return initDeviceDataFrameFromOwnedColumns(self.allocator, source_names, columns, self.rows, self.device);
    }

    pub fn groupByCount(self: DeviceDataFrame, key_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        return switch (key.*) {
            .bool => |typed| groupByCountTyped(DeviceDataFrame, bool, self.allocator, key_name, output_name, typed, self.device),
            .i8 => |typed| groupByCountTyped(DeviceDataFrame, i8, self.allocator, key_name, output_name, typed, self.device),
            .i16 => |typed| groupByCountTyped(DeviceDataFrame, i16, self.allocator, key_name, output_name, typed, self.device),
            .i32 => |typed| groupByCountTyped(DeviceDataFrame, i32, self.allocator, key_name, output_name, typed, self.device),
            .i64 => |typed| groupByCountTyped(DeviceDataFrame, i64, self.allocator, key_name, output_name, typed, self.device),
            .u8 => |typed| groupByCountTyped(DeviceDataFrame, u8, self.allocator, key_name, output_name, typed, self.device),
            .u16 => |typed| groupByCountTyped(DeviceDataFrame, u16, self.allocator, key_name, output_name, typed, self.device),
            .u32 => |typed| groupByCountTyped(DeviceDataFrame, u32, self.allocator, key_name, output_name, typed, self.device),
            .u64 => |typed| groupByCountTyped(DeviceDataFrame, u64, self.allocator, key_name, output_name, typed, self.device),
            .usize => |typed| groupByCountTyped(DeviceDataFrame, usize, self.allocator, key_name, output_name, typed, self.device),
            .isize => |typed| groupByCountTyped(DeviceDataFrame, isize, self.allocator, key_name, output_name, typed, self.device),
            .f16 => |typed| groupByCountTyped(DeviceDataFrame, f16, self.allocator, key_name, output_name, typed, self.device),
            .f32 => |typed| groupByCountTyped(DeviceDataFrame, f32, self.allocator, key_name, output_name, typed, self.device),
            .f64 => |typed| groupByCountTyped(DeviceDataFrame, f64, self.allocator, key_name, output_name, typed, self.device),
            .bf16, .c64, .c128 => error.TypeUnsupported,
        };
    }

    pub fn groupBySum(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByNumericDispatchKey(DeviceDataFrame, .sum, self.allocator, key_name, output_name, key.*, value.*, self.device);
    }

    pub fn groupByMin(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByNumericDispatchKey(DeviceDataFrame, .min, self.allocator, key_name, output_name, key.*, value.*, self.device);
    }

    pub fn groupByMax(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByNumericDispatchKey(DeviceDataFrame, .max, self.allocator, key_name, output_name, key.*, value.*, self.device);
    }

    pub fn groupByMean(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByMeanDispatchKey(DeviceDataFrame, self.allocator, key_name, output_name, key.*, value.*, self.device);
    }

    pub fn groupByStats(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByStatsDispatchKey(DeviceDataFrame, self.allocator, key_name, output_prefix, key.*, value.*, self.device);
    }

    pub fn groupByStatsOn(self: DeviceDataFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        if (key_names.len == 0) return error.LengthMismatch;
        for (key_names) |key_name| _ = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByStatsOnDispatchValue(DeviceDataFrame, self.allocator, self, key_names, output_prefix, value.*, self.device);
    }

    pub fn groupByProfile(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        const key = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByProfileDispatchKey(DeviceDataFrame, self.allocator, key_name, output_prefix, key.*, value.*, self.device);
    }

    pub fn groupByProfileOn(self: DeviceDataFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        if (key_names.len == 0) return error.LengthMismatch;
        for (key_names) |key_name| _ = try self.column(key_name);
        const value = try self.column(value_name);
        return groupByProfileOnDispatchValue(DeviceDataFrame, self.allocator, self, key_names, output_prefix, value.*, self.device);
    }

    pub fn innerJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;

        var pair = try innerJoinRowIndices(self.allocator, left_key.*, right_key.*);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(DeviceDataFrame, self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(DeviceDataFrame, right, pair.right);
        defer right_rows.deinit();

        return concatJoinedTables(DeviceDataFrame, self.allocator, left_rows, right_rows, right_key_name, options_value);
    }

    pub fn innerJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
        for (left_key_names, right_key_names) |left_name, right_name| {
            const left_key = try self.column(left_name);
            const right_key = try right.column(right_name);
            if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        }

        var pair = try innerJoinRowIndicesMulti(self.allocator, self, right, left_key_names, right_key_names);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(DeviceDataFrame, self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(DeviceDataFrame, right, pair.right);
        defer right_rows.deinit();

        return concatJoinedTablesExcludingKeys(DeviceDataFrame, self.allocator, left_rows, right_rows, right_key_names, options_value);
    }

    pub fn leftJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;

        var pair = try leftJoinRowIndices(self.allocator, left_key.*, right_key.*);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(DeviceDataFrame, self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(DeviceDataFrame, right, pair.right);
        defer right_rows.deinit();

        return concatJoinedTables(DeviceDataFrame, self.allocator, left_rows, right_rows, right_key_name, options_value);
    }

    pub fn leftJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
        for (left_key_names, right_key_names) |left_name, right_name| {
            const left_key = try self.column(left_name);
            const right_key = try right.column(right_name);
            if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        }

        var pair = try leftJoinRowIndicesMulti(self.allocator, self, right, left_key_names, right_key_names);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(DeviceDataFrame, self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(DeviceDataFrame, right, pair.right);
        defer right_rows.deinit();

        return concatJoinedTablesExcludingKeys(DeviceDataFrame, self.allocator, left_rows, right_rows, right_key_names, options_value);
    }

    pub fn fullJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;

        var pair = try fullJoinRowIndices(self.allocator, left_key.*, right_key.*);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(DeviceDataFrame, self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(DeviceDataFrame, right, pair.right);
        defer right_rows.deinit();

        return concatFullJoinedTables(DeviceDataFrame, self.allocator, left_rows, right_rows, left_key_name, right_key_name, options_value);
    }

    pub fn fullJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
        for (left_key_names, right_key_names) |left_name, right_name| {
            const left_key = try self.column(left_name);
            const right_key = try right.column(right_name);
            if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        }

        var pair = try fullJoinRowIndicesMulti(self.allocator, self, right, left_key_names, right_key_names);
        defer pair.deinit();

        var left_rows = try takeOptionalRows(DeviceDataFrame, self, pair.left);
        defer left_rows.deinit();
        var right_rows = try takeOptionalRows(DeviceDataFrame, right, pair.right);
        defer right_rows.deinit();

        return concatFullJoinedTablesOn(DeviceDataFrame, self.allocator, left_rows, right_rows, left_key_names, right_key_names, options_value);
    }

    pub fn semiJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const indices = try self.semiAntiJoinIndices(right, left_key_name, right_key_name, true);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn semiJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const indices = try self.semiAntiJoinIndicesOn(right, left_key_names, right_key_names, true);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn antiJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const indices = try self.semiAntiJoinIndices(right, left_key_name, right_key_name, false);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn antiJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
    ) DeviceDataError!DeviceDataFrame {
        const indices = try self.semiAntiJoinIndicesOn(right, left_key_names, right_key_names, false);
        defer self.allocator.free(indices);
        return self.take(indices);
    }

    pub fn asofJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceAsofOptions,
    ) DeviceDataError!DeviceDataFrame {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;

        const right_indices = try asofRightRowIndices(self.allocator, left_key.*, right_key.*, options_value.strategy);
        defer self.allocator.free(right_indices);
        var right_rows = try takeOptionalRows(DeviceDataFrame, right, right_indices);
        defer right_rows.deinit();

        return concatJoinedTables(DeviceDataFrame, self.allocator, self, right_rows, right_key_name, .{ .right_suffix = options_value.right_suffix });
    }

    fn semiAntiJoinIndices(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        keep_matches: bool,
    ) DeviceDataError![]usize {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        const left_key = try self.column(left_key_name);
        const right_key = try right.column(right_key_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        return semiAntiJoinRowIndices(self.allocator, left_key.*, right_key.*, keep_matches);
    }

    fn semiAntiJoinIndicesOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        keep_matches: bool,
    ) DeviceDataError![]usize {
        if (!self.device.sameDevice(right.device)) return error.InvalidDevice;
        if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
        for (left_key_names, right_key_names) |left_name, right_name| {
            const left_key = try self.column(left_name);
            const right_key = try right.column(right_name);
            if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
        }
        return semiAntiJoinRowIndicesMulti(self.allocator, self, right, left_key_names, right_key_names, keep_matches);
    }

    pub fn filter(self: DeviceDataFrame, mask: []const bool) DeviceDataError!DeviceDataFrame {
        if (mask.len != self.rows) return error.LengthMismatch;
        const row_indices = try rowIndicesFromMask(self.allocator, mask);
        defer self.allocator.free(row_indices);
        return self.take(row_indices);
    }

    pub fn to(self: DeviceDataFrame, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        if (!device_value.isAvailable()) return error.InvalidDevice;
        var columns = try self.allocator.alloc(DeviceColumn, self.columns.len);
        var initialized: usize = 0;
        errdefer {
            for (columns[0..initialized]) |*col| col.deinit();
            self.allocator.free(columns);
        }
        for (self.columns, 0..) |col, i| {
            columns[i] = try col.to(device_value);
            initialized += 1;
        }
        return initDeviceDataFrameFromOwnedColumns(self.allocator, self.names, columns, self.rows, device_value);
    }

    pub fn cpu(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.to(.cpu);
    }

    pub fn cuda(self: DeviceDataFrame, index: usize) DeviceDataError!DeviceDataFrame {
        return self.to(array_mod.Device.cuda(index));
    }

    pub fn mps(self: DeviceDataFrame, index: usize) DeviceDataError!DeviceDataFrame {
        return self.to(array_mod.Device.mps(index));
    }

    pub fn toDataFrame(self: DeviceDataFrame) DeviceDataError!DataFrame {
        var defs = try self.allocator.alloc(ColumnDef, self.columns.len);
        defer self.allocator.free(defs);
        var initialized: usize = 0;
        defer {
            for (defs[0..initialized]) |def| freeColumn(self.allocator, def.data);
        }

        for (self.names, self.columns, 0..) |name, col, i| {
            if (col.hasNulls()) return error.TypeUnsupported;
            defs[i].name = name;
            defs[i].data = switch (col) {
                .f64 => |typed| .{ .f64 = try typed.toOwnedSlice(self.allocator) },
                .i64 => |typed| .{ .i64 = try typed.toOwnedSlice(self.allocator) },
                .bool => |typed| .{ .bool = try typed.toOwnedSlice(self.allocator) },
                else => return error.TypeUnsupported,
            };
            initialized += 1;
        }
        return DataFrame.init(self.allocator, defs);
    }
};

const compareFloatSortValues = numeric_mod.compareFloatSortValues;
const groupByStatsOnDispatchValue = group_multi_mod.groupByStatsOnDispatchValue;
const groupByProfileOnDispatchValue = group_multi_mod.groupByProfileOnDispatchValue;
const groupByCountTyped = group_profile_mod.groupByCountTyped;
const initAggregatedDataFrame = group_profile_mod.initAggregatedDataFrame;
const groupByNumericDispatchKey = group_profile_mod.groupByNumericDispatchKey;
const groupByMeanDispatchKey = group_profile_mod.groupByMeanDispatchKey;
const groupByStatsDispatchKey = group_profile_mod.groupByStatsDispatchKey;
const groupByProfileDispatchKey = group_profile_mod.groupByProfileDispatchKey;
const innerJoinRowIndices = join_mod.innerJoinRowIndices;
const innerJoinRowIndicesMulti = join_mod.innerJoinRowIndicesMulti;
const leftJoinRowIndices = join_mod.leftJoinRowIndices;
const leftJoinRowIndicesMulti = join_mod.leftJoinRowIndicesMulti;
const fullJoinRowIndices = join_mod.fullJoinRowIndices;
const fullJoinRowIndicesMulti = join_mod.fullJoinRowIndicesMulti;
const semiAntiJoinRowIndices = join_mod.semiAntiJoinRowIndices;
const semiAntiJoinRowIndicesMulti = join_mod.semiAntiJoinRowIndicesMulti;
const concatJoinedTables = join_mod.concatJoinedTables;
const concatJoinedTablesExcludingKeys = join_mod.concatJoinedTablesExcludingKeys;
const concatFullJoinedTables = join_mod.concatFullJoinedTables;
const concatFullJoinedTablesOn = join_mod.concatFullJoinedTablesOn;
const findGroupIndex = numeric_mod.findGroupIndex;
const groupKeyEqual = numeric_mod.groupKeyEqual;
const castToF64 = numeric_mod.castToF64;

fn initDeviceDataFrameFromOwnedColumns(
    allocator: std.mem.Allocator,
    source_names: []const []const u8,
    columns: []DeviceColumn,
    rows: usize,
    device_value: array_mod.Device,
) DeviceDataError!DeviceDataFrame {
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, allocator, source_names, columns, rows, device_value);
}

pub fn deviceDataFrame(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
    return DeviceDataFrame.init(allocator, defs);
}
