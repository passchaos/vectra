const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_arrow_mod = @import("dataframe_arrow.zig");
const dataframe_column_mod = @import("dataframe_column.zig");
const dataframe_host_mod = @import("dataframe_host.zig");
const expr_mod = @import("dataframe_expr.zig");
const options_mod = @import("dataframe_options.zig");
const dataframe_view_mod = @import("dataframe_view.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const keys_mod = @import("dataframe_keys.zig");
const join_mod = @import("dataframe_join.zig");
const lazy_frame_mod = @import("dataframe_lazy_frame.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
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
const RollingRankProfileColumnCount = rank_mod.RollingRankProfileColumnCount;
const rollingRankProfileOutputNames = rank_mod.rollingRankProfileOutputNames;
const ExpandingRankProfileColumnCount = rank_mod.ExpandingRankProfileColumnCount;
const expandingRankProfileOutputNames = rank_mod.expandingRankProfileOutputNames;
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
const freeOwnedNameItems = names_mod.freeOwnedNameItems;
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

const lazy_frame_types = lazy_frame_mod.DeviceLazyTypes(DeviceDataFrame, DeviceColumnDef, DeviceColumn);
pub const DeviceLazySource = lazy_frame_types.DeviceLazySource;
pub const DeviceLazyFrame = lazy_frame_types.DeviceLazyFrame;
pub const DeviceParquetScan = lazy_frame_types.DeviceParquetScan;

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
        return expr_mod.binaryColumns(self, lhs_name, rhs_name, op);
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
        return expr_mod.binaryColumnScalar(self, name, T, scalar, op);
    }

    pub fn binaryColumnScalarWithDeviceScalar(self: DeviceDataFrame, name: []const u8, scalar: DeviceScalar, op: DeviceColumnBinaryOp) DeviceDataError!DeviceColumn {
        return expr_mod.binaryColumnScalarWithDeviceScalar(self, name, scalar, op);
    }

    pub fn compareColumns(self: DeviceDataFrame, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        return expr_mod.compareColumns(self, lhs_name, rhs_name, op);
    }

    pub fn compareColumnScalar(self: DeviceDataFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        return expr_mod.compareColumnScalar(self, name, T, scalar, op);
    }

    pub fn compareColumnScalarWithDeviceScalar(self: DeviceDataFrame, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!DeviceColumn {
        return expr_mod.compareColumnScalarWithDeviceScalar(self, name, scalar, op);
    }

    pub fn filterColumnMask(self: DeviceDataFrame, mask: DeviceColumn) DeviceDataError!DeviceDataFrame {
        return expr_mod.filterColumnMask(DeviceDataFrame, self, mask);
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
        return dataframe_array_mod.view(DeviceDataFrameView, DeviceColumnView, self);
    }

    pub fn select(self: DeviceDataFrame, wanted_names: []const []const u8) DeviceDataError!DeviceDataFrame {
        return dataframe_array_mod.select(DeviceDataFrame, self, wanted_names);
    }

    pub fn withColumn(self: DeviceDataFrame, name: []const u8, data: DeviceColumn) DeviceDataError!DeviceDataFrame {
        return dataframe_array_mod.withColumn(DeviceDataFrame, self, name, data);
    }

    pub fn head(self: DeviceDataFrame, n: usize) DeviceDataError!DeviceDataFrame {
        return self.sliceRows(0, @min(n, self.rows));
    }

    pub fn tail(self: DeviceDataFrame, n: usize) DeviceDataError!DeviceDataFrame {
        const count = @min(n, self.rows);
        return self.sliceRows(self.rows - count, self.rows);
    }

    pub fn sliceRows(self: DeviceDataFrame, start: usize, stop: usize) DeviceDataError!DeviceDataFrame {
        return dataframe_array_mod.sliceRows(DeviceDataFrame, self, start, stop);
    }

    pub fn take(self: DeviceDataFrame, row_indices: []const usize) DeviceDataError!DeviceDataFrame {
        return dataframe_array_mod.takeRows(DeviceDataFrame, self, row_indices);
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
        return rank_mod.argsortBy(self, name, options_value);
    }

    pub fn sortBy(self: DeviceDataFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        return rank_mod.sortBy(DeviceDataFrame, self, name, options_value);
    }

    pub fn sortByColumn(self: DeviceDataFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        return self.sortBy(name, options_value);
    }

    pub fn topKBy(self: DeviceDataFrame, name: []const u8, k: usize, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        return rank_mod.topKBy(DeviceDataFrame, self, name, k, options_value);
    }

    pub fn rankProfileBy(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceSortOptions) DeviceDataError!DeviceDataFrame {
        return rank_mod.rankProfileBy(DeviceDataFrame, self, name, output_prefix, options_value);
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
        return group_profile_mod.groupByCount(DeviceDataFrame, self, key_name, output_name);
    }

    pub fn groupBySum(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        return group_profile_mod.groupByNumeric(DeviceDataFrame, .sum, self, key_name, value_name, output_name);
    }

    pub fn groupByMin(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        return group_profile_mod.groupByNumeric(DeviceDataFrame, .min, self, key_name, value_name, output_name);
    }

    pub fn groupByMax(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        return group_profile_mod.groupByNumeric(DeviceDataFrame, .max, self, key_name, value_name, output_name);
    }

    pub fn groupByMean(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_name: []const u8) DeviceDataError!DeviceDataFrame {
        return group_profile_mod.groupByMean(DeviceDataFrame, self, key_name, value_name, output_name);
    }

    pub fn groupByStats(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        return group_profile_mod.groupByStats(DeviceDataFrame, self, key_name, value_name, output_prefix);
    }

    pub fn groupByStatsOn(self: DeviceDataFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        return group_multi_mod.groupByStatsOn(DeviceDataFrame, self, key_names, value_name, output_prefix);
    }

    pub fn groupByProfile(self: DeviceDataFrame, key_name: []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        return group_profile_mod.groupByProfile(DeviceDataFrame, self, key_name, value_name, output_prefix);
    }

    pub fn groupByProfileOn(self: DeviceDataFrame, key_names: []const []const u8, value_name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        return group_multi_mod.groupByProfileOn(DeviceDataFrame, self, key_names, value_name, output_prefix);
    }

    pub fn innerJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.innerJoin(DeviceDataFrame, self, right, left_key_name, right_key_name, options_value);
    }

    pub fn innerJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.innerJoinOn(DeviceDataFrame, self, right, left_key_names, right_key_names, options_value);
    }

    pub fn leftJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.leftJoin(DeviceDataFrame, self, right, left_key_name, right_key_name, options_value);
    }

    pub fn leftJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.leftJoinOn(DeviceDataFrame, self, right, left_key_names, right_key_names, options_value);
    }

    pub fn fullJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.fullJoin(DeviceDataFrame, self, right, left_key_name, right_key_name, options_value);
    }

    pub fn fullJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
        options_value: DeviceJoinOptions,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.fullJoinOn(DeviceDataFrame, self, right, left_key_names, right_key_names, options_value);
    }

    pub fn semiJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.semiJoin(DeviceDataFrame, self, right, left_key_name, right_key_name);
    }

    pub fn semiJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.semiJoinOn(DeviceDataFrame, self, right, left_key_names, right_key_names);
    }

    pub fn antiJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.antiJoin(DeviceDataFrame, self, right, left_key_name, right_key_name);
    }

    pub fn antiJoinOn(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_names: []const []const u8,
        right_key_names: []const []const u8,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.antiJoinOn(DeviceDataFrame, self, right, left_key_names, right_key_names);
    }

    pub fn asofJoin(
        self: DeviceDataFrame,
        right: DeviceDataFrame,
        left_key_name: []const u8,
        right_key_name: []const u8,
        options_value: DeviceAsofOptions,
    ) DeviceDataError!DeviceDataFrame {
        return join_mod.asofJoin(DeviceDataFrame, self, right, left_key_name, right_key_name, options_value);
    }

    pub fn filter(self: DeviceDataFrame, mask: []const bool) DeviceDataError!DeviceDataFrame {
        return dataframe_array_mod.filterRows(DeviceDataFrame, self, mask);
    }

    pub fn to(self: DeviceDataFrame, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        return dataframe_array_mod.toDevice(DeviceDataFrame, self, device_value);
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
