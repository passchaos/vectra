const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_arrow_mod = @import("dataframe_arrow.zig");
const dataframe_column_mod = @import("dataframe_column.zig");
const dataframe_core_mod = @import("dataframe_core.zig");
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
const classification_mod = @import("dataframe_classification.zig");
const error_mod = @import("dataframe_error.zig");
const correlation_mod = @import("dataframe_correlation.zig");
const linear_fit_mod = @import("dataframe_linear_fit.zig");
const crossover_mod = @import("dataframe_crossover.zig");
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
const RollingDrawdownProfileColumnCount = risk_mod.RollingDrawdownProfileColumnCount;
const rollingDrawdownProfileOutputNames = risk_mod.rollingDrawdownProfileOutputNames;
const rollingDrawdownProfileColumnsByValue = risk_mod.rollingDrawdownProfileColumnsByValue;
const standardize_mod = @import("dataframe_standardize.zig");
const robust_mod = @import("dataframe_robust.zig");
const RollingRobustProfileColumnCount = robust_mod.RollingRobustProfileColumnCount;
const rollingRobustProfileOutputNames = robust_mod.rollingRobustProfileOutputNames;
const ExpandingRobustProfileColumnCount = robust_mod.ExpandingRobustProfileColumnCount;
const expandingRobustProfileOutputNames = robust_mod.expandingRobustProfileOutputNames;
const rollingRobustProfileColumnsByValue = robust_mod.rollingRobustProfileColumnsByValue;
const expandingRobustProfileColumnsByValue = robust_mod.expandingRobustProfileColumnsByValue;
const trend_mod = @import("dataframe_trend.zig");
const change_mod = @import("dataframe_change.zig");
const sign_mod = @import("dataframe_sign.zig");
const shift_mod = @import("dataframe_shift.zig");
const LagProfileColumnCount = shift_mod.LagProfileColumnCount;
const lagProfileOutputNames = shift_mod.lagProfileOutputNames;
const LeadProfileColumnCount = shift_mod.LeadProfileColumnCount;
const leadProfileOutputNames = shift_mod.leadProfileOutputNames;
const lagProfileColumnsByValue = shift_mod.lagProfileColumnsByValue;
const leadProfileColumnsByValue = shift_mod.leadProfileColumnsByValue;
const ema_mod = @import("dataframe_ema.zig");
const quantile_mod = @import("dataframe_quantile.zig");
const RollingQuantileProfileColumnCount = quantile_mod.RollingQuantileProfileColumnCount;
const rollingQuantileProfileOutputNames = quantile_mod.rollingQuantileProfileOutputNames;
const ExpandingQuantileProfileColumnCount = quantile_mod.ExpandingQuantileProfileColumnCount;
const expandingQuantileProfileOutputNames = quantile_mod.expandingQuantileProfileOutputNames;
const rollingQuantileProfileColumnsByValue = quantile_mod.rollingQuantileProfileColumnsByValue;
const expandingQuantileProfileColumnsByValue = quantile_mod.expandingQuantileProfileColumnsByValue;
const bucket_mod = @import("dataframe_bucket.zig");
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
const validityValues = validity_mod.validityValues;
const freeOwnedNameItems = names_mod.freeOwnedNameItems;
const takeOptionalRows = dataframe_array_mod.takeOptionalRows;
const concatDeviceDataFramesRows = dataframe_array_mod.concatDeviceDataFramesRows;
const concatDeviceColumns = dataframe_array_mod.concatDeviceColumns;
const argsortTypedColumn = dataframe_device_column_mod.argsortTypedColumn;
const distinctRowIndices = keys_mod.distinctRowIndices;
const rowsMatchAllKeys = keys_mod.rowsMatchAllKeys;
const asofRightRowIndices = keys_mod.asofRightRowIndices;

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
        return dataframe_core_mod.init(DeviceDataFrame, allocator, defs);
    }

    pub fn initEmpty(allocator: std.mem.Allocator, rows: usize, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        return dataframe_core_mod.initEmpty(DeviceDataFrame, allocator, rows, device_value);
    }

    pub fn fromDataFrame(allocator: std.mem.Allocator, frame: DataFrame, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        return dataframe_host_mod.deviceDataFrameFromDataFrame(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, frame, device_value);
    }

    pub fn deinit(self: *DeviceDataFrame) void {
        dataframe_core_mod.deinit(self);
    }

    pub fn clone(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return dataframe_core_mod.clone(DeviceDataFrame, self);
    }

    pub fn height(self: DeviceDataFrame) usize {
        return self.rows;
    }

    pub fn width(self: DeviceDataFrame) usize {
        return self.columns.len;
    }

    pub fn shape(self: DeviceDataFrame) struct { rows: usize, cols: usize } {
        return dataframe_core_mod.shape(self);
    }

    pub fn columnIndex(self: DeviceDataFrame, name: []const u8) ?usize {
        return dataframe_core_mod.columnIndex(self, name);
    }

    pub fn column(self: *const DeviceDataFrame, name: []const u8) DataError!*const DeviceColumn {
        return dataframe_core_mod.column(self, name);
    }

    pub fn columnDType(self: DeviceDataFrame, name: []const u8) DataError!DeviceDType {
        return dataframe_core_mod.columnDType(self, name);
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
        return standardize_mod.standardizeProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn robustProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRobustOptions) DeviceDataError!DeviceDataFrame {
        return robust_mod.robustProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn drawdownProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceDrawdownOptions) DeviceDataError!DeviceDataFrame {
        return risk_mod.drawdownProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn extremaProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExtremaOptions) DeviceDataError!DeviceDataFrame {
        return risk_mod.extremaProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn trendProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        return trend_mod.trendProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn changePointProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, threshold: f64, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        return change_mod.changePointProfileFrame(DeviceDataFrame, self, name, output_prefix, threshold, options_value);
    }

    pub fn rollingChangePointProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        return change_mod.rollingChangePointProfileFrame(DeviceDataFrame, self, name, output_prefix, threshold, change_options, options_value);
    }

    pub fn expandingChangePointProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, threshold: f64, change_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        return change_mod.expandingChangePointProfileFrame(DeviceDataFrame, self, name, output_prefix, threshold, change_options, options_value);
    }

    pub fn rollingTrendProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        return trend_mod.rollingTrendProfileFrame(DeviceDataFrame, self, name, output_prefix, trend_options, options_value);
    }

    pub fn expandingTrendProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, trend_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        return trend_mod.expandingTrendProfileFrame(DeviceDataFrame, self, name, output_prefix, trend_options, options_value);
    }

    pub fn signProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        return sign_mod.signProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn rollingSignProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        return sign_mod.rollingSignProfileFrame(DeviceDataFrame, self, name, output_prefix, sign_options, options_value);
    }

    pub fn expandingSignProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, sign_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        return sign_mod.expandingSignProfileFrame(DeviceDataFrame, self, name, output_prefix, sign_options, options_value);
    }

    pub fn crossoverProfile(
        self: DeviceDataFrame,
        lhs_name: []const u8,
        rhs_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceCrossoverOptions,
    ) DeviceDataError!DeviceDataFrame {
        return crossover_mod.crossoverProfileFrame(DeviceDataFrame, self, lhs_name, rhs_name, output_prefix, options_value);
    }

    pub fn rollingCrossoverProfile(
        self: DeviceDataFrame,
        lhs_name: []const u8,
        rhs_name: []const u8,
        output_prefix: []const u8,
        cross_options: DeviceCrossoverOptions,
        options_value: DeviceRollingOptions,
    ) DeviceDataError!DeviceDataFrame {
        return crossover_mod.rollingCrossoverProfileFrame(DeviceDataFrame, self, lhs_name, rhs_name, output_prefix, cross_options, options_value);
    }

    pub fn expandingCrossoverProfile(
        self: DeviceDataFrame,
        lhs_name: []const u8,
        rhs_name: []const u8,
        output_prefix: []const u8,
        cross_options: DeviceCrossoverOptions,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        return crossover_mod.expandingCrossoverProfileFrame(DeviceDataFrame, self, lhs_name, rhs_name, output_prefix, cross_options, options_value);
    }

    pub fn bucketProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceBucketOptions) DeviceDataError!DeviceDataFrame {
        return bucket_mod.bucketProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn emaProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceEmaOptions) DeviceDataError!DeviceDataFrame {
        return ema_mod.emaProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn linearFitProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceLinearFitOptions,
    ) DeviceDataError!DeviceDataFrame {
        return linear_fit_mod.linearFitProfileFrame(DeviceDataFrame, self, x_name, y_name, output_prefix, options_value);
    }

    pub fn errorProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        return error_mod.errorProfileFrame(DeviceDataFrame, self, actual_name, predicted_name, output_prefix);
    }

    pub fn rollingErrorProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingOptions,
    ) DeviceDataError!DeviceDataFrame {
        return error_mod.rollingErrorProfileFrame(DeviceDataFrame, self, actual_name, predicted_name, output_prefix, options_value);
    }

    pub fn expandingErrorProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        return error_mod.expandingErrorProfileFrame(DeviceDataFrame, self, actual_name, predicted_name, output_prefix, options_value);
    }

    pub fn classificationProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
    ) DeviceDataError!DeviceDataFrame {
        return classification_mod.classificationProfileFrame(DeviceDataFrame, self, actual_name, predicted_name, output_prefix);
    }

    pub fn rollingClassificationProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingOptions,
    ) DeviceDataError!DeviceDataFrame {
        return classification_mod.rollingClassificationProfileFrame(DeviceDataFrame, self, actual_name, predicted_name, output_prefix, options_value);
    }

    pub fn expandingClassificationProfile(
        self: DeviceDataFrame,
        actual_name: []const u8,
        predicted_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        return classification_mod.expandingClassificationProfileFrame(DeviceDataFrame, self, actual_name, predicted_name, output_prefix, options_value);
    }

    pub fn boolTransitionProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceTrendOptions) DeviceDataError!DeviceDataFrame {
        return bool_transition_mod.boolTransitionProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn rollingBoolTransitionProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        return bool_transition_mod.rollingBoolTransitionProfileFrame(DeviceDataFrame, self, name, output_prefix, transition_options, options_value);
    }

    pub fn expandingBoolTransitionProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, transition_options: DeviceTrendOptions, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        return bool_transition_mod.expandingBoolTransitionProfileFrame(DeviceDataFrame, self, name, output_prefix, transition_options, options_value);
    }

    pub fn rollingCorrelationProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingCorrelationOptions,
    ) DeviceDataError!DeviceDataFrame {
        return correlation_mod.rollingCorrelationProfileFrame(DeviceDataFrame, self, x_name, y_name, output_prefix, options_value);
    }

    pub fn expandingCorrelationProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        return correlation_mod.expandingCorrelationProfileFrame(DeviceDataFrame, self, x_name, y_name, output_prefix, options_value);
    }

    pub fn expandingLinearFitProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceExpandingOptions,
    ) DeviceDataError!DeviceDataFrame {
        return linear_fit_mod.expandingLinearFitProfileFrame(DeviceDataFrame, self, x_name, y_name, output_prefix, options_value);
    }

    pub fn rollingLinearFitProfile(
        self: DeviceDataFrame,
        x_name: []const u8,
        y_name: []const u8,
        output_prefix: []const u8,
        options_value: DeviceRollingCorrelationOptions,
    ) DeviceDataError!DeviceDataFrame {
        return linear_fit_mod.rollingLinearFitProfileFrame(DeviceDataFrame, self, x_name, y_name, output_prefix, options_value);
    }

    pub fn validityProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8) DeviceDataError!DeviceDataFrame {
        return validity_mod.validityProfileFrame(DeviceDataFrame, self, name, output_prefix);
    }

    pub fn rollingValidityProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingOptions) DeviceDataError!DeviceDataFrame {
        return validity_mod.rollingValidityProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
    }

    pub fn expandingValidityProfile(self: DeviceDataFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingOptions) DeviceDataError!DeviceDataFrame {
        return validity_mod.expandingValidityProfileFrame(DeviceDataFrame, self, name, output_prefix, options_value);
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
        return dataframe_host_mod.deviceDataFrameToDataFrame(self);
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
