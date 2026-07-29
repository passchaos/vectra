const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_arrow_mod = @import("dataframe_arrow.zig");
const dataframe_core_mod = @import("dataframe_core.zig");
const dataframe_host_mod = @import("dataframe_host.zig");
const expr_mod = @import("dataframe_expr.zig");
const options_mod = @import("dataframe_options.zig");
const dataframe_view_mod = @import("dataframe_view.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const keys_mod = @import("dataframe_keys.zig");
const lazy_frame_mod = @import("dataframe_lazy_frame.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const boltha = @import("boltha");
const rank_mod = @import("dataframe_rank.zig");
const profile_methods_mod = @import("dataframe_profile_methods.zig");
const relation_methods_mod = @import("dataframe_relation_methods.zig");

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
        return dataframe_array_mod.concatDeviceDataFramesRows(DeviceDataFrame, self, other);
    }

    pub fn appendRows(self: DeviceDataFrame, other: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.concatRows(other);
    }

    pub fn vstack(self: DeviceDataFrame, other: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return self.concatRows(other);
    }

    pub fn distinctRows(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return keys_mod.distinctRows(DeviceDataFrame, self);
    }

    pub fn distinctOn(self: DeviceDataFrame, key_names: []const []const u8) DeviceDataError!DeviceDataFrame {
        return keys_mod.distinctOn(DeviceDataFrame, self, key_names);
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

    pub const rollingProfile = profile_methods_mod.rollingProfile;
    pub const rollingMomentProfile = profile_methods_mod.rollingMomentProfile;
    pub const rollingRangeProfile = profile_methods_mod.rollingRangeProfile;
    pub const rollingNormalizeProfile = profile_methods_mod.rollingNormalizeProfile;
    pub const expandingNormalizeProfile = profile_methods_mod.expandingNormalizeProfile;
    pub const rollingQuantileProfile = profile_methods_mod.rollingQuantileProfile;
    pub const expandingQuantileProfile = profile_methods_mod.expandingQuantileProfile;
    pub const rollingBoolProfile = profile_methods_mod.rollingBoolProfile;
    pub const rollingDrawdownProfile = profile_methods_mod.rollingDrawdownProfile;
    pub const rollingRobustProfile = profile_methods_mod.rollingRobustProfile;
    pub const rollingRankProfile = profile_methods_mod.rollingRankProfile;
    pub const lagProfile = profile_methods_mod.lagProfile;
    pub const leadProfile = profile_methods_mod.leadProfile;
    pub const clipProfile = profile_methods_mod.clipProfile;
    pub const rollingClipProfile = profile_methods_mod.rollingClipProfile;
    pub const expandingClipProfile = profile_methods_mod.expandingClipProfile;
    pub const thresholdProfile = profile_methods_mod.thresholdProfile;
    pub const rollingThresholdProfile = profile_methods_mod.rollingThresholdProfile;
    pub const expandingThresholdProfile = profile_methods_mod.expandingThresholdProfile;
    pub const expandingProfile = profile_methods_mod.expandingProfile;
    pub const expandingBoolProfile = profile_methods_mod.expandingBoolProfile;
    pub const expandingRankProfile = profile_methods_mod.expandingRankProfile;
    pub const expandingRobustProfile = profile_methods_mod.expandingRobustProfile;
    pub const expandingMomentProfile = profile_methods_mod.expandingMomentProfile;
    pub const standardizeProfile = profile_methods_mod.standardizeProfile;
    pub const robustProfile = profile_methods_mod.robustProfile;
    pub const drawdownProfile = profile_methods_mod.drawdownProfile;
    pub const extremaProfile = profile_methods_mod.extremaProfile;
    pub const trendProfile = profile_methods_mod.trendProfile;
    pub const changePointProfile = profile_methods_mod.changePointProfile;
    pub const rollingChangePointProfile = profile_methods_mod.rollingChangePointProfile;
    pub const expandingChangePointProfile = profile_methods_mod.expandingChangePointProfile;
    pub const rollingTrendProfile = profile_methods_mod.rollingTrendProfile;
    pub const expandingTrendProfile = profile_methods_mod.expandingTrendProfile;
    pub const signProfile = profile_methods_mod.signProfile;
    pub const rollingSignProfile = profile_methods_mod.rollingSignProfile;
    pub const expandingSignProfile = profile_methods_mod.expandingSignProfile;
    pub const crossoverProfile = profile_methods_mod.crossoverProfile;
    pub const rollingCrossoverProfile = profile_methods_mod.rollingCrossoverProfile;
    pub const expandingCrossoverProfile = profile_methods_mod.expandingCrossoverProfile;
    pub const bucketProfile = profile_methods_mod.bucketProfile;
    pub const emaProfile = profile_methods_mod.emaProfile;
    pub const linearFitProfile = profile_methods_mod.linearFitProfile;
    pub const errorProfile = profile_methods_mod.errorProfile;
    pub const rollingErrorProfile = profile_methods_mod.rollingErrorProfile;
    pub const expandingErrorProfile = profile_methods_mod.expandingErrorProfile;
    pub const classificationProfile = profile_methods_mod.classificationProfile;
    pub const rollingClassificationProfile = profile_methods_mod.rollingClassificationProfile;
    pub const expandingClassificationProfile = profile_methods_mod.expandingClassificationProfile;
    pub const boolTransitionProfile = profile_methods_mod.boolTransitionProfile;
    pub const rollingBoolTransitionProfile = profile_methods_mod.rollingBoolTransitionProfile;
    pub const expandingBoolTransitionProfile = profile_methods_mod.expandingBoolTransitionProfile;
    pub const rollingCorrelationProfile = profile_methods_mod.rollingCorrelationProfile;
    pub const expandingCorrelationProfile = profile_methods_mod.expandingCorrelationProfile;
    pub const expandingLinearFitProfile = profile_methods_mod.expandingLinearFitProfile;
    pub const rollingLinearFitProfile = profile_methods_mod.rollingLinearFitProfile;
    pub const validityProfile = profile_methods_mod.validityProfile;
    pub const rollingValidityProfile = profile_methods_mod.rollingValidityProfile;
    pub const expandingValidityProfile = profile_methods_mod.expandingValidityProfile;
    pub const groupByCount = relation_methods_mod.groupByCount;
    pub const groupBySum = relation_methods_mod.groupBySum;
    pub const groupByMin = relation_methods_mod.groupByMin;
    pub const groupByMax = relation_methods_mod.groupByMax;
    pub const groupByMean = relation_methods_mod.groupByMean;
    pub const groupByStats = relation_methods_mod.groupByStats;
    pub const groupByStatsOn = relation_methods_mod.groupByStatsOn;
    pub const groupByProfile = relation_methods_mod.groupByProfile;
    pub const groupByProfileOn = relation_methods_mod.groupByProfileOn;
    pub const innerJoin = relation_methods_mod.innerJoin;
    pub const innerJoinOn = relation_methods_mod.innerJoinOn;
    pub const leftJoin = relation_methods_mod.leftJoin;
    pub const leftJoinOn = relation_methods_mod.leftJoinOn;
    pub const fullJoin = relation_methods_mod.fullJoin;
    pub const fullJoinOn = relation_methods_mod.fullJoinOn;
    pub const semiJoin = relation_methods_mod.semiJoin;
    pub const semiJoinOn = relation_methods_mod.semiJoinOn;
    pub const antiJoin = relation_methods_mod.antiJoin;
    pub const antiJoinOn = relation_methods_mod.antiJoinOn;
    pub const asofJoin = relation_methods_mod.asofJoin;
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

pub fn deviceDataFrame(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
    return DeviceDataFrame.init(allocator, defs);
}
