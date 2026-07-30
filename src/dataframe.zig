const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const dataframe_core_mod = @import("dataframe_core.zig");
const dataframe_host_mod = @import("dataframe_host.zig");
const options_mod = @import("dataframe_options.zig");
const dataframe_view_mod = @import("dataframe_view.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const lazy_frame_mod = @import("dataframe_lazy_frame.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const boltha = @import("boltha");
const profile_methods_mod = @import("dataframe_profile_methods.zig");
const relation_methods_mod = @import("dataframe_relation_methods.zig");
const table_methods_mod = @import("dataframe_table_methods.zig");
const arrow_methods_mod = @import("dataframe_arrow_methods.zig");

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
pub const DeviceDTypeClass = options_mod.DeviceDTypeClass;
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

    pub const binaryColumns = table_methods_mod.binaryColumns;
    pub const addColumns = table_methods_mod.addColumns;
    pub const subColumns = table_methods_mod.subColumns;
    pub const mulColumns = table_methods_mod.mulColumns;
    pub const divColumns = table_methods_mod.divColumns;
    pub const binaryColumnScalar = table_methods_mod.binaryColumnScalar;
    pub const binaryColumnScalarWithDeviceScalar = table_methods_mod.binaryColumnScalarWithDeviceScalar;
    pub const compareColumns = table_methods_mod.compareColumns;
    pub const compareColumnScalar = table_methods_mod.compareColumnScalar;
    pub const compareColumnScalarWithDeviceScalar = table_methods_mod.compareColumnScalarWithDeviceScalar;
    pub const filterColumnMask = table_methods_mod.filterColumnMask;
    pub const filterColumn = table_methods_mod.filterColumn;
    pub const toArrowSchema = arrow_methods_mod.toArrowSchema;
    pub const toArrowRecordBatch = arrow_methods_mod.toArrowRecordBatch;
    pub const toArrowTable = arrow_methods_mod.toArrowTable;
    pub const toParquetBytes = arrow_methods_mod.toParquetBytes;

    pub fn fromParquetBytes(allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) ParquetInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromParquetBytes(DeviceDataFrame, DeviceColumnDef, allocator, bytes, device_value);
    }

    pub fn fromParquetBytesPruned(
        allocator: std.mem.Allocator,
        bytes: []const u8,
        column_name: []const u8,
        predicate: ParquetRangePredicate,
        device_value: array_mod.Device,
    ) ParquetInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromParquetBytesPruned(DeviceDataFrame, DeviceColumnDef, allocator, bytes, column_name, predicate, device_value);
    }

    pub fn fromArrowTable(allocator: std.mem.Allocator, table: boltha.arrow.Table, device_value: array_mod.Device) ArrowInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromArrowTable(DeviceDataFrame, DeviceColumnDef, allocator, table, device_value);
    }

    pub fn fromArrowTableProjection(
        allocator: std.mem.Allocator,
        table: boltha.arrow.Table,
        wanted_names: []const []const u8,
        device_value: array_mod.Device,
    ) ArrowInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromArrowTableProjection(DeviceDataFrame, DeviceColumnDef, allocator, table, wanted_names, device_value);
    }

    pub fn fromArrowRecordBatch(allocator: std.mem.Allocator, batch: boltha.arrow.RecordBatch, device_value: array_mod.Device) ArrowInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromArrowRecordBatch(DeviceDataFrame, DeviceColumnDef, allocator, batch, device_value);
    }

    pub fn fromArrowRecordBatchProjection(
        allocator: std.mem.Allocator,
        batch: boltha.arrow.RecordBatch,
        wanted_names: []const []const u8,
        device_value: array_mod.Device,
    ) ArrowInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromArrowRecordBatchProjection(DeviceDataFrame, DeviceColumnDef, allocator, batch, wanted_names, device_value);
    }

    pub const view = table_methods_mod.view;
    pub const select = table_methods_mod.select;
    pub const selectByColumnIndices = table_methods_mod.selectByColumnIndices;
    pub const selectColumnRange = table_methods_mod.selectColumnRange;
    pub const selectFirstColumns = table_methods_mod.selectFirstColumns;
    pub const selectLastColumns = table_methods_mod.selectLastColumns;
    pub const dropByColumnIndices = table_methods_mod.dropByColumnIndices;
    pub const dropColumnRange = table_methods_mod.dropColumnRange;
    pub const dropFirstColumns = table_methods_mod.dropFirstColumns;
    pub const dropLastColumns = table_methods_mod.dropLastColumns;
    pub const reverseColumns = table_methods_mod.reverseColumns;
    pub const sortColumnsByName = table_methods_mod.sortColumnsByName;
    pub const selectByNamePrefix = table_methods_mod.selectByNamePrefix;
    pub const selectByNameSuffix = table_methods_mod.selectByNameSuffix;
    pub const selectByNameContains = table_methods_mod.selectByNameContains;
    pub const dropByNamePrefix = table_methods_mod.dropByNamePrefix;
    pub const dropByNameSuffix = table_methods_mod.dropByNameSuffix;
    pub const dropByNameContains = table_methods_mod.dropByNameContains;
    pub const selectByDTypes = table_methods_mod.selectByDTypes;
    pub const selectByDTypeClass = table_methods_mod.selectByDTypeClass;
    pub const dropByDTypes = table_methods_mod.dropByDTypes;
    pub const dropByDTypeClass = table_methods_mod.dropByDTypeClass;
    pub const selectNumeric = table_methods_mod.selectNumeric;
    pub const selectReal = table_methods_mod.selectReal;
    pub const selectFloat = table_methods_mod.selectFloat;
    pub const selectInteger = table_methods_mod.selectInteger;
    pub const selectBool = table_methods_mod.selectBool;
    pub const dropNumeric = table_methods_mod.dropNumeric;
    pub const dropReal = table_methods_mod.dropReal;
    pub const dropFloat = table_methods_mod.dropFloat;
    pub const dropInteger = table_methods_mod.dropInteger;
    pub const dropBool = table_methods_mod.dropBool;
    pub const selectNullableColumns = table_methods_mod.selectNullableColumns;
    pub const selectNonNullableColumns = table_methods_mod.selectNonNullableColumns;
    pub const selectColumnsWithNulls = table_methods_mod.selectColumnsWithNulls;
    pub const selectColumnsWithoutNulls = table_methods_mod.selectColumnsWithoutNulls;
    pub const dropNullableColumns = table_methods_mod.dropNullableColumns;
    pub const dropNonNullableColumns = table_methods_mod.dropNonNullableColumns;
    pub const dropColumnsWithNulls = table_methods_mod.dropColumnsWithNulls;
    pub const dropColumnsWithoutNulls = table_methods_mod.dropColumnsWithoutNulls;
    pub const withColumn = table_methods_mod.withColumn;
    pub const withColumnAt = table_methods_mod.withColumnAt;
    pub const withColumnBefore = table_methods_mod.withColumnBefore;
    pub const withColumnAfter = table_methods_mod.withColumnAfter;
    pub const copyColumn = table_methods_mod.copyColumn;
    pub const copyColumnAt = table_methods_mod.copyColumnAt;
    pub const copyColumnBefore = table_methods_mod.copyColumnBefore;
    pub const copyColumnAfter = table_methods_mod.copyColumnAfter;
    pub const castColumn = table_methods_mod.castColumn;
    pub const fillNullColumn = table_methods_mod.fillNullColumn;
    pub const fillNullColumnWithScalar = table_methods_mod.fillNullColumnWithScalar;
    pub const fillNaNColumn = table_methods_mod.fillNaNColumn;
    pub const fillNaNColumnWithScalar = table_methods_mod.fillNaNColumnWithScalar;
    pub const fillInfColumn = table_methods_mod.fillInfColumn;
    pub const fillInfColumnWithScalar = table_methods_mod.fillInfColumnWithScalar;
    pub const coalesceColumns = table_methods_mod.coalesceColumns;
    pub const isNullColumn = table_methods_mod.isNullColumn;
    pub const isValidColumn = table_methods_mod.isValidColumn;
    pub const isNanColumn = table_methods_mod.isNanColumn;
    pub const isFiniteColumn = table_methods_mod.isFiniteColumn;
    pub const isInfColumn = table_methods_mod.isInfColumn;
    pub const withRowNullCount = table_methods_mod.withRowNullCount;
    pub const withRowValidCount = table_methods_mod.withRowValidCount;
    pub const withRowNaNCount = table_methods_mod.withRowNaNCount;
    pub const withRowInfCount = table_methods_mod.withRowInfCount;
    pub const withRowFiniteCount = table_methods_mod.withRowFiniteCount;
    pub const withRowNonFiniteCount = table_methods_mod.withRowNonFiniteCount;
    pub const withColumnLiteral = table_methods_mod.withColumnLiteral;
    pub const withColumnLiteralScalar = table_methods_mod.withColumnLiteralScalar;
    pub const withColumnLiteralAt = table_methods_mod.withColumnLiteralAt;
    pub const withColumnLiteralBefore = table_methods_mod.withColumnLiteralBefore;
    pub const withColumnLiteralAfter = table_methods_mod.withColumnLiteralAfter;
    pub const withColumnLiteralScalarAt = table_methods_mod.withColumnLiteralScalarAt;
    pub const withColumnLiteralScalarBefore = table_methods_mod.withColumnLiteralScalarBefore;
    pub const withColumnLiteralScalarAfter = table_methods_mod.withColumnLiteralScalarAfter;
    pub const withRowIndex = table_methods_mod.withRowIndex;
    pub const renameColumn = table_methods_mod.renameColumn;
    pub const renameColumns = table_methods_mod.renameColumns;
    pub const addColumnNamePrefix = table_methods_mod.addColumnNamePrefix;
    pub const addColumnNameSuffix = table_methods_mod.addColumnNameSuffix;
    pub const moveColumn = table_methods_mod.moveColumn;
    pub const moveColumnBefore = table_methods_mod.moveColumnBefore;
    pub const moveColumnAfter = table_methods_mod.moveColumnAfter;
    pub const dropColumns = table_methods_mod.dropColumns;
    pub const dropColumn = table_methods_mod.dropColumn;
    pub const dropNulls = table_methods_mod.dropNulls;
    pub const dropNullsOn = table_methods_mod.dropNullsOn;
    pub const dropNullsColumn = table_methods_mod.dropNullsColumn;
    pub const filterNullsColumn = table_methods_mod.filterNullsColumn;
    pub const dropNaNs = table_methods_mod.dropNaNs;
    pub const dropNaNsOn = table_methods_mod.dropNaNsOn;
    pub const dropNaNsColumn = table_methods_mod.dropNaNsColumn;
    pub const filterNaNsColumn = table_methods_mod.filterNaNsColumn;
    pub const dropInfs = table_methods_mod.dropInfs;
    pub const dropInfsOn = table_methods_mod.dropInfsOn;
    pub const dropInfsColumn = table_methods_mod.dropInfsColumn;
    pub const filterInfsColumn = table_methods_mod.filterInfsColumn;
    pub const head = table_methods_mod.head;
    pub const tail = table_methods_mod.tail;
    pub const sliceRows = table_methods_mod.sliceRows;
    pub const dropRows = table_methods_mod.dropRows;
    pub const dropRowRange = table_methods_mod.dropRowRange;
    pub const dropFirstRows = table_methods_mod.dropFirstRows;
    pub const dropLastRows = table_methods_mod.dropLastRows;
    pub const sliceRowsStep = table_methods_mod.sliceRowsStep;
    pub const sliceStep = table_methods_mod.sliceStep;
    pub const take = table_methods_mod.take;
    pub const sampleRows = table_methods_mod.sampleRows;
    pub const sampleRowsWithReplacement = table_methods_mod.sampleRowsWithReplacement;
    pub const strideRows = table_methods_mod.strideRows;
    pub const reverseRows = table_methods_mod.reverseRows;
    pub const reverse = table_methods_mod.reverse;
    pub const concatRows = table_methods_mod.concatRows;
    pub const appendRows = table_methods_mod.appendRows;
    pub const vstack = table_methods_mod.vstack;
    pub const distinctRows = table_methods_mod.distinctRows;
    pub const distinctOn = table_methods_mod.distinctOn;
    pub const dropDuplicates = table_methods_mod.dropDuplicates;
    pub const dropDuplicatesOn = table_methods_mod.dropDuplicatesOn;
    pub const uniqueRows = table_methods_mod.uniqueRows;
    pub const argsortBy = table_methods_mod.argsortBy;
    pub const sortBy = table_methods_mod.sortBy;
    pub const sortByColumn = table_methods_mod.sortByColumn;
    pub const topKBy = table_methods_mod.topKBy;
    pub const rankProfileBy = table_methods_mod.rankProfileBy;
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
    pub const filter = table_methods_mod.filter;
    pub const to = table_methods_mod.to;
    pub const cpu = table_methods_mod.cpu;
    pub const cuda = table_methods_mod.cuda;
    pub const mps = table_methods_mod.mps;
    pub fn toDataFrame(self: DeviceDataFrame) DeviceDataError!DataFrame {
        return dataframe_host_mod.deviceDataFrameToDataFrame(self);
    }
};

pub fn deviceDataFrame(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
    return DeviceDataFrame.init(allocator, defs);
}
