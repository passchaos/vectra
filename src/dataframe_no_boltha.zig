const std = @import("std");
const array_mod = @import("array.zig");
const host_mod = @import("dataframe_no_boltha_host.zig");
const options_mod = @import("dataframe_no_boltha_options.zig");
const view_core_mod = @import("dataframe_view_core.zig");
const column_mod = @import("dataframe_no_boltha_column.zig");
const lazy_options_mod = @import("dataframe_no_boltha_lazy_options.zig");
const lazy_mod = @import("dataframe_no_boltha_lazy.zig");
const dataframe_stub_mod = @import("dataframe_no_boltha_dataframe.zig");

pub const DataError = host_mod.DataError;
pub const DType = host_mod.DType;
pub const Column = host_mod.Column;
pub const ColumnDef = host_mod.ColumnDef;
pub const DataFrame = host_mod.DataFrame;
pub const dataframe = host_mod.dataframe;

pub const DeviceDType = array_mod.DType;
pub const DeviceDataError = DataError || array_mod.ArrayError;
pub const ArrowInteropError = DeviceDataError || error{FeatureUnavailable};
pub const ParquetInteropError = ArrowInteropError;

pub const DeviceValidityEncoding = options_mod.DeviceValidityEncoding;
pub const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
pub const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
pub const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;
pub const DeviceDTypeClass = options_mod.DeviceDTypeClass;
pub const DeviceScalar = options_mod.DeviceScalar;
pub const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;
pub const NullPlacement = options_mod.NullPlacement;
pub const DeviceSortOptions = options_mod.DeviceSortOptions;
pub const DeviceClipOptions = options_mod.DeviceClipOptions;
pub const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
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
pub const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
pub const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
pub const DeviceRollingRobustOptions = options_mod.DeviceRollingRobustOptions;
pub const DeviceJoinOptions = options_mod.DeviceJoinOptions;
pub const AsofStrategy = options_mod.AsofStrategy;
pub const DeviceAsofOptions = options_mod.DeviceAsofOptions;
pub const Range = options_mod.Range;
pub const ParquetRangePredicate = options_mod.ParquetRangePredicate;
pub const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
pub const DeviceParquetNullFilter = options_mod.DeviceParquetNullFilter;

const view_types = view_core_mod.DeviceViewTypes(DeviceValidityEncoding, DeviceDTypeClass, DataError);
pub const DeviceDataFrameViewError = view_types.DeviceDataFrameViewError;
pub const DeviceColumnView = view_types.DeviceColumnView;
pub const DeviceDataFrameView = view_types.DeviceDataFrameView;
pub const DeviceTypedColumn = column_mod.DeviceTypedColumn;
pub const DeviceColumn = column_mod.DeviceColumn;
pub const DeviceColumnDef = column_mod.DeviceColumnDef;
pub const DeviceColumnSchema = column_mod.DeviceColumnSchema;
pub const DeviceLazyGroupByAggregation = lazy_options_mod.DeviceLazyGroupByAggregation;
pub const DeviceLazyWeightedGroupByAggregation = lazy_options_mod.DeviceLazyWeightedGroupByAggregation;
pub const DeviceLazyPairGroupByAggregation = lazy_options_mod.DeviceLazyPairGroupByAggregation;
pub const DeviceLazyWeightedPairGroupByAggregation = lazy_options_mod.DeviceLazyWeightedPairGroupByAggregation;
pub const DeviceLazyJoinKind = lazy_options_mod.DeviceLazyJoinKind;

const lazy_types = lazy_mod.DeviceLazyParquetTypes(
    DeviceDataFrame,
    DeviceColumn,
    ParquetRangePredicate,
    DeviceDataError,
    ParquetInteropError,
);
pub const DeviceLazySource = lazy_types.DeviceLazySource;
pub const DeviceLazyFrame = lazy_types.DeviceLazyFrame;
pub const DeviceParquetScan = lazy_types.DeviceParquetScan;

pub const DeviceDataFrame = dataframe_stub_mod.DeviceDataFrameType(
    DataFrame,
    DeviceColumn,
    DeviceColumnDef,
    DeviceColumnView,
    DeviceColumnSchema,
    DeviceDType,
    DeviceDTypeClass,
    ParquetRangePredicate,
    DataError,
    DeviceDataError,
    ParquetInteropError,
);

pub fn deviceDataFrame(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
    return DeviceDataFrame.init(allocator, defs);
}
