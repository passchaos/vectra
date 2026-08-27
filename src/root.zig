//! Vectra is an experimental Zig-native data processing and numerical computing
//! toolkit inspired by NumPy/CuPy/SciPy/Pandas/Polars, with PyTorch-style object
//! methods as the primary public array API.
//!
//! The current implementation provides a compact CPU core: typed arrays,
//! broadcasting arithmetic, reductions, linalg/stat helpers, typed Series,
//! heterogeneous DataFrame operations, sparse matrices, and CSV IO. CUDA/GPU is
//! available through the default Axiom backend when a CUDA device is
//! present; CUDA arrays own device-resident storage and dispatch supported f32/f64
//! operations without staging through host arrays.
//! `DeviceDataFrame` extends the same device model to fixed-width tabular
//! columns on CPU/CUDA/MPS while the existing heterogeneous `DataFrame` remains
//! the compact host/CSV-oriented API.
//! Public backend diagnostics intentionally flow through `axiom_backend` so
//! callers choose targets and inspect capability reports instead of binding to a
//! target-specific bridge module.

const std = @import("std");
const build_options = @import("vectra_build_options");
const array_mod = @import("array.zig");
const series_mod = @import("series.zig");
const dataframe_mod = if (build_options.enable_boltha) @import("dataframe.zig") else @import("dataframe_no_boltha.zig");
const einsum_mod = @import("einsum.zig");
const layered_array_mod = @import("layered_array.zig");
const forge_interop_mod = @import("forge_interop.zig");
const histogram2d_device_mod = @import("histogram2d_device.zig");
pub const linalg = @import("linalg.zig");
pub const stats = @import("stats.zig");
pub const sparse = @import("sparse.zig");
pub const forge_interop = forge_interop_mod;
pub const ForgeInteropBoundary = forge_interop_mod.InteropBoundary;
pub const forgeArrayInteropBoundary = forge_interop_mod.forge_array_interop_boundary;
pub const forgeInteropBoundary = forge_interop_mod.forgeInteropBoundary;

pub const Array = array_mod.Array;
pub const NDArray = array_mod.NDArray;
pub const ArrayView = array_mod.ArrayView;
pub const NDArrayView = array_mod.NDArrayView;
pub const BFloat16 = array_mod.BFloat16;
pub const Complex64 = array_mod.Complex64;
pub const Complex128 = array_mod.Complex128;
pub const Device = array_mod.Device;
pub const DeviceHistogram2DBoundsF32 = histogram2d_device_mod.BoundsF32;
pub const DeviceHistogram2DCountView = histogram2d_device_mod.CountView;
pub const DeviceHistogram2DCountSession = histogram2d_device_mod.DeviceHistogram2DCountSession;
pub const DeviceHistogram2DExtremaOp = histogram2d_device_mod.ExtremaOp;
pub const DeviceHistogram2DExtremaView = histogram2d_device_mod.ExtremaView;
pub const DeviceHistogram2DExtremaSession = histogram2d_device_mod.DeviceHistogram2DExtremaSession;
pub const DeviceHistogram2DSumView = histogram2d_device_mod.SumView;
pub const DeviceHistogram2DSumSession = histogram2d_device_mod.DeviceHistogram2DSumSession;
pub const DeviceCategoricalHistogram2DCountView = histogram2d_device_mod.CategoricalCountView;
pub const DeviceCategoricalHistogram2DCountSession = histogram2d_device_mod.DeviceCategoricalHistogram2DCountSession;
pub const DType = array_mod.DType;
pub const canCastDType = array_mod.canCastDType;
pub const promoteDType = array_mod.promoteDType;
pub const resultDType = array_mod.resultDType;
pub const promoteType = array_mod.promoteType;
pub const Slice = array_mod.Slice;
pub const ScatterReduce = array_mod.ScatterReduce;
pub const SearchSide = array_mod.SearchSide;
pub const IndexMode = array_mod.IndexMode;
pub const MeshGridIndexing = array_mod.MeshGridIndexing;
pub const ConvMode = array_mod.ConvMode;
pub const LossReduction = array_mod.LossReduction;
pub const ArrayError = array_mod.ArrayError;
pub const axiom_backend = @import("backends/axiom_backend.zig");
pub const DialectBackend = axiom_backend.DialectBackend;
pub const PreparedF32Matmul = axiom_backend.PreparedF32Matmul;
pub const PreparedF32TransposedMatmul = axiom_backend.PreparedF32TransposedMatmul;
pub const PreparedF64Matmul = axiom_backend.PreparedF64Matmul;
pub const PreparedF64TransposedMatmul = axiom_backend.PreparedF64TransposedMatmul;
pub const setDefaultDialectBackend = axiom_backend.setDefaultDialectBackend;
pub const defaultDialectBackend = axiom_backend.defaultDialectBackend;
pub const defaultExecutionTarget = axiom_backend.defaultExecutionTarget;
pub const executionTargetForDevice = axiom_backend.executionTargetForDevice;
pub const resetDefaultDialectBackend = axiom_backend.resetDefaultDialectBackend;
pub const cpuMatmulColumnMajorResult = axiom_backend.cpuMatmulColumnMajorResult;
pub const Series = series_mod.Series;
pub const DataFrame = dataframe_mod.DataFrame;
pub const Column = dataframe_mod.Column;
pub const ColumnDef = dataframe_mod.ColumnDef;
pub const DataError = dataframe_mod.DataError;
pub const DeviceDataFrame = dataframe_mod.DeviceDataFrame;
pub const DeviceColumnSchema = dataframe_mod.DeviceColumnSchema;
pub const DeviceLazyFrame = dataframe_mod.DeviceLazyFrame;
pub const DeviceLazyOp = dataframe_mod.DeviceLazyOp;
pub const DeviceLazySource = dataframe_mod.DeviceLazySource;
pub const DeviceLazyGroupByAggregation = dataframe_mod.DeviceLazyGroupByAggregation;
pub const DeviceLazyWeightedGroupByAggregation = dataframe_mod.DeviceLazyWeightedGroupByAggregation;
pub const DeviceLazyPairGroupByAggregation = dataframe_mod.DeviceLazyPairGroupByAggregation;
pub const DeviceLazyWeightedPairGroupByAggregation = dataframe_mod.DeviceLazyWeightedPairGroupByAggregation;
pub const DeviceLazyJoinKind = dataframe_mod.DeviceLazyJoinKind;
pub const DeviceParquetScan = dataframe_mod.DeviceParquetScan;
pub const DeviceParquetRangeFilter = dataframe_mod.DeviceParquetRangeFilter;
pub const DeviceParquetNullFilter = dataframe_mod.DeviceParquetNullFilter;
pub const DeviceParquetFileSummary = dataframe_mod.DeviceParquetFileSummary;
pub const DeviceParquetRowGroupSummary = dataframe_mod.DeviceParquetRowGroupSummary;
pub const DeviceParquetScanSourceRange = dataframe_mod.DeviceParquetScanSourceRange;
pub const DeviceParquetScanSummary = dataframe_mod.DeviceParquetScanSummary;
pub const DeviceParquetScanPushdownSummary = dataframe_mod.DeviceParquetScanPushdownSummary;
pub const ParquetRangePredicate = dataframe_mod.ParquetRangePredicate;
pub const DeviceDataFrameView = dataframe_mod.DeviceDataFrameView;
pub const DeviceColumn = dataframe_mod.DeviceColumn;
pub const DeviceColumnView = dataframe_mod.DeviceColumnView;
pub const DeviceColumnDef = dataframe_mod.DeviceColumnDef;
pub const DeviceTypedColumn = dataframe_mod.DeviceTypedColumn;
pub const DeviceDataError = dataframe_mod.DeviceDataError;
pub const ArrowInteropError = dataframe_mod.ArrowInteropError;
pub const ParquetInteropError = dataframe_mod.ParquetInteropError;
pub const DeviceDType = dataframe_mod.DeviceDType;
pub const DeviceDTypeClass = dataframe_mod.DeviceDTypeClass;
pub const DeviceScalar = dataframe_mod.DeviceScalar;
pub const Range = dataframe_mod.Range;
pub const DeviceColumnBinaryOp = dataframe_mod.DeviceColumnBinaryOp;
pub const DeviceColumnCompareOp = dataframe_mod.DeviceColumnCompareOp;
pub const DeviceColumnLogicalOp = dataframe_mod.DeviceColumnLogicalOp;
pub const DeviceGroupByAggregation = dataframe_mod.DeviceGroupByAggregation;
pub const DeviceSortOptions = dataframe_mod.DeviceSortOptions;
pub const DeviceClipOptions = dataframe_mod.DeviceClipOptions;
pub const DeviceThresholdOptions = dataframe_mod.DeviceThresholdOptions;
pub const DeviceRollingOptions = dataframe_mod.DeviceRollingOptions;
pub const DeviceLagOptions = dataframe_mod.DeviceLagOptions;
pub const DeviceExpandingOptions = dataframe_mod.DeviceExpandingOptions;
pub const DeviceExpandingRankOptions = dataframe_mod.DeviceExpandingRankOptions;
pub const DeviceStandardizeOptions = dataframe_mod.DeviceStandardizeOptions;
pub const DeviceRobustOptions = dataframe_mod.DeviceRobustOptions;
pub const DeviceDrawdownOptions = dataframe_mod.DeviceDrawdownOptions;
pub const DeviceExtremaOptions = dataframe_mod.DeviceExtremaOptions;
pub const DeviceTrendOptions = dataframe_mod.DeviceTrendOptions;
pub const DeviceCrossoverOptions = dataframe_mod.DeviceCrossoverOptions;
pub const DeviceBucketOptions = dataframe_mod.DeviceBucketOptions;
pub const DeviceEmaOptions = dataframe_mod.DeviceEmaOptions;
pub const DeviceLinearFitOptions = dataframe_mod.DeviceLinearFitOptions;
pub const DeviceRollingCorrelationOptions = dataframe_mod.DeviceRollingCorrelationOptions;
pub const DeviceRollingRankOptions = dataframe_mod.DeviceRollingRankOptions;
pub const DeviceRollingRobustOptions = dataframe_mod.DeviceRollingRobustOptions;
pub const NullPlacement = dataframe_mod.NullPlacement;
pub const DeviceJoinOptions = dataframe_mod.DeviceJoinOptions;
pub const AsofStrategy = dataframe_mod.AsofStrategy;
pub const DeviceAsofOptions = dataframe_mod.DeviceAsofOptions;
pub const DeviceValidityEncoding = dataframe_mod.DeviceValidityEncoding;

pub const ArrowExport = if (build_options.enable_boltha) struct {
    pub const DataFrame = struct {
        pub const Arrow = struct {
            pub const hasProjection = dataframe_mod.deviceDataFrameHasArrowProjection;
            pub const hasArrowProjection = dataframe_mod.deviceDataFrameHasArrowProjection;
            pub const toFields = dataframe_mod.DeviceDataFrame.toArrowFields;
            pub const toArrowFields = dataframe_mod.DeviceDataFrame.toArrowFields;
            pub const toSchema = dataframe_mod.DeviceDataFrame.toArrowSchema;
            pub const toArrowSchema = dataframe_mod.DeviceDataFrame.toArrowSchema;
            pub const toRecordBatch = dataframe_mod.DeviceDataFrame.toArrowRecordBatch;
            pub const toArrowRecordBatch = dataframe_mod.DeviceDataFrame.toArrowRecordBatch;
            pub const toTable = dataframe_mod.DeviceDataFrame.toArrowTable;
            pub const toArrowTable = dataframe_mod.DeviceDataFrame.toArrowTable;
            pub const fromRecordBatch = dataframe_mod.DeviceDataFrame.fromArrowRecordBatch;
            pub const fromArrowRecordBatch = dataframe_mod.DeviceDataFrame.fromArrowRecordBatch;
            pub const fromRecordBatchProjection = dataframe_mod.DeviceDataFrame.fromArrowRecordBatchProjection;
            pub const fromArrowRecordBatchProjection = dataframe_mod.DeviceDataFrame.fromArrowRecordBatchProjection;
            pub const fromTable = dataframe_mod.DeviceDataFrame.fromArrowTable;
            pub const fromArrowTable = dataframe_mod.DeviceDataFrame.fromArrowTable;
            pub const fromTableProjection = dataframe_mod.DeviceDataFrame.fromArrowTableProjection;
            pub const fromArrowTableProjection = dataframe_mod.DeviceDataFrame.fromArrowTableProjection;
        };

        pub const Parquet = struct {
            pub const toBytes = dataframe_mod.DeviceDataFrame.toParquetBytes;
            pub const toParquetBytes = dataframe_mod.DeviceDataFrame.toParquetBytes;
            pub const writeFileInDir = dataframe_mod.DeviceDataFrame.writeParquetFileInDir;
            pub const writeFile = dataframe_mod.DeviceDataFrame.writeParquetFile;
            pub const fromBytes = dataframe_mod.DeviceDataFrame.fromParquetBytes;
            pub const fromParquetBytes = dataframe_mod.DeviceDataFrame.fromParquetBytes;
            pub const fromBytesPruned = dataframe_mod.DeviceDataFrame.fromParquetBytesPruned;
            pub const fromParquetBytesPruned = dataframe_mod.DeviceDataFrame.fromParquetBytesPruned;
            pub const fromFileInDir = dataframe_mod.DeviceDataFrame.fromParquetFileInDir;
            pub const fromFile = dataframe_mod.DeviceDataFrame.fromParquetFile;
            pub const fromFilePrunedInDir = dataframe_mod.DeviceDataFrame.fromParquetFilePrunedInDir;
            pub const fromFilePruned = dataframe_mod.DeviceDataFrame.fromParquetFilePruned;
        };

        pub const hasArrowProjection = dataframe_mod.deviceDataFrameHasArrowProjection;
        pub const toArrowFields = dataframe_mod.DeviceDataFrame.toArrowFields;
        pub const toArrowSchema = dataframe_mod.DeviceDataFrame.toArrowSchema;
        pub const toArrowRecordBatch = dataframe_mod.DeviceDataFrame.toArrowRecordBatch;
        pub const toArrowTable = dataframe_mod.DeviceDataFrame.toArrowTable;
        pub const toParquetBytes = dataframe_mod.DeviceDataFrame.toParquetBytes;
        pub const writeParquetFileInDir = dataframe_mod.DeviceDataFrame.writeParquetFileInDir;
        pub const writeParquetFile = dataframe_mod.DeviceDataFrame.writeParquetFile;
        pub const fromParquetFileInDir = dataframe_mod.DeviceDataFrame.fromParquetFileInDir;
        pub const fromParquetFile = dataframe_mod.DeviceDataFrame.fromParquetFile;
        pub const fromParquetFilePrunedInDir = dataframe_mod.DeviceDataFrame.fromParquetFilePrunedInDir;
        pub const fromParquetFilePruned = dataframe_mod.DeviceDataFrame.fromParquetFilePruned;
        pub const fromArrowRecordBatch = dataframe_mod.DeviceDataFrame.fromArrowRecordBatch;
        pub const fromArrowRecordBatchProjection = dataframe_mod.DeviceDataFrame.fromArrowRecordBatchProjection;
        pub const fromArrowTable = dataframe_mod.DeviceDataFrame.fromArrowTable;
        pub const fromArrowTableProjection = dataframe_mod.DeviceDataFrame.fromArrowTableProjection;
        pub const fromParquetBytes = dataframe_mod.DeviceDataFrame.fromParquetBytes;
        pub const fromParquetBytesPruned = dataframe_mod.DeviceDataFrame.fromParquetBytesPruned;
    };

    pub const Column = struct {
        pub const arrowDataType = dataframe_mod.DeviceColumn.arrowDataType;
        pub const toArrowField = dataframe_mod.deviceColumnToArrowField;
        pub const toArrowArray = dataframe_mod.DeviceColumn.toArrowArray;
    };

    pub const ColumnSchema = struct {
        pub const arrowDataType = dataframe_mod.deviceColumnSchemaToArrowDataType;
        pub const toArrowField = dataframe_mod.deviceColumnSchemaToArrowField;
    };

    pub const ColumnView = struct {
        pub const arrowDataType = dataframe_mod.deviceColumnViewToArrowDataType;
        pub const toArrowField = dataframe_mod.deviceColumnViewToArrowField;
    };

    pub const DataFrameView = struct {
        pub const toArrowFields = dataframe_mod.deviceDataFrameViewToArrowFields;
        pub const hasArrowProjection = dataframe_mod.deviceDataFrameViewHasArrowProjection;
        pub const toArrowFieldsProjection = dataframe_mod.deviceDataFrameViewToArrowFieldsProjection;
        pub const toArrowSchema = dataframe_mod.deviceDataFrameViewToArrowSchema;
        pub const toArrowSchemaProjection = dataframe_mod.deviceDataFrameViewToArrowSchemaProjection;
    };

    pub const LazyFrame = struct {
        pub const Arrow = struct {
            pub const hasProjection = dataframe_mod.DeviceLazyFrame.hasArrowProjection;
            pub const hasArrowProjection = dataframe_mod.DeviceLazyFrame.hasArrowProjection;
            pub const toFields = dataframe_mod.DeviceLazyFrame.toArrowFields;
            pub const toArrowFields = dataframe_mod.DeviceLazyFrame.toArrowFields;
            pub const toFieldsProjection = dataframe_mod.DeviceLazyFrame.toArrowFieldsProjection;
            pub const toArrowFieldsProjection = dataframe_mod.DeviceLazyFrame.toArrowFieldsProjection;
            pub const toSchema = dataframe_mod.DeviceLazyFrame.toArrowSchema;
            pub const toArrowSchema = dataframe_mod.DeviceLazyFrame.toArrowSchema;
            pub const toSchemaProjection = dataframe_mod.DeviceLazyFrame.toArrowSchemaProjection;
            pub const toArrowSchemaProjection = dataframe_mod.DeviceLazyFrame.toArrowSchemaProjection;
        };

        pub const hasProjection = dataframe_mod.DeviceLazyFrame.hasArrowProjection;
        pub const hasArrowProjection = dataframe_mod.DeviceLazyFrame.hasArrowProjection;
        pub const toArrowFields = dataframe_mod.DeviceLazyFrame.toArrowFields;
        pub const toArrowFieldsProjection = dataframe_mod.DeviceLazyFrame.toArrowFieldsProjection;
        pub const toArrowSchema = dataframe_mod.DeviceLazyFrame.toArrowSchema;
        pub const toArrowSchemaProjection = dataframe_mod.DeviceLazyFrame.toArrowSchemaProjection;
    };

    pub const ParquetScan = struct {
        pub const Lifecycle = struct {
            pub const init = dataframe_mod.DeviceParquetScan.init;
            pub const initOwnedBytes = dataframe_mod.DeviceParquetScan.initOwnedBytes;
            pub const fromFileInDir = dataframe_mod.DeviceParquetScan.fromFileInDir;
            pub const fromFile = dataframe_mod.DeviceParquetScan.fromFile;
            pub const moveBytes = dataframe_mod.DeviceParquetScan.moveBytes;
            pub const clone = dataframe_mod.DeviceParquetScan.clone;
            pub const lazy = dataframe_mod.DeviceParquetScan.lazy;
            pub const collect = dataframe_mod.DeviceParquetScan.collect;
            pub const explain = dataframe_mod.DeviceParquetScan.explain;
            pub const explainSummary = dataframe_mod.DeviceParquetScan.explainSummary;
        };

        pub const Device = struct {
            pub const setDevice = dataframe_mod.DeviceParquetScan.setDevice;
            pub const retarget = dataframe_mod.DeviceParquetScan.retarget;
            pub const to = dataframe_mod.DeviceParquetScan.to;
            pub const withDevice = dataframe_mod.DeviceParquetScan.withDevice;
            pub const cpu = dataframe_mod.DeviceParquetScan.cpu;
            pub const cuda = dataframe_mod.DeviceParquetScan.cuda;
            pub const mps = dataframe_mod.DeviceParquetScan.mps;
            pub const deviceValue = dataframe_mod.DeviceParquetScan.deviceValue;
            pub const deviceBackend = dataframe_mod.DeviceParquetScan.deviceBackend;
            pub const deviceBackendName = dataframe_mod.DeviceParquetScan.deviceBackendName;
            pub const deviceIndex = dataframe_mod.DeviceParquetScan.deviceIndex;
            pub const isCpu = dataframe_mod.DeviceParquetScan.isCpu;
            pub const isCuda = dataframe_mod.DeviceParquetScan.isCuda;
            pub const isMps = dataframe_mod.DeviceParquetScan.isMps;
            pub const isHostBacked = dataframe_mod.DeviceParquetScan.isHostBacked;
            pub const isCudaBacked = dataframe_mod.DeviceParquetScan.isCudaBacked;
            pub const isMpsBacked = dataframe_mod.DeviceParquetScan.isMpsBacked;
            pub const isAcceleratorBacked = dataframe_mod.DeviceParquetScan.isAcceleratorBacked;
            pub const isRemoteBacked = dataframe_mod.DeviceParquetScan.isRemoteBacked;
            pub const isDeviceBacked = dataframe_mod.DeviceParquetScan.isDeviceBacked;
            pub const isDeviceAvailable = dataframe_mod.DeviceParquetScan.isDeviceAvailable;
            pub const sameDevice = dataframe_mod.DeviceParquetScan.sameDevice;
        };

        pub const Source = struct {
            pub const sourceNbytes = dataframe_mod.DeviceParquetScan.sourceNbytes;
            pub const sourcePtr = dataframe_mod.DeviceParquetScan.sourcePtr;
            pub const dataPtr = dataframe_mod.DeviceParquetScan.dataPtr;
            pub const hasSourcePtr = dataframe_mod.DeviceParquetScan.hasSourcePtr;
            pub const sourceEndPtr = dataframe_mod.DeviceParquetScan.sourceEndPtr;
            pub const sourceRange = dataframe_mod.DeviceParquetScan.sourceRange;
            pub const sharesSource = dataframe_mod.DeviceParquetScan.sharesSource;
            pub const sameSource = dataframe_mod.DeviceParquetScan.sameSource;
            pub const sharesStorage = dataframe_mod.DeviceParquetScan.sharesStorage;
            pub const sameStorage = dataframe_mod.DeviceParquetScan.sameStorage;
            pub const sourceMayOverlap = dataframe_mod.DeviceParquetScan.sourceMayOverlap;
            pub const mayOverlap = dataframe_mod.DeviceParquetScan.mayOverlap;
            pub const sourceByteCount = dataframe_mod.DeviceParquetScan.sourceByteCount;
            pub const nbytes = dataframe_mod.DeviceParquetScan.nbytes;
            pub const byteCount = dataframe_mod.DeviceParquetScan.byteCount;
            pub const isEmpty = dataframe_mod.DeviceParquetScan.isEmpty;
            pub const isNonEmpty = dataframe_mod.DeviceParquetScan.isNonEmpty;
            pub const hasBytes = dataframe_mod.DeviceParquetScan.hasBytes;
            pub const ownedNbytes = dataframe_mod.DeviceParquetScan.ownedNbytes;
            pub const memoryUsage = dataframe_mod.DeviceParquetScan.memoryUsage;
            pub const estimatedSize = dataframe_mod.DeviceParquetScan.estimatedSize;
            pub const summary = dataframe_mod.DeviceParquetScan.summary;
        };

        pub const File = struct {
            pub const parquetFileSummary = dataframe_mod.DeviceParquetScan.parquetFileSummary;
            pub const parquetRowGroupSummaryAt = dataframe_mod.DeviceParquetScan.parquetRowGroupSummaryAt;
            pub const parquetRowGroupSummaries = dataframe_mod.DeviceParquetScan.parquetRowGroupSummaries;
            pub const parquetRowGroupRowCounts = dataframe_mod.DeviceParquetScan.parquetRowGroupRowCounts;
            pub const parquetRowGroupColumnChunkCounts = dataframe_mod.DeviceParquetScan.parquetRowGroupColumnChunkCounts;
            pub const parquetRowGroupTotalNbytes = dataframe_mod.DeviceParquetScan.parquetRowGroupTotalNbytes;
            pub const parquetRowGroupTotalCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetRowGroupTotalCompressedNbytes;
            pub const parquetRowGroupCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetRowGroupCompressedNbytes;
            pub const parquetRowGroupUncompressedNbytes = dataframe_mod.DeviceParquetScan.parquetRowGroupUncompressedNbytes;
            pub const parquetRowGroupCompressionRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupCompressionRatios;
            pub const parquetRowGroupMetadataCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupMetadataCoverageRatios;
            pub const parquetRowGroupMissingMetadataRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupMissingMetadataRatios;
            pub const parquetRowGroupColumnIndexCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupColumnIndexCoverageRatios;
            pub const parquetRowGroupOffsetIndexCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupOffsetIndexCoverageRatios;
            pub const parquetRowGroupPageIndexCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupPageIndexCoverageRatios;
            pub const parquetRowGroupBloomFilterCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupBloomFilterCoverageRatios;
            pub const parquetRowGroupSizedBloomFilterCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupSizedBloomFilterCoverageRatios;
            pub const rowCount = dataframe_mod.DeviceParquetScan.rowCount;
            pub const nRows = dataframe_mod.DeviceParquetScan.nRows;
            pub const rowGroupCount = dataframe_mod.DeviceParquetScan.rowGroupCount;
            pub const parquetColumnChunkCount = dataframe_mod.DeviceParquetScan.parquetColumnChunkCount;
            pub const columnCount = dataframe_mod.DeviceParquetScan.columnCount;
            pub const width = dataframe_mod.DeviceParquetScan.width;
            pub const cols = dataframe_mod.DeviceParquetScan.cols;
            pub const nCols = dataframe_mod.DeviceParquetScan.nCols;
            pub const cellCount = dataframe_mod.DeviceParquetScan.cellCount;
            pub const shape = dataframe_mod.DeviceParquetScan.shape;
            pub const hasRows = dataframe_mod.DeviceParquetScan.hasRows;
            pub const hasColumns = dataframe_mod.DeviceParquetScan.hasColumns;
            pub const hasShape = dataframe_mod.DeviceParquetScan.hasShape;
            pub const sameHeight = dataframe_mod.DeviceParquetScan.sameHeight;
            pub const sameWidth = dataframe_mod.DeviceParquetScan.sameWidth;
            pub const sameShape = dataframe_mod.DeviceParquetScan.sameShape;
            pub const shapeEquals = dataframe_mod.DeviceParquetScan.shapeEquals;
            pub const sameRowGroups = dataframe_mod.DeviceParquetScan.sameRowGroups;
            pub const parquetTotalNbytes = dataframe_mod.DeviceParquetScan.parquetTotalNbytes;
            pub const parquetTotalCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetTotalCompressedNbytes;
            pub const parquetTotalUncompressedNbytes = dataframe_mod.DeviceParquetScan.parquetTotalUncompressedNbytes;
            pub const parquetFieldCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetFieldCompressedNbytes;
            pub const parquetFieldCompressedNbytesProjection = dataframe_mod.DeviceParquetScan.parquetFieldCompressedNbytesProjection;
            pub const parquetFieldUncompressedNbytes = dataframe_mod.DeviceParquetScan.parquetFieldUncompressedNbytes;
            pub const parquetFieldUncompressedNbytesProjection = dataframe_mod.DeviceParquetScan.parquetFieldUncompressedNbytesProjection;
            pub const parquetCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetCompressedNbytes;
            pub const parquetCompressedNbytesProjection = dataframe_mod.DeviceParquetScan.parquetCompressedNbytesProjection;
            pub const parquetUncompressedNbytes = dataframe_mod.DeviceParquetScan.parquetUncompressedNbytes;
            pub const parquetUncompressedNbytesProjection = dataframe_mod.DeviceParquetScan.parquetUncompressedNbytesProjection;
            pub const parquetFieldCompressionRatios = dataframe_mod.DeviceParquetScan.parquetFieldCompressionRatios;
            pub const parquetFieldCompressionRatiosProjection = dataframe_mod.DeviceParquetScan.parquetFieldCompressionRatiosProjection;
            pub const parquetCompressionRatio = dataframe_mod.DeviceParquetScan.parquetCompressionRatio;
            pub const parquetCompressionRatioProjection = dataframe_mod.DeviceParquetScan.parquetCompressionRatioProjection;
            pub const parquetMetadataCoverageRatio = dataframe_mod.DeviceParquetScan.parquetMetadataCoverageRatio;
            pub const parquetPageIndexCoverageRatio = dataframe_mod.DeviceParquetScan.parquetPageIndexCoverageRatio;
            pub const hasRowGroups = dataframe_mod.DeviceParquetScan.hasRowGroups;
        };

        pub const Pushdown = struct {
            pub const projectionMetadataNbytes = dataframe_mod.DeviceParquetScan.projectionMetadataNbytes;
            pub const rangePredicateMetadataNbytes = dataframe_mod.DeviceParquetScan.rangePredicateMetadataNbytes;
            pub const nullPredicateMetadataNbytes = dataframe_mod.DeviceParquetScan.nullPredicateMetadataNbytes;
            pub const predicateMetadataNbytes = dataframe_mod.DeviceParquetScan.predicateMetadataNbytes;
            pub const pushdownMetadataNbytes = dataframe_mod.DeviceParquetScan.pushdownMetadataNbytes;
            pub const hasProjection = dataframe_mod.DeviceParquetScan.hasProjection;
            pub const projectionColumnCount = dataframe_mod.DeviceParquetScan.projectionColumnCount;
            pub const projectionNames = dataframe_mod.DeviceParquetScan.projectionNames;
            pub const projectionNameAt = dataframe_mod.DeviceParquetScan.projectionNameAt;
            pub const projectionIndex = dataframe_mod.DeviceParquetScan.projectionIndex;
            pub const projectionContains = dataframe_mod.DeviceParquetScan.projectionContains;
            pub const projectionNamesUnique = dataframe_mod.DeviceParquetScan.projectionNamesUnique;
            pub const hasDuplicateProjectionNames = dataframe_mod.DeviceParquetScan.hasDuplicateProjectionNames;
            pub const duplicateProjectionNameCount = dataframe_mod.DeviceParquetScan.duplicateProjectionNameCount;
            pub const hasAllProjectionNames = dataframe_mod.DeviceParquetScan.hasAllProjectionNames;
            pub const hasAnyProjectionName = dataframe_mod.DeviceParquetScan.hasAnyProjectionName;
            pub const projectsColumn = dataframe_mod.DeviceParquetScan.projectsColumn;
            pub const hasPredicate = dataframe_mod.DeviceParquetScan.hasPredicate;
            pub const predicateColumn = dataframe_mod.DeviceParquetScan.predicateColumn;
            pub const hasPredicateFor = dataframe_mod.DeviceParquetScan.hasPredicateFor;
            pub const hasRangePredicate = dataframe_mod.DeviceParquetScan.hasRangePredicate;
            pub const rangePredicateColumn = dataframe_mod.DeviceParquetScan.rangePredicateColumn;
            pub const rangePredicate = dataframe_mod.DeviceParquetScan.rangePredicate;
            pub const rangePredicateDType = dataframe_mod.DeviceParquetScan.rangePredicateDType;
            pub const hasRangePredicateFor = dataframe_mod.DeviceParquetScan.hasRangePredicateFor;
            pub const hasNullPredicate = dataframe_mod.DeviceParquetScan.hasNullPredicate;
            pub const nullPredicateColumn = dataframe_mod.DeviceParquetScan.nullPredicateColumn;
            pub const nullPredicateWantNulls = dataframe_mod.DeviceParquetScan.nullPredicateWantNulls;
            pub const hasNullPredicateFor = dataframe_mod.DeviceParquetScan.hasNullPredicateFor;
            pub const hasPushdown = dataframe_mod.DeviceParquetScan.hasPushdown;
            pub const validateProjection = dataframe_mod.DeviceParquetScan.validateProjection;
            pub const validatePredicate = dataframe_mod.DeviceParquetScan.validatePredicate;
            pub const validatePushdown = dataframe_mod.DeviceParquetScan.validatePushdown;
            pub const pushdownValid = dataframe_mod.DeviceParquetScan.pushdownValid;
            pub const validateCollect = dataframe_mod.DeviceParquetScan.validateCollect;
            pub const collectValid = dataframe_mod.DeviceParquetScan.collectValid;
            pub const pushdownSummary = dataframe_mod.DeviceParquetScan.pushdownSummary;
            pub const clearProjection = dataframe_mod.DeviceParquetScan.clearProjection;
            pub const clearRangePredicate = dataframe_mod.DeviceParquetScan.clearRangePredicate;
            pub const clearNullPredicate = dataframe_mod.DeviceParquetScan.clearNullPredicate;
            pub const clearPredicate = dataframe_mod.DeviceParquetScan.clearPredicate;
            pub const clearPushdown = dataframe_mod.DeviceParquetScan.clearPushdown;
            pub const resetPushdown = dataframe_mod.DeviceParquetScan.resetPushdown;
            pub const select = dataframe_mod.DeviceParquetScan.select;
            pub const appendSelect = dataframe_mod.DeviceParquetScan.appendSelect;
            pub const dropSelected = dataframe_mod.DeviceParquetScan.dropSelected;
            pub const selectAll = dataframe_mod.DeviceParquetScan.selectAll;
            pub const selectExcept = dataframe_mod.DeviceParquetScan.selectExcept;
            pub const intersectSelect = dataframe_mod.DeviceParquetScan.intersectSelect;
            pub const whereRange = dataframe_mod.DeviceParquetScan.whereRange;
            pub const whereMin = dataframe_mod.DeviceParquetScan.whereMin;
            pub const whereMax = dataframe_mod.DeviceParquetScan.whereMax;
            pub const whereBetween = dataframe_mod.DeviceParquetScan.whereBetween;
            pub const whereGe = dataframe_mod.DeviceParquetScan.whereGe;
            pub const whereLe = dataframe_mod.DeviceParquetScan.whereLe;
            pub const whereGt = dataframe_mod.DeviceParquetScan.whereGt;
            pub const whereLt = dataframe_mod.DeviceParquetScan.whereLt;
            pub const whereEq = dataframe_mod.DeviceParquetScan.whereEq;
            pub const whereBool = dataframe_mod.DeviceParquetScan.whereBool;
            pub const whereNull = dataframe_mod.DeviceParquetScan.whereNull;
            pub const whereIsNull = dataframe_mod.DeviceParquetScan.whereIsNull;
            pub const whereIsNotNull = dataframe_mod.DeviceParquetScan.whereIsNotNull;
            pub const whereNotNull = dataframe_mod.DeviceParquetScan.whereNotNull;
        };

        pub const Arrow = struct {
            pub const toArrowSchema = dataframe_mod.DeviceParquetScan.toArrowSchema;
            pub const toArrowSchemaProjection = dataframe_mod.DeviceParquetScan.toArrowSchemaProjection;
            pub const toArrowFields = dataframe_mod.DeviceParquetScan.toArrowFields;
            pub const toArrowFieldsProjection = dataframe_mod.DeviceParquetScan.toArrowFieldsProjection;
            pub const arrowFieldCount = dataframe_mod.DeviceParquetScan.arrowFieldCount;
            pub const arrowFieldNameAt = dataframe_mod.DeviceParquetScan.arrowFieldNameAt;
            pub const arrowFieldNames = dataframe_mod.DeviceParquetScan.arrowFieldNames;
            pub const arrowFieldIndex = dataframe_mod.DeviceParquetScan.arrowFieldIndex;
            pub const hasArrowField = dataframe_mod.DeviceParquetScan.hasArrowField;
            pub const hasAllArrowFields = dataframe_mod.DeviceParquetScan.hasAllArrowFields;
            pub const hasAnyArrowField = dataframe_mod.DeviceParquetScan.hasAnyArrowField;
            pub const arrowFieldDTypeAt = dataframe_mod.DeviceParquetScan.arrowFieldDTypeAt;
            pub const arrowFieldDType = dataframe_mod.DeviceParquetScan.arrowFieldDType;
            pub const arrowFieldDTypes = dataframe_mod.DeviceParquetScan.arrowFieldDTypes;
            pub const arrowFieldDTypesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypesProjection;
            pub const arrowFieldDTypeNames = dataframe_mod.DeviceParquetScan.arrowFieldDTypeNames;
            pub const arrowFieldDTypeNamesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeNamesProjection;
            pub const arrowFieldDTypeByteSizes = dataframe_mod.DeviceParquetScan.arrowFieldDTypeByteSizes;
            pub const arrowFieldDTypeByteSizesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeByteSizesProjection;
            pub const arrowFieldDTypeBitSizes = dataframe_mod.DeviceParquetScan.arrowFieldDTypeBitSizes;
            pub const arrowFieldDTypeBitSizesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeBitSizesProjection;
            pub const arrowFieldDTypeClassMask = dataframe_mod.DeviceParquetScan.arrowFieldDTypeClassMask;
            pub const arrowFieldDTypeClassMaskProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeClassMaskProjection;
            pub const arrowFieldDTypeClassCount = dataframe_mod.DeviceParquetScan.arrowFieldDTypeClassCount;
            pub const arrowFieldDTypeClassCountProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeClassCountProjection;
            pub const numericArrowFieldCount = dataframe_mod.DeviceParquetScan.numericArrowFieldCount;
            pub const numericArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.numericArrowFieldCountProjection;
            pub const floatArrowFieldCount = dataframe_mod.DeviceParquetScan.floatArrowFieldCount;
            pub const floatArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.floatArrowFieldCountProjection;
            pub const integerArrowFieldCount = dataframe_mod.DeviceParquetScan.integerArrowFieldCount;
            pub const integerArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.integerArrowFieldCountProjection;
            pub const boolArrowFieldCount = dataframe_mod.DeviceParquetScan.boolArrowFieldCount;
            pub const boolArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.boolArrowFieldCountProjection;
            pub const arrowFieldNullableAt = dataframe_mod.DeviceParquetScan.arrowFieldNullableAt;
            pub const arrowFieldNullable = dataframe_mod.DeviceParquetScan.arrowFieldNullable;
            pub const arrowFieldNullableMask = dataframe_mod.DeviceParquetScan.arrowFieldNullableMask;
            pub const arrowFieldNullableMaskProjection = dataframe_mod.DeviceParquetScan.arrowFieldNullableMaskProjection;
            pub const nullableArrowFieldCount = dataframe_mod.DeviceParquetScan.nullableArrowFieldCount;
            pub const nullableArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.nullableArrowFieldCountProjection;
            pub const nonNullableArrowFieldCount = dataframe_mod.DeviceParquetScan.nonNullableArrowFieldCount;
            pub const nonNullableArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.nonNullableArrowFieldCountProjection;
            pub const arrowFieldNullCount = dataframe_mod.DeviceParquetScan.arrowFieldNullCount;
            pub const arrowFieldValidCount = dataframe_mod.DeviceParquetScan.arrowFieldValidCount;
            pub const arrowFieldNullCounts = dataframe_mod.DeviceParquetScan.arrowFieldNullCounts;
            pub const arrowFieldNullCountsProjection = dataframe_mod.DeviceParquetScan.arrowFieldNullCountsProjection;
            pub const arrowFieldValidCounts = dataframe_mod.DeviceParquetScan.arrowFieldValidCounts;
            pub const arrowFieldValidCountsProjection = dataframe_mod.DeviceParquetScan.arrowFieldValidCountsProjection;
            pub const arrowFieldNullRatios = dataframe_mod.DeviceParquetScan.arrowFieldNullRatios;
            pub const arrowFieldNullRatiosProjection = dataframe_mod.DeviceParquetScan.arrowFieldNullRatiosProjection;
            pub const arrowFieldValidRatios = dataframe_mod.DeviceParquetScan.arrowFieldValidRatios;
            pub const arrowFieldValidRatiosProjection = dataframe_mod.DeviceParquetScan.arrowFieldValidRatiosProjection;
            pub const arrowNullCount = dataframe_mod.DeviceParquetScan.arrowNullCount;
            pub const arrowNullCountProjection = dataframe_mod.DeviceParquetScan.arrowNullCountProjection;
            pub const arrowValidCount = dataframe_mod.DeviceParquetScan.arrowValidCount;
            pub const arrowValidCountProjection = dataframe_mod.DeviceParquetScan.arrowValidCountProjection;
            pub const arrowNullRatio = dataframe_mod.DeviceParquetScan.arrowNullRatio;
            pub const arrowNullRatioProjection = dataframe_mod.DeviceParquetScan.arrowNullRatioProjection;
            pub const arrowValidRatio = dataframe_mod.DeviceParquetScan.arrowValidRatio;
            pub const arrowValidRatioProjection = dataframe_mod.DeviceParquetScan.arrowValidRatioProjection;
            pub const hasNullableArrowFields = dataframe_mod.DeviceParquetScan.hasNullableArrowFields;
            pub const allArrowFieldsNullable = dataframe_mod.DeviceParquetScan.allArrowFieldsNullable;
            pub const arrowColumnSchemaAt = dataframe_mod.DeviceParquetScan.arrowColumnSchemaAt;
            pub const arrowColumnSchema = dataframe_mod.DeviceParquetScan.arrowColumnSchema;
            pub const arrowColumnSchemas = dataframe_mod.DeviceParquetScan.arrowColumnSchemas;
            pub const arrowColumnSchemasProjection = dataframe_mod.DeviceParquetScan.arrowColumnSchemasProjection;
            pub const arrowSchemaSummary = dataframe_mod.DeviceParquetScan.arrowSchemaSummary;
            pub const arrowSchemaSummaryProjection = dataframe_mod.DeviceParquetScan.arrowSchemaSummaryProjection;
            pub const arrowFieldDataNbytes = dataframe_mod.DeviceParquetScan.arrowFieldDataNbytes;
            pub const arrowFieldDataNbytesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDataNbytesProjection;
            pub const arrowFieldValidityNbytes = dataframe_mod.DeviceParquetScan.arrowFieldValidityNbytes;
            pub const arrowFieldValidityNbytesProjection = dataframe_mod.DeviceParquetScan.arrowFieldValidityNbytesProjection;
            pub const arrowFieldTotalNbytes = dataframe_mod.DeviceParquetScan.arrowFieldTotalNbytes;
            pub const arrowFieldTotalNbytesProjection = dataframe_mod.DeviceParquetScan.arrowFieldTotalNbytesProjection;
            pub const arrowDataNbytes = dataframe_mod.DeviceParquetScan.arrowDataNbytes;
            pub const arrowDataNbytesProjection = dataframe_mod.DeviceParquetScan.arrowDataNbytesProjection;
            pub const arrowValidityNbytes = dataframe_mod.DeviceParquetScan.arrowValidityNbytes;
            pub const arrowValidityNbytesProjection = dataframe_mod.DeviceParquetScan.arrowValidityNbytesProjection;
            pub const arrowTotalNbytes = dataframe_mod.DeviceParquetScan.arrowTotalNbytes;
            pub const arrowTotalNbytesProjection = dataframe_mod.DeviceParquetScan.arrowTotalNbytesProjection;
            pub const arrowSchemaEquals = dataframe_mod.DeviceParquetScan.arrowSchemaEquals;
            pub const arrowSameSchema = dataframe_mod.DeviceParquetScan.arrowSameSchema;
            pub const arrowSchemaCompatible = dataframe_mod.DeviceParquetScan.arrowSchemaCompatible;
            pub const arrowSchemaEqualsSchemas = dataframe_mod.DeviceParquetScan.arrowSchemaEqualsSchemas;
            pub const arrowSchemaEqualsFrame = dataframe_mod.DeviceParquetScan.arrowSchemaEqualsFrame;
            pub const arrowSameSchemaFrame = dataframe_mod.DeviceParquetScan.arrowSameSchemaFrame;
            pub const arrowSchemaCompatibleFrame = dataframe_mod.DeviceParquetScan.arrowSchemaCompatibleFrame;
            pub const hasArrowProjection = dataframe_mod.DeviceParquetScan.hasArrowProjection;
        };

        pub const init = dataframe_mod.DeviceParquetScan.init;
        pub const initOwnedBytes = dataframe_mod.DeviceParquetScan.initOwnedBytes;
        pub const fromFileInDir = dataframe_mod.DeviceParquetScan.fromFileInDir;
        pub const fromFile = dataframe_mod.DeviceParquetScan.fromFile;
        pub const moveBytes = dataframe_mod.DeviceParquetScan.moveBytes;
        pub const clone = dataframe_mod.DeviceParquetScan.clone;
        pub const lazy = dataframe_mod.DeviceParquetScan.lazy;
        pub const setDevice = dataframe_mod.DeviceParquetScan.setDevice;
        pub const retarget = dataframe_mod.DeviceParquetScan.retarget;
        pub const to = dataframe_mod.DeviceParquetScan.to;
        pub const withDevice = dataframe_mod.DeviceParquetScan.withDevice;
        pub const cpu = dataframe_mod.DeviceParquetScan.cpu;
        pub const cuda = dataframe_mod.DeviceParquetScan.cuda;
        pub const mps = dataframe_mod.DeviceParquetScan.mps;
        pub const deviceValue = dataframe_mod.DeviceParquetScan.deviceValue;
        pub const deviceBackend = dataframe_mod.DeviceParquetScan.deviceBackend;
        pub const deviceBackendName = dataframe_mod.DeviceParquetScan.deviceBackendName;
        pub const deviceIndex = dataframe_mod.DeviceParquetScan.deviceIndex;
        pub const isCpu = dataframe_mod.DeviceParquetScan.isCpu;
        pub const isCuda = dataframe_mod.DeviceParquetScan.isCuda;
        pub const isMps = dataframe_mod.DeviceParquetScan.isMps;
        pub const isHostBacked = dataframe_mod.DeviceParquetScan.isHostBacked;
        pub const isCudaBacked = dataframe_mod.DeviceParquetScan.isCudaBacked;
        pub const isMpsBacked = dataframe_mod.DeviceParquetScan.isMpsBacked;
        pub const isAcceleratorBacked = dataframe_mod.DeviceParquetScan.isAcceleratorBacked;
        pub const isRemoteBacked = dataframe_mod.DeviceParquetScan.isRemoteBacked;
        pub const isDeviceBacked = dataframe_mod.DeviceParquetScan.isDeviceBacked;
        pub const isDeviceAvailable = dataframe_mod.DeviceParquetScan.isDeviceAvailable;
        pub const sameDevice = dataframe_mod.DeviceParquetScan.sameDevice;
        pub const sourceNbytes = dataframe_mod.DeviceParquetScan.sourceNbytes;
        pub const sourcePtr = dataframe_mod.DeviceParquetScan.sourcePtr;
        pub const dataPtr = dataframe_mod.DeviceParquetScan.dataPtr;
        pub const hasSourcePtr = dataframe_mod.DeviceParquetScan.hasSourcePtr;
        pub const sourceEndPtr = dataframe_mod.DeviceParquetScan.sourceEndPtr;
        pub const sourceRange = dataframe_mod.DeviceParquetScan.sourceRange;
        pub const sharesSource = dataframe_mod.DeviceParquetScan.sharesSource;
        pub const sameSource = dataframe_mod.DeviceParquetScan.sameSource;
        pub const sharesStorage = dataframe_mod.DeviceParquetScan.sharesStorage;
        pub const sameStorage = dataframe_mod.DeviceParquetScan.sameStorage;
        pub const sourceMayOverlap = dataframe_mod.DeviceParquetScan.sourceMayOverlap;
        pub const mayOverlap = dataframe_mod.DeviceParquetScan.mayOverlap;
        pub const sourceByteCount = dataframe_mod.DeviceParquetScan.sourceByteCount;
        pub const nbytes = dataframe_mod.DeviceParquetScan.nbytes;
        pub const byteCount = dataframe_mod.DeviceParquetScan.byteCount;
        pub const isEmpty = dataframe_mod.DeviceParquetScan.isEmpty;
        pub const isNonEmpty = dataframe_mod.DeviceParquetScan.isNonEmpty;
        pub const hasBytes = dataframe_mod.DeviceParquetScan.hasBytes;
        pub const projectionMetadataNbytes = dataframe_mod.DeviceParquetScan.projectionMetadataNbytes;
        pub const rangePredicateMetadataNbytes = dataframe_mod.DeviceParquetScan.rangePredicateMetadataNbytes;
        pub const nullPredicateMetadataNbytes = dataframe_mod.DeviceParquetScan.nullPredicateMetadataNbytes;
        pub const predicateMetadataNbytes = dataframe_mod.DeviceParquetScan.predicateMetadataNbytes;
        pub const pushdownMetadataNbytes = dataframe_mod.DeviceParquetScan.pushdownMetadataNbytes;
        pub const ownedNbytes = dataframe_mod.DeviceParquetScan.ownedNbytes;
        pub const memoryUsage = dataframe_mod.DeviceParquetScan.memoryUsage;
        pub const estimatedSize = dataframe_mod.DeviceParquetScan.estimatedSize;
        pub const parquetFileSummary = dataframe_mod.DeviceParquetScan.parquetFileSummary;
        pub const parquetRowGroupSummaryAt = dataframe_mod.DeviceParquetScan.parquetRowGroupSummaryAt;
        pub const parquetRowGroupSummaries = dataframe_mod.DeviceParquetScan.parquetRowGroupSummaries;
        pub const parquetRowGroupRowCounts = dataframe_mod.DeviceParquetScan.parquetRowGroupRowCounts;
        pub const parquetRowGroupColumnChunkCounts = dataframe_mod.DeviceParquetScan.parquetRowGroupColumnChunkCounts;
        pub const parquetRowGroupTotalNbytes = dataframe_mod.DeviceParquetScan.parquetRowGroupTotalNbytes;
        pub const parquetRowGroupTotalCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetRowGroupTotalCompressedNbytes;
        pub const parquetRowGroupCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetRowGroupCompressedNbytes;
        pub const parquetRowGroupUncompressedNbytes = dataframe_mod.DeviceParquetScan.parquetRowGroupUncompressedNbytes;
        pub const parquetRowGroupCompressionRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupCompressionRatios;
        pub const parquetRowGroupMetadataCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupMetadataCoverageRatios;
        pub const parquetRowGroupMissingMetadataRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupMissingMetadataRatios;
        pub const parquetRowGroupColumnIndexCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupColumnIndexCoverageRatios;
        pub const parquetRowGroupOffsetIndexCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupOffsetIndexCoverageRatios;
        pub const parquetRowGroupPageIndexCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupPageIndexCoverageRatios;
        pub const parquetRowGroupBloomFilterCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupBloomFilterCoverageRatios;
        pub const parquetRowGroupSizedBloomFilterCoverageRatios = dataframe_mod.DeviceParquetScan.parquetRowGroupSizedBloomFilterCoverageRatios;
        pub const rowCount = dataframe_mod.DeviceParquetScan.rowCount;
        pub const nRows = dataframe_mod.DeviceParquetScan.nRows;
        pub const rowGroupCount = dataframe_mod.DeviceParquetScan.rowGroupCount;
        pub const parquetColumnChunkCount = dataframe_mod.DeviceParquetScan.parquetColumnChunkCount;
        pub const columnCount = dataframe_mod.DeviceParquetScan.columnCount;
        pub const width = dataframe_mod.DeviceParquetScan.width;
        pub const cols = dataframe_mod.DeviceParquetScan.cols;
        pub const nCols = dataframe_mod.DeviceParquetScan.nCols;
        pub const cellCount = dataframe_mod.DeviceParquetScan.cellCount;
        pub const shape = dataframe_mod.DeviceParquetScan.shape;
        pub const hasRows = dataframe_mod.DeviceParquetScan.hasRows;
        pub const hasColumns = dataframe_mod.DeviceParquetScan.hasColumns;
        pub const hasShape = dataframe_mod.DeviceParquetScan.hasShape;
        pub const sameHeight = dataframe_mod.DeviceParquetScan.sameHeight;
        pub const sameWidth = dataframe_mod.DeviceParquetScan.sameWidth;
        pub const sameShape = dataframe_mod.DeviceParquetScan.sameShape;
        pub const shapeEquals = dataframe_mod.DeviceParquetScan.shapeEquals;
        pub const sameRowGroups = dataframe_mod.DeviceParquetScan.sameRowGroups;
        pub const parquetTotalNbytes = dataframe_mod.DeviceParquetScan.parquetTotalNbytes;
        pub const parquetTotalCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetTotalCompressedNbytes;
        pub const parquetTotalUncompressedNbytes = dataframe_mod.DeviceParquetScan.parquetTotalUncompressedNbytes;
        pub const parquetFieldCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetFieldCompressedNbytes;
        pub const parquetFieldCompressedNbytesProjection = dataframe_mod.DeviceParquetScan.parquetFieldCompressedNbytesProjection;
        pub const parquetFieldUncompressedNbytes = dataframe_mod.DeviceParquetScan.parquetFieldUncompressedNbytes;
        pub const parquetFieldUncompressedNbytesProjection = dataframe_mod.DeviceParquetScan.parquetFieldUncompressedNbytesProjection;
        pub const parquetCompressedNbytes = dataframe_mod.DeviceParquetScan.parquetCompressedNbytes;
        pub const parquetCompressedNbytesProjection = dataframe_mod.DeviceParquetScan.parquetCompressedNbytesProjection;
        pub const parquetUncompressedNbytes = dataframe_mod.DeviceParquetScan.parquetUncompressedNbytes;
        pub const parquetUncompressedNbytesProjection = dataframe_mod.DeviceParquetScan.parquetUncompressedNbytesProjection;
        pub const parquetFieldCompressionRatios = dataframe_mod.DeviceParquetScan.parquetFieldCompressionRatios;
        pub const parquetFieldCompressionRatiosProjection = dataframe_mod.DeviceParquetScan.parquetFieldCompressionRatiosProjection;
        pub const parquetCompressionRatio = dataframe_mod.DeviceParquetScan.parquetCompressionRatio;
        pub const parquetCompressionRatioProjection = dataframe_mod.DeviceParquetScan.parquetCompressionRatioProjection;
        pub const parquetMetadataCoverageRatio = dataframe_mod.DeviceParquetScan.parquetMetadataCoverageRatio;
        pub const parquetPageIndexCoverageRatio = dataframe_mod.DeviceParquetScan.parquetPageIndexCoverageRatio;
        pub const hasRowGroups = dataframe_mod.DeviceParquetScan.hasRowGroups;
        pub const hasProjection = dataframe_mod.DeviceParquetScan.hasProjection;
        pub const projectionColumnCount = dataframe_mod.DeviceParquetScan.projectionColumnCount;
        pub const projectionNames = dataframe_mod.DeviceParquetScan.projectionNames;
        pub const projectionNameAt = dataframe_mod.DeviceParquetScan.projectionNameAt;
        pub const projectionIndex = dataframe_mod.DeviceParquetScan.projectionIndex;
        pub const projectionContains = dataframe_mod.DeviceParquetScan.projectionContains;
        pub const projectionNamesUnique = dataframe_mod.DeviceParquetScan.projectionNamesUnique;
        pub const hasDuplicateProjectionNames = dataframe_mod.DeviceParquetScan.hasDuplicateProjectionNames;
        pub const duplicateProjectionNameCount = dataframe_mod.DeviceParquetScan.duplicateProjectionNameCount;
        pub const hasAllProjectionNames = dataframe_mod.DeviceParquetScan.hasAllProjectionNames;
        pub const hasAnyProjectionName = dataframe_mod.DeviceParquetScan.hasAnyProjectionName;
        pub const projectsColumn = dataframe_mod.DeviceParquetScan.projectsColumn;
        pub const hasPredicate = dataframe_mod.DeviceParquetScan.hasPredicate;
        pub const predicateColumn = dataframe_mod.DeviceParquetScan.predicateColumn;
        pub const hasPredicateFor = dataframe_mod.DeviceParquetScan.hasPredicateFor;
        pub const hasRangePredicate = dataframe_mod.DeviceParquetScan.hasRangePredicate;
        pub const rangePredicateColumn = dataframe_mod.DeviceParquetScan.rangePredicateColumn;
        pub const rangePredicate = dataframe_mod.DeviceParquetScan.rangePredicate;
        pub const rangePredicateDType = dataframe_mod.DeviceParquetScan.rangePredicateDType;
        pub const hasRangePredicateFor = dataframe_mod.DeviceParquetScan.hasRangePredicateFor;
        pub const hasNullPredicate = dataframe_mod.DeviceParquetScan.hasNullPredicate;
        pub const nullPredicateColumn = dataframe_mod.DeviceParquetScan.nullPredicateColumn;
        pub const nullPredicateWantNulls = dataframe_mod.DeviceParquetScan.nullPredicateWantNulls;
        pub const hasNullPredicateFor = dataframe_mod.DeviceParquetScan.hasNullPredicateFor;
        pub const hasPushdown = dataframe_mod.DeviceParquetScan.hasPushdown;
        pub const validateProjection = dataframe_mod.DeviceParquetScan.validateProjection;
        pub const validatePredicate = dataframe_mod.DeviceParquetScan.validatePredicate;
        pub const validatePushdown = dataframe_mod.DeviceParquetScan.validatePushdown;
        pub const pushdownValid = dataframe_mod.DeviceParquetScan.pushdownValid;
        pub const validateCollect = dataframe_mod.DeviceParquetScan.validateCollect;
        pub const collectValid = dataframe_mod.DeviceParquetScan.collectValid;
        pub const pushdownSummary = dataframe_mod.DeviceParquetScan.pushdownSummary;
        pub const summary = dataframe_mod.DeviceParquetScan.summary;
        pub const toArrowSchema = dataframe_mod.DeviceParquetScan.toArrowSchema;
        pub const toArrowSchemaProjection = dataframe_mod.DeviceParquetScan.toArrowSchemaProjection;
        pub const toArrowFields = dataframe_mod.DeviceParquetScan.toArrowFields;
        pub const toArrowFieldsProjection = dataframe_mod.DeviceParquetScan.toArrowFieldsProjection;
        pub const arrowFieldCount = dataframe_mod.DeviceParquetScan.arrowFieldCount;
        pub const arrowFieldNameAt = dataframe_mod.DeviceParquetScan.arrowFieldNameAt;
        pub const arrowFieldNames = dataframe_mod.DeviceParquetScan.arrowFieldNames;
        pub const arrowFieldIndex = dataframe_mod.DeviceParquetScan.arrowFieldIndex;
        pub const hasArrowField = dataframe_mod.DeviceParquetScan.hasArrowField;
        pub const hasAllArrowFields = dataframe_mod.DeviceParquetScan.hasAllArrowFields;
        pub const hasAnyArrowField = dataframe_mod.DeviceParquetScan.hasAnyArrowField;
        pub const arrowFieldDTypeAt = dataframe_mod.DeviceParquetScan.arrowFieldDTypeAt;
        pub const arrowFieldDType = dataframe_mod.DeviceParquetScan.arrowFieldDType;
        pub const arrowFieldDTypes = dataframe_mod.DeviceParquetScan.arrowFieldDTypes;
        pub const arrowFieldDTypesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypesProjection;
        pub const arrowFieldDTypeNames = dataframe_mod.DeviceParquetScan.arrowFieldDTypeNames;
        pub const arrowFieldDTypeNamesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeNamesProjection;
        pub const arrowFieldDTypeByteSizes = dataframe_mod.DeviceParquetScan.arrowFieldDTypeByteSizes;
        pub const arrowFieldDTypeByteSizesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeByteSizesProjection;
        pub const arrowFieldDTypeBitSizes = dataframe_mod.DeviceParquetScan.arrowFieldDTypeBitSizes;
        pub const arrowFieldDTypeBitSizesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeBitSizesProjection;
        pub const arrowFieldDTypeClassMask = dataframe_mod.DeviceParquetScan.arrowFieldDTypeClassMask;
        pub const arrowFieldDTypeClassMaskProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeClassMaskProjection;
        pub const arrowFieldDTypeClassCount = dataframe_mod.DeviceParquetScan.arrowFieldDTypeClassCount;
        pub const arrowFieldDTypeClassCountProjection = dataframe_mod.DeviceParquetScan.arrowFieldDTypeClassCountProjection;
        pub const numericArrowFieldCount = dataframe_mod.DeviceParquetScan.numericArrowFieldCount;
        pub const numericArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.numericArrowFieldCountProjection;
        pub const floatArrowFieldCount = dataframe_mod.DeviceParquetScan.floatArrowFieldCount;
        pub const floatArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.floatArrowFieldCountProjection;
        pub const integerArrowFieldCount = dataframe_mod.DeviceParquetScan.integerArrowFieldCount;
        pub const integerArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.integerArrowFieldCountProjection;
        pub const boolArrowFieldCount = dataframe_mod.DeviceParquetScan.boolArrowFieldCount;
        pub const boolArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.boolArrowFieldCountProjection;
        pub const arrowFieldNullableAt = dataframe_mod.DeviceParquetScan.arrowFieldNullableAt;
        pub const arrowFieldNullable = dataframe_mod.DeviceParquetScan.arrowFieldNullable;
        pub const arrowFieldNullableMask = dataframe_mod.DeviceParquetScan.arrowFieldNullableMask;
        pub const arrowFieldNullableMaskProjection = dataframe_mod.DeviceParquetScan.arrowFieldNullableMaskProjection;
        pub const nullableArrowFieldCount = dataframe_mod.DeviceParquetScan.nullableArrowFieldCount;
        pub const nullableArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.nullableArrowFieldCountProjection;
        pub const nonNullableArrowFieldCount = dataframe_mod.DeviceParquetScan.nonNullableArrowFieldCount;
        pub const nonNullableArrowFieldCountProjection = dataframe_mod.DeviceParquetScan.nonNullableArrowFieldCountProjection;
        pub const arrowFieldNullCount = dataframe_mod.DeviceParquetScan.arrowFieldNullCount;
        pub const arrowFieldValidCount = dataframe_mod.DeviceParquetScan.arrowFieldValidCount;
        pub const arrowFieldNullCounts = dataframe_mod.DeviceParquetScan.arrowFieldNullCounts;
        pub const arrowFieldNullCountsProjection = dataframe_mod.DeviceParquetScan.arrowFieldNullCountsProjection;
        pub const arrowFieldValidCounts = dataframe_mod.DeviceParquetScan.arrowFieldValidCounts;
        pub const arrowFieldValidCountsProjection = dataframe_mod.DeviceParquetScan.arrowFieldValidCountsProjection;
        pub const arrowFieldNullRatios = dataframe_mod.DeviceParquetScan.arrowFieldNullRatios;
        pub const arrowFieldNullRatiosProjection = dataframe_mod.DeviceParquetScan.arrowFieldNullRatiosProjection;
        pub const arrowFieldValidRatios = dataframe_mod.DeviceParquetScan.arrowFieldValidRatios;
        pub const arrowFieldValidRatiosProjection = dataframe_mod.DeviceParquetScan.arrowFieldValidRatiosProjection;
        pub const arrowNullCount = dataframe_mod.DeviceParquetScan.arrowNullCount;
        pub const arrowNullCountProjection = dataframe_mod.DeviceParquetScan.arrowNullCountProjection;
        pub const arrowValidCount = dataframe_mod.DeviceParquetScan.arrowValidCount;
        pub const arrowValidCountProjection = dataframe_mod.DeviceParquetScan.arrowValidCountProjection;
        pub const arrowNullRatio = dataframe_mod.DeviceParquetScan.arrowNullRatio;
        pub const arrowNullRatioProjection = dataframe_mod.DeviceParquetScan.arrowNullRatioProjection;
        pub const arrowValidRatio = dataframe_mod.DeviceParquetScan.arrowValidRatio;
        pub const arrowValidRatioProjection = dataframe_mod.DeviceParquetScan.arrowValidRatioProjection;
        pub const hasNullableArrowFields = dataframe_mod.DeviceParquetScan.hasNullableArrowFields;
        pub const allArrowFieldsNullable = dataframe_mod.DeviceParquetScan.allArrowFieldsNullable;
        pub const arrowColumnSchemaAt = dataframe_mod.DeviceParquetScan.arrowColumnSchemaAt;
        pub const arrowColumnSchema = dataframe_mod.DeviceParquetScan.arrowColumnSchema;
        pub const arrowColumnSchemas = dataframe_mod.DeviceParquetScan.arrowColumnSchemas;
        pub const arrowColumnSchemasProjection = dataframe_mod.DeviceParquetScan.arrowColumnSchemasProjection;
        pub const arrowSchemaSummary = dataframe_mod.DeviceParquetScan.arrowSchemaSummary;
        pub const arrowSchemaSummaryProjection = dataframe_mod.DeviceParquetScan.arrowSchemaSummaryProjection;
        pub const arrowFieldDataNbytes = dataframe_mod.DeviceParquetScan.arrowFieldDataNbytes;
        pub const arrowFieldDataNbytesProjection = dataframe_mod.DeviceParquetScan.arrowFieldDataNbytesProjection;
        pub const arrowFieldValidityNbytes = dataframe_mod.DeviceParquetScan.arrowFieldValidityNbytes;
        pub const arrowFieldValidityNbytesProjection = dataframe_mod.DeviceParquetScan.arrowFieldValidityNbytesProjection;
        pub const arrowFieldTotalNbytes = dataframe_mod.DeviceParquetScan.arrowFieldTotalNbytes;
        pub const arrowFieldTotalNbytesProjection = dataframe_mod.DeviceParquetScan.arrowFieldTotalNbytesProjection;
        pub const arrowDataNbytes = dataframe_mod.DeviceParquetScan.arrowDataNbytes;
        pub const arrowDataNbytesProjection = dataframe_mod.DeviceParquetScan.arrowDataNbytesProjection;
        pub const arrowValidityNbytes = dataframe_mod.DeviceParquetScan.arrowValidityNbytes;
        pub const arrowValidityNbytesProjection = dataframe_mod.DeviceParquetScan.arrowValidityNbytesProjection;
        pub const arrowTotalNbytes = dataframe_mod.DeviceParquetScan.arrowTotalNbytes;
        pub const arrowTotalNbytesProjection = dataframe_mod.DeviceParquetScan.arrowTotalNbytesProjection;
        pub const arrowSchemaEquals = dataframe_mod.DeviceParquetScan.arrowSchemaEquals;
        pub const arrowSameSchema = dataframe_mod.DeviceParquetScan.arrowSameSchema;
        pub const arrowSchemaCompatible = dataframe_mod.DeviceParquetScan.arrowSchemaCompatible;
        pub const arrowSchemaEqualsSchemas = dataframe_mod.DeviceParquetScan.arrowSchemaEqualsSchemas;
        pub const arrowSchemaEqualsFrame = dataframe_mod.DeviceParquetScan.arrowSchemaEqualsFrame;
        pub const arrowSameSchemaFrame = dataframe_mod.DeviceParquetScan.arrowSameSchemaFrame;
        pub const arrowSchemaCompatibleFrame = dataframe_mod.DeviceParquetScan.arrowSchemaCompatibleFrame;
        pub const hasArrowProjection = dataframe_mod.DeviceParquetScan.hasArrowProjection;
        pub const clearProjection = dataframe_mod.DeviceParquetScan.clearProjection;
        pub const clearRangePredicate = dataframe_mod.DeviceParquetScan.clearRangePredicate;
        pub const clearNullPredicate = dataframe_mod.DeviceParquetScan.clearNullPredicate;
        pub const clearPredicate = dataframe_mod.DeviceParquetScan.clearPredicate;
        pub const clearPushdown = dataframe_mod.DeviceParquetScan.clearPushdown;
        pub const resetPushdown = dataframe_mod.DeviceParquetScan.resetPushdown;
        pub const select = dataframe_mod.DeviceParquetScan.select;
        pub const appendSelect = dataframe_mod.DeviceParquetScan.appendSelect;
        pub const dropSelected = dataframe_mod.DeviceParquetScan.dropSelected;
        pub const selectAll = dataframe_mod.DeviceParquetScan.selectAll;
        pub const selectExcept = dataframe_mod.DeviceParquetScan.selectExcept;
        pub const intersectSelect = dataframe_mod.DeviceParquetScan.intersectSelect;
        pub const whereRange = dataframe_mod.DeviceParquetScan.whereRange;
        pub const whereMin = dataframe_mod.DeviceParquetScan.whereMin;
        pub const whereMax = dataframe_mod.DeviceParquetScan.whereMax;
        pub const whereBetween = dataframe_mod.DeviceParquetScan.whereBetween;
        pub const whereGe = dataframe_mod.DeviceParquetScan.whereGe;
        pub const whereLe = dataframe_mod.DeviceParquetScan.whereLe;
        pub const whereGt = dataframe_mod.DeviceParquetScan.whereGt;
        pub const whereLt = dataframe_mod.DeviceParquetScan.whereLt;
        pub const whereEq = dataframe_mod.DeviceParquetScan.whereEq;
        pub const whereBool = dataframe_mod.DeviceParquetScan.whereBool;
        pub const whereNull = dataframe_mod.DeviceParquetScan.whereNull;
        pub const whereIsNull = dataframe_mod.DeviceParquetScan.whereIsNull;
        pub const whereIsNotNull = dataframe_mod.DeviceParquetScan.whereIsNotNull;
        pub const whereNotNull = dataframe_mod.DeviceParquetScan.whereNotNull;
        pub const collect = dataframe_mod.DeviceParquetScan.collect;
        pub const explain = dataframe_mod.DeviceParquetScan.explain;
        pub const explainSummary = dataframe_mod.DeviceParquetScan.explainSummary;
    };
} else struct {};

pub const DeviceDataFrameArrow = if (build_options.enable_boltha) ArrowExport.DataFrame else struct {};
pub const DeviceDataFrameViewArrow = if (build_options.enable_boltha) ArrowExport.DataFrameView else struct {};
pub const DeviceLazyFrameArrow = if (build_options.enable_boltha) ArrowExport.LazyFrame else struct {};
pub const DeviceColumnArrow = if (build_options.enable_boltha) ArrowExport.Column else struct {};
pub const DeviceColumnViewArrow = if (build_options.enable_boltha) ArrowExport.ColumnView else struct {};
pub const DeviceColumnSchemaArrow = if (build_options.enable_boltha) ArrowExport.ColumnSchema else struct {};

pub const CsrMatrix = sparse.CsrMatrix;
pub const CscMatrix = sparse.CscMatrix;
pub const CooMatrix = sparse.CooMatrix;
pub const SparseError = sparse.SparseError;
pub const SparseProfile = sparse.SparseProfile;
pub const SparseDiffSummary = sparse.SparseDiffSummary;
pub const SparseResidualSummary = sparse.SparseResidualSummary;
pub const csrFromDense = sparse.csrFromDense;
pub const csrFromCompressed = sparse.csrFromCompressed;
pub const cscFromDense = sparse.cscFromDense;
pub const cscFromCompressed = sparse.cscFromCompressed;
pub const cooFromDense = sparse.cooFromDense;
pub const cooFromSlices = sparse.cooFromSlices;
pub const cooEye = sparse.cooEye;
pub const cooIdentity = sparse.cooIdentity;
pub const csrEye = sparse.csrEye;
pub const csrIdentity = sparse.csrIdentity;
pub const cscEye = sparse.cscEye;
pub const cscIdentity = sparse.cscIdentity;
pub const cooFromDiagonal = sparse.cooFromDiagonal;
pub const csrFromDiagonal = sparse.csrFromDiagonal;
pub const cscFromDiagonal = sparse.cscFromDiagonal;

pub const MatrixNormOrder = array_mod.MatrixNormOrder;
pub const Triangle = array_mod.Triangle;
pub const Diagonal = array_mod.Diagonal;
pub const QrResult = array_mod.QrResult;
pub const SvdResult = array_mod.SvdResult;
pub const EighResult = array_mod.EighResult;
pub const LuResult = array_mod.LuResult;

pub const LayoutOrder = layered_array_mod.LayoutOrder;
pub const RuntimeLayout = layered_array_mod.RuntimeLayout;
pub const StaticLayout = layered_array_mod.StaticLayout;
pub const StaticArray = layered_array_mod.StaticArray;
pub const SymbolicLayout = layered_array_mod.SymbolicLayout;
pub const SymbolicArray = layered_array_mod.SymbolicArray;
pub const DimExpr = layered_array_mod.DimExpr;
pub const DimBinding = layered_array_mod.DimBinding;
pub const dim = layered_array_mod.dim;
pub const symbol = layered_array_mod.symbol;
pub const dimAdd = layered_array_mod.dimAdd;
pub const dimSub = layered_array_mod.dimSub;
pub const dimMul = layered_array_mod.dimMul;
pub const AnyArray = layered_array_mod.AnyArray;
pub const CreationOptions = layered_array_mod.CreationOptions;
pub const Context = layered_array_mod.Context;
pub const cpu = Device.cpu;
pub const options = layered_array_mod.options;
pub const onDevice = layered_array_mod.onDevice;
pub const seeded = layered_array_mod.seeded;
pub const seededOn = layered_array_mod.seededOn;

pub fn cuda(index: usize) Device {
    return Device.cuda(index);
}

pub fn mps(index: usize) Device {
    return Device.mps(index);
}

pub fn withAllocator(allocator: @import("std").mem.Allocator) Context {
    return layered_array_mod.withAllocator(allocator);
}

pub fn withSeed(allocator: @import("std").mem.Allocator, seed: u64) Context {
    return layered_array_mod.withSeed(allocator, seed);
}

pub fn add(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return lhs.add(rhs);
}

pub fn matmul(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return lhs.matmul(rhs);
}

pub fn matmulOut(lhs: anytype, rhs: @TypeOf(lhs), out: @TypeOf(lhs)) ArrayError!void {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, out);
    return lhs.matmulOut(rhs, out);
}

pub fn matmulAddOut(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs), out: @TypeOf(lhs)) ArrayError!void {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    try requireSameDevice(lhs, out);
    return lhs.matmulAddOut(rhs, addend, out);
}

pub fn matmulAdd(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return lhs.matmulAdd(rhs, addend);
}

pub fn matmulAddSqrt(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return lhs.matmulAddSqrt(rhs, addend);
}

pub fn einsumUnary(subscripts: []const u8, input: anytype) ArrayError!@TypeOf(input) {
    return einsum_mod.einsumUnary(subscripts, input);
}

pub fn einsum1(subscripts: []const u8, input: anytype) ArrayError!@TypeOf(input) {
    return einsum_mod.einsum1(subscripts, input);
}

pub fn einsum3(subscripts: []const u8, a: anytype, b: @TypeOf(a), c: @TypeOf(a)) ArrayError!@TypeOf(a) {
    return einsum_mod.einsum3(subscripts, a, b, c);
}

pub fn einsum(subscripts: []const u8, lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    return einsum_mod.einsum(subscripts, lhs, rhs);
}

pub fn tensordot(lhs: anytype, rhs: @TypeOf(lhs), axes_lhs: []const usize, axes_rhs: []const usize) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return lhs.contractAxes(rhs, axes_lhs, axes_rhs);
}

pub fn tensorDot(lhs: anytype, rhs: @TypeOf(lhs), axes_lhs: []const usize, axes_rhs: []const usize) ArrayError!@TypeOf(lhs) {
    return tensordot(lhs, rhs, axes_lhs, axes_rhs);
}

pub fn tryMatmulAddTarget(target: DialectBackend, lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!?@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return axiom_backend.executeMatmulAdd(@TypeOf(lhs).Scalar, target, lhs, rhs, addend);
}

fn requireSameDevice(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!void {
    if (!lhs.device.sameDevice(rhs.device)) return error.InvalidDevice;
    if (!lhs.device.isAvailable()) return error.InvalidDevice;
}

pub fn sum(input: anytype, axis_opt: ?isize, keepdims: bool) ArrayError!@TypeOf(input) {
    return input.sum(axis_opt, keepdims);
}

pub fn addScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.addScalar(scalar);
}

pub fn subScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.subScalar(scalar);
}

pub fn mulScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.mulScalar(scalar);
}

pub fn divScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.divScalar(scalar);
}

pub fn rsubScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.rsubScalar(scalar);
}

pub fn rdivScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.rdivScalar(scalar);
}

pub fn scalarSub(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.scalarSub(scalar);
}

pub fn scalarDiv(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.scalarDiv(scalar);
}

pub fn addScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.addScalarPromote(U, scalar);
}

pub fn subScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.subScalarPromote(U, scalar);
}

pub fn mulScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.mulScalarPromote(U, scalar);
}

pub fn divScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.divScalarPromote(U, scalar);
}

pub fn rsubScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.rsubScalarPromote(U, scalar);
}

pub fn rdivScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.rdivScalarPromote(U, scalar);
}

pub fn scalarSubPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.scalarSubPromote(U, scalar);
}

pub fn scalarDivPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.scalarDivPromote(U, scalar);
}

test "top-level ops respect device dispatch" {
    const gpa = @import("std").testing.allocator;
    var lhs = try Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer lhs.deinit();
    var rhs = try Array(f32).fromSlice(gpa, &.{ 10, 20, 30, 40 }, &.{ 2, 2 });
    defer rhs.deinit();

    var sum_cpu = try add(lhs, rhs);
    defer sum_cpu.deinit();
    try @import("std").testing.expect(sum_cpu.device.isCpu());
    try @import("std").testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, sum_cpu.data);

    if (try tryMatmulAddTarget(.cpu, lhs, rhs, sum_cpu)) |target_add_value| {
        var target_add = target_add_value;
        defer target_add.deinit();
        try @import("std").testing.expect(target_add.device.isCpu());
        try @import("std").testing.expectEqualSlices(f32, &.{ 81, 122, 183, 264 }, target_add.data);
    }

    var cuda_rhs = try rhs.clone();
    defer cuda_rhs.deinit();
    cuda_rhs.device = Device.cuda(0);
    try @import("std").testing.expectError(error.InvalidDevice, add(lhs, cuda_rhs));
}

test "no-boltha DeviceDataFrame metadata facade is source-compatible" {
    if (!build_options.enable_boltha) {
        const gpa = std.testing.allocator;
        const frame: DeviceDataFrame = .{};

        try std.testing.expectEqual(@as(usize, 0), frame.height());
        try std.testing.expectEqual(@as(usize, 0), frame.width());
        try std.testing.expectEqual(@as(usize, 0), frame.cellCount());
        try std.testing.expect(frame.isEmpty());
        try std.testing.expect(frame.isHostBacked());
        try std.testing.expect(!frame.isCudaBacked());
        try std.testing.expect(!frame.isMpsBacked());
        try std.testing.expect(!frame.isAcceleratorBacked());
        try std.testing.expect(!frame.isRemoteBacked());
        try std.testing.expect(!frame.isDeviceBacked());
        try std.testing.expect(frame.isDeviceAvailable());
        try std.testing.expectEqualStrings("cpu", frame.deviceBackendName());
        try std.testing.expectEqual(Device.cpu.backend, frame.deviceBackend());
        try std.testing.expect(frame.deviceValue().sameDevice(.cpu));
        try std.testing.expectEqual(@as(usize, 0), frame.deviceIndex());
        try std.testing.expect(frame.columnNamesUnique());
        try std.testing.expect(!frame.hasDuplicateColumnNames());
        try std.testing.expect(frame.hasAllColumns(&.{}));
        try std.testing.expect(!frame.hasAnyColumn(&.{ "missing", "absent" }));

        const parquet_scan: DeviceParquetScan = .{};
        try std.testing.expect(parquet_scan.deviceValue().sameDevice(.cpu));
        try std.testing.expectEqual(Device.cpu.backend, parquet_scan.deviceBackend());
        try std.testing.expectEqualStrings("cpu", parquet_scan.deviceBackendName());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.deviceIndex());
        try std.testing.expect(parquet_scan.isCpu());
        try std.testing.expect(parquet_scan.isHostBacked());
        try std.testing.expect(!parquet_scan.isCuda());
        try std.testing.expect(!parquet_scan.isCudaBacked());
        try std.testing.expect(!parquet_scan.isMps());
        try std.testing.expect(!parquet_scan.isMpsBacked());
        try std.testing.expect(!parquet_scan.isAcceleratorBacked());
        try std.testing.expect(!parquet_scan.isRemoteBacked());
        try std.testing.expect(!parquet_scan.isDeviceBacked());
        try std.testing.expect(parquet_scan.isDeviceAvailable());
        try std.testing.expect(parquet_scan.sameDevice(.{}));
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.sourceNbytes());
        try std.testing.expectEqual(@as(u64, 0), parquet_scan.sourcePtr());
        try std.testing.expectEqual(@as(u64, 0), parquet_scan.dataPtr());
        try std.testing.expect(!parquet_scan.hasSourcePtr());
        try std.testing.expectEqual(@as(u64, 0), parquet_scan.sourceEndPtr());
        const empty_source_range: DeviceParquetScanSourceRange = parquet_scan.sourceRange();
        try std.testing.expect(empty_source_range.isEmpty());
        try std.testing.expect(!empty_source_range.hasPtr());
        try std.testing.expectEqual(@as(u64, 0), empty_source_range.sourcePtr());
        try std.testing.expectEqual(@as(usize, 0), empty_source_range.sourceNbytes());
        try std.testing.expectEqual(@as(u64, 0), empty_source_range.sourceEndPtr());
        try std.testing.expect(empty_source_range.sameRange(.{}));
        try std.testing.expect(empty_source_range.sameStorage(.{}));
        try std.testing.expect(!empty_source_range.mayOverlap(.{}));
        try std.testing.expect(parquet_scan.sharesSource(parquet_scan));
        try std.testing.expect(parquet_scan.sameSource(parquet_scan));
        try std.testing.expect(parquet_scan.sameStorage(parquet_scan));
        try std.testing.expect(!parquet_scan.mayOverlap(parquet_scan));
        var retarget_scan = parquet_scan;
        try std.testing.expectError(error.FeatureUnavailable, retarget_scan.setDevice(.cpu));
        try std.testing.expectError(error.FeatureUnavailable, retarget_scan.retarget(.cpu));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.to(.cpu));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.withDevice(.cpu));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.cpu());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.cuda(0));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.mps(0));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupSummaryAt(0));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupSummaries(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupRowCounts(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupColumnChunkCounts(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupTotalNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupTotalCompressedNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupCompressedNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupUncompressedNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupCompressionRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupMetadataCoverageRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupMissingMetadataRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupColumnIndexCoverageRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupOffsetIndexCoverageRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupPageIndexCoverageRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupBloomFilterCoverageRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetRowGroupSizedBloomFilterCoverageRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetFieldCompressedNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetFieldCompressedNbytesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetFieldUncompressedNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetFieldUncompressedNbytesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetCompressedNbytes());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetCompressedNbytesProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetUncompressedNbytes());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetUncompressedNbytesProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetFieldCompressionRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetFieldCompressionRatiosProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.parquetCompressionRatioProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.rowCount());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.nRows());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.columnCount());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.width());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.cols());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.nCols());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.cellCount());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.shape());
        try std.testing.expect(!parquet_scan.hasRows());
        try std.testing.expect(!parquet_scan.hasColumns());
        try std.testing.expect(!parquet_scan.hasShape(0, 0));
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.sourceByteCount());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.nbytes());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.byteCount());
        try std.testing.expect(parquet_scan.isEmpty());
        try std.testing.expect(!parquet_scan.isNonEmpty());
        try std.testing.expect(!parquet_scan.hasBytes());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.projectionMetadataNbytes());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.rangePredicateMetadataNbytes());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.nullPredicateMetadataNbytes());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.predicateMetadataNbytes());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.pushdownMetadataNbytes());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.ownedNbytes());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.memoryUsage());
        try std.testing.expectEqual(@as(usize, 0), parquet_scan.estimatedSize());
        try std.testing.expect(parquet_scan.projectionNameAt(0) == null);
        try std.testing.expect(parquet_scan.projectionIndex("missing") == null);
        try std.testing.expect(!parquet_scan.projectionContains("missing"));
        try std.testing.expect(parquet_scan.projectsColumn("missing"));
        try std.testing.expect(!parquet_scan.hasPredicate());
        try std.testing.expect(parquet_scan.predicateColumn() == null);
        try std.testing.expect(!parquet_scan.hasPredicateFor("missing"));
        try std.testing.expect(parquet_scan.rangePredicate() == null);
        try std.testing.expect(parquet_scan.rangePredicateDType() == null);
        try std.testing.expect(!parquet_scan.hasRangePredicateFor("missing"));
        try std.testing.expect(parquet_scan.nullPredicateWantNulls() == null);
        try std.testing.expect(!parquet_scan.hasNullPredicateFor("missing"));
        const empty_pushdown: DeviceParquetScanPushdownSummary = parquet_scan.pushdownSummary();
        try std.testing.expect(empty_pushdown.isEmpty());
        try std.testing.expect(!empty_pushdown.hasPushdown());
        try std.testing.expect(!empty_pushdown.hasProjection());
        try std.testing.expect(!empty_pushdown.hasPredicate());
        try std.testing.expect(empty_pushdown.projectionNameAt(0) == null);
        try std.testing.expect(empty_pushdown.projectionIndex("missing") == null);
        try std.testing.expect(empty_pushdown.projectsColumn("missing"));
        try std.testing.expect(empty_pushdown.predicateColumn() == null);
        try std.testing.expect(!empty_pushdown.hasPredicateFor("missing"));
        try std.testing.expect(empty_pushdown.rangePredicateDType() == null);
        try std.testing.expect(empty_pushdown.nullPredicateWantNulls() == null);
        try std.testing.expectEqual(@as(usize, 0), empty_pushdown.pushdownMetadataNbytes());
        const empty_scan_summary: DeviceParquetScanSummary = parquet_scan.summary();
        try std.testing.expect(empty_scan_summary.deviceValue().sameDevice(.cpu));
        try std.testing.expect(empty_scan_summary.isCpu());
        try std.testing.expect(empty_scan_summary.isHostBacked());
        try std.testing.expect(!empty_scan_summary.isDeviceBacked());
        try std.testing.expect(empty_scan_summary.isDeviceAvailable());
        try std.testing.expectEqual(@as(usize, 0), empty_scan_summary.sourceNbytes());
        try std.testing.expectEqual(@as(u64, 0), empty_scan_summary.sourcePtr());
        try std.testing.expectEqual(@as(u64, 0), empty_scan_summary.dataPtr());
        try std.testing.expect(!empty_scan_summary.hasSourcePtr());
        try std.testing.expectEqual(@as(u64, 0), empty_scan_summary.sourceEndPtr());
        try std.testing.expect(empty_scan_summary.sourceRange().isEmpty());
        try std.testing.expectEqual(@as(usize, 0), empty_scan_summary.ownedNbytes());
        try std.testing.expect(empty_scan_summary.isEmpty());
        try std.testing.expect(!empty_scan_summary.hasPushdown());
        try std.testing.expect(empty_scan_summary.pushdownSummary().isEmpty());
        var mutable_scan = parquet_scan;
        mutable_scan.clearProjection();
        mutable_scan.clearRangePredicate();
        mutable_scan.clearNullPredicate();
        mutable_scan.clearPredicate();
        mutable_scan.clearPushdown();
        mutable_scan.resetPushdown();

        const shape_value = frame.shape();
        try std.testing.expectEqual(@as(usize, 0), shape_value.rows);
        try std.testing.expectEqual(@as(usize, 0), shape_value.cols);
        try std.testing.expect(frame.hasShape(0, 0));
        try std.testing.expect(frame.sameStorage(frame));
        try std.testing.expect(frame.columnIndex("missing") == null);
        try std.testing.expectError(error.ColumnNotFound, frame.columnDType("missing"));
        try std.testing.expectError(error.IndexOutOfBounds, frame.columnDTypeAt(0));
        try std.testing.expectError(error.FeatureUnavailable, frame.columnDTypes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, frame.columnNullCounts(gpa));
        try std.testing.expectError(error.FeatureUnavailable, frame.columnSchemas(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowColumnSchemasProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowSchemaSummaryProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldDTypesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldDTypeNamesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldDTypeByteSizesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldDTypeBitSizesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldDTypeClassMaskProjection(gpa, &.{"id"}, .numeric));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldDTypeClassCountProjection(&.{"id"}, .numeric));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.numericArrowFieldCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldNullableMaskProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.nullableArrowFieldCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.nonNullableArrowFieldCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldNullCount("id"));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldValidCount("id"));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldNullCounts(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldNullCountsProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldValidCounts(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldValidCountsProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldNullRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldNullRatiosProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldValidRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldValidRatiosProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowNullCount());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowNullCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowValidCount());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowValidCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowNullRatio());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowNullRatioProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowValidRatio());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowValidRatioProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldDataNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldDataNbytesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldValidityNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldValidityNbytesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldTotalNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowFieldTotalNbytesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowDataNbytes());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowDataNbytesProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowValidityNbytes());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowValidityNbytesProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowTotalNbytes());
        try std.testing.expectError(error.FeatureUnavailable, parquet_scan.arrowTotalNbytesProjection(&.{"id"}));

        const lazy_frame: DeviceLazyFrame = .{};
        try std.testing.expect(!lazy_frame.hasSchemaProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnNullCounts(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnNullCountsProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnValidCounts(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnValidCountsProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.nullCount());
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.nullCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.validCount());
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.validCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.nullRatio());
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.nullRatioProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.validRatio());
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.validRatioProjection(&.{"id"}));
        try std.testing.expect(!lazy_frame.anyNull());
        try std.testing.expect(!lazy_frame.anyNullProjection(&.{"id"}));
        try std.testing.expect(!lazy_frame.allNull());
        try std.testing.expect(!lazy_frame.allNullProjection(&.{"id"}));
        try std.testing.expect(!lazy_frame.anyValid());
        try std.testing.expect(!lazy_frame.anyValidProjection(&.{"id"}));
        try std.testing.expect(!lazy_frame.allValid());
        try std.testing.expect(!lazy_frame.allValidProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.anyNullColumn("id"));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.allNullColumn("id"));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.anyValidColumn("id"));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.allValidColumn("id"));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnNullRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnNullRatiosProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnValidRatios(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnValidRatiosProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnHasNullsMask(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnHasNullsMaskProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnsWithNullsCount());
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnsWithNullsCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnsWithoutNullsCount());
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnsWithoutNullsCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDataNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDataNbytesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDataMemoryUsage(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDataMemoryUsageProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnValidityNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnValidityNbytesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnValidityMemoryUsage(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnValidityMemoryUsageProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnTotalNbytes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnTotalNbytesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnMemoryUsage(gpa));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnMemoryUsageProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.dataNbytes());
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.dataNbytesProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.validityNbytes());
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.validityNbytesProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.totalNbytes());
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.totalNbytesProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.memoryUsageProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.estimatedSizeProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDTypesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDTypeNamesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.dtypeNamesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDTypeByteSizesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDTypeBitSizesProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDTypeClassMaskProjection(gpa, &.{"id"}, .numeric));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnIsNumericMaskProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnIsFloatMaskProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnIsBoolMaskProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnDTypeClassCountProjection(&.{"id"}, .numeric));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.numericColumnCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnNullableMaskProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.nullableColumnCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.nonNullableColumnCountProjection(&.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.columnSchemasProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.schemaProjection(gpa, &.{"id"}));
        try std.testing.expectError(error.FeatureUnavailable, lazy_frame.schemaSummaryProjection(gpa, &.{"id"}));

        const column = DeviceColumn{ .i32 = undefined };
        const bool_column = DeviceColumn{ .bool = undefined };
        try std.testing.expectEqual(@as(usize, 0), column.len());
        try std.testing.expectEqual(column.len(), column.rowCount());
        try std.testing.expectEqual(column.len(), column.height());
        try std.testing.expectEqual(column.len(), column.nRows());
        try std.testing.expectEqual(@as(usize, 0), column.shape().rows);
        try std.testing.expect(column.shapeEquals(0));
        try std.testing.expect(column.hasShape(0));
        try std.testing.expect(column.isEmpty());
        try std.testing.expect(!column.isNonEmpty());
        try std.testing.expect(!column.hasRows());
        try std.testing.expectEqual(column.len(), column.cellCount());
        try std.testing.expectEqual(DeviceDType.i32, column.dtype());
        try std.testing.expectEqualStrings("i32", column.dtypeName());
        try std.testing.expectEqual(DeviceDType.i32.bitSize(), column.dtypeBitSize());
        try std.testing.expect(column.isNumeric());
        try std.testing.expect(column.isReal());
        try std.testing.expect(!column.isFloat());
        try std.testing.expect(column.isInteger());
        try std.testing.expect(column.isSignedInteger());
        try std.testing.expect(!column.isUnsignedInteger());
        try std.testing.expect(!column.isBool());
        try std.testing.expect(!column.isComplex());
        try std.testing.expect(column.isCpu());
        try std.testing.expect(column.isHostBacked());
        try std.testing.expect(!column.isCudaBacked());
        try std.testing.expect(!column.isMpsBacked());
        try std.testing.expect(!column.isAcceleratorBacked());
        try std.testing.expect(!column.isRemoteBacked());
        try std.testing.expect(!column.isDeviceBacked());
        try std.testing.expect(column.isDeviceAvailable());
        try std.testing.expectEqualStrings("cpu", column.deviceBackendName());
        try std.testing.expectEqual(Device.cpu.backend, column.deviceBackend());
        try std.testing.expect(column.deviceValue().sameDevice(.cpu));
        try std.testing.expectEqual(@as(usize, 0), column.deviceIndex());
        try std.testing.expectEqual(@as(usize, 0), column.memoryUsage());
        try std.testing.expectEqual(@as(u64, 0), column.dataPtr());
        try std.testing.expect(!column.hasValidity());
        try std.testing.expect(column.validityPtr() == null);
        try std.testing.expectEqual(DeviceValidityEncoding.none, column.validityEncoding());
        try std.testing.expect(column.sameDevice(bool_column));
        try std.testing.expect(column.sameLength(bool_column));
        try std.testing.expect(column.sameShape(bool_column));
        try std.testing.expect(column.lengthEquals(0));
        try std.testing.expect(!column.sameDType(bool_column));
        try std.testing.expect(column.sameNullability(bool_column));
        try std.testing.expect(!column.schemaEquals(bool_column));
        try std.testing.expect(column.sameStorage(bool_column));
        try std.testing.expect(column.schema("id").schemaEquals(column.schema("id")));

        const view_names = [_][]const u8{"id"};
        var view_columns = [_]DeviceColumnView{.{
            .dtype = .i32,
            .rows = 2,
            .device = .cpu,
            .data_ptr = 0x10,
            .data_nbytes = 2 * @sizeOf(i32),
        }};
        const view = DeviceDataFrameView{
            .allocator = gpa,
            .names = &view_names,
            .columns = &view_columns,
            .rows = 2,
            .device = .cpu,
        };
        try std.testing.expectEqual(@as(usize, 2), view.nRows());
        try std.testing.expectEqual(@as(usize, 1), view.nCols());
        try std.testing.expect(!view.isEmpty());
        try std.testing.expect(view.columnNamesUnique());
        try std.testing.expect(!view.hasDuplicateColumnNames());
        try std.testing.expectEqual(@as(usize, 0), view.duplicateColumnNameCount());
        try std.testing.expect(view.sameDevice(view));
        try std.testing.expect(view.sameShape(view));
        try std.testing.expect(view.sameStorage(view));
        try std.testing.expect(view.hasShape(2, 1));
        try std.testing.expect(view.sameHeight(view));
        try std.testing.expect(view.sameWidth(view));
        const view_dtypes = try view.columnDTypes(gpa);
        defer gpa.free(view_dtypes);
        try std.testing.expectEqualSlices(DeviceDType, &.{.i32}, view_dtypes);
        const view_dtype_names = try view.dtypeNames(gpa);
        defer gpa.free(view_dtype_names);
        try std.testing.expectEqualStrings("i32", view_dtype_names[0]);
        try std.testing.expectEqual(@as(usize, 1), view.numericColumnCount());
        try std.testing.expectEqual(@as(usize, 1), view.integerColumnCount());
        try std.testing.expectEqual(@as(usize, 1), view.signedIntegerColumnCount());
        const view_integer_mask = try view.columnIsIntegerMask(gpa);
        defer gpa.free(view_integer_mask);
        try std.testing.expectEqualSlices(bool, &.{true}, view_integer_mask);
        try std.testing.expect(view.hasColumn("id"));
        try std.testing.expect(view.hasAllColumns(&.{"id"}));
        try std.testing.expectEqual(DeviceDType.i32, try view.columnDType("id"));
        try std.testing.expectEqual(DeviceDType.i32, try view.columnDTypeAt(0));
        const duplicate_view_names = [_][]const u8{ "id", "id" };
        var duplicate_view_columns = [_]DeviceColumnView{ view_columns[0], view_columns[0] };
        const duplicate_view = DeviceDataFrameView{
            .allocator = gpa,
            .names = &duplicate_view_names,
            .columns = &duplicate_view_columns,
            .rows = 2,
            .device = .cpu,
        };
        try std.testing.expect(!duplicate_view.columnNamesUnique());
        try std.testing.expect(duplicate_view.hasDuplicateColumnNames());
        try std.testing.expectEqual(@as(usize, 1), duplicate_view.duplicateColumnNameCount());
        try std.testing.expect(!view.schemaEquals(duplicate_view));
        try std.testing.expect(!view.sameSchema(duplicate_view));
        const id_view = try view.columnView("id");
        try std.testing.expectEqual(DeviceDType.i32, id_view.dtype);
        try std.testing.expectEqual(@as(usize, 2), id_view.len());
        try std.testing.expectEqual(id_view.len(), id_view.rowCount());
        try std.testing.expectEqual(id_view.len(), id_view.height());
        try std.testing.expectEqual(id_view.len(), id_view.nRows());
        try std.testing.expectEqual(@as(usize, 2), id_view.shape().rows);
        try std.testing.expect(id_view.shapeEquals(2));
        try std.testing.expect(id_view.hasShape(2));
        try std.testing.expectEqual(@as(u64, 0x10), id_view.dataPtr());
        try std.testing.expect(!id_view.hasValidity());
        try std.testing.expect(id_view.validityPtr() == null);
        try std.testing.expectEqual(DeviceValidityEncoding.none, id_view.validityEncoding());
        try std.testing.expect(!id_view.isEmpty());
        try std.testing.expect(id_view.isNonEmpty());
        try std.testing.expect(id_view.hasRows());
        try std.testing.expectEqual(id_view.len(), id_view.cellCount());
        try std.testing.expectEqual(@as(usize, 0), id_view.nullCount());
        try std.testing.expectEqual(@as(usize, 2), id_view.validCount());
        try std.testing.expect(!id_view.anyNull());
        try std.testing.expect(!id_view.allNull());
        try std.testing.expect(id_view.anyValid());
        try std.testing.expect(id_view.allValid());
        try std.testing.expectEqual(@as(usize, 2 * @sizeOf(i32)), id_view.dataNbytes());
        try std.testing.expectEqual(id_view.totalNbytes(), id_view.memoryUsage());
        try std.testing.expectEqualStrings("i32", id_view.dtypeName());
        try std.testing.expect(id_view.isCpu());
        try std.testing.expect(id_view.isHostBacked());
        try std.testing.expect(!id_view.isCudaBacked());
        try std.testing.expect(!id_view.isMpsBacked());
        try std.testing.expect(!id_view.isAcceleratorBacked());
        try std.testing.expect(!id_view.isRemoteBacked());
        try std.testing.expect(id_view.isDeviceAvailable());
        try std.testing.expectEqualStrings("cpu", id_view.deviceBackendName());
        try std.testing.expectEqual(Device.cpu.backend, id_view.deviceBackend());
        try std.testing.expect(id_view.deviceValue().sameDevice(.cpu));
        try std.testing.expectEqual(@as(usize, 0), id_view.deviceIndex());
        try std.testing.expect(id_view.sameDevice(view_columns[0]));
        try std.testing.expect(id_view.sameLength(view_columns[0]));
        try std.testing.expect(id_view.sameShape(view_columns[0]));
        try std.testing.expect(id_view.lengthEquals(2));
        try std.testing.expect(id_view.sameDType(view_columns[0]));
        try std.testing.expect(id_view.sameNullability(view_columns[0]));
        try std.testing.expect(id_view.schemaEquals(view_columns[0]));
        try std.testing.expect(id_view.sameSchema(view_columns[0]));
        try std.testing.expect(id_view.schemaCompatible(view_columns[0]));
        try std.testing.expect(id_view.sameStorage(view_columns[0]));
        const view_null_counts = try view.columnNullCounts(gpa);
        defer gpa.free(view_null_counts);
        try std.testing.expectEqualSlices(usize, &.{0}, view_null_counts);
        const view_valid_counts = try view.columnValidCounts(gpa);
        defer gpa.free(view_valid_counts);
        try std.testing.expectEqualSlices(usize, &.{2}, view_valid_counts);
        try std.testing.expectEqual(@as(usize, 0), view.nullCount());
        try std.testing.expectEqual(@as(usize, 2), view.validCount());
        try std.testing.expectEqual(@as(usize, 2), view.cellCount());
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), view.nullRatio(), 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), view.validRatio(), 1e-12);
        const view_memory = try view.columnMemoryUsage(gpa);
        defer gpa.free(view_memory);
        try std.testing.expectEqualSlices(usize, &.{id_view.totalNbytes()}, view_memory);
        try std.testing.expectEqual(id_view.dataNbytes(), view.dataNbytes());
        try std.testing.expectEqual(id_view.validityNbytes(), view.validityNbytes());
        try std.testing.expectEqual(id_view.totalNbytes(), view.totalNbytes());
        try std.testing.expectEqual(view.totalNbytes(), view.estimatedSize());
        const id_schema = try view.columnSchema("id");
        try std.testing.expectEqual(DeviceDType.i32, id_schema.dtype);
        try std.testing.expectEqual(@as(usize, 2), id_schema.rows);
        try std.testing.expectEqual(@as(usize, 2), id_schema.len());
        try std.testing.expectEqual(id_schema.len(), id_schema.rowCount());
        try std.testing.expectEqual(id_schema.len(), id_schema.height());
        try std.testing.expectEqual(id_schema.len(), id_schema.nRows());
        try std.testing.expectEqual(@as(usize, 2), id_schema.shape().rows);
        try std.testing.expect(id_schema.shapeEquals(2));
        try std.testing.expect(id_schema.hasShape(2));
        try std.testing.expect(!id_schema.isEmpty());
        try std.testing.expect(id_schema.isNonEmpty());
        try std.testing.expect(id_schema.hasRows());
        try std.testing.expectEqual(id_schema.len(), id_schema.cellCount());
        try std.testing.expectEqualStrings("i32", id_schema.dtypeName());
        try std.testing.expect(id_schema.isNumeric());
        try std.testing.expect(id_schema.isInteger());
        try std.testing.expect(id_schema.isSignedInteger());
        try std.testing.expect(!id_schema.isBool());
        try std.testing.expect(id_schema.allValid());
        try std.testing.expect(!id_schema.anyNull());
        try std.testing.expect(!id_schema.allNull());
        try std.testing.expect(id_schema.anyValid());
        try std.testing.expectEqual(@as(usize, 0), id_schema.nullCount());
        try std.testing.expectEqual(@as(usize, 2), id_schema.validCount());
        try std.testing.expectEqual(id_view.dataNbytes(), id_schema.dataNbytes());
        try std.testing.expectEqual(id_view.dataPtr(), id_schema.dataPtr());
        try std.testing.expectEqual(id_view.validityNbytes(), id_schema.validityNbytes());
        try std.testing.expect(!id_schema.hasValidity());
        try std.testing.expect(id_schema.validityPtr() == null);
        try std.testing.expectEqual(id_view.totalNbytes(), id_schema.totalNbytes());
        try std.testing.expectEqual(id_view.totalNbytes(), id_schema.memoryUsage());
        try std.testing.expect(id_schema.isCpu());
        try std.testing.expect(id_schema.isHostBacked());
        try std.testing.expect(!id_schema.isCudaBacked());
        try std.testing.expect(!id_schema.isMpsBacked());
        try std.testing.expect(!id_schema.isAcceleratorBacked());
        try std.testing.expect(!id_schema.isRemoteBacked());
        try std.testing.expect(!id_schema.isDeviceBacked());
        try std.testing.expect(id_schema.isDeviceAvailable());
        try std.testing.expectEqualStrings("cpu", id_schema.deviceBackendName());
        try std.testing.expectEqual(Device.cpu.backend, id_schema.deviceBackend());
        try std.testing.expect(id_schema.deviceValue().sameDevice(.cpu));
        try std.testing.expectEqual(@as(usize, 0), id_schema.deviceIndex());
        try std.testing.expect(id_schema.schemaEquals(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.sameSchema(try view.columnSchema("id")));
        try std.testing.expect(id_schema.schemaCompatible(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.sameDevice(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.sameLength(try view.columnSchema("id")));
        try std.testing.expect(id_schema.sameShape(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.sameStorage(try view.columnSchema("id")));
        try std.testing.expect(id_schema.lengthEquals(2));
        try std.testing.expect(id_schema.sameDType(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.sameNullability(try view.columnSchema("id")));
        const schema = try view.schema(gpa);
        defer gpa.free(schema);
        try std.testing.expectEqual(@as(usize, 1), schema.len);
        try std.testing.expectEqual(DeviceDType.i32, schema[0].dtype);
        const view_alias = DeviceDataFrameView{
            .allocator = gpa,
            .names = &view_names,
            .columns = &view_columns,
            .rows = 2,
            .device = .cpu,
        };
        try std.testing.expect(view.schemaEquals(view_alias));
        try std.testing.expect(view.sameSchema(view_alias));
        try std.testing.expect(view.schemaCompatible(view_alias));
        try std.testing.expectError(error.ColumnNotFound, view.columnSchema("missing"));
        try std.testing.expectError(error.IndexOutOfBounds, view.columnSchemaAt(1));
        try std.testing.expect(std.mem.eql(u8, "id", try view.columnNameAt(0)));
        try std.testing.expectError(error.IndexOutOfBounds, view.columnAt(1));
    }
}

test {
    _ = array_mod;
    _ = axiom_backend;
    _ = series_mod;
    _ = dataframe_mod;
    _ = layered_array_mod;
    _ = forge_interop_mod;
    _ = histogram2d_device_mod;
    _ = linalg;
    _ = stats;
    _ = sparse;
}
