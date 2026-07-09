//! Vectra is an experimental Zig-native data processing and numerical computing
//! toolkit inspired by NumPy/CuPy/SciPy/Pandas/Polars, with PyTorch-style object
//! methods as the primary public array API.
//!
//! The current implementation provides a compact CPU core: typed arrays,
//! broadcasting arithmetic, reductions, linalg/stat helpers, typed Series,
//! heterogeneous DataFrame operations, sparse matrices, and CSV IO. CUDA/GPU is
//! represented in the API surface and returns `error.InvalidDevice` until a real
//! backend is wired in.

const array_mod = @import("array.zig");
const series_mod = @import("series.zig");
const dataframe_mod = @import("dataframe.zig");
pub const linalg = @import("linalg.zig");
pub const stats = @import("stats.zig");
pub const sparse = @import("sparse.zig");

pub const Array = array_mod.Array;
pub const NDArray = array_mod.NDArray;
pub const ArrayView = array_mod.ArrayView;
pub const NDArrayView = array_mod.NDArrayView;
pub const BFloat16 = array_mod.BFloat16;
pub const Complex64 = array_mod.Complex64;
pub const Complex128 = array_mod.Complex128;
pub const Device = array_mod.Device;
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
pub const ArrayError = array_mod.ArrayError;

pub const Series = series_mod.Series;
pub const DataFrame = dataframe_mod.DataFrame;
pub const Column = dataframe_mod.Column;
pub const ColumnDef = dataframe_mod.ColumnDef;
pub const DataError = dataframe_mod.DataError;

pub const CsrMatrix = sparse.CsrMatrix;
pub const CscMatrix = sparse.CscMatrix;
pub const csrFromDense = sparse.csrFromDense;
pub const csrFromCompressed = sparse.csrFromCompressed;
pub const cscFromDense = sparse.cscFromDense;
pub const cscFromCompressed = sparse.cscFromCompressed;

pub const MatrixNormOrder = array_mod.MatrixNormOrder;
pub const Triangle = array_mod.Triangle;
pub const Diagonal = array_mod.Diagonal;
pub const QrResult = array_mod.QrResult;
pub const SvdResult = array_mod.SvdResult;
pub const EighResult = array_mod.EighResult;
pub const LuResult = array_mod.LuResult;

test {
    _ = array_mod;
    _ = series_mod;
    _ = dataframe_mod;
    _ = linalg;
    _ = stats;
    _ = sparse;
}
