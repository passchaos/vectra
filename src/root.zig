//! Vectra is an experimental Zig-native data processing and numerical computing
//! toolkit inspired by NumPy/CuPy/SciPy/Pandas/Polars, with PyTorch-style array
//! method names where that makes code more fluent.
//!
//! The current implementation provides a compact, dependency-free CPU core:
//! typed arrays, broadcasting arithmetic, reductions, linalg/stat helpers,
//! typed Series, heterogeneous DataFrame operations, and CSV IO. CUDA/GPU is
//! represented in the API surface and returns `error.InvalidDevice` until a real
//! backend is wired in.

pub const tensor_mod = @import("tensor.zig");
pub const series_mod = @import("series.zig");
pub const dataframe_mod = @import("dataframe.zig");
pub const linalg = @import("linalg.zig");
pub const stats = @import("stats.zig");
pub const sparse = @import("sparse.zig");

pub const Array = tensor_mod.Array;
pub const NDArray = tensor_mod.NDArray;
pub const Tensor = tensor_mod.Tensor; // Deprecated compatibility alias; use Array/NDArray for new APIs.
pub const Device = tensor_mod.Device;
pub const DType = tensor_mod.DType;
pub const Slice = tensor_mod.Slice;
pub const ScatterReduce = tensor_mod.ScatterReduce;
pub const TensorError = tensor_mod.TensorError;

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

pub const array = tensor_mod.array;
pub const ndarray = tensor_mod.ndarray;
pub const tensor = tensor_mod.tensor; // Deprecated compatibility alias; use array/ndarray for new APIs.
pub const zeros = tensor_mod.zeros;
pub const ones = tensor_mod.ones;
pub const full = tensor_mod.full;
pub const empty = tensor_mod.empty;
pub const arrayScalar = tensor_mod.arrayScalar;
pub const emptyLike = tensor_mod.emptyLike;
pub const zerosLike = tensor_mod.zerosLike;
pub const onesLike = tensor_mod.onesLike;
pub const fullLike = tensor_mod.fullLike;
pub const arange = tensor_mod.arange;
pub const linspace = tensor_mod.linspace;
pub const rand = tensor_mod.rand;
pub const randn = tensor_mod.randn;
pub const randint = tensor_mod.randint;
pub const uniform = tensor_mod.uniform;
pub const normal = tensor_mod.normal;
pub const bernoulli = tensor_mod.bernoulli;
pub const exponential = tensor_mod.exponential;
pub const gamma = tensor_mod.gamma;
pub const beta = tensor_mod.beta;
pub const poisson = tensor_mod.poisson;
pub const lognormal = tensor_mod.lognormal;
pub const studentT = tensor_mod.studentT;
pub const cauchy = tensor_mod.cauchy;
pub const laplace = tensor_mod.laplace;
pub const weibull = tensor_mod.weibull;
pub const eye = tensor_mod.eye;
pub const cat = tensor_mod.cat;
pub const stack = tensor_mod.stack;
pub const outer = tensor_mod.outer;
pub const where = tensor_mod.where;
pub const takeAlongAxis = tensor_mod.takeAlongAxis;
pub const putAlongAxis = tensor_mod.putAlongAxis;
pub const maskedFill = tensor_mod.maskedFill;
pub const maskedScatter = tensor_mod.maskedScatter;
pub const nonzero = tensor_mod.nonzero;
pub const countNonzero = tensor_mod.countNonzero;
pub const diag = tensor_mod.diag;
pub const diagflat = tensor_mod.diagflat;
pub const sliceAxis = tensor_mod.sliceAxis;
pub const flip = tensor_mod.flip;
pub const roll = tensor_mod.roll;
pub const padConstant = tensor_mod.padConstant;
pub const cumsumAxis = tensor_mod.cumsumAxis;
pub const cumprodAxis = tensor_mod.cumprodAxis;
pub const diff = tensor_mod.diff;
pub const toBytes = tensor_mod.toBytes;
pub const fromBytes = tensor_mod.fromBytes;
pub const toArchive = tensor_mod.toArchive;
pub const fromArchive = tensor_mod.fromArchive;
pub const matmul = linalg.matmul;
pub const matvec = linalg.matvec;
pub const cholesky = linalg.cholesky;
pub const qr = linalg.qr;
pub const svd = linalg.svd;
pub const lstsq = linalg.lstsq;
pub const singularValues = linalg.singularValues;
pub const matrixRank = linalg.matrixRank;
pub const cond = linalg.cond;
pub const pinv = linalg.pinv;
pub const matrixNorm = linalg.matrixNorm;
pub const MatrixNormOrder = linalg.MatrixNormOrder;
pub const eigh = linalg.eigh;
pub const eigvalsh = linalg.eigvalsh;
pub const lu = linalg.lu;
pub const solveTriangular = linalg.solveTriangular;
pub const Triangle = linalg.Triangle;
pub const Diagonal = linalg.Diagonal;
pub const dataframe = dataframe_mod.dataframe;

test {
    _ = tensor_mod;
    _ = series_mod;
    _ = dataframe_mod;
    _ = linalg;
    _ = stats;
    _ = sparse;
}
