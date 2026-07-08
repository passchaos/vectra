//! Vectra is an experimental Zig-native data processing and numerical computing
//! toolkit inspired by NumPy/CuPy/SciPy/Pandas/Polars, with PyTorch-style tensor
//! method names where that makes code more fluent.
//!
//! The current implementation provides a compact, dependency-free CPU core:
//! typed tensors, broadcasting arithmetic, reductions, linalg/stat helpers,
//! typed Series, heterogeneous DataFrame operations, and CSV IO. CUDA/GPU is
//! represented in the API surface and returns `error.InvalidDevice` until a real
//! backend is wired in.

pub const tensor_mod = @import("tensor.zig");
pub const series_mod = @import("series.zig");
pub const dataframe_mod = @import("dataframe.zig");
pub const linalg = @import("linalg.zig");
pub const stats = @import("stats.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Device = tensor_mod.Device;
pub const DType = tensor_mod.DType;
pub const Slice = tensor_mod.Slice;
pub const TensorError = tensor_mod.TensorError;

pub const Series = series_mod.Series;
pub const DataFrame = dataframe_mod.DataFrame;
pub const Column = dataframe_mod.Column;
pub const ColumnDef = dataframe_mod.ColumnDef;
pub const DataError = dataframe_mod.DataError;

pub const tensor = tensor_mod.tensor;
pub const zeros = tensor_mod.zeros;
pub const ones = tensor_mod.ones;
pub const full = tensor_mod.full;
pub const empty = tensor_mod.empty;
pub const arange = tensor_mod.arange;
pub const linspace = tensor_mod.linspace;
pub const rand = tensor_mod.rand;
pub const randn = tensor_mod.randn;
pub const randint = tensor_mod.randint;
pub const eye = tensor_mod.eye;
pub const cat = tensor_mod.cat;
pub const stack = tensor_mod.stack;
pub const outer = tensor_mod.outer;
pub const where = tensor_mod.where;
pub const matmul = linalg.matmul;
pub const matvec = linalg.matvec;
pub const cholesky = linalg.cholesky;
pub const qr = linalg.qr;
pub const dataframe = dataframe_mod.dataframe;

test {
    _ = tensor_mod;
    _ = series_mod;
    _ = dataframe_mod;
    _ = linalg;
    _ = stats;
}
