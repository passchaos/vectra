const std = @import("std");
const array_mod = @import("array.zig");
const column_arrow_mod = @import("dataframe_device_column_arrow.zig");
const column_ops_mod = @import("dataframe_device_column_ops.zig");
const column_sort_mod = @import("dataframe_device_column_sort.zig");
const dataframe_typed_column_mod = @import("dataframe_device_typed_column.zig");
const dataframe_view_mod = @import("dataframe_view.zig");

const DeviceDType = array_mod.DType;
const DeviceColumnView = dataframe_view_mod.DeviceColumnView;

pub const DeviceTypedColumn = dataframe_typed_column_mod.DeviceTypedColumn;
pub const DeviceColumn = union(DeviceDType) {
    f32: DeviceTypedColumn(f32),
    f64: DeviceTypedColumn(f64),
    i8: DeviceTypedColumn(i8),
    i16: DeviceTypedColumn(i16),
    i32: DeviceTypedColumn(i32),
    i64: DeviceTypedColumn(i64),
    u8: DeviceTypedColumn(u8),
    u16: DeviceTypedColumn(u16),
    u32: DeviceTypedColumn(u32),
    u64: DeviceTypedColumn(u64),
    usize: DeviceTypedColumn(usize),
    bool: DeviceTypedColumn(bool),
    bf16: DeviceTypedColumn(array_mod.BFloat16),
    f16: DeviceTypedColumn(f16),
    c64: DeviceTypedColumn(array_mod.Complex64),
    c128: DeviceTypedColumn(array_mod.Complex128),
    isize: DeviceTypedColumn(isize),

    pub fn fromSlice(comptime T: type, allocator: std.mem.Allocator, values: []const T, device_value: array_mod.Device) array_mod.ArrayError!DeviceColumn {
        const tag = comptime DeviceDType.of(T);
        const typed = try DeviceTypedColumn(T).fromSlice(allocator, values, device_value);
        return @unionInit(DeviceColumn, @tagName(tag), typed);
    }

    pub fn fromSliceWithValidity(
        comptime T: type,
        allocator: std.mem.Allocator,
        values: []const T,
        validity_values: []const bool,
        device_value: array_mod.Device,
    ) array_mod.ArrayError!DeviceColumn {
        const tag = comptime DeviceDType.of(T);
        const typed = try DeviceTypedColumn(T).fromSliceWithValidity(allocator, values, validity_values, device_value);
        return @unionInit(DeviceColumn, @tagName(tag), typed);
    }

    pub fn deinit(self: *DeviceColumn) void {
        switch (self.*) {
            inline else => |*typed| typed.deinit(),
        }
        self.* = undefined;
    }

    pub fn len(self: DeviceColumn) usize {
        return switch (self) {
            inline else => |typed| typed.len(),
        };
    }

    pub fn dtype(self: DeviceColumn) DeviceDType {
        return std.meta.activeTag(self);
    }

    pub fn device(self: DeviceColumn) array_mod.Device {
        return switch (self) {
            inline else => |typed| typed.device(),
        };
    }

    pub fn nullable(self: DeviceColumn) bool {
        return switch (self) {
            inline else => |typed| typed.nullable(),
        };
    }

    pub fn hasNulls(self: DeviceColumn) bool {
        return switch (self) {
            inline else => |typed| typed.hasNulls(),
        };
    }

    pub fn nullCount(self: DeviceColumn) usize {
        return switch (self) {
            inline else => |typed| typed.null_count,
        };
    }

    pub fn dataNbytes(self: DeviceColumn) usize {
        return switch (self) {
            inline else => |typed| typed.dataNbytes(),
        };
    }

    pub fn view(self: DeviceColumn) DeviceColumnView {
        return switch (self) {
            inline else => |typed| typed.view(),
        };
    }

    pub fn clone(self: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.clone()),
        };
    }

    pub fn cast(self: DeviceColumn, comptime T: type) array_mod.ArrayError!DeviceColumn {
        const tag = comptime DeviceDType.of(T);
        const target_is_complex = comptime tag.isComplex();
        return @unionInit(DeviceColumn, @tagName(tag), switch (self) {
            inline else => |typed, source_tag| blk: {
                // Array.astype intentionally rejects complex-to-real casts at
                // compile time because silently discarding the imaginary part is
                // often a data-quality bug. Keep dataframe dynamic dtype casts
                // on the same policy, but return a runtime error before
                // instantiating the unsupported typed cast branch.
                if (comptime source_tag.isComplex() and !target_is_complex) return error.TypeUnsupported;
                break :blk try typed.cast(T);
            },
        });
    }

    pub fn castToDType(self: DeviceColumn, dtype_value: DeviceDType) array_mod.ArrayError!DeviceColumn {
        return switch (dtype_value) {
            .i8 => self.cast(i8),
            .i16 => self.cast(i16),
            .i32 => self.cast(i32),
            .i64 => self.cast(i64),
            .u8 => self.cast(u8),
            .u16 => self.cast(u16),
            .u32 => self.cast(u32),
            .u64 => self.cast(u64),
            .usize => self.cast(usize),
            .isize => self.cast(isize),
            .f16 => self.cast(f16),
            .f32 => self.cast(f32),
            .f64 => self.cast(f64),
            .bool => self.cast(bool),
            .bf16 => self.cast(array_mod.BFloat16),
            .c64 => self.cast(array_mod.Complex64),
            .c128 => self.cast(array_mod.Complex128),
        };
    }

    pub fn to(self: DeviceColumn, device_value: array_mod.Device) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.to(device_value)),
        };
    }

    pub fn cpu(self: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.to(.cpu);
    }

    pub fn cuda(self: DeviceColumn, index: usize) array_mod.ArrayError!DeviceColumn {
        return self.to(array_mod.Device.cuda(index));
    }

    pub fn mps(self: DeviceColumn, index: usize) array_mod.ArrayError!DeviceColumn {
        return self.to(array_mod.Device.mps(index));
    }

    pub fn sliceRows(self: DeviceColumn, start: usize, stop: usize) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.sliceRows(start, stop)),
        };
    }

    pub fn take(self: DeviceColumn, row_indices: []const usize) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.take(row_indices)),
        };
    }

    pub fn takeOptional(self: DeviceColumn, row_indices: []const ?usize) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.takeOptional(row_indices)),
        };
    }

    pub fn filter(self: DeviceColumn, mask: []const bool) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.filter(mask)),
        };
    }

    pub fn fillNull(self: DeviceColumn, comptime T: type, value: T) array_mod.ArrayError!DeviceColumn {
        const tag = comptime DeviceDType.of(T);
        if (self.dtype() != tag) return error.TypeUnsupported;
        return @unionInit(DeviceColumn, @tagName(tag), try @field(self, @tagName(tag)).fillNull(value));
    }

    pub fn fillNullWithScalar(self: DeviceColumn, scalar: @import("dataframe_options.zig").DeviceScalar) array_mod.ArrayError!DeviceColumn {
        return switch (scalar) {
            inline else => |value| self.fillNull(@TypeOf(value), value),
        };
    }

    pub const argsort = column_sort_mod.argsort;

    pub const abs = column_ops_mod.abs;
    pub const neg = column_ops_mod.neg;
    pub const negative = column_ops_mod.neg;
    pub const square = column_ops_mod.square;
    pub const reciprocal = column_ops_mod.reciprocal;
    pub const sign = column_ops_mod.sign;
    pub const sqrt = column_ops_mod.sqrt;
    pub const rsqrt = column_ops_mod.rsqrt;
    pub const cbrt = column_ops_mod.cbrt;
    pub const floor = column_ops_mod.floor;
    pub const ceil = column_ops_mod.ceil;
    pub const round = column_ops_mod.round;
    pub const trunc = column_ops_mod.trunc;
    pub const deg2rad = column_ops_mod.deg2rad;
    pub const rad2deg = column_ops_mod.rad2deg;
    pub const expit = column_ops_mod.expit;
    pub const logit = column_ops_mod.logit;
    pub const softplus = column_ops_mod.softplus;
    pub const logsigmoid = column_ops_mod.logsigmoid;
    pub const relu = column_ops_mod.relu;
    pub const leakyRelu = column_ops_mod.leakyRelu;
    pub const leakyReluWithDeviceScalar = column_ops_mod.leakyReluWithDeviceScalar;
    pub const relu6 = column_ops_mod.relu6;
    pub const powScalar = column_ops_mod.powScalar;
    pub const powWithDeviceScalar = column_ops_mod.powWithDeviceScalar;
    pub const floorDivScalar = column_ops_mod.floorDivScalar;
    pub const floorDivWithDeviceScalar = column_ops_mod.floorDivWithDeviceScalar;
    pub const modScalar = column_ops_mod.modScalar;
    pub const modWithDeviceScalar = column_ops_mod.modWithDeviceScalar;
    pub const remainderScalar = column_ops_mod.remainderScalar;
    pub const remainderWithDeviceScalar = column_ops_mod.remainderWithDeviceScalar;
    pub const logAddExpScalar = column_ops_mod.logAddExpScalar;
    pub const logAddExpWithDeviceScalar = column_ops_mod.logAddExpWithDeviceScalar;
    pub const logAddExp2Scalar = column_ops_mod.logAddExp2Scalar;
    pub const logAddExp2WithDeviceScalar = column_ops_mod.logAddExp2WithDeviceScalar;
    pub const xlogyScalar = column_ops_mod.xlogyScalar;
    pub const xlogyWithDeviceScalar = column_ops_mod.xlogyWithDeviceScalar;
    pub const fmaxScalar = column_ops_mod.fmaxScalar;
    pub const fmaxWithDeviceScalar = column_ops_mod.fmaxWithDeviceScalar;
    pub const fminScalar = column_ops_mod.fminScalar;
    pub const fminWithDeviceScalar = column_ops_mod.fminWithDeviceScalar;
    pub const hypotScalar = column_ops_mod.hypotScalar;
    pub const hypotWithDeviceScalar = column_ops_mod.hypotWithDeviceScalar;
    pub const atan2Scalar = column_ops_mod.atan2Scalar;
    pub const atan2WithDeviceScalar = column_ops_mod.atan2WithDeviceScalar;
    pub const nextAfterScalar = column_ops_mod.nextAfterScalar;
    pub const nextAfterWithDeviceScalar = column_ops_mod.nextAfterWithDeviceScalar;
    pub const copysignScalar = column_ops_mod.copysignScalar;
    pub const copysignWithDeviceScalar = column_ops_mod.copysignWithDeviceScalar;
    pub const heavisideScalar = column_ops_mod.heavisideScalar;
    pub const heavisideWithDeviceScalar = column_ops_mod.heavisideWithDeviceScalar;
    pub const ldexpScalar = column_ops_mod.ldexpScalar;
    pub const threshold = column_ops_mod.threshold;
    pub const thresholdWithDeviceScalars = column_ops_mod.thresholdWithDeviceScalars;
    pub const hardtanh = column_ops_mod.hardtanh;
    pub const hardtanhWithDeviceScalars = column_ops_mod.hardtanhWithDeviceScalars;
    pub const maximumScalar = column_ops_mod.maximumScalar;
    pub const maximumWithDeviceScalar = column_ops_mod.maximumWithDeviceScalar;
    pub const minimumScalar = column_ops_mod.minimumScalar;
    pub const minimumWithDeviceScalar = column_ops_mod.minimumWithDeviceScalar;
    pub const clipMin = column_ops_mod.clipMin;
    pub const clipMinWithDeviceScalar = column_ops_mod.clipMinWithDeviceScalar;
    pub const clipMax = column_ops_mod.clipMax;
    pub const clipMaxWithDeviceScalar = column_ops_mod.clipMaxWithDeviceScalar;
    pub const hardshrink = column_ops_mod.hardshrink;
    pub const hardshrinkWithDeviceScalar = column_ops_mod.hardshrinkWithDeviceScalar;
    pub const softshrink = column_ops_mod.softshrink;
    pub const softshrinkWithDeviceScalar = column_ops_mod.softshrinkWithDeviceScalar;
    pub const tanhshrink = column_ops_mod.tanhshrink;
    pub const elu = column_ops_mod.elu;
    pub const eluWithDeviceScalar = column_ops_mod.eluWithDeviceScalar;
    pub const celu = column_ops_mod.celu;
    pub const celuWithDeviceScalar = column_ops_mod.celuWithDeviceScalar;
    pub const softsign = column_ops_mod.softsign;
    pub const hardsigmoid = column_ops_mod.hardsigmoid;
    pub const hardswish = column_ops_mod.hardswish;
    pub const silu = column_ops_mod.silu;
    pub const swish = column_ops_mod.swish;
    pub const mish = column_ops_mod.mish;
    pub const gelu = column_ops_mod.gelu;
    pub const selu = column_ops_mod.selu;
    pub const exp = column_ops_mod.exp;
    pub const exp2 = column_ops_mod.exp2;
    pub const expm1 = column_ops_mod.expm1;
    pub const sin = column_ops_mod.sin;
    pub const cos = column_ops_mod.cos;
    pub const tan = column_ops_mod.tan;
    pub const asin = column_ops_mod.asin;
    pub const acos = column_ops_mod.acos;
    pub const atan = column_ops_mod.atan;
    pub const sinh = column_ops_mod.sinh;
    pub const cosh = column_ops_mod.cosh;
    pub const tanh = column_ops_mod.tanh;
    pub const asinh = column_ops_mod.asinh;
    pub const acosh = column_ops_mod.acosh;
    pub const atanh = column_ops_mod.atanh;
    pub const log = column_ops_mod.log;
    pub const log1p = column_ops_mod.log1p;
    pub const lgamma = column_ops_mod.lgamma;
    pub const sinc = column_ops_mod.sinc;
    pub const log2 = column_ops_mod.log2;
    pub const log10 = column_ops_mod.log10;
    pub const binary = column_ops_mod.binary;
    pub const add = column_ops_mod.add;
    pub const sub = column_ops_mod.sub;
    pub const mul = column_ops_mod.mul;
    pub const div = column_ops_mod.div;
    pub const binaryScalar = column_ops_mod.binaryScalar;
    pub const addScalar = column_ops_mod.addScalar;
    pub const subScalar = column_ops_mod.subScalar;
    pub const mulScalar = column_ops_mod.mulScalar;
    pub const divScalar = column_ops_mod.divScalar;
    pub const lerpScalar = column_ops_mod.lerpScalar;
    pub const lerpWithDeviceScalar = column_ops_mod.lerpWithDeviceScalar;
    pub const addcmulScalar = column_ops_mod.addcmulScalar;
    pub const addcmulWithDeviceScalar = column_ops_mod.addcmulWithDeviceScalar;
    pub const addcdivScalar = column_ops_mod.addcdivScalar;
    pub const addcdivWithDeviceScalar = column_ops_mod.addcdivWithDeviceScalar;
    pub const clipArray = column_ops_mod.clipArray;
    pub const whereColumn = column_ops_mod.whereColumn;
    pub const whereScalar = column_ops_mod.whereScalar;
    pub const whereWithDeviceScalar = column_ops_mod.whereWithDeviceScalar;
    pub const isinColumn = column_ops_mod.isinColumn;
    pub const maskedPutScalar = column_ops_mod.maskedPutScalar;
    pub const maskedPutWithDeviceScalar = column_ops_mod.maskedPutWithDeviceScalar;
    pub const putFlat = column_ops_mod.putFlat;
    pub const putFlatScalar = column_ops_mod.putFlatScalar;
    pub const putFlatWithDeviceScalar = column_ops_mod.putFlatWithDeviceScalar;
    pub const putFlatScalarMode = column_ops_mod.putFlatScalarMode;
    pub const putFlatModeWithDeviceScalar = column_ops_mod.putFlatModeWithDeviceScalar;
    pub const putFlatScalarSigned = column_ops_mod.putFlatScalarSigned;
    pub const putFlatSignedWithDeviceScalar = column_ops_mod.putFlatSignedWithDeviceScalar;
    pub const compare = column_ops_mod.compare;
    pub const equal = column_ops_mod.equal;
    pub const notEqual = column_ops_mod.notEqual;
    pub const greater = column_ops_mod.greater;
    pub const greaterEqual = column_ops_mod.greaterEqual;
    pub const less = column_ops_mod.less;
    pub const lessEqual = column_ops_mod.lessEqual;
    pub const compareScalar = column_ops_mod.compareScalar;
    pub const equalScalar = column_ops_mod.equalScalar;
    pub const notEqualScalar = column_ops_mod.notEqualScalar;
    pub const greaterScalar = column_ops_mod.greaterScalar;
    pub const greaterEqualScalar = column_ops_mod.greaterEqualScalar;
    pub const lessScalar = column_ops_mod.lessScalar;
    pub const lessEqualScalar = column_ops_mod.lessEqualScalar;
    pub const iscloseScalar = column_ops_mod.iscloseScalar;
    pub const iscloseWithDeviceScalars = column_ops_mod.iscloseWithDeviceScalars;
    pub const allcloseScalar = column_ops_mod.allcloseScalar;
    pub const allcloseWithDeviceScalars = column_ops_mod.allcloseWithDeviceScalars;
    pub const countNonzero = column_ops_mod.countNonzero;
    pub const sum = column_ops_mod.sum;
    pub const mean = column_ops_mod.mean;
    pub const any = column_ops_mod.any;
    pub const all = column_ops_mod.all;
    pub const countTrue = column_ops_mod.countTrue;
    pub const countFalse = column_ops_mod.countFalse;
    pub const logical = column_ops_mod.logical;
    pub const logicalAnd = column_ops_mod.logicalAnd;
    pub const logicalOr = column_ops_mod.logicalOr;
    pub const logicalXor = column_ops_mod.logicalXor;
    pub const logicalAndScalar = column_ops_mod.logicalAndScalar;
    pub const logicalOrScalar = column_ops_mod.logicalOrScalar;
    pub const logicalXorScalar = column_ops_mod.logicalXorScalar;

    pub const arrowDataType = column_arrow_mod.arrowDataType;
    pub const toArrowArray = column_arrow_mod.toArrowArray;
};

pub const DeviceColumnDef = struct {
    name: []const u8,
    data: DeviceColumn,
};

pub const argsortTypedColumn = column_sort_mod.argsortTypedColumn;
