//! Arrow export helpers for tagged device columns.

const std = @import("std");
const array_mod = @import("../../array.zig");
const boltha = @import("boltha");
const dataframe_arrow_mod = @import("../arrow.zig");
const series_mod = @import("../../series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const ArrowInteropError = DeviceDataError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
const deviceDTypeToArrowDataType = dataframe_arrow_mod.deviceDTypeToArrowDataType;
const primitiveColumnToArrow = dataframe_arrow_mod.primitiveColumnToArrow;
const boolColumnToArrow = dataframe_arrow_mod.boolColumnToArrow;
const indexColumnToArrow = dataframe_arrow_mod.indexColumnToArrow;
const extensionColumnToArrow = dataframe_arrow_mod.extensionColumnToArrow;

fn columnValue(self: anytype) switch (@typeInfo(@TypeOf(self))) {
    .pointer => |ptr| ptr.child,
    else => @TypeOf(self),
} {
    return switch (@typeInfo(@TypeOf(self))) {
        .pointer => self.*,
        else => self,
    };
}

pub fn arrowDataType(self: anytype) ArrowInteropError!boltha.arrow.DataType {
    return deviceDTypeToArrowDataType(columnValue(self).dtype());
}

pub fn toArrowArray(self: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.AnyArray {
    return switch (columnValue(self)) {
        .bool => |typed| try boolColumnToArrow(typed, allocator),
        .i8 => |typed| try primitiveColumnToArrow(i8, "int8", typed, allocator),
        .i16 => |typed| try primitiveColumnToArrow(i16, "int16", typed, allocator),
        .i32 => |typed| try primitiveColumnToArrow(i32, "int32", typed, allocator),
        .i64 => |typed| try primitiveColumnToArrow(i64, "int64", typed, allocator),
        .u8 => |typed| try primitiveColumnToArrow(u8, "uint8", typed, allocator),
        .u16 => |typed| try primitiveColumnToArrow(u16, "uint16", typed, allocator),
        .u32 => |typed| try primitiveColumnToArrow(u32, "uint32", typed, allocator),
        .u64 => |typed| try primitiveColumnToArrow(u64, "uint64", typed, allocator),
        .f16 => |typed| try primitiveColumnToArrow(f16, "float16", typed, allocator),
        .f32 => |typed| try primitiveColumnToArrow(f32, "float32", typed, allocator),
        .f64 => |typed| try primitiveColumnToArrow(f64, "float64", typed, allocator),
        .usize => |typed| try indexColumnToArrow(usize, typed, allocator),
        .isize => |typed| try indexColumnToArrow(isize, typed, allocator),
        .bf16 => |typed| try extensionColumnToArrow(array_mod.BFloat16, typed, allocator),
        .c64 => |typed| try extensionColumnToArrow(array_mod.Complex64, typed, allocator),
        .c128 => |typed| try extensionColumnToArrow(array_mod.Complex128, typed, allocator),
    };
}
