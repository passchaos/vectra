//! Column-level Arrow conversion helpers for device dataframe interop.

const std = @import("std");
const array_mod = @import("../../array.zig");
const boltha = @import("boltha");
const dataframe_array_mod = @import("../../dataframe_array.zig");
const arrow_extensions_mod = @import("extensions.zig");
const validity_mod = @import("../validity/core.zig");

pub const DataFrameInitError = std.mem.Allocator.Error || std.Io.Writer.Error || error{ LengthMismatch, ColumnNotFound, TypeMismatch, InvalidCsv, EmptyDataFrame, UnsupportedType, InvalidDevice };
pub const ArrowInteropError = DataFrameInitError || array_mod.ArrayError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;

const zeroValue = dataframe_array_mod.zeroValue;
const validityValues = validity_mod.validityValues;

fn fixedBytesWidth(comptime T: type) usize {
    if (comptime T == array_mod.BFloat16) return 2;
    if (comptime T == array_mod.Complex64) return 8;
    if (comptime T == array_mod.Complex128) return 16;
    @compileError("unsupported Vectra Arrow extension dtype: " ++ @typeName(T));
}

fn writeExtensionValue(comptime T: type, value: T, dst: []u8) void {
    if (comptime T == array_mod.BFloat16) {
        std.mem.writeInt(u16, dst[0..2], value.bits, .little);
    } else if (comptime T == array_mod.Complex64) {
        std.mem.writeInt(u32, dst[0..4], @bitCast(value.re), .little);
        std.mem.writeInt(u32, dst[4..8], @bitCast(value.im), .little);
    } else if (comptime T == array_mod.Complex128) {
        std.mem.writeInt(u64, dst[0..8], @bitCast(value.re), .little);
        std.mem.writeInt(u64, dst[8..16], @bitCast(value.im), .little);
    } else {
        @compileError("unsupported Vectra Arrow extension dtype: " ++ @typeName(T));
    }
}

fn readExtensionValue(comptime T: type, src: []const u8) T {
    if (comptime T == array_mod.BFloat16) {
        return .{ .bits = std.mem.readInt(u16, src[0..2], .little) };
    } else if (comptime T == array_mod.Complex64) {
        return .{
            .re = @bitCast(std.mem.readInt(u32, src[0..4], .little)),
            .im = @bitCast(std.mem.readInt(u32, src[4..8], .little)),
        };
    } else if (comptime T == array_mod.Complex128) {
        return .{
            .re = @bitCast(std.mem.readInt(u64, src[0..8], .little)),
            .im = @bitCast(std.mem.readInt(u64, src[8..16], .little)),
        };
    } else {
        @compileError("unsupported Vectra Arrow extension dtype: " ++ @typeName(T));
    }
}

pub fn primitiveColumnToArrow(
    comptime T: type,
    comptime tag_name: []const u8,
    column: anytype,
    allocator: std.mem.Allocator,
) ArrowInteropError!boltha.arrow.AnyArray {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const primitive = if (maybe_validity) |validity| blk: {
        const optional_values = try allocator.alloc(?T, values.len);
        defer allocator.free(optional_values);
        for (values, validity, optional_values) |value, valid, *slot| {
            slot.* = if (valid) value else null;
        }
        break :blk try boltha.arrow.PrimitiveArray(T).fromOptionalSlice(allocator, optional_values, zeroValue(T));
    } else try boltha.arrow.PrimitiveArray(T).fromSlice(allocator, values);
    return @unionInit(boltha.arrow.AnyArray, tag_name, primitive);
}

pub fn boolColumnToArrow(column: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.AnyArray {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const array_value = if (maybe_validity) |validity| blk: {
        const optional_values = try allocator.alloc(?bool, values.len);
        defer allocator.free(optional_values);
        for (values, validity, optional_values) |value, valid, *slot| {
            slot.* = if (valid) value else null;
        }
        break :blk try boltha.arrow.BooleanArray.fromOptionalSlice(allocator, optional_values);
    } else try boltha.arrow.BooleanArray.fromSlice(allocator, values);
    return .{ .boolean = array_value };
}

pub fn indexColumnToArrow(comptime T: type, column: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.AnyArray {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    if (comptime T == usize) {
        const converted = try allocator.alloc(u64, values.len);
        defer allocator.free(converted);
        for (values, converted) |value, *slot| {
            slot.* = std.math.cast(u64, value) orelse return error.TypeUnsupported;
        }
        if (maybe_validity) |validity| {
            const optional_values = try allocator.alloc(?u64, values.len);
            defer allocator.free(optional_values);
            for (converted, validity, optional_values) |value, valid, *slot| {
                slot.* = if (valid) value else null;
            }
            return .{ .uint64 = try boltha.arrow.PrimitiveArray(u64).fromOptionalSlice(allocator, optional_values, 0) };
        }
        return .{ .uint64 = try boltha.arrow.PrimitiveArray(u64).fromSlice(allocator, converted) };
    }

    if (comptime T == isize) {
        const converted = try allocator.alloc(i64, values.len);
        defer allocator.free(converted);
        for (values, converted) |value, *slot| {
            slot.* = std.math.cast(i64, value) orelse return error.TypeUnsupported;
        }
        if (maybe_validity) |validity| {
            const optional_values = try allocator.alloc(?i64, values.len);
            defer allocator.free(optional_values);
            for (converted, validity, optional_values) |value, valid, *slot| {
                slot.* = if (valid) value else null;
            }
            return .{ .int64 = try boltha.arrow.PrimitiveArray(i64).fromOptionalSlice(allocator, optional_values, 0) };
        }
        return .{ .int64 = try boltha.arrow.PrimitiveArray(i64).fromSlice(allocator, converted) };
    }

    unreachable;
}

pub fn extensionColumnToArrow(comptime T: type, column: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.AnyArray {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const byte_width = comptime fixedBytesWidth(T);
    const encoded = try allocator.alloc([]const u8, values.len);
    defer allocator.free(encoded);
    const bytes = try allocator.alloc(u8, byte_width * values.len);
    defer allocator.free(bytes);
    for (values, encoded, 0..) |value, *slot, i| {
        const dst = bytes[i * byte_width ..][0..byte_width];
        writeExtensionValue(T, value, dst);
        slot.* = dst;
    }

    const fixed = if (maybe_validity) |validity| blk: {
        const optional_values = try allocator.alloc(?[]const u8, values.len);
        defer allocator.free(optional_values);
        for (encoded, validity, optional_values) |value, valid, *slot| {
            slot.* = if (valid) value else null;
        }
        break :blk try boltha.arrow.FixedSizeBinaryArray.fromOptionalSlice(allocator, byte_width, optional_values);
    } else try boltha.arrow.FixedSizeBinaryArray.fromSlice(allocator, byte_width, encoded);
    return .{ .fixed_size_binary = fixed };
}

pub fn primitiveDeviceColumnFromArrow(
    comptime T: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    arrow_array: boltha.arrow.PrimitiveArray(T),
    device_value: array_mod.Device,
) ArrowInteropError!DeviceColumn {
    if (arrow_array.null_count == 0) return DeviceColumn.fromSlice(T, allocator, arrow_array.values, device_value);

    const validity = try allocator.alloc(bool, arrow_array.values.len);
    defer allocator.free(validity);
    for (validity, 0..) |*slot, i| slot.* = !arrow_array.isNull(i);
    return DeviceColumn.fromSliceWithValidity(T, allocator, arrow_array.values, validity, device_value);
}

pub fn boolDeviceColumnFromArrow(
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    arrow_array: boltha.arrow.BooleanArray,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceColumn {
    const values = try allocator.alloc(bool, arrow_array.len());
    defer allocator.free(values);
    const validity = try allocator.alloc(bool, arrow_array.len());
    defer allocator.free(validity);
    for (values, validity, 0..) |*value_slot, *valid_slot, i| {
        if (arrow_array.value(i)) |value| {
            value_slot.* = value;
            valid_slot.* = true;
        } else {
            value_slot.* = false;
            valid_slot.* = false;
        }
    }
    if (arrow_array.null_count == 0) return DeviceColumn.fromSlice(bool, allocator, values, device_value);
    return DeviceColumn.fromSliceWithValidity(bool, allocator, values, validity, device_value);
}

pub fn extensionDeviceColumnFromArrow(
    comptime T: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    arrow_array: boltha.arrow.FixedSizeBinaryArray,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceColumn {
    if (arrow_array.byte_width != comptime fixedBytesWidth(T)) return error.TypeUnsupported;

    const values = try allocator.alloc(T, arrow_array.len());
    defer allocator.free(values);
    const validity = try allocator.alloc(bool, arrow_array.len());
    defer allocator.free(validity);

    for (values, validity, 0..) |*value_slot, *valid_slot, i| {
        if (arrow_array.value(i)) |bytes| {
            value_slot.* = readExtensionValue(T, bytes);
            valid_slot.* = true;
        } else {
            value_slot.* = zeroValue(T);
            valid_slot.* = false;
        }
    }
    if (arrow_array.null_count == 0) return DeviceColumn.fromSlice(T, allocator, values, device_value);
    return DeviceColumn.fromSliceWithValidity(T, allocator, values, validity, device_value);
}

pub fn deviceColumnFromArrowFieldArray(
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    field: boltha.arrow.Field,
    column: boltha.arrow.AnyArray,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceColumn {
    if (arrow_extensions_mod.dtypeFromField(field)) |dtype| {
        if (column != .fixed_size_binary) return error.TypeUnsupported;
        return switch (dtype) {
            .bf16 => extensionDeviceColumnFromArrow(array_mod.BFloat16, DeviceColumn, allocator, column.fixed_size_binary, device_value),
            .c64 => extensionDeviceColumnFromArrow(array_mod.Complex64, DeviceColumn, allocator, column.fixed_size_binary, device_value),
            .c128 => extensionDeviceColumnFromArrow(array_mod.Complex128, DeviceColumn, allocator, column.fixed_size_binary, device_value),
            else => unreachable,
        };
    }
    return deviceColumnFromArrowArray(DeviceColumn, allocator, column, device_value);
}

pub fn deviceColumnFromArrowArray(
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    column: boltha.arrow.AnyArray,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceColumn {
    return switch (column) {
        .boolean => |array| boolDeviceColumnFromArrow(DeviceColumn, allocator, array, device_value),
        .int8 => |array| primitiveDeviceColumnFromArrow(i8, DeviceColumn, allocator, array, device_value),
        .uint8 => |array| primitiveDeviceColumnFromArrow(u8, DeviceColumn, allocator, array, device_value),
        .int16 => |array| primitiveDeviceColumnFromArrow(i16, DeviceColumn, allocator, array, device_value),
        .uint16 => |array| primitiveDeviceColumnFromArrow(u16, DeviceColumn, allocator, array, device_value),
        .int32 => |array| primitiveDeviceColumnFromArrow(i32, DeviceColumn, allocator, array, device_value),
        .uint32 => |array| primitiveDeviceColumnFromArrow(u32, DeviceColumn, allocator, array, device_value),
        .int64 => |array| primitiveDeviceColumnFromArrow(i64, DeviceColumn, allocator, array, device_value),
        .uint64 => |array| primitiveDeviceColumnFromArrow(u64, DeviceColumn, allocator, array, device_value),
        .float16 => |array| primitiveDeviceColumnFromArrow(f16, DeviceColumn, allocator, array, device_value),
        .float32 => |array| primitiveDeviceColumnFromArrow(f32, DeviceColumn, allocator, array, device_value),
        .float64 => |array| primitiveDeviceColumnFromArrow(f64, DeviceColumn, allocator, array, device_value),
        else => error.TypeUnsupported,
    };
}

pub fn emptyDeviceColumnFromArrowField(
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    field: boltha.arrow.Field,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceColumn {
    if (arrow_extensions_mod.dtypeFromField(field)) |dtype| {
        return switch (dtype) {
            .bf16 => DeviceColumn.fromSlice(array_mod.BFloat16, allocator, &.{}, device_value),
            .c64 => DeviceColumn.fromSlice(array_mod.Complex64, allocator, &.{}, device_value),
            .c128 => DeviceColumn.fromSlice(array_mod.Complex128, allocator, &.{}, device_value),
            else => unreachable,
        };
    }
    return emptyDeviceColumnFromArrowType(DeviceColumn, allocator, field.data_type, device_value);
}

pub fn emptyDeviceColumnFromArrowType(
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    dtype: boltha.arrow.DataType,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceColumn {
    return switch (dtype) {
        .bool => DeviceColumn.fromSlice(bool, allocator, &.{}, device_value),
        .int => |info| if (info.signed) switch (info.bit_width) {
            8 => DeviceColumn.fromSlice(i8, allocator, &.{}, device_value),
            16 => DeviceColumn.fromSlice(i16, allocator, &.{}, device_value),
            32 => DeviceColumn.fromSlice(i32, allocator, &.{}, device_value),
            64 => DeviceColumn.fromSlice(i64, allocator, &.{}, device_value),
            else => error.TypeUnsupported,
        } else switch (info.bit_width) {
            8 => DeviceColumn.fromSlice(u8, allocator, &.{}, device_value),
            16 => DeviceColumn.fromSlice(u16, allocator, &.{}, device_value),
            32 => DeviceColumn.fromSlice(u32, allocator, &.{}, device_value),
            64 => DeviceColumn.fromSlice(u64, allocator, &.{}, device_value),
            else => error.TypeUnsupported,
        },
        .floating_point => |fp| switch (fp) {
            .half => DeviceColumn.fromSlice(f16, allocator, &.{}, device_value),
            .single => DeviceColumn.fromSlice(f32, allocator, &.{}, device_value),
            .double => DeviceColumn.fromSlice(f64, allocator, &.{}, device_value),
        },
        else => error.TypeUnsupported,
    };
}
