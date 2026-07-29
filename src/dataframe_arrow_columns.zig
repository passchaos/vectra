//! Column-level Arrow conversion helpers for device dataframe interop.

const std = @import("std");
const array_mod = @import("array.zig");
const boltha = @import("boltha");
const dataframe_array_mod = @import("dataframe_array.zig");
const validity_mod = @import("dataframe_validity_core.zig");

pub const DataFrameInitError = std.mem.Allocator.Error || std.Io.Writer.Error || error{ LengthMismatch, ColumnNotFound, TypeMismatch, InvalidCsv, EmptyDataFrame, UnsupportedType, InvalidDevice };
pub const ArrowInteropError = DataFrameInitError || array_mod.ArrayError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;

const zeroValue = dataframe_array_mod.zeroValue;
const validityValues = validity_mod.validityValues;

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
