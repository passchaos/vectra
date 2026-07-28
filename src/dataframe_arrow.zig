const std = @import("std");
const array_mod = @import("array.zig");
const boltha = @import("boltha");
const numeric_mod = @import("dataframe_numeric.zig");
const validity_mod = @import("dataframe_validity.zig");
const dataframe_array_mod = @import("dataframe_array.zig");

pub const DataFrameInitError = std.mem.Allocator.Error || std.Io.Writer.Error || error{ LengthMismatch, ColumnNotFound, TypeMismatch, InvalidCsv, EmptyDataFrame, UnsupportedType, InvalidDevice };
pub const ArrowInteropError = DataFrameInitError || array_mod.ArrayError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
pub const ParquetInteropError = ArrowInteropError || boltha.parquet.SimpleError;
const optionalCast = numeric_mod.optionalCast;
const validityValues = validity_mod.validityValues;
const zeroValue = dataframe_array_mod.zeroValue;

pub fn deviceDTypeToArrowDataType(dtype: array_mod.DType) ArrowInteropError!boltha.arrow.DataType {
    return switch (dtype) {
        .bool => .bool,
        .i8 => .{ .int = .{ .bit_width = 8, .signed = true } },
        .i16 => .{ .int = .{ .bit_width = 16, .signed = true } },
        .i32 => .{ .int = .{ .bit_width = 32, .signed = true } },
        .i64, .isize => .{ .int = .{ .bit_width = 64, .signed = true } },
        .u8 => .{ .int = .{ .bit_width = 8, .signed = false } },
        .u16 => .{ .int = .{ .bit_width = 16, .signed = false } },
        .u32 => .{ .int = .{ .bit_width = 32, .signed = false } },
        .u64, .usize => .{ .int = .{ .bit_width = 64, .signed = false } },
        .f16 => .{ .floating_point = .half },
        .f32 => .{ .floating_point = .single },
        .f64 => .{ .floating_point = .double },
        // Boltha already models Arrow primitive/fixed/nested types. Vectra's
        // BFloat16 and complex values need explicit logical-extension metadata
        // before they can be exported without losing semantics, so keep them
        // rejected rather than pretending they are plain fixed-size binaries.
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn readBolthaTableWithBoolRangePruning(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    column_name: []const u8,
    range: anytype,
) ParquetInteropError!boltha.arrow.Table {
    if (range.min) |min_value| {
        if (range.max) |max_value| {
            if (min_value == max_value) {
                return boltha.parquet.readTableWithBooleanPruning(allocator, bytes, column_name, .{ .value = min_value });
            }
            if (!min_value and max_value) return boltha.parquet.readTable(allocator, bytes);
            return emptyBolthaTableForParquetBytes(allocator, bytes);
        }
        return if (min_value)
            boltha.parquet.readTableWithBooleanPruning(allocator, bytes, column_name, .{ .value = true })
        else
            boltha.parquet.readTable(allocator, bytes);
    }
    if (range.max) |max_value| {
        return if (!max_value)
            boltha.parquet.readTableWithBooleanPruning(allocator, bytes, column_name, .{ .value = false })
        else
            boltha.parquet.readTable(allocator, bytes);
    }
    return boltha.parquet.readTable(allocator, bytes);
}

pub fn readBolthaTableWithRangePruning(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    column_name: []const u8,
    predicate: anytype,
) ParquetInteropError!boltha.arrow.Table {
    return switch (predicate) {
        .bool => |range| readBolthaTableWithBoolRangePruning(allocator, bytes, column_name, range),
        .i8 => |range| boltha.parquet.readTableWithInt8Pruning(allocator, bytes, column_name, .{ .min = optionalCast(i32, range.min), .max = optionalCast(i32, range.max) }),
        .i16 => |range| boltha.parquet.readTableWithInt16Pruning(allocator, bytes, column_name, .{ .min = optionalCast(i32, range.min), .max = optionalCast(i32, range.max) }),
        .i32 => |range| boltha.parquet.readTableWithInt32Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .i64 => |range| boltha.parquet.readTableWithInt64Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .isize => |range| boltha.parquet.readTableWithInt64Pruning(allocator, bytes, column_name, .{ .min = optionalCast(i64, range.min), .max = optionalCast(i64, range.max) }),
        .u8 => |range| boltha.parquet.readTableWithUInt8Pruning(allocator, bytes, column_name, .{ .min = optionalCast(u32, range.min), .max = optionalCast(u32, range.max) }),
        .u16 => |range| boltha.parquet.readTableWithUInt16Pruning(allocator, bytes, column_name, .{ .min = optionalCast(u32, range.min), .max = optionalCast(u32, range.max) }),
        .u32 => |range| boltha.parquet.readTableWithUInt32Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .u64 => |range| boltha.parquet.readTableWithUInt64Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .usize => |range| boltha.parquet.readTableWithUInt64Pruning(allocator, bytes, column_name, .{ .min = optionalCast(u64, range.min), .max = optionalCast(u64, range.max) }),
        .f16 => |range| boltha.parquet.readTableWithFloat16Pruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .f32 => |range| boltha.parquet.readTableWithFloatPruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .f64 => |range| boltha.parquet.readTableWithDoublePruning(allocator, bytes, column_name, .{ .min = range.min, .max = range.max }),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn emptyBolthaTableForParquetBytes(allocator: std.mem.Allocator, bytes: []const u8) ParquetInteropError!boltha.arrow.Table {
    var schema = try boltha.parquet.readSchema(allocator, bytes);
    errdefer schema.deinit(allocator);
    const batches = try allocator.alloc(boltha.arrow.RecordBatch, 0);
    errdefer allocator.free(batches);
    return boltha.arrow.Table.initOwned(schema, batches);
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

pub fn emptyFromArrowSchema(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    schema: boltha.arrow.Schema,
    rows: usize,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceDataFrame {
    if (rows != 0) return error.TypeUnsupported;
    var defs = try allocator.alloc(DeviceColumnDef, schema.fields.len);
    defer allocator.free(defs);
    var initialized: usize = 0;
    defer {
        for (defs[0..initialized]) |*def| def.data.deinit();
    }
    for (schema.fields, 0..) |field, i| {
        defs[i] = .{
            .name = field.name,
            .data = try emptyDeviceColumnFromArrowType(DeviceColumn, allocator, field.data_type, device_value),
        };
        initialized += 1;
    }
    return DeviceDataFrame.init(allocator, defs);
}

pub fn emptyFromArrowSchemaProjection(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    schema: boltha.arrow.Schema,
    rows: usize,
    wanted_names: []const []const u8,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceDataFrame {
    if (rows != 0) return error.TypeUnsupported;
    var defs = try allocator.alloc(DeviceColumnDef, wanted_names.len);
    defer allocator.free(defs);
    var initialized: usize = 0;
    defer {
        for (defs[0..initialized]) |*def| def.data.deinit();
    }
    for (wanted_names, 0..) |name, i| {
        const column_index = schema.fieldIndexByName(name) orelse return error.ColumnNotFound;
        const field = schema.fields[column_index];
        defs[i] = .{
            .name = field.name,
            .data = try emptyDeviceColumnFromArrowType(DeviceColumn, allocator, field.data_type, device_value),
        };
        initialized += 1;
    }
    return DeviceDataFrame.init(allocator, defs);
}
