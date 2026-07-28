const std = @import("std");
const array_mod = @import("array.zig");
const boltha = @import("boltha");
const numeric_mod = @import("dataframe_numeric.zig");
const validity_mod = @import("dataframe_validity.zig");
const dataframe_array_mod = @import("dataframe_array.zig");

pub const ArrowInteropError = array_mod.ArrayError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
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
