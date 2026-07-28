const std = @import("std");
const array_mod = @import("array.zig");
const boltha = @import("boltha");

pub const ArrowInteropError = array_mod.ArrayError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
pub const ParquetInteropError = ArrowInteropError || boltha.parquet.SimpleError;

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

pub fn emptyBolthaTableForParquetBytes(allocator: std.mem.Allocator, bytes: []const u8) ParquetInteropError!boltha.arrow.Table {
    var schema = try boltha.parquet.readSchema(allocator, bytes);
    errdefer schema.deinit(allocator);
    const batches = try allocator.alloc(boltha.arrow.RecordBatch, 0);
    errdefer allocator.free(batches);
    return boltha.arrow.Table.initOwned(schema, batches);
}
