const std = @import("std");
const array_mod = @import("../array.zig");
const boltha = @import("boltha");
const numeric_mod = @import("../dataframe_numeric.zig");
const dataframe_array_mod = @import("../dataframe_array.zig");
const schema_mod = @import("../dataframe_schema.zig");
const arrow_columns_mod = @import("arrow/columns.zig");
const arrow_extensions_mod = @import("arrow/extensions.zig");
const arrow_import_mod = @import("arrow/import.zig");

pub const DataFrameInitError = std.mem.Allocator.Error || std.Io.Writer.Error || error{ LengthMismatch, ColumnNotFound, TypeMismatch, InvalidCsv, EmptyDataFrame, UnsupportedType, InvalidDevice };
pub const ArrowInteropError = DataFrameInitError || array_mod.ArrayError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
pub const ParquetInteropError = ArrowInteropError || boltha.parquet.SimpleError;
const optionalCast = numeric_mod.optionalCast;
const DeviceColumnSchema = schema_mod.DeviceColumnSchema;

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
        // Arrow has no built-in BFloat16 or complex scalar type. Vectra stores
        // these as fixed-size binary arrays and attaches extension metadata at
        // the Field level in `toArrowField`.
        .bf16 => .{ .fixed_size_binary = 2 },
        .c64 => .{ .fixed_size_binary = 8 },
        .c128 => .{ .fixed_size_binary = 16 },
    };
}

pub fn deviceDTypeToArrowField(allocator: std.mem.Allocator, name: []const u8, dtype: array_mod.DType, nullable: bool) ArrowInteropError!boltha.arrow.Field {
    const data_type = try deviceDTypeToArrowDataType(dtype);
    if (arrow_extensions_mod.forDType(dtype)) |spec| {
        var metadata: [2]boltha.arrow.KeyValue = undefined;
        metadata[0] = try boltha.arrow.KeyValue.init(allocator, boltha.arrow.extensionTypeNameKey, spec.name);
        metadata[1] = boltha.arrow.KeyValue.init(allocator, boltha.arrow.extensionTypeMetadataKey, spec.metadata) catch |err| {
            metadata[0].deinit(allocator);
            return err;
        };
        defer {
            metadata[0].deinit(allocator);
            metadata[1].deinit(allocator);
        }
        return boltha.arrow.Field.initWithMetadata(allocator, name, data_type, nullable, &metadata);
    }
    return boltha.arrow.Field.init(allocator, name, data_type, nullable);
}

pub fn deviceColumnSchemaToArrowField(allocator: std.mem.Allocator, schema: DeviceColumnSchema) ArrowInteropError!boltha.arrow.Field {
    return deviceDTypeToArrowField(allocator, schema.name, schema.dtype, schema.nullableColumn());
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

pub const primitiveColumnToArrow = arrow_columns_mod.primitiveColumnToArrow;
pub const boolColumnToArrow = arrow_columns_mod.boolColumnToArrow;
pub const indexColumnToArrow = arrow_columns_mod.indexColumnToArrow;
pub const extensionColumnToArrow = arrow_columns_mod.extensionColumnToArrow;
pub const primitiveDeviceColumnFromArrow = arrow_columns_mod.primitiveDeviceColumnFromArrow;
pub const boolDeviceColumnFromArrow = arrow_columns_mod.boolDeviceColumnFromArrow;
pub const deviceColumnFromArrowArray = arrow_columns_mod.deviceColumnFromArrowArray;
pub const deviceColumnFromArrowFieldArray = arrow_columns_mod.deviceColumnFromArrowFieldArray;
pub const emptyDeviceColumnFromArrowType = arrow_columns_mod.emptyDeviceColumnFromArrowType;
pub const emptyDeviceColumnFromArrowField = arrow_columns_mod.emptyDeviceColumnFromArrowField;

pub const emptyFromArrowSchema = arrow_import_mod.emptyFromArrowSchema;
pub const emptyFromArrowSchemaProjection = arrow_import_mod.emptyFromArrowSchemaProjection;
pub const fromArrowTable = arrow_import_mod.fromArrowTable;
pub const fromArrowTableProjection = arrow_import_mod.fromArrowTableProjection;
pub const fromArrowRecordBatch = arrow_import_mod.fromArrowRecordBatch;
pub const fromArrowRecordBatchProjection = arrow_import_mod.fromArrowRecordBatchProjection;

/// Export a Boltha/Arrow schema for a fixed-width device dataframe.
///
/// The dataframe may own CPU, CUDA, or MPS columns; schema export only inspects
/// column metadata, while array materialization remains in `toArrowRecordBatch`.
pub fn toArrowFields(frame: anytype, allocator: std.mem.Allocator) ArrowInteropError![]boltha.arrow.Field {
    var fields = try allocator.alloc(boltha.arrow.Field, frame.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (fields[0..initialized]) |*field| field.deinit(allocator);
        allocator.free(fields);
    }

    for (frame.names, frame.columns, 0..) |name, col, i| {
        fields[i] = try deviceColumnSchemaToArrowField(allocator, col.schema(name));
        initialized += 1;
    }
    return fields;
}

pub fn toArrowSchema(frame: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.Schema {
    const fields = try toArrowFields(frame, allocator);
    defer {
        for (fields) |*field| field.deinit(allocator);
        allocator.free(fields);
    }
    return boltha.arrow.Schema.init(allocator, fields);
}

/// Materialize a device dataframe as a single Boltha/Arrow record batch.
///
/// This is the host-side interchange boundary used by Arrow IPC/Parquet and
/// external consumers. Device-resident columns are downloaded through the column
/// conversion APIs so one implementation serves CPU, CUDA, and MPS storage.
pub fn toArrowRecordBatch(frame: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.RecordBatch {
    var schema = try toArrowSchema(frame, allocator);
    errdefer schema.deinit(allocator);

    const columns = try allocator.alloc(boltha.arrow.AnyArray, frame.columns.len);
    errdefer allocator.free(columns);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*column_value| column_value.deinit(allocator);
    }

    for (frame.columns, columns) |col, *slot| {
        slot.* = try col.toArrowArray(allocator);
        initialized += 1;
    }
    // Arrow permits a record batch with no columns and a non-zero row count.
    // Preserve that row-count metadata here so projecting a dataframe down to
    // zero columns remains lossless across the Boltha Arrow boundary.
    return boltha.arrow.RecordBatch.initOwnedWithRowCount(schema, columns, frame.rows);
}

pub fn toArrowTable(frame: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.Table {
    var batch = try toArrowRecordBatch(frame, allocator);
    errdefer batch.deinit(allocator);

    var schema = try batch.schema.clone(allocator);
    errdefer schema.deinit(allocator);

    const batches = try allocator.alloc(boltha.arrow.RecordBatch, 1);
    errdefer allocator.free(batches);
    batches[0] = batch;
    return boltha.arrow.Table.initOwned(schema, batches);
}

pub fn toParquetBytes(frame: anytype, allocator: std.mem.Allocator) ParquetInteropError![]u8 {
    var batch = try toArrowRecordBatch(frame, allocator);
    defer batch.deinit(allocator);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try boltha.parquet.writeRecordBatch(allocator, &out, batch);
    return out.toOwnedSlice(allocator);
}

pub fn fromParquetBytes(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    bytes: []const u8,
    device_value: array_mod.Device,
) ParquetInteropError!DeviceDataFrame {
    var batch = try boltha.parquet.readRecordBatch(allocator, bytes);
    defer batch.deinit(allocator);
    return fromArrowRecordBatch(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, batch, device_value);
}

pub fn fromParquetBytesPruned(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    bytes: []const u8,
    column_name: []const u8,
    predicate: anytype,
    device_value: array_mod.Device,
) ParquetInteropError!DeviceDataFrame {
    var table = try readBolthaTableWithRangePruning(allocator, bytes, column_name, predicate);
    defer table.deinit(allocator);
    return fromArrowTable(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, table, device_value);
}
