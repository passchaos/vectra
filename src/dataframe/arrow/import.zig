//! Arrow/Boltha table and record-batch import helpers for device dataframes.

const std = @import("std");
const array_mod = @import("../../array.zig");
const boltha = @import("boltha");
const dataframe_array_mod = @import("../../dataframe_array.zig");
const arrow_columns_mod = @import("columns.zig");

pub const DataFrameInitError = std.mem.Allocator.Error || std.Io.Writer.Error || error{ LengthMismatch, ColumnNotFound, TypeMismatch, InvalidCsv, EmptyDataFrame, UnsupportedType, InvalidDevice };
pub const ArrowInteropError = DataFrameInitError || array_mod.ArrayError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;

const emptyDeviceColumnFromArrowType = arrow_columns_mod.emptyDeviceColumnFromArrowType;
const deviceColumnFromArrowArray = arrow_columns_mod.deviceColumnFromArrowArray;

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

pub fn fromArrowTable(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    table: boltha.arrow.Table,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceDataFrame {
    if (table.batches.len == 0) return emptyFromArrowSchema(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, table.schema, table.row_count, device_value);
    var out = try fromArrowRecordBatch(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, table.batches[0], device_value);
    errdefer out.deinit();
    for (table.batches[1..]) |batch| {
        var next = try fromArrowRecordBatch(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, batch, device_value);
        defer next.deinit();
        const combined = try dataframe_array_mod.concatDeviceDataFramesRows(DeviceDataFrame, out, next);
        out.deinit();
        out = combined;
    }
    return out;
}

pub fn fromArrowTableProjection(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    table: boltha.arrow.Table,
    wanted_names: []const []const u8,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceDataFrame {
    if (wanted_names.len == 0) return DeviceDataFrame.initEmpty(allocator, table.row_count, device_value);
    if (table.batches.len == 0) return emptyFromArrowSchemaProjection(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, table.schema, table.row_count, wanted_names, device_value);

    // Projection is applied while crossing the Arrow -> DeviceDataFrame
    // boundary. Boltha's current simple Parquet reader still decodes full row
    // groups, but dropped columns are not uploaded/materialized into Vectra
    // arrays, preserving the public projection-pushdown contract.
    var out = try fromArrowRecordBatchProjection(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, table.batches[0], wanted_names, device_value);
    errdefer out.deinit();
    for (table.batches[1..]) |batch| {
        var next = try fromArrowRecordBatchProjection(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, batch, wanted_names, device_value);
        defer next.deinit();
        const combined = try dataframe_array_mod.concatDeviceDataFramesRows(DeviceDataFrame, out, next);
        out.deinit();
        out = combined;
    }
    return out;
}

pub fn fromArrowRecordBatch(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    batch: boltha.arrow.RecordBatch,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceDataFrame {
    if (!device_value.isAvailable()) return error.InvalidDevice;
    var defs = try allocator.alloc(DeviceColumnDef, batch.columns.len);
    defer allocator.free(defs);
    var initialized: usize = 0;
    defer {
        for (defs[0..initialized]) |*def| def.data.deinit();
    }
    for (batch.schema.fields, batch.columns, 0..) |field, arrow_column, i| {
        defs[i] = .{
            .name = field.name,
            .data = try deviceColumnFromArrowArray(DeviceColumn, allocator, arrow_column, device_value),
        };
        initialized += 1;
    }
    return DeviceDataFrame.init(allocator, defs);
}

pub fn fromArrowRecordBatchProjection(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    batch: boltha.arrow.RecordBatch,
    wanted_names: []const []const u8,
    device_value: array_mod.Device,
) ArrowInteropError!DeviceDataFrame {
    if (!device_value.isAvailable()) return error.InvalidDevice;
    if (wanted_names.len == 0) return DeviceDataFrame.initEmpty(allocator, batch.row_count, device_value);

    var defs = try allocator.alloc(DeviceColumnDef, wanted_names.len);
    defer allocator.free(defs);
    var initialized: usize = 0;
    defer {
        for (defs[0..initialized]) |*def| def.data.deinit();
    }
    for (wanted_names, 0..) |name, i| {
        const column_index = batch.schema.fieldIndexByName(name) orelse return error.ColumnNotFound;
        defs[i] = .{
            .name = batch.schema.fields[column_index].name,
            .data = try deviceColumnFromArrowArray(DeviceColumn, allocator, batch.columns[column_index], device_value),
        };
        initialized += 1;
    }
    return DeviceDataFrame.init(allocator, defs);
}
