//! Arrow and Parquet wrapper methods for eager DeviceDataFrame.

const std = @import("std");
const array_mod = @import("../../array.zig");
const boltha = @import("boltha");
const dataframe_arrow_mod = @import("../arrow.zig");
const options_mod = @import("../../dataframe_options.zig");
const series_mod = @import("../../series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const ArrowInteropError = DeviceDataError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
const ParquetInteropError = ArrowInteropError || boltha.parquet.SimpleError;
const ParquetRangePredicate = options_mod.ParquetRangePredicate;

fn FrameType(comptime Frame: type) type {
    return switch (@typeInfo(Frame)) {
        .pointer => |ptr| ptr.child,
        else => Frame,
    };
}

fn frameValue(self: anytype) FrameType(@TypeOf(self)) {
    return switch (@typeInfo(@TypeOf(self))) {
        .pointer => self.*,
        else => self,
    };
}

pub fn toArrowSchema(self: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.Schema {
    return dataframe_arrow_mod.toArrowSchema(frameValue(self), allocator);
}

pub fn toArrowFields(self: anytype, allocator: std.mem.Allocator) ArrowInteropError![]boltha.arrow.Field {
    return dataframe_arrow_mod.toArrowFields(frameValue(self), allocator);
}

pub fn toArrowFieldFromView(view: anytype, allocator: std.mem.Allocator, name: []const u8) ArrowInteropError!boltha.arrow.Field {
    return dataframe_arrow_mod.deviceColumnSchemaToArrowField(allocator, view.schema(name));
}

pub fn arrowDataTypeFromSchema(schema_value: anytype) ArrowInteropError!boltha.arrow.DataType {
    return dataframe_arrow_mod.deviceDTypeToArrowDataType(schema_value.dtype);
}

pub fn toArrowFieldFromSchema(schema_value: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.Field {
    return dataframe_arrow_mod.deviceColumnSchemaToArrowField(allocator, schema_value);
}

pub fn toArrowRecordBatch(self: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.RecordBatch {
    return dataframe_arrow_mod.toArrowRecordBatch(frameValue(self), allocator);
}

pub fn toArrowTable(self: anytype, allocator: std.mem.Allocator) ArrowInteropError!boltha.arrow.Table {
    return dataframe_arrow_mod.toArrowTable(frameValue(self), allocator);
}

pub fn toParquetBytes(self: anytype, allocator: std.mem.Allocator) ParquetInteropError![]u8 {
    return dataframe_arrow_mod.toParquetBytes(frameValue(self), allocator);
}

pub fn writeParquetFileInDir(self: anytype, dir: std.Io.Dir, io: std.Io, path: []const u8) (ParquetInteropError || std.Io.Dir.WriteFileError)!void {
    const frame = frameValue(self);
    const bytes = try dataframe_arrow_mod.toParquetBytes(frame, frame.allocator);
    defer frame.allocator.free(bytes);
    try dir.writeFile(io, .{ .sub_path = path, .data = bytes });
}

pub fn writeParquetFile(self: anytype, io: std.Io, path: []const u8) (ParquetInteropError || std.Io.Dir.WriteFileError)!void {
    return writeParquetFileInDir(self, std.Io.Dir.cwd(), io, path);
}

pub fn fromParquetFileInDir(comptime Frame: type, comptime DeviceColumnDef: type, allocator: std.mem.Allocator, dir: std.Io.Dir, io: std.Io, path: []const u8, read_limit: std.Io.Limit, device_value: array_mod.Device) !Frame {
    const bytes = try dir.readFileAlloc(io, path, allocator, read_limit);
    defer allocator.free(bytes);
    return fromParquetBytes(Frame, DeviceColumnDef, allocator, bytes, device_value);
}

pub fn fromParquetFile(comptime Frame: type, comptime DeviceColumnDef: type, allocator: std.mem.Allocator, io: std.Io, path: []const u8, read_limit: std.Io.Limit, device_value: array_mod.Device) !Frame {
    return fromParquetFileInDir(Frame, DeviceColumnDef, allocator, std.Io.Dir.cwd(), io, path, read_limit, device_value);
}

pub fn fromParquetBytes(comptime Frame: type, comptime DeviceColumnDef: type, allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) ParquetInteropError!Frame {
    return dataframe_arrow_mod.fromParquetBytes(Frame, DeviceColumnDef, @FieldType(DeviceColumnDef, "data"), allocator, bytes, device_value);
}

pub fn fromParquetBytesPruned(
    comptime Frame: type,
    comptime DeviceColumnDef: type,
    allocator: std.mem.Allocator,
    bytes: []const u8,
    column_name: []const u8,
    predicate: ParquetRangePredicate,
    device_value: array_mod.Device,
) ParquetInteropError!Frame {
    return dataframe_arrow_mod.fromParquetBytesPruned(Frame, DeviceColumnDef, @FieldType(DeviceColumnDef, "data"), allocator, bytes, column_name, predicate, device_value);
}

pub fn fromParquetFilePrunedInDir(
    comptime Frame: type,
    comptime DeviceColumnDef: type,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    io: std.Io,
    path: []const u8,
    read_limit: std.Io.Limit,
    column_name: []const u8,
    predicate: ParquetRangePredicate,
    device_value: array_mod.Device,
) !Frame {
    const bytes = try dir.readFileAlloc(io, path, allocator, read_limit);
    defer allocator.free(bytes);
    return fromParquetBytesPruned(Frame, DeviceColumnDef, allocator, bytes, column_name, predicate, device_value);
}

pub fn fromParquetFilePruned(
    comptime Frame: type,
    comptime DeviceColumnDef: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    read_limit: std.Io.Limit,
    column_name: []const u8,
    predicate: ParquetRangePredicate,
    device_value: array_mod.Device,
) !Frame {
    return fromParquetFilePrunedInDir(Frame, DeviceColumnDef, allocator, std.Io.Dir.cwd(), io, path, read_limit, column_name, predicate, device_value);
}

pub fn fromArrowTable(comptime Frame: type, comptime DeviceColumnDef: type, allocator: std.mem.Allocator, table: boltha.arrow.Table, device_value: array_mod.Device) ArrowInteropError!Frame {
    return dataframe_arrow_mod.fromArrowTable(Frame, DeviceColumnDef, @FieldType(DeviceColumnDef, "data"), allocator, table, device_value);
}

pub fn fromArrowTableProjection(
    comptime Frame: type,
    comptime DeviceColumnDef: type,
    allocator: std.mem.Allocator,
    table: boltha.arrow.Table,
    wanted_names: []const []const u8,
    device_value: array_mod.Device,
) ArrowInteropError!Frame {
    return dataframe_arrow_mod.fromArrowTableProjection(Frame, DeviceColumnDef, @FieldType(DeviceColumnDef, "data"), allocator, table, wanted_names, device_value);
}

pub fn fromArrowRecordBatch(comptime Frame: type, comptime DeviceColumnDef: type, allocator: std.mem.Allocator, batch: boltha.arrow.RecordBatch, device_value: array_mod.Device) ArrowInteropError!Frame {
    return dataframe_arrow_mod.fromArrowRecordBatch(Frame, DeviceColumnDef, @FieldType(DeviceColumnDef, "data"), allocator, batch, device_value);
}

pub fn fromArrowRecordBatchProjection(
    comptime Frame: type,
    comptime DeviceColumnDef: type,
    allocator: std.mem.Allocator,
    batch: boltha.arrow.RecordBatch,
    wanted_names: []const []const u8,
    device_value: array_mod.Device,
) ArrowInteropError!Frame {
    return dataframe_arrow_mod.fromArrowRecordBatchProjection(Frame, DeviceColumnDef, @FieldType(DeviceColumnDef, "data"), allocator, batch, wanted_names, device_value);
}
