//! Shared metadata helpers for `DeviceParquetScan`.
//!
//! The scan object owns bytes and mutable pushdown state; this module keeps the
//! Boltha schema/footer conversion routines out of the lifecycle code so the
//! public scan type remains easier to read and extend.

const std = @import("std");
const boltha = @import("boltha");
const array_mod = @import("../../array.zig");
const arrow_extensions_mod = @import("../arrow/extensions.zig");
const dataframe_arrow_mod = @import("../arrow.zig");
const options_mod = @import("../../dataframe_options.zig");
const scan_summary_mod = @import("../../dataframe_parquet_scan_summary.zig");

const ParquetInteropError = dataframe_arrow_mod.ParquetInteropError;
const DeviceParquetFileSummary = scan_summary_mod.DeviceParquetFileSummary;

pub fn cloneArrowFields(
    allocator: std.mem.Allocator,
    fields: []const boltha.arrow.Field,
) ParquetInteropError![]boltha.arrow.Field {
    const owned = try allocator.alloc(boltha.arrow.Field, fields.len);
    var initialized: usize = 0;
    errdefer {
        for (owned[0..initialized]) |*field| field.deinit(allocator);
        allocator.free(owned);
    }
    for (fields, owned) |field, *slot| {
        slot.* = try field.clone(allocator);
        initialized += 1;
    }
    return owned;
}

fn schemaProjectionIndices(
    allocator: std.mem.Allocator,
    schema: boltha.arrow.Schema,
    wanted_names: []const []const u8,
) ParquetInteropError![]usize {
    const indices = try allocator.alloc(usize, wanted_names.len);
    errdefer allocator.free(indices);
    for (wanted_names, indices) |name, *slot| {
        slot.* = schema.fieldIndexByName(name) orelse return error.ColumnNotFound;
    }
    return indices;
}

pub fn projectArrowSchemaByName(
    allocator: std.mem.Allocator,
    schema: boltha.arrow.Schema,
    wanted_names: []const []const u8,
) ParquetInteropError!boltha.arrow.Schema {
    const indices = try schemaProjectionIndices(allocator, schema, wanted_names);
    defer allocator.free(indices);
    return schema.project(allocator, indices) catch |err| switch (err) {
        // `schemaProjectionIndices` resolves every name against this exact
        // schema, so invalid-index would indicate internal logic drift.
        error.InvalidFieldIndex => unreachable,
        else => |e| return e,
    };
}

pub fn toArrowSchema(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError!boltha.arrow.Schema {
    var schema = try boltha.parquet.readSchema(allocator, scan.bytes);
    errdefer schema.deinit(allocator);
    if (scan.projection) |names| {
        var projected = try projectArrowSchemaByName(allocator, schema, names);
        errdefer projected.deinit(allocator);
        schema.deinit(allocator);
        return projected;
    }
    return schema;
}

pub fn toArrowSchemaProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError!boltha.arrow.Schema {
    var schema = try boltha.parquet.readSchema(allocator, scan.bytes);
    defer schema.deinit(allocator);
    return projectArrowSchemaByName(allocator, schema, wanted_names);
}

pub fn toArrowFields(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]boltha.arrow.Field {
    var schema = try toArrowSchema(scan, allocator);
    defer schema.deinit(allocator);
    return cloneArrowFields(allocator, schema.fieldsView());
}

pub fn toArrowFieldsProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]boltha.arrow.Field {
    var schema = try toArrowSchemaProjection(scan, allocator, wanted_names);
    defer schema.deinit(allocator);
    return cloneArrowFields(allocator, schema.fieldsView());
}

pub fn arrowFieldCount(scan: anytype) ParquetInteropError!usize {
    var schema = try toArrowSchema(scan, scan.allocator);
    defer schema.deinit(scan.allocator);
    return schema.fieldCount();
}

pub fn arrowFieldNameAt(scan: anytype, allocator: std.mem.Allocator, index: usize) ParquetInteropError!?[]const u8 {
    var schema = try toArrowSchema(scan, allocator);
    defer schema.deinit(allocator);
    const field = schema.fieldAt(index) orelse return null;
    return try allocator.dupe(u8, field.name);
}

pub fn arrowFieldNames(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![][]const u8 {
    var schema = try toArrowSchema(scan, allocator);
    defer schema.deinit(allocator);

    const names = try allocator.alloc([]const u8, schema.fields.len);
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
        allocator.free(names);
    }
    for (schema.fields, names) |field, *slot| {
        slot.* = try allocator.dupe(u8, field.name);
        initialized += 1;
    }
    return names;
}

pub fn arrowFieldIndex(scan: anytype, name: []const u8) ParquetInteropError!?usize {
    var schema = try toArrowSchema(scan, scan.allocator);
    defer schema.deinit(scan.allocator);
    return schema.fieldIndexByName(name);
}

pub fn hasArrowField(scan: anytype, name: []const u8) bool {
    return (arrowFieldIndex(scan, name) catch null) != null;
}

pub fn hasAllArrowFields(scan: anytype, wanted_names: []const []const u8) bool {
    for (wanted_names) |name| {
        if (!hasArrowField(scan, name)) return false;
    }
    return true;
}

pub fn hasAnyArrowField(scan: anytype, wanted_names: []const []const u8) bool {
    for (wanted_names) |name| {
        if (hasArrowField(scan, name)) return true;
    }
    return false;
}

pub fn arrowFieldDTypeAt(scan: anytype, index: usize) ParquetInteropError!?array_mod.DType {
    var schema = try toArrowSchema(scan, scan.allocator);
    defer schema.deinit(scan.allocator);
    const field = schema.fieldAt(index) orelse return null;
    return try deviceDTypeFromArrowField(field.*);
}

pub fn arrowFieldDType(scan: anytype, name: []const u8) ParquetInteropError!array_mod.DType {
    var schema = try toArrowSchema(scan, scan.allocator);
    defer schema.deinit(scan.allocator);
    const field = schema.fieldByName(name) orelse return error.ColumnNotFound;
    return try deviceDTypeFromArrowField(field.*);
}

pub fn arrowFieldDTypes(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]array_mod.DType {
    var schema = try toArrowSchema(scan, allocator);
    defer schema.deinit(allocator);
    const dtypes = try allocator.alloc(array_mod.DType, schema.fields.len);
    errdefer allocator.free(dtypes);
    for (schema.fields, dtypes) |field, *slot| slot.* = try deviceDTypeFromArrowField(field);
    return dtypes;
}

pub fn arrowFieldDTypeNames(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![][]const u8 {
    const dtypes = try arrowFieldDTypes(scan, allocator);
    defer allocator.free(dtypes);
    const names = try allocator.alloc([]const u8, dtypes.len);
    for (dtypes, names) |dtype, *slot| slot.* = dtype.name();
    return names;
}

pub fn arrowFieldDTypeByteSizes(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]usize {
    const dtypes = try arrowFieldDTypes(scan, allocator);
    defer allocator.free(dtypes);
    const sizes = try allocator.alloc(usize, dtypes.len);
    for (dtypes, sizes) |dtype, *slot| slot.* = dtype.byteSize();
    return sizes;
}

pub fn arrowFieldDTypeBitSizes(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]usize {
    const dtypes = try arrowFieldDTypes(scan, allocator);
    defer allocator.free(dtypes);
    const sizes = try allocator.alloc(usize, dtypes.len);
    for (dtypes, sizes) |dtype, *slot| slot.* = dtype.bitSize();
    return sizes;
}

pub fn arrowFieldDTypeClassMask(scan: anytype, allocator: std.mem.Allocator, class: options_mod.DeviceDTypeClass) ParquetInteropError![]bool {
    const dtypes = try arrowFieldDTypes(scan, allocator);
    defer allocator.free(dtypes);
    const mask = try allocator.alloc(bool, dtypes.len);
    for (dtypes, mask) |dtype, *slot| slot.* = class.matches(dtype);
    return mask;
}

pub fn arrowFieldDTypeClassCount(scan: anytype, class: options_mod.DeviceDTypeClass) ParquetInteropError!usize {
    const dtypes = try arrowFieldDTypes(scan, scan.allocator);
    defer scan.allocator.free(dtypes);
    var count: usize = 0;
    for (dtypes) |dtype| {
        if (class.matches(dtype)) count += 1;
    }
    return count;
}

pub fn arrowFieldNullableAt(scan: anytype, index: usize) ParquetInteropError!?bool {
    var schema = try toArrowSchema(scan, scan.allocator);
    defer schema.deinit(scan.allocator);
    const field = schema.fieldAt(index) orelse return null;
    return field.nullable;
}

pub fn arrowFieldNullable(scan: anytype, name: []const u8) ParquetInteropError!bool {
    var schema = try toArrowSchema(scan, scan.allocator);
    defer schema.deinit(scan.allocator);
    const field = schema.fieldByName(name) orelse return error.ColumnNotFound;
    return field.nullable;
}

pub fn arrowFieldNullableMask(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]bool {
    var schema = try toArrowSchema(scan, allocator);
    defer schema.deinit(allocator);
    const mask = try allocator.alloc(bool, schema.fields.len);
    for (schema.fields, mask) |field, *slot| slot.* = field.nullable;
    return mask;
}

pub fn nullableArrowFieldCount(scan: anytype) ParquetInteropError!usize {
    var schema = try toArrowSchema(scan, scan.allocator);
    defer schema.deinit(scan.allocator);
    var count: usize = 0;
    for (schema.fields) |field| {
        if (field.nullable) count += 1;
    }
    return count;
}

pub fn allArrowFieldsNullable(scan: anytype) bool {
    const total = arrowFieldCount(scan) catch return false;
    return total != 0 and (nullableArrowFieldCount(scan) catch return false) == total;
}

pub fn hasArrowProjection(scan: anytype, wanted_names: []const []const u8) bool {
    var schema = boltha.parquet.readSchema(scan.allocator, scan.bytes) catch return false;
    defer schema.deinit(scan.allocator);
    for (wanted_names) |name| {
        if (schema.fieldIndexByName(name) == null) return false;
    }
    return true;
}

pub fn deviceDTypeFromArrowField(field: boltha.arrow.Field) ParquetInteropError!array_mod.DType {
    if (arrow_extensions_mod.dtypeFromField(field)) |dtype| return dtype;
    return switch (field.data_type) {
        .bool => .bool,
        .int => |info| if (info.signed) switch (info.bit_width) {
            8 => .i8,
            16 => .i16,
            32 => .i32,
            64 => .i64,
            else => error.TypeUnsupported,
        } else switch (info.bit_width) {
            8 => .u8,
            16 => .u16,
            32 => .u32,
            64 => .u64,
            else => error.TypeUnsupported,
        },
        .floating_point => |fp| switch (fp) {
            .half => .f16,
            .single => .f32,
            .double => .f64,
        },
        else => error.TypeUnsupported,
    };
}

fn nonNegativeI64ToUsize(value: i64) ParquetInteropError!usize {
    if (value < 0) return error.UnsupportedParquetSchema;
    return std.math.cast(usize, value) orelse error.UnsupportedParquetSchema;
}

pub fn bolthaFileSummaryToDeviceSummary(summary: boltha.parquet.FileSummary) ParquetInteropError!DeviceParquetFileSummary {
    return .{
        .rows = try nonNegativeI64ToUsize(summary.num_rows),
        .row_group_rows = try nonNegativeI64ToUsize(summary.row_group_num_rows),
        .row_groups = summary.row_group_count,
        .column_chunks = summary.column_count,
        .columns_with_metadata = summary.columns_with_metadata,
        .columns_without_metadata = summary.columns_without_metadata,
        .columns_with_column_index = summary.columns_with_column_index,
        .columns_with_offset_index = summary.columns_with_offset_index,
        .columns_with_page_index = summary.columns_with_page_index,
        .columns_with_bloom_filter = summary.columns_with_bloom_filter,
        .columns_with_sized_bloom_filter = summary.columns_with_sized_bloom_filter,
        .row_group_total_nbytes = try nonNegativeI64ToUsize(summary.row_group_total_byte_size),
        .row_group_total_compressed_nbytes = try nonNegativeI64ToUsize(summary.row_group_total_compressed_size),
        .row_groups_with_compressed_size = summary.row_groups_with_compressed_size,
        .total_compressed_nbytes = try nonNegativeI64ToUsize(summary.total_compressed_size),
        .total_uncompressed_nbytes = try nonNegativeI64ToUsize(summary.total_uncompressed_size),
    };
}
