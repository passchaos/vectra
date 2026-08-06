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
const schema_mod = @import("../../dataframe_schema.zig");
const scan_summary_mod = @import("../../dataframe_parquet_scan_summary.zig");

const ParquetInteropError = dataframe_arrow_mod.ParquetInteropError;
const DeviceColumnSchema = schema_mod.DeviceColumnSchema;
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

fn orderedOptionalBounds(comptime T: type, min_value: ?T, max_value: ?T) bool {
    if (min_value) |min_unwrapped| {
        if (max_value) |max_unwrapped| return min_unwrapped <= max_unwrapped;
    }
    return true;
}

fn orderedBoolBounds(min_value: ?bool, max_value: ?bool) bool {
    if (min_value) |min_unwrapped| {
        if (max_value) |max_unwrapped| return !min_unwrapped or max_unwrapped;
    }
    return true;
}

pub fn rangePredicateBoundsValid(predicate: anytype) bool {
    return switch (predicate) {
        .bool => |range| orderedBoolBounds(range.min, range.max),
        .i8 => |range| orderedOptionalBounds(i8, range.min, range.max),
        .i16 => |range| orderedOptionalBounds(i16, range.min, range.max),
        .i32 => |range| orderedOptionalBounds(i32, range.min, range.max),
        .i64 => |range| orderedOptionalBounds(i64, range.min, range.max),
        .u8 => |range| orderedOptionalBounds(u8, range.min, range.max),
        .u16 => |range| orderedOptionalBounds(u16, range.min, range.max),
        .u32 => |range| orderedOptionalBounds(u32, range.min, range.max),
        .u64 => |range| orderedOptionalBounds(u64, range.min, range.max),
        .usize => |range| orderedOptionalBounds(usize, range.min, range.max),
        .isize => |range| orderedOptionalBounds(isize, range.min, range.max),
        .f16 => |range| orderedOptionalBounds(f16, range.min, range.max),
        .f32 => |range| orderedOptionalBounds(f32, range.min, range.max),
        .f64 => |range| orderedOptionalBounds(f64, range.min, range.max),
        .bf16, .c64, .c128 => true,
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

pub fn arrowFieldDTypesProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]array_mod.DType {
    var schema = try toArrowSchemaProjection(scan, allocator, wanted_names);
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

pub fn arrowFieldDTypeNamesProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![][]const u8 {
    const dtypes = try arrowFieldDTypesProjection(scan, allocator, wanted_names);
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

pub fn arrowFieldDTypeByteSizesProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]usize {
    const dtypes = try arrowFieldDTypesProjection(scan, allocator, wanted_names);
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

pub fn arrowFieldDTypeBitSizesProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]usize {
    const dtypes = try arrowFieldDTypesProjection(scan, allocator, wanted_names);
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

pub fn arrowFieldDTypeClassMaskProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8, class: options_mod.DeviceDTypeClass) ParquetInteropError![]bool {
    const dtypes = try arrowFieldDTypesProjection(scan, allocator, wanted_names);
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

pub fn arrowFieldDTypeClassCountProjection(scan: anytype, wanted_names: []const []const u8, class: options_mod.DeviceDTypeClass) ParquetInteropError!usize {
    const dtypes = try arrowFieldDTypesProjection(scan, scan.allocator, wanted_names);
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

pub fn arrowFieldNullableMaskProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]bool {
    var schema = try toArrowSchemaProjection(scan, allocator, wanted_names);
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

pub fn nullableArrowFieldCountProjection(scan: anytype, wanted_names: []const []const u8) ParquetInteropError!usize {
    var schema = try toArrowSchemaProjection(scan, scan.allocator, wanted_names);
    defer schema.deinit(scan.allocator);
    var count: usize = 0;
    for (schema.fields) |field| {
        if (field.nullable) count += 1;
    }
    return count;
}

pub fn nonNullableArrowFieldCountProjection(scan: anytype, wanted_names: []const []const u8) ParquetInteropError!usize {
    return wanted_names.len - try nullableArrowFieldCountProjection(scan, wanted_names);
}

fn checkedMulUsize(left: usize, right: usize) ParquetInteropError!usize {
    return std.math.mul(usize, left, right) catch error.UnsupportedParquetSchema;
}

fn checkedAddUsize(left: usize, right: usize) ParquetInteropError!usize {
    return std.math.add(usize, left, right) catch error.UnsupportedParquetSchema;
}

fn checkedSubUsize(left: usize, right: usize) ParquetInteropError!usize {
    return std.math.sub(usize, left, right) catch error.UnsupportedParquetSchema;
}

fn ratioFromCount(count: usize, total: usize) f64 {
    if (total == 0) return std.math.nan(f64);
    return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(total));
}

fn parquetNullCountForFieldIndex(
    file_metadata: boltha.parquet.FileMetaData,
    field_index: usize,
    nullable: bool,
) ParquetInteropError!usize {
    var total: usize = 0;
    for (file_metadata.row_groups) |row_group| {
        if (field_index >= row_group.columns.len) return error.UnsupportedParquetSchema;
        const column_metadata = row_group.columns[field_index].metadata orelse {
            // Required fields are known to have no nulls even if the file omits
            // chunk metadata. Optional fields need statistics to answer this
            // metadata query without materializing column pages.
            if (nullable) return error.UnsupportedParquetSchema;
            continue;
        };
        const statistics = column_metadata.statistics orelse {
            if (nullable) return error.UnsupportedParquetSchema;
            continue;
        };
        const raw_null_count = statistics.null_count orelse {
            if (nullable) return error.UnsupportedParquetSchema;
            continue;
        };
        total = try checkedAddUsize(total, try nonNegativeI64ToUsize(raw_null_count));
    }
    return total;
}

fn parquetNullCountForField(scan: anytype, name: []const u8, nullable: bool) ParquetInteropError!usize {
    var full_schema = try boltha.parquet.readSchema(scan.allocator, scan.bytes);
    defer full_schema.deinit(scan.allocator);
    const full_index = full_schema.fieldIndexByName(name) orelse return error.ColumnNotFound;

    var file_metadata = try boltha.parquet.readFileMetadata(scan.allocator, scan.bytes);
    defer file_metadata.deinit(scan.allocator);
    return parquetNullCountForFieldIndex(file_metadata, full_index, nullable);
}

fn parquetNullCountsForFields(
    scan: anytype,
    allocator: std.mem.Allocator,
    fields: []const boltha.arrow.Field,
) ParquetInteropError![]usize {
    var full_schema = try boltha.parquet.readSchema(scan.allocator, scan.bytes);
    defer full_schema.deinit(scan.allocator);
    var file_metadata = try boltha.parquet.readFileMetadata(scan.allocator, scan.bytes);
    defer file_metadata.deinit(scan.allocator);

    const counts = try allocator.alloc(usize, fields.len);
    errdefer allocator.free(counts);
    for (fields, counts) |field, *slot| {
        const full_index = full_schema.fieldIndexByName(field.name) orelse return error.ColumnNotFound;
        slot.* = try parquetNullCountForFieldIndex(file_metadata, full_index, field.nullable);
    }
    return counts;
}

pub fn arrowFieldNullCount(scan: anytype, name: []const u8) ParquetInteropError!usize {
    var schema_value = try toArrowSchema(scan, scan.allocator);
    defer schema_value.deinit(scan.allocator);
    const field = schema_value.fieldByName(name) orelse return error.ColumnNotFound;
    return parquetNullCountForField(scan, name, field.nullable);
}

pub fn arrowFieldValidCount(scan: anytype, name: []const u8) ParquetInteropError!usize {
    return checkedSubUsize(try scan.rowCount(), try arrowFieldNullCount(scan, name));
}

pub fn arrowFieldNullCounts(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]usize {
    var schema_value = try toArrowSchema(scan, allocator);
    defer schema_value.deinit(allocator);
    return parquetNullCountsForFields(scan, allocator, schema_value.fields);
}

pub fn arrowFieldNullCountsProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]usize {
    var schema_value = try toArrowSchemaProjection(scan, allocator, wanted_names);
    defer schema_value.deinit(allocator);
    return parquetNullCountsForFields(scan, allocator, schema_value.fields);
}

pub fn arrowFieldValidCounts(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]usize {
    const null_counts = try arrowFieldNullCounts(scan, allocator);
    errdefer allocator.free(null_counts);
    const rows = try scan.rowCount();
    for (null_counts) |*count| count.* = try checkedSubUsize(rows, count.*);
    return null_counts;
}

pub fn arrowFieldValidCountsProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]usize {
    const null_counts = try arrowFieldNullCountsProjection(scan, allocator, wanted_names);
    errdefer allocator.free(null_counts);
    const rows = try scan.rowCount();
    for (null_counts) |*count| count.* = try checkedSubUsize(rows, count.*);
    return null_counts;
}

pub fn arrowFieldNullRatios(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]f64 {
    const counts = try arrowFieldNullCounts(scan, allocator);
    defer allocator.free(counts);
    const rows = try scan.rowCount();
    const ratios = try allocator.alloc(f64, counts.len);
    errdefer allocator.free(ratios);
    for (counts, ratios) |count, *slot| slot.* = ratioFromCount(count, rows);
    return ratios;
}

pub fn arrowFieldNullRatiosProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]f64 {
    const counts = try arrowFieldNullCountsProjection(scan, allocator, wanted_names);
    defer allocator.free(counts);
    const rows = try scan.rowCount();
    const ratios = try allocator.alloc(f64, counts.len);
    errdefer allocator.free(ratios);
    for (counts, ratios) |count, *slot| slot.* = ratioFromCount(count, rows);
    return ratios;
}

pub fn arrowFieldValidRatios(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]f64 {
    const counts = try arrowFieldValidCounts(scan, allocator);
    defer allocator.free(counts);
    const rows = try scan.rowCount();
    const ratios = try allocator.alloc(f64, counts.len);
    errdefer allocator.free(ratios);
    for (counts, ratios) |count, *slot| slot.* = ratioFromCount(count, rows);
    return ratios;
}

pub fn arrowFieldValidRatiosProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]f64 {
    const counts = try arrowFieldValidCountsProjection(scan, allocator, wanted_names);
    defer allocator.free(counts);
    const rows = try scan.rowCount();
    const ratios = try allocator.alloc(f64, counts.len);
    errdefer allocator.free(ratios);
    for (counts, ratios) |count, *slot| slot.* = ratioFromCount(count, rows);
    return ratios;
}

pub fn arrowNullCount(scan: anytype) ParquetInteropError!usize {
    const counts = try arrowFieldNullCounts(scan, scan.allocator);
    defer scan.allocator.free(counts);
    var total: usize = 0;
    for (counts) |count| total = try checkedAddUsize(total, count);
    return total;
}

pub fn arrowNullCountProjection(scan: anytype, wanted_names: []const []const u8) ParquetInteropError!usize {
    const counts = try arrowFieldNullCountsProjection(scan, scan.allocator, wanted_names);
    defer scan.allocator.free(counts);
    var total: usize = 0;
    for (counts) |count| total = try checkedAddUsize(total, count);
    return total;
}

pub fn arrowValidCount(scan: anytype) ParquetInteropError!usize {
    return checkedSubUsize(try checkedMulUsize(try scan.rowCount(), try arrowFieldCount(scan)), try arrowNullCount(scan));
}

pub fn arrowValidCountProjection(scan: anytype, wanted_names: []const []const u8) ParquetInteropError!usize {
    return checkedSubUsize(try checkedMulUsize(try scan.rowCount(), wanted_names.len), try arrowNullCountProjection(scan, wanted_names));
}

pub fn arrowNullRatio(scan: anytype) ParquetInteropError!f64 {
    return ratioFromCount(try arrowNullCount(scan), try checkedMulUsize(try scan.rowCount(), try arrowFieldCount(scan)));
}

pub fn arrowNullRatioProjection(scan: anytype, wanted_names: []const []const u8) ParquetInteropError!f64 {
    return ratioFromCount(try arrowNullCountProjection(scan, wanted_names), try checkedMulUsize(try scan.rowCount(), wanted_names.len));
}

pub fn arrowValidRatio(scan: anytype) ParquetInteropError!f64 {
    return ratioFromCount(try arrowValidCount(scan), try checkedMulUsize(try scan.rowCount(), try arrowFieldCount(scan)));
}

pub fn arrowValidRatioProjection(scan: anytype, wanted_names: []const []const u8) ParquetInteropError!f64 {
    return ratioFromCount(try arrowValidCountProjection(scan, wanted_names), try checkedMulUsize(try scan.rowCount(), wanted_names.len));
}

pub fn allArrowFieldsNullable(scan: anytype) bool {
    const total = arrowFieldCount(scan) catch return false;
    return total != 0 and (nullableArrowFieldCount(scan) catch return false) == total;
}

fn columnSchemaFromArrowField(scan: anytype, schema_name: []const u8, lookup_name: []const u8, field: boltha.arrow.Field) ParquetInteropError!DeviceColumnSchema {
    const rows = try scan.rowCount();
    const dtype = try deviceDTypeFromArrowField(field);
    const null_count = try parquetNullCountForField(scan, lookup_name, field.nullable);
    return .{
        .name = schema_name,
        .dtype = dtype,
        .rows = rows,
        .nullable = field.nullable,
        .null_count = null_count,
        .valid_count = try checkedSubUsize(rows, null_count),
        .data_nbytes = 0,
        .validity_nbytes = 0,
        .total_nbytes = 0,
        .device = scan.device,
    };
}

pub fn arrowColumnSchemaAt(scan: anytype, index: usize) ParquetInteropError!?DeviceColumnSchema {
    var schema_value = try toArrowSchema(scan, scan.allocator);
    defer schema_value.deinit(scan.allocator);
    const field = schema_value.fieldAt(index) orelse return null;
    return try columnSchemaFromArrowField(scan, "", field.name, field.*);
}

pub fn arrowColumnSchema(scan: anytype, name: []const u8) ParquetInteropError!DeviceColumnSchema {
    var schema_value = try toArrowSchema(scan, scan.allocator);
    defer schema_value.deinit(scan.allocator);
    const field = schema_value.fieldByName(name) orelse return error.ColumnNotFound;
    return try columnSchemaFromArrowField(scan, name, field.name, field.*);
}

pub fn arrowColumnSchemas(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
    var schema_value = try toArrowSchema(scan, allocator);
    defer schema_value.deinit(allocator);
    const rows = try scan.rowCount();
    const schemas = try allocator.alloc(DeviceColumnSchema, schema_value.fields.len);
    errdefer allocator.free(schemas);
    for (schema_value.fields, schemas, 0..) |field, *slot, index| {
        const dtype = try deviceDTypeFromArrowField(field);
        const schema_name = if (scan.projection) |names| names[index] else "";
        const null_count = try parquetNullCountForField(scan, field.name, field.nullable);
        slot.* = .{
            .name = schema_name,
            .dtype = dtype,
            .rows = rows,
            .nullable = field.nullable,
            .null_count = null_count,
            .valid_count = try checkedSubUsize(rows, null_count),
            .data_nbytes = 0,
            .validity_nbytes = 0,
            .total_nbytes = 0,
            .device = scan.device,
        };
    }
    return schemas;
}

pub fn arrowColumnSchemasProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]DeviceColumnSchema {
    var schema_value = try toArrowSchemaProjection(scan, allocator, wanted_names);
    defer schema_value.deinit(allocator);
    const rows = try scan.rowCount();
    const schemas = try allocator.alloc(DeviceColumnSchema, schema_value.fields.len);
    errdefer allocator.free(schemas);
    for (schema_value.fields, schemas, wanted_names) |field, *slot, name| {
        const dtype = try deviceDTypeFromArrowField(field);
        const null_count = try parquetNullCountForField(scan, field.name, field.nullable);
        slot.* = .{
            .name = name,
            .dtype = dtype,
            .rows = rows,
            .nullable = field.nullable,
            .null_count = null_count,
            .valid_count = try checkedSubUsize(rows, null_count),
            .data_nbytes = 0,
            .validity_nbytes = 0,
            .total_nbytes = 0,
            .device = scan.device,
        };
    }
    return schemas;
}

pub fn arrowSchemaSummary(scan: anytype, allocator: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
    return arrowColumnSchemas(scan, allocator);
}

pub fn arrowSchemaSummaryProjection(scan: anytype, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]DeviceColumnSchema {
    return arrowColumnSchemasProjection(scan, allocator, wanted_names);
}

fn schemasEqual(left: []const DeviceColumnSchema, right: []const DeviceColumnSchema) bool {
    if (left.len != right.len) return false;
    for (left, right) |left_schema, right_schema| {
        if (!left_schema.schemaEquals(right_schema)) return false;
    }
    return true;
}

pub fn arrowSchemaEquals(scan: anytype, other: @TypeOf(scan)) bool {
    const left = arrowColumnSchemas(scan, scan.allocator) catch return false;
    defer scan.allocator.free(left);
    const right = arrowColumnSchemas(other, other.allocator) catch return false;
    defer other.allocator.free(right);
    return schemasEqual(left, right);
}

pub fn arrowSameSchema(scan: anytype, other: @TypeOf(scan)) bool {
    return arrowSchemaEquals(scan, other);
}

pub fn arrowSchemaCompatible(scan: anytype, other: @TypeOf(scan)) bool {
    return arrowSchemaEquals(scan, other);
}

pub fn arrowSchemaEqualsSchemas(scan: anytype, schemas: []const DeviceColumnSchema) bool {
    const left = arrowColumnSchemas(scan, scan.allocator) catch return false;
    defer scan.allocator.free(left);
    return schemasEqual(left, schemas);
}

fn schemasCompatibleIgnoringNames(left: []const DeviceColumnSchema, right: []const DeviceColumnSchema) bool {
    if (left.len != right.len) return false;
    for (left, right) |left_schema, right_schema| {
        if (!left_schema.sameDType(right_schema) or
            !left_schema.sameNullability(right_schema) or
            !left_schema.sameShape(right_schema)) return false;
    }
    return true;
}

pub fn arrowSchemaEqualsFrame(scan: anytype, frame: anytype) bool {
    const scan_schema = arrowColumnSchemas(scan, scan.allocator) catch return false;
    defer scan.allocator.free(scan_schema);
    const frame_schema = frame.schema(scan.allocator) catch return false;
    defer scan.allocator.free(frame_schema);
    return schemasCompatibleIgnoringNames(scan_schema, frame_schema);
}

pub fn arrowSameSchemaFrame(scan: anytype, frame: anytype) bool {
    return arrowSchemaEqualsFrame(scan, frame);
}

pub fn arrowSchemaCompatibleFrame(scan: anytype, frame: anytype) bool {
    return arrowSchemaEqualsFrame(scan, frame);
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
