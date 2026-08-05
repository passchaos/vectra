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
