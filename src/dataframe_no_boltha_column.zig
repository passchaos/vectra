//! No-Boltha fixed-width device column facade.
//!
//! Boltha-free builds cannot construct real device dataframe columns, but keeping
//! the tagged column, definition, and schema surfaces in this small module lets
//! downstream code type-check metadata-only paths without importing Boltha.

const std = @import("std");
const array_mod = @import("array.zig");
const options_mod = @import("dataframe_no_boltha_options.zig");
const schema_mod = @import("dataframe_schema.zig");

pub const DeviceDType = array_mod.DType;
const DeviceValidityEncoding = options_mod.DeviceValidityEncoding;

pub fn DeviceTypedColumn(comptime T: type) type {
    return struct {
        data: array_mod.Array(T),
        validity: ?array_mod.Array(bool) = null,
        null_count: usize = 0,

        pub fn deinit(self: *@This()) void {
            self.data.deinit();
            if (self.validity) |*validity| validity.deinit();
            self.* = undefined;
        }
    };
}

pub const DeviceColumn = union(DeviceDType) {
    f32: DeviceTypedColumn(f32),
    f64: DeviceTypedColumn(f64),
    i8: DeviceTypedColumn(i8),
    i16: DeviceTypedColumn(i16),
    i32: DeviceTypedColumn(i32),
    i64: DeviceTypedColumn(i64),
    u8: DeviceTypedColumn(u8),
    u16: DeviceTypedColumn(u16),
    u32: DeviceTypedColumn(u32),
    u64: DeviceTypedColumn(u64),
    usize: DeviceTypedColumn(usize),
    bool: DeviceTypedColumn(bool),
    bf16: DeviceTypedColumn(array_mod.BFloat16),
    f16: DeviceTypedColumn(f16),
    c64: DeviceTypedColumn(array_mod.Complex64),
    c128: DeviceTypedColumn(array_mod.Complex128),
    isize: DeviceTypedColumn(isize),

    pub fn len(_: DeviceColumn) usize {
        return 0;
    }

    pub fn rowCount(_: DeviceColumn) usize {
        return 0;
    }

    pub fn height(_: DeviceColumn) usize {
        return 0;
    }

    pub fn nRows(_: DeviceColumn) usize {
        return 0;
    }

    pub fn shape(_: DeviceColumn) struct { rows: usize } {
        return .{ .rows = 0 };
    }

    pub fn isEmpty(_: DeviceColumn) bool {
        return true;
    }

    pub fn isNonEmpty(_: DeviceColumn) bool {
        return false;
    }

    pub fn hasRows(_: DeviceColumn) bool {
        return false;
    }

    pub fn cellCount(_: DeviceColumn) usize {
        return 0;
    }

    pub fn dtype(self: DeviceColumn) DeviceDType {
        return std.meta.activeTag(self);
    }

    pub fn dtypeName(self: DeviceColumn) []const u8 {
        return self.dtype().name();
    }

    pub fn dtypeByteSize(self: DeviceColumn) usize {
        return self.dtype().byteSize();
    }

    pub fn dtypeBitSize(self: DeviceColumn) usize {
        return self.dtype().bitSize();
    }

    pub fn isNumeric(self: DeviceColumn) bool {
        return self.dtype().isNumeric();
    }

    pub fn isReal(self: DeviceColumn) bool {
        return self.dtype().isReal();
    }

    pub fn isFloat(self: DeviceColumn) bool {
        return self.dtype().isFloat();
    }

    pub fn isInteger(self: DeviceColumn) bool {
        return self.dtype().isInteger();
    }

    pub fn isSignedInteger(self: DeviceColumn) bool {
        return self.dtype().isSigned();
    }

    pub fn isUnsignedInteger(self: DeviceColumn) bool {
        return self.dtype().isUnsigned();
    }

    pub fn isBool(self: DeviceColumn) bool {
        return self.dtype().isBool();
    }

    pub fn isComplex(self: DeviceColumn) bool {
        return self.dtype().isComplex();
    }

    pub fn device(_: DeviceColumn) array_mod.Device {
        return .cpu;
    }

    pub fn deviceValue(_: DeviceColumn) array_mod.Device {
        return .cpu;
    }

    pub fn isCpu(_: DeviceColumn) bool {
        return true;
    }

    pub fn isCuda(_: DeviceColumn) bool {
        return false;
    }

    pub fn isMps(_: DeviceColumn) bool {
        return false;
    }

    pub fn isDeviceBacked(_: DeviceColumn) bool {
        return false;
    }

    pub fn isDeviceAvailable(_: DeviceColumn) bool {
        return true;
    }

    pub fn deviceBackendName(_: DeviceColumn) []const u8 {
        return "cpu";
    }

    pub fn deviceBackend(_: DeviceColumn) array_mod.Backend {
        return .cpu;
    }

    pub fn deviceIndex(_: DeviceColumn) usize {
        return 0;
    }

    pub fn nullable(_: DeviceColumn) bool {
        return false;
    }

    pub fn hasNulls(_: DeviceColumn) bool {
        return false;
    }

    pub fn nullCount(_: DeviceColumn) usize {
        return 0;
    }

    pub fn validCount(_: DeviceColumn) usize {
        return 0;
    }

    pub fn anyNull(_: DeviceColumn) bool {
        return false;
    }

    pub fn allNull(_: DeviceColumn) bool {
        return true;
    }

    pub fn anyValid(_: DeviceColumn) bool {
        return false;
    }

    pub fn allValid(_: DeviceColumn) bool {
        return true;
    }

    fn ratioFromCount(count: usize, rows: usize) f64 {
        _ = count;
        if (rows == 0) return std.math.nan(f64);
        return 0.0;
    }

    pub fn nullRatio(_: DeviceColumn) f64 {
        return ratioFromCount(0, 0);
    }

    pub fn validRatio(_: DeviceColumn) f64 {
        return ratioFromCount(0, 0);
    }

    pub fn dataNbytes(_: DeviceColumn) usize {
        return 0;
    }

    pub fn dataMemoryUsage(_: DeviceColumn) usize {
        return 0;
    }

    pub fn dataPtr(_: DeviceColumn) u64 {
        return 0;
    }

    pub fn validityNbytes(_: DeviceColumn) usize {
        return 0;
    }

    pub fn validityMemoryUsage(_: DeviceColumn) usize {
        return 0;
    }

    pub fn validityPtr(_: DeviceColumn) ?u64 {
        return null;
    }

    pub fn hasValidity(_: DeviceColumn) bool {
        return false;
    }

    pub fn validityEncoding(_: DeviceColumn) DeviceValidityEncoding {
        return .none;
    }

    pub fn totalNbytes(_: DeviceColumn) usize {
        return 0;
    }

    pub fn memoryUsage(_: DeviceColumn) usize {
        return 0;
    }

    pub fn estimatedSize(_: DeviceColumn) usize {
        return 0;
    }

    pub fn sameDevice(_: DeviceColumn, _: DeviceColumn) bool {
        return true;
    }

    pub fn sameLength(_: DeviceColumn, _: DeviceColumn) bool {
        return true;
    }

    pub fn sameShape(_: DeviceColumn, _: DeviceColumn) bool {
        return true;
    }

    pub fn lengthEquals(_: DeviceColumn, rows: usize) bool {
        return rows == 0;
    }

    pub fn shapeEquals(_: DeviceColumn, rows: usize) bool {
        return rows == 0;
    }

    pub fn hasShape(self: DeviceColumn, rows: usize) bool {
        return self.shapeEquals(rows);
    }

    pub fn sameDType(self: DeviceColumn, other: DeviceColumn) bool {
        return self.dtype() == other.dtype();
    }

    pub fn sameNullability(_: DeviceColumn, _: DeviceColumn) bool {
        return true;
    }

    pub fn schemaEquals(self: DeviceColumn, other: DeviceColumn) bool {
        return self.sameDType(other) and self.sameNullability(other);
    }

    pub const sameSchema = schemaEquals;
    pub const schemaCompatible = schemaEquals;

    pub fn sameStorage(_: DeviceColumn, _: DeviceColumn) bool {
        return true;
    }

    pub fn schema(self: DeviceColumn, name: []const u8) DeviceColumnSchema {
        return .{
            .name = name,
            .dtype = self.dtype(),
            .rows = self.len(),
            .nullable = self.nullable(),
            .null_count = self.nullCount(),
            .valid_count = self.validCount(),
            .data_nbytes = self.dataNbytes(),
            .validity_nbytes = self.validityNbytes(),
            .total_nbytes = self.totalNbytes(),
            .device = self.device(),
        };
    }
};

pub const DeviceColumnDef = struct {
    name: []const u8,
    data: DeviceColumn,
};

pub const DeviceColumnSchema = schema_mod.DeviceColumnSchema;
