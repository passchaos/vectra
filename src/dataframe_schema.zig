const std = @import("std");
const array_mod = @import("array.zig");

/// Portable per-column metadata shared by owning dataframes and non-owning
/// views.  Keeping the layout Boltha-free lets no-Boltha builds expose the same
/// schema facade while Arrow/Parquet bridges remain optional.
pub const DeviceColumnSchema = struct {
    name: []const u8,
    dtype: array_mod.DType,
    rows: usize,
    nullable: bool,
    null_count: usize,
    valid_count: usize,
    data_nbytes: usize,
    validity_nbytes: usize,
    total_nbytes: usize,
    device: array_mod.Device,

    pub fn len(self: @This()) usize {
        return self.rows;
    }

    pub fn rowCount(self: @This()) usize {
        return self.len();
    }

    pub fn nRows(self: @This()) usize {
        return self.len();
    }

    pub fn shape(self: @This()) struct { rows: usize } {
        return .{ .rows = self.rows };
    }

    pub fn isEmpty(self: @This()) bool {
        return self.rows == 0;
    }

    pub fn isNonEmpty(self: @This()) bool {
        return !self.isEmpty();
    }

    pub fn hasRows(self: @This()) bool {
        return self.rows != 0;
    }

    pub fn dtypeName(self: @This()) []const u8 {
        return self.dtype.name();
    }

    pub fn dtypeByteSize(self: @This()) usize {
        return self.dtype.byteSize();
    }

    pub fn dtypeBitSize(self: @This()) usize {
        return self.dtype.bitSize();
    }

    pub fn isNumeric(self: @This()) bool {
        return self.dtype.isNumeric();
    }

    pub fn isReal(self: @This()) bool {
        return self.dtype.isReal();
    }

    pub fn isFloat(self: @This()) bool {
        return self.dtype.isFloat();
    }

    pub fn isInteger(self: @This()) bool {
        return self.dtype.isInteger();
    }

    pub fn isSignedInteger(self: @This()) bool {
        return self.dtype.isSigned();
    }

    pub fn isUnsignedInteger(self: @This()) bool {
        return self.dtype.isUnsigned();
    }

    pub fn isBool(self: @This()) bool {
        return self.dtype.isBool();
    }

    pub fn isComplex(self: @This()) bool {
        return self.dtype.isComplex();
    }

    pub fn nullableColumn(self: @This()) bool {
        return self.nullable;
    }

    pub fn nullCount(self: @This()) usize {
        return self.null_count;
    }

    pub fn validCount(self: @This()) usize {
        return self.valid_count;
    }

    pub fn hasNulls(self: @This()) bool {
        return self.null_count != 0;
    }

    pub fn anyNull(self: @This()) bool {
        return self.nullCount() != 0;
    }

    pub fn allNull(self: @This()) bool {
        return self.validCount() == 0;
    }

    pub fn anyValid(self: @This()) bool {
        return self.validCount() != 0;
    }

    pub fn allValid(self: @This()) bool {
        return self.null_count == 0;
    }

    fn ratioFromSchemaCount(count: usize, rows: usize) f64 {
        if (rows == 0) return std.math.nan(f64);
        return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(rows));
    }

    pub fn nullRatio(self: @This()) f64 {
        return ratioFromSchemaCount(self.null_count, self.rows);
    }

    pub fn validRatio(self: @This()) f64 {
        return ratioFromSchemaCount(self.valid_count, self.rows);
    }

    pub fn dataNbytes(self: @This()) usize {
        return self.data_nbytes;
    }

    pub fn dataMemoryUsage(self: @This()) usize {
        return self.data_nbytes;
    }

    pub fn validityNbytes(self: @This()) usize {
        return self.validity_nbytes;
    }

    pub fn validityMemoryUsage(self: @This()) usize {
        return self.validity_nbytes;
    }

    pub fn totalNbytes(self: @This()) usize {
        return self.total_nbytes;
    }

    pub fn memoryUsage(self: @This()) usize {
        return self.total_nbytes;
    }

    pub fn estimatedSize(self: @This()) usize {
        return self.total_nbytes;
    }

    pub fn isCpu(self: @This()) bool {
        return self.device.isCpu();
    }

    pub fn isCuda(self: @This()) bool {
        return self.device.isCuda();
    }

    pub fn isMps(self: @This()) bool {
        return self.device.isMps();
    }

    pub fn isDeviceBacked(self: @This()) bool {
        return !self.isCpu();
    }

    pub fn deviceBackendName(self: @This()) []const u8 {
        return self.device.backendName();
    }

    pub fn sameDevice(self: @This(), other: @This()) bool {
        return self.device.sameDevice(other.device);
    }

    pub fn sameLength(self: @This(), other: @This()) bool {
        return self.rows == other.rows;
    }

    pub fn sameShape(self: @This(), other: @This()) bool {
        return self.sameLength(other);
    }

    pub fn lengthEquals(self: @This(), rows: usize) bool {
        return self.rows == rows;
    }

    pub fn shapeEquals(self: @This(), rows: usize) bool {
        return self.lengthEquals(rows);
    }

    pub fn hasShape(self: @This(), rows: usize) bool {
        return self.shapeEquals(rows);
    }

    pub fn sameDType(self: @This(), other: @This()) bool {
        return self.dtype == other.dtype;
    }

    pub fn sameNullability(self: @This(), other: @This()) bool {
        return self.nullable == other.nullable;
    }

    pub fn schemaEquals(self: @This(), other: @This()) bool {
        return std.mem.eql(u8, self.name, other.name) and
            self.dtype == other.dtype and
            self.nullable == other.nullable;
    }

    pub const sameSchema = schemaEquals;
    pub const schemaCompatible = schemaEquals;
};
