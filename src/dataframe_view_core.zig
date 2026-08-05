const std = @import("std");
const array_mod = @import("array.zig");
const schema_mod = @import("dataframe_schema.zig");

/// Builds the shared non-owning dataframe view API for both the Boltha-backed
/// facade and the no-Boltha fallback facade.  The options module supplies small
/// enum types while the error set keeps name-based access aligned with the
/// importing facade.
pub fn DeviceViewTypes(
    comptime DeviceValidityEncoding: type,
    comptime DeviceDTypeClass: type,
    comptime DataError: type,
) type {
    const ViewError = DataError || array_mod.ArrayError;

    return struct {
        pub const DeviceDataFrameViewError = ViewError;
        pub const DeviceColumnSchema = schema_mod.DeviceColumnSchema;

        pub const DeviceColumnView = struct {
            dtype: array_mod.DType,
            rows: usize,
            device: array_mod.Device,
            data_ptr: u64,
            data_nbytes: usize,
            validity_ptr: ?u64 = null,
            validity_nbytes: usize = 0,
            null_count: usize = 0,
            validity_encoding: DeviceValidityEncoding = .none,

            pub fn len(self: DeviceColumnView) usize {
                return self.rows;
            }

            pub fn dtypeName(self: DeviceColumnView) []const u8 {
                return self.dtype.name();
            }

            pub fn dtypeByteSize(self: DeviceColumnView) usize {
                return self.dtype.byteSize();
            }

            pub fn dtypeBitSize(self: DeviceColumnView) usize {
                return self.dtype.bitSize();
            }

            pub fn isNumeric(self: DeviceColumnView) bool {
                return self.dtype.isNumeric();
            }

            pub fn isReal(self: DeviceColumnView) bool {
                return self.dtype.isReal();
            }

            pub fn isFloat(self: DeviceColumnView) bool {
                return self.dtype.isFloat();
            }

            pub fn isInteger(self: DeviceColumnView) bool {
                return self.dtype.isInteger();
            }

            pub fn isSignedInteger(self: DeviceColumnView) bool {
                return self.dtype.isSigned();
            }

            pub fn isUnsignedInteger(self: DeviceColumnView) bool {
                return self.dtype.isUnsigned();
            }

            pub fn isBool(self: DeviceColumnView) bool {
                return self.dtype.isBool();
            }

            pub fn isComplex(self: DeviceColumnView) bool {
                return self.dtype.isComplex();
            }

            pub fn nullable(self: DeviceColumnView) bool {
                return self.validity_ptr != null;
            }

            pub fn hasNulls(self: DeviceColumnView) bool {
                return self.null_count != 0;
            }

            pub fn anyNull(self: DeviceColumnView) bool {
                return self.nullCount() != 0;
            }

            pub fn allNull(self: DeviceColumnView) bool {
                return self.validCount() == 0;
            }

            pub fn anyValid(self: DeviceColumnView) bool {
                return self.validCount() != 0;
            }

            pub fn allValid(self: DeviceColumnView) bool {
                return self.nullCount() == 0;
            }

            pub fn nullCount(self: DeviceColumnView) usize {
                return self.null_count;
            }

            pub fn validCount(self: DeviceColumnView) usize {
                return self.rows - self.null_count;
            }

            fn ratioFromCount(count: usize, rows: usize) f64 {
                if (rows == 0) return std.math.nan(f64);
                return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(rows));
            }

            pub fn nullRatio(self: DeviceColumnView) f64 {
                return ratioFromCount(self.null_count, self.rows);
            }

            pub fn validRatio(self: DeviceColumnView) f64 {
                return ratioFromCount(self.validCount(), self.rows);
            }

            pub fn dataNbytes(self: DeviceColumnView) usize {
                return self.data_nbytes;
            }

            pub fn dataMemoryUsage(self: DeviceColumnView) usize {
                return self.data_nbytes;
            }

            pub fn validityNbytes(self: DeviceColumnView) usize {
                return self.validity_nbytes;
            }

            pub fn validityMemoryUsage(self: DeviceColumnView) usize {
                return self.validity_nbytes;
            }

            pub fn totalNbytes(self: DeviceColumnView) usize {
                return self.data_nbytes + self.validity_nbytes;
            }

            pub fn memoryUsage(self: DeviceColumnView) usize {
                return self.totalNbytes();
            }

            pub fn estimatedSize(self: DeviceColumnView) usize {
                return self.totalNbytes();
            }

            pub fn isCpu(self: DeviceColumnView) bool {
                return self.device.isCpu();
            }

            pub fn isCuda(self: DeviceColumnView) bool {
                return self.device.isCuda();
            }

            pub fn isMps(self: DeviceColumnView) bool {
                return self.device.isMps();
            }

            pub fn isDeviceBacked(self: DeviceColumnView) bool {
                return !self.device.isCpu();
            }

            pub fn deviceBackendName(self: DeviceColumnView) []const u8 {
                return self.device.backendName();
            }

            pub fn sameDevice(self: DeviceColumnView, other: DeviceColumnView) bool {
                return self.device.sameDevice(other.device);
            }

            pub fn sameLength(self: DeviceColumnView, other: DeviceColumnView) bool {
                return self.rows == other.rows;
            }

            pub fn lengthEquals(self: DeviceColumnView, rows: usize) bool {
                return self.rows == rows;
            }

            pub fn sameDType(self: DeviceColumnView, other: DeviceColumnView) bool {
                return self.dtype == other.dtype;
            }

            pub fn sameNullability(self: DeviceColumnView, other: DeviceColumnView) bool {
                return self.nullable() == other.nullable();
            }

            pub fn schemaEquals(self: DeviceColumnView, other: DeviceColumnView) bool {
                return self.sameDType(other) and self.sameNullability(other);
            }

            pub const sameSchema = schemaEquals;
            pub const schemaCompatible = schemaEquals;

            pub fn sameStorage(self: DeviceColumnView, other: DeviceColumnView) bool {
                return self.data_ptr == other.data_ptr and
                    self.data_nbytes == other.data_nbytes and
                    self.validity_ptr == other.validity_ptr and
                    self.validity_nbytes == other.validity_nbytes and
                    self.validity_encoding == other.validity_encoding;
            }
        };

        /// Non-owning table metadata modeled after cuDF's `table_view`.
        ///
        /// The view does not own column storage or names; it only owns the small
        /// `columns` metadata slice allocated by `DeviceDataFrame.view()`. Users may pass
        /// this compact description to backend bridges without copying column buffers.
        pub const DeviceDataFrameView = struct {
            allocator: std.mem.Allocator,
            names: []const []const u8,
            columns: []DeviceColumnView,
            rows: usize,
            device: array_mod.Device,

            pub fn deinit(self: *DeviceDataFrameView) void {
                if (self.columns.len != 0) self.allocator.free(self.columns);
                self.* = undefined;
            }

            pub fn height(self: DeviceDataFrameView) usize {
                return self.rows;
            }

            pub fn rowCount(self: DeviceDataFrameView) usize {
                return self.height();
            }

            pub fn nRows(self: DeviceDataFrameView) usize {
                return self.height();
            }

            pub fn width(self: DeviceDataFrameView) usize {
                return self.columns.len;
            }

            pub fn columnCount(self: DeviceDataFrameView) usize {
                return self.width();
            }

            pub fn cols(self: DeviceDataFrameView) usize {
                return self.width();
            }

            pub fn nCols(self: DeviceDataFrameView) usize {
                return self.width();
            }

            pub fn shape(self: DeviceDataFrameView) struct { rows: usize, cols: usize } {
                return .{ .rows = self.rows, .cols = self.columns.len };
            }

            pub fn columnDTypes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]array_mod.DType {
                const out = try allocator.alloc(array_mod.DType, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.dtype;
                return out;
            }

            pub fn dtypes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]array_mod.DType {
                return self.columnDTypes(allocator);
            }

            pub fn columnDTypeNames(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![][]const u8 {
                const out = try allocator.alloc([]const u8, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.dtype.name();
                return out;
            }

            pub fn dtypeNames(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![][]const u8 {
                return self.columnDTypeNames(allocator);
            }

            pub fn columnDTypeByteSizes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                const out = try allocator.alloc(usize, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.dtype.byteSize();
                return out;
            }

            pub fn columnDTypeBitSizes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                const out = try allocator.alloc(usize, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.dtype.bitSize();
                return out;
            }

            pub fn columnDTypeClassMask(
                self: DeviceDataFrameView,
                allocator: std.mem.Allocator,
                class: DeviceDTypeClass,
            ) std.mem.Allocator.Error![]bool {
                const out = try allocator.alloc(bool, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = class.matches(column_value.dtype);
                return out;
            }

            pub fn columnDTypeClassCount(self: DeviceDataFrameView, class: DeviceDTypeClass) usize {
                var count: usize = 0;
                for (self.columns) |column_value| {
                    if (class.matches(column_value.dtype)) count += 1;
                }
                return count;
            }

            pub fn numericColumnCount(self: DeviceDataFrameView) usize {
                return self.columnDTypeClassCount(.numeric);
            }

            pub fn realColumnCount(self: DeviceDataFrameView) usize {
                return self.columnDTypeClassCount(.real);
            }

            pub fn floatColumnCount(self: DeviceDataFrameView) usize {
                return self.columnDTypeClassCount(.float);
            }

            pub fn integerColumnCount(self: DeviceDataFrameView) usize {
                return self.columnDTypeClassCount(.integer);
            }

            pub fn signedIntegerColumnCount(self: DeviceDataFrameView) usize {
                return self.columnDTypeClassCount(.signed_integer);
            }

            pub fn unsignedIntegerColumnCount(self: DeviceDataFrameView) usize {
                return self.columnDTypeClassCount(.unsigned_integer);
            }

            pub fn boolColumnCount(self: DeviceDataFrameView) usize {
                return self.columnDTypeClassCount(.bool);
            }

            pub fn complexColumnCount(self: DeviceDataFrameView) usize {
                return self.columnDTypeClassCount(.complex);
            }

            pub fn columnIsNumericMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                return self.columnDTypeClassMask(allocator, .numeric);
            }

            pub fn columnIsRealMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                return self.columnDTypeClassMask(allocator, .real);
            }

            pub fn columnIsFloatMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                return self.columnDTypeClassMask(allocator, .float);
            }

            pub fn columnIsIntegerMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                return self.columnDTypeClassMask(allocator, .integer);
            }

            pub fn columnIsSignedIntegerMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                return self.columnDTypeClassMask(allocator, .signed_integer);
            }

            pub fn columnIsUnsignedIntegerMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                return self.columnDTypeClassMask(allocator, .unsigned_integer);
            }

            pub fn columnIsBoolMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                return self.columnDTypeClassMask(allocator, .bool);
            }

            pub fn columnIsComplexMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                return self.columnDTypeClassMask(allocator, .complex);
            }

            pub fn columnNullCounts(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                const out = try allocator.alloc(usize, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.nullCount();
                return out;
            }

            pub fn columnValidCounts(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                const out = try allocator.alloc(usize, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.validCount();
                return out;
            }

            pub fn nullCount(self: DeviceDataFrameView) usize {
                var count: usize = 0;
                for (self.columns) |column_value| count += column_value.nullCount();
                return count;
            }

            pub fn validCount(self: DeviceDataFrameView) usize {
                var count: usize = 0;
                for (self.columns) |column_value| count += column_value.validCount();
                return count;
            }

            pub fn cellCount(self: DeviceDataFrameView) usize {
                return self.rows * self.columns.len;
            }

            fn ratioFromCount(count: usize, rows: usize) f64 {
                if (rows == 0) return std.math.nan(f64);
                return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(rows));
            }

            pub fn nullRatio(self: DeviceDataFrameView) f64 {
                return ratioFromCount(self.nullCount(), self.cellCount());
            }

            pub fn validRatio(self: DeviceDataFrameView) f64 {
                return ratioFromCount(self.validCount(), self.cellCount());
            }

            pub fn columnNullRatios(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]f64 {
                const out = try allocator.alloc(f64, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = ratioFromCount(column_value.nullCount(), self.rows);
                return out;
            }

            pub fn columnValidRatios(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]f64 {
                const out = try allocator.alloc(f64, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = ratioFromCount(column_value.validCount(), self.rows);
                return out;
            }

            pub fn columnNullableMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                const out = try allocator.alloc(bool, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.nullable();
                return out;
            }

            pub fn nullableColumnCount(self: DeviceDataFrameView) usize {
                var count: usize = 0;
                for (self.columns) |column_value| {
                    if (column_value.nullable()) count += 1;
                }
                return count;
            }

            pub fn nonNullableColumnCount(self: DeviceDataFrameView) usize {
                return self.columns.len - self.nullableColumnCount();
            }

            pub fn columnHasNullsMask(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
                const out = try allocator.alloc(bool, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.hasNulls();
                return out;
            }

            pub fn columnsWithNullsCount(self: DeviceDataFrameView) usize {
                var count: usize = 0;
                for (self.columns) |column_value| {
                    if (column_value.hasNulls()) count += 1;
                }
                return count;
            }

            pub fn columnsWithoutNullsCount(self: DeviceDataFrameView) usize {
                return self.columns.len - self.columnsWithNullsCount();
            }

            pub fn columnDataNbytes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                const out = try allocator.alloc(usize, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.dataNbytes();
                return out;
            }

            pub fn columnDataMemoryUsage(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                return self.columnDataNbytes(allocator);
            }

            pub fn columnValidityNbytes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                const out = try allocator.alloc(usize, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.validityNbytes();
                return out;
            }

            pub fn columnValidityMemoryUsage(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                return self.columnValidityNbytes(allocator);
            }

            pub fn columnTotalNbytes(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                const out = try allocator.alloc(usize, self.columns.len);
                for (self.columns, out) |column_value, *slot| slot.* = column_value.totalNbytes();
                return out;
            }

            pub fn columnMemoryUsage(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
                return self.columnTotalNbytes(allocator);
            }

            pub fn dataNbytes(self: DeviceDataFrameView) usize {
                var total: usize = 0;
                for (self.columns) |column_value| total += column_value.dataNbytes();
                return total;
            }

            pub fn dataMemoryUsage(self: DeviceDataFrameView) usize {
                return self.dataNbytes();
            }

            pub fn validityNbytes(self: DeviceDataFrameView) usize {
                var total: usize = 0;
                for (self.columns) |column_value| total += column_value.validityNbytes();
                return total;
            }

            pub fn validityMemoryUsage(self: DeviceDataFrameView) usize {
                return self.validityNbytes();
            }

            pub fn totalNbytes(self: DeviceDataFrameView) usize {
                return self.dataNbytes() + self.validityNbytes();
            }

            pub fn memoryUsage(self: DeviceDataFrameView) usize {
                return self.totalNbytes();
            }

            pub fn estimatedSize(self: DeviceDataFrameView) usize {
                return self.totalNbytes();
            }

            pub fn columnSchemaAt(self: DeviceDataFrameView, index: usize) DeviceDataFrameViewError!DeviceColumnSchema {
                if (index >= self.columns.len) return error.IndexOutOfBounds;
                const column_value = self.columns[index];
                return .{
                    .name = self.names[index],
                    .dtype = column_value.dtype,
                    .rows = column_value.rows,
                    .nullable = column_value.nullable(),
                    .null_count = column_value.nullCount(),
                    .valid_count = column_value.validCount(),
                    .data_nbytes = column_value.dataNbytes(),
                    .validity_nbytes = column_value.validityNbytes(),
                    .total_nbytes = column_value.totalNbytes(),
                    .device = column_value.device,
                };
            }

            pub fn columnSchema(self: DeviceDataFrameView, name: []const u8) DataError!DeviceColumnSchema {
                const index = self.columnIndex(name) orelse return error.ColumnNotFound;
                return self.columnSchemaAt(index) catch unreachable;
            }

            pub fn columnSchemas(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]DeviceColumnSchema {
                const out = try allocator.alloc(DeviceColumnSchema, self.columns.len);
                for (out, 0..) |*slot, index| slot.* = self.columnSchemaAt(index) catch unreachable;
                return out;
            }

            pub fn schema(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]DeviceColumnSchema {
                return self.columnSchemas(allocator);
            }

            pub fn schemaSummary(self: DeviceDataFrameView, allocator: std.mem.Allocator) std.mem.Allocator.Error![]DeviceColumnSchema {
                return self.columnSchemas(allocator);
            }

            pub fn schemaEquals(self: DeviceDataFrameView, other: DeviceDataFrameView) bool {
                if (self.names.len != other.names.len) return false;
                for (self.names, other.names, self.columns, other.columns) |left_name, right_name, left_column, right_column| {
                    if (!std.mem.eql(u8, left_name, right_name)) return false;
                    if (left_column.dtype != right_column.dtype) return false;
                    if (left_column.nullable() != right_column.nullable()) return false;
                }
                return true;
            }

            pub const sameSchema = schemaEquals;
            pub const schemaCompatible = schemaEquals;

            pub fn isEmpty(self: DeviceDataFrameView) bool {
                return self.rows == 0 or self.columns.len == 0;
            }

            pub fn isNonEmpty(self: DeviceDataFrameView) bool {
                return !self.isEmpty();
            }

            pub fn hasRows(self: DeviceDataFrameView) bool {
                return self.rows != 0;
            }

            pub fn hasColumns(self: DeviceDataFrameView) bool {
                return self.columns.len != 0;
            }

            pub fn isCpu(self: DeviceDataFrameView) bool {
                return self.device.isCpu();
            }

            pub fn isCuda(self: DeviceDataFrameView) bool {
                return self.device.isCuda();
            }

            pub fn isMps(self: DeviceDataFrameView) bool {
                return self.device.isMps();
            }

            pub fn isDeviceBacked(self: DeviceDataFrameView) bool {
                return !self.isCpu();
            }

            pub fn deviceBackendName(self: DeviceDataFrameView) []const u8 {
                return self.device.backendName();
            }

            pub fn sameDevice(self: DeviceDataFrameView, other: DeviceDataFrameView) bool {
                return self.device.sameDevice(other.device);
            }

            pub fn sameShape(self: DeviceDataFrameView, other: DeviceDataFrameView) bool {
                return self.rows == other.rows and self.columns.len == other.columns.len;
            }

            pub fn shapeEquals(self: DeviceDataFrameView, rows: usize, columns: usize) bool {
                return self.rows == rows and self.columns.len == columns;
            }

            pub fn hasShape(self: DeviceDataFrameView, rows: usize, columns: usize) bool {
                return self.shapeEquals(rows, columns);
            }

            pub fn sameHeight(self: DeviceDataFrameView, other: DeviceDataFrameView) bool {
                return self.rows == other.rows;
            }

            pub fn sameWidth(self: DeviceDataFrameView, other: DeviceDataFrameView) bool {
                return self.columns.len == other.columns.len;
            }

            pub fn columnNames(self: DeviceDataFrameView) []const []const u8 {
                return self.names;
            }

            pub fn columnLabels(self: DeviceDataFrameView) []const []const u8 {
                return self.columnNames();
            }

            pub fn columnNamesUnique(self: DeviceDataFrameView) bool {
                for (self.names, 0..) |name, index| {
                    for (self.names[0..index]) |previous| {
                        if (std.mem.eql(u8, name, previous)) return false;
                    }
                }
                return true;
            }

            pub fn hasDuplicateColumnNames(self: DeviceDataFrameView) bool {
                return !self.columnNamesUnique();
            }

            pub fn duplicateColumnNameCount(self: DeviceDataFrameView) usize {
                var duplicates: usize = 0;
                for (self.names, 0..) |name, index| {
                    for (self.names[0..index]) |previous| {
                        if (std.mem.eql(u8, name, previous)) {
                            duplicates += 1;
                            break;
                        }
                    }
                }
                return duplicates;
            }

            pub fn columnIndex(self: DeviceDataFrameView, name: []const u8) ?usize {
                for (self.names, 0..) |existing, i| {
                    if (std.mem.eql(u8, existing, name)) return i;
                }
                return null;
            }

            pub fn hasColumn(self: DeviceDataFrameView, name: []const u8) bool {
                return self.columnIndex(name) != null;
            }

            pub fn hasAllColumns(self: DeviceDataFrameView, names: []const []const u8) bool {
                for (names) |name| {
                    if (!self.hasColumn(name)) return false;
                }
                return true;
            }

            pub fn hasAnyColumn(self: DeviceDataFrameView, names: []const []const u8) bool {
                for (names) |name| {
                    if (self.hasColumn(name)) return true;
                }
                return false;
            }

            pub fn column(self: DeviceDataFrameView, name: []const u8) DataError!DeviceColumnView {
                const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
                return self.columns[idx];
            }

            pub fn columnView(self: DeviceDataFrameView, name: []const u8) DataError!DeviceColumnView {
                return self.column(name);
            }

            pub fn columnAt(self: DeviceDataFrameView, index: usize) DeviceDataFrameViewError!DeviceColumnView {
                if (index >= self.columns.len) return error.IndexOutOfBounds;
                return self.columns[index];
            }

            pub fn columnViewAt(self: DeviceDataFrameView, index: usize) DeviceDataFrameViewError!DeviceColumnView {
                return self.columnAt(index);
            }

            pub fn columnNameAt(self: DeviceDataFrameView, index: usize) DeviceDataFrameViewError![]const u8 {
                if (index >= self.names.len) return error.IndexOutOfBounds;
                return self.names[index];
            }

            pub fn columnDType(self: DeviceDataFrameView, name: []const u8) DataError!array_mod.DType {
                const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
                return self.columns[idx].dtype;
            }

            pub fn columnDTypeAt(self: DeviceDataFrameView, index: usize) DeviceDataFrameViewError!array_mod.DType {
                const column_value = try self.columnAt(index);
                return column_value.dtype;
            }
        };
    };
}
