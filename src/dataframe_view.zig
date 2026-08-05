const std = @import("std");
const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

pub const DeviceDataFrameViewError = series_mod.DataError || array_mod.ArrayError;

pub const DeviceColumnView = struct {
    dtype: array_mod.DType,
    rows: usize,
    device: array_mod.Device,
    data_ptr: u64,
    data_nbytes: usize,
    validity_ptr: ?u64 = null,
    validity_nbytes: usize = 0,
    null_count: usize = 0,
    validity_encoding: options_mod.DeviceValidityEncoding = .none,

    pub fn nullable(self: DeviceColumnView) bool {
        return self.validity_ptr != null;
    }

    pub fn hasNulls(self: DeviceColumnView) bool {
        return self.null_count != 0;
    }

    pub fn isDeviceBacked(self: DeviceColumnView) bool {
        return !self.device.isCpu();
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

    pub fn column(self: DeviceDataFrameView, name: []const u8) series_mod.DataError!DeviceColumnView {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx];
    }

    pub fn columnView(self: DeviceDataFrameView, name: []const u8) series_mod.DataError!DeviceColumnView {
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

    pub fn columnDType(self: DeviceDataFrameView, name: []const u8) series_mod.DataError!array_mod.DType {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx].dtype;
    }

    pub fn columnDTypeAt(self: DeviceDataFrameView, index: usize) DeviceDataFrameViewError!array_mod.DType {
        const column_value = try self.columnAt(index);
        return column_value.dtype;
    }
};
