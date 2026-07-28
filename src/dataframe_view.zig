const std = @import("std");
const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

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

    pub fn width(self: DeviceDataFrameView) usize {
        return self.columns.len;
    }

    pub fn shape(self: DeviceDataFrameView) struct { rows: usize, cols: usize } {
        return .{ .rows = self.rows, .cols = self.columns.len };
    }

    pub fn columnIndex(self: DeviceDataFrameView, name: []const u8) ?usize {
        for (self.names, 0..) |existing, i| {
            if (std.mem.eql(u8, existing, name)) return i;
        }
        return null;
    }

    pub fn column(self: DeviceDataFrameView, name: []const u8) series_mod.DataError!DeviceColumnView {
        const idx = self.columnIndex(name) orelse return error.ColumnNotFound;
        return self.columns[idx];
    }
};
