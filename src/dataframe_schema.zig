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

    pub fn nullableColumn(self: @This()) bool {
        return self.nullable;
    }

    pub fn hasNulls(self: @This()) bool {
        return self.null_count != 0;
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

    pub fn dataMemoryUsage(self: @This()) usize {
        return self.data_nbytes;
    }

    pub fn validityMemoryUsage(self: @This()) usize {
        return self.validity_nbytes;
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
};
