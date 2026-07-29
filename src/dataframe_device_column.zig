const std = @import("std");
const array_mod = @import("array.zig");
const column_arrow_mod = @import("dataframe_device_column_arrow.zig");
const column_ops_mod = @import("dataframe_device_column_ops.zig");
const column_sort_mod = @import("dataframe_device_column_sort.zig");
const dataframe_typed_column_mod = @import("dataframe_device_typed_column.zig");
const dataframe_view_mod = @import("dataframe_view.zig");

const DeviceDType = array_mod.DType;
const DeviceColumnView = dataframe_view_mod.DeviceColumnView;

pub const DeviceTypedColumn = dataframe_typed_column_mod.DeviceTypedColumn;
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

    pub fn fromSlice(comptime T: type, allocator: std.mem.Allocator, values: []const T, device_value: array_mod.Device) array_mod.ArrayError!DeviceColumn {
        const tag = comptime DeviceDType.of(T);
        const typed = try DeviceTypedColumn(T).fromSlice(allocator, values, device_value);
        return @unionInit(DeviceColumn, @tagName(tag), typed);
    }

    pub fn fromSliceWithValidity(
        comptime T: type,
        allocator: std.mem.Allocator,
        values: []const T,
        validity_values: []const bool,
        device_value: array_mod.Device,
    ) array_mod.ArrayError!DeviceColumn {
        const tag = comptime DeviceDType.of(T);
        const typed = try DeviceTypedColumn(T).fromSliceWithValidity(allocator, values, validity_values, device_value);
        return @unionInit(DeviceColumn, @tagName(tag), typed);
    }

    pub fn deinit(self: *DeviceColumn) void {
        switch (self.*) {
            inline else => |*typed| typed.deinit(),
        }
        self.* = undefined;
    }

    pub fn len(self: DeviceColumn) usize {
        return switch (self) {
            inline else => |typed| typed.len(),
        };
    }

    pub fn dtype(self: DeviceColumn) DeviceDType {
        return std.meta.activeTag(self);
    }

    pub fn device(self: DeviceColumn) array_mod.Device {
        return switch (self) {
            inline else => |typed| typed.device(),
        };
    }

    pub fn nullable(self: DeviceColumn) bool {
        return switch (self) {
            inline else => |typed| typed.nullable(),
        };
    }

    pub fn hasNulls(self: DeviceColumn) bool {
        return switch (self) {
            inline else => |typed| typed.hasNulls(),
        };
    }

    pub fn nullCount(self: DeviceColumn) usize {
        return switch (self) {
            inline else => |typed| typed.null_count,
        };
    }

    pub fn dataNbytes(self: DeviceColumn) usize {
        return switch (self) {
            inline else => |typed| typed.dataNbytes(),
        };
    }

    pub fn view(self: DeviceColumn) DeviceColumnView {
        return switch (self) {
            inline else => |typed| typed.view(),
        };
    }

    pub fn clone(self: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.clone()),
        };
    }

    pub fn to(self: DeviceColumn, device_value: array_mod.Device) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.to(device_value)),
        };
    }

    pub fn cpu(self: DeviceColumn) array_mod.ArrayError!DeviceColumn {
        return self.to(.cpu);
    }

    pub fn cuda(self: DeviceColumn, index: usize) array_mod.ArrayError!DeviceColumn {
        return self.to(array_mod.Device.cuda(index));
    }

    pub fn mps(self: DeviceColumn, index: usize) array_mod.ArrayError!DeviceColumn {
        return self.to(array_mod.Device.mps(index));
    }

    pub fn sliceRows(self: DeviceColumn, start: usize, stop: usize) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.sliceRows(start, stop)),
        };
    }

    pub fn take(self: DeviceColumn, row_indices: []const usize) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.take(row_indices)),
        };
    }

    pub fn takeOptional(self: DeviceColumn, row_indices: []const ?usize) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.takeOptional(row_indices)),
        };
    }

    pub fn filter(self: DeviceColumn, mask: []const bool) array_mod.ArrayError!DeviceColumn {
        return switch (self) {
            inline else => |typed, tag| @unionInit(DeviceColumn, @tagName(tag), try typed.filter(mask)),
        };
    }

    pub const argsort = column_sort_mod.argsort;

    pub const binary = column_ops_mod.binary;
    pub const add = column_ops_mod.add;
    pub const sub = column_ops_mod.sub;
    pub const mul = column_ops_mod.mul;
    pub const div = column_ops_mod.div;
    pub const binaryScalar = column_ops_mod.binaryScalar;
    pub const addScalar = column_ops_mod.addScalar;
    pub const subScalar = column_ops_mod.subScalar;
    pub const mulScalar = column_ops_mod.mulScalar;
    pub const divScalar = column_ops_mod.divScalar;
    pub const compare = column_ops_mod.compare;
    pub const equal = column_ops_mod.equal;
    pub const notEqual = column_ops_mod.notEqual;
    pub const greater = column_ops_mod.greater;
    pub const greaterEqual = column_ops_mod.greaterEqual;
    pub const less = column_ops_mod.less;
    pub const lessEqual = column_ops_mod.lessEqual;
    pub const compareScalar = column_ops_mod.compareScalar;
    pub const equalScalar = column_ops_mod.equalScalar;
    pub const notEqualScalar = column_ops_mod.notEqualScalar;
    pub const greaterScalar = column_ops_mod.greaterScalar;
    pub const greaterEqualScalar = column_ops_mod.greaterEqualScalar;
    pub const lessScalar = column_ops_mod.lessScalar;
    pub const lessEqualScalar = column_ops_mod.lessEqualScalar;

    pub const arrowDataType = column_arrow_mod.arrowDataType;
    pub const toArrowArray = column_arrow_mod.toArrowArray;
};

pub const DeviceColumnDef = struct {
    name: []const u8,
    data: DeviceColumn,
};

pub const argsortTypedColumn = column_sort_mod.argsortTypedColumn;
