const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceLagOptions = options_mod.DeviceLagOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const ShiftMetrics = struct {
    allocator: std.mem.Allocator,
    shifted: []f64,
    diff: []f64,
    pct_change: []f64,
    shift_validity: []bool,
    change_validity: []bool,

    pub fn deinit(self: *ShiftMetrics) void {
        self.allocator.free(self.shifted);
        self.allocator.free(self.diff);
        self.allocator.free(self.pct_change);
        self.allocator.free(self.shift_validity);
        self.allocator.free(self.change_validity);
        self.* = undefined;
    }
};

pub const LagProfileColumnCount = 3;

pub fn lagProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![LagProfileColumnCount][]const u8 {
    var names: [LagProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "lag", "diff", "pct_change" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const LeadProfileColumnCount = 3;

pub fn leadProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![LeadProfileColumnCount][]const u8 {
    var names: [LeadProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "lead", "forward_diff", "forward_pct_change" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values: []const f64, maybe_validity: ?[]const bool, periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn allocMetrics(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!ShiftMetrics {
    const shifted = try allocator.alloc(f64, rows);
    errdefer allocator.free(shifted);
    const diff = try allocator.alloc(f64, rows);
    errdefer allocator.free(diff);
    const pct_change = try allocator.alloc(f64, rows);
    errdefer allocator.free(pct_change);
    const shift_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(shift_validity);
    const change_validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(change_validity);
    return .{
        .allocator = allocator,
        .shifted = shifted,
        .diff = diff,
        .pct_change = pct_change,
        .shift_validity = shift_validity,
        .change_validity = change_validity,
    };
}

pub fn lagProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ShiftMetrics {
    try validate(values, maybe_validity, periods);
    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    for (values, 0..) |value, row| {
        if (row < periods) {
            out.shifted[row] = 0;
            out.diff[row] = 0;
            out.pct_change[row] = 0;
            out.shift_validity[row] = false;
            out.change_validity[row] = false;
            continue;
        }

        const lag_row = row - periods;
        const lag_valid = rowValid(maybe_validity, lag_row);
        const previous = values[lag_row];
        out.shifted[row] = previous;
        out.shift_validity[row] = lag_valid;

        const can_change = rowValid(maybe_validity, row) and lag_valid;
        out.change_validity[row] = can_change;
        if (can_change) {
            const diff = value - previous;
            out.diff[row] = diff;
            out.pct_change[row] = if (previous == 0) std.math.nan(f64) else diff / previous;
        } else {
            out.diff[row] = 0;
            out.pct_change[row] = 0;
        }
    }

    return out;
}

pub fn leadProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ShiftMetrics {
    try validate(values, maybe_validity, periods);
    var out = try allocMetrics(allocator, values.len);
    errdefer out.deinit();

    for (values, 0..) |value, row| {
        const lead_row = row + periods;
        if (lead_row >= values.len) {
            out.shifted[row] = 0;
            out.diff[row] = 0;
            out.pct_change[row] = 0;
            out.shift_validity[row] = false;
            out.change_validity[row] = false;
            continue;
        }

        const lead_valid = rowValid(maybe_validity, lead_row);
        const future = values[lead_row];
        out.shifted[row] = future;
        out.shift_validity[row] = lead_valid;

        const can_change = rowValid(maybe_validity, row) and lead_valid;
        out.change_validity[row] = can_change;
        if (can_change) {
            const diff = future - value;
            out.diff[row] = diff;
            out.pct_change[row] = if (value == 0) std.math.nan(f64) else diff / value;
        } else {
            out.diff[row] = 0;
            out.pct_change[row] = 0;
        }
    }

    return out;
}

pub fn lagProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![LagProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| lagProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| lagProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| lagProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| lagProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| lagProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| lagProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| lagProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| lagProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| lagProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| lagProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| lagProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| lagProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| lagProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn lagProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![LagProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try lagProfile(allocator, values, maybe_validity, options_value.periods);
    defer metrics.deinit();

    var columns: [LagProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.shifted, metrics.shift_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.diff, metrics.change_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.pct_change, metrics.change_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn leadProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![LeadProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| leadProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| leadProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| leadProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| leadProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| leadProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| leadProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| leadProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| leadProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| leadProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| leadProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| leadProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| leadProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| leadProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn leadProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceLagOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![LeadProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try leadProfile(allocator, values, maybe_validity, options_value.periods);
    defer metrics.deinit();

    var columns: [LeadProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.shifted, metrics.shift_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.diff, metrics.change_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.pct_change, metrics.change_validity, device_value);
    initialized += 1;
    return columns;
}
