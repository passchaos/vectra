const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceClipOptions = options_mod.DeviceClipOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const ClipMetrics = struct {
    allocator: std.mem.Allocator,
    clipped: []f64,
    below: []bool,
    above: []bool,
    in_range: []bool,
    validity: []bool,

    pub fn deinit(self: *ClipMetrics) void {
        self.allocator.free(self.clipped);
        self.allocator.free(self.below);
        self.allocator.free(self.above);
        self.allocator.free(self.in_range);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ClipSummaryMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    mean_clipped: []f64,
    clipped_rates: []f64,
    below_rates: []f64,
    above_rates: []f64,
    in_range_rates: []f64,
    validity: []bool,

    pub fn deinit(self: *ClipSummaryMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.mean_clipped);
        self.allocator.free(self.clipped_rates);
        self.allocator.free(self.below_rates);
        self.allocator.free(self.above_rates);
        self.allocator.free(self.in_range_rates);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ClipProfileColumnCount = 4;

pub fn clipProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ClipProfileColumnCount][]const u8 {
    var names: [ClipProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "clipped", "below", "above", "in_range" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingClipProfileColumnCount = 6;

pub fn rollingClipProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingClipProfileColumnCount][]const u8 {
    var names: [RollingClipProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{
        "rolling_clip_count",
        "rolling_mean_clipped",
        "rolling_clipped_rate",
        "rolling_clip_below_rate",
        "rolling_clip_above_rate",
        "rolling_clip_in_range_rate",
    };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingClipProfileColumnCount = 6;

pub fn expandingClipProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingClipProfileColumnCount][]const u8 {
    var names: [ExpandingClipProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{
        "expanding_clip_count",
        "expanding_mean_clipped",
        "expanding_clipped_rate",
        "expanding_clip_below_rate",
        "expanding_clip_above_rate",
        "expanding_clip_in_range_rate",
    };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values: []const f64, maybe_validity: ?[]const bool, lower: f64, upper: f64) error{ InvalidShape, LengthMismatch }!void {
    if (lower > upper) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn clippedValue(value: f64, lower: f64, upper: f64) f64 {
    return @min(@max(value, lower), upper);
}

fn classify(value: f64, lower: f64, upper: f64) struct { clipped: f64, below: bool, above: bool, in_range: bool } {
    const below = value < lower;
    const above = value > upper;
    return .{
        .clipped = clippedValue(value, lower, upper),
        .below = below,
        .above = above,
        .in_range = !below and !above,
    };
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!ClipSummaryMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const mean_clipped = try allocator.alloc(f64, rows);
    errdefer allocator.free(mean_clipped);
    const clipped_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(clipped_rates);
    const below_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(below_rates);
    const above_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(above_rates);
    const in_range_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(in_range_rates);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .counts = counts,
        .mean_clipped = mean_clipped,
        .clipped_rates = clipped_rates,
        .below_rates = below_rates,
        .above_rates = above_rates,
        .in_range_rates = in_range_rates,
        .validity = validity,
    };
}

fn addClass(c: anytype, count: *usize, clipped_sum: *f64, below_count: *usize, above_count: *usize, in_range_count: *usize) void {
    clipped_sum.* += c.clipped;
    if (c.below) {
        below_count.* += 1;
    } else if (c.above) {
        above_count.* += 1;
    } else {
        in_range_count.* += 1;
    }
    count.* += 1;
}

fn removeClass(c: anytype, count: *usize, clipped_sum: *f64, below_count: *usize, above_count: *usize, in_range_count: *usize) void {
    clipped_sum.* -= c.clipped;
    if (c.below) {
        below_count.* -= 1;
    } else if (c.above) {
        above_count.* -= 1;
    } else {
        in_range_count.* -= 1;
    }
    count.* -= 1;
}

fn writeSummary(row: usize, min_periods: usize, count: usize, clipped_sum: f64, below_count: usize, above_count: usize, in_range_count: usize, out: ClipSummaryMetrics) void {
    out.counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        const clipped_count = below_count + above_count;
        out.mean_clipped[row] = clipped_sum / n;
        out.clipped_rates[row] = @as(f64, @floatFromInt(clipped_count)) / n;
        out.below_rates[row] = @as(f64, @floatFromInt(below_count)) / n;
        out.above_rates[row] = @as(f64, @floatFromInt(above_count)) / n;
        out.in_range_rates[row] = @as(f64, @floatFromInt(in_range_count)) / n;
    } else {
        out.mean_clipped[row] = 0;
        out.clipped_rates[row] = 0;
        out.below_rates[row] = 0;
        out.above_rates[row] = 0;
        out.in_range_rates[row] = 0;
    }
}

pub fn clipProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    lower: f64,
    upper: f64,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ClipMetrics {
    try validate(values, maybe_validity, lower, upper);

    const rows = values.len;
    const clipped = try allocator.alloc(f64, rows);
    errdefer allocator.free(clipped);
    const below = try allocator.alloc(bool, rows);
    errdefer allocator.free(below);
    const above = try allocator.alloc(bool, rows);
    errdefer allocator.free(above);
    const in_range = try allocator.alloc(bool, rows);
    errdefer allocator.free(in_range);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row);
        validity[row] = valid;
        if (valid) {
            const c = classify(value, lower, upper);
            clipped[row] = c.clipped;
            below[row] = c.below;
            above[row] = c.above;
            in_range[row] = c.in_range;
        } else {
            clipped[row] = 0;
            below[row] = false;
            above[row] = false;
            in_range[row] = false;
        }
    }

    return .{
        .allocator = allocator,
        .clipped = clipped,
        .below = below,
        .above = above,
        .in_range = in_range,
        .validity = validity,
    };
}

pub fn rollingClipProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    lower: f64,
    upper: f64,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ClipSummaryMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity, lower, upper);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();

    var count: usize = 0;
    var clipped_sum: f64 = 0;
    var below_count: usize = 0;
    var above_count: usize = 0;
    var in_range_count: usize = 0;

    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            addClass(classify(value, lower, upper), &count, &clipped_sum, &below_count, &above_count, &in_range_count);
        }

        if (row >= window) {
            const evict_row = row - window;
            if (rowValid(maybe_validity, evict_row)) {
                removeClass(classify(values[evict_row], lower, upper), &count, &clipped_sum, &below_count, &above_count, &in_range_count);
            }
        }

        writeSummary(row, min_periods, count, clipped_sum, below_count, above_count, in_range_count, out);
    }

    return out;
}

pub fn expandingClipProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    lower: f64,
    upper: f64,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ClipSummaryMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity, lower, upper);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();

    var count: usize = 0;
    var clipped_sum: f64 = 0;
    var below_count: usize = 0;
    var above_count: usize = 0;
    var in_range_count: usize = 0;

    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            addClass(classify(value, lower, upper), &count, &clipped_sum, &below_count, &above_count, &in_range_count);
        }

        writeSummary(row, min_periods, count, clipped_sum, below_count, above_count, in_range_count, out);
    }

    return out;
}

pub fn clipProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceClipOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ClipProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| clipProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| clipProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| clipProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| clipProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| clipProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| clipProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| clipProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| clipProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| clipProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| clipProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| clipProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| clipProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| clipProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn clipProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceClipOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ClipProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try clipProfile(allocator, values, maybe_validity, options_value.lower, options_value.upper);
    defer metrics.deinit();

    var columns: [ClipProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.clipped, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.below, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.above, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.in_range, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingClipProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    clip_options: DeviceClipOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingClipProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingClipProfileColumnsTyped(i8, allocator, typed, clip_options, options_value, device_value),
        .i16 => |typed| rollingClipProfileColumnsTyped(i16, allocator, typed, clip_options, options_value, device_value),
        .i32 => |typed| rollingClipProfileColumnsTyped(i32, allocator, typed, clip_options, options_value, device_value),
        .i64 => |typed| rollingClipProfileColumnsTyped(i64, allocator, typed, clip_options, options_value, device_value),
        .u8 => |typed| rollingClipProfileColumnsTyped(u8, allocator, typed, clip_options, options_value, device_value),
        .u16 => |typed| rollingClipProfileColumnsTyped(u16, allocator, typed, clip_options, options_value, device_value),
        .u32 => |typed| rollingClipProfileColumnsTyped(u32, allocator, typed, clip_options, options_value, device_value),
        .u64 => |typed| rollingClipProfileColumnsTyped(u64, allocator, typed, clip_options, options_value, device_value),
        .usize => |typed| rollingClipProfileColumnsTyped(usize, allocator, typed, clip_options, options_value, device_value),
        .isize => |typed| rollingClipProfileColumnsTyped(isize, allocator, typed, clip_options, options_value, device_value),
        .f16 => |typed| rollingClipProfileColumnsTyped(f16, allocator, typed, clip_options, options_value, device_value),
        .f32 => |typed| rollingClipProfileColumnsTyped(f32, allocator, typed, clip_options, options_value, device_value),
        .f64 => |typed| rollingClipProfileColumnsTyped(f64, allocator, typed, clip_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingClipProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    clip_options: DeviceClipOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingClipProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try rollingClipProfile(allocator, values, maybe_validity, clip_options.lower, clip_options.upper, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingClipProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_clipped, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.clipped_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.below_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.above_rates, metrics.validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.in_range_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingClipProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    clip_options: DeviceClipOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingClipProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingClipProfileColumnsTyped(i8, allocator, typed, clip_options, options_value, device_value),
        .i16 => |typed| expandingClipProfileColumnsTyped(i16, allocator, typed, clip_options, options_value, device_value),
        .i32 => |typed| expandingClipProfileColumnsTyped(i32, allocator, typed, clip_options, options_value, device_value),
        .i64 => |typed| expandingClipProfileColumnsTyped(i64, allocator, typed, clip_options, options_value, device_value),
        .u8 => |typed| expandingClipProfileColumnsTyped(u8, allocator, typed, clip_options, options_value, device_value),
        .u16 => |typed| expandingClipProfileColumnsTyped(u16, allocator, typed, clip_options, options_value, device_value),
        .u32 => |typed| expandingClipProfileColumnsTyped(u32, allocator, typed, clip_options, options_value, device_value),
        .u64 => |typed| expandingClipProfileColumnsTyped(u64, allocator, typed, clip_options, options_value, device_value),
        .usize => |typed| expandingClipProfileColumnsTyped(usize, allocator, typed, clip_options, options_value, device_value),
        .isize => |typed| expandingClipProfileColumnsTyped(isize, allocator, typed, clip_options, options_value, device_value),
        .f16 => |typed| expandingClipProfileColumnsTyped(f16, allocator, typed, clip_options, options_value, device_value),
        .f32 => |typed| expandingClipProfileColumnsTyped(f32, allocator, typed, clip_options, options_value, device_value),
        .f64 => |typed| expandingClipProfileColumnsTyped(f64, allocator, typed, clip_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingClipProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    clip_options: DeviceClipOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingClipProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try expandingClipProfile(allocator, values, maybe_validity, clip_options.lower, clip_options.upper, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingClipProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.mean_clipped, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.clipped_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.below_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.above_rates, metrics.validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.in_range_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
