const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceDrawdownOptions = options_mod.DeviceDrawdownOptions;
const DeviceExtremaOptions = options_mod.DeviceExtremaOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const DrawdownMetrics = struct {
    allocator: std.mem.Allocator,
    running_peak: []f64,
    drawdown: []f64,
    drawdown_pct: []f64,
    validity: []bool,

    pub fn deinit(self: *DrawdownMetrics) void {
        self.allocator.free(self.running_peak);
        self.allocator.free(self.drawdown);
        self.allocator.free(self.drawdown_pct);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const RollingDrawdownMetrics = struct {
    allocator: std.mem.Allocator,
    peaks: []f64,
    drawdowns: []f64,
    drawdown_pcts: []f64,
    peak_ages: []i64,
    validity: []bool,

    pub fn deinit(self: *RollingDrawdownMetrics) void {
        self.allocator.free(self.peaks);
        self.allocator.free(self.drawdowns);
        self.allocator.free(self.drawdown_pcts);
        self.allocator.free(self.peak_ages);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const ExtremaMetrics = struct {
    allocator: std.mem.Allocator,
    running_low: []f64,
    running_high: []f64,
    new_low: []bool,
    new_high: []bool,
    validity: []bool,

    pub fn deinit(self: *ExtremaMetrics) void {
        self.allocator.free(self.running_low);
        self.allocator.free(self.running_high);
        self.allocator.free(self.new_low);
        self.allocator.free(self.new_high);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const DrawdownProfileColumnCount = 3;

pub fn drawdownProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![DrawdownProfileColumnCount][]const u8 {
    var names: [DrawdownProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "running_peak", "drawdown", "drawdown_pct" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingDrawdownProfileColumnCount = 4;

pub fn rollingDrawdownProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingDrawdownProfileColumnCount][]const u8 {
    var names: [RollingDrawdownProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_peak", "rolling_drawdown", "rolling_drawdown_pct", "rolling_peak_age" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExtremaProfileColumnCount = 4;

pub fn extremaProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExtremaProfileColumnCount][]const u8 {
    var names: [ExtremaProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "running_low", "running_high", "new_low", "new_high" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

fn validateLength(values: []const f64, maybe_validity: ?[]const bool) error{LengthMismatch}!void {
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

pub fn drawdownProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!DrawdownMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    const rows = values.len;
    const running_peak = try allocator.alloc(f64, rows);
    errdefer allocator.free(running_peak);
    const drawdown = try allocator.alloc(f64, rows);
    errdefer allocator.free(drawdown);
    const drawdown_pct = try allocator.alloc(f64, rows);
    errdefer allocator.free(drawdown_pct);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    var valid_count: usize = 0;
    var peak: f64 = 0;
    // Drawdown is order-sensitive: null rows do not advance the running peak
    // and their derived metrics are null, while output remains row-aligned.
    for (values, 0..) |current, row| {
        if (rowValid(maybe_validity, row)) {
            if (valid_count == 0 or current > peak) peak = current;
            valid_count += 1;

            const has_enough = valid_count >= min_periods;
            validity[row] = has_enough;
            if (has_enough) {
                const dd = current - peak;
                running_peak[row] = peak;
                drawdown[row] = dd;
                drawdown_pct[row] = if (peak == 0) std.math.nan(f64) else dd / peak;
            } else {
                running_peak[row] = 0;
                drawdown[row] = 0;
                drawdown_pct[row] = 0;
            }
        } else {
            validity[row] = false;
            running_peak[row] = 0;
            drawdown[row] = 0;
            drawdown_pct[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .running_peak = running_peak,
        .drawdown = drawdown,
        .drawdown_pct = drawdown_pct,
        .validity = validity,
    };
}

pub fn rollingDrawdownProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!RollingDrawdownMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    const rows = values.len;
    const peaks = try allocator.alloc(f64, rows);
    errdefer allocator.free(peaks);
    const drawdowns = try allocator.alloc(f64, rows);
    errdefer allocator.free(drawdowns);
    const drawdown_pcts = try allocator.alloc(f64, rows);
    errdefer allocator.free(drawdown_pcts);
    const peak_ages = try allocator.alloc(i64, rows);
    errdefer allocator.free(peak_ages);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    // Equal highs choose the most recent row so peak_age answers "rows since the
    // latest peak that still defines this window".
    for (values, 0..) |current, row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var peak: f64 = 0;
        var peak_row: usize = start;
        for (start..row + 1) |window_row| {
            if (!rowValid(maybe_validity, window_row)) continue;
            const x = values[window_row];
            if (count == 0 or x >= peak) {
                peak = x;
                peak_row = window_row;
            }
            count += 1;
        }

        const has_enough = rowValid(maybe_validity, row) and count >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            const dd = current - peak;
            peaks[row] = peak;
            drawdowns[row] = dd;
            drawdown_pcts[row] = if (peak == 0) std.math.nan(f64) else dd / peak;
            peak_ages[row] = @intCast(row - peak_row);
        } else {
            peaks[row] = 0;
            drawdowns[row] = 0;
            drawdown_pcts[row] = 0;
            peak_ages[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .peaks = peaks,
        .drawdowns = drawdowns,
        .drawdown_pcts = drawdown_pcts,
        .peak_ages = peak_ages,
        .validity = validity,
    };
}

pub fn extremaProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!ExtremaMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validateLength(values, maybe_validity);

    const rows = values.len;
    const running_low = try allocator.alloc(f64, rows);
    errdefer allocator.free(running_low);
    const running_high = try allocator.alloc(f64, rows);
    errdefer allocator.free(running_high);
    const new_low = try allocator.alloc(bool, rows);
    errdefer allocator.free(new_low);
    const new_high = try allocator.alloc(bool, rows);
    errdefer allocator.free(new_high);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);

    var seen: usize = 0;
    var low: f64 = 0;
    var high: f64 = 0;
    for (values, 0..) |value, row| {
        if (!rowValid(maybe_validity, row)) {
            running_low[row] = 0;
            running_high[row] = 0;
            new_low[row] = false;
            new_high[row] = false;
            validity[row] = false;
            continue;
        }

        const first = seen == 0;
        const is_new_low = first or value < low;
        const is_new_high = first or value > high;
        if (first) {
            low = value;
            high = value;
        } else {
            if (is_new_low) low = value;
            if (is_new_high) high = value;
        }
        seen += 1;
        const has_enough = seen >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            running_low[row] = low;
            running_high[row] = high;
            new_low[row] = is_new_low;
            new_high[row] = is_new_high;
        } else {
            running_low[row] = 0;
            running_high[row] = 0;
            new_low[row] = false;
            new_high[row] = false;
        }
    }

    return .{
        .allocator = allocator,
        .running_low = running_low,
        .running_high = running_high,
        .new_low = new_low,
        .new_high = new_high,
        .validity = validity,
    };
}

pub fn rollingDrawdownProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingDrawdownProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingDrawdownProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingDrawdownProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingDrawdownProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingDrawdownProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingDrawdownProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingDrawdownProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingDrawdownProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingDrawdownProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingDrawdownProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingDrawdownProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingDrawdownProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingDrawdownProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingDrawdownProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingDrawdownProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingDrawdownProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try rollingDrawdownProfile(allocator, values, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingDrawdownProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.peaks, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.drawdowns, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.drawdown_pcts, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.peak_ages, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn drawdownProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceDrawdownOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![DrawdownProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| drawdownProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| drawdownProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| drawdownProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| drawdownProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| drawdownProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| drawdownProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| drawdownProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| drawdownProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| drawdownProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| drawdownProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| drawdownProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| drawdownProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| drawdownProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn drawdownProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceDrawdownOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![DrawdownProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try drawdownProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [DrawdownProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.running_peak, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.drawdown, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.drawdown_pct, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn extremaProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExtremaOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExtremaProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| extremaProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| extremaProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| extremaProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| extremaProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| extremaProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| extremaProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| extremaProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| extremaProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| extremaProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| extremaProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| extremaProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| extremaProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| extremaProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn extremaProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExtremaOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExtremaProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try extremaProfile(allocator, values, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExtremaProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.running_low, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.running_high, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.new_low, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.new_high, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
