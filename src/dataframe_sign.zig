const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const SignMetrics = struct {
    allocator: std.mem.Allocator,
    signs: []i64,
    flips: []bool,
    positive_streak: []i64,
    negative_streak: []i64,
    zero_streak: []i64,
    sign_validity: []bool,
    flip_validity: []bool,

    pub fn deinit(self: *SignMetrics) void {
        self.allocator.free(self.signs);
        self.allocator.free(self.flips);
        self.allocator.free(self.positive_streak);
        self.allocator.free(self.negative_streak);
        self.allocator.free(self.zero_streak);
        self.allocator.free(self.sign_validity);
        self.allocator.free(self.flip_validity);
        self.* = undefined;
    }
};

pub const SignSummaryMetrics = struct {
    allocator: std.mem.Allocator,
    counts: []i64,
    positive_rates: []f64,
    negative_rates: []f64,
    zero_rates: []f64,
    flip_rates: []f64,
    validity: []bool,

    pub fn deinit(self: *SignSummaryMetrics) void {
        self.allocator.free(self.counts);
        self.allocator.free(self.positive_rates);
        self.allocator.free(self.negative_rates);
        self.allocator.free(self.zero_rates);
        self.allocator.free(self.flip_rates);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const SignProfileColumnCount = 5;

pub fn signProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![SignProfileColumnCount][]const u8 {
    var names: [SignProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "sign", "sign_flip", "positive_streak", "negative_streak", "zero_streak" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const RollingSignProfileColumnCount = 5;

pub fn rollingSignProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![RollingSignProfileColumnCount][]const u8 {
    var names: [RollingSignProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "rolling_sign_count", "rolling_positive_rate", "rolling_negative_rate", "rolling_zero_rate", "rolling_sign_flip_rate" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

pub const ExpandingSignProfileColumnCount = 5;

pub fn expandingSignProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![ExpandingSignProfileColumnCount][]const u8 {
    var names: [ExpandingSignProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "expanding_sign_count", "expanding_positive_rate", "expanding_negative_rate", "expanding_zero_rate", "expanding_sign_flip_rate" };
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

fn signOf(value: f64) i64 {
    return if (value > 0) 1 else if (value < 0) -1 else 0;
}

fn allocSummary(allocator: std.mem.Allocator, rows: usize) std.mem.Allocator.Error!SignSummaryMetrics {
    const counts = try allocator.alloc(i64, rows);
    errdefer allocator.free(counts);
    const positive_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(positive_rates);
    const negative_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(negative_rates);
    const zero_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(zero_rates);
    const flip_rates = try allocator.alloc(f64, rows);
    errdefer allocator.free(flip_rates);
    const validity = try allocator.alloc(bool, rows);
    errdefer allocator.free(validity);
    return .{
        .allocator = allocator,
        .counts = counts,
        .positive_rates = positive_rates,
        .negative_rates = negative_rates,
        .zero_rates = zero_rates,
        .flip_rates = flip_rates,
        .validity = validity,
    };
}

fn writeSummary(row: usize, min_periods: usize, count: usize, positive_count: usize, negative_count: usize, zero_count: usize, flip_count: usize, out: SignSummaryMetrics) void {
    out.counts[row] = @intCast(count);
    const has_enough = count >= min_periods;
    out.validity[row] = has_enough;
    if (has_enough) {
        const n: f64 = @floatFromInt(count);
        out.positive_rates[row] = @as(f64, @floatFromInt(positive_count)) / n;
        out.negative_rates[row] = @as(f64, @floatFromInt(negative_count)) / n;
        out.zero_rates[row] = @as(f64, @floatFromInt(zero_count)) / n;
        out.flip_rates[row] = @as(f64, @floatFromInt(flip_count)) / n;
    } else {
        out.positive_rates[row] = 0;
        out.negative_rates[row] = 0;
        out.zero_rates[row] = 0;
        out.flip_rates[row] = 0;
    }
}

fn computeSignEvents(allocator: std.mem.Allocator, values: []const f64, maybe_validity: ?[]const bool, periods: usize) !struct { signs: []i64, flips: []bool, sign_validity: []bool, flip_validity: []bool } {
    const signs = try allocator.alloc(i64, values.len);
    errdefer allocator.free(signs);
    const flips = try allocator.alloc(bool, values.len);
    errdefer allocator.free(flips);
    const sign_validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(sign_validity);
    const flip_validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(flip_validity);

    for (values, 0..) |value, row| {
        const valid = rowValid(maybe_validity, row);
        sign_validity[row] = valid;
        if (!valid) {
            signs[row] = 0;
            flips[row] = false;
            flip_validity[row] = false;
            continue;
        }

        const sign = signOf(value);
        signs[row] = sign;
        if (row < periods) {
            flips[row] = false;
            flip_validity[row] = false;
        } else {
            const previous_row = row - periods;
            const previous_valid = rowValid(maybe_validity, previous_row);
            flip_validity[row] = previous_valid;
            flips[row] = if (previous_valid) sign != signOf(values[previous_row]) else false;
        }
    }

    return .{ .signs = signs, .flips = flips, .sign_validity = sign_validity, .flip_validity = flip_validity };
}

pub fn signProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!SignMetrics {
    try validate(values, maybe_validity, periods);

    const events = try computeSignEvents(allocator, values, maybe_validity, periods);
    errdefer allocator.free(events.signs);
    errdefer allocator.free(events.flips);
    errdefer allocator.free(events.sign_validity);
    errdefer allocator.free(events.flip_validity);

    const positive_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(positive_streak);
    const negative_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(negative_streak);
    const zero_streak = try allocator.alloc(i64, values.len);
    errdefer allocator.free(zero_streak);

    var pos: i64 = 0;
    var neg: i64 = 0;
    var zero: i64 = 0;
    for (0..values.len) |row| {
        if (!events.sign_validity[row]) {
            positive_streak[row] = 0;
            negative_streak[row] = 0;
            zero_streak[row] = 0;
            pos = 0;
            neg = 0;
            zero = 0;
            continue;
        }

        switch (events.signs[row]) {
            1 => {
                pos += 1;
                neg = 0;
                zero = 0;
            },
            -1 => {
                neg += 1;
                pos = 0;
                zero = 0;
            },
            else => {
                zero += 1;
                pos = 0;
                neg = 0;
            },
        }
        positive_streak[row] = pos;
        negative_streak[row] = neg;
        zero_streak[row] = zero;
    }

    return .{
        .allocator = allocator,
        .signs = events.signs,
        .flips = events.flips,
        .positive_streak = positive_streak,
        .negative_streak = negative_streak,
        .zero_streak = zero_streak,
        .sign_validity = events.sign_validity,
        .flip_validity = events.flip_validity,
    };
}

pub fn rollingSignProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
    window: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!SignSummaryMetrics {
    if (window == 0) return error.InvalidShape;
    if (min_periods == 0 or min_periods > window) return error.InvalidShape;
    try validate(values, maybe_validity, periods);

    const events = try computeSignEvents(allocator, values, maybe_validity, periods);
    defer allocator.free(events.signs);
    defer allocator.free(events.flips);
    defer allocator.free(events.sign_validity);
    defer allocator.free(events.flip_validity);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();
    for (0..values.len) |row| {
        const start = if (row + 1 > window) row + 1 - window else 0;
        var count: usize = 0;
        var positive_count: usize = 0;
        var negative_count: usize = 0;
        var zero_count: usize = 0;
        var flip_count: usize = 0;
        for (start..row + 1) |window_row| {
            if (!events.sign_validity[window_row]) continue;
            switch (events.signs[window_row]) {
                1 => positive_count += 1,
                -1 => negative_count += 1,
                else => zero_count += 1,
            }
            if (events.flip_validity[window_row] and events.flips[window_row]) flip_count += 1;
            count += 1;
        }
        writeSummary(row, min_periods, count, positive_count, negative_count, zero_count, flip_count, out);
    }
    return out;
}

pub fn expandingSignProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    periods: usize,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!SignSummaryMetrics {
    if (min_periods == 0) return error.InvalidShape;
    try validate(values, maybe_validity, periods);

    var out = try allocSummary(allocator, values.len);
    errdefer out.deinit();

    var count: usize = 0;
    var positive_count: usize = 0;
    var negative_count: usize = 0;
    var zero_count: usize = 0;
    var flip_count: usize = 0;

    for (values, 0..) |value, row| {
        if (rowValid(maybe_validity, row)) {
            const sign = signOf(value);
            switch (sign) {
                1 => positive_count += 1,
                -1 => negative_count += 1,
                else => zero_count += 1,
            }
            if (row >= periods) {
                const previous_row = row - periods;
                if (rowValid(maybe_validity, previous_row) and sign != signOf(values[previous_row])) flip_count += 1;
            }
            count += 1;
        }
        writeSummary(row, min_periods, count, positive_count, negative_count, zero_count, flip_count, out);
    }

    return out;
}

pub fn signProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![SignProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| signProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| signProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| signProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| signProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| signProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| signProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| signProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| signProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| signProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| signProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| signProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| signProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| signProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn signProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![SignProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try signProfile(allocator, values, maybe_validity, options_value.periods);
    defer metrics.deinit();

    var columns: [SignProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.signs, metrics.sign_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.flips, metrics.flip_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.positive_streak, metrics.sign_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.negative_streak, metrics.sign_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.zero_streak, metrics.sign_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingSignProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    sign_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingSignProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| rollingSignProfileColumnsTyped(i8, allocator, typed, sign_options, options_value, device_value),
        .i16 => |typed| rollingSignProfileColumnsTyped(i16, allocator, typed, sign_options, options_value, device_value),
        .i32 => |typed| rollingSignProfileColumnsTyped(i32, allocator, typed, sign_options, options_value, device_value),
        .i64 => |typed| rollingSignProfileColumnsTyped(i64, allocator, typed, sign_options, options_value, device_value),
        .u8 => |typed| rollingSignProfileColumnsTyped(u8, allocator, typed, sign_options, options_value, device_value),
        .u16 => |typed| rollingSignProfileColumnsTyped(u16, allocator, typed, sign_options, options_value, device_value),
        .u32 => |typed| rollingSignProfileColumnsTyped(u32, allocator, typed, sign_options, options_value, device_value),
        .u64 => |typed| rollingSignProfileColumnsTyped(u64, allocator, typed, sign_options, options_value, device_value),
        .usize => |typed| rollingSignProfileColumnsTyped(usize, allocator, typed, sign_options, options_value, device_value),
        .isize => |typed| rollingSignProfileColumnsTyped(isize, allocator, typed, sign_options, options_value, device_value),
        .f16 => |typed| rollingSignProfileColumnsTyped(f16, allocator, typed, sign_options, options_value, device_value),
        .f32 => |typed| rollingSignProfileColumnsTyped(f32, allocator, typed, sign_options, options_value, device_value),
        .f64 => |typed| rollingSignProfileColumnsTyped(f64, allocator, typed, sign_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingSignProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    sign_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingSignProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try rollingSignProfile(allocator, values, maybe_validity, sign_options.periods, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingSignProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.positive_rates, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.negative_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.zero_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.flip_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingSignProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    sign_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingSignProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| expandingSignProfileColumnsTyped(i8, allocator, typed, sign_options, options_value, device_value),
        .i16 => |typed| expandingSignProfileColumnsTyped(i16, allocator, typed, sign_options, options_value, device_value),
        .i32 => |typed| expandingSignProfileColumnsTyped(i32, allocator, typed, sign_options, options_value, device_value),
        .i64 => |typed| expandingSignProfileColumnsTyped(i64, allocator, typed, sign_options, options_value, device_value),
        .u8 => |typed| expandingSignProfileColumnsTyped(u8, allocator, typed, sign_options, options_value, device_value),
        .u16 => |typed| expandingSignProfileColumnsTyped(u16, allocator, typed, sign_options, options_value, device_value),
        .u32 => |typed| expandingSignProfileColumnsTyped(u32, allocator, typed, sign_options, options_value, device_value),
        .u64 => |typed| expandingSignProfileColumnsTyped(u64, allocator, typed, sign_options, options_value, device_value),
        .usize => |typed| expandingSignProfileColumnsTyped(usize, allocator, typed, sign_options, options_value, device_value),
        .isize => |typed| expandingSignProfileColumnsTyped(isize, allocator, typed, sign_options, options_value, device_value),
        .f16 => |typed| expandingSignProfileColumnsTyped(f16, allocator, typed, sign_options, options_value, device_value),
        .f32 => |typed| expandingSignProfileColumnsTyped(f32, allocator, typed, sign_options, options_value, device_value),
        .f64 => |typed| expandingSignProfileColumnsTyped(f64, allocator, typed, sign_options, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingSignProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    sign_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingSignProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);
    var metrics = try expandingSignProfile(allocator, values, maybe_validity, sign_options.periods, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingSignProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.positive_rates, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.negative_rates, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.zero_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.flip_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const SignFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendSignColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    sign_columns: anytype,
) SignFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + sign_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&sign_columns) |*sign_col| {
        columns[initialized] = sign_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn signFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    sign_columns_value: anytype,
    comptime namesFn: anytype,
) SignFrameError!DeviceDataFrame {
    var sign_columns = sign_columns_value;
    var sign_columns_transferred: usize = 0;
    errdefer {
        for (sign_columns[sign_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + sign_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var sign_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, sign_names[0..]);
    for (sign_names, 0..) |sign_name, i| source_names[frame.columns.len + i] = sign_name;

    const out = try appendSignColumns(DeviceDataFrame, frame, source_names, sign_columns);
    sign_columns_transferred = sign_columns.len;
    return out;
}

pub fn signProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceTrendOptions,
) SignFrameError!DeviceDataFrame {
    const sign_value = try frame.column(name);
    const sign_columns = try signProfileColumnsByValue(frame.allocator, sign_value.*, options_value, frame.device, frame.rows);
    return signFrameFromColumns(DeviceDataFrame, frame, output_prefix, sign_columns, signProfileOutputNames);
}

pub fn rollingSignProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    sign_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
) SignFrameError!DeviceDataFrame {
    const sign_value = try frame.column(name);
    const sign_columns = try rollingSignProfileColumnsByValue(frame.allocator, sign_value.*, sign_options, options_value, frame.device, frame.rows);
    return signFrameFromColumns(DeviceDataFrame, frame, output_prefix, sign_columns, rollingSignProfileOutputNames);
}

pub fn expandingSignProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    sign_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
) SignFrameError!DeviceDataFrame {
    const sign_value = try frame.column(name);
    const sign_columns = try expandingSignProfileColumnsByValue(frame.allocator, sign_value.*, sign_options, options_value, frame.device, frame.rows);
    return signFrameFromColumns(DeviceDataFrame, frame, output_prefix, sign_columns, expandingSignProfileOutputNames);
}
