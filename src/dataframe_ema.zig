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
const DeviceEmaOptions = options_mod.DeviceEmaOptions;
const castToF64 = numeric_mod.castToF64;
const validityValues = validity_mod.validityValues;

pub const EmaMetrics = struct {
    allocator: std.mem.Allocator,
    ema_values: []f64,
    residuals: []f64,
    ratios: []f64,
    validity: []bool,

    pub fn deinit(self: *EmaMetrics) void {
        self.allocator.free(self.ema_values);
        self.allocator.free(self.residuals);
        self.allocator.free(self.ratios);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const EmaProfileColumnCount = 3;

pub fn emaProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![EmaProfileColumnCount][]const u8 {
    var names: [EmaProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "ema", "ema_residual", "ema_ratio" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values: []const f64, maybe_validity: ?[]const bool, alpha: f64, min_periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (alpha <= 0 or alpha > 1 or min_periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn emaProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    alpha: f64,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!EmaMetrics {
    try validate(values, maybe_validity, alpha, min_periods);

    const ema_values = try allocator.alloc(f64, values.len);
    errdefer allocator.free(ema_values);
    const residuals = try allocator.alloc(f64, values.len);
    errdefer allocator.free(residuals);
    const ratios = try allocator.alloc(f64, values.len);
    errdefer allocator.free(ratios);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    var seen: usize = 0;
    var ema: f64 = 0;
    // Null observations do not update EMA state. This keeps sequence gaps from
    // biasing the smoother while preserving row-aligned nullable outputs.
    for (values, 0..) |x, row| {
        if (!rowValid(maybe_validity, row)) {
            ema_values[row] = 0;
            residuals[row] = 0;
            ratios[row] = 0;
            validity[row] = false;
            continue;
        }

        if (seen == 0) {
            ema = x;
        } else {
            ema = alpha * x + (1.0 - alpha) * ema;
        }
        seen += 1;

        const has_enough = seen >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            ema_values[row] = ema;
            residuals[row] = x - ema;
            ratios[row] = if (ema == 0) std.math.nan(f64) else x / ema;
        } else {
            ema_values[row] = 0;
            residuals[row] = 0;
            ratios[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .ema_values = ema_values,
        .residuals = residuals,
        .ratios = ratios,
        .validity = validity,
    };
}

pub fn emaProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceEmaOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![EmaProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .i8 => |typed| emaProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| emaProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| emaProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| emaProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| emaProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| emaProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| emaProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| emaProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| emaProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| emaProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| emaProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| emaProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| emaProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn emaProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceEmaOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![EmaProfileColumnCount]DeviceColumn {
    const values_typed = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values_typed);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const rows = values_typed.len;
    const values = try allocator.alloc(f64, rows);
    defer allocator.free(values);
    for (values_typed, 0..) |value, row| values[row] = castToF64(T, value);

    var metrics = try emaProfile(allocator, values, maybe_validity, options_value.alpha, options_value.min_periods);
    defer metrics.deinit();

    var columns: [EmaProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.ema_values, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.residuals, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.ratios, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const EmaFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

pub fn emaProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceEmaOptions,
) EmaFrameError!DeviceDataFrame {
    const ema_value = try frame.column(name);
    var ema_columns = try emaProfileColumnsByValue(frame.allocator, ema_value.*, options_value, frame.device, frame.rows);
    var ema_columns_transferred: usize = 0;
    errdefer {
        for (ema_columns[ema_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + ema_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var ema_names = try emaProfileOutputNames(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, ema_names[0..]);
    for (ema_names, 0..) |ema_name, i| source_names[frame.columns.len + i] = ema_name;

    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + ema_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&ema_columns) |*ema_col| {
        columns[initialized] = ema_col.*;
        initialized += 1;
        ema_columns_transferred += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}
