const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const validity_core_mod = @import("dataframe_validity_core.zig");
const validity_metrics_mod = @import("dataframe_validity_metrics.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;

pub const countNulls = validity_core_mod.countNulls;
pub const countNullsInArray = validity_core_mod.countNullsInArray;
pub const validityValues = validity_core_mod.validityValues;

pub const ValidityMetrics = validity_metrics_mod.ValidityMetrics;
pub const ValiditySummaryMetrics = validity_metrics_mod.ValiditySummaryMetrics;
pub const ValidityProfileColumnCount = validity_metrics_mod.ValidityProfileColumnCount;
pub const RollingValidityProfileColumnCount = validity_metrics_mod.RollingValidityProfileColumnCount;
pub const ExpandingValidityProfileColumnCount = validity_metrics_mod.ExpandingValidityProfileColumnCount;
pub const validityProfileOutputNames = validity_metrics_mod.validityProfileOutputNames;
pub const rollingValidityProfileOutputNames = validity_metrics_mod.rollingValidityProfileOutputNames;
pub const expandingValidityProfileOutputNames = validity_metrics_mod.expandingValidityProfileOutputNames;
pub const validityProfile = validity_metrics_mod.validityProfile;
pub const rollingValidityProfile = validity_metrics_mod.rollingValidityProfile;
pub const expandingValidityProfile = validity_metrics_mod.expandingValidityProfile;

pub fn validityProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ValidityProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        inline else => |typed| validityProfileColumnsTyped(allocator, typed, device_value),
    };
}
fn validityProfileColumnsTyped(
    allocator: std.mem.Allocator,
    column: anytype,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ValidityProfileColumnCount]DeviceColumn {
    const rows = column.len();
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var metrics = try validityProfile(allocator, rows, maybe_validity);
    defer metrics.deinit();

    var columns: [ValidityProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(bool, allocator, metrics.is_null, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(bool, allocator, metrics.is_valid, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, metrics.valid_streak, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(i64, allocator, metrics.null_streak, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingValidityProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingValidityProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        inline else => |typed| rollingValidityProfileColumnsTyped(allocator, typed, options_value, device_value),
    };
}
fn rollingValidityProfileColumnsTyped(
    allocator: std.mem.Allocator,
    column: anytype,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingValidityProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const rows = column.len();
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var metrics = try rollingValidityProfile(allocator, rows, maybe_validity, options_value.window, min_periods);
    defer metrics.deinit();

    var columns: [RollingValidityProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.total_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.valid_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, metrics.null_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.valid_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.null_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingValidityProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingValidityProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        inline else => |typed| expandingValidityProfileColumnsTyped(allocator, typed, options_value, device_value),
    };
}
fn expandingValidityProfileColumnsTyped(
    allocator: std.mem.Allocator,
    column: anytype,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingValidityProfileColumnCount]DeviceColumn {
    const rows = column.len();
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var metrics = try expandingValidityProfile(allocator, rows, maybe_validity, options_value.min_periods);
    defer metrics.deinit();

    var columns: [ExpandingValidityProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.total_counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.valid_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, metrics.null_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.valid_rates, metrics.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.null_rates, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const ValidityFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendValidityColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    validity_columns: anytype,
) ValidityFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + validity_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&validity_columns) |*validity_col| {
        columns[initialized] = validity_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn validityFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    validity_columns_value: anytype,
    comptime namesFn: anytype,
) ValidityFrameError!DeviceDataFrame {
    var validity_columns = validity_columns_value;
    var validity_columns_transferred: usize = 0;
    errdefer {
        for (validity_columns[validity_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + validity_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var validity_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, validity_names[0..]);
    for (validity_names, 0..) |validity_name, i| source_names[frame.columns.len + i] = validity_name;

    const out = try appendValidityColumns(DeviceDataFrame, frame, source_names, validity_columns);
    validity_columns_transferred = validity_columns.len;
    return out;
}

pub fn validityProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
) ValidityFrameError!DeviceDataFrame {
    const source = try frame.column(name);
    const validity_columns = try validityProfileColumnsByValue(frame.allocator, source.*, frame.device, frame.rows);
    return validityFrameFromColumns(DeviceDataFrame, frame, output_prefix, validity_columns, validityProfileOutputNames);
}

pub fn rollingValidityProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) ValidityFrameError!DeviceDataFrame {
    const source = try frame.column(name);
    const validity_columns = try rollingValidityProfileColumnsByValue(frame.allocator, source.*, options_value, frame.device, frame.rows);
    return validityFrameFromColumns(DeviceDataFrame, frame, output_prefix, validity_columns, rollingValidityProfileOutputNames);
}

pub fn expandingValidityProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) ValidityFrameError!DeviceDataFrame {
    const source = try frame.column(name);
    const validity_columns = try expandingValidityProfileColumnsByValue(frame.allocator, source.*, options_value, frame.device, frame.rows);
    return validityFrameFromColumns(DeviceDataFrame, frame, output_prefix, validity_columns, expandingValidityProfileOutputNames);
}
