const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const transition_metrics_mod = @import("dataframe_bool_transition_metrics.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe_validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceTrendOptions = options_mod.DeviceTrendOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const validityValues = validity_mod.validityValues;

pub const BoolTransitionProfile = transition_metrics_mod.BoolTransitionProfile;
pub const BoolTransitionProfileColumnCount = transition_metrics_mod.BoolTransitionProfileColumnCount;
pub const boolTransitionProfileOutputNames = transition_metrics_mod.boolTransitionProfileOutputNames;
pub const boolTransitionProfile = transition_metrics_mod.boolTransitionProfile;
pub const RollingBoolTransitionMetrics = transition_metrics_mod.RollingBoolTransitionMetrics;
pub const RollingBoolTransitionProfileColumnCount = transition_metrics_mod.RollingBoolTransitionProfileColumnCount;
pub const rollingBoolTransitionProfileOutputNames = transition_metrics_mod.rollingBoolTransitionProfileOutputNames;
pub const rollingBoolTransitionProfile = transition_metrics_mod.rollingBoolTransitionProfile;
pub const ExpandingBoolTransitionMetrics = transition_metrics_mod.ExpandingBoolTransitionMetrics;
pub const ExpandingBoolTransitionProfileColumnCount = transition_metrics_mod.ExpandingBoolTransitionProfileColumnCount;
pub const expandingBoolTransitionProfileOutputNames = transition_metrics_mod.expandingBoolTransitionProfileOutputNames;
pub const expandingBoolTransitionProfile = transition_metrics_mod.expandingBoolTransitionProfile;

pub fn boolTransitionProfileColumns(
    allocator: std.mem.Allocator,
    source: DeviceTypedColumn(bool),
    options_value: DeviceTrendOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![BoolTransitionProfileColumnCount]DeviceColumn {
    if (source.len() != rows) return error.LengthMismatch;

    const values = try source.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(source, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var profile = try boolTransitionProfile(allocator, values, maybe_validity, options_value.periods);
    defer profile.deinit();

    var columns: [BoolTransitionProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.rising, profile.transition_validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.falling, profile.transition_validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.toggled, profile.transition_validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(i64, allocator, profile.true_streak, profile.streak_validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(i64, allocator, profile.false_streak, profile.streak_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingBoolTransitionProfileColumns(
    allocator: std.mem.Allocator,
    source: DeviceTypedColumn(bool),
    transition_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingBoolTransitionProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (source.len() != rows) return error.LengthMismatch;

    const values = try source.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(source, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var metrics = try rollingBoolTransitionProfile(
        allocator,
        values,
        maybe_validity,
        transition_options.periods,
        options_value.window,
        min_periods,
    );
    defer metrics.deinit();

    var columns: [RollingBoolTransitionProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.rising_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, metrics.falling_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(i64, allocator, metrics.toggle_counts, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.rising_rates, metrics.metric_validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.falling_rates, metrics.metric_validity, device_value);
    initialized += 1;
    columns[6] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.toggle_rates, metrics.metric_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingBoolTransitionProfileColumns(
    allocator: std.mem.Allocator,
    source: DeviceTypedColumn(bool),
    transition_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingBoolTransitionProfileColumnCount]DeviceColumn {
    if (source.len() != rows) return error.LengthMismatch;

    const values = try source.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(source, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    var metrics = try expandingBoolTransitionProfile(
        allocator,
        values,
        maybe_validity,
        transition_options.periods,
        options_value.min_periods,
    );
    defer metrics.deinit();

    var columns: [ExpandingBoolTransitionProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.rising_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, metrics.falling_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(i64, allocator, metrics.toggle_counts, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.rising_rates, metrics.metric_validity, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.falling_rates, metrics.metric_validity, device_value);
    initialized += 1;
    columns[6] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.toggle_rates, metrics.metric_validity, device_value);
    initialized += 1;
    return columns;
}

const BoolTransitionFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendBoolTransitionColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    transition_columns: anytype,
) BoolTransitionFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + transition_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&transition_columns) |*transition_col| {
        columns[initialized] = transition_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn boolTransitionFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    transition_columns_value: anytype,
    comptime namesFn: anytype,
) BoolTransitionFrameError!DeviceDataFrame {
    var transition_columns = transition_columns_value;
    var transition_columns_transferred: usize = 0;
    errdefer {
        for (transition_columns[transition_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + transition_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var transition_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, transition_names[0..]);
    for (transition_names, 0..) |transition_name, i| source_names[frame.columns.len + i] = transition_name;

    const out = try appendBoolTransitionColumns(DeviceDataFrame, frame, source_names, transition_columns);
    transition_columns_transferred = transition_columns.len;
    return out;
}

fn boolSource(frame: anytype, name: []const u8) BoolTransitionFrameError!@TypeOf((frame.column(name) catch unreachable).bool) {
    const source = try frame.column(name);
    if (source.dtype() != .bool) return error.TypeMismatch;
    return source.bool;
}

pub fn boolTransitionProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceTrendOptions,
) BoolTransitionFrameError!DeviceDataFrame {
    const source = try boolSource(frame, name);
    const transition_columns = try boolTransitionProfileColumns(frame.allocator, source, options_value, frame.device, frame.rows);
    return boolTransitionFrameFromColumns(DeviceDataFrame, frame, output_prefix, transition_columns, boolTransitionProfileOutputNames);
}

pub fn rollingBoolTransitionProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    transition_options: DeviceTrendOptions,
    options_value: DeviceRollingOptions,
) BoolTransitionFrameError!DeviceDataFrame {
    const source = try boolSource(frame, name);
    const transition_columns = try rollingBoolTransitionProfileColumns(frame.allocator, source, transition_options, options_value, frame.device, frame.rows);
    return boolTransitionFrameFromColumns(DeviceDataFrame, frame, output_prefix, transition_columns, rollingBoolTransitionProfileOutputNames);
}

pub fn expandingBoolTransitionProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    transition_options: DeviceTrendOptions,
    options_value: DeviceExpandingOptions,
) BoolTransitionFrameError!DeviceDataFrame {
    const source = try boolSource(frame, name);
    const transition_columns = try expandingBoolTransitionProfileColumns(frame.allocator, source, transition_options, options_value, frame.device, frame.rows);
    return boolTransitionFrameFromColumns(DeviceDataFrame, frame, output_prefix, transition_columns, expandingBoolTransitionProfileOutputNames);
}
