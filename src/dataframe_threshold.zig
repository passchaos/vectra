const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const threshold_columns_mod = @import("dataframe_threshold_columns.zig");
const threshold_metrics_mod = @import("dataframe_threshold_metrics.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;

pub const ThresholdMetrics = threshold_metrics_mod.ThresholdMetrics;
pub const ThresholdSummaryMetrics = threshold_metrics_mod.ThresholdSummaryMetrics;
pub const ThresholdProfileColumnCount = threshold_metrics_mod.ThresholdProfileColumnCount;
pub const RollingThresholdProfileColumnCount = threshold_metrics_mod.RollingThresholdProfileColumnCount;
pub const ExpandingThresholdProfileColumnCount = threshold_metrics_mod.ExpandingThresholdProfileColumnCount;
pub const thresholdProfileOutputNames = threshold_metrics_mod.thresholdProfileOutputNames;
pub const rollingThresholdProfileOutputNames = threshold_metrics_mod.rollingThresholdProfileOutputNames;
pub const expandingThresholdProfileOutputNames = threshold_metrics_mod.expandingThresholdProfileOutputNames;
pub const thresholdProfile = threshold_metrics_mod.thresholdProfile;
pub const rollingThresholdProfile = threshold_metrics_mod.rollingThresholdProfile;
pub const expandingThresholdProfile = threshold_metrics_mod.expandingThresholdProfile;
pub const thresholdProfileColumnsByValue = threshold_columns_mod.thresholdProfileColumnsByValue;
pub const rollingThresholdProfileColumnsByValue = threshold_columns_mod.rollingThresholdProfileColumnsByValue;
pub const expandingThresholdProfileColumnsByValue = threshold_columns_mod.expandingThresholdProfileColumnsByValue;

const ThresholdFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendThresholdColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    threshold_columns: anytype,
) ThresholdFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + threshold_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&threshold_columns) |*threshold_col| {
        columns[initialized] = threshold_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn thresholdFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    threshold_columns_value: anytype,
    comptime namesFn: anytype,
) ThresholdFrameError!DeviceDataFrame {
    var threshold_columns = threshold_columns_value;
    var threshold_columns_transferred: usize = 0;
    errdefer {
        for (threshold_columns[threshold_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + threshold_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var threshold_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, threshold_names[0..]);
    for (threshold_names, 0..) |threshold_name, i| source_names[frame.columns.len + i] = threshold_name;

    const out = try appendThresholdColumns(DeviceDataFrame, frame, source_names, threshold_columns);
    threshold_columns_transferred = threshold_columns.len;
    return out;
}

pub fn thresholdProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceThresholdOptions,
) ThresholdFrameError!DeviceDataFrame {
    const threshold_value = try frame.column(name);
    const threshold_columns = try thresholdProfileColumnsByValue(frame.allocator, threshold_value.*, options_value, frame.device, frame.rows);
    return thresholdFrameFromColumns(DeviceDataFrame, frame, output_prefix, threshold_columns, thresholdProfileOutputNames);
}

pub fn rollingThresholdProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    options_value: DeviceRollingOptions,
) ThresholdFrameError!DeviceDataFrame {
    const threshold_value = try frame.column(name);
    const threshold_columns = try rollingThresholdProfileColumnsByValue(frame.allocator, threshold_value.*, threshold, options_value, frame.device, frame.rows);
    return thresholdFrameFromColumns(DeviceDataFrame, frame, output_prefix, threshold_columns, rollingThresholdProfileOutputNames);
}

pub fn expandingThresholdProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    options_value: DeviceExpandingOptions,
) ThresholdFrameError!DeviceDataFrame {
    const threshold_value = try frame.column(name);
    const threshold_columns = try expandingThresholdProfileColumnsByValue(frame.allocator, threshold_value.*, threshold, options_value, frame.device, frame.rows);
    return thresholdFrameFromColumns(DeviceDataFrame, frame, output_prefix, threshold_columns, expandingThresholdProfileOutputNames);
}
