const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const clip_columns_mod = @import("dataframe_clip_columns.zig");
const clip_metrics_mod = @import("dataframe_clip_metrics.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceClipOptions = options_mod.DeviceClipOptions;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;

pub const ClipMetrics = clip_metrics_mod.ClipMetrics;
pub const ClipSummaryMetrics = clip_metrics_mod.ClipSummaryMetrics;
pub const ClipProfileColumnCount = clip_metrics_mod.ClipProfileColumnCount;
pub const RollingClipProfileColumnCount = clip_metrics_mod.RollingClipProfileColumnCount;
pub const ExpandingClipProfileColumnCount = clip_metrics_mod.ExpandingClipProfileColumnCount;
pub const clipProfileOutputNames = clip_metrics_mod.clipProfileOutputNames;
pub const rollingClipProfileOutputNames = clip_metrics_mod.rollingClipProfileOutputNames;
pub const expandingClipProfileOutputNames = clip_metrics_mod.expandingClipProfileOutputNames;
pub const clipProfile = clip_metrics_mod.clipProfile;
pub const rollingClipProfile = clip_metrics_mod.rollingClipProfile;
pub const expandingClipProfile = clip_metrics_mod.expandingClipProfile;
pub const clipProfileColumnsByValue = clip_columns_mod.clipProfileColumnsByValue;
pub const rollingClipProfileColumnsByValue = clip_columns_mod.rollingClipProfileColumnsByValue;
pub const expandingClipProfileColumnsByValue = clip_columns_mod.expandingClipProfileColumnsByValue;

const ClipFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendClipColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    clip_columns: anytype,
) ClipFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + clip_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&clip_columns) |*clip_col| {
        columns[initialized] = clip_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn clipFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    clip_columns_value: anytype,
    comptime namesFn: anytype,
) ClipFrameError!DeviceDataFrame {
    var clip_columns = clip_columns_value;
    var clip_columns_transferred: usize = 0;
    errdefer {
        for (clip_columns[clip_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + clip_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var clip_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, clip_names[0..]);
    for (clip_names, 0..) |clip_name, i| source_names[frame.columns.len + i] = clip_name;

    const out = try appendClipColumns(DeviceDataFrame, frame, source_names, clip_columns);
    clip_columns_transferred = clip_columns.len;
    return out;
}

pub fn clipProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceClipOptions,
) ClipFrameError!DeviceDataFrame {
    const clip_value = try frame.column(name);
    const clip_columns = try clipProfileColumnsByValue(frame.allocator, clip_value.*, options_value, frame.device, frame.rows);
    return clipFrameFromColumns(DeviceDataFrame, frame, output_prefix, clip_columns, clipProfileOutputNames);
}

pub fn rollingClipProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    clip_options: DeviceClipOptions,
    options_value: DeviceRollingOptions,
) ClipFrameError!DeviceDataFrame {
    const clip_value = try frame.column(name);
    const clip_columns = try rollingClipProfileColumnsByValue(frame.allocator, clip_value.*, clip_options, options_value, frame.device, frame.rows);
    return clipFrameFromColumns(DeviceDataFrame, frame, output_prefix, clip_columns, rollingClipProfileOutputNames);
}

pub fn expandingClipProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    clip_options: DeviceClipOptions,
    options_value: DeviceExpandingOptions,
) ClipFrameError!DeviceDataFrame {
    const clip_value = try frame.column(name);
    const clip_columns = try expandingClipProfileColumnsByValue(frame.allocator, clip_value.*, clip_options, options_value, frame.device, frame.rows);
    return clipFrameFromColumns(DeviceDataFrame, frame, output_prefix, clip_columns, expandingClipProfileOutputNames);
}
