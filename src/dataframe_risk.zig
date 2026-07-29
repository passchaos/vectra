const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const risk_columns_mod = @import("dataframe_risk_columns.zig");
const risk_metrics_mod = @import("dataframe_risk_metrics.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceDrawdownOptions = options_mod.DeviceDrawdownOptions;
const DeviceExtremaOptions = options_mod.DeviceExtremaOptions;

pub const DrawdownMetrics = risk_metrics_mod.DrawdownMetrics;
pub const RollingDrawdownMetrics = risk_metrics_mod.RollingDrawdownMetrics;
pub const ExtremaMetrics = risk_metrics_mod.ExtremaMetrics;
pub const DrawdownProfileColumnCount = risk_metrics_mod.DrawdownProfileColumnCount;
pub const RollingDrawdownProfileColumnCount = risk_metrics_mod.RollingDrawdownProfileColumnCount;
pub const ExtremaProfileColumnCount = risk_metrics_mod.ExtremaProfileColumnCount;
pub const drawdownProfileOutputNames = risk_metrics_mod.drawdownProfileOutputNames;
pub const rollingDrawdownProfileOutputNames = risk_metrics_mod.rollingDrawdownProfileOutputNames;
pub const extremaProfileOutputNames = risk_metrics_mod.extremaProfileOutputNames;
pub const drawdownProfile = risk_metrics_mod.drawdownProfile;
pub const rollingDrawdownProfile = risk_metrics_mod.rollingDrawdownProfile;
pub const extremaProfile = risk_metrics_mod.extremaProfile;
pub const rollingDrawdownProfileColumnsByValue = risk_columns_mod.rollingDrawdownProfileColumnsByValue;
pub const drawdownProfileColumnsByValue = risk_columns_mod.drawdownProfileColumnsByValue;
pub const extremaProfileColumnsByValue = risk_columns_mod.extremaProfileColumnsByValue;

const RiskFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendProfileColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    profile_columns: anytype,
) RiskFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + profile_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&profile_columns) |*profile_col| {
        columns[initialized] = profile_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

pub fn rollingDrawdownProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) RiskFrameError!DeviceDataFrame {
    const rolling_value = try frame.column(name);
    var rolling_columns = try rollingDrawdownProfileColumnsByValue(frame.allocator, rolling_value.*, options_value, frame.device, frame.rows);
    var rolling_columns_transferred: usize = 0;
    errdefer {
        for (rolling_columns[rolling_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + rolling_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var rolling_names = try rollingDrawdownProfileOutputNames(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, rolling_names[0..]);
    for (rolling_names, 0..) |rolling_name, i| source_names[frame.columns.len + i] = rolling_name;

    const out = try appendProfileColumns(DeviceDataFrame, frame, source_names, rolling_columns);
    rolling_columns_transferred = rolling_columns.len;
    return out;
}

pub fn drawdownProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceDrawdownOptions,
) RiskFrameError!DeviceDataFrame {
    const drawdown_value = try frame.column(name);
    var drawdown_columns = try drawdownProfileColumnsByValue(frame.allocator, drawdown_value.*, options_value, frame.device, frame.rows);
    var drawdown_columns_transferred: usize = 0;
    errdefer {
        for (drawdown_columns[drawdown_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + drawdown_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var drawdown_names = try drawdownProfileOutputNames(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, drawdown_names[0..]);
    for (drawdown_names, 0..) |drawdown_name, i| source_names[frame.columns.len + i] = drawdown_name;

    const out = try appendProfileColumns(DeviceDataFrame, frame, source_names, drawdown_columns);
    drawdown_columns_transferred = drawdown_columns.len;
    return out;
}

pub fn extremaProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExtremaOptions,
) RiskFrameError!DeviceDataFrame {
    const extrema_value = try frame.column(name);
    var extrema_columns = try extremaProfileColumnsByValue(frame.allocator, extrema_value.*, options_value, frame.device, frame.rows);
    var extrema_columns_transferred: usize = 0;
    errdefer {
        for (extrema_columns[extrema_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + extrema_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var extrema_names = try extremaProfileOutputNames(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, extrema_names[0..]);
    for (extrema_names, 0..) |extrema_name, i| source_names[frame.columns.len + i] = extrema_name;

    const out = try appendProfileColumns(DeviceDataFrame, frame, source_names, extrema_columns);
    extrema_columns_transferred = extrema_columns.len;
    return out;
}
