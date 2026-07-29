const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const group_basic_mod = @import("dataframe_group_basic.zig");
const group_dispatch_mod = @import("dataframe_group_dispatch.zig");
const metrics_mod = @import("dataframe_group_metrics.zig");
const options_mod = @import("dataframe_options.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;

const GroupByMethodError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

pub const MomentProfile = metrics_mod.MomentProfile;
pub const MetricSlices = metrics_mod.MetricSlices;
pub const materializeMetrics = metrics_mod.materializeMetrics;
pub const initProfileDataFrame = metrics_mod.initProfileDataFrame;

pub const groupByCountTyped = group_basic_mod.groupByCountTyped;
pub const initAggregatedDataFrame = group_basic_mod.initAggregatedDataFrame;
pub const groupByNumericDispatchKey = group_basic_mod.groupByNumericDispatchKey;
pub const groupByMeanDispatchKey = group_basic_mod.groupByMeanDispatchKey;

pub const groupByStatsDispatchKey = group_dispatch_mod.groupByStatsDispatchKey;
pub const groupByProfileDispatchKey = group_dispatch_mod.groupByProfileDispatchKey;

pub fn groupByCount(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_name: []const u8,
    output_name: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    return switch (key.*) {
        .bool => |typed| groupByCountTyped(DeviceDataFrame, bool, frame.allocator, key_name, output_name, typed, frame.device),
        .i8 => |typed| groupByCountTyped(DeviceDataFrame, i8, frame.allocator, key_name, output_name, typed, frame.device),
        .i16 => |typed| groupByCountTyped(DeviceDataFrame, i16, frame.allocator, key_name, output_name, typed, frame.device),
        .i32 => |typed| groupByCountTyped(DeviceDataFrame, i32, frame.allocator, key_name, output_name, typed, frame.device),
        .i64 => |typed| groupByCountTyped(DeviceDataFrame, i64, frame.allocator, key_name, output_name, typed, frame.device),
        .u8 => |typed| groupByCountTyped(DeviceDataFrame, u8, frame.allocator, key_name, output_name, typed, frame.device),
        .u16 => |typed| groupByCountTyped(DeviceDataFrame, u16, frame.allocator, key_name, output_name, typed, frame.device),
        .u32 => |typed| groupByCountTyped(DeviceDataFrame, u32, frame.allocator, key_name, output_name, typed, frame.device),
        .u64 => |typed| groupByCountTyped(DeviceDataFrame, u64, frame.allocator, key_name, output_name, typed, frame.device),
        .usize => |typed| groupByCountTyped(DeviceDataFrame, usize, frame.allocator, key_name, output_name, typed, frame.device),
        .isize => |typed| groupByCountTyped(DeviceDataFrame, isize, frame.allocator, key_name, output_name, typed, frame.device),
        .f16 => |typed| groupByCountTyped(DeviceDataFrame, f16, frame.allocator, key_name, output_name, typed, frame.device),
        .f32 => |typed| groupByCountTyped(DeviceDataFrame, f32, frame.allocator, key_name, output_name, typed, frame.device),
        .f64 => |typed| groupByCountTyped(DeviceDataFrame, f64, frame.allocator, key_name, output_name, typed, frame.device),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn groupByNumeric(
    comptime DeviceDataFrame: type,
    op: DeviceGroupByAggregation,
    frame: DeviceDataFrame,
    key_name: []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByNumericDispatchKey(DeviceDataFrame, op, frame.allocator, key_name, output_name, key.*, value.*, frame.device);
}

pub fn groupByMean(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_name: []const u8,
    value_name: []const u8,
    output_name: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByMeanDispatchKey(DeviceDataFrame, frame.allocator, key_name, output_name, key.*, value.*, frame.device);
}

pub fn groupByStats(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_name: []const u8,
    value_name: []const u8,
    output_prefix: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByStatsDispatchKey(DeviceDataFrame, frame.allocator, key_name, output_prefix, key.*, value.*, frame.device);
}

pub fn groupByProfile(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    key_name: []const u8,
    value_name: []const u8,
    output_prefix: []const u8,
) GroupByMethodError!DeviceDataFrame {
    const key = try frame.column(key_name);
    const value = try frame.column(value_name);
    return groupByProfileDispatchKey(DeviceDataFrame, frame.allocator, key_name, output_prefix, key.*, value.*, frame.device);
}
