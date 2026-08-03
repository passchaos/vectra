//! Column materializers for eager rank profile operations.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const rank_metrics_mod = @import("dataframe_rank_metrics.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe/validity/core.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
const argsortTypedColumn = dataframe_device_column_mod.argsortTypedColumn;
const compareSortValues = numeric_mod.compareSortValues;
const validityValues = validity_mod.validityValues;

pub const RankProfileColumnCount = rank_metrics_mod.RankProfileColumnCount;
pub const RollingRankProfileColumnCount = rank_metrics_mod.RollingRankProfileColumnCount;
pub const ExpandingRankProfileColumnCount = rank_metrics_mod.ExpandingRankProfileColumnCount;
pub const rankProfile = rank_metrics_mod.rankProfile;
pub const rollingRankProfile = rank_metrics_mod.rollingRankProfile;
pub const expandingRankProfile = rank_metrics_mod.expandingRankProfile;

pub fn rankProfileColumnsByKey(
    allocator: std.mem.Allocator,
    key: DeviceColumn,
    options_value: DeviceSortOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RankProfileColumnCount]DeviceColumn {
    if (key.len() != rows) return error.LengthMismatch;
    return switch (key) {
        .bool => |typed| rankProfileColumnsTyped(bool, allocator, typed, options_value, device_value),
        .i8 => |typed| rankProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rankProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rankProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rankProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rankProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rankProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rankProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rankProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rankProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rankProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rankProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rankProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rankProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rankProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceSortOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RankProfileColumnCount]DeviceColumn {
    const rows = column.len();
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);
    const order = try argsortTypedColumn(T, column, allocator, options_value);
    defer allocator.free(order);

    const TieCtx = struct {
        values: []const T,
        validity: ?[]const bool,
        fn keysTie(ctx: @This(), lhs: usize, rhs: usize) bool {
            const lhs_valid = if (ctx.validity) |validity| validity[lhs] else true;
            const rhs_valid = if (ctx.validity) |validity| validity[rhs] else true;
            if (lhs_valid != rhs_valid) return false;
            if (!lhs_valid) return true;
            return compareSortValues(T, ctx.values[lhs], ctx.values[rhs]) == 0;
        }
    };
    const tie_ctx = TieCtx{ .values = values, .validity = maybe_validity };
    const tie = struct {
        var context: TieCtx = undefined;
        fn call(lhs: usize, rhs: usize) bool {
            return context.keysTie(lhs, rhs);
        }
    };
    tie.context = tie_ctx;

    var metrics = try rankProfile(allocator, rows, order, tie.call);
    defer metrics.deinit();

    var columns: [RankProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.ordinal, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, metrics.competition, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, metrics.dense, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(f64, allocator, metrics.percent, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSlice(f64, allocator, metrics.cume, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingRankProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceRollingRankOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRankProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .bool => |typed| rollingRankProfileColumnsTyped(bool, allocator, typed, options_value, device_value),
        .i8 => |typed| rollingRankProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| rollingRankProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| rollingRankProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| rollingRankProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| rollingRankProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| rollingRankProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| rollingRankProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| rollingRankProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| rollingRankProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| rollingRankProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| rollingRankProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| rollingRankProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| rollingRankProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn rollingRankProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceRollingRankOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![RollingRankProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);
    const rows = values.len;

    const CmpCtx = struct {
        values: []const T,
        fn compare(ctx: @This(), lhs: usize, rhs: usize) i8 {
            return compareSortValues(T, ctx.values[lhs], ctx.values[rhs]);
        }
    };
    const cmp_ctx = CmpCtx{ .values = values };
    const cmp = struct {
        var context: CmpCtx = undefined;
        fn call(lhs: usize, rhs: usize) i8 {
            return context.compare(lhs, rhs);
        }
    };
    cmp.context = cmp_ctx;
    var metrics = try rollingRankProfile(allocator, rows, maybe_validity, options_value.window, min_periods, options_value.descending, cmp.call);
    defer metrics.deinit();

    var columns: [RollingRankProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.ranks, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.percent_ranks, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.cume_dist, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingRankProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceExpandingRankOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingRankProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .bool => |typed| expandingRankProfileColumnsTyped(bool, allocator, typed, options_value, device_value),
        .i8 => |typed| expandingRankProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| expandingRankProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| expandingRankProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| expandingRankProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| expandingRankProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| expandingRankProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| expandingRankProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| expandingRankProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| expandingRankProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| expandingRankProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| expandingRankProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| expandingRankProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| expandingRankProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn expandingRankProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceExpandingRankOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![ExpandingRankProfileColumnCount]DeviceColumn {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);
    const rows = values.len;

    const CmpCtx = struct {
        values: []const T,
        fn compare(ctx: @This(), lhs: usize, rhs: usize) i8 {
            return compareSortValues(T, ctx.values[lhs], ctx.values[rhs]);
        }
    };
    const cmp_ctx = CmpCtx{ .values = values };
    const cmp = struct {
        var context: CmpCtx = undefined;
        fn call(lhs: usize, rhs: usize) i8 {
            return context.compare(lhs, rhs);
        }
    };
    cmp.context = cmp_ctx;
    var metrics = try expandingRankProfile(allocator, rows, maybe_validity, options_value.min_periods, options_value.descending, cmp.call);
    defer metrics.deinit();

    var columns: [ExpandingRankProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, metrics.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.ranks, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.percent_ranks, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.cume_dist, metrics.validity, device_value);
    initialized += 1;
    return columns;
}
