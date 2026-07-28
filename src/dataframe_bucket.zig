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
const DeviceBucketOptions = options_mod.DeviceBucketOptions;
const argsortTypedColumn = dataframe_device_column_mod.argsortTypedColumn;
const compareSortValues = numeric_mod.compareSortValues;
const validityValues = validity_mod.validityValues;

pub const BucketMetrics = struct {
    allocator: std.mem.Allocator,
    ecdf: []f64,
    buckets: []i64,
    lower_tail: []bool,
    upper_tail: []bool,
    validity: []bool,

    pub fn deinit(self: *BucketMetrics) void {
        self.allocator.free(self.ecdf);
        self.allocator.free(self.buckets);
        self.allocator.free(self.lower_tail);
        self.allocator.free(self.upper_tail);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

pub const BucketProfileColumnCount = 4;

pub fn bucketProfileOutputNames(allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![BucketProfileColumnCount][]const u8 {
    var names: [BucketProfileColumnCount][]const u8 = undefined;
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    const suffixes = [_][]const u8{ "ecdf", "bucket", "lower_tail", "upper_tail" };
    for (suffixes, 0..) |suffix, i| {
        names[i] = try std.fmt.allocPrint(allocator, "{s}_{s}", .{ prefix, suffix });
        initialized += 1;
    }
    return names;
}

fn validate(values_len: usize, maybe_validity: ?[]const bool, buckets: usize, lower_quantile: f64, upper_quantile: f64, min_periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (buckets == 0 or min_periods == 0) return error.InvalidShape;
    if (lower_quantile < 0 or lower_quantile > 1) return error.InvalidShape;
    if (upper_quantile < 0 or upper_quantile > 1) return error.InvalidShape;
    if (lower_quantile > upper_quantile) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values_len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn bucketProfile(
    allocator: std.mem.Allocator,
    order: []const usize,
    maybe_validity: ?[]const bool,
    keysTie: *const fn (usize, usize) bool,
    buckets_count: usize,
    lower_quantile: f64,
    upper_quantile: f64,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!BucketMetrics {
    try validate(order.len, maybe_validity, buckets_count, lower_quantile, upper_quantile, min_periods);

    const ecdf = try allocator.alloc(f64, order.len);
    errdefer allocator.free(ecdf);
    const buckets = try allocator.alloc(i64, order.len);
    errdefer allocator.free(buckets);
    const lower_tail = try allocator.alloc(bool, order.len);
    errdefer allocator.free(lower_tail);
    const upper_tail = try allocator.alloc(bool, order.len);
    errdefer allocator.free(upper_tail);
    const validity = try allocator.alloc(bool, order.len);
    errdefer allocator.free(validity);

    @memset(ecdf, 0);
    @memset(buckets, 0);
    @memset(lower_tail, false);
    @memset(upper_tail, false);
    @memset(validity, false);

    var valid_count: usize = 0;
    for (0..order.len) |row| {
        if (rowValid(maybe_validity, row)) valid_count += 1;
    }

    if (valid_count >= min_periods and valid_count != 0) {
        var group_start: usize = 0;
        while (group_start < valid_count) {
            var group_end = group_start + 1;
            while (group_end < valid_count and keysTie(order[group_start], order[group_end])) {
                group_end += 1;
            }

            const rank_position = group_end; // right-continuous ECDF, 1-based.
            const ecdf_value = @as(f64, @floatFromInt(rank_position)) / @as(f64, @floatFromInt(valid_count));
            var bucket_index = @divFloor((rank_position - 1) * buckets_count, valid_count);
            if (bucket_index >= buckets_count) bucket_index = buckets_count - 1;
            const is_lower = ecdf_value <= lower_quantile;
            const is_upper = ecdf_value >= upper_quantile;

            for (order[group_start..group_end]) |row| {
                ecdf[row] = ecdf_value;
                buckets[row] = @intCast(bucket_index);
                lower_tail[row] = is_lower;
                upper_tail[row] = is_upper;
                validity[row] = true;
            }
            group_start = group_end;
        }
    }

    return .{
        .allocator = allocator,
        .ecdf = ecdf,
        .buckets = buckets,
        .lower_tail = lower_tail,
        .upper_tail = upper_tail,
        .validity = validity,
    };
}

pub fn bucketProfileColumnsByValue(
    allocator: std.mem.Allocator,
    value: DeviceColumn,
    options_value: DeviceBucketOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{LengthMismatch})![BucketProfileColumnCount]DeviceColumn {
    if (value.len() != rows) return error.LengthMismatch;
    return switch (value) {
        .bool => |typed| bucketProfileColumnsTyped(bool, allocator, typed, options_value, device_value),
        .i8 => |typed| bucketProfileColumnsTyped(i8, allocator, typed, options_value, device_value),
        .i16 => |typed| bucketProfileColumnsTyped(i16, allocator, typed, options_value, device_value),
        .i32 => |typed| bucketProfileColumnsTyped(i32, allocator, typed, options_value, device_value),
        .i64 => |typed| bucketProfileColumnsTyped(i64, allocator, typed, options_value, device_value),
        .u8 => |typed| bucketProfileColumnsTyped(u8, allocator, typed, options_value, device_value),
        .u16 => |typed| bucketProfileColumnsTyped(u16, allocator, typed, options_value, device_value),
        .u32 => |typed| bucketProfileColumnsTyped(u32, allocator, typed, options_value, device_value),
        .u64 => |typed| bucketProfileColumnsTyped(u64, allocator, typed, options_value, device_value),
        .usize => |typed| bucketProfileColumnsTyped(usize, allocator, typed, options_value, device_value),
        .isize => |typed| bucketProfileColumnsTyped(isize, allocator, typed, options_value, device_value),
        .f16 => |typed| bucketProfileColumnsTyped(f16, allocator, typed, options_value, device_value),
        .f32 => |typed| bucketProfileColumnsTyped(f32, allocator, typed, options_value, device_value),
        .f64 => |typed| bucketProfileColumnsTyped(f64, allocator, typed, options_value, device_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
fn bucketProfileColumnsTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    column: DeviceTypedColumn(T),
    options_value: DeviceBucketOptions,
    device_value: array_mod.Device,
) (array_mod.ArrayError || error{LengthMismatch})![BucketProfileColumnCount]DeviceColumn {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const order = try argsortTypedColumn(T, column, allocator, .{ .descending = false, .nulls = .last });
    defer allocator.free(order);

    const TieCtx = struct {
        values: []const T,
        fn keysTie(ctx: @This(), lhs: usize, rhs: usize) bool {
            if (comptime T == bool) return ctx.values[lhs] == ctx.values[rhs];
            return compareSortValues(T, ctx.values[lhs], ctx.values[rhs]) == 0;
        }
    };
    const ctx = TieCtx{ .values = values };
    const keysTie = struct {
        var context: TieCtx = undefined;
        fn call(lhs: usize, rhs: usize) bool {
            return context.keysTie(lhs, rhs);
        }
    };
    keysTie.context = ctx;

    var metrics = try bucketProfile(
        allocator,
        order,
        maybe_validity,
        keysTie.call,
        options_value.buckets,
        options_value.lower_quantile,
        options_value.upper_quantile,
        options_value.min_periods,
    );
    defer metrics.deinit();

    var columns: [BucketProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(f64, allocator, metrics.ecdf, metrics.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(i64, allocator, metrics.buckets, metrics.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.lower_tail, metrics.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, metrics.upper_tail, metrics.validity, device_value);
    initialized += 1;
    return columns;
}

const BucketFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

pub fn bucketProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceBucketOptions,
) BucketFrameError!DeviceDataFrame {
    const bucket_value = try frame.column(name);
    var bucket_columns = try bucketProfileColumnsByValue(frame.allocator, bucket_value.*, options_value, frame.device, frame.rows);
    var bucket_columns_transferred: usize = 0;
    errdefer {
        for (bucket_columns[bucket_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + bucket_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var bucket_names = try bucketProfileOutputNames(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, bucket_names[0..]);
    for (bucket_names, 0..) |bucket_name, i| source_names[frame.columns.len + i] = bucket_name;

    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + bucket_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&bucket_columns) |*bucket_col| {
        columns[initialized] = bucket_col.*;
        initialized += 1;
        bucket_columns_transferred += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}
