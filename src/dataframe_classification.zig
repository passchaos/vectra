const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const names_mod = @import("dataframe_names.zig");
const dataframe_device_column_mod = @import("dataframe/device_column.zig");
const classification_metrics_mod = @import("dataframe_classification_metrics.zig");
const options_mod = @import("dataframe_options.zig");
const validity_mod = @import("dataframe/validity.zig");

const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;
const DeviceRollingOptions = options_mod.DeviceRollingOptions;
const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
const validityValues = validity_mod.validityValues;

pub const ClassificationProfile = classification_metrics_mod.ClassificationProfile;
pub const ClassificationSummaryProfile = classification_metrics_mod.ClassificationSummaryProfile;
pub const ClassificationProfileColumnCount = classification_metrics_mod.ClassificationProfileColumnCount;
pub const RollingClassificationProfileColumnCount = classification_metrics_mod.RollingClassificationProfileColumnCount;
pub const ExpandingClassificationProfileColumnCount = classification_metrics_mod.ExpandingClassificationProfileColumnCount;
pub const classificationProfileOutputNames = classification_metrics_mod.classificationProfileOutputNames;
pub const rollingClassificationProfileOutputNames = classification_metrics_mod.rollingClassificationProfileOutputNames;
pub const expandingClassificationProfileOutputNames = classification_metrics_mod.expandingClassificationProfileOutputNames;
pub const classificationProfile = classification_metrics_mod.classificationProfile;
pub const rollingClassificationProfile = classification_metrics_mod.rollingClassificationProfile;
pub const expandingClassificationProfile = classification_metrics_mod.expandingClassificationProfile;

pub fn classificationProfileColumns(
    allocator: std.mem.Allocator,
    actual: DeviceTypedColumn(bool),
    predicted: DeviceTypedColumn(bool),
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })![ClassificationProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (!actual.device().sameDevice(predicted.device())) return error.InvalidDevice;

    const actual_values = try actual.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values);
    const predicted_values = try predicted.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values);
    const maybe_actual_validity = try validityValues(actual, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    var profile = try classificationProfile(allocator, actual_values, predicted_values, maybe_actual_validity, maybe_predicted_validity);
    defer profile.deinit();

    var columns: [ClassificationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.tp, profile.validity, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.fp, profile.validity, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.tn, profile.validity, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.fn_values, profile.validity, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSliceWithValidity(bool, allocator, profile.correct, profile.validity, device_value);
    initialized += 1;
    return columns;
}
pub fn rollingClassificationProfileColumns(
    allocator: std.mem.Allocator,
    actual: DeviceTypedColumn(bool),
    predicted: DeviceTypedColumn(bool),
    options_value: DeviceRollingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })![RollingClassificationProfileColumnCount]DeviceColumn {
    const min_periods = options_value.min_periods orelse options_value.window;
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (!actual.device().sameDevice(predicted.device())) return error.InvalidDevice;

    const actual_values = try actual.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values);
    const predicted_values = try predicted.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values);
    const maybe_actual_validity = try validityValues(actual, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    var profile = try rollingClassificationProfile(
        allocator,
        actual_values,
        predicted_values,
        maybe_actual_validity,
        maybe_predicted_validity,
        options_value.window,
        min_periods,
    );
    defer profile.deinit();

    var columns: [RollingClassificationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, profile.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, profile.tp_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, profile.fp_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(i64, allocator, profile.tn_counts, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSlice(i64, allocator, profile.fn_counts, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.accuracies, profile.metric_validity, device_value);
    initialized += 1;
    columns[6] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.precisions, profile.metric_validity, device_value);
    initialized += 1;
    columns[7] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.recalls, profile.metric_validity, device_value);
    initialized += 1;
    return columns;
}
pub fn expandingClassificationProfileColumns(
    allocator: std.mem.Allocator,
    actual: DeviceTypedColumn(bool),
    predicted: DeviceTypedColumn(bool),
    options_value: DeviceExpandingOptions,
    device_value: array_mod.Device,
    rows: usize,
) (array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })![ExpandingClassificationProfileColumnCount]DeviceColumn {
    if (actual.len() != rows or predicted.len() != rows) return error.LengthMismatch;
    if (!actual.device().sameDevice(predicted.device())) return error.InvalidDevice;

    const actual_values = try actual.values.toOwnedSlice(allocator);
    defer allocator.free(actual_values);
    const predicted_values = try predicted.values.toOwnedSlice(allocator);
    defer allocator.free(predicted_values);
    const maybe_actual_validity = try validityValues(actual, allocator);
    defer if (maybe_actual_validity) |validity| allocator.free(validity);
    const maybe_predicted_validity = try validityValues(predicted, allocator);
    defer if (maybe_predicted_validity) |validity| allocator.free(validity);

    var profile = try expandingClassificationProfile(
        allocator,
        actual_values,
        predicted_values,
        maybe_actual_validity,
        maybe_predicted_validity,
        options_value.min_periods,
    );
    defer profile.deinit();

    var columns: [ExpandingClassificationProfileColumnCount]DeviceColumn = undefined;
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
    }
    columns[0] = try DeviceColumn.fromSlice(i64, allocator, profile.counts, device_value);
    initialized += 1;
    columns[1] = try DeviceColumn.fromSlice(i64, allocator, profile.tp_counts, device_value);
    initialized += 1;
    columns[2] = try DeviceColumn.fromSlice(i64, allocator, profile.fp_counts, device_value);
    initialized += 1;
    columns[3] = try DeviceColumn.fromSlice(i64, allocator, profile.tn_counts, device_value);
    initialized += 1;
    columns[4] = try DeviceColumn.fromSlice(i64, allocator, profile.fn_counts, device_value);
    initialized += 1;
    columns[5] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.accuracies, profile.metric_validity, device_value);
    initialized += 1;
    columns[6] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.precisions, profile.metric_validity, device_value);
    initialized += 1;
    columns[7] = try DeviceColumn.fromSliceWithValidity(f64, allocator, profile.recalls, profile.metric_validity, device_value);
    initialized += 1;
    return columns;
}

const ClassificationFrameError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
};

fn appendClassificationColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    source_names: []const []const u8,
    class_columns: anytype,
) ClassificationFrameError!DeviceDataFrame {
    const DeviceColumnType = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumnType, frame.columns.len + class_columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    for (&class_columns) |*class_col| {
        columns[initialized] = class_col.*;
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, source_names, columns, frame.rows, frame.device);
}

fn classificationFrameFromColumns(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    output_prefix: []const u8,
    class_columns_value: anytype,
    comptime namesFn: anytype,
) ClassificationFrameError!DeviceDataFrame {
    var class_columns = class_columns_value;
    var class_columns_transferred: usize = 0;
    errdefer {
        for (class_columns[class_columns_transferred..]) |*col| col.deinit();
    }

    const source_names = try frame.allocator.alloc([]const u8, frame.columns.len + class_columns.len);
    defer frame.allocator.free(source_names);
    for (frame.names, 0..) |source_name, i| source_names[i] = source_name;

    var class_names = try namesFn(frame.allocator, output_prefix);
    defer names_mod.freeOwnedNameItems(frame.allocator, class_names[0..]);
    for (class_names, 0..) |class_name, i| source_names[frame.columns.len + i] = class_name;

    const out = try appendClassificationColumns(DeviceDataFrame, frame, source_names, class_columns);
    class_columns_transferred = class_columns.len;
    return out;
}

fn validateClassificationInputs(frame: anytype, actual_name: []const u8, predicted_name: []const u8) ClassificationFrameError!struct { actual: @TypeOf(frame.column(actual_name) catch unreachable), predicted: @TypeOf(frame.column(predicted_name) catch unreachable) } {
    const actual = try frame.column(actual_name);
    const predicted = try frame.column(predicted_name);
    if (actual.dtype() != .bool or predicted.dtype() != .bool) return error.TypeMismatch;
    return .{ .actual = actual, .predicted = predicted };
}

pub fn classificationProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
) ClassificationFrameError!DeviceDataFrame {
    const inputs = try validateClassificationInputs(frame, actual_name, predicted_name);
    const class_columns = try classificationProfileColumns(frame.allocator, inputs.actual.bool, inputs.predicted.bool, frame.device, frame.rows);
    return classificationFrameFromColumns(DeviceDataFrame, frame, output_prefix, class_columns, classificationProfileOutputNames);
}

pub fn rollingClassificationProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceRollingOptions,
) ClassificationFrameError!DeviceDataFrame {
    const inputs = try validateClassificationInputs(frame, actual_name, predicted_name);
    const class_columns = try rollingClassificationProfileColumns(frame.allocator, inputs.actual.bool, inputs.predicted.bool, options_value, frame.device, frame.rows);
    return classificationFrameFromColumns(DeviceDataFrame, frame, output_prefix, class_columns, rollingClassificationProfileOutputNames);
}

pub fn expandingClassificationProfileFrame(
    comptime DeviceDataFrame: type,
    frame: DeviceDataFrame,
    actual_name: []const u8,
    predicted_name: []const u8,
    output_prefix: []const u8,
    options_value: DeviceExpandingOptions,
) ClassificationFrameError!DeviceDataFrame {
    const inputs = try validateClassificationInputs(frame, actual_name, predicted_name);
    const class_columns = try expandingClassificationProfileColumns(frame.allocator, inputs.actual.bool, inputs.predicted.bool, options_value, frame.device, frame.rows);
    return classificationFrameFromColumns(DeviceDataFrame, frame, output_prefix, class_columns, expandingClassificationProfileOutputNames);
}
