//! Lazy single-column profile operation builders for DeviceLazyFrame.

const std = @import("std");
const array_mod = @import("../../array.zig");
const series_mod = @import("../../series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

fn LazyPayload(comptime LazyOp: type, comptime tag_name: []const u8) type {
    return @FieldType(LazyOp, tag_name);
}

pub fn appendNameOutputOptions(frame: anytype, comptime tag_name: []const u8, name: []const u8, output_prefix: []const u8, options: anytype) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, .{
        .name = owned_name,
        .output_prefix = owned_prefix,
        .options = options,
    }));
}

pub fn appendNameOutput(frame: anytype, comptime tag_name: []const u8, name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, .{
        .name = owned_name,
        .output_prefix = owned_prefix,
    }));
}

/// Build a lazy profile payload whose schema is `{ name, output_prefix, <extra>, options }`.
///
/// The tag and extra field name are comptime parameters so Zig validates the
/// target union payload at compile time. This keeps DeviceLazyFrame as a thin
/// public API facade while centralizing the allocation/rollback rules for the
/// many near-identical lazy profile builders.
pub fn appendNameOutputExtraOptions(
    frame: anytype,
    comptime tag_name: []const u8,
    name: []const u8,
    output_prefix: []const u8,
    comptime extra_field_name: []const u8,
    extra: anytype,
    options: anytype,
) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    var payload: LazyPayload(LazyOp, tag_name) = undefined;
    payload.name = owned_name;
    payload.output_prefix = owned_prefix;
    @field(payload, extra_field_name) = extra;
    payload.options = options;
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, payload));
}

pub fn appendNameOutputThresholdOptions(
    frame: anytype,
    comptime tag_name: []const u8,
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    options: anytype,
) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    var payload: LazyPayload(LazyOp, tag_name) = undefined;
    payload.name = owned_name;
    payload.output_prefix = owned_prefix;
    payload.threshold = threshold;
    payload.options = options;
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, payload));
}

pub fn appendNameOutputThresholdExtraOptions(
    frame: anytype,
    comptime tag_name: []const u8,
    name: []const u8,
    output_prefix: []const u8,
    threshold: f64,
    comptime extra_field_name: []const u8,
    extra: anytype,
    options: anytype,
) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    var payload: LazyPayload(LazyOp, tag_name) = undefined;
    payload.name = owned_name;
    payload.output_prefix = owned_prefix;
    payload.threshold = threshold;
    @field(payload, extra_field_name) = extra;
    payload.options = options;
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, payload));
}

pub fn appendPairOutput(
    frame: anytype,
    comptime tag_name: []const u8,
    comptime lhs_field_name: []const u8,
    lhs_name: []const u8,
    comptime rhs_field_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
) DeviceDataError!void {
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    var payload: LazyPayload(LazyOp, tag_name) = undefined;
    @field(payload, lhs_field_name) = owned_lhs;
    @field(payload, rhs_field_name) = owned_rhs;
    payload.output_prefix = owned_prefix;
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, payload));
}

pub fn appendPairOutputOptions(
    frame: anytype,
    comptime tag_name: []const u8,
    comptime lhs_field_name: []const u8,
    lhs_name: []const u8,
    comptime rhs_field_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    options: anytype,
) DeviceDataError!void {
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    var payload: LazyPayload(LazyOp, tag_name) = undefined;
    @field(payload, lhs_field_name) = owned_lhs;
    @field(payload, rhs_field_name) = owned_rhs;
    payload.output_prefix = owned_prefix;
    payload.options = options;
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, payload));
}

pub fn appendPairOutputExtraOptions(
    frame: anytype,
    comptime tag_name: []const u8,
    comptime lhs_field_name: []const u8,
    lhs_name: []const u8,
    comptime rhs_field_name: []const u8,
    rhs_name: []const u8,
    output_prefix: []const u8,
    comptime extra_field_name: []const u8,
    extra: anytype,
    options: anytype,
) DeviceDataError!void {
    const owned_lhs = try frame.allocator.dupe(u8, lhs_name);
    errdefer frame.allocator.free(owned_lhs);
    const owned_rhs = try frame.allocator.dupe(u8, rhs_name);
    errdefer frame.allocator.free(owned_rhs);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    var payload: LazyPayload(LazyOp, tag_name) = undefined;
    @field(payload, lhs_field_name) = owned_lhs;
    @field(payload, rhs_field_name) = owned_rhs;
    payload.output_prefix = owned_prefix;
    @field(payload, extra_field_name) = extra;
    payload.options = options;
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, payload));
}
