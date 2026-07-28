//! Clone helpers for lazy profile operation payloads.
//!
//! Lazy profile operations have a small set of repeated ownership shapes. These
//! helpers keep `dataframe_lazy_op_ownership.zig` focused on the public clone
//! switch while centralizing the string duplication/rollback invariants here.

const std = @import("std");
const array_mod = @import("array.zig");
const series_mod = @import("series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

fn LazyPayload(comptime Self: type, comptime tag_name: []const u8) type {
    return @FieldType(Self, tag_name);
}

pub fn cloneNameOutputOptions(comptime Self: type, allocator: std.mem.Allocator, comptime tag_name: []const u8, payload: anytype) DeviceDataError!Self {
    const name = try allocator.dupe(u8, payload.name);
    errdefer allocator.free(name);
    const output_prefix = try allocator.dupe(u8, payload.output_prefix);
    errdefer allocator.free(output_prefix);
    var cloned: LazyPayload(Self, tag_name) = undefined;
    cloned.name = name;
    cloned.output_prefix = output_prefix;
    cloned.options = payload.options;
    return @unionInit(Self, tag_name, cloned);
}

pub fn cloneNameOutput(comptime Self: type, allocator: std.mem.Allocator, comptime tag_name: []const u8, payload: anytype) DeviceDataError!Self {
    const name = try allocator.dupe(u8, payload.name);
    errdefer allocator.free(name);
    const output_prefix = try allocator.dupe(u8, payload.output_prefix);
    errdefer allocator.free(output_prefix);
    var cloned: LazyPayload(Self, tag_name) = undefined;
    cloned.name = name;
    cloned.output_prefix = output_prefix;
    return @unionInit(Self, tag_name, cloned);
}

pub fn cloneNameOutputExtraOptions(
    comptime Self: type,
    allocator: std.mem.Allocator,
    comptime tag_name: []const u8,
    payload: anytype,
    comptime extra_field_name: []const u8,
) DeviceDataError!Self {
    const name = try allocator.dupe(u8, payload.name);
    errdefer allocator.free(name);
    const output_prefix = try allocator.dupe(u8, payload.output_prefix);
    errdefer allocator.free(output_prefix);
    var cloned: LazyPayload(Self, tag_name) = undefined;
    cloned.name = name;
    cloned.output_prefix = output_prefix;
    @field(cloned, extra_field_name) = @field(payload, extra_field_name);
    cloned.options = payload.options;
    return @unionInit(Self, tag_name, cloned);
}

pub fn cloneNameOutputThresholdOptions(comptime Self: type, allocator: std.mem.Allocator, comptime tag_name: []const u8, payload: anytype) DeviceDataError!Self {
    const name = try allocator.dupe(u8, payload.name);
    errdefer allocator.free(name);
    const output_prefix = try allocator.dupe(u8, payload.output_prefix);
    errdefer allocator.free(output_prefix);
    var cloned: LazyPayload(Self, tag_name) = undefined;
    cloned.name = name;
    cloned.output_prefix = output_prefix;
    cloned.threshold = payload.threshold;
    cloned.options = payload.options;
    return @unionInit(Self, tag_name, cloned);
}

pub fn cloneNameOutputThresholdExtraOptions(
    comptime Self: type,
    allocator: std.mem.Allocator,
    comptime tag_name: []const u8,
    payload: anytype,
    comptime extra_field_name: []const u8,
) DeviceDataError!Self {
    const name = try allocator.dupe(u8, payload.name);
    errdefer allocator.free(name);
    const output_prefix = try allocator.dupe(u8, payload.output_prefix);
    errdefer allocator.free(output_prefix);
    var cloned: LazyPayload(Self, tag_name) = undefined;
    cloned.name = name;
    cloned.output_prefix = output_prefix;
    cloned.threshold = payload.threshold;
    @field(cloned, extra_field_name) = @field(payload, extra_field_name);
    cloned.options = payload.options;
    return @unionInit(Self, tag_name, cloned);
}

pub fn clonePairOutput(
    comptime Self: type,
    allocator: std.mem.Allocator,
    comptime tag_name: []const u8,
    payload: anytype,
    comptime lhs_field_name: []const u8,
    comptime rhs_field_name: []const u8,
) DeviceDataError!Self {
    const lhs_name = try allocator.dupe(u8, @field(payload, lhs_field_name));
    errdefer allocator.free(lhs_name);
    const rhs_name = try allocator.dupe(u8, @field(payload, rhs_field_name));
    errdefer allocator.free(rhs_name);
    const output_prefix = try allocator.dupe(u8, payload.output_prefix);
    errdefer allocator.free(output_prefix);
    var cloned: LazyPayload(Self, tag_name) = undefined;
    @field(cloned, lhs_field_name) = lhs_name;
    @field(cloned, rhs_field_name) = rhs_name;
    cloned.output_prefix = output_prefix;
    return @unionInit(Self, tag_name, cloned);
}

pub fn clonePairOutputOptions(
    comptime Self: type,
    allocator: std.mem.Allocator,
    comptime tag_name: []const u8,
    payload: anytype,
    comptime lhs_field_name: []const u8,
    comptime rhs_field_name: []const u8,
) DeviceDataError!Self {
    const lhs_name = try allocator.dupe(u8, @field(payload, lhs_field_name));
    errdefer allocator.free(lhs_name);
    const rhs_name = try allocator.dupe(u8, @field(payload, rhs_field_name));
    errdefer allocator.free(rhs_name);
    const output_prefix = try allocator.dupe(u8, payload.output_prefix);
    errdefer allocator.free(output_prefix);
    var cloned: LazyPayload(Self, tag_name) = undefined;
    @field(cloned, lhs_field_name) = lhs_name;
    @field(cloned, rhs_field_name) = rhs_name;
    cloned.output_prefix = output_prefix;
    cloned.options = payload.options;
    return @unionInit(Self, tag_name, cloned);
}

pub fn clonePairOutputExtraOptions(
    comptime Self: type,
    allocator: std.mem.Allocator,
    comptime tag_name: []const u8,
    payload: anytype,
    comptime lhs_field_name: []const u8,
    comptime rhs_field_name: []const u8,
    comptime extra_field_name: []const u8,
) DeviceDataError!Self {
    const lhs_name = try allocator.dupe(u8, @field(payload, lhs_field_name));
    errdefer allocator.free(lhs_name);
    const rhs_name = try allocator.dupe(u8, @field(payload, rhs_field_name));
    errdefer allocator.free(rhs_name);
    const output_prefix = try allocator.dupe(u8, payload.output_prefix);
    errdefer allocator.free(output_prefix);
    var cloned: LazyPayload(Self, tag_name) = undefined;
    @field(cloned, lhs_field_name) = lhs_name;
    @field(cloned, rhs_field_name) = rhs_name;
    cloned.output_prefix = output_prefix;
    @field(cloned, extra_field_name) = @field(payload, extra_field_name);
    cloned.options = payload.options;
    return @unionInit(Self, tag_name, cloned);
}
