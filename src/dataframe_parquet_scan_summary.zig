//! Lightweight, Boltha-independent metadata snapshots for Parquet scan pushdown.
//!
//! `DeviceParquetScan` owns the byte buffer and mutable pushdown filters, while
//! this module keeps the immutable inspection surface shared between Boltha and
//! no-Boltha facades.  The summary borrows slices from the scan that produced it;
//! callers should treat it as a short-lived view rather than an owning value.

const std = @import("std");
const array_mod = @import("array.zig");

pub const DeviceParquetScanSummary = struct {
    device: array_mod.Device = .cpu,
    source_ptr: u64 = 0,
    source_nbytes: usize = 0,
    owned_nbytes: usize = 0,
    pushdown: DeviceParquetScanPushdownSummary = .{},

    const Self = @This();

    pub fn deviceValue(self: Self) array_mod.Device {
        return self.device;
    }

    pub fn deviceBackend(self: Self) array_mod.Backend {
        return self.device.backend;
    }

    pub fn deviceBackendName(self: Self) []const u8 {
        return self.device.backendName();
    }

    pub fn deviceIndex(self: Self) usize {
        return self.device.index;
    }

    pub fn isCpu(self: Self) bool {
        return self.device.isCpu();
    }

    pub fn isHostBacked(self: Self) bool {
        return self.isCpu();
    }

    pub fn isCuda(self: Self) bool {
        return self.device.isCuda();
    }

    pub fn isCudaBacked(self: Self) bool {
        return self.isCuda();
    }

    pub fn isMps(self: Self) bool {
        return self.device.isMps();
    }

    pub fn isMpsBacked(self: Self) bool {
        return self.isMps();
    }

    pub fn isAcceleratorBacked(self: Self) bool {
        return self.isCudaBacked() or self.isMpsBacked();
    }

    pub fn isRemoteBacked(self: Self) bool {
        return self.isAcceleratorBacked();
    }

    pub fn isDeviceBacked(self: Self) bool {
        return !self.isCpu();
    }

    pub fn isDeviceAvailable(self: Self) bool {
        return self.device.isAvailable();
    }

    pub fn sameDevice(self: Self, other: Self) bool {
        return self.device.sameDevice(other.device);
    }

    pub fn sourceNbytes(self: Self) usize {
        return self.source_nbytes;
    }

    pub fn sourcePtr(self: Self) u64 {
        return self.source_ptr;
    }

    pub fn dataPtr(self: Self) u64 {
        return self.sourcePtr();
    }

    pub fn hasSourcePtr(self: Self) bool {
        return self.source_ptr != 0;
    }

    pub fn sourceEndPtr(self: Self) u64 {
        return self.source_ptr + self.source_nbytes;
    }

    pub fn sourceRange(self: Self) SourceRange {
        return .{
            .ptr = self.source_ptr,
            .nbytes = self.source_nbytes,
        };
    }

    pub fn sourceByteCount(self: Self) usize {
        return self.sourceNbytes();
    }

    pub fn nbytes(self: Self) usize {
        return self.sourceNbytes();
    }

    pub fn byteCount(self: Self) usize {
        return self.sourceNbytes();
    }

    pub fn hasBytes(self: Self) bool {
        return self.source_nbytes != 0;
    }

    pub fn isEmpty(self: Self) bool {
        return !self.hasBytes();
    }

    pub fn isNonEmpty(self: Self) bool {
        return self.hasBytes();
    }

    pub fn pushdownSummary(self: Self) DeviceParquetScanPushdownSummary {
        return self.pushdown;
    }

    pub fn hasPushdown(self: Self) bool {
        return self.pushdown.hasPushdown();
    }

    pub fn ownedNbytes(self: Self) usize {
        return self.owned_nbytes;
    }

    pub fn memoryUsage(self: Self) usize {
        return self.owned_nbytes;
    }

    pub fn estimatedSize(self: Self) usize {
        return self.owned_nbytes;
    }
};

pub const SourceRange = struct {
    ptr: u64 = 0,
    nbytes: usize = 0,

    pub fn endPtr(self: SourceRange) u64 {
        return self.ptr + self.nbytes;
    }

    pub fn isEmpty(self: SourceRange) bool {
        return self.nbytes == 0;
    }

    pub fn isNonEmpty(self: SourceRange) bool {
        return self.nbytes != 0;
    }
};

pub const DeviceParquetScanPushdownSummary = struct {
    has_projection: bool = false,
    projection_count: usize = 0,
    projection_names: []const []const u8 = &.{},
    has_range_predicate: bool = false,
    range_predicate_column: ?[]const u8 = null,
    range_predicate_dtype: ?array_mod.DType = null,
    has_null_predicate: bool = false,
    null_predicate_column: ?[]const u8 = null,
    null_predicate_want_nulls: ?bool = null,
    projection_metadata_nbytes: usize = 0,
    range_predicate_metadata_nbytes: usize = 0,
    null_predicate_metadata_nbytes: usize = 0,
    predicate_metadata_nbytes: usize = 0,
    pushdown_metadata_nbytes: usize = 0,

    const Self = @This();

    pub fn hasPushdown(self: Self) bool {
        return self.hasProjection() or self.hasPredicate();
    }

    pub fn isEmpty(self: Self) bool {
        return !self.hasPushdown();
    }

    pub fn isNonEmpty(self: Self) bool {
        return self.hasPushdown();
    }

    pub fn hasProjection(self: Self) bool {
        return self.has_projection;
    }

    pub fn projectionColumnCount(self: Self) usize {
        return self.projection_count;
    }

    pub fn projectionNames(self: Self) []const []const u8 {
        return self.projection_names;
    }

    pub fn projectionNameAt(self: Self, index: usize) ?[]const u8 {
        if (index >= self.projection_names.len) return null;
        return self.projection_names[index];
    }

    pub fn projectionIndex(self: Self, name: []const u8) ?usize {
        for (self.projection_names, 0..) |candidate, index| {
            if (std.mem.eql(u8, candidate, name)) return index;
        }
        return null;
    }

    pub fn projectionContains(self: Self, name: []const u8) bool {
        return self.projectionIndex(name) != null;
    }

    pub fn projectsColumn(self: Self, name: []const u8) bool {
        return !self.has_projection or self.projectionContains(name);
    }

    pub fn hasPredicate(self: Self) bool {
        return self.hasRangePredicate() or self.hasNullPredicate();
    }

    pub fn predicateColumn(self: Self) ?[]const u8 {
        if (self.range_predicate_column) |column| return column;
        if (self.null_predicate_column) |column| return column;
        return null;
    }

    pub fn hasPredicateFor(self: Self, column: []const u8) bool {
        const active_column = self.predicateColumn() orelse return false;
        return std.mem.eql(u8, active_column, column);
    }

    pub fn hasRangePredicate(self: Self) bool {
        return self.has_range_predicate;
    }

    pub fn rangePredicateColumn(self: Self) ?[]const u8 {
        return self.range_predicate_column;
    }

    pub fn rangePredicateDType(self: Self) ?array_mod.DType {
        return self.range_predicate_dtype;
    }

    pub fn hasRangePredicateFor(self: Self, column: []const u8) bool {
        const active_column = self.rangePredicateColumn() orelse return false;
        return std.mem.eql(u8, active_column, column);
    }

    pub fn hasNullPredicate(self: Self) bool {
        return self.has_null_predicate;
    }

    pub fn nullPredicateColumn(self: Self) ?[]const u8 {
        return self.null_predicate_column;
    }

    pub fn nullPredicateWantNulls(self: Self) ?bool {
        return self.null_predicate_want_nulls;
    }

    pub fn hasNullPredicateFor(self: Self, column: []const u8) bool {
        const active_column = self.nullPredicateColumn() orelse return false;
        return std.mem.eql(u8, active_column, column);
    }

    pub fn projectionMetadataNbytes(self: Self) usize {
        return self.projection_metadata_nbytes;
    }

    pub fn rangePredicateMetadataNbytes(self: Self) usize {
        return self.range_predicate_metadata_nbytes;
    }

    pub fn nullPredicateMetadataNbytes(self: Self) usize {
        return self.null_predicate_metadata_nbytes;
    }

    pub fn predicateMetadataNbytes(self: Self) usize {
        return self.predicate_metadata_nbytes;
    }

    pub fn pushdownMetadataNbytes(self: Self) usize {
        return self.pushdown_metadata_nbytes;
    }
};
