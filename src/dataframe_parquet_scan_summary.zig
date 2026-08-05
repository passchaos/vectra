//! Lightweight, Boltha-independent metadata snapshots for Parquet scan pushdown.
//!
//! `DeviceParquetScan` owns the byte buffer and mutable pushdown filters, while
//! this module keeps the immutable inspection surface shared between Boltha and
//! no-Boltha facades.  The summary borrows slices from the scan that produced it;
//! callers should treat it as a short-lived view rather than an owning value.

const std = @import("std");
const array_mod = @import("array.zig");

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
