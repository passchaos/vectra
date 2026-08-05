//! No-Boltha lazy dataframe and Parquet scan stubs.
//!
//! This module is generic over the fallback dataframe/column types to avoid a
//! circular import with the main no-Boltha facade while preserving the same
//! public API shape.

const std = @import("std");
const array_mod = @import("array.zig");

pub fn DeviceLazyParquetTypes(
    comptime DeviceDataFrame: type,
    comptime DeviceColumn: type,
    comptime ParquetRangePredicate: type,
    comptime DeviceDataError: type,
    comptime ParquetInteropError: type,
) type {
    return struct {
        pub const DeviceLazySource = union(enum) {
            unsupported: void,
        };

        pub const DeviceLazyFrame = struct {
            pub fn init(_: std.mem.Allocator, _: DeviceDataFrame) DeviceDataError!DeviceLazyFrame {
                return error.FeatureUnavailable;
            }

            pub fn scanParquetBytes(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceLazyFrame {
                return error.FeatureUnavailable;
            }

            pub fn filterIsInValuesColumn(_: *DeviceLazyFrame, _: []const u8, _: DeviceColumn) DeviceDataError!void {
                return error.FeatureUnavailable;
            }

            pub fn filterNotInValuesColumn(_: *DeviceLazyFrame, _: []const u8, _: DeviceColumn) DeviceDataError!void {
                return error.FeatureUnavailable;
            }
        };

        pub const DeviceParquetScan = struct {
            pub fn init(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn deinit(_: *DeviceParquetScan) void {}

            pub fn clone(_: DeviceParquetScan) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn lazy(_: DeviceParquetScan) ParquetInteropError!DeviceLazyFrame {
                return error.FeatureUnavailable;
            }

            pub fn deviceValue(_: DeviceParquetScan) array_mod.Device {
                return .cpu;
            }

            pub fn deviceBackend(_: DeviceParquetScan) array_mod.Backend {
                return .cpu;
            }

            pub fn deviceBackendName(_: DeviceParquetScan) []const u8 {
                return "cpu";
            }

            pub fn deviceIndex(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn isCpu(_: DeviceParquetScan) bool {
                return true;
            }

            pub fn isHostBacked(_: DeviceParquetScan) bool {
                return true;
            }

            pub fn isCuda(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isCudaBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isMps(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isMpsBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isAcceleratorBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isRemoteBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isDeviceBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isDeviceAvailable(_: DeviceParquetScan) bool {
                return true;
            }

            pub fn sameDevice(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return true;
            }

            pub fn sourceNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn sourceByteCount(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn nbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn byteCount(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn isEmpty(_: DeviceParquetScan) bool {
                return true;
            }

            pub fn isNonEmpty(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn hasBytes(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn projectionMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn rangePredicateMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn nullPredicateMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn predicateMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn pushdownMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn ownedNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn memoryUsage(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn estimatedSize(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn hasProjection(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn projectionColumnCount(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn projectionNames(_: DeviceParquetScan) []const []const u8 {
                return &.{};
            }

            pub fn projectionNameAt(_: DeviceParquetScan, _: usize) ?[]const u8 {
                return null;
            }

            pub fn projectionIndex(_: DeviceParquetScan, _: []const u8) ?usize {
                return null;
            }

            pub fn projectionContains(_: DeviceParquetScan, _: []const u8) bool {
                return false;
            }

            pub fn projectsColumn(_: DeviceParquetScan, _: []const u8) bool {
                return true;
            }

            pub fn hasPredicate(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn predicateColumn(_: DeviceParquetScan) ?[]const u8 {
                return null;
            }

            pub fn hasPredicateFor(_: DeviceParquetScan, _: []const u8) bool {
                return false;
            }

            pub fn hasRangePredicate(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn rangePredicateColumn(_: DeviceParquetScan) ?[]const u8 {
                return null;
            }

            pub fn rangePredicate(_: DeviceParquetScan) ?ParquetRangePredicate {
                return null;
            }

            pub fn rangePredicateDType(_: DeviceParquetScan) ?array_mod.DType {
                return null;
            }

            pub fn hasRangePredicateFor(_: DeviceParquetScan, _: []const u8) bool {
                return false;
            }

            pub fn hasNullPredicate(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn nullPredicateColumn(_: DeviceParquetScan) ?[]const u8 {
                return null;
            }

            pub fn nullPredicateWantNulls(_: DeviceParquetScan) ?bool {
                return null;
            }

            pub fn hasNullPredicateFor(_: DeviceParquetScan, _: []const u8) bool {
                return false;
            }

            pub fn hasPushdown(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn clearProjection(_: *DeviceParquetScan) void {}

            pub fn clearRangePredicate(_: *DeviceParquetScan) void {}

            pub fn clearNullPredicate(_: *DeviceParquetScan) void {}

            pub fn clearPredicate(_: *DeviceParquetScan) void {}

            pub fn clearPushdown(_: *DeviceParquetScan) void {}

            pub fn resetPushdown(_: *DeviceParquetScan) void {}

            pub fn select(_: *DeviceParquetScan, _: []const []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereRange(_: *DeviceParquetScan, _: []const u8, _: ParquetRangePredicate) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereNull(_: *DeviceParquetScan, _: []const u8, _: bool) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn collect(_: DeviceParquetScan) ParquetInteropError!DeviceDataFrame {
                return error.FeatureUnavailable;
            }

            pub fn explain(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]u8 {
                return error.FeatureUnavailable;
            }
        };
    };
}
