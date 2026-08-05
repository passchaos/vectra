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

            pub fn hasProjection(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn hasRangePredicate(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn hasNullPredicate(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn hasPushdown(_: DeviceParquetScan) bool {
                return false;
            }

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
