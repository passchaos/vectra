//! Lazy dataframe frame/source execution machinery.
//!
//! `DeviceLazyFrame` is intentionally generic over the concrete eager
//! dataframe/column types so it can live outside `dataframe.zig` without an
//! import cycle. This keeps the public facade small while preserving all lazy
//! dataframe method signatures and ownership semantics.

const std = @import("std");
const array_mod = @import("array.zig");
const lazy_exec_mod = @import("dataframe_lazy_frame_exec.zig");
const lazy_expr_mod = @import("dataframe_lazy_expr_plan.zig");
const lazy_relation_methods_mod = @import("dataframe_lazy_relation_methods.zig");
const lazy_profile_methods_mod = @import("dataframe_lazy_profile_methods.zig");
const lazy_sort_mod = @import("dataframe_lazy_sort_plan.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const names_mod = @import("dataframe_names.zig");
const options_mod = @import("dataframe_options.zig");
const parquet_scan_mod = @import("dataframe_parquet_scan.zig");
const series_mod = @import("series.zig");

const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceScalar = options_mod.DeviceScalar;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const ParquetInteropError = lazy_exec_mod.ParquetInteropError;
const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;
const freeOwnedNameItems = names_mod.freeOwnedNameItems;

pub fn DeviceLazyTypes(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
) type {
    const DeviceLazyOp = lazy_op_mod.DeviceLazyOp(DeviceDataFrame, DeviceColumn);

    return struct {
        pub const DeviceParquetScan = parquet_scan_mod.DeviceParquetScan(DeviceDataFrame, DeviceLazyFrame, DeviceColumnDef, DeviceColumn);

        pub const DeviceLazySource = union(enum) {
            dataframe: DeviceDataFrame,
            parquet_scan: DeviceParquetScan,

            fn deinit(self: *DeviceLazySource) void {
                switch (self.*) {
                    .dataframe => |*frame| frame.deinit(),
                    .parquet_scan => |*scan| scan.deinit(),
                }
                self.* = undefined;
            }

            fn clone(self: DeviceLazySource) DeviceDataError!DeviceLazySource {
                return switch (self) {
                    .dataframe => |frame| .{ .dataframe = try frame.clone() },
                    .parquet_scan => |scan| .{ .parquet_scan = try scan.clone() },
                };
            }

            pub fn name(self: DeviceLazySource) []const u8 {
                return switch (self) {
                    .dataframe => "dataframe",
                    .parquet_scan => "parquet_scan",
                };
            }
        };

        /// A compact eager-backed lazy plan for `DeviceDataFrame`.
        ///
        /// Polars' lazy API is valuable because it gives the planner a concrete list of
        /// projections, filters, and ordering operations before execution.  Vectra keeps
        /// the plan small and still executes through the existing `DeviceDataFrame`
        /// methods in `collect()`, but scan sources are represented explicitly so the
        /// planner can push conservative Parquet row-group pruning and column projection
        /// toward Boltha before materializing CPU/CUDA/MPS columns.  That gives callers a
        /// stable API today and gives Axiom a single future lowering boundary for
        /// fusing/reordering dataframe operations across CPU/CUDA/MPS.
        pub const DeviceLazyFrame = struct {
            allocator: std.mem.Allocator,
            source: DeviceLazySource,
            ops: std.ArrayList(DeviceLazyOp) = .empty,

            pub fn init(allocator: std.mem.Allocator, source: DeviceDataFrame) DeviceDataError!DeviceLazyFrame {
                return .{
                    .allocator = allocator,
                    .source = .{ .dataframe = try source.clone() },
                };
            }

            pub fn initParquetScan(allocator: std.mem.Allocator, scan: DeviceParquetScan) DeviceDataError!DeviceLazyFrame {
                return .{
                    .allocator = allocator,
                    .source = .{ .parquet_scan = try scan.clone() },
                };
            }

            pub fn scanParquetBytes(allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) DeviceDataError!DeviceLazyFrame {
                return .{
                    .allocator = allocator,
                    .source = .{ .parquet_scan = try DeviceParquetScan.init(allocator, bytes, device_value) },
                };
            }

            pub fn clone(self: DeviceLazyFrame) DeviceDataError!DeviceLazyFrame {
                var cloned = DeviceLazyFrame{
                    .allocator = self.allocator,
                    .source = try self.source.clone(),
                };
                errdefer cloned.source.deinit();
                errdefer deinitLazyOps(self.allocator, &cloned.ops);
                for (self.ops.items) |op| {
                    var cloned_op = try op.clone(self.allocator);
                    errdefer cloned_op.deinit(self.allocator);
                    try cloned.ops.append(self.allocator, cloned_op);
                }
                return cloned;
            }

            pub fn deinit(self: *DeviceLazyFrame) void {
                self.source.deinit();
                for (self.ops.items) |*op| op.deinit(self.allocator);
                self.ops.deinit(self.allocator);
                self.* = undefined;
            }

            fn sourceDevice(self: *const DeviceLazyFrame) array_mod.Device {
                return switch (self.source) {
                    .dataframe => |frame| frame.device,
                    .parquet_scan => |scan| scan.device,
                };
            }

            pub fn select(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.select(self, names);
            }

            pub fn selectByColumnIndices(self: *DeviceLazyFrame, indices: []const usize) DeviceDataError!void {
                const owned = try self.allocator.dupe(usize, indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .select_column_indices = owned });
            }

            pub fn selectColumnRange(self: *DeviceLazyFrame, start: usize, stop: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_column_range = .{
                    .start = start,
                    .stop = stop,
                } });
            }

            pub fn selectFirstColumns(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                return self.selectColumnRange(0, n);
            }

            pub fn selectLastColumns(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_last_columns = n });
            }

            pub fn dropByColumnIndices(self: *DeviceLazyFrame, indices: []const usize) DeviceDataError!void {
                const owned = try self.allocator.dupe(usize, indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .drop_column_indices = owned });
            }

            pub fn dropColumnRange(self: *DeviceLazyFrame, start: usize, stop: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_column_range = .{
                    .start = start,
                    .stop = stop,
                } });
            }

            pub fn dropFirstColumns(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                return self.dropColumnRange(0, n);
            }

            pub fn dropLastColumns(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_last_columns = n });
            }

            pub fn reverseColumns(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .reverse_columns = {} });
            }

            pub fn sortColumnsByName(self: *DeviceLazyFrame, descending: bool) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .sort_columns_by_name = .{ .descending = descending } });
            }

            pub fn selectByNamePrefix(self: *DeviceLazyFrame, prefix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.selectByNamePrefix(self, prefix);
            }

            pub fn selectByNameSuffix(self: *DeviceLazyFrame, suffix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.selectByNameSuffix(self, suffix);
            }

            pub fn selectByNameContains(self: *DeviceLazyFrame, needle: []const u8) DeviceDataError!void {
                return lazy_expr_mod.selectByNameContains(self, needle);
            }

            pub fn selectByNameGlob(self: *DeviceLazyFrame, pattern: []const u8) DeviceDataError!void {
                return lazy_expr_mod.selectByNameGlob(self, pattern);
            }

            pub fn dropByNamePrefix(self: *DeviceLazyFrame, prefix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropByNamePrefix(self, prefix);
            }

            pub fn dropByNameSuffix(self: *DeviceLazyFrame, suffix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropByNameSuffix(self, suffix);
            }

            pub fn dropByNameContains(self: *DeviceLazyFrame, needle: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropByNameContains(self, needle);
            }

            pub fn dropByNameGlob(self: *DeviceLazyFrame, pattern: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropByNameGlob(self, pattern);
            }

            pub fn selectByDTypes(self: *DeviceLazyFrame, dtypes: []const array_mod.DType) DeviceDataError!void {
                return lazy_expr_mod.selectByDTypes(self, dtypes);
            }

            pub fn selectByDTypeClass(self: *DeviceLazyFrame, class: options_mod.DeviceDTypeClass) DeviceDataError!void {
                return lazy_expr_mod.selectByDTypeClass(self, class);
            }

            pub fn dropByDTypes(self: *DeviceLazyFrame, dtypes: []const array_mod.DType) DeviceDataError!void {
                return lazy_expr_mod.dropByDTypes(self, dtypes);
            }

            pub fn dropByDTypeClass(self: *DeviceLazyFrame, class: options_mod.DeviceDTypeClass) DeviceDataError!void {
                return lazy_expr_mod.dropByDTypeClass(self, class);
            }

            pub fn selectNumeric(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.selectNumeric(self);
            }

            pub fn selectReal(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.selectReal(self);
            }

            pub fn selectFloat(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.selectFloat(self);
            }

            pub fn selectInteger(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.selectInteger(self);
            }

            pub fn selectBool(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.selectBool(self);
            }

            pub fn dropNumeric(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.dropNumeric(self);
            }

            pub fn dropReal(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.dropReal(self);
            }

            pub fn dropFloat(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.dropFloat(self);
            }

            pub fn dropInteger(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.dropInteger(self);
            }

            pub fn dropBool(self: *DeviceLazyFrame) DeviceDataError!void {
                return lazy_expr_mod.dropBool(self);
            }

            pub fn selectNullableColumns(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_nullable_columns = {} });
            }

            pub fn selectNonNullableColumns(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_non_nullable_columns = {} });
            }

            pub fn selectColumnsWithNulls(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_nulls = {} });
            }

            pub fn selectColumnsWithoutNulls(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_nulls = {} });
            }

            pub fn dropNullableColumns(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_nullable_columns = {} });
            }

            pub fn dropNonNullableColumns(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_non_nullable_columns = {} });
            }

            pub fn dropColumnsWithNulls(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_nulls = {} });
            }

            pub fn dropColumnsWithoutNulls(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_nulls = {} });
            }

            pub fn selectColumnsWithNaNs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_nans = {} });
            }

            pub fn selectColumnsWithoutNaNs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_nans = {} });
            }

            pub fn dropColumnsWithNaNs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_nans = {} });
            }

            pub fn dropColumnsWithoutNaNs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_nans = {} });
            }

            pub fn selectColumnsWithInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_infs = {} });
            }

            pub fn selectColumnsWithoutInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_infs = {} });
            }

            pub fn dropColumnsWithInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_infs = {} });
            }

            pub fn dropColumnsWithoutInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_infs = {} });
            }

            pub fn selectColumnsWithPositiveInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_positive_infs = {} });
            }

            pub fn selectColumnsWithoutPositiveInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_positive_infs = {} });
            }

            pub fn dropColumnsWithPositiveInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_positive_infs = {} });
            }

            pub fn dropColumnsWithoutPositiveInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_positive_infs = {} });
            }

            pub fn selectColumnsWithNegativeInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_negative_infs = {} });
            }

            pub fn selectColumnsWithoutNegativeInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_negative_infs = {} });
            }

            pub fn dropColumnsWithNegativeInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_negative_infs = {} });
            }

            pub fn dropColumnsWithoutNegativeInfs(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_negative_infs = {} });
            }

            pub fn selectColumnsWithZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_zeros = {} });
            }

            pub fn selectColumnsWithoutZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_zeros = {} });
            }

            pub fn dropColumnsWithZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_zeros = {} });
            }

            pub fn dropColumnsWithoutZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_zeros = {} });
            }

            pub fn selectColumnsWithPositiveZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_positive_zeros = {} });
            }

            pub fn selectColumnsWithoutPositiveZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_positive_zeros = {} });
            }

            pub fn dropColumnsWithPositiveZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_positive_zeros = {} });
            }

            pub fn dropColumnsWithoutPositiveZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_positive_zeros = {} });
            }

            pub fn selectColumnsWithNegativeZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_negative_zeros = {} });
            }

            pub fn selectColumnsWithoutNegativeZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_negative_zeros = {} });
            }

            pub fn dropColumnsWithNegativeZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_negative_zeros = {} });
            }

            pub fn dropColumnsWithoutNegativeZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_negative_zeros = {} });
            }

            pub fn selectColumnsWithNonZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_non_zeros = {} });
            }

            pub fn selectColumnsWithoutNonZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_non_zeros = {} });
            }

            pub fn dropColumnsWithNonZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_non_zeros = {} });
            }

            pub fn dropColumnsWithoutNonZeros(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_non_zeros = {} });
            }

            pub fn selectColumnsWithPositives(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_positives = {} });
            }

            pub fn selectColumnsWithoutPositives(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_positives = {} });
            }

            pub fn dropColumnsWithPositives(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_positives = {} });
            }

            pub fn dropColumnsWithoutPositives(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_positives = {} });
            }

            pub fn selectColumnsWithSignBits(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_signbits = {} });
            }

            pub fn selectColumnsWithoutSignBits(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_signbits = {} });
            }

            pub fn dropColumnsWithSignBits(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_signbits = {} });
            }

            pub fn dropColumnsWithoutSignBits(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_signbits = {} });
            }

            pub fn selectColumnsWithNegatives(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_negatives = {} });
            }

            pub fn selectColumnsWithoutNegatives(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_negatives = {} });
            }

            pub fn dropColumnsWithNegatives(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_negatives = {} });
            }

            pub fn dropColumnsWithoutNegatives(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_negatives = {} });
            }

            pub fn selectColumnsWithFinites(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_finites = {} });
            }

            pub fn selectColumnsWithoutFinites(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_finites = {} });
            }

            pub fn dropColumnsWithFinites(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_finites = {} });
            }

            pub fn dropColumnsWithoutFinites(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_finites = {} });
            }

            pub fn selectColumnsWithNormals(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_normals = {} });
            }

            pub fn selectColumnsWithoutNormals(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_normals = {} });
            }

            pub fn dropColumnsWithNormals(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_normals = {} });
            }

            pub fn dropColumnsWithoutNormals(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_normals = {} });
            }

            pub fn selectColumnsWithSubnormals(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_subnormals = {} });
            }

            pub fn selectColumnsWithoutSubnormals(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_subnormals = {} });
            }

            pub fn dropColumnsWithSubnormals(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_subnormals = {} });
            }

            pub fn dropColumnsWithoutSubnormals(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_subnormals = {} });
            }

            pub fn selectColumnsWithNonFinites(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_with_non_finites = {} });
            }

            pub fn selectColumnsWithoutNonFinites(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .select_columns_without_non_finites = {} });
            }

            pub fn dropColumnsWithNonFinites(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_with_non_finites = {} });
            }

            pub fn dropColumnsWithoutNonFinites(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_columns_without_non_finites = {} });
            }

            pub fn withRowIndex(self: *DeviceLazyFrame, name: []const u8, row_offset: usize) DeviceDataError!void {
                return lazy_expr_mod.withRowIndex(self, name, row_offset);
            }

            pub fn renameColumn(self: *DeviceLazyFrame, old_name: []const u8, new_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.renameColumn(self, old_name, new_name);
            }

            pub fn renameColumns(self: *DeviceLazyFrame, old_names: []const []const u8, new_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.renameColumns(self, old_names, new_names);
            }

            pub fn addColumnNamePrefix(self: *DeviceLazyFrame, prefix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.addColumnNamePrefix(self, prefix);
            }

            pub fn addColumnNameSuffix(self: *DeviceLazyFrame, suffix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.addColumnNameSuffix(self, suffix);
            }

            pub fn stripColumnNamePrefix(self: *DeviceLazyFrame, prefix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.stripColumnNamePrefix(self, prefix);
            }

            pub const removeColumnNamePrefix = stripColumnNamePrefix;

            pub fn stripColumnNameSuffix(self: *DeviceLazyFrame, suffix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.stripColumnNameSuffix(self, suffix);
            }

            pub const removeColumnNameSuffix = stripColumnNameSuffix;

            pub fn replaceColumnNamePrefix(self: *DeviceLazyFrame, old_prefix: []const u8, new_prefix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.replaceColumnNamePrefix(self, old_prefix, new_prefix);
            }

            pub fn replaceColumnNameSuffix(self: *DeviceLazyFrame, old_suffix: []const u8, new_suffix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.replaceColumnNameSuffix(self, old_suffix, new_suffix);
            }

            pub fn moveColumn(self: *DeviceLazyFrame, name: []const u8, target_index: usize) DeviceDataError!void {
                return lazy_expr_mod.moveColumn(self, name, target_index);
            }

            pub fn moveColumnBefore(self: *DeviceLazyFrame, name: []const u8, before_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.moveColumnBefore(self, name, before_name);
            }

            pub fn moveColumnAfter(self: *DeviceLazyFrame, name: []const u8, after_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.moveColumnAfter(self, name, after_name);
            }

            pub fn copyColumn(self: *DeviceLazyFrame, source_name: []const u8, new_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.copyColumn(self, source_name, new_name);
            }

            pub fn copyColumnAt(self: *DeviceLazyFrame, source_name: []const u8, new_name: []const u8, target_index: usize) DeviceDataError!void {
                return lazy_expr_mod.copyColumnAt(self, source_name, new_name, target_index);
            }

            pub fn copyColumnBefore(self: *DeviceLazyFrame, source_name: []const u8, new_name: []const u8, before_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.copyColumnBefore(self, source_name, new_name, before_name);
            }

            pub fn copyColumnAfter(self: *DeviceLazyFrame, source_name: []const u8, new_name: []const u8, after_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.copyColumnAfter(self, source_name, new_name, after_name);
            }

            pub fn dropColumns(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropColumns(self, names);
            }

            pub fn selectExcept(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropColumns(names);
            }

            pub const selectAllExcept = selectExcept;
            pub const excludeColumns = selectExcept;

            pub fn dropColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropColumn(self, name);
            }

            pub fn dropNulls(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNulls(self, names);
            }

            pub fn dropNullsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropNulls(names);
            }

            pub fn dropNullsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNullsColumn(self, name);
            }

            pub fn dropAllNulls(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropAllNulls(self, names);
            }

            pub fn dropAllNullsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropAllNulls(names);
            }

            pub fn filterAllNulls(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterAllNulls(self, names);
            }

            pub fn filterAllNullsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.filterAllNulls(names);
            }

            pub fn filterNullsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterNullsColumn(self, name);
            }

            pub fn dropNaNs(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNaNs(self, names);
            }

            pub fn dropNaNsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropNaNs(names);
            }

            pub fn dropNaNsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNaNsColumn(self, name);
            }

            pub fn filterNaNsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterNaNsColumn(self, name);
            }

            pub fn dropInfs(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropInfs(self, names);
            }

            pub fn dropInfsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropInfs(names);
            }

            pub fn dropInfsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropInfsColumn(self, name);
            }

            pub fn filterInfsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterInfsColumn(self, name);
            }

            pub fn dropPositiveInfs(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropPositiveInfs(self, names);
            }

            pub fn dropPositiveInfsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropPositiveInfs(names);
            }

            pub fn dropPositiveInfsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropPositiveInfsColumn(self, name);
            }

            pub fn filterPositiveInfsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterPositiveInfsColumn(self, name);
            }

            pub fn dropNegativeInfs(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNegativeInfs(self, names);
            }

            pub fn dropNegativeInfsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropNegativeInfs(names);
            }

            pub fn dropNegativeInfsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNegativeInfsColumn(self, name);
            }

            pub fn filterNegativeInfsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterNegativeInfsColumn(self, name);
            }

            pub fn dropZeros(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropZeros(self, names);
            }

            pub fn dropZerosOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropZeros(names);
            }

            pub fn dropZerosColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropZerosColumn(self, name);
            }

            pub fn filterZerosColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterZerosColumn(self, name);
            }

            pub fn dropPositiveZeros(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropPositiveZeros(self, names);
            }

            pub fn dropPositiveZerosOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropPositiveZeros(names);
            }

            pub fn dropPositiveZerosColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropPositiveZerosColumn(self, name);
            }

            pub fn filterPositiveZerosColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterPositiveZerosColumn(self, name);
            }

            pub fn dropNegativeZeros(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNegativeZeros(self, names);
            }

            pub fn dropNegativeZerosOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropNegativeZeros(names);
            }

            pub fn dropNegativeZerosColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNegativeZerosColumn(self, name);
            }

            pub fn filterNegativeZerosColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterNegativeZerosColumn(self, name);
            }

            pub fn dropNonZeros(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNonZeros(self, names);
            }

            pub fn dropNonZerosOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropNonZeros(names);
            }

            pub fn dropNonZerosColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNonZerosColumn(self, name);
            }

            pub fn filterNonZerosColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterNonZerosColumn(self, name);
            }

            pub fn dropPositives(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropPositives(self, names);
            }

            pub fn dropPositivesOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropPositives(names);
            }

            pub fn dropPositivesColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropPositivesColumn(self, name);
            }

            pub fn filterPositivesColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterPositivesColumn(self, name);
            }

            pub fn dropSignBits(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropSignBits(self, names);
            }

            pub fn dropSignBitsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropSignBits(names);
            }

            pub fn dropSignBitsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropSignBitsColumn(self, name);
            }

            pub fn filterSignBitsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterSignBitsColumn(self, name);
            }

            pub fn dropNegatives(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNegatives(self, names);
            }

            pub fn dropNegativesOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropNegatives(names);
            }

            pub fn dropNegativesColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNegativesColumn(self, name);
            }

            pub fn filterNegativesColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterNegativesColumn(self, name);
            }

            pub fn dropFinites(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropFinites(self, names);
            }

            pub fn dropFinitesOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropFinites(names);
            }

            pub fn dropFinitesColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropFinitesColumn(self, name);
            }

            pub fn filterFinitesColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterFinitesColumn(self, name);
            }

            pub fn dropNormals(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNormals(self, names);
            }

            pub fn dropNormalsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropNormals(names);
            }

            pub fn dropNormalsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNormalsColumn(self, name);
            }

            pub fn filterNormalsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterNormalsColumn(self, name);
            }

            pub fn dropSubnormals(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropSubnormals(self, names);
            }

            pub fn dropSubnormalsOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropSubnormals(names);
            }

            pub fn dropSubnormalsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropSubnormalsColumn(self, name);
            }

            pub fn filterSubnormalsColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterSubnormalsColumn(self, name);
            }

            pub fn dropNonFinites(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNonFinites(self, names);
            }

            pub fn dropNonFinitesOn(self: *DeviceLazyFrame, names: []const []const u8) DeviceDataError!void {
                return self.dropNonFinites(names);
            }

            pub fn dropNonFinitesColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNonFinitesColumn(self, name);
            }

            pub fn filterNonFinitesColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterNonFinitesColumn(self, name);
            }

            pub fn filter(self: *DeviceLazyFrame, mask: DeviceColumn) DeviceDataError!void {
                return lazy_expr_mod.filter(self, mask);
            }

            pub fn filterColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterColumn(self, name);
            }

            pub fn filterIsInColumn(self: *DeviceLazyFrame, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterIsInColumn(self, input_name, test_name);
            }

            pub fn filterNotInColumn(self: *DeviceLazyFrame, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.filterNotInColumn(self, input_name, test_name);
            }

            pub const filterIsinColumn = filterIsInColumn;
            pub const filterIsInColumnInverted = filterNotInColumn;
            pub const filterIsinColumnInverted = filterNotInColumn;

            pub fn filterIsInValues(self: *DeviceLazyFrame, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!void {
                var value_column = try DeviceColumn.fromSlice(T, self.allocator, values, self.sourceDevice());
                defer value_column.deinit();
                return lazy_expr_mod.filterIsInValuesColumn(self, input_name, value_column);
            }

            pub fn filterNotInValues(self: *DeviceLazyFrame, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!void {
                var value_column = try DeviceColumn.fromSlice(T, self.allocator, values, self.sourceDevice());
                defer value_column.deinit();
                return lazy_expr_mod.filterNotInValuesColumn(self, input_name, value_column);
            }

            pub const filterIsinValues = filterIsInValues;
            pub const filterIsInValuesInverted = filterNotInValues;
            pub const filterIsinValuesInverted = filterNotInValues;

            pub fn dropIsInColumn(self: *DeviceLazyFrame, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropIsInColumn(self, input_name, test_name);
            }

            pub fn dropNotInColumn(self: *DeviceLazyFrame, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropNotInColumn(self, input_name, test_name);
            }

            pub const dropIsinColumn = dropIsInColumn;
            pub const dropIsInColumnInverted = dropNotInColumn;
            pub const dropIsinColumnInverted = dropNotInColumn;

            pub fn dropIsInValues(self: *DeviceLazyFrame, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!void {
                var value_column = try DeviceColumn.fromSlice(T, self.allocator, values, self.sourceDevice());
                defer value_column.deinit();
                return lazy_expr_mod.dropIsInValuesColumn(self, input_name, value_column);
            }

            pub fn dropNotInValues(self: *DeviceLazyFrame, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!void {
                var value_column = try DeviceColumn.fromSlice(T, self.allocator, values, self.sourceDevice());
                defer value_column.deinit();
                return lazy_expr_mod.dropNotInValuesColumn(self, input_name, value_column);
            }

            pub const dropIsinValues = dropIsInValues;
            pub const dropIsInValuesInverted = dropNotInValues;
            pub const dropIsinValuesInverted = dropNotInValues;

            pub fn filterBetweenColumnWithDeviceScalars(self: *DeviceLazyFrame, name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
                return lazy_expr_mod.filterBetweenColumnWithDeviceScalars(self, name, lower, upper, lower_inclusive, upper_inclusive);
            }

            pub fn filterBetweenColumnClosed(self: *DeviceLazyFrame, name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
                return lazy_expr_mod.filterBetweenColumnClosed(self, name, T, lower, upper, lower_inclusive, upper_inclusive);
            }

            pub fn filterBetweenColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.filterBetweenColumn(self, name, T, lower, upper);
            }

            pub fn filterOutsideColumnWithDeviceScalars(self: *DeviceLazyFrame, name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
                return lazy_expr_mod.filterOutsideColumnWithDeviceScalars(self, name, lower, upper, lower_inclusive, upper_inclusive);
            }

            pub fn filterOutsideColumnClosed(self: *DeviceLazyFrame, name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
                return lazy_expr_mod.filterOutsideColumnClosed(self, name, T, lower, upper, lower_inclusive, upper_inclusive);
            }

            pub fn filterOutsideColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.filterOutsideColumn(self, name, T, lower, upper);
            }

            pub fn dropBetweenColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.dropBetweenColumn(self, name, T, lower, upper);
            }

            pub fn dropOutsideColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.dropOutsideColumn(self, name, T, lower, upper);
            }

            pub fn dropRowsByColumnMask(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropRowsByColumnMask(self, name);
            }

            pub fn whereIndicesColumn(self: *DeviceLazyFrame, mask_name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.whereIndicesColumn(self, mask_name, output_name);
            }

            pub fn argwhereColumn(self: *DeviceLazyFrame, mask_name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.argwhereColumn(self, mask_name, output_name);
            }

            pub fn withColumnAbs(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnAbs(self, name, input_name);
            }

            pub fn withColumnNeg(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNeg(self, name, input_name);
            }

            pub fn withColumnNegative(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNegative(self, name, input_name);
            }

            pub fn withColumnSquare(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSquare(self, name, input_name);
            }

            pub fn withColumnReciprocal(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnReciprocal(self, name, input_name);
            }

            pub fn withColumnSign(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSign(self, name, input_name);
            }

            pub fn withColumnSqrt(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSqrt(self, name, input_name);
            }

            pub fn withColumnRsqrt(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnRsqrt(self, name, input_name);
            }

            pub fn withColumnCbrt(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnCbrt(self, name, input_name);
            }

            pub fn withColumnFloor(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnFloor(self, name, input_name);
            }

            pub fn withColumnCeil(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnCeil(self, name, input_name);
            }

            pub fn withColumnRound(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnRound(self, name, input_name);
            }

            pub fn withColumnTrunc(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnTrunc(self, name, input_name);
            }

            pub fn withColumnDeg2rad(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnDeg2rad(self, name, input_name);
            }

            pub fn withColumnRad2deg(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnRad2deg(self, name, input_name);
            }

            pub fn withColumnExpit(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnExpit(self, name, input_name);
            }

            pub fn withColumnLogit(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogit(self, name, input_name);
            }

            pub fn withColumnSoftplus(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSoftplus(self, name, input_name);
            }

            pub fn withColumnLogsigmoid(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogsigmoid(self, name, input_name);
            }

            pub fn withColumnRelu(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnRelu(self, name, input_name);
            }

            pub fn withColumnLeakyRelu(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, negative_slope: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnLeakyRelu(self, name, input_name, T, negative_slope);
            }

            pub fn withColumnLeakyReluWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, negative_slope: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnLeakyReluWithDeviceScalar(self, name, input_name, negative_slope);
            }

            pub fn withColumnRelu6(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnRelu6(self, name, input_name);
            }

            pub fn withColumnPowScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, exponent: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnPowScalar(self, name, input_name, T, exponent);
            }

            pub fn withColumnPowWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, exponent: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnPowWithDeviceScalar(self, name, input_name, exponent);
            }

            pub fn withColumnFloorDivScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFloorDivScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnFloorDivWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFloorDivWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnModScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnModScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnModWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnModWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnRemainderScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnRemainderScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnRemainderWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnRemainderWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnLogAddExpScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogAddExpScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnLogAddExpWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogAddExpWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnLogAddExp2Scalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogAddExp2Scalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnLogAddExp2WithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogAddExp2WithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnXlogyScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnXlogyScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnXlogyWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnXlogyWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnFmaxScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFmaxScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnFmaxWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFmaxWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnFminScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFminScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnFminWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFminWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnHypotScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnHypotScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnHypotWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnHypotWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnAtan2Scalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnAtan2Scalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnAtan2WithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnAtan2WithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnNextAfterScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnNextAfterScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnNextAfterWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnNextAfterWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnCopysignScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnCopysignScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnCopysignWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnCopysignWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnHeavisideScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, value_at_zero: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnHeavisideScalar(self, name, input_name, T, value_at_zero);
            }

            pub fn withColumnHeavisideWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, value_at_zero: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnHeavisideWithDeviceScalar(self, name, input_name, value_at_zero);
            }

            pub fn withColumnLdexpScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, exponent: i32) DeviceDataError!void {
                return lazy_expr_mod.withColumnLdexpScalar(self, name, input_name, exponent);
            }

            pub fn withColumnThreshold(
                self: *DeviceLazyFrame,
                name: []const u8,
                input_name: []const u8,
                comptime T: type,
                threshold_value: T,
                replacement_value: T,
            ) DeviceDataError!void {
                return lazy_expr_mod.withColumnThreshold(self, name, input_name, T, threshold_value, replacement_value);
            }

            pub fn withColumnThresholdWithDeviceScalars(
                self: *DeviceLazyFrame,
                name: []const u8,
                input_name: []const u8,
                threshold_value: DeviceScalar,
                replacement_value: DeviceScalar,
            ) DeviceDataError!void {
                return lazy_expr_mod.withColumnThresholdWithDeviceScalars(self, name, input_name, threshold_value, replacement_value);
            }

            pub fn withColumnHardtanh(
                self: *DeviceLazyFrame,
                name: []const u8,
                input_name: []const u8,
                comptime T: type,
                min_value: T,
                max_value: T,
            ) DeviceDataError!void {
                return lazy_expr_mod.withColumnHardtanh(self, name, input_name, T, min_value, max_value);
            }

            pub fn withColumnHardtanhWithDeviceScalars(
                self: *DeviceLazyFrame,
                name: []const u8,
                input_name: []const u8,
                min_value: DeviceScalar,
                max_value: DeviceScalar,
            ) DeviceDataError!void {
                return lazy_expr_mod.withColumnHardtanhWithDeviceScalars(self, name, input_name, min_value, max_value);
            }

            pub fn withColumnBetween(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnBetween(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnIsBetween(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnIsBetween(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnBetweenClosed(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
                return lazy_expr_mod.withColumnBetweenClosed(self, name, input_name, T, lower, upper, lower_inclusive, upper_inclusive);
            }

            pub fn withColumnBetweenExclusive(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnBetweenExclusive(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnBetweenLeftClosed(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnBetweenLeftClosed(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnBetweenRightClosed(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnBetweenRightClosed(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnBetweenWithDeviceScalars(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
                return lazy_expr_mod.withColumnBetweenWithDeviceScalars(self, name, input_name, lower, upper, lower_inclusive, upper_inclusive);
            }

            pub fn withColumnNotBetween(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnNotBetween(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnOutside(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnOutside(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnNotBetweenClosed(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
                return lazy_expr_mod.withColumnNotBetweenClosed(self, name, input_name, T, lower, upper, lower_inclusive, upper_inclusive);
            }

            pub fn withColumnNotBetweenExclusive(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnNotBetweenExclusive(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnNotBetweenLeftClosed(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnNotBetweenLeftClosed(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnNotBetweenRightClosed(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lower: T, upper: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnNotBetweenRightClosed(self, name, input_name, T, lower, upper);
            }

            pub fn withColumnNotBetweenWithDeviceScalars(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) DeviceDataError!void {
                return lazy_expr_mod.withColumnNotBetweenWithDeviceScalars(self, name, input_name, lower, upper, lower_inclusive, upper_inclusive);
            }

            pub fn withColumnMaximumScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnMaximumScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnMaximumWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnMaximumWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnMinimumScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnMinimumScalar(self, name, input_name, T, scalar);
            }

            pub fn withColumnMinimumWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnMinimumWithDeviceScalar(self, name, input_name, scalar);
            }

            pub fn withColumnClipMin(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, min_value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnClipMin(self, name, input_name, T, min_value);
            }

            pub fn withColumnClipMinWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, min_value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnClipMinWithDeviceScalar(self, name, input_name, min_value);
            }

            pub fn withColumnClipMax(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, max_value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnClipMax(self, name, input_name, T, max_value);
            }

            pub fn withColumnClipMaxWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, max_value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnClipMaxWithDeviceScalar(self, name, input_name, max_value);
            }

            pub fn withColumnHardshrink(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lambd: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnHardshrink(self, name, input_name, T, lambd);
            }

            pub fn withColumnHardshrinkWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, lambd: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnHardshrinkWithDeviceScalar(self, name, input_name, lambd);
            }

            pub fn withColumnSoftshrink(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, lambd: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnSoftshrink(self, name, input_name, T, lambd);
            }

            pub fn withColumnSoftshrinkWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, lambd: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnSoftshrinkWithDeviceScalar(self, name, input_name, lambd);
            }

            pub fn withColumnTanhshrink(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnTanhshrink(self, name, input_name);
            }

            pub fn withColumnElu(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, alpha: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnElu(self, name, input_name, T, alpha);
            }

            pub fn withColumnEluWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, alpha: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnEluWithDeviceScalar(self, name, input_name, alpha);
            }

            pub fn withColumnCelu(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, alpha: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnCelu(self, name, input_name, T, alpha);
            }

            pub fn withColumnCeluWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, alpha: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnCeluWithDeviceScalar(self, name, input_name, alpha);
            }

            pub fn withColumnSoftsign(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSoftsign(self, name, input_name);
            }

            pub fn withColumnHardsigmoid(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnHardsigmoid(self, name, input_name);
            }

            pub fn withColumnHardswish(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnHardswish(self, name, input_name);
            }

            pub fn withColumnSilu(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSilu(self, name, input_name);
            }

            pub fn withColumnSwish(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSwish(self, name, input_name);
            }

            pub fn withColumnMish(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnMish(self, name, input_name);
            }

            pub fn withColumnGelu(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnGelu(self, name, input_name);
            }

            pub fn withColumnSelu(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSelu(self, name, input_name);
            }

            pub fn withColumnExp(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnExp(self, name, input_name);
            }

            pub fn withColumnExp2(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnExp2(self, name, input_name);
            }

            pub fn withColumnExpm1(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnExpm1(self, name, input_name);
            }

            pub fn withColumnSin(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSin(self, name, input_name);
            }

            pub fn withColumnCos(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnCos(self, name, input_name);
            }

            pub fn withColumnTan(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnTan(self, name, input_name);
            }

            pub fn withColumnAsin(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnAsin(self, name, input_name);
            }

            pub fn withColumnAcos(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnAcos(self, name, input_name);
            }

            pub fn withColumnAtan(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnAtan(self, name, input_name);
            }

            pub fn withColumnSinh(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSinh(self, name, input_name);
            }

            pub fn withColumnCosh(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnCosh(self, name, input_name);
            }

            pub fn withColumnTanh(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnTanh(self, name, input_name);
            }

            pub fn withColumnAsinh(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnAsinh(self, name, input_name);
            }

            pub fn withColumnAcosh(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnAcosh(self, name, input_name);
            }

            pub fn withColumnAtanh(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnAtanh(self, name, input_name);
            }

            pub fn withColumnLog(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLog(self, name, input_name);
            }

            pub fn withColumnLog1p(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLog1p(self, name, input_name);
            }

            pub fn withColumnLgamma(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLgamma(self, name, input_name);
            }

            pub fn withColumnSinc(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSinc(self, name, input_name);
            }

            pub fn withColumnLog2(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLog2(self, name, input_name);
            }

            pub fn withColumnLog10(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLog10(self, name, input_name);
            }

            pub fn withColumnBinary(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnBinaryOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnBinary(self, name, lhs_name, rhs_name, op);
            }

            pub fn withColumnScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnBinaryOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnScalar(self, name, input_name, T, scalar, op);
            }

            pub fn withColumnLerpScalar(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, comptime T: type, weight: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnLerpScalar(self, name, lhs_name, rhs_name, T, weight);
            }

            pub fn withColumnLerpWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, weight: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnLerpWithDeviceScalar(self, name, lhs_name, rhs_name, weight);
            }

            pub fn withColumnAddcmulScalar(self: *DeviceLazyFrame, name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnAddcmulScalar(self, name, base_name, input1_name, input2_name, T, value);
            }

            pub fn withColumnAddcmulWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnAddcmulWithDeviceScalar(self, name, base_name, input1_name, input2_name, value);
            }

            pub fn withColumnAddcdivScalar(self: *DeviceLazyFrame, name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnAddcdivScalar(self, name, base_name, input1_name, input2_name, T, value);
            }

            pub fn withColumnAddcdivWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, base_name: []const u8, input1_name: []const u8, input2_name: []const u8, value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnAddcdivWithDeviceScalar(self, name, base_name, input1_name, input2_name, value);
            }

            pub fn withColumnClipArray(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, min_name: []const u8, max_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnClipArray(self, name, input_name, min_name, max_name);
            }

            pub fn withColumnWhereScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, mask_name: []const u8, comptime T: type, other_value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnWhereScalar(self, name, input_name, mask_name, T, other_value);
            }

            pub fn withColumnWhereWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, mask_name: []const u8, other_value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnWhereWithDeviceScalar(self, name, input_name, mask_name, other_value);
            }

            pub fn withColumnWhere(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, mask_name: []const u8, other_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnWhere(self, name, input_name, mask_name, other_name);
            }

            pub fn withColumnIsIn(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnIsIn(self, name, input_name, test_name);
            }

            pub fn withColumnIsInInverted(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, test_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnIsInInverted(self, name, input_name, test_name);
            }

            pub const withColumnIsin = withColumnIsIn;
            pub const withColumnIsinInverted = withColumnIsInInverted;

            pub fn withColumnIsInValues(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!void {
                var value_column = try DeviceColumn.fromSlice(T, self.allocator, values, self.sourceDevice());
                defer value_column.deinit();
                return lazy_expr_mod.withColumnIsInValuesColumn(self, name, input_name, value_column);
            }

            pub fn withColumnIsInValuesInverted(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!void {
                var value_column = try DeviceColumn.fromSlice(T, self.allocator, values, self.sourceDevice());
                defer value_column.deinit();
                return lazy_expr_mod.withColumnIsInValuesInvertedColumn(self, name, input_name, value_column);
            }

            pub const withColumnIsinValues = withColumnIsInValues;
            pub const withColumnIsinValuesInverted = withColumnIsInValuesInverted;

            pub fn withColumnMaskedPutScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, mask_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnMaskedPutScalar(self, name, input_name, mask_name, T, value);
            }

            pub fn withColumnMaskedPutWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, mask_name: []const u8, value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnMaskedPutWithDeviceScalar(self, name, input_name, mask_name, value);
            }

            pub fn withColumnPutMaskScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, mask_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnPutMaskScalar(self, name, input_name, mask_name, T, value);
            }

            pub fn withColumnPutMaskWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, mask_name: []const u8, value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnPutMaskWithDeviceScalar(self, name, input_name, mask_name, value);
            }

            pub fn withColumnPutFlatScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnPutFlatScalar(self, name, input_name, row_indices, T, value);
            }

            pub fn withColumnPutFlatWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const usize, value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnPutFlatWithDeviceScalar(self, name, input_name, row_indices, value);
            }

            pub fn withColumnPutFlat(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const usize, value_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnPutFlat(self, name, input_name, row_indices, value_name);
            }

            pub fn withColumnIndexPut(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const usize, value_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnIndexPut(self, name, input_name, row_indices, value_name);
            }

            pub fn withColumnPutFlatScalarMode(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T, mode: array_mod.IndexMode) DeviceDataError!void {
                return lazy_expr_mod.withColumnPutFlatScalarMode(self, name, input_name, row_indices, T, value, mode);
            }

            pub fn withColumnPutFlatModeWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const usize, value: DeviceScalar, mode: array_mod.IndexMode) DeviceDataError!void {
                return lazy_expr_mod.withColumnPutFlatModeWithDeviceScalar(self, name, input_name, row_indices, value, mode);
            }

            pub fn withColumnIndexPutScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const usize, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnIndexPutScalar(self, name, input_name, row_indices, T, value);
            }

            pub fn withColumnIndexPutWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const usize, value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnIndexPutWithDeviceScalar(self, name, input_name, row_indices, value);
            }

            pub fn withColumnPutFlatScalarSigned(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const isize, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnPutFlatScalarSigned(self, name, input_name, row_indices, T, value);
            }

            pub fn withColumnPutFlatSignedWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const isize, value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnPutFlatSignedWithDeviceScalar(self, name, input_name, row_indices, value);
            }

            pub fn withColumnIndexPutScalarSigned(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const isize, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnIndexPutScalarSigned(self, name, input_name, row_indices, T, value);
            }

            pub fn withColumnIndexPutSignedWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, row_indices: []const isize, value: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnIndexPutSignedWithDeviceScalar(self, name, input_name, row_indices, value);
            }

            pub fn withColumnIscloseScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnIscloseScalar(self, name, input_name, T, scalar, rtol, atol);
            }

            pub fn withColumnIscloseScalarEqualNan(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, rtol: T, atol: T, equal_nan: bool) DeviceDataError!void {
                return lazy_expr_mod.withColumnIscloseScalarEqualNan(self, name, input_name, T, scalar, rtol, atol, equal_nan);
            }

            pub fn withColumnIscloseWithDeviceScalars(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnIscloseWithDeviceScalars(self, name, input_name, scalar, rtol, atol);
            }

            pub fn withColumnIscloseWithDeviceScalarsEqualNan(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: DeviceScalar, rtol: DeviceScalar, atol: DeviceScalar, equal_nan: bool) DeviceDataError!void {
                return lazy_expr_mod.withColumnIscloseWithDeviceScalarsEqualNan(self, name, input_name, scalar, rtol, atol, equal_nan);
            }

            pub fn withColumnLogicalScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: bool, op: options_mod.DeviceColumnLogicalOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogicalScalar(self, name, input_name, scalar, op);
            }

            pub fn withColumnLogicalAndScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: bool) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogicalAndScalar(self, name, input_name, scalar);
            }

            pub fn withColumnLogicalOrScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: bool) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogicalOrScalar(self, name, input_name, scalar);
            }

            pub fn withColumnLogicalXorScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, scalar: bool) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogicalXorScalar(self, name, input_name, scalar);
            }

            pub fn withColumnLogicalNot(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogicalNot(self, name, input_name);
            }

            pub fn withColumnNot(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNot(self, name, input_name);
            }

            pub fn withColumnLogical(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: options_mod.DeviceColumnLogicalOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogical(self, name, lhs_name, rhs_name, op);
            }

            pub fn withColumnLogicalAnd(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogicalAnd(self, name, lhs_name, rhs_name);
            }

            pub fn withColumnLogicalOr(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogicalOr(self, name, lhs_name, rhs_name);
            }

            pub fn withColumnLogicalXor(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLogicalXor(self, name, lhs_name, rhs_name);
            }

            pub fn withColumnLiteral(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnLiteral(self, name, T, value);
            }

            pub fn withColumnLiteralScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnLiteralScalar(self, name, scalar);
            }

            pub fn withColumnLiteralAt(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T, target_index: usize) DeviceDataError!void {
                return lazy_expr_mod.withColumnLiteralAt(self, name, T, value, target_index);
            }

            pub fn withColumnLiteralBefore(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T, before_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLiteralBefore(self, name, T, value, before_name);
            }

            pub fn withColumnLiteralAfter(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T, after_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLiteralAfter(self, name, T, value, after_name);
            }

            pub fn withColumnLiteralScalarAt(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar, target_index: usize) DeviceDataError!void {
                return lazy_expr_mod.withColumnLiteralScalarAt(self, name, scalar, target_index);
            }

            pub fn withColumnLiteralScalarBefore(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar, before_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLiteralScalarBefore(self, name, scalar, before_name);
            }

            pub fn withColumnLiteralScalarAfter(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar, after_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLiteralScalarAfter(self, name, scalar, after_name);
            }

            pub fn castColumn(self: *DeviceLazyFrame, name: []const u8, dtype_value: array_mod.DType) DeviceDataError!void {
                return lazy_expr_mod.castColumn(self, name, dtype_value);
            }

            pub fn fillNullColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillNullColumn(self, name, T, value);
            }

            pub fn fillNullColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillNullColumnWithScalar(self, name, scalar);
            }

            pub fn withColumnFillNull(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNull(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillNullScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNullScalar(self, output_name, input_name, scalar);
            }

            pub fn fillNullForwardColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.fillNullForwardColumn(self, name);
            }

            pub fn fillNullBackwardColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.fillNullBackwardColumn(self, name);
            }

            pub fn withColumnFillNullForward(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNullForward(self, output_name, input_name);
            }

            pub fn withColumnFillNullBackward(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNullBackward(self, output_name, input_name);
            }

            pub fn nullIfColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.nullIfColumn(self, name, T, value);
            }

            pub fn nullIfColumnScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.nullIfColumnScalar(self, name, scalar);
            }

            pub fn nullIfValuesColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, values: []const T) DeviceDataError!void {
                var value_column = try DeviceColumn.fromSlice(T, self.allocator, values, self.sourceDevice());
                defer value_column.deinit();
                return lazy_expr_mod.nullIfValuesColumnWithDeviceColumn(self, name, value_column);
            }

            pub fn withColumnNullIf(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIf(self, output_name, input_name, T, value);
            }

            pub fn withColumnNullIfScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnNullIfValues(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, values: []const T) DeviceDataError!void {
                var value_column = try DeviceColumn.fromSlice(T, self.allocator, values, self.sourceDevice());
                defer value_column.deinit();
                return lazy_expr_mod.withColumnNullIfValuesWithDeviceColumn(self, output_name, input_name, value_column);
            }

            pub fn nullIfNaNColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfNaNColumn(self, name);
            }

            pub fn withColumnNullIfNaN(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfNaN(self, output_name, input_name);
            }

            pub fn nullIfInfColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfInfColumn(self, name);
            }

            pub fn withColumnNullIfInf(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfInf(self, output_name, input_name);
            }

            pub fn nullIfPositiveInfColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfPositiveInfColumn(self, name);
            }

            pub fn withColumnNullIfPositiveInf(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfPositiveInf(self, output_name, input_name);
            }

            pub fn nullIfNegativeInfColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfNegativeInfColumn(self, name);
            }

            pub fn withColumnNullIfNegativeInf(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfNegativeInf(self, output_name, input_name);
            }

            pub fn nullIfZeroColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfZeroColumn(self, name);
            }

            pub fn withColumnNullIfZero(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfZero(self, output_name, input_name);
            }

            pub fn nullIfPositiveZeroColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfPositiveZeroColumn(self, name);
            }

            pub fn withColumnNullIfPositiveZero(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfPositiveZero(self, output_name, input_name);
            }

            pub fn nullIfNegativeZeroColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfNegativeZeroColumn(self, name);
            }

            pub fn withColumnNullIfNegativeZero(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfNegativeZero(self, output_name, input_name);
            }

            pub fn nullIfNonZeroColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfNonZeroColumn(self, name);
            }

            pub fn withColumnNullIfNonZero(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfNonZero(self, output_name, input_name);
            }

            pub fn nullIfPositiveColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfPositiveColumn(self, name);
            }

            pub fn withColumnNullIfPositive(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfPositive(self, output_name, input_name);
            }

            pub fn nullIfSignBitColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfSignBitColumn(self, name);
            }

            pub fn withColumnNullIfSignBit(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfSignBit(self, output_name, input_name);
            }

            pub fn nullIfNegativeColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfNegativeColumn(self, name);
            }

            pub fn withColumnNullIfNegative(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfNegative(self, output_name, input_name);
            }

            pub fn nullIfFiniteColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfFiniteColumn(self, name);
            }

            pub fn withColumnNullIfFinite(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfFinite(self, output_name, input_name);
            }

            pub fn nullIfNormalColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfNormalColumn(self, name);
            }

            pub fn withColumnNullIfNormal(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfNormal(self, output_name, input_name);
            }

            pub fn nullIfSubnormalColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfSubnormalColumn(self, name);
            }

            pub fn withColumnNullIfSubnormal(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfSubnormal(self, output_name, input_name);
            }

            pub fn nullIfNonFiniteColumn(self: *DeviceLazyFrame, name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.nullIfNonFiniteColumn(self, name);
            }

            pub fn withColumnNullIfNonFinite(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnNullIfNonFinite(self, output_name, input_name);
            }

            pub fn withColumnFillNaN(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNaN(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillNaNScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNaNScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillInf(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillInf(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillInfScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillInfScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillPositiveInf(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillPositiveInf(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillPositiveInfScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillPositiveInfScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillNegativeInf(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNegativeInf(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillNegativeInfScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNegativeInfScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillZero(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillZero(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillZeroScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillZeroScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillPositiveZero(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillPositiveZero(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillPositiveZeroScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillPositiveZeroScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillNegativeZero(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNegativeZero(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillNegativeZeroScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNegativeZeroScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillNonZero(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNonZero(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillNonZeroScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNonZeroScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillPositive(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillPositive(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillPositiveScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillPositiveScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillSignBit(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillSignBit(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillSignBitScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillSignBitScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillNegative(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNegative(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillNegativeScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNegativeScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillFinite(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillFinite(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillFiniteScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillFiniteScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillNormal(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNormal(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillNormalScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNormalScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillSubnormal(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillSubnormal(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillSubnormalScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillSubnormalScalar(self, output_name, input_name, scalar);
            }

            pub fn withColumnFillNonFinite(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNonFinite(self, output_name, input_name, T, value);
            }

            pub fn withColumnFillNonFiniteScalar(self: *DeviceLazyFrame, output_name: []const u8, input_name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.withColumnFillNonFiniteScalar(self, output_name, input_name, scalar);
            }

            pub fn fillNaNColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillNaNColumn(self, name, T, value);
            }

            pub fn fillNaNColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillNaNColumnWithScalar(self, name, scalar);
            }

            pub fn fillInfColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillInfColumn(self, name, T, value);
            }

            pub fn fillInfColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillInfColumnWithScalar(self, name, scalar);
            }

            pub fn fillPositiveInfColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillPositiveInfColumn(self, name, T, value);
            }

            pub fn fillPositiveInfColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillPositiveInfColumnWithScalar(self, name, scalar);
            }

            pub fn fillNegativeInfColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillNegativeInfColumn(self, name, T, value);
            }

            pub fn fillNegativeInfColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillNegativeInfColumnWithScalar(self, name, scalar);
            }

            pub fn fillZeroColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillZeroColumn(self, name, T, value);
            }

            pub fn fillZeroColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillZeroColumnWithScalar(self, name, scalar);
            }

            pub fn fillPositiveZeroColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillPositiveZeroColumn(self, name, T, value);
            }

            pub fn fillPositiveZeroColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillPositiveZeroColumnWithScalar(self, name, scalar);
            }

            pub fn fillNegativeZeroColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillNegativeZeroColumn(self, name, T, value);
            }

            pub fn fillNegativeZeroColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillNegativeZeroColumnWithScalar(self, name, scalar);
            }

            pub fn fillNonZeroColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillNonZeroColumn(self, name, T, value);
            }

            pub fn fillNonZeroColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillNonZeroColumnWithScalar(self, name, scalar);
            }

            pub fn fillPositiveColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillPositiveColumn(self, name, T, value);
            }

            pub fn fillPositiveColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillPositiveColumnWithScalar(self, name, scalar);
            }

            pub fn fillSignBitColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillSignBitColumn(self, name, T, value);
            }

            pub fn fillSignBitColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillSignBitColumnWithScalar(self, name, scalar);
            }

            pub fn fillNegativeColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillNegativeColumn(self, name, T, value);
            }

            pub fn fillNegativeColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillNegativeColumnWithScalar(self, name, scalar);
            }

            pub fn fillFiniteColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillFiniteColumn(self, name, T, value);
            }

            pub fn fillFiniteColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillFiniteColumnWithScalar(self, name, scalar);
            }

            pub fn fillNormalColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillNormalColumn(self, name, T, value);
            }

            pub fn fillNormalColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillNormalColumnWithScalar(self, name, scalar);
            }

            pub fn fillSubnormalColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillSubnormalColumn(self, name, T, value);
            }

            pub fn fillSubnormalColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillSubnormalColumnWithScalar(self, name, scalar);
            }

            pub fn fillNonFiniteColumn(self: *DeviceLazyFrame, name: []const u8, comptime T: type, value: T) DeviceDataError!void {
                return lazy_expr_mod.fillNonFiniteColumn(self, name, T, value);
            }

            pub fn fillNonFiniteColumnWithScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar) DeviceDataError!void {
                return lazy_expr_mod.fillNonFiniteColumnWithScalar(self, name, scalar);
            }

            pub fn coalesceColumns(self: *DeviceLazyFrame, primary_name: []const u8, fallback_name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.coalesceColumns(self, primary_name, fallback_name, output_name);
            }

            pub fn coalesceColumnsMany(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.coalesceColumnsMany(self, names, output_name);
            }

            pub fn coalesceManyColumns(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.coalesceManyColumns(self, names, output_name);
            }

            pub fn coalesceFirstValidColumns(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.coalesceFirstValidColumns(self, names, output_name);
            }

            pub fn isNullColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isNullColumn(self, name, output_name);
            }

            pub fn isValidColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isValidColumn(self, name, output_name);
            }

            pub fn isNanColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isNanColumn(self, name, output_name);
            }

            pub fn isZeroColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isZeroColumn(self, name, output_name);
            }

            pub fn isPositiveZeroColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isPositiveZeroColumn(self, name, output_name);
            }

            pub fn isNegativeZeroColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isNegativeZeroColumn(self, name, output_name);
            }

            pub fn isNonZeroColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isNonZeroColumn(self, name, output_name);
            }

            pub fn isPositiveColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isPositiveColumn(self, name, output_name);
            }

            pub fn isSignBitColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isSignBitColumn(self, name, output_name);
            }

            pub fn isNegativeColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isNegativeColumn(self, name, output_name);
            }

            pub fn isFiniteColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isFiniteColumn(self, name, output_name);
            }

            pub fn isNormalColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isNormalColumn(self, name, output_name);
            }

            pub fn isSubnormalColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isSubnormalColumn(self, name, output_name);
            }

            pub fn isNonFiniteColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isNonFiniteColumn(self, name, output_name);
            }

            pub fn isInfColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isInfColumn(self, name, output_name);
            }

            pub fn isPositiveInfColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isPositiveInfColumn(self, name, output_name);
            }

            pub fn isNegativeInfColumn(self: *DeviceLazyFrame, name: []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.isNegativeInfColumn(self, name, output_name);
            }

            pub fn withRowNullCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNullCount(self, names, output_name);
            }

            pub fn withRowValidCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowValidCount(self, names, output_name);
            }

            pub fn withRowAnyNull(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyNull(self, names, output_name);
            }

            pub fn withRowAllNull(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllNull(self, names, output_name);
            }

            pub fn withRowAnyValid(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyValid(self, names, output_name);
            }

            pub fn withRowAllValid(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllValid(self, names, output_name);
            }

            pub fn withRowCumulativeAnyNull(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyNull(self, names, output_names);
            }

            pub fn withRowCumAnyNull(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyNull(self, names, output_names);
            }

            pub fn withRowPrefixAnyNull(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyNull(self, names, output_names);
            }

            pub fn withRowCumulativeAllNull(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllNull(self, names, output_names);
            }

            pub fn withRowCumAllNull(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllNull(self, names, output_names);
            }

            pub fn withRowPrefixAllNull(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllNull(self, names, output_names);
            }

            pub fn withRowCumulativeAnyValid(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyValid(self, names, output_names);
            }

            pub fn withRowCumAnyValid(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyValid(self, names, output_names);
            }

            pub fn withRowPrefixAnyValid(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyValid(self, names, output_names);
            }

            pub fn withRowCumulativeAllValid(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllValid(self, names, output_names);
            }

            pub fn withRowCumAllValid(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllValid(self, names, output_names);
            }

            pub fn withRowPrefixAllValid(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllValid(self, names, output_names);
            }

            pub fn withRowCumulativeNullCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNullCount(self, names, output_names);
            }

            pub fn withRowCumNullCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNullCount(self, names, output_names);
            }

            pub fn withRowPrefixNullCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNullCount(self, names, output_names);
            }

            pub fn withRowCumulativeValidCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeValidCount(self, names, output_names);
            }

            pub fn withRowCumValidCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumValidCount(self, names, output_names);
            }

            pub fn withRowPrefixValidCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixValidCount(self, names, output_names);
            }

            pub fn withRowCumulativeNullRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNullRatio(self, names, output_names);
            }

            pub fn withRowCumNullRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNullRatio(self, names, output_names);
            }

            pub fn withRowPrefixNullRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNullRatio(self, names, output_names);
            }

            pub fn withRowCumulativeValidRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeValidRatio(self, names, output_names);
            }

            pub fn withRowCumValidRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumValidRatio(self, names, output_names);
            }

            pub fn withRowPrefixValidRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixValidRatio(self, names, output_names);
            }

            pub fn withRowNullRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNullRatio(self, names, output_name);
            }

            pub fn withRowValidRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowValidRatio(self, names, output_name);
            }

            pub fn withRowFirstValidIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstValidIndex(self, names, output_name);
            }

            pub fn withRowLastValidIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastValidIndex(self, names, output_name);
            }

            pub fn withRowFirstNullIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNullIndex(self, names, output_name);
            }

            pub fn withRowLastNullIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNullIndex(self, names, output_name);
            }

            pub fn withRowCumulativeFirstValidIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstValidIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstValidIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstValidIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastValidIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastValidIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastValidIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastValidIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstNullIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstNullIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstNullIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstNullIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastNullIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastNullIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastNullIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastNullIndex(self, names, output_names);
            }

            pub fn withRowPairCount(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPairCount(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowWeightedMean(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedMean(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedVariance(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedVariance(self, value_names, weight_names, output_name, correction);
            }

            pub fn withRowWeightedVar(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedVar(self, value_names, weight_names, output_name, correction);
            }

            pub fn withRowWeightedStddev(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedStddev(self, value_names, weight_names, output_name, correction);
            }

            pub fn withRowWeightedStd(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedStd(self, value_names, weight_names, output_name, correction);
            }

            pub fn withRowWeightedCovariance(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedCovariance(self, lhs_names, rhs_names, weight_names, output_name, correction);
            }

            pub fn withRowWeightedCorrelation(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedCorrelation(self, lhs_names, rhs_names, weight_names, output_name, correction);
            }

            pub fn withRowWeightedBeta(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedBeta(self, lhs_names, rhs_names, weight_names, output_name, correction);
            }

            pub fn withRowWeightedQuantile(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedQuantile(self, value_names, weight_names, output_name, q);
            }

            pub fn withRowWeightedMedian(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedMedian(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedIqr(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedIqr(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedMad(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedMad(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedTrimmedMean(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedTrimmedMean(self, value_names, weight_names, output_name, trim_fraction);
            }

            pub fn withRowWeightedWinsorizedMean(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedWinsorizedMean(self, value_names, weight_names, output_name, winsor_fraction);
            }

            pub fn withRowWeightedMode(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedMode(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedModeWeight(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedModeWeight(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedModeRatio(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedModeRatio(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedModeMargin(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedModeMargin(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedModeMarginRatio(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedModeMarginRatio(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedEntropy(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedEntropy(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedGiniImpurity(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedGiniImpurity(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedPerplexity(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedPerplexity(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedInverseSimpson(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedInverseSimpson(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedSimpsonConcentration(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedSimpsonConcentration(self, value_names, weight_names, output_name);
            }

            pub fn withRowWeightedEvenness(self: *DeviceLazyFrame, value_names: []const []const u8, weight_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowWeightedEvenness(self, value_names, weight_names, output_name);
            }

            pub fn withRowDot(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowDot(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowCosineSimilarity(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCosineSimilarity(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowCosine(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCosine(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowSquaredEuclideanDistance(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSquaredEuclideanDistance(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowEuclideanDistance(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowEuclideanDistance(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowManhattanDistance(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowManhattanDistance(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowChebyshevDistance(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowChebyshevDistance(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowCanberraDistance(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCanberraDistance(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowBrayCurtisDistance(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowBrayCurtisDistance(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowMeanError(self: *DeviceLazyFrame, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMeanError(self, actual_names, predicted_names, output_name);
            }

            pub fn withRowBias(self: *DeviceLazyFrame, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowBias(self, actual_names, predicted_names, output_name);
            }

            pub fn withRowMae(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMae(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowMse(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMse(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowRmse(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowRmse(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowMape(self: *DeviceLazyFrame, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMape(self, actual_names, predicted_names, output_name);
            }

            pub fn withRowSmape(self: *DeviceLazyFrame, actual_names: []const []const u8, predicted_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSmape(self, actual_names, predicted_names, output_name);
            }

            pub fn withRowCovariance(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCovariance(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowCorrelation(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCorrelation(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowBeta(self: *DeviceLazyFrame, lhs_names: []const []const u8, rhs_names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowBeta(self, lhs_names, rhs_names, output_name);
            }

            pub fn withRowArgMin(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowArgMin(self, names, output_name);
            }

            pub fn withRowArgMax(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowArgMax(self, names, output_name);
            }

            pub fn withRowCumulativeArgMin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeArgMin(self, names, output_names);
            }

            pub fn withRowCumArgMin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumArgMin(self, names, output_names);
            }

            pub fn withRowPrefixArgMin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixArgMin(self, names, output_names);
            }

            pub fn withRowCumulativeArgMax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeArgMax(self, names, output_names);
            }

            pub fn withRowCumArgMax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumArgMax(self, names, output_names);
            }

            pub fn withRowPrefixArgMax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixArgMax(self, names, output_names);
            }

            pub fn withRowQuantile(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, q: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowQuantile(self, names, output_name, q);
            }

            pub fn withRowQuantileRange(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, low_q: f64, high_q: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowQuantileRange(self, names, output_name, low_q, high_q);
            }

            pub fn withRowTrimmedMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, trim_fraction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowTrimmedMean(self, names, output_name, trim_fraction);
            }

            pub fn withRowWinsorizedMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, winsor_fraction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowWinsorizedMean(self, names, output_name, winsor_fraction);
            }

            pub fn withRowMedian(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMedian(self, names, output_name);
            }

            pub fn withRowIqr(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIqr(self, names, output_name);
            }

            pub fn withRowInterdecileRange(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowInterdecileRange(self, names, output_name);
            }

            pub fn withRowIdr(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIdr(self, names, output_name);
            }

            pub fn withRowMidhinge(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMidhinge(self, names, output_name);
            }

            pub fn withRowTrimean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowTrimean(self, names, output_name);
            }

            pub fn withRowBowleySkewness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowBowleySkewness(self, names, output_name);
            }

            pub fn withRowBowleySkew(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowBowleySkew(self, names, output_name);
            }

            pub fn withRowQuartileCoeffDispersion(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowQuartileCoeffDispersion(self, names, output_name);
            }

            pub fn withRowQcd(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowQcd(self, names, output_name);
            }

            pub fn withRowKelleySkewness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowKelleySkewness(self, names, output_name);
            }

            pub fn withRowKelleySkew(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowKelleySkew(self, names, output_name);
            }

            pub fn withRowMad(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMad(self, names, output_name);
            }

            pub fn withRowMedianAbsDev(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMedianAbsDev(self, names, output_name);
            }

            pub fn withRowMode(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMode(self, names, output_name);
            }

            pub fn withRowCumulativeMode(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMode(self, names, output_names);
            }

            pub fn withRowCumMode(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMode(self, names, output_names);
            }

            pub fn withRowPrefixMode(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMode(self, names, output_names);
            }

            pub fn withRowCumulativeModeCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeModeCount(self, names, output_names);
            }

            pub fn withRowCumModeCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumModeCount(self, names, output_names);
            }

            pub fn withRowPrefixModeCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixModeCount(self, names, output_names);
            }

            pub fn withRowCumulativeModeRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeModeRatio(self, names, output_names);
            }

            pub fn withRowCumModeRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumModeRatio(self, names, output_names);
            }

            pub fn withRowPrefixModeRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixModeRatio(self, names, output_names);
            }

            pub fn withRowCumulativeModeMargin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeModeMargin(self, names, output_names);
            }

            pub fn withRowCumModeMargin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumModeMargin(self, names, output_names);
            }

            pub fn withRowPrefixModeMargin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixModeMargin(self, names, output_names);
            }

            pub fn withRowCumulativeModeMarginRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeModeMarginRatio(self, names, output_names);
            }

            pub fn withRowCumModeMarginRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumModeMarginRatio(self, names, output_names);
            }

            pub fn withRowPrefixModeMarginRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixModeMarginRatio(self, names, output_names);
            }

            pub fn withRowEntropy(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowEntropy(self, names, output_name);
            }

            pub fn withRowGiniImpurity(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowGiniImpurity(self, names, output_name);
            }

            pub fn withRowPerplexity(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPerplexity(self, names, output_name);
            }

            pub fn withRowInverseSimpson(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowInverseSimpson(self, names, output_name);
            }

            pub fn withRowSimpsonConcentration(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSimpsonConcentration(self, names, output_name);
            }

            pub fn withRowEvenness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowEvenness(self, names, output_name);
            }

            pub fn withRowModeCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowModeCount(self, names, output_name);
            }

            pub fn withRowModeRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowModeRatio(self, names, output_name);
            }

            pub fn withRowModeMargin(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowModeMargin(self, names, output_name);
            }

            pub fn withRowModeMarginRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowModeMarginRatio(self, names, output_name);
            }

            pub fn withRowCountDistinct(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCountDistinct(self, names, output_name);
            }

            pub fn withRowNUnique(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNUnique(self, names, output_name);
            }

            pub fn withRowIsDuplicated(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIsDuplicated(self, names, output_name);
            }

            pub fn withRowIsUnique(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIsUnique(self, names, output_name);
            }

            pub fn withRowCumulativeDistinctCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeDistinctCount(self, names, output_names);
            }

            pub fn withRowCumDistinctCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumDistinctCount(self, names, output_names);
            }

            pub fn withRowPrefixDistinctCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixDistinctCount(self, names, output_names);
            }

            pub fn withRowCumulativeNUnique(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNUnique(self, names, output_names);
            }

            pub fn withRowPrefixNUnique(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNUnique(self, names, output_names);
            }

            pub fn withRowSum(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSum(self, names, output_name);
            }

            pub fn withRowMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMean(self, names, output_name);
            }

            pub fn withRowLogSumExp(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLogSumExp(self, names, output_name);
            }

            pub fn withRowLogsumexp(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLogsumexp(self, names, output_name);
            }

            pub fn withRowLogMeanExp(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLogMeanExp(self, names, output_name);
            }

            pub fn withRowLogmeanexp(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLogmeanexp(self, names, output_name);
            }

            pub fn withRowCentered(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCentered(self, names, output_names);
            }

            pub fn withRowDemean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowDemean(self, names, output_names);
            }

            pub fn withRowZScore(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowZScore(self, names, output_names);
            }

            pub fn withRowZscore(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowZscore(self, names, output_names);
            }

            pub fn withRowStandardize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowStandardize(self, names, output_names);
            }

            pub fn withRowRobustZScore(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowRobustZScore(self, names, output_names);
            }

            pub fn withRowRobustZscore(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowRobustZscore(self, names, output_names);
            }

            pub fn withRowMadZScore(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMadZScore(self, names, output_names);
            }

            pub fn withRowMadZscore(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMadZscore(self, names, output_names);
            }

            pub fn withRowAverageRank(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAverageRank(self, names, output_names);
            }

            pub fn withRowAverageRanks(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAverageRanks(self, names, output_names);
            }

            pub fn withRowAvgRank(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAvgRank(self, names, output_names);
            }

            pub fn withRowAvgRanks(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAvgRanks(self, names, output_names);
            }

            pub fn withRowFractionalRank(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFractionalRank(self, names, output_names);
            }

            pub fn withRowFractionalRanks(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFractionalRanks(self, names, output_names);
            }

            pub fn withRowOrdinalRank(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowOrdinalRank(self, names, output_names);
            }

            pub fn withRowOrdinalRanks(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowOrdinalRanks(self, names, output_names);
            }

            pub fn withRowDenseRank(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowDenseRank(self, names, output_names);
            }

            pub fn withRowDenseRanks(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowDenseRanks(self, names, output_names);
            }

            pub fn withRowCompetitionRank(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCompetitionRank(self, names, output_names);
            }

            pub fn withRowCompetitionRanks(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCompetitionRanks(self, names, output_names);
            }

            pub fn withRowMinRank(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMinRank(self, names, output_names);
            }

            pub fn withRowMinRanks(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMinRanks(self, names, output_names);
            }

            pub fn withRowPercentRank(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPercentRank(self, names, output_names);
            }

            pub fn withRowPercentRanks(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPercentRanks(self, names, output_names);
            }

            pub fn withRowPercentileRank(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPercentileRank(self, names, output_names);
            }

            pub fn withRowPercentileRanks(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPercentileRanks(self, names, output_names);
            }

            pub fn withRowCumeDist(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumeDist(self, names, output_names);
            }

            pub fn withRowCumeDistribution(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumeDistribution(self, names, output_names);
            }

            pub fn withRowCumulativeDistribution(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeDistribution(self, names, output_names);
            }

            pub fn withRowCumulativeSum(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeSum(self, names, output_names);
            }

            pub fn withRowCumsum(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumsum(self, names, output_names);
            }

            pub fn withRowCumSum(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumSum(self, names, output_names);
            }

            pub fn withRowPrefixSum(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixSum(self, names, output_names);
            }

            pub fn withRowCumulativeMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMean(self, names, output_names);
            }

            pub fn withRowCummean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCummean(self, names, output_names);
            }

            pub fn withRowCumMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMean(self, names, output_names);
            }

            pub fn withRowPrefixMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMean(self, names, output_names);
            }

            pub fn withRowCumulativeAverage(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAverage(self, names, output_names);
            }

            pub fn withRowCumAverage(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAverage(self, names, output_names);
            }

            pub fn withRowCumAvg(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAvg(self, names, output_names);
            }

            pub fn withRowPrefixAverage(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAverage(self, names, output_names);
            }

            pub fn withRowPrefixAvg(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAvg(self, names, output_names);
            }

            pub fn withRowCumulativeLogSumExp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLogSumExp(self, names, output_names);
            }

            pub fn withRowCumulativeLogsumexp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLogsumexp(self, names, output_names);
            }

            pub fn withRowCumLogSumExp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumLogSumExp(self, names, output_names);
            }

            pub fn withRowCumLogsumexp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumLogsumexp(self, names, output_names);
            }

            pub fn withRowPrefixLogSumExp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLogSumExp(self, names, output_names);
            }

            pub fn withRowPrefixLogsumexp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLogsumexp(self, names, output_names);
            }

            pub fn withRowCumulativeLogMeanExp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLogMeanExp(self, names, output_names);
            }

            pub fn withRowCumulativeLogmeanexp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLogmeanexp(self, names, output_names);
            }

            pub fn withRowCumLogMeanExp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumLogMeanExp(self, names, output_names);
            }

            pub fn withRowCumLogmeanexp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumLogmeanexp(self, names, output_names);
            }

            pub fn withRowPrefixLogMeanExp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLogMeanExp(self, names, output_names);
            }

            pub fn withRowPrefixLogmeanexp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLogmeanexp(self, names, output_names);
            }

            pub fn withRowCumulativeGeometricMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeGeometricMean(self, names, output_names);
            }

            pub fn withRowCumulativeGeoMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeGeoMean(self, names, output_names);
            }

            pub fn withRowCumGeometricMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumGeometricMean(self, names, output_names);
            }

            pub fn withRowCumGeoMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumGeoMean(self, names, output_names);
            }

            pub fn withRowPrefixGeometricMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixGeometricMean(self, names, output_names);
            }

            pub fn withRowPrefixGeoMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixGeoMean(self, names, output_names);
            }

            pub fn withRowCumulativeHarmonicMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeHarmonicMean(self, names, output_names);
            }

            pub fn withRowCumulativeHarmMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeHarmMean(self, names, output_names);
            }

            pub fn withRowCumHarmonicMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumHarmonicMean(self, names, output_names);
            }

            pub fn withRowCumHarmMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumHarmMean(self, names, output_names);
            }

            pub fn withRowPrefixHarmonicMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixHarmonicMean(self, names, output_names);
            }

            pub fn withRowPrefixHarmMean(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixHarmMean(self, names, output_names);
            }

            pub fn withRowCumulativeVariance(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeVariance(self, names, output_names, correction);
            }

            pub fn withRowCumulativeVar(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeVar(self, names, output_names, correction);
            }

            pub fn withRowCumVariance(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumVariance(self, names, output_names, correction);
            }

            pub fn withRowCumVar(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumVar(self, names, output_names, correction);
            }

            pub fn withRowPrefixVariance(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixVariance(self, names, output_names, correction);
            }

            pub fn withRowPrefixVar(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixVar(self, names, output_names, correction);
            }

            pub fn withRowCumulativeStddev(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeStddev(self, names, output_names, correction);
            }

            pub fn withRowCumulativeStd(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeStd(self, names, output_names, correction);
            }

            pub fn withRowCumStddev(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumStddev(self, names, output_names, correction);
            }

            pub fn withRowCumStd(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumStd(self, names, output_names, correction);
            }

            pub fn withRowPrefixStddev(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixStddev(self, names, output_names, correction);
            }

            pub fn withRowPrefixStd(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixStd(self, names, output_names, correction);
            }

            pub fn withRowCumulativeSem(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeSem(self, names, output_names, correction);
            }

            pub fn withRowCumSem(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumSem(self, names, output_names, correction);
            }

            pub fn withRowPrefixSem(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixSem(self, names, output_names, correction);
            }

            pub fn withRowCumulativeCv(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeCv(self, names, output_names, correction);
            }

            pub fn withRowCumCv(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumCv(self, names, output_names, correction);
            }

            pub fn withRowPrefixCv(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixCv(self, names, output_names, correction);
            }

            pub fn withRowCumulativeFano(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFano(self, names, output_names, correction);
            }

            pub fn withRowCumFano(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumFano(self, names, output_names, correction);
            }

            pub fn withRowPrefixFano(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFano(self, names, output_names, correction);
            }

            pub fn withRowCumulativeIndexOfDispersion(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeIndexOfDispersion(self, names, output_names, correction);
            }

            pub fn withRowCumIndexOfDispersion(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCumIndexOfDispersion(self, names, output_names, correction);
            }

            pub fn withRowPrefixIndexOfDispersion(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixIndexOfDispersion(self, names, output_names, correction);
            }

            pub fn withRowCumulativeSkewness(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeSkewness(self, names, output_names);
            }

            pub fn withRowCumulativeSkew(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeSkew(self, names, output_names);
            }

            pub fn withRowCumSkewness(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumSkewness(self, names, output_names);
            }

            pub fn withRowCumSkew(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumSkew(self, names, output_names);
            }

            pub fn withRowPrefixSkewness(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixSkewness(self, names, output_names);
            }

            pub fn withRowPrefixSkew(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixSkew(self, names, output_names);
            }

            pub fn withRowCumulativeKurtosis(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeKurtosis(self, names, output_names);
            }

            pub fn withRowCumulativeKurt(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeKurt(self, names, output_names);
            }

            pub fn withRowCumKurtosis(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumKurtosis(self, names, output_names);
            }

            pub fn withRowCumKurt(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumKurt(self, names, output_names);
            }

            pub fn withRowPrefixKurtosis(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixKurtosis(self, names, output_names);
            }

            pub fn withRowPrefixKurt(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixKurt(self, names, output_names);
            }

            pub fn withRowCumulativeRms(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeRms(self, names, output_names);
            }

            pub fn withRowCumRms(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumRms(self, names, output_names);
            }

            pub fn withRowPrefixRms(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixRms(self, names, output_names);
            }

            pub fn withRowCumulativeMeanAbs(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMeanAbs(self, names, output_names);
            }

            pub fn withRowCumulativeMeanAbsolute(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMeanAbsolute(self, names, output_names);
            }

            pub fn withRowCumMeanAbs(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMeanAbs(self, names, output_names);
            }

            pub fn withRowCumMeanAbsolute(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMeanAbsolute(self, names, output_names);
            }

            pub fn withRowPrefixMeanAbs(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMeanAbs(self, names, output_names);
            }

            pub fn withRowPrefixMeanAbsolute(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMeanAbsolute(self, names, output_names);
            }

            pub fn withRowCumulativeMeanSquare(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMeanSquare(self, names, output_names);
            }

            pub fn withRowCumulativeMeanSquared(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMeanSquared(self, names, output_names);
            }

            pub fn withRowCumMeanSquare(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMeanSquare(self, names, output_names);
            }

            pub fn withRowCumMeanSquared(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMeanSquared(self, names, output_names);
            }

            pub fn withRowPrefixMeanSquare(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMeanSquare(self, names, output_names);
            }

            pub fn withRowPrefixMeanSquared(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMeanSquared(self, names, output_names);
            }

            pub fn withRowCumulativeMaxAbs(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMaxAbs(self, names, output_names);
            }

            pub fn withRowCumulativeMaxAbsolute(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMaxAbsolute(self, names, output_names);
            }

            pub fn withRowCumulativeLInfNorm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLInfNorm(self, names, output_names);
            }

            pub fn withRowCumulativeLinfNorm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLinfNorm(self, names, output_names);
            }

            pub fn withRowCumMaxAbs(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMaxAbs(self, names, output_names);
            }

            pub fn withRowCumMaxAbsolute(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMaxAbsolute(self, names, output_names);
            }

            pub fn withRowCumLInfNorm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumLInfNorm(self, names, output_names);
            }

            pub fn withRowCumLinfNorm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumLinfNorm(self, names, output_names);
            }

            pub fn withRowPrefixMaxAbs(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMaxAbs(self, names, output_names);
            }

            pub fn withRowPrefixMaxAbsolute(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMaxAbsolute(self, names, output_names);
            }

            pub fn withRowPrefixLInfNorm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLInfNorm(self, names, output_names);
            }

            pub fn withRowPrefixLinfNorm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLinfNorm(self, names, output_names);
            }

            pub fn withRowCumulativeMinAbs(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMinAbs(self, names, output_names);
            }

            pub fn withRowCumulativeMinAbsolute(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMinAbsolute(self, names, output_names);
            }

            pub fn withRowCumMinAbs(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMinAbs(self, names, output_names);
            }

            pub fn withRowCumMinAbsolute(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMinAbsolute(self, names, output_names);
            }

            pub fn withRowPrefixMinAbs(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMinAbs(self, names, output_names);
            }

            pub fn withRowPrefixMinAbsolute(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMinAbsolute(self, names, output_names);
            }

            pub fn withRowCumulativeL1Norm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeL1Norm(self, names, output_names);
            }

            pub fn withRowCumL1Norm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumL1Norm(self, names, output_names);
            }

            pub fn withRowPrefixL1Norm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixL1Norm(self, names, output_names);
            }

            pub fn withRowCumulativeL2Norm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeL2Norm(self, names, output_names);
            }

            pub fn withRowCumL2Norm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumL2Norm(self, names, output_names);
            }

            pub fn withRowPrefixL2Norm(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixL2Norm(self, names, output_names);
            }

            pub fn withRowCumulativeProduct(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeProduct(self, names, output_names);
            }

            pub fn withRowCumprod(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumprod(self, names, output_names);
            }

            pub fn withRowCumProd(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumProd(self, names, output_names);
            }

            pub fn withRowPrefixProduct(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixProduct(self, names, output_names);
            }

            pub fn withRowCumulativeMax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMax(self, names, output_names);
            }

            pub fn withRowCummax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCummax(self, names, output_names);
            }

            pub fn withRowCumMax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMax(self, names, output_names);
            }

            pub fn withRowPrefixMax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMax(self, names, output_names);
            }

            pub fn withRowCumulativeMin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeMin(self, names, output_names);
            }

            pub fn withRowCummin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCummin(self, names, output_names);
            }

            pub fn withRowCumMin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumMin(self, names, output_names);
            }

            pub fn withRowPrefixMin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixMin(self, names, output_names);
            }

            pub fn withRowCumulativeRange(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeRange(self, names, output_names);
            }

            pub fn withRowCumRange(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumRange(self, names, output_names);
            }

            pub fn withRowPrefixRange(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixRange(self, names, output_names);
            }

            pub fn withRowCumulativePtp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativePtp(self, names, output_names);
            }

            pub fn withRowCumPtp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumPtp(self, names, output_names);
            }

            pub fn withRowPrefixPtp(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixPtp(self, names, output_names);
            }

            pub fn withRowIqrOutlier(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIqrOutlier(self, names, output_names);
            }

            pub fn withRowIqrOutliers(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIqrOutliers(self, names, output_names);
            }

            pub fn withRowTukeyOutlier(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowTukeyOutlier(self, names, output_names);
            }

            pub fn withRowTukeyOutliers(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowTukeyOutliers(self, names, output_names);
            }

            pub fn withRowMaxIndicator(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMaxIndicator(self, names, output_names);
            }

            pub fn withRowMaxIndicators(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMaxIndicators(self, names, output_names);
            }

            pub fn withRowIsMax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIsMax(self, names, output_names);
            }

            pub fn withRowMaxMask(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMaxMask(self, names, output_names);
            }

            pub fn withRowMinIndicator(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMinIndicator(self, names, output_names);
            }

            pub fn withRowMinIndicators(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMinIndicators(self, names, output_names);
            }

            pub fn withRowIsMin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIsMin(self, names, output_names);
            }

            pub fn withRowMinMask(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMinMask(self, names, output_names);
            }

            pub fn withRowTukeyWinsorize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowTukeyWinsorize(self, names, output_names);
            }

            pub fn withRowTukeyWinsorized(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowTukeyWinsorized(self, names, output_names);
            }

            pub fn withRowIqrWinsorize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIqrWinsorize(self, names, output_names);
            }

            pub fn withRowIqrWinsorized(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowIqrWinsorized(self, names, output_names);
            }

            pub fn withRowMinMaxScale(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMinMaxScale(self, names, output_names);
            }

            pub fn withRowMinmaxScale(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMinmaxScale(self, names, output_names);
            }

            pub fn withRowL2Normalize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowL2Normalize(self, names, output_names);
            }

            pub fn withRowL2Normalized(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowL2Normalized(self, names, output_names);
            }

            pub fn withRowL1Normalize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowL1Normalize(self, names, output_names);
            }

            pub fn withRowL1Normalized(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowL1Normalized(self, names, output_names);
            }

            pub fn withRowSumNormalize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSumNormalize(self, names, output_names);
            }

            pub fn withRowProportion(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowProportion(self, names, output_names);
            }

            pub fn withRowShare(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowShare(self, names, output_names);
            }

            pub fn withRowMeanNormalize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMeanNormalize(self, names, output_names);
            }

            pub fn withRowMeanNormalized(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMeanNormalized(self, names, output_names);
            }

            pub fn withRowMeanRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMeanRatio(self, names, output_names);
            }

            pub fn withRowMaxAbsNormalize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMaxAbsNormalize(self, names, output_names);
            }

            pub fn withRowMaxabsNormalize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMaxabsNormalize(self, names, output_names);
            }

            pub fn withRowLInfNormalize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLInfNormalize(self, names, output_names);
            }

            pub fn withRowLinfNormalize(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLinfNormalize(self, names, output_names);
            }

            pub fn withRowSoftmax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmax(self, names, output_names);
            }

            pub fn withRowLogSoftmax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLogSoftmax(self, names, output_names);
            }

            pub fn withRowLogsoftmax(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLogsoftmax(self, names, output_names);
            }

            pub fn withRowSoftmin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmin(self, names, output_names);
            }

            pub fn withRowLogSoftmin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLogSoftmin(self, names, output_names);
            }

            pub fn withRowLogsoftmin(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLogsoftmin(self, names, output_names);
            }

            pub fn withRowSoftmaxEntropy(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxEntropy(self, names, output_name);
            }

            pub fn withRowSoftmaxPerplexity(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxPerplexity(self, names, output_name);
            }

            pub fn withRowSoftmaxConfidence(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxConfidence(self, names, output_name);
            }

            pub fn withRowSoftmaxMargin(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxMargin(self, names, output_name);
            }

            pub fn withRowSoftmaxEvenness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxEvenness(self, names, output_name);
            }

            pub fn withRowSoftmaxNormalizedEntropy(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxNormalizedEntropy(self, names, output_name);
            }

            pub fn withRowSoftmaxConcentration(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxConcentration(self, names, output_name);
            }

            pub fn withRowSoftmaxNormalizedHhi(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxNormalizedHhi(self, names, output_name);
            }

            pub fn withRowSoftmaxNormalizedHHI(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxNormalizedHHI(self, names, output_name);
            }

            pub fn withRowSoftmaxNhhi(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxNhhi(self, names, output_name);
            }

            pub fn withRowSoftmaxGiniImpurity(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxGiniImpurity(self, names, output_name);
            }

            pub fn withRowSoftmaxGini(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxGini(self, names, output_name);
            }

            pub fn withRowSoftmaxInverseSimpson(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxInverseSimpson(self, names, output_name);
            }

            pub fn withRowSoftmaxSimpsonEvenness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxSimpsonEvenness(self, names, output_name);
            }

            pub fn withRowSoftmaxSimpsonEven(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSoftmaxSimpsonEven(self, names, output_name);
            }

            pub fn withRowLogitMargin(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLogitMargin(self, names, output_name);
            }

            pub fn withRowGeometricMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowGeometricMean(self, names, output_name);
            }

            pub fn withRowGeoMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowGeoMean(self, names, output_name);
            }

            pub fn withRowMagnitudeGeometricMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeGeometricMean(self, names, output_name);
            }

            pub fn withRowAbsGeometricMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsGeometricMean(self, names, output_name);
            }

            pub fn withRowMagnitudeGeoMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeGeoMean(self, names, output_name);
            }

            pub fn withRowAbsGeoMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsGeoMean(self, names, output_name);
            }

            pub fn withRowHarmonicMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowHarmonicMean(self, names, output_name);
            }

            pub fn withRowHarmMean(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowHarmMean(self, names, output_name);
            }

            pub fn withRowSkewness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSkewness(self, names, output_name);
            }

            pub fn withRowSkew(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSkew(self, names, output_name);
            }

            pub fn withRowMagnitudeSkewness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeSkewness(self, names, output_name);
            }

            pub fn withRowAbsSkewness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsSkewness(self, names, output_name);
            }

            pub fn withRowMagnitudeSkew(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeSkew(self, names, output_name);
            }

            pub fn withRowAbsSkew(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsSkew(self, names, output_name);
            }

            pub fn withRowKurtosis(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowKurtosis(self, names, output_name);
            }

            pub fn withRowKurt(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowKurt(self, names, output_name);
            }

            pub fn withRowMagnitudeKurtosis(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeKurtosis(self, names, output_name);
            }

            pub fn withRowAbsKurtosis(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsKurtosis(self, names, output_name);
            }

            pub fn withRowMagnitudeKurt(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeKurt(self, names, output_name);
            }

            pub fn withRowAbsKurt(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsKurt(self, names, output_name);
            }

            pub fn withRowProd(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowProd(self, names, output_name);
            }

            pub fn withRowMin(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMin(self, names, output_name);
            }

            pub fn withRowMax(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMax(self, names, output_name);
            }

            pub fn withRowPtp(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPtp(self, names, output_name);
            }

            pub fn withRowMagnitudePtp(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudePtp(self, names, output_name);
            }

            pub fn withRowAbsPtp(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsPtp(self, names, output_name);
            }

            pub fn withRowMagnitudePeakToPeak(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudePeakToPeak(self, names, output_name);
            }

            pub fn withRowAbsPeakToPeak(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsPeakToPeak(self, names, output_name);
            }

            pub fn withRowMidrange(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMidrange(self, names, output_name);
            }

            pub fn withRowMagnitudeMidrange(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeMidrange(self, names, output_name);
            }

            pub fn withRowAbsMidrange(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsMidrange(self, names, output_name);
            }

            pub fn withRowRangeCoeff(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowRangeCoeff(self, names, output_name);
            }

            pub fn withRowRangeCoefficient(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowRangeCoefficient(self, names, output_name);
            }

            pub fn withRowMagnitudeRangeCoeff(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeRangeCoeff(self, names, output_name);
            }

            pub fn withRowAbsRangeCoeff(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsRangeCoeff(self, names, output_name);
            }

            pub fn withRowMagnitudeRangeCoefficient(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeRangeCoefficient(self, names, output_name);
            }

            pub fn withRowAbsRangeCoefficient(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsRangeCoefficient(self, names, output_name);
            }

            pub fn withRowMeanAbs(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMeanAbs(self, names, output_name);
            }

            pub fn withRowHhi(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowHhi(self, names, output_name);
            }

            pub fn withRowHerfindahl(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowHerfindahl(self, names, output_name);
            }

            pub fn withRowHerfindahlHirschman(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowHerfindahlHirschman(self, names, output_name);
            }

            pub fn withRowMagnitudeNormalizedHhi(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeNormalizedHhi(self, names, output_name);
            }

            pub fn withRowAbsNormalizedHhi(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsNormalizedHhi(self, names, output_name);
            }

            pub fn withRowMagnitudeSparsity(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeSparsity(self, names, output_name);
            }

            pub fn withRowAbsSparsity(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsSparsity(self, names, output_name);
            }

            pub fn withRowMagnitudeInverseSimpson(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeInverseSimpson(self, names, output_name);
            }

            pub fn withRowAbsInverseSimpson(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsInverseSimpson(self, names, output_name);
            }

            pub fn withRowMagnitudeSimpsonEvenness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeSimpsonEvenness(self, names, output_name);
            }

            pub fn withRowAbsSimpsonEvenness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsSimpsonEvenness(self, names, output_name);
            }

            pub fn withRowMagnitudeDominance(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeDominance(self, names, output_name);
            }

            pub fn withRowAbsDominance(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsDominance(self, names, output_name);
            }

            pub fn withRowMagnitudeDominanceMargin(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeDominanceMargin(self, names, output_name);
            }

            pub fn withRowAbsDominanceMargin(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsDominanceMargin(self, names, output_name);
            }

            pub fn withRowMagnitudeEntropy(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeEntropy(self, names, output_name);
            }

            pub fn withRowAbsEntropy(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsEntropy(self, names, output_name);
            }

            pub fn withRowMagnitudePerplexity(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudePerplexity(self, names, output_name);
            }

            pub fn withRowAbsPerplexity(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsPerplexity(self, names, output_name);
            }

            pub fn withRowMagnitudeEvenness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeEvenness(self, names, output_name);
            }

            pub fn withRowAbsEvenness(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsEvenness(self, names, output_name);
            }

            pub fn withRowMeanAbsDev(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMeanAbsDev(self, names, output_name);
            }

            pub fn withRowGiniMeanDiff(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowGiniMeanDiff(self, names, output_name);
            }

            pub fn withRowGiniCoefficient(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowGiniCoefficient(self, names, output_name);
            }

            pub fn withRowGiniCoeff(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowGiniCoeff(self, names, output_name);
            }

            pub fn withRowMeanAbsDevRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowMeanAbsDevRatio(self, names, output_name);
            }

            pub fn withRowRms(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowRms(self, names, output_name);
            }

            pub fn withRowL1Norm(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowL1Norm(self, names, output_name);
            }

            pub fn withRowL2Norm(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowL2Norm(self, names, output_name);
            }

            pub fn withRowVariance(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowVariance(self, names, output_name, correction);
            }

            pub fn withRowVar(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowVar(self, names, output_name, correction);
            }

            pub fn withRowMagnitudeVariance(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeVariance(self, names, output_name, correction);
            }

            pub fn withRowAbsVariance(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsVariance(self, names, output_name, correction);
            }

            pub fn withRowMagnitudeVar(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeVar(self, names, output_name, correction);
            }

            pub fn withRowAbsVar(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsVar(self, names, output_name, correction);
            }

            pub fn withRowStddev(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowStddev(self, names, output_name, correction);
            }

            pub fn withRowStd(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowStd(self, names, output_name, correction);
            }

            pub fn withRowMagnitudeStddev(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeStddev(self, names, output_name, correction);
            }

            pub fn withRowAbsStddev(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsStddev(self, names, output_name, correction);
            }

            pub fn withRowMagnitudeStd(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeStd(self, names, output_name, correction);
            }

            pub fn withRowAbsStd(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsStd(self, names, output_name, correction);
            }

            pub fn withRowSem(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowSem(self, names, output_name, correction);
            }

            pub fn withRowMagnitudeSem(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeSem(self, names, output_name, correction);
            }

            pub fn withRowAbsSem(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsSem(self, names, output_name, correction);
            }

            pub fn withRowCv(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowCv(self, names, output_name, correction);
            }

            pub fn withRowMagnitudeCv(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeCv(self, names, output_name, correction);
            }

            pub fn withRowAbsCv(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsCv(self, names, output_name, correction);
            }

            pub fn withRowMagnitudeFano(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeFano(self, names, output_name, correction);
            }

            pub fn withRowAbsFano(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsFano(self, names, output_name, correction);
            }

            pub fn withRowMagnitudeIndexOfDispersion(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowMagnitudeIndexOfDispersion(self, names, output_name, correction);
            }

            pub fn withRowAbsIndexOfDispersion(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowAbsIndexOfDispersion(self, names, output_name, correction);
            }

            pub fn withRowFano(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowFano(self, names, output_name, correction);
            }

            pub fn withRowIndexOfDispersion(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8, correction: f64) DeviceDataError!void {
                return lazy_expr_mod.withRowIndexOfDispersion(self, names, output_name, correction);
            }

            pub fn withRowTrueCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowTrueCount(self, names, output_name);
            }

            pub fn withRowFalseCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFalseCount(self, names, output_name);
            }

            pub fn withRowCumulativeTrueCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeTrueCount(self, names, output_names);
            }

            pub fn withRowCumTrueCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumTrueCount(self, names, output_names);
            }

            pub fn withRowPrefixTrueCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixTrueCount(self, names, output_names);
            }

            pub fn withRowCumulativeFalseCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFalseCount(self, names, output_names);
            }

            pub fn withRowCumFalseCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumFalseCount(self, names, output_names);
            }

            pub fn withRowPrefixFalseCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFalseCount(self, names, output_names);
            }

            pub fn withRowCumulativeTrueRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeTrueRatio(self, names, output_names);
            }

            pub fn withRowCumTrueRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumTrueRatio(self, names, output_names);
            }

            pub fn withRowPrefixTrueRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixTrueRatio(self, names, output_names);
            }

            pub fn withRowCumulativeFalseRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFalseRatio(self, names, output_names);
            }

            pub fn withRowCumFalseRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumFalseRatio(self, names, output_names);
            }

            pub fn withRowPrefixFalseRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFalseRatio(self, names, output_names);
            }

            pub fn withRowAnyTrue(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyTrue(self, names, output_name);
            }

            pub fn withRowAllTrue(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllTrue(self, names, output_name);
            }

            pub fn withRowAnyFalse(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyFalse(self, names, output_name);
            }

            pub fn withRowAllFalse(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllFalse(self, names, output_name);
            }

            pub fn withRowCumulativeAnyTrue(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyTrue(self, names, output_names);
            }

            pub fn withRowCumAnyTrue(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyTrue(self, names, output_names);
            }

            pub fn withRowPrefixAnyTrue(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyTrue(self, names, output_names);
            }

            pub fn withRowCumulativeAllTrue(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllTrue(self, names, output_names);
            }

            pub fn withRowCumAllTrue(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllTrue(self, names, output_names);
            }

            pub fn withRowPrefixAllTrue(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllTrue(self, names, output_names);
            }

            pub fn withRowCumulativeAnyFalse(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyFalse(self, names, output_names);
            }

            pub fn withRowCumAnyFalse(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyFalse(self, names, output_names);
            }

            pub fn withRowPrefixAnyFalse(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyFalse(self, names, output_names);
            }

            pub fn withRowCumulativeAllFalse(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllFalse(self, names, output_names);
            }

            pub fn withRowCumAllFalse(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllFalse(self, names, output_names);
            }

            pub fn withRowPrefixAllFalse(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllFalse(self, names, output_names);
            }

            pub fn withRowFirstTrueIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstTrueIndex(self, names, output_name);
            }

            pub fn withRowLastTrueIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastTrueIndex(self, names, output_name);
            }

            pub fn withRowFirstFalseIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstFalseIndex(self, names, output_name);
            }

            pub fn withRowLastFalseIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastFalseIndex(self, names, output_name);
            }

            pub fn withRowCumulativeFirstTrueIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstTrueIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstTrueIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstTrueIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastTrueIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastTrueIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastTrueIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastTrueIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstFalseIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstFalseIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstFalseIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstFalseIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastFalseIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastFalseIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastFalseIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastFalseIndex(self, names, output_names);
            }

            pub fn withRowTrueRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowTrueRatio(self, names, output_name);
            }

            pub fn withRowFalseRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFalseRatio(self, names, output_name);
            }

            pub fn withRowNaNCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNaNCount(self, names, output_name);
            }

            pub fn withRowNaNRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNaNRatio(self, names, output_name);
            }

            pub fn withRowNanRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNanRatio(self, names, output_name);
            }

            pub fn withRowInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowInfCount(self, names, output_name);
            }

            pub fn withRowInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowInfRatio(self, names, output_name);
            }

            pub fn withRowPositiveInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPositiveInfCount(self, names, output_name);
            }

            pub fn withRowNegativeInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNegativeInfCount(self, names, output_name);
            }

            pub fn withRowPositiveInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPositiveInfRatio(self, names, output_name);
            }

            pub fn withRowNegativeInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNegativeInfRatio(self, names, output_name);
            }

            pub fn withRowZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowZeroCount(self, names, output_name);
            }

            pub fn withRowZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowZeroRatio(self, names, output_name);
            }

            pub fn withRowPositiveZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPositiveZeroCount(self, names, output_name);
            }

            pub fn withRowNegativeZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNegativeZeroCount(self, names, output_name);
            }

            pub fn withRowPositiveZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPositiveZeroRatio(self, names, output_name);
            }

            pub fn withRowNegativeZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNegativeZeroRatio(self, names, output_name);
            }

            pub fn withRowNonZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNonZeroCount(self, names, output_name);
            }

            pub fn withRowNonZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNonZeroRatio(self, names, output_name);
            }

            pub fn withRowAnyZero(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyZero(self, names, output_name);
            }

            pub fn withRowAllZero(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllZero(self, names, output_name);
            }

            pub fn withRowCumulativeAnyZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyZero(self, names, output_names);
            }

            pub fn withRowCumAnyZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyZero(self, names, output_names);
            }

            pub fn withRowPrefixAnyZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyZero(self, names, output_names);
            }

            pub fn withRowCumulativeAllZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllZero(self, names, output_names);
            }

            pub fn withRowCumAllZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllZero(self, names, output_names);
            }

            pub fn withRowPrefixAllZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllZero(self, names, output_names);
            }

            pub fn withRowAnyNonZero(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyNonZero(self, names, output_name);
            }

            pub fn withRowAllNonZero(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllNonZero(self, names, output_name);
            }

            pub fn withRowCumulativeAnyNonZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyNonZero(self, names, output_names);
            }

            pub fn withRowCumAnyNonZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyNonZero(self, names, output_names);
            }

            pub fn withRowPrefixAnyNonZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyNonZero(self, names, output_names);
            }

            pub fn withRowCumulativeAllNonZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllNonZero(self, names, output_names);
            }

            pub fn withRowCumAllNonZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllNonZero(self, names, output_names);
            }

            pub fn withRowPrefixAllNonZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllNonZero(self, names, output_names);
            }

            pub fn withRowAnyPositiveZero(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyPositiveZero(self, names, output_name);
            }

            pub fn withRowAllPositiveZero(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllPositiveZero(self, names, output_name);
            }

            pub fn withRowCumulativeAnyPositiveZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyPositiveZero(self, names, output_names);
            }

            pub fn withRowCumAnyPositiveZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyPositiveZero(self, names, output_names);
            }

            pub fn withRowPrefixAnyPositiveZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyPositiveZero(self, names, output_names);
            }

            pub fn withRowCumulativeAllPositiveZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllPositiveZero(self, names, output_names);
            }

            pub fn withRowCumAllPositiveZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllPositiveZero(self, names, output_names);
            }

            pub fn withRowPrefixAllPositiveZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllPositiveZero(self, names, output_names);
            }

            pub fn withRowAnyNegativeZero(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyNegativeZero(self, names, output_name);
            }

            pub fn withRowAllNegativeZero(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllNegativeZero(self, names, output_name);
            }

            pub fn withRowCumulativeAnyNegativeZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyNegativeZero(self, names, output_names);
            }

            pub fn withRowCumAnyNegativeZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyNegativeZero(self, names, output_names);
            }

            pub fn withRowPrefixAnyNegativeZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyNegativeZero(self, names, output_names);
            }

            pub fn withRowCumulativeAllNegativeZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllNegativeZero(self, names, output_names);
            }

            pub fn withRowCumAllNegativeZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllNegativeZero(self, names, output_names);
            }

            pub fn withRowPrefixAllNegativeZero(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllNegativeZero(self, names, output_names);
            }

            pub fn withRowAnyPositive(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyPositive(self, names, output_name);
            }

            pub fn withRowAllPositive(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllPositive(self, names, output_name);
            }

            pub fn withRowCumulativeAnyPositive(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyPositive(self, names, output_names);
            }

            pub fn withRowCumAnyPositive(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyPositive(self, names, output_names);
            }

            pub fn withRowPrefixAnyPositive(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyPositive(self, names, output_names);
            }

            pub fn withRowCumulativeAllPositive(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllPositive(self, names, output_names);
            }

            pub fn withRowCumAllPositive(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllPositive(self, names, output_names);
            }

            pub fn withRowPrefixAllPositive(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllPositive(self, names, output_names);
            }

            pub fn withRowAnySignBit(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnySignBit(self, names, output_name);
            }

            pub fn withRowAllSignBit(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllSignBit(self, names, output_name);
            }

            pub fn withRowCumulativeAnySignBit(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnySignBit(self, names, output_names);
            }

            pub fn withRowCumAnySignBit(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnySignBit(self, names, output_names);
            }

            pub fn withRowPrefixAnySignBit(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnySignBit(self, names, output_names);
            }

            pub fn withRowCumulativeAllSignBit(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllSignBit(self, names, output_names);
            }

            pub fn withRowCumAllSignBit(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllSignBit(self, names, output_names);
            }

            pub fn withRowPrefixAllSignBit(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllSignBit(self, names, output_names);
            }

            pub fn withRowAnyNegative(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyNegative(self, names, output_name);
            }

            pub fn withRowAllNegative(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllNegative(self, names, output_name);
            }

            pub fn withRowCumulativeAnyNegative(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyNegative(self, names, output_names);
            }

            pub fn withRowCumAnyNegative(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyNegative(self, names, output_names);
            }

            pub fn withRowPrefixAnyNegative(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyNegative(self, names, output_names);
            }

            pub fn withRowCumulativeAllNegative(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllNegative(self, names, output_names);
            }

            pub fn withRowCumAllNegative(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllNegative(self, names, output_names);
            }

            pub fn withRowPrefixAllNegative(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllNegative(self, names, output_names);
            }

            pub fn withRowAnyNaN(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyNaN(self, names, output_name);
            }

            pub fn withRowAllNaN(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllNaN(self, names, output_name);
            }

            pub fn withRowCumulativeAnyNaN(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyNaN(self, names, output_names);
            }

            pub fn withRowCumAnyNaN(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyNaN(self, names, output_names);
            }

            pub fn withRowPrefixAnyNaN(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyNaN(self, names, output_names);
            }

            pub fn withRowCumulativeAllNaN(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllNaN(self, names, output_names);
            }

            pub fn withRowCumAllNaN(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllNaN(self, names, output_names);
            }

            pub fn withRowPrefixAllNaN(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllNaN(self, names, output_names);
            }

            pub fn withRowAnyInf(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyInf(self, names, output_name);
            }

            pub fn withRowAllInf(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllInf(self, names, output_name);
            }

            pub fn withRowCumulativeAnyInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyInf(self, names, output_names);
            }

            pub fn withRowCumAnyInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyInf(self, names, output_names);
            }

            pub fn withRowPrefixAnyInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyInf(self, names, output_names);
            }

            pub fn withRowCumulativeAllInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllInf(self, names, output_names);
            }

            pub fn withRowCumAllInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllInf(self, names, output_names);
            }

            pub fn withRowPrefixAllInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllInf(self, names, output_names);
            }

            pub fn withRowAnyPositiveInf(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyPositiveInf(self, names, output_name);
            }

            pub fn withRowAllPositiveInf(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllPositiveInf(self, names, output_name);
            }

            pub fn withRowCumulativeAnyPositiveInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyPositiveInf(self, names, output_names);
            }

            pub fn withRowCumAnyPositiveInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyPositiveInf(self, names, output_names);
            }

            pub fn withRowPrefixAnyPositiveInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyPositiveInf(self, names, output_names);
            }

            pub fn withRowCumulativeAllPositiveInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllPositiveInf(self, names, output_names);
            }

            pub fn withRowCumAllPositiveInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllPositiveInf(self, names, output_names);
            }

            pub fn withRowPrefixAllPositiveInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllPositiveInf(self, names, output_names);
            }

            pub fn withRowAnyNegativeInf(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyNegativeInf(self, names, output_name);
            }

            pub fn withRowAllNegativeInf(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllNegativeInf(self, names, output_name);
            }

            pub fn withRowCumulativeAnyNegativeInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyNegativeInf(self, names, output_names);
            }

            pub fn withRowCumAnyNegativeInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyNegativeInf(self, names, output_names);
            }

            pub fn withRowPrefixAnyNegativeInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyNegativeInf(self, names, output_names);
            }

            pub fn withRowCumulativeAllNegativeInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllNegativeInf(self, names, output_names);
            }

            pub fn withRowCumAllNegativeInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllNegativeInf(self, names, output_names);
            }

            pub fn withRowPrefixAllNegativeInf(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllNegativeInf(self, names, output_names);
            }

            pub fn withRowAnyFinite(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyFinite(self, names, output_name);
            }

            pub fn withRowAllFinite(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllFinite(self, names, output_name);
            }

            pub fn withRowCumulativeAnyFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyFinite(self, names, output_names);
            }

            pub fn withRowCumAnyFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyFinite(self, names, output_names);
            }

            pub fn withRowPrefixAnyFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyFinite(self, names, output_names);
            }

            pub fn withRowCumulativeAllFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllFinite(self, names, output_names);
            }

            pub fn withRowCumAllFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllFinite(self, names, output_names);
            }

            pub fn withRowPrefixAllFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllFinite(self, names, output_names);
            }

            pub fn withRowAnyNormal(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyNormal(self, names, output_name);
            }

            pub fn withRowAllNormal(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllNormal(self, names, output_name);
            }

            pub fn withRowCumulativeAnyNormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyNormal(self, names, output_names);
            }

            pub fn withRowCumAnyNormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyNormal(self, names, output_names);
            }

            pub fn withRowPrefixAnyNormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyNormal(self, names, output_names);
            }

            pub fn withRowCumulativeAllNormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllNormal(self, names, output_names);
            }

            pub fn withRowCumAllNormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllNormal(self, names, output_names);
            }

            pub fn withRowPrefixAllNormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllNormal(self, names, output_names);
            }

            pub fn withRowAnySubnormal(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnySubnormal(self, names, output_name);
            }

            pub fn withRowAllSubnormal(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllSubnormal(self, names, output_name);
            }

            pub fn withRowCumulativeAnySubnormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnySubnormal(self, names, output_names);
            }

            pub fn withRowCumAnySubnormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnySubnormal(self, names, output_names);
            }

            pub fn withRowPrefixAnySubnormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnySubnormal(self, names, output_names);
            }

            pub fn withRowCumulativeAllSubnormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllSubnormal(self, names, output_names);
            }

            pub fn withRowCumAllSubnormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllSubnormal(self, names, output_names);
            }

            pub fn withRowPrefixAllSubnormal(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllSubnormal(self, names, output_names);
            }

            pub fn withRowAnyNonFinite(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAnyNonFinite(self, names, output_name);
            }

            pub fn withRowAllNonFinite(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowAllNonFinite(self, names, output_name);
            }

            pub fn withRowCumulativeAnyNonFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAnyNonFinite(self, names, output_names);
            }

            pub fn withRowCumAnyNonFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAnyNonFinite(self, names, output_names);
            }

            pub fn withRowPrefixAnyNonFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAnyNonFinite(self, names, output_names);
            }

            pub fn withRowCumulativeAllNonFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeAllNonFinite(self, names, output_names);
            }

            pub fn withRowCumAllNonFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumAllNonFinite(self, names, output_names);
            }

            pub fn withRowPrefixAllNonFinite(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixAllNonFinite(self, names, output_names);
            }

            pub fn withRowFirstNaNIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNaNIndex(self, names, output_name);
            }

            pub fn withRowFirstNanIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNanIndex(self, names, output_name);
            }

            pub fn withRowLastNaNIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNaNIndex(self, names, output_name);
            }

            pub fn withRowLastNanIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNanIndex(self, names, output_name);
            }

            pub fn withRowFirstInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstInfIndex(self, names, output_name);
            }

            pub fn withRowLastInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastInfIndex(self, names, output_name);
            }

            pub fn withRowFirstPositiveInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstPositiveInfIndex(self, names, output_name);
            }

            pub fn withRowLastPositiveInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastPositiveInfIndex(self, names, output_name);
            }

            pub fn withRowFirstNegativeInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNegativeInfIndex(self, names, output_name);
            }

            pub fn withRowLastNegativeInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNegativeInfIndex(self, names, output_name);
            }

            pub fn withRowFirstPositiveZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstPositiveZeroIndex(self, names, output_name);
            }

            pub fn withRowLastPositiveZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastPositiveZeroIndex(self, names, output_name);
            }

            pub fn withRowFirstNegativeZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNegativeZeroIndex(self, names, output_name);
            }

            pub fn withRowLastNegativeZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNegativeZeroIndex(self, names, output_name);
            }

            pub fn withRowFirstSignBitIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstSignBitIndex(self, names, output_name);
            }

            pub fn withRowLastSignBitIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastSignBitIndex(self, names, output_name);
            }

            pub fn withRowFirstFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstFiniteIndex(self, names, output_name);
            }

            pub fn withRowLastFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastFiniteIndex(self, names, output_name);
            }

            pub fn withRowFirstNormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNormalIndex(self, names, output_name);
            }

            pub fn withRowLastNormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNormalIndex(self, names, output_name);
            }

            pub fn withRowFirstSubnormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstSubnormalIndex(self, names, output_name);
            }

            pub fn withRowLastSubnormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastSubnormalIndex(self, names, output_name);
            }

            pub fn withRowFirstNonFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNonFiniteIndex(self, names, output_name);
            }

            pub fn withRowFirstNonfiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNonfiniteIndex(self, names, output_name);
            }

            pub fn withRowLastNonFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNonFiniteIndex(self, names, output_name);
            }

            pub fn withRowLastNonfiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNonfiniteIndex(self, names, output_name);
            }

            pub fn withRowFirstZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstZeroIndex(self, names, output_name);
            }

            pub fn withRowLastZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastZeroIndex(self, names, output_name);
            }

            pub fn withRowFirstNonZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNonZeroIndex(self, names, output_name);
            }

            pub fn withRowFirstNonzeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNonzeroIndex(self, names, output_name);
            }

            pub fn withRowLastNonZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNonZeroIndex(self, names, output_name);
            }

            pub fn withRowLastNonzeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNonzeroIndex(self, names, output_name);
            }

            pub fn withRowFirstPositiveIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstPositiveIndex(self, names, output_name);
            }

            pub fn withRowLastPositiveIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastPositiveIndex(self, names, output_name);
            }

            pub fn withRowFirstNegativeIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFirstNegativeIndex(self, names, output_name);
            }

            pub fn withRowLastNegativeIndex(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowLastNegativeIndex(self, names, output_name);
            }

            pub fn withRowPositiveCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPositiveCount(self, names, output_name);
            }

            pub fn withRowPositiveRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPositiveRatio(self, names, output_name);
            }

            pub fn withRowSignBitCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSignBitCount(self, names, output_name);
            }

            pub fn withRowSignBitRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSignBitRatio(self, names, output_name);
            }

            pub fn withRowNegativeCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNegativeCount(self, names, output_name);
            }

            pub fn withRowNegativeRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNegativeRatio(self, names, output_name);
            }

            pub fn withRowCumulativePositiveZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativePositiveZeroCount(self, names, output_names);
            }

            pub fn withRowCumPositiveZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumPositiveZeroCount(self, names, output_names);
            }

            pub fn withRowPrefixPositiveZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixPositiveZeroCount(self, names, output_names);
            }

            pub fn withRowCumulativePositiveZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativePositiveZeroRatio(self, names, output_names);
            }

            pub fn withRowCumPositiveZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumPositiveZeroRatio(self, names, output_names);
            }

            pub fn withRowPrefixPositiveZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixPositiveZeroRatio(self, names, output_names);
            }

            pub fn withRowCumulativeNegativeZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNegativeZeroCount(self, names, output_names);
            }

            pub fn withRowCumNegativeZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNegativeZeroCount(self, names, output_names);
            }

            pub fn withRowPrefixNegativeZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNegativeZeroCount(self, names, output_names);
            }

            pub fn withRowCumulativeNegativeZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNegativeZeroRatio(self, names, output_names);
            }

            pub fn withRowCumNegativeZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNegativeZeroRatio(self, names, output_names);
            }

            pub fn withRowPrefixNegativeZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNegativeZeroRatio(self, names, output_names);
            }

            pub fn withRowCumulativeSignBitCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeSignBitCount(self, names, output_names);
            }

            pub fn withRowCumSignBitCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumSignBitCount(self, names, output_names);
            }

            pub fn withRowPrefixSignBitCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixSignBitCount(self, names, output_names);
            }

            pub fn withRowCumulativeSignBitRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeSignBitRatio(self, names, output_names);
            }

            pub fn withRowCumSignBitRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumSignBitRatio(self, names, output_names);
            }

            pub fn withRowPrefixSignBitRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixSignBitRatio(self, names, output_names);
            }

            pub fn withRowCumulativeNaNCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNaNCount(self, names, output_names);
            }

            pub fn withRowCumNaNCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNaNCount(self, names, output_names);
            }

            pub fn withRowPrefixNaNCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNaNCount(self, names, output_names);
            }

            pub fn withRowCumulativeNaNRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNaNRatio(self, names, output_names);
            }

            pub fn withRowCumNaNRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNaNRatio(self, names, output_names);
            }

            pub fn withRowPrefixNaNRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNaNRatio(self, names, output_names);
            }

            pub fn withRowCumulativeInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeInfCount(self, names, output_names);
            }

            pub fn withRowCumInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumInfCount(self, names, output_names);
            }

            pub fn withRowPrefixInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixInfCount(self, names, output_names);
            }

            pub fn withRowCumulativeInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeInfRatio(self, names, output_names);
            }

            pub fn withRowCumInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumInfRatio(self, names, output_names);
            }

            pub fn withRowPrefixInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixInfRatio(self, names, output_names);
            }

            pub fn withRowCumulativePositiveInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativePositiveInfCount(self, names, output_names);
            }

            pub fn withRowCumPositiveInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumPositiveInfCount(self, names, output_names);
            }

            pub fn withRowPrefixPositiveInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixPositiveInfCount(self, names, output_names);
            }

            pub fn withRowCumulativePositiveInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativePositiveInfRatio(self, names, output_names);
            }

            pub fn withRowCumPositiveInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumPositiveInfRatio(self, names, output_names);
            }

            pub fn withRowPrefixPositiveInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixPositiveInfRatio(self, names, output_names);
            }

            pub fn withRowCumulativeNegativeInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNegativeInfCount(self, names, output_names);
            }

            pub fn withRowCumNegativeInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNegativeInfCount(self, names, output_names);
            }

            pub fn withRowPrefixNegativeInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNegativeInfCount(self, names, output_names);
            }

            pub fn withRowCumulativeNegativeInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNegativeInfRatio(self, names, output_names);
            }

            pub fn withRowCumNegativeInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNegativeInfRatio(self, names, output_names);
            }

            pub fn withRowPrefixNegativeInfRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNegativeInfRatio(self, names, output_names);
            }

            pub fn withRowCumulativeFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFiniteCount(self, names, output_names);
            }

            pub fn withRowCumFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumFiniteCount(self, names, output_names);
            }

            pub fn withRowPrefixFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFiniteCount(self, names, output_names);
            }

            pub fn withRowCumulativeFiniteRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFiniteRatio(self, names, output_names);
            }

            pub fn withRowCumFiniteRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumFiniteRatio(self, names, output_names);
            }

            pub fn withRowPrefixFiniteRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFiniteRatio(self, names, output_names);
            }

            pub fn withRowCumulativeNormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNormalCount(self, names, output_names);
            }

            pub fn withRowCumNormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNormalCount(self, names, output_names);
            }

            pub fn withRowPrefixNormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNormalCount(self, names, output_names);
            }

            pub fn withRowCumulativeNormalRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNormalRatio(self, names, output_names);
            }

            pub fn withRowCumNormalRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNormalRatio(self, names, output_names);
            }

            pub fn withRowPrefixNormalRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNormalRatio(self, names, output_names);
            }

            pub fn withRowCumulativeSubnormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeSubnormalCount(self, names, output_names);
            }

            pub fn withRowCumSubnormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumSubnormalCount(self, names, output_names);
            }

            pub fn withRowPrefixSubnormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixSubnormalCount(self, names, output_names);
            }

            pub fn withRowCumulativeSubnormalRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeSubnormalRatio(self, names, output_names);
            }

            pub fn withRowCumSubnormalRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumSubnormalRatio(self, names, output_names);
            }

            pub fn withRowPrefixSubnormalRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixSubnormalRatio(self, names, output_names);
            }

            pub fn withRowCumulativeNonFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNonFiniteCount(self, names, output_names);
            }

            pub fn withRowCumNonFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNonFiniteCount(self, names, output_names);
            }

            pub fn withRowPrefixNonFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNonFiniteCount(self, names, output_names);
            }

            pub fn withRowCumulativeNonFiniteRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNonFiniteRatio(self, names, output_names);
            }

            pub fn withRowCumNonFiniteRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNonFiniteRatio(self, names, output_names);
            }

            pub fn withRowPrefixNonFiniteRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNonFiniteRatio(self, names, output_names);
            }

            pub fn withRowCumulativeZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeZeroCount(self, names, output_names);
            }

            pub fn withRowCumZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumZeroCount(self, names, output_names);
            }

            pub fn withRowPrefixZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixZeroCount(self, names, output_names);
            }

            pub fn withRowCumulativeFirstNaNIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstNaNIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstNaNIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstNaNIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastNaNIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastNaNIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastNaNIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastNaNIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstInfIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstInfIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastInfIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastInfIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstPositiveInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstPositiveInfIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstPositiveInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstPositiveInfIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastPositiveInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastPositiveInfIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastPositiveInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastPositiveInfIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstNegativeInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstNegativeInfIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstNegativeInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstNegativeInfIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastNegativeInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastNegativeInfIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastNegativeInfIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastNegativeInfIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstFiniteIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstFiniteIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastFiniteIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastFiniteIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstNormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstNormalIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstNormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstNormalIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastNormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastNormalIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastNormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastNormalIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstSubnormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstSubnormalIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstSubnormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstSubnormalIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastSubnormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastSubnormalIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastSubnormalIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastSubnormalIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstNonFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstNonFiniteIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstNonFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstNonFiniteIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastNonFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastNonFiniteIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastNonFiniteIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastNonFiniteIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstZeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstZeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastZeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastZeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstPositiveZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstPositiveZeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstPositiveZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstPositiveZeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastPositiveZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastPositiveZeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastPositiveZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastPositiveZeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstNegativeZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstNegativeZeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstNegativeZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstNegativeZeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastNegativeZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastNegativeZeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastNegativeZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastNegativeZeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeNonZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNonZeroCount(self, names, output_names);
            }

            pub fn withRowCumNonZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNonZeroCount(self, names, output_names);
            }

            pub fn withRowPrefixNonZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNonZeroCount(self, names, output_names);
            }

            pub fn withRowCumulativeFirstNonZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstNonZeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstNonzeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstNonzeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstNonZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstNonZeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstNonzeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstNonzeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastNonZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastNonZeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastNonzeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastNonzeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastNonZeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastNonZeroIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastNonzeroIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastNonzeroIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstPositiveIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstPositiveIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstPositiveIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstPositiveIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastPositiveIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastPositiveIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastPositiveIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastPositiveIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstSignBitIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstSignBitIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstSignBitIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstSignBitIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastSignBitIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastSignBitIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastSignBitIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastSignBitIndex(self, names, output_names);
            }

            pub fn withRowCumulativeFirstNegativeIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeFirstNegativeIndex(self, names, output_names);
            }

            pub fn withRowPrefixFirstNegativeIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixFirstNegativeIndex(self, names, output_names);
            }

            pub fn withRowCumulativeLastNegativeIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeLastNegativeIndex(self, names, output_names);
            }

            pub fn withRowPrefixLastNegativeIndex(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixLastNegativeIndex(self, names, output_names);
            }

            pub fn withRowCumulativePositiveCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativePositiveCount(self, names, output_names);
            }

            pub fn withRowCumPositiveCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumPositiveCount(self, names, output_names);
            }

            pub fn withRowPrefixPositiveCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixPositiveCount(self, names, output_names);
            }

            pub fn withRowCumulativeNegativeCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNegativeCount(self, names, output_names);
            }

            pub fn withRowCumNegativeCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNegativeCount(self, names, output_names);
            }

            pub fn withRowPrefixNegativeCount(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNegativeCount(self, names, output_names);
            }

            pub fn withRowCumulativeZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeZeroRatio(self, names, output_names);
            }

            pub fn withRowCumZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumZeroRatio(self, names, output_names);
            }

            pub fn withRowPrefixZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixZeroRatio(self, names, output_names);
            }

            pub fn withRowCumulativeNonZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNonZeroRatio(self, names, output_names);
            }

            pub fn withRowCumNonZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNonZeroRatio(self, names, output_names);
            }

            pub fn withRowPrefixNonZeroRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNonZeroRatio(self, names, output_names);
            }

            pub fn withRowCumulativePositiveRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativePositiveRatio(self, names, output_names);
            }

            pub fn withRowCumPositiveRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumPositiveRatio(self, names, output_names);
            }

            pub fn withRowPrefixPositiveRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixPositiveRatio(self, names, output_names);
            }

            pub fn withRowCumulativeNegativeRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumulativeNegativeRatio(self, names, output_names);
            }

            pub fn withRowCumNegativeRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowCumNegativeRatio(self, names, output_names);
            }

            pub fn withRowPrefixNegativeRatio(self: *DeviceLazyFrame, names: []const []const u8, output_names: []const []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPrefixNegativeRatio(self, names, output_names);
            }

            pub fn withRowFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFiniteCount(self, names, output_name);
            }

            pub fn withRowFiniteRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFiniteRatio(self, names, output_name);
            }

            pub fn withRowNormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNormalCount(self, names, output_name);
            }

            pub fn withRowNormalRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNormalRatio(self, names, output_name);
            }

            pub fn withRowSubnormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSubnormalCount(self, names, output_name);
            }

            pub fn withRowSubnormalRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSubnormalRatio(self, names, output_name);
            }

            pub fn withRowNonFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNonFiniteCount(self, names, output_name);
            }

            pub fn withRowNonFiniteRatio(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNonFiniteRatio(self, names, output_name);
            }

            pub fn withColumnCompare(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnCompare(self, name, lhs_name, rhs_name, op);
            }

            pub fn withColumnCompareScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnCompareScalar(self, name, input_name, T, scalar, op);
            }

            pub const groupByCount = lazy_relation_methods_mod.groupByCount;
            pub const groupByCountOn = lazy_relation_methods_mod.groupByCountOn;
            pub const groupByHeadRows = lazy_relation_methods_mod.groupByHeadRows;
            pub const groupByHeadRowsOn = lazy_relation_methods_mod.groupByHeadRowsOn;
            pub const groupByTailRows = lazy_relation_methods_mod.groupByTailRows;
            pub const groupByTailRowsOn = lazy_relation_methods_mod.groupByTailRowsOn;
            pub const groupBySliceRows = lazy_relation_methods_mod.groupBySliceRows;
            pub const groupBySliceRowsOn = lazy_relation_methods_mod.groupBySliceRowsOn;
            pub const groupBySliceRowsStep = lazy_relation_methods_mod.groupBySliceRowsStep;
            pub const groupBySliceRowsStepOn = lazy_relation_methods_mod.groupBySliceRowsStepOn;
            pub const groupBySliceRowsSigned = lazy_relation_methods_mod.groupBySliceRowsSigned;
            pub const groupBySliceRowsSignedOn = lazy_relation_methods_mod.groupBySliceRowsSignedOn;
            pub const groupBySliceRowsSignedStep = lazy_relation_methods_mod.groupBySliceRowsSignedStep;
            pub const groupBySliceRowsSignedStepOn = lazy_relation_methods_mod.groupBySliceRowsSignedStepOn;
            pub const groupByTopRows = lazy_relation_methods_mod.groupByTopRows;
            pub const groupByTopRowsOn = lazy_relation_methods_mod.groupByTopRowsOn;
            pub const groupByBottomRows = lazy_relation_methods_mod.groupByBottomRows;
            pub const groupByBottomRowsOn = lazy_relation_methods_mod.groupByBottomRowsOn;
            pub const groupByTopRowsByColumns = lazy_relation_methods_mod.groupByTopRowsByColumns;
            pub const groupByTopRowsByColumnsOn = lazy_relation_methods_mod.groupByTopRowsByColumnsOn;
            pub const groupByBottomRowsByColumns = lazy_relation_methods_mod.groupByBottomRowsByColumns;
            pub const groupByBottomRowsByColumnsOn = lazy_relation_methods_mod.groupByBottomRowsByColumnsOn;
            pub const withGroupId = lazy_relation_methods_mod.withGroupId;
            pub const withGroupIdOn = lazy_relation_methods_mod.withGroupIdOn;
            pub const withGroupIndex = lazy_relation_methods_mod.withGroupIndex;
            pub const withGroupIndexOn = lazy_relation_methods_mod.withGroupIndexOn;
            pub const withGroupFirstRowIndex = lazy_relation_methods_mod.withGroupFirstRowIndex;
            pub const withGroupFirstRowIndexOn = lazy_relation_methods_mod.withGroupFirstRowIndexOn;
            pub const withGroupLastRowIndex = lazy_relation_methods_mod.withGroupLastRowIndex;
            pub const withGroupLastRowIndexOn = lazy_relation_methods_mod.withGroupLastRowIndexOn;
            pub const withGroupIsFirstRow = lazy_relation_methods_mod.withGroupIsFirstRow;
            pub const withGroupIsFirstRowOn = lazy_relation_methods_mod.withGroupIsFirstRowOn;
            pub const withGroupIsLastRow = lazy_relation_methods_mod.withGroupIsLastRow;
            pub const withGroupIsLastRowOn = lazy_relation_methods_mod.withGroupIsLastRowOn;
            pub const withGroupIsSingleton = lazy_relation_methods_mod.withGroupIsSingleton;
            pub const withGroupIsSingletonOn = lazy_relation_methods_mod.withGroupIsSingletonOn;
            pub const withGroupIsDuplicated = lazy_relation_methods_mod.withGroupIsDuplicated;
            pub const withGroupIsDuplicatedOn = lazy_relation_methods_mod.withGroupIsDuplicatedOn;
            pub const withGroupCumeDist = lazy_relation_methods_mod.withGroupCumeDist;
            pub const withGroupCumeDistOn = lazy_relation_methods_mod.withGroupCumeDistOn;
            pub const withGroupCumulativeDistribution = lazy_relation_methods_mod.withGroupCumulativeDistribution;
            pub const withGroupCumulativeDistributionOn = lazy_relation_methods_mod.withGroupCumulativeDistributionOn;
            pub const withGroupPercentRank = lazy_relation_methods_mod.withGroupPercentRank;
            pub const withGroupPercentRankOn = lazy_relation_methods_mod.withGroupPercentRankOn;
            pub const withGroupPercentileRank = lazy_relation_methods_mod.withGroupPercentileRank;
            pub const withGroupPercentileRankOn = lazy_relation_methods_mod.withGroupPercentileRankOn;
            pub const withGroupReverseCumeDist = lazy_relation_methods_mod.withGroupReverseCumeDist;
            pub const withGroupReverseCumeDistOn = lazy_relation_methods_mod.withGroupReverseCumeDistOn;
            pub const withGroupReverseCumulativeDistribution = lazy_relation_methods_mod.withGroupReverseCumulativeDistribution;
            pub const withGroupReverseCumulativeDistributionOn = lazy_relation_methods_mod.withGroupReverseCumulativeDistributionOn;
            pub const withGroupReversePercentRank = lazy_relation_methods_mod.withGroupReversePercentRank;
            pub const withGroupReversePercentRankOn = lazy_relation_methods_mod.withGroupReversePercentRankOn;
            pub const withGroupReversePercentileRank = lazy_relation_methods_mod.withGroupReversePercentileRank;
            pub const withGroupReversePercentileRankOn = lazy_relation_methods_mod.withGroupReversePercentileRankOn;
            pub const withGroupLag = lazy_relation_methods_mod.withGroupLag;
            pub const withGroupLagOn = lazy_relation_methods_mod.withGroupLagOn;
            pub const withGroupLead = lazy_relation_methods_mod.withGroupLead;
            pub const withGroupLeadOn = lazy_relation_methods_mod.withGroupLeadOn;
            pub const withGroupFirstRowValue = lazy_relation_methods_mod.withGroupFirstRowValue;
            pub const withGroupFirstRowValueOn = lazy_relation_methods_mod.withGroupFirstRowValueOn;
            pub const withGroupLastRowValue = lazy_relation_methods_mod.withGroupLastRowValue;
            pub const withGroupLastRowValueOn = lazy_relation_methods_mod.withGroupLastRowValueOn;
            pub const withGroupNthRowValue = lazy_relation_methods_mod.withGroupNthRowValue;
            pub const withGroupNthRowValueOn = lazy_relation_methods_mod.withGroupNthRowValueOn;
            pub const withGroupNthValue = lazy_relation_methods_mod.withGroupNthValue;
            pub const withGroupNthValueOn = lazy_relation_methods_mod.withGroupNthValueOn;
            pub const withGroupFirstValidValue = lazy_relation_methods_mod.withGroupFirstValidValue;
            pub const withGroupFirstValidValueOn = lazy_relation_methods_mod.withGroupFirstValidValueOn;
            pub const withGroupLastValidValue = lazy_relation_methods_mod.withGroupLastValidValue;
            pub const withGroupLastValidValueOn = lazy_relation_methods_mod.withGroupLastValidValueOn;
            pub const withGroupNthValidValue = lazy_relation_methods_mod.withGroupNthValidValue;
            pub const withGroupNthValidValueOn = lazy_relation_methods_mod.withGroupNthValidValueOn;
            pub const withGroupFillNullForward = lazy_relation_methods_mod.withGroupFillNullForward;
            pub const withGroupFillNullForwardOn = lazy_relation_methods_mod.withGroupFillNullForwardOn;
            pub const withGroupFillNullBackward = lazy_relation_methods_mod.withGroupFillNullBackward;
            pub const withGroupFillNullBackwardOn = lazy_relation_methods_mod.withGroupFillNullBackwardOn;
            pub const withGroupCumulativeValidCount = lazy_relation_methods_mod.withGroupCumulativeValidCount;
            pub const withGroupCumulativeValidCountOn = lazy_relation_methods_mod.withGroupCumulativeValidCountOn;
            pub const withGroupCumulativeNullCount = lazy_relation_methods_mod.withGroupCumulativeNullCount;
            pub const withGroupCumulativeNullCountOn = lazy_relation_methods_mod.withGroupCumulativeNullCountOn;
            pub const withGroupCumValidCount = lazy_relation_methods_mod.withGroupCumValidCount;
            pub const withGroupCumValidCountOn = lazy_relation_methods_mod.withGroupCumValidCountOn;
            pub const withGroupCumNullCount = lazy_relation_methods_mod.withGroupCumNullCount;
            pub const withGroupCumNullCountOn = lazy_relation_methods_mod.withGroupCumNullCountOn;
            pub const withGroupCumulativeValidRatio = lazy_relation_methods_mod.withGroupCumulativeValidRatio;
            pub const withGroupCumulativeValidRatioOn = lazy_relation_methods_mod.withGroupCumulativeValidRatioOn;
            pub const withGroupCumulativeNullRatio = lazy_relation_methods_mod.withGroupCumulativeNullRatio;
            pub const withGroupCumulativeNullRatioOn = lazy_relation_methods_mod.withGroupCumulativeNullRatioOn;
            pub const withGroupCumValidRatio = lazy_relation_methods_mod.withGroupCumValidRatio;
            pub const withGroupCumValidRatioOn = lazy_relation_methods_mod.withGroupCumValidRatioOn;
            pub const withGroupCumNullRatio = lazy_relation_methods_mod.withGroupCumNullRatio;
            pub const withGroupCumNullRatioOn = lazy_relation_methods_mod.withGroupCumNullRatioOn;
            pub const withGroupCumulativeFirstValidIndex = lazy_relation_methods_mod.withGroupCumulativeFirstValidIndex;
            pub const withGroupCumulativeFirstValidIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstValidIndexOn;
            pub const withGroupCumulativeLastValidIndex = lazy_relation_methods_mod.withGroupCumulativeLastValidIndex;
            pub const withGroupCumulativeLastValidIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastValidIndexOn;
            pub const withGroupCumulativeFirstNullIndex = lazy_relation_methods_mod.withGroupCumulativeFirstNullIndex;
            pub const withGroupCumulativeFirstNullIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstNullIndexOn;
            pub const withGroupCumulativeLastNullIndex = lazy_relation_methods_mod.withGroupCumulativeLastNullIndex;
            pub const withGroupCumulativeLastNullIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastNullIndexOn;
            pub const withGroupCumFirstValidIndex = lazy_relation_methods_mod.withGroupCumFirstValidIndex;
            pub const withGroupCumFirstValidIndexOn = lazy_relation_methods_mod.withGroupCumFirstValidIndexOn;
            pub const withGroupCumLastValidIndex = lazy_relation_methods_mod.withGroupCumLastValidIndex;
            pub const withGroupCumLastValidIndexOn = lazy_relation_methods_mod.withGroupCumLastValidIndexOn;
            pub const withGroupCumFirstNullIndex = lazy_relation_methods_mod.withGroupCumFirstNullIndex;
            pub const withGroupCumFirstNullIndexOn = lazy_relation_methods_mod.withGroupCumFirstNullIndexOn;
            pub const withGroupCumLastNullIndex = lazy_relation_methods_mod.withGroupCumLastNullIndex;
            pub const withGroupCumLastNullIndexOn = lazy_relation_methods_mod.withGroupCumLastNullIndexOn;
            pub const withGroupCumulativeNaNCount = lazy_relation_methods_mod.withGroupCumulativeNaNCount;
            pub const withGroupCumulativeNaNCountOn = lazy_relation_methods_mod.withGroupCumulativeNaNCountOn;
            pub const withGroupCumulativeNaNRatio = lazy_relation_methods_mod.withGroupCumulativeNaNRatio;
            pub const withGroupCumulativeNaNRatioOn = lazy_relation_methods_mod.withGroupCumulativeNaNRatioOn;
            pub const withGroupCumulativeNanCount = lazy_relation_methods_mod.withGroupCumulativeNanCount;
            pub const withGroupCumulativeNanCountOn = lazy_relation_methods_mod.withGroupCumulativeNanCountOn;
            pub const withGroupCumulativeNanRatio = lazy_relation_methods_mod.withGroupCumulativeNanRatio;
            pub const withGroupCumulativeNanRatioOn = lazy_relation_methods_mod.withGroupCumulativeNanRatioOn;
            pub const withGroupCumulativeInfCount = lazy_relation_methods_mod.withGroupCumulativeInfCount;
            pub const withGroupCumulativeInfCountOn = lazy_relation_methods_mod.withGroupCumulativeInfCountOn;
            pub const withGroupCumulativeInfRatio = lazy_relation_methods_mod.withGroupCumulativeInfRatio;
            pub const withGroupCumulativeInfRatioOn = lazy_relation_methods_mod.withGroupCumulativeInfRatioOn;
            pub const withGroupCumulativePositiveInfCount = lazy_relation_methods_mod.withGroupCumulativePositiveInfCount;
            pub const withGroupCumulativePositiveInfCountOn = lazy_relation_methods_mod.withGroupCumulativePositiveInfCountOn;
            pub const withGroupCumulativePositiveInfRatio = lazy_relation_methods_mod.withGroupCumulativePositiveInfRatio;
            pub const withGroupCumulativePositiveInfRatioOn = lazy_relation_methods_mod.withGroupCumulativePositiveInfRatioOn;
            pub const withGroupCumulativeNegativeInfCount = lazy_relation_methods_mod.withGroupCumulativeNegativeInfCount;
            pub const withGroupCumulativeNegativeInfCountOn = lazy_relation_methods_mod.withGroupCumulativeNegativeInfCountOn;
            pub const withGroupCumulativeNegativeInfRatio = lazy_relation_methods_mod.withGroupCumulativeNegativeInfRatio;
            pub const withGroupCumulativeNegativeInfRatioOn = lazy_relation_methods_mod.withGroupCumulativeNegativeInfRatioOn;
            pub const withGroupCumulativeFiniteCount = lazy_relation_methods_mod.withGroupCumulativeFiniteCount;
            pub const withGroupCumulativeFiniteCountOn = lazy_relation_methods_mod.withGroupCumulativeFiniteCountOn;
            pub const withGroupCumulativeFiniteRatio = lazy_relation_methods_mod.withGroupCumulativeFiniteRatio;
            pub const withGroupCumulativeFiniteRatioOn = lazy_relation_methods_mod.withGroupCumulativeFiniteRatioOn;
            pub const withGroupCumulativeNormalCount = lazy_relation_methods_mod.withGroupCumulativeNormalCount;
            pub const withGroupCumulativeNormalCountOn = lazy_relation_methods_mod.withGroupCumulativeNormalCountOn;
            pub const withGroupCumulativeNormalRatio = lazy_relation_methods_mod.withGroupCumulativeNormalRatio;
            pub const withGroupCumulativeNormalRatioOn = lazy_relation_methods_mod.withGroupCumulativeNormalRatioOn;
            pub const withGroupCumulativeSubnormalCount = lazy_relation_methods_mod.withGroupCumulativeSubnormalCount;
            pub const withGroupCumulativeSubnormalCountOn = lazy_relation_methods_mod.withGroupCumulativeSubnormalCountOn;
            pub const withGroupCumulativeSubnormalRatio = lazy_relation_methods_mod.withGroupCumulativeSubnormalRatio;
            pub const withGroupCumulativeSubnormalRatioOn = lazy_relation_methods_mod.withGroupCumulativeSubnormalRatioOn;
            pub const withGroupCumulativeNonFiniteCount = lazy_relation_methods_mod.withGroupCumulativeNonFiniteCount;
            pub const withGroupCumulativeNonFiniteCountOn = lazy_relation_methods_mod.withGroupCumulativeNonFiniteCountOn;
            pub const withGroupCumulativeNonFiniteRatio = lazy_relation_methods_mod.withGroupCumulativeNonFiniteRatio;
            pub const withGroupCumulativeNonFiniteRatioOn = lazy_relation_methods_mod.withGroupCumulativeNonFiniteRatioOn;
            pub const withGroupCumulativeZeroCount = lazy_relation_methods_mod.withGroupCumulativeZeroCount;
            pub const withGroupCumulativeZeroCountOn = lazy_relation_methods_mod.withGroupCumulativeZeroCountOn;
            pub const withGroupCumulativeZeroRatio = lazy_relation_methods_mod.withGroupCumulativeZeroRatio;
            pub const withGroupCumulativeZeroRatioOn = lazy_relation_methods_mod.withGroupCumulativeZeroRatioOn;
            pub const withGroupCumulativePositiveZeroCount = lazy_relation_methods_mod.withGroupCumulativePositiveZeroCount;
            pub const withGroupCumulativePositiveZeroCountOn = lazy_relation_methods_mod.withGroupCumulativePositiveZeroCountOn;
            pub const withGroupCumulativePositiveZeroRatio = lazy_relation_methods_mod.withGroupCumulativePositiveZeroRatio;
            pub const withGroupCumulativePositiveZeroRatioOn = lazy_relation_methods_mod.withGroupCumulativePositiveZeroRatioOn;
            pub const withGroupCumulativeNegativeZeroCount = lazy_relation_methods_mod.withGroupCumulativeNegativeZeroCount;
            pub const withGroupCumulativeNegativeZeroCountOn = lazy_relation_methods_mod.withGroupCumulativeNegativeZeroCountOn;
            pub const withGroupCumulativeNegativeZeroRatio = lazy_relation_methods_mod.withGroupCumulativeNegativeZeroRatio;
            pub const withGroupCumulativeNegativeZeroRatioOn = lazy_relation_methods_mod.withGroupCumulativeNegativeZeroRatioOn;
            pub const withGroupCumulativeNonZeroCount = lazy_relation_methods_mod.withGroupCumulativeNonZeroCount;
            pub const withGroupCumulativeNonZeroCountOn = lazy_relation_methods_mod.withGroupCumulativeNonZeroCountOn;
            pub const withGroupCumulativeNonZeroRatio = lazy_relation_methods_mod.withGroupCumulativeNonZeroRatio;
            pub const withGroupCumulativeNonZeroRatioOn = lazy_relation_methods_mod.withGroupCumulativeNonZeroRatioOn;
            pub const withGroupCumulativePositiveCount = lazy_relation_methods_mod.withGroupCumulativePositiveCount;
            pub const withGroupCumulativePositiveCountOn = lazy_relation_methods_mod.withGroupCumulativePositiveCountOn;
            pub const withGroupCumulativePositiveRatio = lazy_relation_methods_mod.withGroupCumulativePositiveRatio;
            pub const withGroupCumulativePositiveRatioOn = lazy_relation_methods_mod.withGroupCumulativePositiveRatioOn;
            pub const withGroupCumulativeSignBitCount = lazy_relation_methods_mod.withGroupCumulativeSignBitCount;
            pub const withGroupCumulativeSignBitCountOn = lazy_relation_methods_mod.withGroupCumulativeSignBitCountOn;
            pub const withGroupCumulativeSignBitRatio = lazy_relation_methods_mod.withGroupCumulativeSignBitRatio;
            pub const withGroupCumulativeSignBitRatioOn = lazy_relation_methods_mod.withGroupCumulativeSignBitRatioOn;
            pub const withGroupCumulativeNegativeCount = lazy_relation_methods_mod.withGroupCumulativeNegativeCount;
            pub const withGroupCumulativeNegativeCountOn = lazy_relation_methods_mod.withGroupCumulativeNegativeCountOn;
            pub const withGroupCumulativeNegativeRatio = lazy_relation_methods_mod.withGroupCumulativeNegativeRatio;
            pub const withGroupCumulativeNegativeRatioOn = lazy_relation_methods_mod.withGroupCumulativeNegativeRatioOn;
            pub const withGroupCumNaNCount = lazy_relation_methods_mod.withGroupCumNaNCount;
            pub const withGroupCumNaNCountOn = lazy_relation_methods_mod.withGroupCumNaNCountOn;
            pub const withGroupCumNaNRatio = lazy_relation_methods_mod.withGroupCumNaNRatio;
            pub const withGroupCumNaNRatioOn = lazy_relation_methods_mod.withGroupCumNaNRatioOn;
            pub const withGroupCumNanCount = lazy_relation_methods_mod.withGroupCumNanCount;
            pub const withGroupCumNanCountOn = lazy_relation_methods_mod.withGroupCumNanCountOn;
            pub const withGroupCumNanRatio = lazy_relation_methods_mod.withGroupCumNanRatio;
            pub const withGroupCumNanRatioOn = lazy_relation_methods_mod.withGroupCumNanRatioOn;
            pub const withGroupCumInfCount = lazy_relation_methods_mod.withGroupCumInfCount;
            pub const withGroupCumInfCountOn = lazy_relation_methods_mod.withGroupCumInfCountOn;
            pub const withGroupCumInfRatio = lazy_relation_methods_mod.withGroupCumInfRatio;
            pub const withGroupCumInfRatioOn = lazy_relation_methods_mod.withGroupCumInfRatioOn;
            pub const withGroupCumPositiveInfCount = lazy_relation_methods_mod.withGroupCumPositiveInfCount;
            pub const withGroupCumPositiveInfCountOn = lazy_relation_methods_mod.withGroupCumPositiveInfCountOn;
            pub const withGroupCumPositiveInfRatio = lazy_relation_methods_mod.withGroupCumPositiveInfRatio;
            pub const withGroupCumPositiveInfRatioOn = lazy_relation_methods_mod.withGroupCumPositiveInfRatioOn;
            pub const withGroupCumNegativeInfCount = lazy_relation_methods_mod.withGroupCumNegativeInfCount;
            pub const withGroupCumNegativeInfCountOn = lazy_relation_methods_mod.withGroupCumNegativeInfCountOn;
            pub const withGroupCumNegativeInfRatio = lazy_relation_methods_mod.withGroupCumNegativeInfRatio;
            pub const withGroupCumNegativeInfRatioOn = lazy_relation_methods_mod.withGroupCumNegativeInfRatioOn;
            pub const withGroupCumFiniteCount = lazy_relation_methods_mod.withGroupCumFiniteCount;
            pub const withGroupCumFiniteCountOn = lazy_relation_methods_mod.withGroupCumFiniteCountOn;
            pub const withGroupCumFiniteRatio = lazy_relation_methods_mod.withGroupCumFiniteRatio;
            pub const withGroupCumFiniteRatioOn = lazy_relation_methods_mod.withGroupCumFiniteRatioOn;
            pub const withGroupCumNormalCount = lazy_relation_methods_mod.withGroupCumNormalCount;
            pub const withGroupCumNormalCountOn = lazy_relation_methods_mod.withGroupCumNormalCountOn;
            pub const withGroupCumNormalRatio = lazy_relation_methods_mod.withGroupCumNormalRatio;
            pub const withGroupCumNormalRatioOn = lazy_relation_methods_mod.withGroupCumNormalRatioOn;
            pub const withGroupCumSubnormalCount = lazy_relation_methods_mod.withGroupCumSubnormalCount;
            pub const withGroupCumSubnormalCountOn = lazy_relation_methods_mod.withGroupCumSubnormalCountOn;
            pub const withGroupCumSubnormalRatio = lazy_relation_methods_mod.withGroupCumSubnormalRatio;
            pub const withGroupCumSubnormalRatioOn = lazy_relation_methods_mod.withGroupCumSubnormalRatioOn;
            pub const withGroupCumNonFiniteCount = lazy_relation_methods_mod.withGroupCumNonFiniteCount;
            pub const withGroupCumNonFiniteCountOn = lazy_relation_methods_mod.withGroupCumNonFiniteCountOn;
            pub const withGroupCumNonFiniteRatio = lazy_relation_methods_mod.withGroupCumNonFiniteRatio;
            pub const withGroupCumNonFiniteRatioOn = lazy_relation_methods_mod.withGroupCumNonFiniteRatioOn;
            pub const withGroupCumZeroCount = lazy_relation_methods_mod.withGroupCumZeroCount;
            pub const withGroupCumZeroCountOn = lazy_relation_methods_mod.withGroupCumZeroCountOn;
            pub const withGroupCumZeroRatio = lazy_relation_methods_mod.withGroupCumZeroRatio;
            pub const withGroupCumZeroRatioOn = lazy_relation_methods_mod.withGroupCumZeroRatioOn;
            pub const withGroupCumPositiveZeroCount = lazy_relation_methods_mod.withGroupCumPositiveZeroCount;
            pub const withGroupCumPositiveZeroCountOn = lazy_relation_methods_mod.withGroupCumPositiveZeroCountOn;
            pub const withGroupCumPositiveZeroRatio = lazy_relation_methods_mod.withGroupCumPositiveZeroRatio;
            pub const withGroupCumPositiveZeroRatioOn = lazy_relation_methods_mod.withGroupCumPositiveZeroRatioOn;
            pub const withGroupCumNegativeZeroCount = lazy_relation_methods_mod.withGroupCumNegativeZeroCount;
            pub const withGroupCumNegativeZeroCountOn = lazy_relation_methods_mod.withGroupCumNegativeZeroCountOn;
            pub const withGroupCumNegativeZeroRatio = lazy_relation_methods_mod.withGroupCumNegativeZeroRatio;
            pub const withGroupCumNegativeZeroRatioOn = lazy_relation_methods_mod.withGroupCumNegativeZeroRatioOn;
            pub const withGroupCumNonZeroCount = lazy_relation_methods_mod.withGroupCumNonZeroCount;
            pub const withGroupCumNonZeroCountOn = lazy_relation_methods_mod.withGroupCumNonZeroCountOn;
            pub const withGroupCumNonZeroRatio = lazy_relation_methods_mod.withGroupCumNonZeroRatio;
            pub const withGroupCumNonZeroRatioOn = lazy_relation_methods_mod.withGroupCumNonZeroRatioOn;
            pub const withGroupCumPositiveCount = lazy_relation_methods_mod.withGroupCumPositiveCount;
            pub const withGroupCumPositiveCountOn = lazy_relation_methods_mod.withGroupCumPositiveCountOn;
            pub const withGroupCumPositiveRatio = lazy_relation_methods_mod.withGroupCumPositiveRatio;
            pub const withGroupCumPositiveRatioOn = lazy_relation_methods_mod.withGroupCumPositiveRatioOn;
            pub const withGroupCumSignBitCount = lazy_relation_methods_mod.withGroupCumSignBitCount;
            pub const withGroupCumSignBitCountOn = lazy_relation_methods_mod.withGroupCumSignBitCountOn;
            pub const withGroupCumSignBitRatio = lazy_relation_methods_mod.withGroupCumSignBitRatio;
            pub const withGroupCumSignBitRatioOn = lazy_relation_methods_mod.withGroupCumSignBitRatioOn;
            pub const withGroupCumNegativeCount = lazy_relation_methods_mod.withGroupCumNegativeCount;
            pub const withGroupCumNegativeCountOn = lazy_relation_methods_mod.withGroupCumNegativeCountOn;
            pub const withGroupCumNegativeRatio = lazy_relation_methods_mod.withGroupCumNegativeRatio;
            pub const withGroupCumNegativeRatioOn = lazy_relation_methods_mod.withGroupCumNegativeRatioOn;
            pub const withGroupCumulativeFirstNaNIndex = lazy_relation_methods_mod.withGroupCumulativeFirstNaNIndex;
            pub const withGroupCumulativeFirstNaNIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstNaNIndexOn;
            pub const withGroupCumulativeLastNaNIndex = lazy_relation_methods_mod.withGroupCumulativeLastNaNIndex;
            pub const withGroupCumulativeLastNaNIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastNaNIndexOn;
            pub const withGroupCumulativeFirstNanIndex = lazy_relation_methods_mod.withGroupCumulativeFirstNanIndex;
            pub const withGroupCumulativeFirstNanIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstNanIndexOn;
            pub const withGroupCumulativeLastNanIndex = lazy_relation_methods_mod.withGroupCumulativeLastNanIndex;
            pub const withGroupCumulativeLastNanIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastNanIndexOn;
            pub const withGroupCumulativeFirstInfIndex = lazy_relation_methods_mod.withGroupCumulativeFirstInfIndex;
            pub const withGroupCumulativeFirstInfIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstInfIndexOn;
            pub const withGroupCumulativeLastInfIndex = lazy_relation_methods_mod.withGroupCumulativeLastInfIndex;
            pub const withGroupCumulativeLastInfIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastInfIndexOn;
            pub const withGroupCumulativeFirstPositiveInfIndex = lazy_relation_methods_mod.withGroupCumulativeFirstPositiveInfIndex;
            pub const withGroupCumulativeFirstPositiveInfIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstPositiveInfIndexOn;
            pub const withGroupCumulativeLastPositiveInfIndex = lazy_relation_methods_mod.withGroupCumulativeLastPositiveInfIndex;
            pub const withGroupCumulativeLastPositiveInfIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastPositiveInfIndexOn;
            pub const withGroupCumulativeFirstNegativeInfIndex = lazy_relation_methods_mod.withGroupCumulativeFirstNegativeInfIndex;
            pub const withGroupCumulativeFirstNegativeInfIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstNegativeInfIndexOn;
            pub const withGroupCumulativeLastNegativeInfIndex = lazy_relation_methods_mod.withGroupCumulativeLastNegativeInfIndex;
            pub const withGroupCumulativeLastNegativeInfIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastNegativeInfIndexOn;
            pub const withGroupCumulativeFirstFiniteIndex = lazy_relation_methods_mod.withGroupCumulativeFirstFiniteIndex;
            pub const withGroupCumulativeFirstFiniteIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstFiniteIndexOn;
            pub const withGroupCumulativeLastFiniteIndex = lazy_relation_methods_mod.withGroupCumulativeLastFiniteIndex;
            pub const withGroupCumulativeLastFiniteIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastFiniteIndexOn;
            pub const withGroupCumulativeFirstNormalIndex = lazy_relation_methods_mod.withGroupCumulativeFirstNormalIndex;
            pub const withGroupCumulativeFirstNormalIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstNormalIndexOn;
            pub const withGroupCumulativeLastNormalIndex = lazy_relation_methods_mod.withGroupCumulativeLastNormalIndex;
            pub const withGroupCumulativeLastNormalIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastNormalIndexOn;
            pub const withGroupCumulativeFirstSubnormalIndex = lazy_relation_methods_mod.withGroupCumulativeFirstSubnormalIndex;
            pub const withGroupCumulativeFirstSubnormalIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstSubnormalIndexOn;
            pub const withGroupCumulativeLastSubnormalIndex = lazy_relation_methods_mod.withGroupCumulativeLastSubnormalIndex;
            pub const withGroupCumulativeLastSubnormalIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastSubnormalIndexOn;
            pub const withGroupCumulativeFirstNonFiniteIndex = lazy_relation_methods_mod.withGroupCumulativeFirstNonFiniteIndex;
            pub const withGroupCumulativeFirstNonFiniteIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstNonFiniteIndexOn;
            pub const withGroupCumulativeLastNonFiniteIndex = lazy_relation_methods_mod.withGroupCumulativeLastNonFiniteIndex;
            pub const withGroupCumulativeLastNonFiniteIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastNonFiniteIndexOn;
            pub const withGroupCumulativeFirstZeroIndex = lazy_relation_methods_mod.withGroupCumulativeFirstZeroIndex;
            pub const withGroupCumulativeFirstZeroIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstZeroIndexOn;
            pub const withGroupCumulativeLastZeroIndex = lazy_relation_methods_mod.withGroupCumulativeLastZeroIndex;
            pub const withGroupCumulativeLastZeroIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastZeroIndexOn;
            pub const withGroupCumulativeFirstPositiveZeroIndex = lazy_relation_methods_mod.withGroupCumulativeFirstPositiveZeroIndex;
            pub const withGroupCumulativeFirstPositiveZeroIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstPositiveZeroIndexOn;
            pub const withGroupCumulativeLastPositiveZeroIndex = lazy_relation_methods_mod.withGroupCumulativeLastPositiveZeroIndex;
            pub const withGroupCumulativeLastPositiveZeroIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastPositiveZeroIndexOn;
            pub const withGroupCumulativeFirstNegativeZeroIndex = lazy_relation_methods_mod.withGroupCumulativeFirstNegativeZeroIndex;
            pub const withGroupCumulativeFirstNegativeZeroIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstNegativeZeroIndexOn;
            pub const withGroupCumulativeLastNegativeZeroIndex = lazy_relation_methods_mod.withGroupCumulativeLastNegativeZeroIndex;
            pub const withGroupCumulativeLastNegativeZeroIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastNegativeZeroIndexOn;
            pub const withGroupCumulativeFirstNonZeroIndex = lazy_relation_methods_mod.withGroupCumulativeFirstNonZeroIndex;
            pub const withGroupCumulativeFirstNonZeroIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstNonZeroIndexOn;
            pub const withGroupCumulativeLastNonZeroIndex = lazy_relation_methods_mod.withGroupCumulativeLastNonZeroIndex;
            pub const withGroupCumulativeLastNonZeroIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastNonZeroIndexOn;
            pub const withGroupCumulativeFirstPositiveIndex = lazy_relation_methods_mod.withGroupCumulativeFirstPositiveIndex;
            pub const withGroupCumulativeFirstPositiveIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstPositiveIndexOn;
            pub const withGroupCumulativeLastPositiveIndex = lazy_relation_methods_mod.withGroupCumulativeLastPositiveIndex;
            pub const withGroupCumulativeLastPositiveIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastPositiveIndexOn;
            pub const withGroupCumulativeFirstSignBitIndex = lazy_relation_methods_mod.withGroupCumulativeFirstSignBitIndex;
            pub const withGroupCumulativeFirstSignBitIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstSignBitIndexOn;
            pub const withGroupCumulativeLastSignBitIndex = lazy_relation_methods_mod.withGroupCumulativeLastSignBitIndex;
            pub const withGroupCumulativeLastSignBitIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastSignBitIndexOn;
            pub const withGroupCumulativeFirstNegativeIndex = lazy_relation_methods_mod.withGroupCumulativeFirstNegativeIndex;
            pub const withGroupCumulativeFirstNegativeIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstNegativeIndexOn;
            pub const withGroupCumulativeLastNegativeIndex = lazy_relation_methods_mod.withGroupCumulativeLastNegativeIndex;
            pub const withGroupCumulativeLastNegativeIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastNegativeIndexOn;
            pub const withGroupCumFirstNaNIndex = lazy_relation_methods_mod.withGroupCumFirstNaNIndex;
            pub const withGroupCumFirstNaNIndexOn = lazy_relation_methods_mod.withGroupCumFirstNaNIndexOn;
            pub const withGroupCumLastNaNIndex = lazy_relation_methods_mod.withGroupCumLastNaNIndex;
            pub const withGroupCumLastNaNIndexOn = lazy_relation_methods_mod.withGroupCumLastNaNIndexOn;
            pub const withGroupCumFirstNanIndex = lazy_relation_methods_mod.withGroupCumFirstNanIndex;
            pub const withGroupCumFirstNanIndexOn = lazy_relation_methods_mod.withGroupCumFirstNanIndexOn;
            pub const withGroupCumLastNanIndex = lazy_relation_methods_mod.withGroupCumLastNanIndex;
            pub const withGroupCumLastNanIndexOn = lazy_relation_methods_mod.withGroupCumLastNanIndexOn;
            pub const withGroupCumFirstInfIndex = lazy_relation_methods_mod.withGroupCumFirstInfIndex;
            pub const withGroupCumFirstInfIndexOn = lazy_relation_methods_mod.withGroupCumFirstInfIndexOn;
            pub const withGroupCumLastInfIndex = lazy_relation_methods_mod.withGroupCumLastInfIndex;
            pub const withGroupCumLastInfIndexOn = lazy_relation_methods_mod.withGroupCumLastInfIndexOn;
            pub const withGroupCumFirstPositiveInfIndex = lazy_relation_methods_mod.withGroupCumFirstPositiveInfIndex;
            pub const withGroupCumFirstPositiveInfIndexOn = lazy_relation_methods_mod.withGroupCumFirstPositiveInfIndexOn;
            pub const withGroupCumLastPositiveInfIndex = lazy_relation_methods_mod.withGroupCumLastPositiveInfIndex;
            pub const withGroupCumLastPositiveInfIndexOn = lazy_relation_methods_mod.withGroupCumLastPositiveInfIndexOn;
            pub const withGroupCumFirstNegativeInfIndex = lazy_relation_methods_mod.withGroupCumFirstNegativeInfIndex;
            pub const withGroupCumFirstNegativeInfIndexOn = lazy_relation_methods_mod.withGroupCumFirstNegativeInfIndexOn;
            pub const withGroupCumLastNegativeInfIndex = lazy_relation_methods_mod.withGroupCumLastNegativeInfIndex;
            pub const withGroupCumLastNegativeInfIndexOn = lazy_relation_methods_mod.withGroupCumLastNegativeInfIndexOn;
            pub const withGroupCumFirstFiniteIndex = lazy_relation_methods_mod.withGroupCumFirstFiniteIndex;
            pub const withGroupCumFirstFiniteIndexOn = lazy_relation_methods_mod.withGroupCumFirstFiniteIndexOn;
            pub const withGroupCumLastFiniteIndex = lazy_relation_methods_mod.withGroupCumLastFiniteIndex;
            pub const withGroupCumLastFiniteIndexOn = lazy_relation_methods_mod.withGroupCumLastFiniteIndexOn;
            pub const withGroupCumFirstNormalIndex = lazy_relation_methods_mod.withGroupCumFirstNormalIndex;
            pub const withGroupCumFirstNormalIndexOn = lazy_relation_methods_mod.withGroupCumFirstNormalIndexOn;
            pub const withGroupCumLastNormalIndex = lazy_relation_methods_mod.withGroupCumLastNormalIndex;
            pub const withGroupCumLastNormalIndexOn = lazy_relation_methods_mod.withGroupCumLastNormalIndexOn;
            pub const withGroupCumFirstSubnormalIndex = lazy_relation_methods_mod.withGroupCumFirstSubnormalIndex;
            pub const withGroupCumFirstSubnormalIndexOn = lazy_relation_methods_mod.withGroupCumFirstSubnormalIndexOn;
            pub const withGroupCumLastSubnormalIndex = lazy_relation_methods_mod.withGroupCumLastSubnormalIndex;
            pub const withGroupCumLastSubnormalIndexOn = lazy_relation_methods_mod.withGroupCumLastSubnormalIndexOn;
            pub const withGroupCumFirstNonFiniteIndex = lazy_relation_methods_mod.withGroupCumFirstNonFiniteIndex;
            pub const withGroupCumFirstNonFiniteIndexOn = lazy_relation_methods_mod.withGroupCumFirstNonFiniteIndexOn;
            pub const withGroupCumLastNonFiniteIndex = lazy_relation_methods_mod.withGroupCumLastNonFiniteIndex;
            pub const withGroupCumLastNonFiniteIndexOn = lazy_relation_methods_mod.withGroupCumLastNonFiniteIndexOn;
            pub const withGroupCumFirstZeroIndex = lazy_relation_methods_mod.withGroupCumFirstZeroIndex;
            pub const withGroupCumFirstZeroIndexOn = lazy_relation_methods_mod.withGroupCumFirstZeroIndexOn;
            pub const withGroupCumLastZeroIndex = lazy_relation_methods_mod.withGroupCumLastZeroIndex;
            pub const withGroupCumLastZeroIndexOn = lazy_relation_methods_mod.withGroupCumLastZeroIndexOn;
            pub const withGroupCumFirstPositiveZeroIndex = lazy_relation_methods_mod.withGroupCumFirstPositiveZeroIndex;
            pub const withGroupCumFirstPositiveZeroIndexOn = lazy_relation_methods_mod.withGroupCumFirstPositiveZeroIndexOn;
            pub const withGroupCumLastPositiveZeroIndex = lazy_relation_methods_mod.withGroupCumLastPositiveZeroIndex;
            pub const withGroupCumLastPositiveZeroIndexOn = lazy_relation_methods_mod.withGroupCumLastPositiveZeroIndexOn;
            pub const withGroupCumFirstNegativeZeroIndex = lazy_relation_methods_mod.withGroupCumFirstNegativeZeroIndex;
            pub const withGroupCumFirstNegativeZeroIndexOn = lazy_relation_methods_mod.withGroupCumFirstNegativeZeroIndexOn;
            pub const withGroupCumLastNegativeZeroIndex = lazy_relation_methods_mod.withGroupCumLastNegativeZeroIndex;
            pub const withGroupCumLastNegativeZeroIndexOn = lazy_relation_methods_mod.withGroupCumLastNegativeZeroIndexOn;
            pub const withGroupCumFirstNonZeroIndex = lazy_relation_methods_mod.withGroupCumFirstNonZeroIndex;
            pub const withGroupCumFirstNonZeroIndexOn = lazy_relation_methods_mod.withGroupCumFirstNonZeroIndexOn;
            pub const withGroupCumLastNonZeroIndex = lazy_relation_methods_mod.withGroupCumLastNonZeroIndex;
            pub const withGroupCumLastNonZeroIndexOn = lazy_relation_methods_mod.withGroupCumLastNonZeroIndexOn;
            pub const withGroupCumFirstPositiveIndex = lazy_relation_methods_mod.withGroupCumFirstPositiveIndex;
            pub const withGroupCumFirstPositiveIndexOn = lazy_relation_methods_mod.withGroupCumFirstPositiveIndexOn;
            pub const withGroupCumLastPositiveIndex = lazy_relation_methods_mod.withGroupCumLastPositiveIndex;
            pub const withGroupCumLastPositiveIndexOn = lazy_relation_methods_mod.withGroupCumLastPositiveIndexOn;
            pub const withGroupCumFirstSignBitIndex = lazy_relation_methods_mod.withGroupCumFirstSignBitIndex;
            pub const withGroupCumFirstSignBitIndexOn = lazy_relation_methods_mod.withGroupCumFirstSignBitIndexOn;
            pub const withGroupCumLastSignBitIndex = lazy_relation_methods_mod.withGroupCumLastSignBitIndex;
            pub const withGroupCumLastSignBitIndexOn = lazy_relation_methods_mod.withGroupCumLastSignBitIndexOn;
            pub const withGroupCumFirstNegativeIndex = lazy_relation_methods_mod.withGroupCumFirstNegativeIndex;
            pub const withGroupCumFirstNegativeIndexOn = lazy_relation_methods_mod.withGroupCumFirstNegativeIndexOn;
            pub const withGroupCumLastNegativeIndex = lazy_relation_methods_mod.withGroupCumLastNegativeIndex;
            pub const withGroupCumLastNegativeIndexOn = lazy_relation_methods_mod.withGroupCumLastNegativeIndexOn;
            pub const withGroupCumulativeDistinctCount = lazy_relation_methods_mod.withGroupCumulativeDistinctCount;
            pub const withGroupCumulativeDistinctCountOn = lazy_relation_methods_mod.withGroupCumulativeDistinctCountOn;
            pub const withGroupCumulativeCountDistinct = lazy_relation_methods_mod.withGroupCumulativeCountDistinct;
            pub const withGroupCumulativeCountDistinctOn = lazy_relation_methods_mod.withGroupCumulativeCountDistinctOn;
            pub const withGroupCumulativeNUnique = lazy_relation_methods_mod.withGroupCumulativeNUnique;
            pub const withGroupCumulativeNUniqueOn = lazy_relation_methods_mod.withGroupCumulativeNUniqueOn;
            pub const withGroupCumulativeNunique = lazy_relation_methods_mod.withGroupCumulativeNunique;
            pub const withGroupCumulativeNuniqueOn = lazy_relation_methods_mod.withGroupCumulativeNuniqueOn;
            pub const withGroupCumDistinctCount = lazy_relation_methods_mod.withGroupCumDistinctCount;
            pub const withGroupCumDistinctCountOn = lazy_relation_methods_mod.withGroupCumDistinctCountOn;
            pub const withGroupCumCountDistinct = lazy_relation_methods_mod.withGroupCumCountDistinct;
            pub const withGroupCumCountDistinctOn = lazy_relation_methods_mod.withGroupCumCountDistinctOn;
            pub const withGroupCumNUnique = lazy_relation_methods_mod.withGroupCumNUnique;
            pub const withGroupCumNUniqueOn = lazy_relation_methods_mod.withGroupCumNUniqueOn;
            pub const withGroupCumNunique = lazy_relation_methods_mod.withGroupCumNunique;
            pub const withGroupCumNuniqueOn = lazy_relation_methods_mod.withGroupCumNuniqueOn;
            pub const withGroupCumulativeMode = lazy_relation_methods_mod.withGroupCumulativeMode;
            pub const withGroupCumulativeModeOn = lazy_relation_methods_mod.withGroupCumulativeModeOn;
            pub const withGroupCumulativeModeCount = lazy_relation_methods_mod.withGroupCumulativeModeCount;
            pub const withGroupCumulativeModeCountOn = lazy_relation_methods_mod.withGroupCumulativeModeCountOn;
            pub const withGroupCumulativeModeRatio = lazy_relation_methods_mod.withGroupCumulativeModeRatio;
            pub const withGroupCumulativeModeRatioOn = lazy_relation_methods_mod.withGroupCumulativeModeRatioOn;
            pub const withGroupCumulativeModeMargin = lazy_relation_methods_mod.withGroupCumulativeModeMargin;
            pub const withGroupCumulativeModeMarginOn = lazy_relation_methods_mod.withGroupCumulativeModeMarginOn;
            pub const withGroupCumulativeModeMarginRatio = lazy_relation_methods_mod.withGroupCumulativeModeMarginRatio;
            pub const withGroupCumulativeModeMarginRatioOn = lazy_relation_methods_mod.withGroupCumulativeModeMarginRatioOn;
            pub const withGroupCumMode = lazy_relation_methods_mod.withGroupCumMode;
            pub const withGroupCumModeOn = lazy_relation_methods_mod.withGroupCumModeOn;
            pub const withGroupCumModeCount = lazy_relation_methods_mod.withGroupCumModeCount;
            pub const withGroupCumModeCountOn = lazy_relation_methods_mod.withGroupCumModeCountOn;
            pub const withGroupCumModeRatio = lazy_relation_methods_mod.withGroupCumModeRatio;
            pub const withGroupCumModeRatioOn = lazy_relation_methods_mod.withGroupCumModeRatioOn;
            pub const withGroupCumModeMargin = lazy_relation_methods_mod.withGroupCumModeMargin;
            pub const withGroupCumModeMarginOn = lazy_relation_methods_mod.withGroupCumModeMarginOn;
            pub const withGroupCumModeMarginRatio = lazy_relation_methods_mod.withGroupCumModeMarginRatio;
            pub const withGroupCumModeMarginRatioOn = lazy_relation_methods_mod.withGroupCumModeMarginRatioOn;
            pub const withGroupCumulativeEntropy = lazy_relation_methods_mod.withGroupCumulativeEntropy;
            pub const withGroupCumulativeEntropyOn = lazy_relation_methods_mod.withGroupCumulativeEntropyOn;
            pub const withGroupCumulativeGiniImpurity = lazy_relation_methods_mod.withGroupCumulativeGiniImpurity;
            pub const withGroupCumulativeGiniImpurityOn = lazy_relation_methods_mod.withGroupCumulativeGiniImpurityOn;
            pub const withGroupCumulativePerplexity = lazy_relation_methods_mod.withGroupCumulativePerplexity;
            pub const withGroupCumulativePerplexityOn = lazy_relation_methods_mod.withGroupCumulativePerplexityOn;
            pub const withGroupCumulativeInverseSimpson = lazy_relation_methods_mod.withGroupCumulativeInverseSimpson;
            pub const withGroupCumulativeInverseSimpsonOn = lazy_relation_methods_mod.withGroupCumulativeInverseSimpsonOn;
            pub const withGroupCumulativeSimpsonConcentration = lazy_relation_methods_mod.withGroupCumulativeSimpsonConcentration;
            pub const withGroupCumulativeSimpsonConcentrationOn = lazy_relation_methods_mod.withGroupCumulativeSimpsonConcentrationOn;
            pub const withGroupCumulativeEvenness = lazy_relation_methods_mod.withGroupCumulativeEvenness;
            pub const withGroupCumulativeEvennessOn = lazy_relation_methods_mod.withGroupCumulativeEvennessOn;
            pub const withGroupCumulativeGini = lazy_relation_methods_mod.withGroupCumulativeGini;
            pub const withGroupCumulativeGiniOn = lazy_relation_methods_mod.withGroupCumulativeGiniOn;
            pub const withGroupCumulativeConcentration = lazy_relation_methods_mod.withGroupCumulativeConcentration;
            pub const withGroupCumulativeConcentrationOn = lazy_relation_methods_mod.withGroupCumulativeConcentrationOn;
            pub const withGroupCumEntropy = lazy_relation_methods_mod.withGroupCumEntropy;
            pub const withGroupCumEntropyOn = lazy_relation_methods_mod.withGroupCumEntropyOn;
            pub const withGroupCumGiniImpurity = lazy_relation_methods_mod.withGroupCumGiniImpurity;
            pub const withGroupCumGiniImpurityOn = lazy_relation_methods_mod.withGroupCumGiniImpurityOn;
            pub const withGroupCumPerplexity = lazy_relation_methods_mod.withGroupCumPerplexity;
            pub const withGroupCumPerplexityOn = lazy_relation_methods_mod.withGroupCumPerplexityOn;
            pub const withGroupCumInverseSimpson = lazy_relation_methods_mod.withGroupCumInverseSimpson;
            pub const withGroupCumInverseSimpsonOn = lazy_relation_methods_mod.withGroupCumInverseSimpsonOn;
            pub const withGroupCumSimpsonConcentration = lazy_relation_methods_mod.withGroupCumSimpsonConcentration;
            pub const withGroupCumSimpsonConcentrationOn = lazy_relation_methods_mod.withGroupCumSimpsonConcentrationOn;
            pub const withGroupCumEvenness = lazy_relation_methods_mod.withGroupCumEvenness;
            pub const withGroupCumEvennessOn = lazy_relation_methods_mod.withGroupCumEvennessOn;
            pub const withGroupCumGini = lazy_relation_methods_mod.withGroupCumGini;
            pub const withGroupCumGiniOn = lazy_relation_methods_mod.withGroupCumGiniOn;
            pub const withGroupCumConcentration = lazy_relation_methods_mod.withGroupCumConcentration;
            pub const withGroupCumConcentrationOn = lazy_relation_methods_mod.withGroupCumConcentrationOn;
            pub const withGroupCumulativeMeanAbsDev = lazy_relation_methods_mod.withGroupCumulativeMeanAbsDev;
            pub const withGroupCumulativeMeanAbsDevOn = lazy_relation_methods_mod.withGroupCumulativeMeanAbsDevOn;
            pub const withGroupCumulativeMeanAbsDevRatio = lazy_relation_methods_mod.withGroupCumulativeMeanAbsDevRatio;
            pub const withGroupCumulativeMeanAbsDevRatioOn = lazy_relation_methods_mod.withGroupCumulativeMeanAbsDevRatioOn;
            pub const withGroupCumulativeGiniMeanDiff = lazy_relation_methods_mod.withGroupCumulativeGiniMeanDiff;
            pub const withGroupCumulativeGiniMeanDiffOn = lazy_relation_methods_mod.withGroupCumulativeGiniMeanDiffOn;
            pub const withGroupCumulativeGiniCoefficient = lazy_relation_methods_mod.withGroupCumulativeGiniCoefficient;
            pub const withGroupCumulativeGiniCoefficientOn = lazy_relation_methods_mod.withGroupCumulativeGiniCoefficientOn;
            pub const withGroupCumulativeMeanAbsoluteDeviation = lazy_relation_methods_mod.withGroupCumulativeMeanAbsoluteDeviation;
            pub const withGroupCumulativeMeanAbsoluteDeviationOn = lazy_relation_methods_mod.withGroupCumulativeMeanAbsoluteDeviationOn;
            pub const withGroupCumulativeGiniCoeff = lazy_relation_methods_mod.withGroupCumulativeGiniCoeff;
            pub const withGroupCumulativeGiniCoeffOn = lazy_relation_methods_mod.withGroupCumulativeGiniCoeffOn;
            pub const withGroupCumMeanAbsDev = lazy_relation_methods_mod.withGroupCumMeanAbsDev;
            pub const withGroupCumMeanAbsDevOn = lazy_relation_methods_mod.withGroupCumMeanAbsDevOn;
            pub const withGroupCumMeanAbsDevRatio = lazy_relation_methods_mod.withGroupCumMeanAbsDevRatio;
            pub const withGroupCumMeanAbsDevRatioOn = lazy_relation_methods_mod.withGroupCumMeanAbsDevRatioOn;
            pub const withGroupCumGiniMeanDiff = lazy_relation_methods_mod.withGroupCumGiniMeanDiff;
            pub const withGroupCumGiniMeanDiffOn = lazy_relation_methods_mod.withGroupCumGiniMeanDiffOn;
            pub const withGroupCumGiniCoefficient = lazy_relation_methods_mod.withGroupCumGiniCoefficient;
            pub const withGroupCumGiniCoefficientOn = lazy_relation_methods_mod.withGroupCumGiniCoefficientOn;
            pub const withGroupCumMeanAbsoluteDeviation = lazy_relation_methods_mod.withGroupCumMeanAbsoluteDeviation;
            pub const withGroupCumMeanAbsoluteDeviationOn = lazy_relation_methods_mod.withGroupCumMeanAbsoluteDeviationOn;
            pub const withGroupCumGiniCoeff = lazy_relation_methods_mod.withGroupCumGiniCoeff;
            pub const withGroupCumGiniCoeffOn = lazy_relation_methods_mod.withGroupCumGiniCoeffOn;
            pub const withGroupCumulativeMedian = lazy_relation_methods_mod.withGroupCumulativeMedian;
            pub const withGroupCumulativeMedianOn = lazy_relation_methods_mod.withGroupCumulativeMedianOn;
            pub const withGroupCumulativeQuantile = lazy_relation_methods_mod.withGroupCumulativeQuantile;
            pub const withGroupCumulativeQuantileOn = lazy_relation_methods_mod.withGroupCumulativeQuantileOn;
            pub const withGroupCumQuantile = lazy_relation_methods_mod.withGroupCumQuantile;
            pub const withGroupCumQuantileOn = lazy_relation_methods_mod.withGroupCumQuantileOn;
            pub const withGroupCumMedian = lazy_relation_methods_mod.withGroupCumMedian;
            pub const withGroupCumMedianOn = lazy_relation_methods_mod.withGroupCumMedianOn;
            pub const withGroupCumulativeIqr = lazy_relation_methods_mod.withGroupCumulativeIqr;
            pub const withGroupCumulativeIqrOn = lazy_relation_methods_mod.withGroupCumulativeIqrOn;
            pub const withGroupCumulativeIQR = lazy_relation_methods_mod.withGroupCumulativeIQR;
            pub const withGroupCumulativeIQROn = lazy_relation_methods_mod.withGroupCumulativeIQROn;
            pub const withGroupCumIqr = lazy_relation_methods_mod.withGroupCumIqr;
            pub const withGroupCumIqrOn = lazy_relation_methods_mod.withGroupCumIqrOn;
            pub const withGroupCumIQR = lazy_relation_methods_mod.withGroupCumIQR;
            pub const withGroupCumIQROn = lazy_relation_methods_mod.withGroupCumIQROn;
            pub const withGroupCumulativeMad = lazy_relation_methods_mod.withGroupCumulativeMad;
            pub const withGroupCumulativeMadOn = lazy_relation_methods_mod.withGroupCumulativeMadOn;
            pub const withGroupCumulativeMAD = lazy_relation_methods_mod.withGroupCumulativeMAD;
            pub const withGroupCumulativeMADOn = lazy_relation_methods_mod.withGroupCumulativeMADOn;
            pub const withGroupCumulativeMedianAbsDev = lazy_relation_methods_mod.withGroupCumulativeMedianAbsDev;
            pub const withGroupCumulativeMedianAbsDevOn = lazy_relation_methods_mod.withGroupCumulativeMedianAbsDevOn;
            pub const withGroupCumMad = lazy_relation_methods_mod.withGroupCumMad;
            pub const withGroupCumMadOn = lazy_relation_methods_mod.withGroupCumMadOn;
            pub const withGroupCumMAD = lazy_relation_methods_mod.withGroupCumMAD;
            pub const withGroupCumMADOn = lazy_relation_methods_mod.withGroupCumMADOn;
            pub const withGroupCumMedianAbsDev = lazy_relation_methods_mod.withGroupCumMedianAbsDev;
            pub const withGroupCumMedianAbsDevOn = lazy_relation_methods_mod.withGroupCumMedianAbsDevOn;
            pub const withGroupCumulativeTrimmedMean = lazy_relation_methods_mod.withGroupCumulativeTrimmedMean;
            pub const withGroupCumulativeTrimmedMeanOn = lazy_relation_methods_mod.withGroupCumulativeTrimmedMeanOn;
            pub const withGroupCumTrimmedMean = lazy_relation_methods_mod.withGroupCumTrimmedMean;
            pub const withGroupCumTrimmedMeanOn = lazy_relation_methods_mod.withGroupCumTrimmedMeanOn;
            pub const withGroupCumulativeWinsorizedMean = lazy_relation_methods_mod.withGroupCumulativeWinsorizedMean;
            pub const withGroupCumulativeWinsorizedMeanOn = lazy_relation_methods_mod.withGroupCumulativeWinsorizedMeanOn;
            pub const withGroupCumWinsorizedMean = lazy_relation_methods_mod.withGroupCumWinsorizedMean;
            pub const withGroupCumWinsorizedMeanOn = lazy_relation_methods_mod.withGroupCumWinsorizedMeanOn;
            pub const withGroupCumulativeInterdecileRange = lazy_relation_methods_mod.withGroupCumulativeInterdecileRange;
            pub const withGroupCumulativeInterdecileRangeOn = lazy_relation_methods_mod.withGroupCumulativeInterdecileRangeOn;
            pub const withGroupCumulativeIdr = lazy_relation_methods_mod.withGroupCumulativeIdr;
            pub const withGroupCumulativeIdrOn = lazy_relation_methods_mod.withGroupCumulativeIdrOn;
            pub const withGroupCumulativeIDR = lazy_relation_methods_mod.withGroupCumulativeIDR;
            pub const withGroupCumulativeIDROn = lazy_relation_methods_mod.withGroupCumulativeIDROn;
            pub const withGroupCumIdr = lazy_relation_methods_mod.withGroupCumIdr;
            pub const withGroupCumIdrOn = lazy_relation_methods_mod.withGroupCumIdrOn;
            pub const withGroupCumIDR = lazy_relation_methods_mod.withGroupCumIDR;
            pub const withGroupCumIDROn = lazy_relation_methods_mod.withGroupCumIDROn;
            pub const withGroupCumulativeMidhinge = lazy_relation_methods_mod.withGroupCumulativeMidhinge;
            pub const withGroupCumulativeMidhingeOn = lazy_relation_methods_mod.withGroupCumulativeMidhingeOn;
            pub const withGroupCumMidhinge = lazy_relation_methods_mod.withGroupCumMidhinge;
            pub const withGroupCumMidhingeOn = lazy_relation_methods_mod.withGroupCumMidhingeOn;
            pub const withGroupCumulativeTrimean = lazy_relation_methods_mod.withGroupCumulativeTrimean;
            pub const withGroupCumulativeTrimeanOn = lazy_relation_methods_mod.withGroupCumulativeTrimeanOn;
            pub const withGroupCumTrimean = lazy_relation_methods_mod.withGroupCumTrimean;
            pub const withGroupCumTrimeanOn = lazy_relation_methods_mod.withGroupCumTrimeanOn;
            pub const withGroupCumulativeBowleySkewness = lazy_relation_methods_mod.withGroupCumulativeBowleySkewness;
            pub const withGroupCumulativeBowleySkewnessOn = lazy_relation_methods_mod.withGroupCumulativeBowleySkewnessOn;
            pub const withGroupCumulativeBowleySkew = lazy_relation_methods_mod.withGroupCumulativeBowleySkew;
            pub const withGroupCumulativeBowleySkewOn = lazy_relation_methods_mod.withGroupCumulativeBowleySkewOn;
            pub const withGroupCumBowleySkewness = lazy_relation_methods_mod.withGroupCumBowleySkewness;
            pub const withGroupCumBowleySkewnessOn = lazy_relation_methods_mod.withGroupCumBowleySkewnessOn;
            pub const withGroupCumBowleySkew = lazy_relation_methods_mod.withGroupCumBowleySkew;
            pub const withGroupCumBowleySkewOn = lazy_relation_methods_mod.withGroupCumBowleySkewOn;
            pub const withGroupCumulativeQuartileCoeffDispersion = lazy_relation_methods_mod.withGroupCumulativeQuartileCoeffDispersion;
            pub const withGroupCumulativeQuartileCoeffDispersionOn = lazy_relation_methods_mod.withGroupCumulativeQuartileCoeffDispersionOn;
            pub const withGroupCumulativeQcd = lazy_relation_methods_mod.withGroupCumulativeQcd;
            pub const withGroupCumulativeQcdOn = lazy_relation_methods_mod.withGroupCumulativeQcdOn;
            pub const withGroupCumulativeQCD = lazy_relation_methods_mod.withGroupCumulativeQCD;
            pub const withGroupCumulativeQCDOn = lazy_relation_methods_mod.withGroupCumulativeQCDOn;
            pub const withGroupCumQuartileCoeffDispersion = lazy_relation_methods_mod.withGroupCumQuartileCoeffDispersion;
            pub const withGroupCumQuartileCoeffDispersionOn = lazy_relation_methods_mod.withGroupCumQuartileCoeffDispersionOn;
            pub const withGroupCumQcd = lazy_relation_methods_mod.withGroupCumQcd;
            pub const withGroupCumQcdOn = lazy_relation_methods_mod.withGroupCumQcdOn;
            pub const withGroupCumQCD = lazy_relation_methods_mod.withGroupCumQCD;
            pub const withGroupCumQCDOn = lazy_relation_methods_mod.withGroupCumQCDOn;
            pub const withGroupCumulativeKelleySkewness = lazy_relation_methods_mod.withGroupCumulativeKelleySkewness;
            pub const withGroupCumulativeKelleySkewnessOn = lazy_relation_methods_mod.withGroupCumulativeKelleySkewnessOn;
            pub const withGroupCumulativeKelleySkew = lazy_relation_methods_mod.withGroupCumulativeKelleySkew;
            pub const withGroupCumulativeKelleySkewOn = lazy_relation_methods_mod.withGroupCumulativeKelleySkewOn;
            pub const withGroupCumKelleySkewness = lazy_relation_methods_mod.withGroupCumKelleySkewness;
            pub const withGroupCumKelleySkewnessOn = lazy_relation_methods_mod.withGroupCumKelleySkewnessOn;
            pub const withGroupCumKelleySkew = lazy_relation_methods_mod.withGroupCumKelleySkew;
            pub const withGroupCumKelleySkewOn = lazy_relation_methods_mod.withGroupCumKelleySkewOn;
            pub const withGroupCumulativeAny = lazy_relation_methods_mod.withGroupCumulativeAny;
            pub const withGroupCumulativeAnyOn = lazy_relation_methods_mod.withGroupCumulativeAnyOn;
            pub const withGroupCumulativeAll = lazy_relation_methods_mod.withGroupCumulativeAll;
            pub const withGroupCumulativeAllOn = lazy_relation_methods_mod.withGroupCumulativeAllOn;
            pub const withGroupCumulativeTrueCount = lazy_relation_methods_mod.withGroupCumulativeTrueCount;
            pub const withGroupCumulativeTrueCountOn = lazy_relation_methods_mod.withGroupCumulativeTrueCountOn;
            pub const withGroupCumulativeFalseCount = lazy_relation_methods_mod.withGroupCumulativeFalseCount;
            pub const withGroupCumulativeFalseCountOn = lazy_relation_methods_mod.withGroupCumulativeFalseCountOn;
            pub const withGroupCumulativeTrueRatio = lazy_relation_methods_mod.withGroupCumulativeTrueRatio;
            pub const withGroupCumulativeTrueRatioOn = lazy_relation_methods_mod.withGroupCumulativeTrueRatioOn;
            pub const withGroupCumulativeFalseRatio = lazy_relation_methods_mod.withGroupCumulativeFalseRatio;
            pub const withGroupCumulativeFalseRatioOn = lazy_relation_methods_mod.withGroupCumulativeFalseRatioOn;
            pub const withGroupCumAny = lazy_relation_methods_mod.withGroupCumAny;
            pub const withGroupCumAnyOn = lazy_relation_methods_mod.withGroupCumAnyOn;
            pub const withGroupCumAll = lazy_relation_methods_mod.withGroupCumAll;
            pub const withGroupCumAllOn = lazy_relation_methods_mod.withGroupCumAllOn;
            pub const withGroupCumTrueCount = lazy_relation_methods_mod.withGroupCumTrueCount;
            pub const withGroupCumTrueCountOn = lazy_relation_methods_mod.withGroupCumTrueCountOn;
            pub const withGroupCumFalseCount = lazy_relation_methods_mod.withGroupCumFalseCount;
            pub const withGroupCumFalseCountOn = lazy_relation_methods_mod.withGroupCumFalseCountOn;
            pub const withGroupCumTrueRatio = lazy_relation_methods_mod.withGroupCumTrueRatio;
            pub const withGroupCumTrueRatioOn = lazy_relation_methods_mod.withGroupCumTrueRatioOn;
            pub const withGroupCumFalseRatio = lazy_relation_methods_mod.withGroupCumFalseRatio;
            pub const withGroupCumFalseRatioOn = lazy_relation_methods_mod.withGroupCumFalseRatioOn;
            pub const withGroupCumulativeFirstTrueIndex = lazy_relation_methods_mod.withGroupCumulativeFirstTrueIndex;
            pub const withGroupCumulativeFirstTrueIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstTrueIndexOn;
            pub const withGroupCumulativeLastTrueIndex = lazy_relation_methods_mod.withGroupCumulativeLastTrueIndex;
            pub const withGroupCumulativeLastTrueIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastTrueIndexOn;
            pub const withGroupCumulativeFirstFalseIndex = lazy_relation_methods_mod.withGroupCumulativeFirstFalseIndex;
            pub const withGroupCumulativeFirstFalseIndexOn = lazy_relation_methods_mod.withGroupCumulativeFirstFalseIndexOn;
            pub const withGroupCumulativeLastFalseIndex = lazy_relation_methods_mod.withGroupCumulativeLastFalseIndex;
            pub const withGroupCumulativeLastFalseIndexOn = lazy_relation_methods_mod.withGroupCumulativeLastFalseIndexOn;
            pub const withGroupCumFirstTrueIndex = lazy_relation_methods_mod.withGroupCumFirstTrueIndex;
            pub const withGroupCumFirstTrueIndexOn = lazy_relation_methods_mod.withGroupCumFirstTrueIndexOn;
            pub const withGroupCumLastTrueIndex = lazy_relation_methods_mod.withGroupCumLastTrueIndex;
            pub const withGroupCumLastTrueIndexOn = lazy_relation_methods_mod.withGroupCumLastTrueIndexOn;
            pub const withGroupCumFirstFalseIndex = lazy_relation_methods_mod.withGroupCumFirstFalseIndex;
            pub const withGroupCumFirstFalseIndexOn = lazy_relation_methods_mod.withGroupCumFirstFalseIndexOn;
            pub const withGroupCumLastFalseIndex = lazy_relation_methods_mod.withGroupCumLastFalseIndex;
            pub const withGroupCumLastFalseIndexOn = lazy_relation_methods_mod.withGroupCumLastFalseIndexOn;
            pub const withGroupCumulativeSum = lazy_relation_methods_mod.withGroupCumulativeSum;
            pub const withGroupCumulativeSumOn = lazy_relation_methods_mod.withGroupCumulativeSumOn;
            pub const withGroupCumSum = lazy_relation_methods_mod.withGroupCumSum;
            pub const withGroupCumSumOn = lazy_relation_methods_mod.withGroupCumSumOn;
            pub const withGroupCumulativeMean = lazy_relation_methods_mod.withGroupCumulativeMean;
            pub const withGroupCumulativeMeanOn = lazy_relation_methods_mod.withGroupCumulativeMeanOn;
            pub const withGroupCumMean = lazy_relation_methods_mod.withGroupCumMean;
            pub const withGroupCumMeanOn = lazy_relation_methods_mod.withGroupCumMeanOn;
            pub const withGroupCumulativeWeightedMean = lazy_relation_methods_mod.withGroupCumulativeWeightedMean;
            pub const withGroupCumulativeWeightedMeanOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanOn;
            pub const withGroupCumulativeWeightedSum = lazy_relation_methods_mod.withGroupCumulativeWeightedSum;
            pub const withGroupCumulativeWeightedSumOn = lazy_relation_methods_mod.withGroupCumulativeWeightedSumOn;
            pub const withGroupCumulativeWeightedProduct = lazy_relation_methods_mod.withGroupCumulativeWeightedProduct;
            pub const withGroupCumulativeWeightedProductOn = lazy_relation_methods_mod.withGroupCumulativeWeightedProductOn;
            pub const withGroupCumulativeWeightedWeightSum = lazy_relation_methods_mod.withGroupCumulativeWeightedWeightSum;
            pub const withGroupCumulativeWeightedWeightSumOn = lazy_relation_methods_mod.withGroupCumulativeWeightedWeightSumOn;
            pub const withGroupCumulativeWeightedPositiveCount = lazy_relation_methods_mod.withGroupCumulativeWeightedPositiveCount;
            pub const withGroupCumulativeWeightedPositiveCountOn = lazy_relation_methods_mod.withGroupCumulativeWeightedPositiveCountOn;
            pub const withGroupCumulativeWeightedEffectiveN = lazy_relation_methods_mod.withGroupCumulativeWeightedEffectiveN;
            pub const withGroupCumulativeWeightedEffectiveNOn = lazy_relation_methods_mod.withGroupCumulativeWeightedEffectiveNOn;
            pub const withGroupCumulativeWeightedEffectiveCount = lazy_relation_methods_mod.withGroupCumulativeWeightedEffectiveCount;
            pub const withGroupCumulativeWeightedEffectiveCountOn = lazy_relation_methods_mod.withGroupCumulativeWeightedEffectiveCountOn;
            pub const withGroupCumulativeWeightedTrimmedMean = lazy_relation_methods_mod.withGroupCumulativeWeightedTrimmedMean;
            pub const withGroupCumulativeWeightedTrimmedMeanOn = lazy_relation_methods_mod.withGroupCumulativeWeightedTrimmedMeanOn;
            pub const withGroupCumulativeWeightedWinsorizedMean = lazy_relation_methods_mod.withGroupCumulativeWeightedWinsorizedMean;
            pub const withGroupCumulativeWeightedWinsorizedMeanOn = lazy_relation_methods_mod.withGroupCumulativeWeightedWinsorizedMeanOn;
            pub const withGroupCumulativeWeightedProd = lazy_relation_methods_mod.withGroupCumulativeWeightedProd;
            pub const withGroupCumulativeWeightedProdOn = lazy_relation_methods_mod.withGroupCumulativeWeightedProdOn;
            pub const withGroupCumWeightedMean = lazy_relation_methods_mod.withGroupCumWeightedMean;
            pub const withGroupCumWeightedMeanOn = lazy_relation_methods_mod.withGroupCumWeightedMeanOn;
            pub const withGroupCumWeightedSum = lazy_relation_methods_mod.withGroupCumWeightedSum;
            pub const withGroupCumWeightedSumOn = lazy_relation_methods_mod.withGroupCumWeightedSumOn;
            pub const withGroupCumWeightedProduct = lazy_relation_methods_mod.withGroupCumWeightedProduct;
            pub const withGroupCumWeightedProductOn = lazy_relation_methods_mod.withGroupCumWeightedProductOn;
            pub const withGroupCumWeightedWeightSum = lazy_relation_methods_mod.withGroupCumWeightedWeightSum;
            pub const withGroupCumWeightedWeightSumOn = lazy_relation_methods_mod.withGroupCumWeightedWeightSumOn;
            pub const withGroupCumWeightedPositiveCount = lazy_relation_methods_mod.withGroupCumWeightedPositiveCount;
            pub const withGroupCumWeightedPositiveCountOn = lazy_relation_methods_mod.withGroupCumWeightedPositiveCountOn;
            pub const withGroupCumWeightedEffectiveN = lazy_relation_methods_mod.withGroupCumWeightedEffectiveN;
            pub const withGroupCumWeightedEffectiveNOn = lazy_relation_methods_mod.withGroupCumWeightedEffectiveNOn;
            pub const withGroupCumWeightedEffectiveCount = lazy_relation_methods_mod.withGroupCumWeightedEffectiveCount;
            pub const withGroupCumWeightedEffectiveCountOn = lazy_relation_methods_mod.withGroupCumWeightedEffectiveCountOn;
            pub const withGroupCumWeightedTrimmedMean = lazy_relation_methods_mod.withGroupCumWeightedTrimmedMean;
            pub const withGroupCumWeightedTrimmedMeanOn = lazy_relation_methods_mod.withGroupCumWeightedTrimmedMeanOn;
            pub const withGroupCumWeightedWinsorizedMean = lazy_relation_methods_mod.withGroupCumWeightedWinsorizedMean;
            pub const withGroupCumWeightedWinsorizedMeanOn = lazy_relation_methods_mod.withGroupCumWeightedWinsorizedMeanOn;
            pub const withGroupCumulativeWeightedInterdecileRange = lazy_relation_methods_mod.withGroupCumulativeWeightedInterdecileRange;
            pub const withGroupCumulativeWeightedInterdecileRangeOn = lazy_relation_methods_mod.withGroupCumulativeWeightedInterdecileRangeOn;
            pub const withGroupCumulativeWeightedIdr = lazy_relation_methods_mod.withGroupCumulativeWeightedIdr;
            pub const withGroupCumulativeWeightedIdrOn = lazy_relation_methods_mod.withGroupCumulativeWeightedIdrOn;
            pub const withGroupCumulativeWeightedIDR = lazy_relation_methods_mod.withGroupCumulativeWeightedIDR;
            pub const withGroupCumulativeWeightedIDROn = lazy_relation_methods_mod.withGroupCumulativeWeightedIDROn;
            pub const withGroupCumulativeWeightedMidhinge = lazy_relation_methods_mod.withGroupCumulativeWeightedMidhinge;
            pub const withGroupCumulativeWeightedMidhingeOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMidhingeOn;
            pub const withGroupCumulativeWeightedTrimean = lazy_relation_methods_mod.withGroupCumulativeWeightedTrimean;
            pub const withGroupCumulativeWeightedTrimeanOn = lazy_relation_methods_mod.withGroupCumulativeWeightedTrimeanOn;
            pub const withGroupCumulativeWeightedBowleySkewness = lazy_relation_methods_mod.withGroupCumulativeWeightedBowleySkewness;
            pub const withGroupCumulativeWeightedBowleySkewnessOn = lazy_relation_methods_mod.withGroupCumulativeWeightedBowleySkewnessOn;
            pub const withGroupCumulativeWeightedBowleySkew = lazy_relation_methods_mod.withGroupCumulativeWeightedBowleySkew;
            pub const withGroupCumulativeWeightedBowleySkewOn = lazy_relation_methods_mod.withGroupCumulativeWeightedBowleySkewOn;
            pub const withGroupCumulativeWeightedQuartileCoeffDispersion = lazy_relation_methods_mod.withGroupCumulativeWeightedQuartileCoeffDispersion;
            pub const withGroupCumulativeWeightedQuartileCoeffDispersionOn = lazy_relation_methods_mod.withGroupCumulativeWeightedQuartileCoeffDispersionOn;
            pub const withGroupCumulativeWeightedQcd = lazy_relation_methods_mod.withGroupCumulativeWeightedQcd;
            pub const withGroupCumulativeWeightedQcdOn = lazy_relation_methods_mod.withGroupCumulativeWeightedQcdOn;
            pub const withGroupCumulativeWeightedQCD = lazy_relation_methods_mod.withGroupCumulativeWeightedQCD;
            pub const withGroupCumulativeWeightedQCDOn = lazy_relation_methods_mod.withGroupCumulativeWeightedQCDOn;
            pub const withGroupCumulativeWeightedKelleySkewness = lazy_relation_methods_mod.withGroupCumulativeWeightedKelleySkewness;
            pub const withGroupCumulativeWeightedKelleySkewnessOn = lazy_relation_methods_mod.withGroupCumulativeWeightedKelleySkewnessOn;
            pub const withGroupCumulativeWeightedKelleySkew = lazy_relation_methods_mod.withGroupCumulativeWeightedKelleySkew;
            pub const withGroupCumulativeWeightedKelleySkewOn = lazy_relation_methods_mod.withGroupCumulativeWeightedKelleySkewOn;
            pub const withGroupCumWeightedIdr = lazy_relation_methods_mod.withGroupCumWeightedIdr;
            pub const withGroupCumWeightedIdrOn = lazy_relation_methods_mod.withGroupCumWeightedIdrOn;
            pub const withGroupCumWeightedIDR = lazy_relation_methods_mod.withGroupCumWeightedIDR;
            pub const withGroupCumWeightedIDROn = lazy_relation_methods_mod.withGroupCumWeightedIDROn;
            pub const withGroupCumWeightedMidhinge = lazy_relation_methods_mod.withGroupCumWeightedMidhinge;
            pub const withGroupCumWeightedMidhingeOn = lazy_relation_methods_mod.withGroupCumWeightedMidhingeOn;
            pub const withGroupCumWeightedTrimean = lazy_relation_methods_mod.withGroupCumWeightedTrimean;
            pub const withGroupCumWeightedTrimeanOn = lazy_relation_methods_mod.withGroupCumWeightedTrimeanOn;
            pub const withGroupCumWeightedBowleySkewness = lazy_relation_methods_mod.withGroupCumWeightedBowleySkewness;
            pub const withGroupCumWeightedBowleySkewnessOn = lazy_relation_methods_mod.withGroupCumWeightedBowleySkewnessOn;
            pub const withGroupCumWeightedBowleySkew = lazy_relation_methods_mod.withGroupCumWeightedBowleySkew;
            pub const withGroupCumWeightedBowleySkewOn = lazy_relation_methods_mod.withGroupCumWeightedBowleySkewOn;
            pub const withGroupCumWeightedQuartileCoeffDispersion = lazy_relation_methods_mod.withGroupCumWeightedQuartileCoeffDispersion;
            pub const withGroupCumWeightedQuartileCoeffDispersionOn = lazy_relation_methods_mod.withGroupCumWeightedQuartileCoeffDispersionOn;
            pub const withGroupCumWeightedQcd = lazy_relation_methods_mod.withGroupCumWeightedQcd;
            pub const withGroupCumWeightedQcdOn = lazy_relation_methods_mod.withGroupCumWeightedQcdOn;
            pub const withGroupCumWeightedQCD = lazy_relation_methods_mod.withGroupCumWeightedQCD;
            pub const withGroupCumWeightedQCDOn = lazy_relation_methods_mod.withGroupCumWeightedQCDOn;
            pub const withGroupCumWeightedKelleySkewness = lazy_relation_methods_mod.withGroupCumWeightedKelleySkewness;
            pub const withGroupCumWeightedKelleySkewnessOn = lazy_relation_methods_mod.withGroupCumWeightedKelleySkewnessOn;
            pub const withGroupCumWeightedKelleySkew = lazy_relation_methods_mod.withGroupCumWeightedKelleySkew;
            pub const withGroupCumWeightedKelleySkewOn = lazy_relation_methods_mod.withGroupCumWeightedKelleySkewOn;
            pub const withGroupCumWeightedProd = lazy_relation_methods_mod.withGroupCumWeightedProd;
            pub const withGroupCumWeightedProdOn = lazy_relation_methods_mod.withGroupCumWeightedProdOn;
            pub const withGroupCumulativeWeightedMedian = lazy_relation_methods_mod.withGroupCumulativeWeightedMedian;
            pub const withGroupCumulativeWeightedMedianOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMedianOn;
            pub const withGroupCumulativeWeightedQuantile = lazy_relation_methods_mod.withGroupCumulativeWeightedQuantile;
            pub const withGroupCumulativeWeightedQuantileOn = lazy_relation_methods_mod.withGroupCumulativeWeightedQuantileOn;
            pub const withGroupCumWeightedMedian = lazy_relation_methods_mod.withGroupCumWeightedMedian;
            pub const withGroupCumWeightedMedianOn = lazy_relation_methods_mod.withGroupCumWeightedMedianOn;
            pub const withGroupCumWeightedQuantile = lazy_relation_methods_mod.withGroupCumWeightedQuantile;
            pub const withGroupCumWeightedQuantileOn = lazy_relation_methods_mod.withGroupCumWeightedQuantileOn;
            pub const withGroupCumulativeWeightedIqr = lazy_relation_methods_mod.withGroupCumulativeWeightedIqr;
            pub const withGroupCumulativeWeightedIqrOn = lazy_relation_methods_mod.withGroupCumulativeWeightedIqrOn;
            pub const withGroupCumulativeWeightedIQR = lazy_relation_methods_mod.withGroupCumulativeWeightedIQR;
            pub const withGroupCumulativeWeightedIQROn = lazy_relation_methods_mod.withGroupCumulativeWeightedIQROn;
            pub const withGroupCumulativeWeightedMad = lazy_relation_methods_mod.withGroupCumulativeWeightedMad;
            pub const withGroupCumulativeWeightedMadOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMadOn;
            pub const withGroupCumulativeWeightedMAD = lazy_relation_methods_mod.withGroupCumulativeWeightedMAD;
            pub const withGroupCumulativeWeightedMADOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMADOn;
            pub const withGroupCumulativeWeightedMedianAbsDev = lazy_relation_methods_mod.withGroupCumulativeWeightedMedianAbsDev;
            pub const withGroupCumulativeWeightedMedianAbsDevOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMedianAbsDevOn;
            pub const withGroupCumWeightedIqr = lazy_relation_methods_mod.withGroupCumWeightedIqr;
            pub const withGroupCumWeightedIqrOn = lazy_relation_methods_mod.withGroupCumWeightedIqrOn;
            pub const withGroupCumWeightedIQR = lazy_relation_methods_mod.withGroupCumWeightedIQR;
            pub const withGroupCumWeightedIQROn = lazy_relation_methods_mod.withGroupCumWeightedIQROn;
            pub const withGroupCumWeightedMad = lazy_relation_methods_mod.withGroupCumWeightedMad;
            pub const withGroupCumWeightedMadOn = lazy_relation_methods_mod.withGroupCumWeightedMadOn;
            pub const withGroupCumWeightedMAD = lazy_relation_methods_mod.withGroupCumWeightedMAD;
            pub const withGroupCumWeightedMADOn = lazy_relation_methods_mod.withGroupCumWeightedMADOn;
            pub const withGroupCumWeightedMedianAbsDev = lazy_relation_methods_mod.withGroupCumWeightedMedianAbsDev;
            pub const withGroupCumWeightedMedianAbsDevOn = lazy_relation_methods_mod.withGroupCumWeightedMedianAbsDevOn;
            pub const withGroupCumulativeWeightedMode = lazy_relation_methods_mod.withGroupCumulativeWeightedMode;
            pub const withGroupCumulativeWeightedModeOn = lazy_relation_methods_mod.withGroupCumulativeWeightedModeOn;
            pub const withGroupCumulativeWeightedModeWeight = lazy_relation_methods_mod.withGroupCumulativeWeightedModeWeight;
            pub const withGroupCumulativeWeightedModeWeightOn = lazy_relation_methods_mod.withGroupCumulativeWeightedModeWeightOn;
            pub const withGroupCumulativeWeightedModeRatio = lazy_relation_methods_mod.withGroupCumulativeWeightedModeRatio;
            pub const withGroupCumulativeWeightedModeRatioOn = lazy_relation_methods_mod.withGroupCumulativeWeightedModeRatioOn;
            pub const withGroupCumulativeWeightedModeMargin = lazy_relation_methods_mod.withGroupCumulativeWeightedModeMargin;
            pub const withGroupCumulativeWeightedModeMarginOn = lazy_relation_methods_mod.withGroupCumulativeWeightedModeMarginOn;
            pub const withGroupCumulativeWeightedModeMarginRatio = lazy_relation_methods_mod.withGroupCumulativeWeightedModeMarginRatio;
            pub const withGroupCumulativeWeightedModeMarginRatioOn = lazy_relation_methods_mod.withGroupCumulativeWeightedModeMarginRatioOn;
            pub const withGroupCumWeightedMode = lazy_relation_methods_mod.withGroupCumWeightedMode;
            pub const withGroupCumWeightedModeOn = lazy_relation_methods_mod.withGroupCumWeightedModeOn;
            pub const withGroupCumWeightedModeWeight = lazy_relation_methods_mod.withGroupCumWeightedModeWeight;
            pub const withGroupCumWeightedModeWeightOn = lazy_relation_methods_mod.withGroupCumWeightedModeWeightOn;
            pub const withGroupCumWeightedModeRatio = lazy_relation_methods_mod.withGroupCumWeightedModeRatio;
            pub const withGroupCumWeightedModeRatioOn = lazy_relation_methods_mod.withGroupCumWeightedModeRatioOn;
            pub const withGroupCumWeightedModeMargin = lazy_relation_methods_mod.withGroupCumWeightedModeMargin;
            pub const withGroupCumWeightedModeMarginOn = lazy_relation_methods_mod.withGroupCumWeightedModeMarginOn;
            pub const withGroupCumWeightedModeMarginRatio = lazy_relation_methods_mod.withGroupCumWeightedModeMarginRatio;
            pub const withGroupCumWeightedModeMarginRatioOn = lazy_relation_methods_mod.withGroupCumWeightedModeMarginRatioOn;
            pub const withGroupCumulativeWeightedEntropy = lazy_relation_methods_mod.withGroupCumulativeWeightedEntropy;
            pub const withGroupCumulativeWeightedEntropyOn = lazy_relation_methods_mod.withGroupCumulativeWeightedEntropyOn;
            pub const withGroupCumulativeWeightedGiniImpurity = lazy_relation_methods_mod.withGroupCumulativeWeightedGiniImpurity;
            pub const withGroupCumulativeWeightedGiniImpurityOn = lazy_relation_methods_mod.withGroupCumulativeWeightedGiniImpurityOn;
            pub const withGroupCumulativeWeightedGini = lazy_relation_methods_mod.withGroupCumulativeWeightedGini;
            pub const withGroupCumulativeWeightedGiniOn = lazy_relation_methods_mod.withGroupCumulativeWeightedGiniOn;
            pub const withGroupCumulativeWeightedPerplexity = lazy_relation_methods_mod.withGroupCumulativeWeightedPerplexity;
            pub const withGroupCumulativeWeightedPerplexityOn = lazy_relation_methods_mod.withGroupCumulativeWeightedPerplexityOn;
            pub const withGroupCumulativeWeightedInverseSimpson = lazy_relation_methods_mod.withGroupCumulativeWeightedInverseSimpson;
            pub const withGroupCumulativeWeightedInverseSimpsonOn = lazy_relation_methods_mod.withGroupCumulativeWeightedInverseSimpsonOn;
            pub const withGroupCumulativeWeightedSimpsonConcentration = lazy_relation_methods_mod.withGroupCumulativeWeightedSimpsonConcentration;
            pub const withGroupCumulativeWeightedSimpsonConcentrationOn = lazy_relation_methods_mod.withGroupCumulativeWeightedSimpsonConcentrationOn;
            pub const withGroupCumulativeWeightedConcentration = lazy_relation_methods_mod.withGroupCumulativeWeightedConcentration;
            pub const withGroupCumulativeWeightedConcentrationOn = lazy_relation_methods_mod.withGroupCumulativeWeightedConcentrationOn;
            pub const withGroupCumulativeWeightedEvenness = lazy_relation_methods_mod.withGroupCumulativeWeightedEvenness;
            pub const withGroupCumulativeWeightedEvennessOn = lazy_relation_methods_mod.withGroupCumulativeWeightedEvennessOn;
            pub const withGroupCumulativeWeightedMeanAbsDev = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanAbsDev;
            pub const withGroupCumulativeWeightedMeanAbsDevOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanAbsDevOn;
            pub const withGroupCumulativeWeightedMeanAbsDevRatio = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanAbsDevRatio;
            pub const withGroupCumulativeWeightedMeanAbsDevRatioOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanAbsDevRatioOn;
            pub const withGroupCumulativeWeightedMeanAbsoluteDeviation = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanAbsoluteDeviation;
            pub const withGroupCumulativeWeightedMeanAbsoluteDeviationOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanAbsoluteDeviationOn;
            pub const withGroupCumulativeWeightedGiniMeanDiff = lazy_relation_methods_mod.withGroupCumulativeWeightedGiniMeanDiff;
            pub const withGroupCumulativeWeightedGiniMeanDiffOn = lazy_relation_methods_mod.withGroupCumulativeWeightedGiniMeanDiffOn;
            pub const withGroupCumulativeWeightedGiniCoefficient = lazy_relation_methods_mod.withGroupCumulativeWeightedGiniCoefficient;
            pub const withGroupCumulativeWeightedGiniCoefficientOn = lazy_relation_methods_mod.withGroupCumulativeWeightedGiniCoefficientOn;
            pub const withGroupCumulativeWeightedGiniCoeff = lazy_relation_methods_mod.withGroupCumulativeWeightedGiniCoeff;
            pub const withGroupCumulativeWeightedGiniCoeffOn = lazy_relation_methods_mod.withGroupCumulativeWeightedGiniCoeffOn;
            pub const withGroupCumWeightedEntropy = lazy_relation_methods_mod.withGroupCumWeightedEntropy;
            pub const withGroupCumWeightedEntropyOn = lazy_relation_methods_mod.withGroupCumWeightedEntropyOn;
            pub const withGroupCumWeightedGiniImpurity = lazy_relation_methods_mod.withGroupCumWeightedGiniImpurity;
            pub const withGroupCumWeightedGiniImpurityOn = lazy_relation_methods_mod.withGroupCumWeightedGiniImpurityOn;
            pub const withGroupCumWeightedGini = lazy_relation_methods_mod.withGroupCumWeightedGini;
            pub const withGroupCumWeightedGiniOn = lazy_relation_methods_mod.withGroupCumWeightedGiniOn;
            pub const withGroupCumWeightedPerplexity = lazy_relation_methods_mod.withGroupCumWeightedPerplexity;
            pub const withGroupCumWeightedPerplexityOn = lazy_relation_methods_mod.withGroupCumWeightedPerplexityOn;
            pub const withGroupCumWeightedInverseSimpson = lazy_relation_methods_mod.withGroupCumWeightedInverseSimpson;
            pub const withGroupCumWeightedInverseSimpsonOn = lazy_relation_methods_mod.withGroupCumWeightedInverseSimpsonOn;
            pub const withGroupCumWeightedSimpsonConcentration = lazy_relation_methods_mod.withGroupCumWeightedSimpsonConcentration;
            pub const withGroupCumWeightedSimpsonConcentrationOn = lazy_relation_methods_mod.withGroupCumWeightedSimpsonConcentrationOn;
            pub const withGroupCumWeightedConcentration = lazy_relation_methods_mod.withGroupCumWeightedConcentration;
            pub const withGroupCumWeightedConcentrationOn = lazy_relation_methods_mod.withGroupCumWeightedConcentrationOn;
            pub const withGroupCumWeightedEvenness = lazy_relation_methods_mod.withGroupCumWeightedEvenness;
            pub const withGroupCumWeightedEvennessOn = lazy_relation_methods_mod.withGroupCumWeightedEvennessOn;
            pub const withGroupCumWeightedMeanAbsDev = lazy_relation_methods_mod.withGroupCumWeightedMeanAbsDev;
            pub const withGroupCumWeightedMeanAbsDevOn = lazy_relation_methods_mod.withGroupCumWeightedMeanAbsDevOn;
            pub const withGroupCumWeightedMeanAbsDevRatio = lazy_relation_methods_mod.withGroupCumWeightedMeanAbsDevRatio;
            pub const withGroupCumWeightedMeanAbsDevRatioOn = lazy_relation_methods_mod.withGroupCumWeightedMeanAbsDevRatioOn;
            pub const withGroupCumWeightedMeanAbsoluteDeviation = lazy_relation_methods_mod.withGroupCumWeightedMeanAbsoluteDeviation;
            pub const withGroupCumWeightedMeanAbsoluteDeviationOn = lazy_relation_methods_mod.withGroupCumWeightedMeanAbsoluteDeviationOn;
            pub const withGroupCumWeightedGiniMeanDiff = lazy_relation_methods_mod.withGroupCumWeightedGiniMeanDiff;
            pub const withGroupCumWeightedGiniMeanDiffOn = lazy_relation_methods_mod.withGroupCumWeightedGiniMeanDiffOn;
            pub const withGroupCumWeightedGiniCoefficient = lazy_relation_methods_mod.withGroupCumWeightedGiniCoefficient;
            pub const withGroupCumWeightedGiniCoefficientOn = lazy_relation_methods_mod.withGroupCumWeightedGiniCoefficientOn;
            pub const withGroupCumWeightedGiniCoeff = lazy_relation_methods_mod.withGroupCumWeightedGiniCoeff;
            pub const withGroupCumWeightedGiniCoeffOn = lazy_relation_methods_mod.withGroupCumWeightedGiniCoeffOn;
            pub const withGroupCumulativeWeightedDot = lazy_relation_methods_mod.withGroupCumulativeWeightedDot;
            pub const withGroupCumulativeWeightedDotOn = lazy_relation_methods_mod.withGroupCumulativeWeightedDotOn;
            pub const withGroupCumWeightedDot = lazy_relation_methods_mod.withGroupCumWeightedDot;
            pub const withGroupCumWeightedDotOn = lazy_relation_methods_mod.withGroupCumWeightedDotOn;
            pub const withGroupCumulativeWeightedCosineSimilarity = lazy_relation_methods_mod.withGroupCumulativeWeightedCosineSimilarity;
            pub const withGroupCumulativeWeightedCosineSimilarityOn = lazy_relation_methods_mod.withGroupCumulativeWeightedCosineSimilarityOn;
            pub const withGroupCumWeightedCosineSimilarity = lazy_relation_methods_mod.withGroupCumWeightedCosineSimilarity;
            pub const withGroupCumWeightedCosineSimilarityOn = lazy_relation_methods_mod.withGroupCumWeightedCosineSimilarityOn;
            pub const withGroupCumulativeWeightedSquaredEuclideanDistance = lazy_relation_methods_mod.withGroupCumulativeWeightedSquaredEuclideanDistance;
            pub const withGroupCumulativeWeightedSquaredEuclideanDistanceOn = lazy_relation_methods_mod.withGroupCumulativeWeightedSquaredEuclideanDistanceOn;
            pub const withGroupCumWeightedSquaredEuclideanDistance = lazy_relation_methods_mod.withGroupCumWeightedSquaredEuclideanDistance;
            pub const withGroupCumWeightedSquaredEuclideanDistanceOn = lazy_relation_methods_mod.withGroupCumWeightedSquaredEuclideanDistanceOn;
            pub const withGroupCumulativeWeightedEuclideanDistance = lazy_relation_methods_mod.withGroupCumulativeWeightedEuclideanDistance;
            pub const withGroupCumulativeWeightedEuclideanDistanceOn = lazy_relation_methods_mod.withGroupCumulativeWeightedEuclideanDistanceOn;
            pub const withGroupCumWeightedEuclideanDistance = lazy_relation_methods_mod.withGroupCumWeightedEuclideanDistance;
            pub const withGroupCumWeightedEuclideanDistanceOn = lazy_relation_methods_mod.withGroupCumWeightedEuclideanDistanceOn;
            pub const withGroupCumulativeWeightedManhattanDistance = lazy_relation_methods_mod.withGroupCumulativeWeightedManhattanDistance;
            pub const withGroupCumulativeWeightedManhattanDistanceOn = lazy_relation_methods_mod.withGroupCumulativeWeightedManhattanDistanceOn;
            pub const withGroupCumWeightedManhattanDistance = lazy_relation_methods_mod.withGroupCumWeightedManhattanDistance;
            pub const withGroupCumWeightedManhattanDistanceOn = lazy_relation_methods_mod.withGroupCumWeightedManhattanDistanceOn;
            pub const withGroupCumulativeWeightedChebyshevDistance = lazy_relation_methods_mod.withGroupCumulativeWeightedChebyshevDistance;
            pub const withGroupCumulativeWeightedChebyshevDistanceOn = lazy_relation_methods_mod.withGroupCumulativeWeightedChebyshevDistanceOn;
            pub const withGroupCumWeightedChebyshevDistance = lazy_relation_methods_mod.withGroupCumWeightedChebyshevDistance;
            pub const withGroupCumWeightedChebyshevDistanceOn = lazy_relation_methods_mod.withGroupCumWeightedChebyshevDistanceOn;
            pub const withGroupCumulativeWeightedCanberraDistance = lazy_relation_methods_mod.withGroupCumulativeWeightedCanberraDistance;
            pub const withGroupCumulativeWeightedCanberraDistanceOn = lazy_relation_methods_mod.withGroupCumulativeWeightedCanberraDistanceOn;
            pub const withGroupCumWeightedCanberraDistance = lazy_relation_methods_mod.withGroupCumWeightedCanberraDistance;
            pub const withGroupCumWeightedCanberraDistanceOn = lazy_relation_methods_mod.withGroupCumWeightedCanberraDistanceOn;
            pub const withGroupCumulativeWeightedBrayCurtisDistance = lazy_relation_methods_mod.withGroupCumulativeWeightedBrayCurtisDistance;
            pub const withGroupCumulativeWeightedBrayCurtisDistanceOn = lazy_relation_methods_mod.withGroupCumulativeWeightedBrayCurtisDistanceOn;
            pub const withGroupCumWeightedBrayCurtisDistance = lazy_relation_methods_mod.withGroupCumWeightedBrayCurtisDistance;
            pub const withGroupCumWeightedBrayCurtisDistanceOn = lazy_relation_methods_mod.withGroupCumWeightedBrayCurtisDistanceOn;
            pub const withGroupCumulativeWeightedMeanError = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanError;
            pub const withGroupCumulativeWeightedMeanErrorOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanErrorOn;
            pub const withGroupCumWeightedMeanError = lazy_relation_methods_mod.withGroupCumWeightedMeanError;
            pub const withGroupCumWeightedMeanErrorOn = lazy_relation_methods_mod.withGroupCumWeightedMeanErrorOn;
            pub const withGroupCumulativeWeightedMae = lazy_relation_methods_mod.withGroupCumulativeWeightedMae;
            pub const withGroupCumulativeWeightedMaeOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMaeOn;
            pub const withGroupCumWeightedMae = lazy_relation_methods_mod.withGroupCumWeightedMae;
            pub const withGroupCumWeightedMaeOn = lazy_relation_methods_mod.withGroupCumWeightedMaeOn;
            pub const withGroupCumulativeWeightedMse = lazy_relation_methods_mod.withGroupCumulativeWeightedMse;
            pub const withGroupCumulativeWeightedMseOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMseOn;
            pub const withGroupCumWeightedMse = lazy_relation_methods_mod.withGroupCumWeightedMse;
            pub const withGroupCumWeightedMseOn = lazy_relation_methods_mod.withGroupCumWeightedMseOn;
            pub const withGroupCumulativeWeightedRmse = lazy_relation_methods_mod.withGroupCumulativeWeightedRmse;
            pub const withGroupCumulativeWeightedRmseOn = lazy_relation_methods_mod.withGroupCumulativeWeightedRmseOn;
            pub const withGroupCumWeightedRmse = lazy_relation_methods_mod.withGroupCumWeightedRmse;
            pub const withGroupCumWeightedRmseOn = lazy_relation_methods_mod.withGroupCumWeightedRmseOn;
            pub const withGroupCumulativeWeightedMape = lazy_relation_methods_mod.withGroupCumulativeWeightedMape;
            pub const withGroupCumulativeWeightedMapeOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMapeOn;
            pub const withGroupCumWeightedMape = lazy_relation_methods_mod.withGroupCumWeightedMape;
            pub const withGroupCumWeightedMapeOn = lazy_relation_methods_mod.withGroupCumWeightedMapeOn;
            pub const withGroupCumulativeWeightedSmape = lazy_relation_methods_mod.withGroupCumulativeWeightedSmape;
            pub const withGroupCumulativeWeightedSmapeOn = lazy_relation_methods_mod.withGroupCumulativeWeightedSmapeOn;
            pub const withGroupCumWeightedSmape = lazy_relation_methods_mod.withGroupCumWeightedSmape;
            pub const withGroupCumWeightedSmapeOn = lazy_relation_methods_mod.withGroupCumWeightedSmapeOn;
            pub const withGroupCumulativeWeightedCosine = lazy_relation_methods_mod.withGroupCumulativeWeightedCosine;
            pub const withGroupCumulativeWeightedCosineOn = lazy_relation_methods_mod.withGroupCumulativeWeightedCosineOn;
            pub const withGroupCumulativeWeightedBias = lazy_relation_methods_mod.withGroupCumulativeWeightedBias;
            pub const withGroupCumulativeWeightedBiasOn = lazy_relation_methods_mod.withGroupCumulativeWeightedBiasOn;
            pub const withGroupCumWeightedCosine = lazy_relation_methods_mod.withGroupCumWeightedCosine;
            pub const withGroupCumWeightedCosineOn = lazy_relation_methods_mod.withGroupCumWeightedCosineOn;
            pub const withGroupCumWeightedBias = lazy_relation_methods_mod.withGroupCumWeightedBias;
            pub const withGroupCumWeightedBiasOn = lazy_relation_methods_mod.withGroupCumWeightedBiasOn;
            pub const withGroupCumulativeWeightedCovariance = lazy_relation_methods_mod.withGroupCumulativeWeightedCovariance;
            pub const withGroupCumulativeWeightedCovarianceOn = lazy_relation_methods_mod.withGroupCumulativeWeightedCovarianceOn;
            pub const withGroupCumulativeWeightedCov = lazy_relation_methods_mod.withGroupCumulativeWeightedCov;
            pub const withGroupCumulativeWeightedCovOn = lazy_relation_methods_mod.withGroupCumulativeWeightedCovOn;
            pub const withGroupCumulativeWeightedCorrelation = lazy_relation_methods_mod.withGroupCumulativeWeightedCorrelation;
            pub const withGroupCumulativeWeightedCorrelationOn = lazy_relation_methods_mod.withGroupCumulativeWeightedCorrelationOn;
            pub const withGroupCumulativeWeightedCorr = lazy_relation_methods_mod.withGroupCumulativeWeightedCorr;
            pub const withGroupCumulativeWeightedCorrOn = lazy_relation_methods_mod.withGroupCumulativeWeightedCorrOn;
            pub const withGroupCumulativeWeightedBeta = lazy_relation_methods_mod.withGroupCumulativeWeightedBeta;
            pub const withGroupCumulativeWeightedBetaOn = lazy_relation_methods_mod.withGroupCumulativeWeightedBetaOn;
            pub const withGroupCumWeightedCovariance = lazy_relation_methods_mod.withGroupCumWeightedCovariance;
            pub const withGroupCumWeightedCovarianceOn = lazy_relation_methods_mod.withGroupCumWeightedCovarianceOn;
            pub const withGroupCumWeightedCov = lazy_relation_methods_mod.withGroupCumWeightedCov;
            pub const withGroupCumWeightedCovOn = lazy_relation_methods_mod.withGroupCumWeightedCovOn;
            pub const withGroupCumWeightedCorrelation = lazy_relation_methods_mod.withGroupCumWeightedCorrelation;
            pub const withGroupCumWeightedCorrelationOn = lazy_relation_methods_mod.withGroupCumWeightedCorrelationOn;
            pub const withGroupCumWeightedCorr = lazy_relation_methods_mod.withGroupCumWeightedCorr;
            pub const withGroupCumWeightedCorrOn = lazy_relation_methods_mod.withGroupCumWeightedCorrOn;
            pub const withGroupCumWeightedBeta = lazy_relation_methods_mod.withGroupCumWeightedBeta;
            pub const withGroupCumWeightedBetaOn = lazy_relation_methods_mod.withGroupCumWeightedBetaOn;
            pub const withGroupCumulativeWeightedSem = lazy_relation_methods_mod.withGroupCumulativeWeightedSem;
            pub const withGroupCumulativeWeightedSemOn = lazy_relation_methods_mod.withGroupCumulativeWeightedSemOn;
            pub const withGroupCumWeightedSem = lazy_relation_methods_mod.withGroupCumWeightedSem;
            pub const withGroupCumWeightedSemOn = lazy_relation_methods_mod.withGroupCumWeightedSemOn;
            pub const withGroupCumulativeWeightedCv = lazy_relation_methods_mod.withGroupCumulativeWeightedCv;
            pub const withGroupCumulativeWeightedCvOn = lazy_relation_methods_mod.withGroupCumulativeWeightedCvOn;
            pub const withGroupCumWeightedCv = lazy_relation_methods_mod.withGroupCumWeightedCv;
            pub const withGroupCumWeightedCvOn = lazy_relation_methods_mod.withGroupCumWeightedCvOn;
            pub const withGroupCumulativeWeightedFano = lazy_relation_methods_mod.withGroupCumulativeWeightedFano;
            pub const withGroupCumulativeWeightedFanoOn = lazy_relation_methods_mod.withGroupCumulativeWeightedFanoOn;
            pub const withGroupCumulativeWeightedSkewness = lazy_relation_methods_mod.withGroupCumulativeWeightedSkewness;
            pub const withGroupCumulativeWeightedSkewnessOn = lazy_relation_methods_mod.withGroupCumulativeWeightedSkewnessOn;
            pub const withGroupCumulativeWeightedSkew = lazy_relation_methods_mod.withGroupCumulativeWeightedSkew;
            pub const withGroupCumulativeWeightedSkewOn = lazy_relation_methods_mod.withGroupCumulativeWeightedSkewOn;
            pub const withGroupCumulativeWeightedKurtosis = lazy_relation_methods_mod.withGroupCumulativeWeightedKurtosis;
            pub const withGroupCumulativeWeightedKurtosisOn = lazy_relation_methods_mod.withGroupCumulativeWeightedKurtosisOn;
            pub const withGroupCumulativeWeightedKurt = lazy_relation_methods_mod.withGroupCumulativeWeightedKurt;
            pub const withGroupCumulativeWeightedKurtOn = lazy_relation_methods_mod.withGroupCumulativeWeightedKurtOn;
            pub const withGroupCumWeightedFano = lazy_relation_methods_mod.withGroupCumWeightedFano;
            pub const withGroupCumWeightedFanoOn = lazy_relation_methods_mod.withGroupCumWeightedFanoOn;
            pub const withGroupCumWeightedSkewness = lazy_relation_methods_mod.withGroupCumWeightedSkewness;
            pub const withGroupCumWeightedSkewnessOn = lazy_relation_methods_mod.withGroupCumWeightedSkewnessOn;
            pub const withGroupCumWeightedSkew = lazy_relation_methods_mod.withGroupCumWeightedSkew;
            pub const withGroupCumWeightedSkewOn = lazy_relation_methods_mod.withGroupCumWeightedSkewOn;
            pub const withGroupCumWeightedKurtosis = lazy_relation_methods_mod.withGroupCumWeightedKurtosis;
            pub const withGroupCumWeightedKurtosisOn = lazy_relation_methods_mod.withGroupCumWeightedKurtosisOn;
            pub const withGroupCumWeightedKurt = lazy_relation_methods_mod.withGroupCumWeightedKurt;
            pub const withGroupCumWeightedKurtOn = lazy_relation_methods_mod.withGroupCumWeightedKurtOn;
            pub const withGroupCumulativeWeightedSEM = lazy_relation_methods_mod.withGroupCumulativeWeightedSEM;
            pub const withGroupCumulativeWeightedSEMOn = lazy_relation_methods_mod.withGroupCumulativeWeightedSEMOn;
            pub const withGroupCumulativeWeightedCV = lazy_relation_methods_mod.withGroupCumulativeWeightedCV;
            pub const withGroupCumulativeWeightedCVOn = lazy_relation_methods_mod.withGroupCumulativeWeightedCVOn;
            pub const withGroupCumWeightedSEM = lazy_relation_methods_mod.withGroupCumWeightedSEM;
            pub const withGroupCumWeightedSEMOn = lazy_relation_methods_mod.withGroupCumWeightedSEMOn;
            pub const withGroupCumWeightedCV = lazy_relation_methods_mod.withGroupCumWeightedCV;
            pub const withGroupCumWeightedCVOn = lazy_relation_methods_mod.withGroupCumWeightedCVOn;
            pub const withGroupCumulativeWeightedMeanSquare = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanSquare;
            pub const withGroupCumulativeWeightedMeanSquareOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanSquareOn;
            pub const withGroupCumulativeWeightedRms = lazy_relation_methods_mod.withGroupCumulativeWeightedRms;
            pub const withGroupCumulativeWeightedRmsOn = lazy_relation_methods_mod.withGroupCumulativeWeightedRmsOn;
            pub const withGroupCumulativeWeightedMeanSquared = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanSquared;
            pub const withGroupCumulativeWeightedMeanSquaredOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanSquaredOn;
            pub const withGroupCumulativeWeightedMeanSq = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanSq;
            pub const withGroupCumulativeWeightedMeanSqOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanSqOn;
            pub const withGroupCumulativeWeightedRMS = lazy_relation_methods_mod.withGroupCumulativeWeightedRMS;
            pub const withGroupCumulativeWeightedRMSOn = lazy_relation_methods_mod.withGroupCumulativeWeightedRMSOn;
            pub const withGroupCumWeightedMeanSquare = lazy_relation_methods_mod.withGroupCumWeightedMeanSquare;
            pub const withGroupCumWeightedMeanSquareOn = lazy_relation_methods_mod.withGroupCumWeightedMeanSquareOn;
            pub const withGroupCumWeightedMeanSquared = lazy_relation_methods_mod.withGroupCumWeightedMeanSquared;
            pub const withGroupCumWeightedMeanSquaredOn = lazy_relation_methods_mod.withGroupCumWeightedMeanSquaredOn;
            pub const withGroupCumWeightedMeanSq = lazy_relation_methods_mod.withGroupCumWeightedMeanSq;
            pub const withGroupCumWeightedMeanSqOn = lazy_relation_methods_mod.withGroupCumWeightedMeanSqOn;
            pub const withGroupCumWeightedRms = lazy_relation_methods_mod.withGroupCumWeightedRms;
            pub const withGroupCumWeightedRmsOn = lazy_relation_methods_mod.withGroupCumWeightedRmsOn;
            pub const withGroupCumWeightedRMS = lazy_relation_methods_mod.withGroupCumWeightedRMS;
            pub const withGroupCumWeightedRMSOn = lazy_relation_methods_mod.withGroupCumWeightedRMSOn;
            pub const withGroupCumulativeWeightedMin = lazy_relation_methods_mod.withGroupCumulativeWeightedMin;
            pub const withGroupCumulativeWeightedMinOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMinOn;
            pub const withGroupCumulativeWeightedMinimum = lazy_relation_methods_mod.withGroupCumulativeWeightedMinimum;
            pub const withGroupCumulativeWeightedMinimumOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMinimumOn;
            pub const withGroupCumulativeWeightedMax = lazy_relation_methods_mod.withGroupCumulativeWeightedMax;
            pub const withGroupCumulativeWeightedMaxOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMaxOn;
            pub const withGroupCumulativeWeightedMaximum = lazy_relation_methods_mod.withGroupCumulativeWeightedMaximum;
            pub const withGroupCumulativeWeightedMaximumOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMaximumOn;
            pub const withGroupCumWeightedMin = lazy_relation_methods_mod.withGroupCumWeightedMin;
            pub const withGroupCumWeightedMinOn = lazy_relation_methods_mod.withGroupCumWeightedMinOn;
            pub const withGroupCumWeightedMinimum = lazy_relation_methods_mod.withGroupCumWeightedMinimum;
            pub const withGroupCumWeightedMinimumOn = lazy_relation_methods_mod.withGroupCumWeightedMinimumOn;
            pub const withGroupCumWeightedMax = lazy_relation_methods_mod.withGroupCumWeightedMax;
            pub const withGroupCumWeightedMaxOn = lazy_relation_methods_mod.withGroupCumWeightedMaxOn;
            pub const withGroupCumWeightedMaximum = lazy_relation_methods_mod.withGroupCumWeightedMaximum;
            pub const withGroupCumWeightedMaximumOn = lazy_relation_methods_mod.withGroupCumWeightedMaximumOn;
            pub const withGroupCumulativeWeightedMeanAbs = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanAbs;
            pub const withGroupCumulativeWeightedMeanAbsOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMeanAbsOn;
            pub const withGroupCumulativeWeightedL1Norm = lazy_relation_methods_mod.withGroupCumulativeWeightedL1Norm;
            pub const withGroupCumulativeWeightedL1NormOn = lazy_relation_methods_mod.withGroupCumulativeWeightedL1NormOn;
            pub const withGroupCumulativeWeightedL2Norm = lazy_relation_methods_mod.withGroupCumulativeWeightedL2Norm;
            pub const withGroupCumulativeWeightedL2NormOn = lazy_relation_methods_mod.withGroupCumulativeWeightedL2NormOn;
            pub const withGroupCumulativeWeightedMaxAbs = lazy_relation_methods_mod.withGroupCumulativeWeightedMaxAbs;
            pub const withGroupCumulativeWeightedMaxAbsOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMaxAbsOn;
            pub const withGroupCumulativeWeightedMinAbs = lazy_relation_methods_mod.withGroupCumulativeWeightedMinAbs;
            pub const withGroupCumulativeWeightedMinAbsOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMinAbsOn;
            pub const withGroupCumulativeWeightedL1 = lazy_relation_methods_mod.withGroupCumulativeWeightedL1;
            pub const withGroupCumulativeWeightedL1On = lazy_relation_methods_mod.withGroupCumulativeWeightedL1On;
            pub const withGroupCumulativeWeightedL2 = lazy_relation_methods_mod.withGroupCumulativeWeightedL2;
            pub const withGroupCumulativeWeightedL2On = lazy_relation_methods_mod.withGroupCumulativeWeightedL2On;
            pub const withGroupCumulativeWeightedMaxAbsolute = lazy_relation_methods_mod.withGroupCumulativeWeightedMaxAbsolute;
            pub const withGroupCumulativeWeightedMaxAbsoluteOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMaxAbsoluteOn;
            pub const withGroupCumulativeWeightedMinAbsolute = lazy_relation_methods_mod.withGroupCumulativeWeightedMinAbsolute;
            pub const withGroupCumulativeWeightedMinAbsoluteOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMinAbsoluteOn;
            pub const withGroupCumWeightedMeanAbs = lazy_relation_methods_mod.withGroupCumWeightedMeanAbs;
            pub const withGroupCumWeightedMeanAbsOn = lazy_relation_methods_mod.withGroupCumWeightedMeanAbsOn;
            pub const withGroupCumWeightedL1Norm = lazy_relation_methods_mod.withGroupCumWeightedL1Norm;
            pub const withGroupCumWeightedL1NormOn = lazy_relation_methods_mod.withGroupCumWeightedL1NormOn;
            pub const withGroupCumWeightedL1 = lazy_relation_methods_mod.withGroupCumWeightedL1;
            pub const withGroupCumWeightedL1On = lazy_relation_methods_mod.withGroupCumWeightedL1On;
            pub const withGroupCumWeightedL2Norm = lazy_relation_methods_mod.withGroupCumWeightedL2Norm;
            pub const withGroupCumWeightedL2NormOn = lazy_relation_methods_mod.withGroupCumWeightedL2NormOn;
            pub const withGroupCumWeightedL2 = lazy_relation_methods_mod.withGroupCumWeightedL2;
            pub const withGroupCumWeightedL2On = lazy_relation_methods_mod.withGroupCumWeightedL2On;
            pub const withGroupCumWeightedMaxAbs = lazy_relation_methods_mod.withGroupCumWeightedMaxAbs;
            pub const withGroupCumWeightedMaxAbsOn = lazy_relation_methods_mod.withGroupCumWeightedMaxAbsOn;
            pub const withGroupCumWeightedMinAbs = lazy_relation_methods_mod.withGroupCumWeightedMinAbs;
            pub const withGroupCumWeightedMinAbsOn = lazy_relation_methods_mod.withGroupCumWeightedMinAbsOn;
            pub const withGroupCumulativeWeightedGeometricMean = lazy_relation_methods_mod.withGroupCumulativeWeightedGeometricMean;
            pub const withGroupCumulativeWeightedGeometricMeanOn = lazy_relation_methods_mod.withGroupCumulativeWeightedGeometricMeanOn;
            pub const withGroupCumulativeWeightedGeoMean = lazy_relation_methods_mod.withGroupCumulativeWeightedGeoMean;
            pub const withGroupCumulativeWeightedGeoMeanOn = lazy_relation_methods_mod.withGroupCumulativeWeightedGeoMeanOn;
            pub const withGroupCumulativeWeightedHarmonicMean = lazy_relation_methods_mod.withGroupCumulativeWeightedHarmonicMean;
            pub const withGroupCumulativeWeightedHarmonicMeanOn = lazy_relation_methods_mod.withGroupCumulativeWeightedHarmonicMeanOn;
            pub const withGroupCumulativeWeightedHarmMean = lazy_relation_methods_mod.withGroupCumulativeWeightedHarmMean;
            pub const withGroupCumulativeWeightedHarmMeanOn = lazy_relation_methods_mod.withGroupCumulativeWeightedHarmMeanOn;
            pub const withGroupCumulativeWeightedLogSumExp = lazy_relation_methods_mod.withGroupCumulativeWeightedLogSumExp;
            pub const withGroupCumulativeWeightedLogSumExpOn = lazy_relation_methods_mod.withGroupCumulativeWeightedLogSumExpOn;
            pub const withGroupCumulativeWeightedLogsumexp = lazy_relation_methods_mod.withGroupCumulativeWeightedLogsumexp;
            pub const withGroupCumulativeWeightedLogsumexpOn = lazy_relation_methods_mod.withGroupCumulativeWeightedLogsumexpOn;
            pub const withGroupCumulativeWeightedLogMeanExp = lazy_relation_methods_mod.withGroupCumulativeWeightedLogMeanExp;
            pub const withGroupCumulativeWeightedLogMeanExpOn = lazy_relation_methods_mod.withGroupCumulativeWeightedLogMeanExpOn;
            pub const withGroupCumulativeWeightedLogmeanexp = lazy_relation_methods_mod.withGroupCumulativeWeightedLogmeanexp;
            pub const withGroupCumulativeWeightedLogmeanexpOn = lazy_relation_methods_mod.withGroupCumulativeWeightedLogmeanexpOn;
            pub const withGroupCumulativeWeightedRange = lazy_relation_methods_mod.withGroupCumulativeWeightedRange;
            pub const withGroupCumulativeWeightedRangeOn = lazy_relation_methods_mod.withGroupCumulativeWeightedRangeOn;
            pub const withGroupCumulativeWeightedMidrange = lazy_relation_methods_mod.withGroupCumulativeWeightedMidrange;
            pub const withGroupCumulativeWeightedMidrangeOn = lazy_relation_methods_mod.withGroupCumulativeWeightedMidrangeOn;
            pub const withGroupCumulativeWeightedRangeCoeff = lazy_relation_methods_mod.withGroupCumulativeWeightedRangeCoeff;
            pub const withGroupCumulativeWeightedRangeCoeffOn = lazy_relation_methods_mod.withGroupCumulativeWeightedRangeCoeffOn;
            pub const withGroupCumulativeWeightedRangeCoefficient = lazy_relation_methods_mod.withGroupCumulativeWeightedRangeCoefficient;
            pub const withGroupCumulativeWeightedRangeCoefficientOn = lazy_relation_methods_mod.withGroupCumulativeWeightedRangeCoefficientOn;
            pub const withGroupCumWeightedGeometricMean = lazy_relation_methods_mod.withGroupCumWeightedGeometricMean;
            pub const withGroupCumWeightedGeometricMeanOn = lazy_relation_methods_mod.withGroupCumWeightedGeometricMeanOn;
            pub const withGroupCumWeightedGeoMean = lazy_relation_methods_mod.withGroupCumWeightedGeoMean;
            pub const withGroupCumWeightedGeoMeanOn = lazy_relation_methods_mod.withGroupCumWeightedGeoMeanOn;
            pub const withGroupCumWeightedHarmonicMean = lazy_relation_methods_mod.withGroupCumWeightedHarmonicMean;
            pub const withGroupCumWeightedHarmonicMeanOn = lazy_relation_methods_mod.withGroupCumWeightedHarmonicMeanOn;
            pub const withGroupCumWeightedHarmMean = lazy_relation_methods_mod.withGroupCumWeightedHarmMean;
            pub const withGroupCumWeightedHarmMeanOn = lazy_relation_methods_mod.withGroupCumWeightedHarmMeanOn;
            pub const withGroupCumWeightedLogSumExp = lazy_relation_methods_mod.withGroupCumWeightedLogSumExp;
            pub const withGroupCumWeightedLogSumExpOn = lazy_relation_methods_mod.withGroupCumWeightedLogSumExpOn;
            pub const withGroupCumWeightedLogsumexp = lazy_relation_methods_mod.withGroupCumWeightedLogsumexp;
            pub const withGroupCumWeightedLogsumexpOn = lazy_relation_methods_mod.withGroupCumWeightedLogsumexpOn;
            pub const withGroupCumWeightedLogMeanExp = lazy_relation_methods_mod.withGroupCumWeightedLogMeanExp;
            pub const withGroupCumWeightedLogMeanExpOn = lazy_relation_methods_mod.withGroupCumWeightedLogMeanExpOn;
            pub const withGroupCumWeightedLogmeanexp = lazy_relation_methods_mod.withGroupCumWeightedLogmeanexp;
            pub const withGroupCumWeightedLogmeanexpOn = lazy_relation_methods_mod.withGroupCumWeightedLogmeanexpOn;
            pub const withGroupCumWeightedRange = lazy_relation_methods_mod.withGroupCumWeightedRange;
            pub const withGroupCumWeightedRangeOn = lazy_relation_methods_mod.withGroupCumWeightedRangeOn;
            pub const withGroupCumWeightedMidrange = lazy_relation_methods_mod.withGroupCumWeightedMidrange;
            pub const withGroupCumWeightedMidrangeOn = lazy_relation_methods_mod.withGroupCumWeightedMidrangeOn;
            pub const withGroupCumWeightedRangeCoeff = lazy_relation_methods_mod.withGroupCumWeightedRangeCoeff;
            pub const withGroupCumWeightedRangeCoeffOn = lazy_relation_methods_mod.withGroupCumWeightedRangeCoeffOn;
            pub const withGroupCumWeightedRangeCoefficient = lazy_relation_methods_mod.withGroupCumWeightedRangeCoefficient;
            pub const withGroupCumWeightedRangeCoefficientOn = lazy_relation_methods_mod.withGroupCumWeightedRangeCoefficientOn;
            pub const withGroupCumulativeWeightedVariance = lazy_relation_methods_mod.withGroupCumulativeWeightedVariance;
            pub const withGroupCumulativeWeightedVarianceOn = lazy_relation_methods_mod.withGroupCumulativeWeightedVarianceOn;
            pub const withGroupCumulativeWeightedVar = lazy_relation_methods_mod.withGroupCumulativeWeightedVar;
            pub const withGroupCumulativeWeightedVarOn = lazy_relation_methods_mod.withGroupCumulativeWeightedVarOn;
            pub const withGroupCumulativeWeightedStddev = lazy_relation_methods_mod.withGroupCumulativeWeightedStddev;
            pub const withGroupCumulativeWeightedStddevOn = lazy_relation_methods_mod.withGroupCumulativeWeightedStddevOn;
            pub const withGroupCumulativeWeightedStd = lazy_relation_methods_mod.withGroupCumulativeWeightedStd;
            pub const withGroupCumulativeWeightedStdOn = lazy_relation_methods_mod.withGroupCumulativeWeightedStdOn;
            pub const withGroupCumWeightedVariance = lazy_relation_methods_mod.withGroupCumWeightedVariance;
            pub const withGroupCumWeightedVarianceOn = lazy_relation_methods_mod.withGroupCumWeightedVarianceOn;
            pub const withGroupCumWeightedVar = lazy_relation_methods_mod.withGroupCumWeightedVar;
            pub const withGroupCumWeightedVarOn = lazy_relation_methods_mod.withGroupCumWeightedVarOn;
            pub const withGroupCumWeightedStddev = lazy_relation_methods_mod.withGroupCumWeightedStddev;
            pub const withGroupCumWeightedStddevOn = lazy_relation_methods_mod.withGroupCumWeightedStddevOn;
            pub const withGroupCumWeightedStd = lazy_relation_methods_mod.withGroupCumWeightedStd;
            pub const withGroupCumWeightedStdOn = lazy_relation_methods_mod.withGroupCumWeightedStdOn;
            pub const withGroupCumulativeProduct = lazy_relation_methods_mod.withGroupCumulativeProduct;
            pub const withGroupCumulativeProductOn = lazy_relation_methods_mod.withGroupCumulativeProductOn;
            pub const withGroupCumProduct = lazy_relation_methods_mod.withGroupCumProduct;
            pub const withGroupCumProductOn = lazy_relation_methods_mod.withGroupCumProductOn;
            pub const withGroupCumProd = lazy_relation_methods_mod.withGroupCumProd;
            pub const withGroupCumProdOn = lazy_relation_methods_mod.withGroupCumProdOn;
            pub const withGroupCumulativeMin = lazy_relation_methods_mod.withGroupCumulativeMin;
            pub const withGroupCumulativeMinOn = lazy_relation_methods_mod.withGroupCumulativeMinOn;
            pub const withGroupCumulativeMax = lazy_relation_methods_mod.withGroupCumulativeMax;
            pub const withGroupCumulativeMaxOn = lazy_relation_methods_mod.withGroupCumulativeMaxOn;
            pub const withGroupCumMin = lazy_relation_methods_mod.withGroupCumMin;
            pub const withGroupCumMinOn = lazy_relation_methods_mod.withGroupCumMinOn;
            pub const withGroupCumMax = lazy_relation_methods_mod.withGroupCumMax;
            pub const withGroupCumMaxOn = lazy_relation_methods_mod.withGroupCumMaxOn;
            pub const withGroupCumulativeVariance = lazy_relation_methods_mod.withGroupCumulativeVariance;
            pub const withGroupCumulativeVarianceOn = lazy_relation_methods_mod.withGroupCumulativeVarianceOn;
            pub const withGroupCumulativeVar = lazy_relation_methods_mod.withGroupCumulativeVar;
            pub const withGroupCumulativeVarOn = lazy_relation_methods_mod.withGroupCumulativeVarOn;
            pub const withGroupCumVariance = lazy_relation_methods_mod.withGroupCumVariance;
            pub const withGroupCumVarianceOn = lazy_relation_methods_mod.withGroupCumVarianceOn;
            pub const withGroupCumVar = lazy_relation_methods_mod.withGroupCumVar;
            pub const withGroupCumVarOn = lazy_relation_methods_mod.withGroupCumVarOn;
            pub const withGroupCumulativeStddev = lazy_relation_methods_mod.withGroupCumulativeStddev;
            pub const withGroupCumulativeStddevOn = lazy_relation_methods_mod.withGroupCumulativeStddevOn;
            pub const withGroupCumulativeStd = lazy_relation_methods_mod.withGroupCumulativeStd;
            pub const withGroupCumulativeStdOn = lazy_relation_methods_mod.withGroupCumulativeStdOn;
            pub const withGroupCumStddev = lazy_relation_methods_mod.withGroupCumStddev;
            pub const withGroupCumStddevOn = lazy_relation_methods_mod.withGroupCumStddevOn;
            pub const withGroupCumStd = lazy_relation_methods_mod.withGroupCumStd;
            pub const withGroupCumStdOn = lazy_relation_methods_mod.withGroupCumStdOn;
            pub const withGroupCumulativeSem = lazy_relation_methods_mod.withGroupCumulativeSem;
            pub const withGroupCumulativeSemOn = lazy_relation_methods_mod.withGroupCumulativeSemOn;
            pub const withGroupCumulativeSEM = lazy_relation_methods_mod.withGroupCumulativeSEM;
            pub const withGroupCumulativeSEMOn = lazy_relation_methods_mod.withGroupCumulativeSEMOn;
            pub const withGroupCumSem = lazy_relation_methods_mod.withGroupCumSem;
            pub const withGroupCumSemOn = lazy_relation_methods_mod.withGroupCumSemOn;
            pub const withGroupCumulativeCv = lazy_relation_methods_mod.withGroupCumulativeCv;
            pub const withGroupCumulativeCvOn = lazy_relation_methods_mod.withGroupCumulativeCvOn;
            pub const withGroupCumulativeCV = lazy_relation_methods_mod.withGroupCumulativeCV;
            pub const withGroupCumulativeCVOn = lazy_relation_methods_mod.withGroupCumulativeCVOn;
            pub const withGroupCumCv = lazy_relation_methods_mod.withGroupCumCv;
            pub const withGroupCumCvOn = lazy_relation_methods_mod.withGroupCumCvOn;
            pub const withGroupCumulativeFano = lazy_relation_methods_mod.withGroupCumulativeFano;
            pub const withGroupCumulativeFanoOn = lazy_relation_methods_mod.withGroupCumulativeFanoOn;
            pub const withGroupCumFano = lazy_relation_methods_mod.withGroupCumFano;
            pub const withGroupCumFanoOn = lazy_relation_methods_mod.withGroupCumFanoOn;
            pub const withGroupCumulativeIndexOfDispersion = lazy_relation_methods_mod.withGroupCumulativeIndexOfDispersion;
            pub const withGroupCumulativeIndexOfDispersionOn = lazy_relation_methods_mod.withGroupCumulativeIndexOfDispersionOn;
            pub const withGroupCumIndexOfDispersion = lazy_relation_methods_mod.withGroupCumIndexOfDispersion;
            pub const withGroupCumIndexOfDispersionOn = lazy_relation_methods_mod.withGroupCumIndexOfDispersionOn;
            pub const withGroupCumulativeSkewness = lazy_relation_methods_mod.withGroupCumulativeSkewness;
            pub const withGroupCumulativeSkewnessOn = lazy_relation_methods_mod.withGroupCumulativeSkewnessOn;
            pub const withGroupCumulativeSkew = lazy_relation_methods_mod.withGroupCumulativeSkew;
            pub const withGroupCumulativeSkewOn = lazy_relation_methods_mod.withGroupCumulativeSkewOn;
            pub const withGroupCumSkewness = lazy_relation_methods_mod.withGroupCumSkewness;
            pub const withGroupCumSkewnessOn = lazy_relation_methods_mod.withGroupCumSkewnessOn;
            pub const withGroupCumSkew = lazy_relation_methods_mod.withGroupCumSkew;
            pub const withGroupCumSkewOn = lazy_relation_methods_mod.withGroupCumSkewOn;
            pub const withGroupCumulativeKurtosis = lazy_relation_methods_mod.withGroupCumulativeKurtosis;
            pub const withGroupCumulativeKurtosisOn = lazy_relation_methods_mod.withGroupCumulativeKurtosisOn;
            pub const withGroupCumulativeKurt = lazy_relation_methods_mod.withGroupCumulativeKurt;
            pub const withGroupCumulativeKurtOn = lazy_relation_methods_mod.withGroupCumulativeKurtOn;
            pub const withGroupCumKurtosis = lazy_relation_methods_mod.withGroupCumKurtosis;
            pub const withGroupCumKurtosisOn = lazy_relation_methods_mod.withGroupCumKurtosisOn;
            pub const withGroupCumKurt = lazy_relation_methods_mod.withGroupCumKurt;
            pub const withGroupCumKurtOn = lazy_relation_methods_mod.withGroupCumKurtOn;
            pub const withGroupCumulativeMeanAbs = lazy_relation_methods_mod.withGroupCumulativeMeanAbs;
            pub const withGroupCumulativeMeanAbsOn = lazy_relation_methods_mod.withGroupCumulativeMeanAbsOn;
            pub const withGroupCumulativeMeanAbsolute = lazy_relation_methods_mod.withGroupCumulativeMeanAbsolute;
            pub const withGroupCumulativeMeanAbsoluteOn = lazy_relation_methods_mod.withGroupCumulativeMeanAbsoluteOn;
            pub const withGroupCumMeanAbs = lazy_relation_methods_mod.withGroupCumMeanAbs;
            pub const withGroupCumMeanAbsOn = lazy_relation_methods_mod.withGroupCumMeanAbsOn;
            pub const withGroupCumMeanAbsolute = lazy_relation_methods_mod.withGroupCumMeanAbsolute;
            pub const withGroupCumMeanAbsoluteOn = lazy_relation_methods_mod.withGroupCumMeanAbsoluteOn;
            pub const withGroupCumulativeMeanSquare = lazy_relation_methods_mod.withGroupCumulativeMeanSquare;
            pub const withGroupCumulativeMeanSquareOn = lazy_relation_methods_mod.withGroupCumulativeMeanSquareOn;
            pub const withGroupCumulativeMeanSquared = lazy_relation_methods_mod.withGroupCumulativeMeanSquared;
            pub const withGroupCumulativeMeanSquaredOn = lazy_relation_methods_mod.withGroupCumulativeMeanSquaredOn;
            pub const withGroupCumulativeMeanSq = lazy_relation_methods_mod.withGroupCumulativeMeanSq;
            pub const withGroupCumulativeMeanSqOn = lazy_relation_methods_mod.withGroupCumulativeMeanSqOn;
            pub const withGroupCumMeanSquare = lazy_relation_methods_mod.withGroupCumMeanSquare;
            pub const withGroupCumMeanSquareOn = lazy_relation_methods_mod.withGroupCumMeanSquareOn;
            pub const withGroupCumMeanSquared = lazy_relation_methods_mod.withGroupCumMeanSquared;
            pub const withGroupCumMeanSquaredOn = lazy_relation_methods_mod.withGroupCumMeanSquaredOn;
            pub const withGroupCumMeanSq = lazy_relation_methods_mod.withGroupCumMeanSq;
            pub const withGroupCumMeanSqOn = lazy_relation_methods_mod.withGroupCumMeanSqOn;
            pub const withGroupCumulativeRms = lazy_relation_methods_mod.withGroupCumulativeRms;
            pub const withGroupCumulativeRmsOn = lazy_relation_methods_mod.withGroupCumulativeRmsOn;
            pub const withGroupCumulativeRMS = lazy_relation_methods_mod.withGroupCumulativeRMS;
            pub const withGroupCumulativeRMSOn = lazy_relation_methods_mod.withGroupCumulativeRMSOn;
            pub const withGroupCumRms = lazy_relation_methods_mod.withGroupCumRms;
            pub const withGroupCumRmsOn = lazy_relation_methods_mod.withGroupCumRmsOn;
            pub const withGroupCumRMS = lazy_relation_methods_mod.withGroupCumRMS;
            pub const withGroupCumRMSOn = lazy_relation_methods_mod.withGroupCumRMSOn;
            pub const withGroupCumulativeMaxAbs = lazy_relation_methods_mod.withGroupCumulativeMaxAbs;
            pub const withGroupCumulativeMaxAbsOn = lazy_relation_methods_mod.withGroupCumulativeMaxAbsOn;
            pub const withGroupCumulativeMaxAbsolute = lazy_relation_methods_mod.withGroupCumulativeMaxAbsolute;
            pub const withGroupCumulativeMaxAbsoluteOn = lazy_relation_methods_mod.withGroupCumulativeMaxAbsoluteOn;
            pub const withGroupCumMaxAbs = lazy_relation_methods_mod.withGroupCumMaxAbs;
            pub const withGroupCumMaxAbsOn = lazy_relation_methods_mod.withGroupCumMaxAbsOn;
            pub const withGroupCumMaxAbsolute = lazy_relation_methods_mod.withGroupCumMaxAbsolute;
            pub const withGroupCumMaxAbsoluteOn = lazy_relation_methods_mod.withGroupCumMaxAbsoluteOn;
            pub const withGroupCumulativeLInfNorm = lazy_relation_methods_mod.withGroupCumulativeLInfNorm;
            pub const withGroupCumulativeLInfNormOn = lazy_relation_methods_mod.withGroupCumulativeLInfNormOn;
            pub const withGroupCumulativeLinfNorm = lazy_relation_methods_mod.withGroupCumulativeLinfNorm;
            pub const withGroupCumulativeLinfNormOn = lazy_relation_methods_mod.withGroupCumulativeLinfNormOn;
            pub const withGroupCumLInfNorm = lazy_relation_methods_mod.withGroupCumLInfNorm;
            pub const withGroupCumLInfNormOn = lazy_relation_methods_mod.withGroupCumLInfNormOn;
            pub const withGroupCumLinfNorm = lazy_relation_methods_mod.withGroupCumLinfNorm;
            pub const withGroupCumLinfNormOn = lazy_relation_methods_mod.withGroupCumLinfNormOn;
            pub const withGroupCumulativeMinAbs = lazy_relation_methods_mod.withGroupCumulativeMinAbs;
            pub const withGroupCumulativeMinAbsOn = lazy_relation_methods_mod.withGroupCumulativeMinAbsOn;
            pub const withGroupCumulativeMinAbsolute = lazy_relation_methods_mod.withGroupCumulativeMinAbsolute;
            pub const withGroupCumulativeMinAbsoluteOn = lazy_relation_methods_mod.withGroupCumulativeMinAbsoluteOn;
            pub const withGroupCumMinAbs = lazy_relation_methods_mod.withGroupCumMinAbs;
            pub const withGroupCumMinAbsOn = lazy_relation_methods_mod.withGroupCumMinAbsOn;
            pub const withGroupCumMinAbsolute = lazy_relation_methods_mod.withGroupCumMinAbsolute;
            pub const withGroupCumMinAbsoluteOn = lazy_relation_methods_mod.withGroupCumMinAbsoluteOn;
            pub const withGroupCumulativeL1Norm = lazy_relation_methods_mod.withGroupCumulativeL1Norm;
            pub const withGroupCumulativeL1NormOn = lazy_relation_methods_mod.withGroupCumulativeL1NormOn;
            pub const withGroupCumL1Norm = lazy_relation_methods_mod.withGroupCumL1Norm;
            pub const withGroupCumL1NormOn = lazy_relation_methods_mod.withGroupCumL1NormOn;
            pub const withGroupCumulativeL2Norm = lazy_relation_methods_mod.withGroupCumulativeL2Norm;
            pub const withGroupCumulativeL2NormOn = lazy_relation_methods_mod.withGroupCumulativeL2NormOn;
            pub const withGroupCumL2Norm = lazy_relation_methods_mod.withGroupCumL2Norm;
            pub const withGroupCumL2NormOn = lazy_relation_methods_mod.withGroupCumL2NormOn;
            pub const withGroupCumulativeRange = lazy_relation_methods_mod.withGroupCumulativeRange;
            pub const withGroupCumulativeRangeOn = lazy_relation_methods_mod.withGroupCumulativeRangeOn;
            pub const withGroupCumulativePtp = lazy_relation_methods_mod.withGroupCumulativePtp;
            pub const withGroupCumulativePtpOn = lazy_relation_methods_mod.withGroupCumulativePtpOn;
            pub const withGroupCumulativePTP = lazy_relation_methods_mod.withGroupCumulativePTP;
            pub const withGroupCumulativePTPOn = lazy_relation_methods_mod.withGroupCumulativePTPOn;
            pub const withGroupCumulativePeakToPeak = lazy_relation_methods_mod.withGroupCumulativePeakToPeak;
            pub const withGroupCumulativePeakToPeakOn = lazy_relation_methods_mod.withGroupCumulativePeakToPeakOn;
            pub const withGroupCumRange = lazy_relation_methods_mod.withGroupCumRange;
            pub const withGroupCumRangeOn = lazy_relation_methods_mod.withGroupCumRangeOn;
            pub const withGroupCumPtp = lazy_relation_methods_mod.withGroupCumPtp;
            pub const withGroupCumPtpOn = lazy_relation_methods_mod.withGroupCumPtpOn;
            pub const withGroupCumPTP = lazy_relation_methods_mod.withGroupCumPTP;
            pub const withGroupCumPTPOn = lazy_relation_methods_mod.withGroupCumPTPOn;
            pub const withGroupCumPeakToPeak = lazy_relation_methods_mod.withGroupCumPeakToPeak;
            pub const withGroupCumPeakToPeakOn = lazy_relation_methods_mod.withGroupCumPeakToPeakOn;
            pub const withGroupCumulativeMidrange = lazy_relation_methods_mod.withGroupCumulativeMidrange;
            pub const withGroupCumulativeMidrangeOn = lazy_relation_methods_mod.withGroupCumulativeMidrangeOn;
            pub const withGroupCumMidrange = lazy_relation_methods_mod.withGroupCumMidrange;
            pub const withGroupCumMidrangeOn = lazy_relation_methods_mod.withGroupCumMidrangeOn;
            pub const withGroupCumulativeRangeCoeff = lazy_relation_methods_mod.withGroupCumulativeRangeCoeff;
            pub const withGroupCumulativeRangeCoeffOn = lazy_relation_methods_mod.withGroupCumulativeRangeCoeffOn;
            pub const withGroupCumulativeRangeCoefficient = lazy_relation_methods_mod.withGroupCumulativeRangeCoefficient;
            pub const withGroupCumulativeRangeCoefficientOn = lazy_relation_methods_mod.withGroupCumulativeRangeCoefficientOn;
            pub const withGroupCumRangeCoeff = lazy_relation_methods_mod.withGroupCumRangeCoeff;
            pub const withGroupCumRangeCoeffOn = lazy_relation_methods_mod.withGroupCumRangeCoeffOn;
            pub const withGroupCumRangeCoefficient = lazy_relation_methods_mod.withGroupCumRangeCoefficient;
            pub const withGroupCumRangeCoefficientOn = lazy_relation_methods_mod.withGroupCumRangeCoefficientOn;
            pub const withGroupCumulativeLogSumExp = lazy_relation_methods_mod.withGroupCumulativeLogSumExp;
            pub const withGroupCumulativeLogSumExpOn = lazy_relation_methods_mod.withGroupCumulativeLogSumExpOn;
            pub const withGroupCumulativeLogsumexp = lazy_relation_methods_mod.withGroupCumulativeLogsumexp;
            pub const withGroupCumulativeLogsumexpOn = lazy_relation_methods_mod.withGroupCumulativeLogsumexpOn;
            pub const withGroupCumLogSumExp = lazy_relation_methods_mod.withGroupCumLogSumExp;
            pub const withGroupCumLogSumExpOn = lazy_relation_methods_mod.withGroupCumLogSumExpOn;
            pub const withGroupCumLogsumexp = lazy_relation_methods_mod.withGroupCumLogsumexp;
            pub const withGroupCumLogsumexpOn = lazy_relation_methods_mod.withGroupCumLogsumexpOn;
            pub const withGroupCumulativeLogMeanExp = lazy_relation_methods_mod.withGroupCumulativeLogMeanExp;
            pub const withGroupCumulativeLogMeanExpOn = lazy_relation_methods_mod.withGroupCumulativeLogMeanExpOn;
            pub const withGroupCumulativeLogmeanexp = lazy_relation_methods_mod.withGroupCumulativeLogmeanexp;
            pub const withGroupCumulativeLogmeanexpOn = lazy_relation_methods_mod.withGroupCumulativeLogmeanexpOn;
            pub const withGroupCumLogMeanExp = lazy_relation_methods_mod.withGroupCumLogMeanExp;
            pub const withGroupCumLogMeanExpOn = lazy_relation_methods_mod.withGroupCumLogMeanExpOn;
            pub const withGroupCumLogmeanexp = lazy_relation_methods_mod.withGroupCumLogmeanexp;
            pub const withGroupCumLogmeanexpOn = lazy_relation_methods_mod.withGroupCumLogmeanexpOn;
            pub const withGroupCumulativeGeometricMean = lazy_relation_methods_mod.withGroupCumulativeGeometricMean;
            pub const withGroupCumulativeGeometricMeanOn = lazy_relation_methods_mod.withGroupCumulativeGeometricMeanOn;
            pub const withGroupCumulativeGeoMean = lazy_relation_methods_mod.withGroupCumulativeGeoMean;
            pub const withGroupCumulativeGeoMeanOn = lazy_relation_methods_mod.withGroupCumulativeGeoMeanOn;
            pub const withGroupCumGeometricMean = lazy_relation_methods_mod.withGroupCumGeometricMean;
            pub const withGroupCumGeometricMeanOn = lazy_relation_methods_mod.withGroupCumGeometricMeanOn;
            pub const withGroupCumGeoMean = lazy_relation_methods_mod.withGroupCumGeoMean;
            pub const withGroupCumGeoMeanOn = lazy_relation_methods_mod.withGroupCumGeoMeanOn;
            pub const withGroupCumulativeHarmonicMean = lazy_relation_methods_mod.withGroupCumulativeHarmonicMean;
            pub const withGroupCumulativeHarmonicMeanOn = lazy_relation_methods_mod.withGroupCumulativeHarmonicMeanOn;
            pub const withGroupCumulativeHarmMean = lazy_relation_methods_mod.withGroupCumulativeHarmMean;
            pub const withGroupCumulativeHarmMeanOn = lazy_relation_methods_mod.withGroupCumulativeHarmMeanOn;
            pub const withGroupCumHarmonicMean = lazy_relation_methods_mod.withGroupCumHarmonicMean;
            pub const withGroupCumHarmonicMeanOn = lazy_relation_methods_mod.withGroupCumHarmonicMeanOn;
            pub const withGroupCumHarmMean = lazy_relation_methods_mod.withGroupCumHarmMean;
            pub const withGroupCumHarmMeanOn = lazy_relation_methods_mod.withGroupCumHarmMeanOn;
            pub const withGroupCumulativeArgMin = lazy_relation_methods_mod.withGroupCumulativeArgMin;
            pub const withGroupCumulativeArgMinOn = lazy_relation_methods_mod.withGroupCumulativeArgMinOn;
            pub const withGroupCumArgMin = lazy_relation_methods_mod.withGroupCumArgMin;
            pub const withGroupCumArgMinOn = lazy_relation_methods_mod.withGroupCumArgMinOn;
            pub const withGroupCumulativeArgmin = lazy_relation_methods_mod.withGroupCumulativeArgmin;
            pub const withGroupCumulativeArgminOn = lazy_relation_methods_mod.withGroupCumulativeArgminOn;
            pub const withGroupCumArgmin = lazy_relation_methods_mod.withGroupCumArgmin;
            pub const withGroupCumArgminOn = lazy_relation_methods_mod.withGroupCumArgminOn;
            pub const withGroupCumulativeArgMax = lazy_relation_methods_mod.withGroupCumulativeArgMax;
            pub const withGroupCumulativeArgMaxOn = lazy_relation_methods_mod.withGroupCumulativeArgMaxOn;
            pub const withGroupCumArgMax = lazy_relation_methods_mod.withGroupCumArgMax;
            pub const withGroupCumArgMaxOn = lazy_relation_methods_mod.withGroupCumArgMaxOn;
            pub const withGroupCumulativeArgmax = lazy_relation_methods_mod.withGroupCumulativeArgmax;
            pub const withGroupCumulativeArgmaxOn = lazy_relation_methods_mod.withGroupCumulativeArgmaxOn;
            pub const withGroupCumArgmax = lazy_relation_methods_mod.withGroupCumArgmax;
            pub const withGroupCumArgmaxOn = lazy_relation_methods_mod.withGroupCumArgmaxOn;
            pub const withGroupRowNumber = lazy_relation_methods_mod.withGroupRowNumber;
            pub const withGroupRowNumberOn = lazy_relation_methods_mod.withGroupRowNumberOn;
            pub const withGroupCumCount = lazy_relation_methods_mod.withGroupCumCount;
            pub const withGroupCumCountOn = lazy_relation_methods_mod.withGroupCumCountOn;
            pub const withGroupSize = lazy_relation_methods_mod.withGroupSize;
            pub const withGroupSizeOn = lazy_relation_methods_mod.withGroupSizeOn;
            pub const withGroupCount = lazy_relation_methods_mod.withGroupCount;
            pub const withGroupCountOn = lazy_relation_methods_mod.withGroupCountOn;
            pub const withGroupReverseRowNumber = lazy_relation_methods_mod.withGroupReverseRowNumber;
            pub const withGroupReverseRowNumberOn = lazy_relation_methods_mod.withGroupReverseRowNumberOn;
            pub const withGroupReverseCumCount = lazy_relation_methods_mod.withGroupReverseCumCount;
            pub const withGroupReverseCumCountOn = lazy_relation_methods_mod.withGroupReverseCumCountOn;
            pub const valueCounts = lazy_relation_methods_mod.valueCounts;
            pub const valueCountsAs = lazy_relation_methods_mod.valueCountsAs;
            pub const valueCountsOn = lazy_relation_methods_mod.valueCountsOn;
            pub const valueCountsOnAs = lazy_relation_methods_mod.valueCountsOnAs;
            pub const valueCountsSorted = lazy_relation_methods_mod.valueCountsSorted;
            pub const valueCountsSortedAs = lazy_relation_methods_mod.valueCountsSortedAs;
            pub const valueCountsOnSorted = lazy_relation_methods_mod.valueCountsOnSorted;
            pub const valueCountsOnSortedAs = lazy_relation_methods_mod.valueCountsOnSortedAs;
            pub const valueCountsSortedOn = lazy_relation_methods_mod.valueCountsSortedOn;
            pub const valueCountsSortedOnAs = lazy_relation_methods_mod.valueCountsSortedOnAs;
            pub const groupByValue = lazy_relation_methods_mod.groupByValue;
            pub const groupByValueOn = lazy_relation_methods_mod.groupByValueOn;
            pub const groupByWeighted = lazy_relation_methods_mod.groupByWeighted;
            pub const groupByWeightedOn = lazy_relation_methods_mod.groupByWeightedOn;
            pub const groupBySum = lazy_relation_methods_mod.groupBySum;
            pub const groupBySumOn = lazy_relation_methods_mod.groupBySumOn;
            pub const groupByProd = lazy_relation_methods_mod.groupByProd;
            pub const groupByProduct = lazy_relation_methods_mod.groupByProduct;
            pub const groupByProdOn = lazy_relation_methods_mod.groupByProdOn;
            pub const groupByProductOn = lazy_relation_methods_mod.groupByProductOn;
            pub const groupByMin = lazy_relation_methods_mod.groupByMin;
            pub const groupByMinOn = lazy_relation_methods_mod.groupByMinOn;
            pub const groupByMax = lazy_relation_methods_mod.groupByMax;
            pub const groupByMaxOn = lazy_relation_methods_mod.groupByMaxOn;
            pub const groupByMean = lazy_relation_methods_mod.groupByMean;
            pub const groupByMeanOn = lazy_relation_methods_mod.groupByMeanOn;
            pub const groupByFirst = lazy_relation_methods_mod.groupByFirst;
            pub const groupByFirstOn = lazy_relation_methods_mod.groupByFirstOn;
            pub const groupByLast = lazy_relation_methods_mod.groupByLast;
            pub const groupByLastOn = lazy_relation_methods_mod.groupByLastOn;
            pub const groupByFirstRow = lazy_relation_methods_mod.groupByFirstRow;
            pub const groupByFirstRowOn = lazy_relation_methods_mod.groupByFirstRowOn;
            pub const groupByLastRow = lazy_relation_methods_mod.groupByLastRow;
            pub const groupByLastRowOn = lazy_relation_methods_mod.groupByLastRowOn;
            pub const groupByNth = lazy_relation_methods_mod.groupByNth;
            pub const groupByNthOn = lazy_relation_methods_mod.groupByNthOn;
            pub const groupByNthRow = lazy_relation_methods_mod.groupByNthRow;
            pub const groupByNthRowOn = lazy_relation_methods_mod.groupByNthRowOn;
            pub const groupByNthIndex = lazy_relation_methods_mod.groupByNthIndex;
            pub const groupByNthIndexOn = lazy_relation_methods_mod.groupByNthIndexOn;
            pub const groupByNthRowIndex = lazy_relation_methods_mod.groupByNthRowIndex;
            pub const groupByNthRowIndexOn = lazy_relation_methods_mod.groupByNthRowIndexOn;
            pub const groupByNUnique = lazy_relation_methods_mod.groupByNUnique;
            pub const groupByNUniqueOn = lazy_relation_methods_mod.groupByNUniqueOn;
            pub const groupByNunique = lazy_relation_methods_mod.groupByNunique;
            pub const groupByNuniqueOn = lazy_relation_methods_mod.groupByNuniqueOn;
            pub const groupByMode = lazy_relation_methods_mod.groupByMode;
            pub const groupByModeOn = lazy_relation_methods_mod.groupByModeOn;
            pub const groupByModeCount = lazy_relation_methods_mod.groupByModeCount;
            pub const groupByModeCountOn = lazy_relation_methods_mod.groupByModeCountOn;
            pub const groupByModeRatio = lazy_relation_methods_mod.groupByModeRatio;
            pub const groupByModeRatioOn = lazy_relation_methods_mod.groupByModeRatioOn;
            pub const groupByModeMargin = lazy_relation_methods_mod.groupByModeMargin;
            pub const groupByModeMarginOn = lazy_relation_methods_mod.groupByModeMarginOn;
            pub const groupByModeMarginRatio = lazy_relation_methods_mod.groupByModeMarginRatio;
            pub const groupByModeMarginRatioOn = lazy_relation_methods_mod.groupByModeMarginRatioOn;
            pub const groupByEntropy = lazy_relation_methods_mod.groupByEntropy;
            pub const groupByEntropyOn = lazy_relation_methods_mod.groupByEntropyOn;
            pub const groupByGiniImpurity = lazy_relation_methods_mod.groupByGiniImpurity;
            pub const groupByGiniImpurityOn = lazy_relation_methods_mod.groupByGiniImpurityOn;
            pub const groupByGini = lazy_relation_methods_mod.groupByGini;
            pub const groupByGiniOn = lazy_relation_methods_mod.groupByGiniOn;
            pub const groupByPerplexity = lazy_relation_methods_mod.groupByPerplexity;
            pub const groupByPerplexityOn = lazy_relation_methods_mod.groupByPerplexityOn;
            pub const groupByInverseSimpson = lazy_relation_methods_mod.groupByInverseSimpson;
            pub const groupByInverseSimpsonOn = lazy_relation_methods_mod.groupByInverseSimpsonOn;
            pub const groupBySimpsonConcentration = lazy_relation_methods_mod.groupBySimpsonConcentration;
            pub const groupBySimpsonConcentrationOn = lazy_relation_methods_mod.groupBySimpsonConcentrationOn;
            pub const groupByConcentration = lazy_relation_methods_mod.groupByConcentration;
            pub const groupByConcentrationOn = lazy_relation_methods_mod.groupByConcentrationOn;
            pub const groupByEvenness = lazy_relation_methods_mod.groupByEvenness;
            pub const groupByEvennessOn = lazy_relation_methods_mod.groupByEvennessOn;
            pub const groupByGiniMeanDiff = lazy_relation_methods_mod.groupByGiniMeanDiff;
            pub const groupByGiniMeanDiffOn = lazy_relation_methods_mod.groupByGiniMeanDiffOn;
            pub const groupByGiniCoefficient = lazy_relation_methods_mod.groupByGiniCoefficient;
            pub const groupByGiniCoefficientOn = lazy_relation_methods_mod.groupByGiniCoefficientOn;
            pub const groupByGiniCoeff = lazy_relation_methods_mod.groupByGiniCoeff;
            pub const groupByGiniCoeffOn = lazy_relation_methods_mod.groupByGiniCoeffOn;
            pub const groupByWeightedMean = lazy_relation_methods_mod.groupByWeightedMean;
            pub const groupByWeightedMeanOn = lazy_relation_methods_mod.groupByWeightedMeanOn;
            pub const groupByWeightedSum = lazy_relation_methods_mod.groupByWeightedSum;
            pub const groupByWeightedSumOn = lazy_relation_methods_mod.groupByWeightedSumOn;
            pub const groupByWeightedProduct = lazy_relation_methods_mod.groupByWeightedProduct;
            pub const groupByWeightedProductOn = lazy_relation_methods_mod.groupByWeightedProductOn;
            pub const groupByWeightedWeightSum = lazy_relation_methods_mod.groupByWeightedWeightSum;
            pub const groupByWeightedWeightSumOn = lazy_relation_methods_mod.groupByWeightedWeightSumOn;
            pub const groupByWeightedPositiveCount = lazy_relation_methods_mod.groupByWeightedPositiveCount;
            pub const groupByWeightedPositiveCountOn = lazy_relation_methods_mod.groupByWeightedPositiveCountOn;
            pub const groupByWeightedEffectiveN = lazy_relation_methods_mod.groupByWeightedEffectiveN;
            pub const groupByWeightedEffectiveNOn = lazy_relation_methods_mod.groupByWeightedEffectiveNOn;
            pub const groupByWeightedEffectiveCount = lazy_relation_methods_mod.groupByWeightedEffectiveCount;
            pub const groupByWeightedEffectiveCountOn = lazy_relation_methods_mod.groupByWeightedEffectiveCountOn;
            pub const groupByWeightedProd = lazy_relation_methods_mod.groupByWeightedProd;
            pub const groupByWeightedProdOn = lazy_relation_methods_mod.groupByWeightedProdOn;
            pub const groupByWeightedMeanSquare = lazy_relation_methods_mod.groupByWeightedMeanSquare;
            pub const groupByWeightedMeanSquareOn = lazy_relation_methods_mod.groupByWeightedMeanSquareOn;
            pub const groupByWeightedRms = lazy_relation_methods_mod.groupByWeightedRms;
            pub const groupByWeightedRmsOn = lazy_relation_methods_mod.groupByWeightedRmsOn;
            pub const groupByWeightedMeanSquared = lazy_relation_methods_mod.groupByWeightedMeanSquared;
            pub const groupByWeightedMeanSquaredOn = lazy_relation_methods_mod.groupByWeightedMeanSquaredOn;
            pub const groupByWeightedMeanSq = lazy_relation_methods_mod.groupByWeightedMeanSq;
            pub const groupByWeightedMeanSqOn = lazy_relation_methods_mod.groupByWeightedMeanSqOn;
            pub const groupByWeightedRMS = lazy_relation_methods_mod.groupByWeightedRMS;
            pub const groupByWeightedRMSOn = lazy_relation_methods_mod.groupByWeightedRMSOn;
            pub const groupByWeightedMin = lazy_relation_methods_mod.groupByWeightedMin;
            pub const groupByWeightedMinOn = lazy_relation_methods_mod.groupByWeightedMinOn;
            pub const groupByWeightedMinimum = lazy_relation_methods_mod.groupByWeightedMinimum;
            pub const groupByWeightedMinimumOn = lazy_relation_methods_mod.groupByWeightedMinimumOn;
            pub const groupByWeightedMax = lazy_relation_methods_mod.groupByWeightedMax;
            pub const groupByWeightedMaxOn = lazy_relation_methods_mod.groupByWeightedMaxOn;
            pub const groupByWeightedMaximum = lazy_relation_methods_mod.groupByWeightedMaximum;
            pub const groupByWeightedMaximumOn = lazy_relation_methods_mod.groupByWeightedMaximumOn;
            pub const groupByWeightedMeanAbs = lazy_relation_methods_mod.groupByWeightedMeanAbs;
            pub const groupByWeightedMeanAbsOn = lazy_relation_methods_mod.groupByWeightedMeanAbsOn;
            pub const groupByWeightedL1Norm = lazy_relation_methods_mod.groupByWeightedL1Norm;
            pub const groupByWeightedL1NormOn = lazy_relation_methods_mod.groupByWeightedL1NormOn;
            pub const groupByWeightedL2Norm = lazy_relation_methods_mod.groupByWeightedL2Norm;
            pub const groupByWeightedL2NormOn = lazy_relation_methods_mod.groupByWeightedL2NormOn;
            pub const groupByWeightedMaxAbs = lazy_relation_methods_mod.groupByWeightedMaxAbs;
            pub const groupByWeightedMaxAbsOn = lazy_relation_methods_mod.groupByWeightedMaxAbsOn;
            pub const groupByWeightedMinAbs = lazy_relation_methods_mod.groupByWeightedMinAbs;
            pub const groupByWeightedMinAbsOn = lazy_relation_methods_mod.groupByWeightedMinAbsOn;
            pub const groupByWeightedL1 = lazy_relation_methods_mod.groupByWeightedL1;
            pub const groupByWeightedL1On = lazy_relation_methods_mod.groupByWeightedL1On;
            pub const groupByWeightedL2 = lazy_relation_methods_mod.groupByWeightedL2;
            pub const groupByWeightedL2On = lazy_relation_methods_mod.groupByWeightedL2On;
            pub const groupByWeightedMaxAbsolute = lazy_relation_methods_mod.groupByWeightedMaxAbsolute;
            pub const groupByWeightedMaxAbsoluteOn = lazy_relation_methods_mod.groupByWeightedMaxAbsoluteOn;
            pub const groupByWeightedMinAbsolute = lazy_relation_methods_mod.groupByWeightedMinAbsolute;
            pub const groupByWeightedMinAbsoluteOn = lazy_relation_methods_mod.groupByWeightedMinAbsoluteOn;
            pub const groupByWeightedGeometricMean = lazy_relation_methods_mod.groupByWeightedGeometricMean;
            pub const groupByWeightedGeometricMeanOn = lazy_relation_methods_mod.groupByWeightedGeometricMeanOn;
            pub const groupByWeightedGeoMean = lazy_relation_methods_mod.groupByWeightedGeoMean;
            pub const groupByWeightedGeoMeanOn = lazy_relation_methods_mod.groupByWeightedGeoMeanOn;
            pub const groupByWeightedHarmonicMean = lazy_relation_methods_mod.groupByWeightedHarmonicMean;
            pub const groupByWeightedHarmonicMeanOn = lazy_relation_methods_mod.groupByWeightedHarmonicMeanOn;
            pub const groupByWeightedHarmMean = lazy_relation_methods_mod.groupByWeightedHarmMean;
            pub const groupByWeightedHarmMeanOn = lazy_relation_methods_mod.groupByWeightedHarmMeanOn;
            pub const groupByWeightedLogSumExp = lazy_relation_methods_mod.groupByWeightedLogSumExp;
            pub const groupByWeightedLogSumExpOn = lazy_relation_methods_mod.groupByWeightedLogSumExpOn;
            pub const groupByWeightedLogsumexp = lazy_relation_methods_mod.groupByWeightedLogsumexp;
            pub const groupByWeightedLogsumexpOn = lazy_relation_methods_mod.groupByWeightedLogsumexpOn;
            pub const groupByWeightedLogMeanExp = lazy_relation_methods_mod.groupByWeightedLogMeanExp;
            pub const groupByWeightedLogMeanExpOn = lazy_relation_methods_mod.groupByWeightedLogMeanExpOn;
            pub const groupByWeightedLogmeanexp = lazy_relation_methods_mod.groupByWeightedLogmeanexp;
            pub const groupByWeightedLogmeanexpOn = lazy_relation_methods_mod.groupByWeightedLogmeanexpOn;
            pub const groupByWeightedRange = lazy_relation_methods_mod.groupByWeightedRange;
            pub const groupByWeightedRangeOn = lazy_relation_methods_mod.groupByWeightedRangeOn;
            pub const groupByWeightedMidrange = lazy_relation_methods_mod.groupByWeightedMidrange;
            pub const groupByWeightedMidrangeOn = lazy_relation_methods_mod.groupByWeightedMidrangeOn;
            pub const groupByWeightedRangeCoeff = lazy_relation_methods_mod.groupByWeightedRangeCoeff;
            pub const groupByWeightedRangeCoeffOn = lazy_relation_methods_mod.groupByWeightedRangeCoeffOn;
            pub const groupByWeightedRangeCoefficient = lazy_relation_methods_mod.groupByWeightedRangeCoefficient;
            pub const groupByWeightedRangeCoefficientOn = lazy_relation_methods_mod.groupByWeightedRangeCoefficientOn;
            pub const groupByWeightedVariance = lazy_relation_methods_mod.groupByWeightedVariance;
            pub const groupByWeightedVarianceOn = lazy_relation_methods_mod.groupByWeightedVarianceOn;
            pub const groupByWeightedVar = lazy_relation_methods_mod.groupByWeightedVar;
            pub const groupByWeightedVarOn = lazy_relation_methods_mod.groupByWeightedVarOn;
            pub const groupByWeightedStddev = lazy_relation_methods_mod.groupByWeightedStddev;
            pub const groupByWeightedStddevOn = lazy_relation_methods_mod.groupByWeightedStddevOn;
            pub const groupByWeightedStd = lazy_relation_methods_mod.groupByWeightedStd;
            pub const groupByWeightedStdOn = lazy_relation_methods_mod.groupByWeightedStdOn;
            pub const groupByWeightedSem = lazy_relation_methods_mod.groupByWeightedSem;
            pub const groupByWeightedSemOn = lazy_relation_methods_mod.groupByWeightedSemOn;
            pub const groupByWeightedCv = lazy_relation_methods_mod.groupByWeightedCv;
            pub const groupByWeightedCvOn = lazy_relation_methods_mod.groupByWeightedCvOn;
            pub const groupByWeightedFano = lazy_relation_methods_mod.groupByWeightedFano;
            pub const groupByWeightedFanoOn = lazy_relation_methods_mod.groupByWeightedFanoOn;
            pub const groupByWeightedSkewness = lazy_relation_methods_mod.groupByWeightedSkewness;
            pub const groupByWeightedSkewnessOn = lazy_relation_methods_mod.groupByWeightedSkewnessOn;
            pub const groupByWeightedSkew = lazy_relation_methods_mod.groupByWeightedSkew;
            pub const groupByWeightedSkewOn = lazy_relation_methods_mod.groupByWeightedSkewOn;
            pub const groupByWeightedKurtosis = lazy_relation_methods_mod.groupByWeightedKurtosis;
            pub const groupByWeightedKurtosisOn = lazy_relation_methods_mod.groupByWeightedKurtosisOn;
            pub const groupByWeightedKurt = lazy_relation_methods_mod.groupByWeightedKurt;
            pub const groupByWeightedKurtOn = lazy_relation_methods_mod.groupByWeightedKurtOn;
            pub const groupByWeightedSEM = lazy_relation_methods_mod.groupByWeightedSEM;
            pub const groupByWeightedSEMOn = lazy_relation_methods_mod.groupByWeightedSEMOn;
            pub const groupByWeightedCV = lazy_relation_methods_mod.groupByWeightedCV;
            pub const groupByWeightedCVOn = lazy_relation_methods_mod.groupByWeightedCVOn;
            pub const groupByWeightedQuantile = lazy_relation_methods_mod.groupByWeightedQuantile;
            pub const groupByWeightedQuantileOn = lazy_relation_methods_mod.groupByWeightedQuantileOn;
            pub const groupByWeightedMedian = lazy_relation_methods_mod.groupByWeightedMedian;
            pub const groupByWeightedMedianOn = lazy_relation_methods_mod.groupByWeightedMedianOn;
            pub const groupByWeightedIqr = lazy_relation_methods_mod.groupByWeightedIqr;
            pub const groupByWeightedIqrOn = lazy_relation_methods_mod.groupByWeightedIqrOn;
            pub const groupByWeightedIQR = lazy_relation_methods_mod.groupByWeightedIQR;
            pub const groupByWeightedIQROn = lazy_relation_methods_mod.groupByWeightedIQROn;
            pub const groupByWeightedMad = lazy_relation_methods_mod.groupByWeightedMad;
            pub const groupByWeightedMadOn = lazy_relation_methods_mod.groupByWeightedMadOn;
            pub const groupByWeightedMAD = lazy_relation_methods_mod.groupByWeightedMAD;
            pub const groupByWeightedMADOn = lazy_relation_methods_mod.groupByWeightedMADOn;
            pub const groupByWeightedTrimmedMean = lazy_relation_methods_mod.groupByWeightedTrimmedMean;
            pub const groupByWeightedTrimmedMeanOn = lazy_relation_methods_mod.groupByWeightedTrimmedMeanOn;
            pub const groupByWeightedWinsorizedMean = lazy_relation_methods_mod.groupByWeightedWinsorizedMean;
            pub const groupByWeightedWinsorizedMeanOn = lazy_relation_methods_mod.groupByWeightedWinsorizedMeanOn;
            pub const groupByWeightedMode = lazy_relation_methods_mod.groupByWeightedMode;
            pub const groupByWeightedModeOn = lazy_relation_methods_mod.groupByWeightedModeOn;
            pub const groupByWeightedModeWeight = lazy_relation_methods_mod.groupByWeightedModeWeight;
            pub const groupByWeightedModeWeightOn = lazy_relation_methods_mod.groupByWeightedModeWeightOn;
            pub const groupByWeightedModeRatio = lazy_relation_methods_mod.groupByWeightedModeRatio;
            pub const groupByWeightedModeRatioOn = lazy_relation_methods_mod.groupByWeightedModeRatioOn;
            pub const groupByWeightedModeMargin = lazy_relation_methods_mod.groupByWeightedModeMargin;
            pub const groupByWeightedModeMarginOn = lazy_relation_methods_mod.groupByWeightedModeMarginOn;
            pub const groupByWeightedModeMarginRatio = lazy_relation_methods_mod.groupByWeightedModeMarginRatio;
            pub const groupByWeightedModeMarginRatioOn = lazy_relation_methods_mod.groupByWeightedModeMarginRatioOn;
            pub const groupByWeightedEntropy = lazy_relation_methods_mod.groupByWeightedEntropy;
            pub const groupByWeightedEntropyOn = lazy_relation_methods_mod.groupByWeightedEntropyOn;
            pub const groupByWeightedGiniImpurity = lazy_relation_methods_mod.groupByWeightedGiniImpurity;
            pub const groupByWeightedGiniImpurityOn = lazy_relation_methods_mod.groupByWeightedGiniImpurityOn;
            pub const groupByWeightedGini = lazy_relation_methods_mod.groupByWeightedGini;
            pub const groupByWeightedGiniOn = lazy_relation_methods_mod.groupByWeightedGiniOn;
            pub const groupByWeightedPerplexity = lazy_relation_methods_mod.groupByWeightedPerplexity;
            pub const groupByWeightedPerplexityOn = lazy_relation_methods_mod.groupByWeightedPerplexityOn;
            pub const groupByWeightedInverseSimpson = lazy_relation_methods_mod.groupByWeightedInverseSimpson;
            pub const groupByWeightedInverseSimpsonOn = lazy_relation_methods_mod.groupByWeightedInverseSimpsonOn;
            pub const groupByWeightedSimpsonConcentration = lazy_relation_methods_mod.groupByWeightedSimpsonConcentration;
            pub const groupByWeightedSimpsonConcentrationOn = lazy_relation_methods_mod.groupByWeightedSimpsonConcentrationOn;
            pub const groupByWeightedConcentration = lazy_relation_methods_mod.groupByWeightedConcentration;
            pub const groupByWeightedConcentrationOn = lazy_relation_methods_mod.groupByWeightedConcentrationOn;
            pub const groupByWeightedEvenness = lazy_relation_methods_mod.groupByWeightedEvenness;
            pub const groupByWeightedEvennessOn = lazy_relation_methods_mod.groupByWeightedEvennessOn;
            pub const groupByWeightedMeanAbsDev = lazy_relation_methods_mod.groupByWeightedMeanAbsDev;
            pub const groupByWeightedMeanAbsDevOn = lazy_relation_methods_mod.groupByWeightedMeanAbsDevOn;
            pub const groupByWeightedMeanAbsDevRatio = lazy_relation_methods_mod.groupByWeightedMeanAbsDevRatio;
            pub const groupByWeightedMeanAbsDevRatioOn = lazy_relation_methods_mod.groupByWeightedMeanAbsDevRatioOn;
            pub const groupByWeightedMeanAbsoluteDeviation = lazy_relation_methods_mod.groupByWeightedMeanAbsoluteDeviation;
            pub const groupByWeightedMeanAbsoluteDeviationOn = lazy_relation_methods_mod.groupByWeightedMeanAbsoluteDeviationOn;
            pub const groupByWeightedGiniMeanDiff = lazy_relation_methods_mod.groupByWeightedGiniMeanDiff;
            pub const groupByWeightedGiniMeanDiffOn = lazy_relation_methods_mod.groupByWeightedGiniMeanDiffOn;
            pub const groupByWeightedGiniCoefficient = lazy_relation_methods_mod.groupByWeightedGiniCoefficient;
            pub const groupByWeightedGiniCoefficientOn = lazy_relation_methods_mod.groupByWeightedGiniCoefficientOn;
            pub const groupByWeightedGiniCoeff = lazy_relation_methods_mod.groupByWeightedGiniCoeff;
            pub const groupByWeightedGiniCoeffOn = lazy_relation_methods_mod.groupByWeightedGiniCoeffOn;
            pub const groupByWeightedInterdecileRange = lazy_relation_methods_mod.groupByWeightedInterdecileRange;
            pub const groupByWeightedInterdecileRangeOn = lazy_relation_methods_mod.groupByWeightedInterdecileRangeOn;
            pub const groupByWeightedIdr = lazy_relation_methods_mod.groupByWeightedIdr;
            pub const groupByWeightedIdrOn = lazy_relation_methods_mod.groupByWeightedIdrOn;
            pub const groupByWeightedIDR = lazy_relation_methods_mod.groupByWeightedIDR;
            pub const groupByWeightedIDROn = lazy_relation_methods_mod.groupByWeightedIDROn;
            pub const groupByWeightedMidhinge = lazy_relation_methods_mod.groupByWeightedMidhinge;
            pub const groupByWeightedMidhingeOn = lazy_relation_methods_mod.groupByWeightedMidhingeOn;
            pub const groupByWeightedTrimean = lazy_relation_methods_mod.groupByWeightedTrimean;
            pub const groupByWeightedTrimeanOn = lazy_relation_methods_mod.groupByWeightedTrimeanOn;
            pub const groupByWeightedBowleySkewness = lazy_relation_methods_mod.groupByWeightedBowleySkewness;
            pub const groupByWeightedBowleySkewnessOn = lazy_relation_methods_mod.groupByWeightedBowleySkewnessOn;
            pub const groupByWeightedBowleySkew = lazy_relation_methods_mod.groupByWeightedBowleySkew;
            pub const groupByWeightedBowleySkewOn = lazy_relation_methods_mod.groupByWeightedBowleySkewOn;
            pub const groupByWeightedQuartileCoeffDispersion = lazy_relation_methods_mod.groupByWeightedQuartileCoeffDispersion;
            pub const groupByWeightedQuartileCoeffDispersionOn = lazy_relation_methods_mod.groupByWeightedQuartileCoeffDispersionOn;
            pub const groupByWeightedQcd = lazy_relation_methods_mod.groupByWeightedQcd;
            pub const groupByWeightedQcdOn = lazy_relation_methods_mod.groupByWeightedQcdOn;
            pub const groupByWeightedQCD = lazy_relation_methods_mod.groupByWeightedQCD;
            pub const groupByWeightedQCDOn = lazy_relation_methods_mod.groupByWeightedQCDOn;
            pub const groupByWeightedKelleySkewness = lazy_relation_methods_mod.groupByWeightedKelleySkewness;
            pub const groupByWeightedKelleySkewnessOn = lazy_relation_methods_mod.groupByWeightedKelleySkewnessOn;
            pub const groupByWeightedKelleySkew = lazy_relation_methods_mod.groupByWeightedKelleySkew;
            pub const groupByWeightedKelleySkewOn = lazy_relation_methods_mod.groupByWeightedKelleySkewOn;
            pub const groupByDot = lazy_relation_methods_mod.groupByDot;
            pub const groupByDotOn = lazy_relation_methods_mod.groupByDotOn;
            pub const groupByCosineSimilarity = lazy_relation_methods_mod.groupByCosineSimilarity;
            pub const groupByCosineSimilarityOn = lazy_relation_methods_mod.groupByCosineSimilarityOn;
            pub const groupByCosine = lazy_relation_methods_mod.groupByCosine;
            pub const groupByCosineOn = lazy_relation_methods_mod.groupByCosineOn;
            pub const groupBySquaredEuclideanDistance = lazy_relation_methods_mod.groupBySquaredEuclideanDistance;
            pub const groupBySquaredEuclideanDistanceOn = lazy_relation_methods_mod.groupBySquaredEuclideanDistanceOn;
            pub const groupByEuclideanDistance = lazy_relation_methods_mod.groupByEuclideanDistance;
            pub const groupByEuclideanDistanceOn = lazy_relation_methods_mod.groupByEuclideanDistanceOn;
            pub const groupByManhattanDistance = lazy_relation_methods_mod.groupByManhattanDistance;
            pub const groupByManhattanDistanceOn = lazy_relation_methods_mod.groupByManhattanDistanceOn;
            pub const groupByChebyshevDistance = lazy_relation_methods_mod.groupByChebyshevDistance;
            pub const groupByChebyshevDistanceOn = lazy_relation_methods_mod.groupByChebyshevDistanceOn;
            pub const groupByCanberraDistance = lazy_relation_methods_mod.groupByCanberraDistance;
            pub const groupByCanberraDistanceOn = lazy_relation_methods_mod.groupByCanberraDistanceOn;
            pub const groupByBrayCurtisDistance = lazy_relation_methods_mod.groupByBrayCurtisDistance;
            pub const groupByBrayCurtisDistanceOn = lazy_relation_methods_mod.groupByBrayCurtisDistanceOn;
            pub const groupByMeanError = lazy_relation_methods_mod.groupByMeanError;
            pub const groupByMeanErrorOn = lazy_relation_methods_mod.groupByMeanErrorOn;
            pub const groupByBias = lazy_relation_methods_mod.groupByBias;
            pub const groupByBiasOn = lazy_relation_methods_mod.groupByBiasOn;
            pub const groupByMae = lazy_relation_methods_mod.groupByMae;
            pub const groupByMaeOn = lazy_relation_methods_mod.groupByMaeOn;
            pub const groupByMse = lazy_relation_methods_mod.groupByMse;
            pub const groupByMseOn = lazy_relation_methods_mod.groupByMseOn;
            pub const groupByRmse = lazy_relation_methods_mod.groupByRmse;
            pub const groupByRmseOn = lazy_relation_methods_mod.groupByRmseOn;
            pub const groupByMape = lazy_relation_methods_mod.groupByMape;
            pub const groupByMapeOn = lazy_relation_methods_mod.groupByMapeOn;
            pub const groupBySmape = lazy_relation_methods_mod.groupBySmape;
            pub const groupBySmapeOn = lazy_relation_methods_mod.groupBySmapeOn;
            pub const groupByPairCount = lazy_relation_methods_mod.groupByPairCount;
            pub const groupByPairCountOn = lazy_relation_methods_mod.groupByPairCountOn;
            pub const groupByCovariance = lazy_relation_methods_mod.groupByCovariance;
            pub const groupByCovarianceOn = lazy_relation_methods_mod.groupByCovarianceOn;
            pub const groupByCov = lazy_relation_methods_mod.groupByCov;
            pub const groupByCovOn = lazy_relation_methods_mod.groupByCovOn;
            pub const groupByCorrelation = lazy_relation_methods_mod.groupByCorrelation;
            pub const groupByCorrelationOn = lazy_relation_methods_mod.groupByCorrelationOn;
            pub const groupByCorr = lazy_relation_methods_mod.groupByCorr;
            pub const groupByCorrOn = lazy_relation_methods_mod.groupByCorrOn;
            pub const groupByBeta = lazy_relation_methods_mod.groupByBeta;
            pub const groupByBetaOn = lazy_relation_methods_mod.groupByBetaOn;
            pub const groupByWeightedDot = lazy_relation_methods_mod.groupByWeightedDot;
            pub const groupByWeightedDotOn = lazy_relation_methods_mod.groupByWeightedDotOn;
            pub const groupByWeightedCosineSimilarity = lazy_relation_methods_mod.groupByWeightedCosineSimilarity;
            pub const groupByWeightedCosineSimilarityOn = lazy_relation_methods_mod.groupByWeightedCosineSimilarityOn;
            pub const groupByWeightedCosine = lazy_relation_methods_mod.groupByWeightedCosine;
            pub const groupByWeightedCosineOn = lazy_relation_methods_mod.groupByWeightedCosineOn;
            pub const groupByWeightedSquaredEuclideanDistance = lazy_relation_methods_mod.groupByWeightedSquaredEuclideanDistance;
            pub const groupByWeightedSquaredEuclideanDistanceOn = lazy_relation_methods_mod.groupByWeightedSquaredEuclideanDistanceOn;
            pub const groupByWeightedEuclideanDistance = lazy_relation_methods_mod.groupByWeightedEuclideanDistance;
            pub const groupByWeightedEuclideanDistanceOn = lazy_relation_methods_mod.groupByWeightedEuclideanDistanceOn;
            pub const groupByWeightedManhattanDistance = lazy_relation_methods_mod.groupByWeightedManhattanDistance;
            pub const groupByWeightedManhattanDistanceOn = lazy_relation_methods_mod.groupByWeightedManhattanDistanceOn;
            pub const groupByWeightedChebyshevDistance = lazy_relation_methods_mod.groupByWeightedChebyshevDistance;
            pub const groupByWeightedChebyshevDistanceOn = lazy_relation_methods_mod.groupByWeightedChebyshevDistanceOn;
            pub const groupByWeightedCanberraDistance = lazy_relation_methods_mod.groupByWeightedCanberraDistance;
            pub const groupByWeightedCanberraDistanceOn = lazy_relation_methods_mod.groupByWeightedCanberraDistanceOn;
            pub const groupByWeightedBrayCurtisDistance = lazy_relation_methods_mod.groupByWeightedBrayCurtisDistance;
            pub const groupByWeightedBrayCurtisDistanceOn = lazy_relation_methods_mod.groupByWeightedBrayCurtisDistanceOn;
            pub const groupByWeightedMeanError = lazy_relation_methods_mod.groupByWeightedMeanError;
            pub const groupByWeightedMeanErrorOn = lazy_relation_methods_mod.groupByWeightedMeanErrorOn;
            pub const groupByWeightedBias = lazy_relation_methods_mod.groupByWeightedBias;
            pub const groupByWeightedBiasOn = lazy_relation_methods_mod.groupByWeightedBiasOn;
            pub const groupByWeightedMae = lazy_relation_methods_mod.groupByWeightedMae;
            pub const groupByWeightedMaeOn = lazy_relation_methods_mod.groupByWeightedMaeOn;
            pub const groupByWeightedMse = lazy_relation_methods_mod.groupByWeightedMse;
            pub const groupByWeightedMseOn = lazy_relation_methods_mod.groupByWeightedMseOn;
            pub const groupByWeightedRmse = lazy_relation_methods_mod.groupByWeightedRmse;
            pub const groupByWeightedRmseOn = lazy_relation_methods_mod.groupByWeightedRmseOn;
            pub const groupByWeightedMape = lazy_relation_methods_mod.groupByWeightedMape;
            pub const groupByWeightedMapeOn = lazy_relation_methods_mod.groupByWeightedMapeOn;
            pub const groupByWeightedSmape = lazy_relation_methods_mod.groupByWeightedSmape;
            pub const groupByWeightedSmapeOn = lazy_relation_methods_mod.groupByWeightedSmapeOn;
            pub const groupByWeightedCovariance = lazy_relation_methods_mod.groupByWeightedCovariance;
            pub const groupByWeightedCovarianceOn = lazy_relation_methods_mod.groupByWeightedCovarianceOn;
            pub const groupByWeightedCov = lazy_relation_methods_mod.groupByWeightedCov;
            pub const groupByWeightedCovOn = lazy_relation_methods_mod.groupByWeightedCovOn;
            pub const groupByWeightedCorrelation = lazy_relation_methods_mod.groupByWeightedCorrelation;
            pub const groupByWeightedCorrelationOn = lazy_relation_methods_mod.groupByWeightedCorrelationOn;
            pub const groupByWeightedCorr = lazy_relation_methods_mod.groupByWeightedCorr;
            pub const groupByWeightedCorrOn = lazy_relation_methods_mod.groupByWeightedCorrOn;
            pub const groupByWeightedBeta = lazy_relation_methods_mod.groupByWeightedBeta;
            pub const groupByWeightedBetaOn = lazy_relation_methods_mod.groupByWeightedBetaOn;
            pub const groupByMeanAbsDev = lazy_relation_methods_mod.groupByMeanAbsDev;
            pub const groupByMeanAbsDevOn = lazy_relation_methods_mod.groupByMeanAbsDevOn;
            pub const groupByMeanAbsDevRatio = lazy_relation_methods_mod.groupByMeanAbsDevRatio;
            pub const groupByMeanAbsDevRatioOn = lazy_relation_methods_mod.groupByMeanAbsDevRatioOn;
            pub const groupByMedian = lazy_relation_methods_mod.groupByMedian;
            pub const groupByMedianOn = lazy_relation_methods_mod.groupByMedianOn;
            pub const groupByQuantile = lazy_relation_methods_mod.groupByQuantile;
            pub const groupByQuantileOn = lazy_relation_methods_mod.groupByQuantileOn;
            pub const groupByIqr = lazy_relation_methods_mod.groupByIqr;
            pub const groupByIqrOn = lazy_relation_methods_mod.groupByIqrOn;
            pub const groupByIQR = lazy_relation_methods_mod.groupByIQR;
            pub const groupByIQROn = lazy_relation_methods_mod.groupByIQROn;
            pub const groupByMad = lazy_relation_methods_mod.groupByMad;
            pub const groupByMadOn = lazy_relation_methods_mod.groupByMadOn;
            pub const groupByMAD = lazy_relation_methods_mod.groupByMAD;
            pub const groupByMADOn = lazy_relation_methods_mod.groupByMADOn;
            pub const groupByMedianAbsDev = lazy_relation_methods_mod.groupByMedianAbsDev;
            pub const groupByMedianAbsDevOn = lazy_relation_methods_mod.groupByMedianAbsDevOn;
            pub const groupByTrimmedMean = lazy_relation_methods_mod.groupByTrimmedMean;
            pub const groupByTrimmedMeanOn = lazy_relation_methods_mod.groupByTrimmedMeanOn;
            pub const groupByWinsorizedMean = lazy_relation_methods_mod.groupByWinsorizedMean;
            pub const groupByWinsorizedMeanOn = lazy_relation_methods_mod.groupByWinsorizedMeanOn;
            pub const groupByInterdecileRange = lazy_relation_methods_mod.groupByInterdecileRange;
            pub const groupByInterdecileRangeOn = lazy_relation_methods_mod.groupByInterdecileRangeOn;
            pub const groupByIdr = lazy_relation_methods_mod.groupByIdr;
            pub const groupByIdrOn = lazy_relation_methods_mod.groupByIdrOn;
            pub const groupByIDR = lazy_relation_methods_mod.groupByIDR;
            pub const groupByIDROn = lazy_relation_methods_mod.groupByIDROn;
            pub const groupByMidhinge = lazy_relation_methods_mod.groupByMidhinge;
            pub const groupByMidhingeOn = lazy_relation_methods_mod.groupByMidhingeOn;
            pub const groupByTrimean = lazy_relation_methods_mod.groupByTrimean;
            pub const groupByTrimeanOn = lazy_relation_methods_mod.groupByTrimeanOn;
            pub const groupByBowleySkewness = lazy_relation_methods_mod.groupByBowleySkewness;
            pub const groupByBowleySkewnessOn = lazy_relation_methods_mod.groupByBowleySkewnessOn;
            pub const groupByBowleySkew = lazy_relation_methods_mod.groupByBowleySkew;
            pub const groupByBowleySkewOn = lazy_relation_methods_mod.groupByBowleySkewOn;
            pub const groupByQuartileCoeffDispersion = lazy_relation_methods_mod.groupByQuartileCoeffDispersion;
            pub const groupByQuartileCoeffDispersionOn = lazy_relation_methods_mod.groupByQuartileCoeffDispersionOn;
            pub const groupByQcd = lazy_relation_methods_mod.groupByQcd;
            pub const groupByQcdOn = lazy_relation_methods_mod.groupByQcdOn;
            pub const groupByKelleySkewness = lazy_relation_methods_mod.groupByKelleySkewness;
            pub const groupByKelleySkewnessOn = lazy_relation_methods_mod.groupByKelleySkewnessOn;
            pub const groupByKelleySkew = lazy_relation_methods_mod.groupByKelleySkew;
            pub const groupByKelleySkewOn = lazy_relation_methods_mod.groupByKelleySkewOn;
            pub const groupByVariance = lazy_relation_methods_mod.groupByVariance;
            pub const groupByVarianceOn = lazy_relation_methods_mod.groupByVarianceOn;
            pub const groupByStddev = lazy_relation_methods_mod.groupByStddev;
            pub const groupByStddevOn = lazy_relation_methods_mod.groupByStddevOn;
            pub const groupByStd = lazy_relation_methods_mod.groupByStd;
            pub const groupByStdOn = lazy_relation_methods_mod.groupByStdOn;
            pub const groupBySem = lazy_relation_methods_mod.groupBySem;
            pub const groupBySemOn = lazy_relation_methods_mod.groupBySemOn;
            pub const groupBySEM = lazy_relation_methods_mod.groupBySEM;
            pub const groupBySEMOn = lazy_relation_methods_mod.groupBySEMOn;
            pub const groupByCv = lazy_relation_methods_mod.groupByCv;
            pub const groupByCvOn = lazy_relation_methods_mod.groupByCvOn;
            pub const groupByCV = lazy_relation_methods_mod.groupByCV;
            pub const groupByCVOn = lazy_relation_methods_mod.groupByCVOn;
            pub const groupByFano = lazy_relation_methods_mod.groupByFano;
            pub const groupByFanoOn = lazy_relation_methods_mod.groupByFanoOn;
            pub const groupByIndexOfDispersion = lazy_relation_methods_mod.groupByIndexOfDispersion;
            pub const groupByIndexOfDispersionOn = lazy_relation_methods_mod.groupByIndexOfDispersionOn;
            pub const groupBySkewness = lazy_relation_methods_mod.groupBySkewness;
            pub const groupBySkewnessOn = lazy_relation_methods_mod.groupBySkewnessOn;
            pub const groupByKurtosis = lazy_relation_methods_mod.groupByKurtosis;
            pub const groupByKurtosisOn = lazy_relation_methods_mod.groupByKurtosisOn;
            pub const groupBySkew = lazy_relation_methods_mod.groupBySkew;
            pub const groupBySkewOn = lazy_relation_methods_mod.groupBySkewOn;
            pub const groupByKurt = lazy_relation_methods_mod.groupByKurt;
            pub const groupByKurtOn = lazy_relation_methods_mod.groupByKurtOn;
            pub const groupByMagnitudeVariance = lazy_relation_methods_mod.groupByMagnitudeVariance;
            pub const groupByMagnitudeVarianceOn = lazy_relation_methods_mod.groupByMagnitudeVarianceOn;
            pub const groupByAbsVariance = lazy_relation_methods_mod.groupByAbsVariance;
            pub const groupByAbsVarianceOn = lazy_relation_methods_mod.groupByAbsVarianceOn;
            pub const groupByMagnitudeVar = lazy_relation_methods_mod.groupByMagnitudeVar;
            pub const groupByMagnitudeVarOn = lazy_relation_methods_mod.groupByMagnitudeVarOn;
            pub const groupByAbsVar = lazy_relation_methods_mod.groupByAbsVar;
            pub const groupByAbsVarOn = lazy_relation_methods_mod.groupByAbsVarOn;
            pub const groupByMagnitudeStddev = lazy_relation_methods_mod.groupByMagnitudeStddev;
            pub const groupByMagnitudeStddevOn = lazy_relation_methods_mod.groupByMagnitudeStddevOn;
            pub const groupByAbsStddev = lazy_relation_methods_mod.groupByAbsStddev;
            pub const groupByAbsStddevOn = lazy_relation_methods_mod.groupByAbsStddevOn;
            pub const groupByMagnitudeStd = lazy_relation_methods_mod.groupByMagnitudeStd;
            pub const groupByMagnitudeStdOn = lazy_relation_methods_mod.groupByMagnitudeStdOn;
            pub const groupByAbsStd = lazy_relation_methods_mod.groupByAbsStd;
            pub const groupByAbsStdOn = lazy_relation_methods_mod.groupByAbsStdOn;
            pub const groupByMagnitudeSem = lazy_relation_methods_mod.groupByMagnitudeSem;
            pub const groupByMagnitudeSemOn = lazy_relation_methods_mod.groupByMagnitudeSemOn;
            pub const groupByAbsSem = lazy_relation_methods_mod.groupByAbsSem;
            pub const groupByAbsSemOn = lazy_relation_methods_mod.groupByAbsSemOn;
            pub const groupByMagnitudeCv = lazy_relation_methods_mod.groupByMagnitudeCv;
            pub const groupByMagnitudeCvOn = lazy_relation_methods_mod.groupByMagnitudeCvOn;
            pub const groupByAbsCv = lazy_relation_methods_mod.groupByAbsCv;
            pub const groupByAbsCvOn = lazy_relation_methods_mod.groupByAbsCvOn;
            pub const groupByAbsCV = lazy_relation_methods_mod.groupByAbsCV;
            pub const groupByAbsCVOn = lazy_relation_methods_mod.groupByAbsCVOn;
            pub const groupByMagnitudeFano = lazy_relation_methods_mod.groupByMagnitudeFano;
            pub const groupByMagnitudeFanoOn = lazy_relation_methods_mod.groupByMagnitudeFanoOn;
            pub const groupByAbsFano = lazy_relation_methods_mod.groupByAbsFano;
            pub const groupByAbsFanoOn = lazy_relation_methods_mod.groupByAbsFanoOn;
            pub const groupByMagnitudeIndexOfDispersion = lazy_relation_methods_mod.groupByMagnitudeIndexOfDispersion;
            pub const groupByMagnitudeIndexOfDispersionOn = lazy_relation_methods_mod.groupByMagnitudeIndexOfDispersionOn;
            pub const groupByAbsIndexOfDispersion = lazy_relation_methods_mod.groupByAbsIndexOfDispersion;
            pub const groupByAbsIndexOfDispersionOn = lazy_relation_methods_mod.groupByAbsIndexOfDispersionOn;
            pub const groupByMagnitudeSkewness = lazy_relation_methods_mod.groupByMagnitudeSkewness;
            pub const groupByMagnitudeSkewnessOn = lazy_relation_methods_mod.groupByMagnitudeSkewnessOn;
            pub const groupByAbsSkewness = lazy_relation_methods_mod.groupByAbsSkewness;
            pub const groupByAbsSkewnessOn = lazy_relation_methods_mod.groupByAbsSkewnessOn;
            pub const groupByMagnitudeSkew = lazy_relation_methods_mod.groupByMagnitudeSkew;
            pub const groupByMagnitudeSkewOn = lazy_relation_methods_mod.groupByMagnitudeSkewOn;
            pub const groupByAbsSkew = lazy_relation_methods_mod.groupByAbsSkew;
            pub const groupByAbsSkewOn = lazy_relation_methods_mod.groupByAbsSkewOn;
            pub const groupByMagnitudeKurtosis = lazy_relation_methods_mod.groupByMagnitudeKurtosis;
            pub const groupByMagnitudeKurtosisOn = lazy_relation_methods_mod.groupByMagnitudeKurtosisOn;
            pub const groupByAbsKurtosis = lazy_relation_methods_mod.groupByAbsKurtosis;
            pub const groupByAbsKurtosisOn = lazy_relation_methods_mod.groupByAbsKurtosisOn;
            pub const groupByMagnitudeKurt = lazy_relation_methods_mod.groupByMagnitudeKurt;
            pub const groupByMagnitudeKurtOn = lazy_relation_methods_mod.groupByMagnitudeKurtOn;
            pub const groupByAbsKurt = lazy_relation_methods_mod.groupByAbsKurt;
            pub const groupByAbsKurtOn = lazy_relation_methods_mod.groupByAbsKurtOn;
            pub const groupByMeanAbs = lazy_relation_methods_mod.groupByMeanAbs;
            pub const groupByMeanAbsOn = lazy_relation_methods_mod.groupByMeanAbsOn;
            pub const groupByMeanSquare = lazy_relation_methods_mod.groupByMeanSquare;
            pub const groupByMeanSquareOn = lazy_relation_methods_mod.groupByMeanSquareOn;
            pub const groupByMeanSq = lazy_relation_methods_mod.groupByMeanSq;
            pub const groupByMeanSqOn = lazy_relation_methods_mod.groupByMeanSqOn;
            pub const groupByRms = lazy_relation_methods_mod.groupByRms;
            pub const groupByRmsOn = lazy_relation_methods_mod.groupByRmsOn;
            pub const groupByRMS = lazy_relation_methods_mod.groupByRMS;
            pub const groupByRMSOn = lazy_relation_methods_mod.groupByRMSOn;
            pub const groupByL1Norm = lazy_relation_methods_mod.groupByL1Norm;
            pub const groupByL1NormOn = lazy_relation_methods_mod.groupByL1NormOn;
            pub const groupByL2Norm = lazy_relation_methods_mod.groupByL2Norm;
            pub const groupByL2NormOn = lazy_relation_methods_mod.groupByL2NormOn;
            pub const groupByMaxAbs = lazy_relation_methods_mod.groupByMaxAbs;
            pub const groupByMaxAbsOn = lazy_relation_methods_mod.groupByMaxAbsOn;
            pub const groupByMinAbs = lazy_relation_methods_mod.groupByMinAbs;
            pub const groupByMinAbsOn = lazy_relation_methods_mod.groupByMinAbsOn;
            pub const groupByHhi = lazy_relation_methods_mod.groupByHhi;
            pub const groupByHhiOn = lazy_relation_methods_mod.groupByHhiOn;
            pub const groupByHerfindahl = lazy_relation_methods_mod.groupByHerfindahl;
            pub const groupByHerfindahlOn = lazy_relation_methods_mod.groupByHerfindahlOn;
            pub const groupByHerfindahlHirschman = lazy_relation_methods_mod.groupByHerfindahlHirschman;
            pub const groupByHerfindahlHirschmanOn = lazy_relation_methods_mod.groupByHerfindahlHirschmanOn;
            pub const groupByMagnitudeNormalizedHhi = lazy_relation_methods_mod.groupByMagnitudeNormalizedHhi;
            pub const groupByMagnitudeNormalizedHhiOn = lazy_relation_methods_mod.groupByMagnitudeNormalizedHhiOn;
            pub const groupByAbsNormalizedHhi = lazy_relation_methods_mod.groupByAbsNormalizedHhi;
            pub const groupByAbsNormalizedHhiOn = lazy_relation_methods_mod.groupByAbsNormalizedHhiOn;
            pub const groupByMagnitudeSparsity = lazy_relation_methods_mod.groupByMagnitudeSparsity;
            pub const groupByMagnitudeSparsityOn = lazy_relation_methods_mod.groupByMagnitudeSparsityOn;
            pub const groupByAbsSparsity = lazy_relation_methods_mod.groupByAbsSparsity;
            pub const groupByAbsSparsityOn = lazy_relation_methods_mod.groupByAbsSparsityOn;
            pub const groupByMagnitudeInverseSimpson = lazy_relation_methods_mod.groupByMagnitudeInverseSimpson;
            pub const groupByMagnitudeInverseSimpsonOn = lazy_relation_methods_mod.groupByMagnitudeInverseSimpsonOn;
            pub const groupByAbsInverseSimpson = lazy_relation_methods_mod.groupByAbsInverseSimpson;
            pub const groupByAbsInverseSimpsonOn = lazy_relation_methods_mod.groupByAbsInverseSimpsonOn;
            pub const groupByMagnitudeSimpsonEvenness = lazy_relation_methods_mod.groupByMagnitudeSimpsonEvenness;
            pub const groupByMagnitudeSimpsonEvennessOn = lazy_relation_methods_mod.groupByMagnitudeSimpsonEvennessOn;
            pub const groupByAbsSimpsonEvenness = lazy_relation_methods_mod.groupByAbsSimpsonEvenness;
            pub const groupByAbsSimpsonEvennessOn = lazy_relation_methods_mod.groupByAbsSimpsonEvennessOn;
            pub const groupByMagnitudeDominance = lazy_relation_methods_mod.groupByMagnitudeDominance;
            pub const groupByMagnitudeDominanceOn = lazy_relation_methods_mod.groupByMagnitudeDominanceOn;
            pub const groupByAbsDominance = lazy_relation_methods_mod.groupByAbsDominance;
            pub const groupByAbsDominanceOn = lazy_relation_methods_mod.groupByAbsDominanceOn;
            pub const groupByMagnitudeDominanceMargin = lazy_relation_methods_mod.groupByMagnitudeDominanceMargin;
            pub const groupByMagnitudeDominanceMarginOn = lazy_relation_methods_mod.groupByMagnitudeDominanceMarginOn;
            pub const groupByAbsDominanceMargin = lazy_relation_methods_mod.groupByAbsDominanceMargin;
            pub const groupByAbsDominanceMarginOn = lazy_relation_methods_mod.groupByAbsDominanceMarginOn;
            pub const groupByMagnitudeEntropy = lazy_relation_methods_mod.groupByMagnitudeEntropy;
            pub const groupByMagnitudeEntropyOn = lazy_relation_methods_mod.groupByMagnitudeEntropyOn;
            pub const groupByAbsEntropy = lazy_relation_methods_mod.groupByAbsEntropy;
            pub const groupByAbsEntropyOn = lazy_relation_methods_mod.groupByAbsEntropyOn;
            pub const groupByMagnitudePerplexity = lazy_relation_methods_mod.groupByMagnitudePerplexity;
            pub const groupByMagnitudePerplexityOn = lazy_relation_methods_mod.groupByMagnitudePerplexityOn;
            pub const groupByAbsPerplexity = lazy_relation_methods_mod.groupByAbsPerplexity;
            pub const groupByAbsPerplexityOn = lazy_relation_methods_mod.groupByAbsPerplexityOn;
            pub const groupByMagnitudeEvenness = lazy_relation_methods_mod.groupByMagnitudeEvenness;
            pub const groupByMagnitudeEvennessOn = lazy_relation_methods_mod.groupByMagnitudeEvennessOn;
            pub const groupByAbsEvenness = lazy_relation_methods_mod.groupByAbsEvenness;
            pub const groupByAbsEvennessOn = lazy_relation_methods_mod.groupByAbsEvennessOn;
            pub const groupByGeometricMean = lazy_relation_methods_mod.groupByGeometricMean;
            pub const groupByGeometricMeanOn = lazy_relation_methods_mod.groupByGeometricMeanOn;
            pub const groupByGeoMean = lazy_relation_methods_mod.groupByGeoMean;
            pub const groupByGeoMeanOn = lazy_relation_methods_mod.groupByGeoMeanOn;
            pub const groupByHarmonicMean = lazy_relation_methods_mod.groupByHarmonicMean;
            pub const groupByHarmonicMeanOn = lazy_relation_methods_mod.groupByHarmonicMeanOn;
            pub const groupByLogSumExp = lazy_relation_methods_mod.groupByLogSumExp;
            pub const groupByLogSumExpOn = lazy_relation_methods_mod.groupByLogSumExpOn;
            pub const groupByLogsumexp = lazy_relation_methods_mod.groupByLogsumexp;
            pub const groupByLogsumexpOn = lazy_relation_methods_mod.groupByLogsumexpOn;
            pub const groupByLogMeanExp = lazy_relation_methods_mod.groupByLogMeanExp;
            pub const groupByLogMeanExpOn = lazy_relation_methods_mod.groupByLogMeanExpOn;
            pub const groupByLogmeanexp = lazy_relation_methods_mod.groupByLogmeanexp;
            pub const groupByLogmeanexpOn = lazy_relation_methods_mod.groupByLogmeanexpOn;
            pub const groupByPtp = lazy_relation_methods_mod.groupByPtp;
            pub const groupByPtpOn = lazy_relation_methods_mod.groupByPtpOn;
            pub const groupByPTP = lazy_relation_methods_mod.groupByPTP;
            pub const groupByPTPOn = lazy_relation_methods_mod.groupByPTPOn;
            pub const groupByPeakToPeak = lazy_relation_methods_mod.groupByPeakToPeak;
            pub const groupByPeakToPeakOn = lazy_relation_methods_mod.groupByPeakToPeakOn;
            pub const groupByMidrange = lazy_relation_methods_mod.groupByMidrange;
            pub const groupByMidrangeOn = lazy_relation_methods_mod.groupByMidrangeOn;
            pub const groupByRangeCoeff = lazy_relation_methods_mod.groupByRangeCoeff;
            pub const groupByRangeCoeffOn = lazy_relation_methods_mod.groupByRangeCoeffOn;
            pub const groupByRangeCoefficient = lazy_relation_methods_mod.groupByRangeCoefficient;
            pub const groupByRangeCoefficientOn = lazy_relation_methods_mod.groupByRangeCoefficientOn;
            pub const groupByAny = lazy_relation_methods_mod.groupByAny;
            pub const groupByAnyOn = lazy_relation_methods_mod.groupByAnyOn;
            pub const groupByAll = lazy_relation_methods_mod.groupByAll;
            pub const groupByAllOn = lazy_relation_methods_mod.groupByAllOn;
            pub const groupByTrueCount = lazy_relation_methods_mod.groupByTrueCount;
            pub const groupByTrueCountOn = lazy_relation_methods_mod.groupByTrueCountOn;
            pub const groupByFalseCount = lazy_relation_methods_mod.groupByFalseCount;
            pub const groupByFalseCountOn = lazy_relation_methods_mod.groupByFalseCountOn;
            pub const groupByTrueRatio = lazy_relation_methods_mod.groupByTrueRatio;
            pub const groupByTrueRatioOn = lazy_relation_methods_mod.groupByTrueRatioOn;
            pub const groupByFalseRatio = lazy_relation_methods_mod.groupByFalseRatio;
            pub const groupByFalseRatioOn = lazy_relation_methods_mod.groupByFalseRatioOn;
            pub const groupByFirstTrueIndex = lazy_relation_methods_mod.groupByFirstTrueIndex;
            pub const groupByFirstTrueIndexOn = lazy_relation_methods_mod.groupByFirstTrueIndexOn;
            pub const groupByLastTrueIndex = lazy_relation_methods_mod.groupByLastTrueIndex;
            pub const groupByLastTrueIndexOn = lazy_relation_methods_mod.groupByLastTrueIndexOn;
            pub const groupByFirstFalseIndex = lazy_relation_methods_mod.groupByFirstFalseIndex;
            pub const groupByFirstFalseIndexOn = lazy_relation_methods_mod.groupByFirstFalseIndexOn;
            pub const groupByLastFalseIndex = lazy_relation_methods_mod.groupByLastFalseIndex;
            pub const groupByLastFalseIndexOn = lazy_relation_methods_mod.groupByLastFalseIndexOn;
            pub const groupByAnyValid = lazy_relation_methods_mod.groupByAnyValid;
            pub const groupByAnyValidOn = lazy_relation_methods_mod.groupByAnyValidOn;
            pub const groupByAllValid = lazy_relation_methods_mod.groupByAllValid;
            pub const groupByAllValidOn = lazy_relation_methods_mod.groupByAllValidOn;
            pub const groupByAnyNull = lazy_relation_methods_mod.groupByAnyNull;
            pub const groupByAnyNullOn = lazy_relation_methods_mod.groupByAnyNullOn;
            pub const groupByAllNull = lazy_relation_methods_mod.groupByAllNull;
            pub const groupByAllNullOn = lazy_relation_methods_mod.groupByAllNullOn;
            pub const groupByValidCount = lazy_relation_methods_mod.groupByValidCount;
            pub const groupByValidCountOn = lazy_relation_methods_mod.groupByValidCountOn;
            pub const groupByNullCount = lazy_relation_methods_mod.groupByNullCount;
            pub const groupByNullCountOn = lazy_relation_methods_mod.groupByNullCountOn;
            pub const groupByValidRatio = lazy_relation_methods_mod.groupByValidRatio;
            pub const groupByValidRatioOn = lazy_relation_methods_mod.groupByValidRatioOn;
            pub const groupByNullRatio = lazy_relation_methods_mod.groupByNullRatio;
            pub const groupByNullRatioOn = lazy_relation_methods_mod.groupByNullRatioOn;
            pub const groupByFirstValidIndex = lazy_relation_methods_mod.groupByFirstValidIndex;
            pub const groupByFirstValidIndexOn = lazy_relation_methods_mod.groupByFirstValidIndexOn;
            pub const groupByLastValidIndex = lazy_relation_methods_mod.groupByLastValidIndex;
            pub const groupByLastValidIndexOn = lazy_relation_methods_mod.groupByLastValidIndexOn;
            pub const groupByFirstNullIndex = lazy_relation_methods_mod.groupByFirstNullIndex;
            pub const groupByFirstNullIndexOn = lazy_relation_methods_mod.groupByFirstNullIndexOn;
            pub const groupByLastNullIndex = lazy_relation_methods_mod.groupByLastNullIndex;
            pub const groupByLastNullIndexOn = lazy_relation_methods_mod.groupByLastNullIndexOn;
            pub const groupByNaNCount = lazy_relation_methods_mod.groupByNaNCount;
            pub const groupByNaNCountOn = lazy_relation_methods_mod.groupByNaNCountOn;
            pub const groupByNaNRatio = lazy_relation_methods_mod.groupByNaNRatio;
            pub const groupByNaNRatioOn = lazy_relation_methods_mod.groupByNaNRatioOn;
            pub const groupByInfCount = lazy_relation_methods_mod.groupByInfCount;
            pub const groupByInfCountOn = lazy_relation_methods_mod.groupByInfCountOn;
            pub const groupByInfRatio = lazy_relation_methods_mod.groupByInfRatio;
            pub const groupByInfRatioOn = lazy_relation_methods_mod.groupByInfRatioOn;
            pub const groupByPositiveInfCount = lazy_relation_methods_mod.groupByPositiveInfCount;
            pub const groupByPositiveInfCountOn = lazy_relation_methods_mod.groupByPositiveInfCountOn;
            pub const groupByPositiveInfRatio = lazy_relation_methods_mod.groupByPositiveInfRatio;
            pub const groupByPositiveInfRatioOn = lazy_relation_methods_mod.groupByPositiveInfRatioOn;
            pub const groupByNegativeInfCount = lazy_relation_methods_mod.groupByNegativeInfCount;
            pub const groupByNegativeInfCountOn = lazy_relation_methods_mod.groupByNegativeInfCountOn;
            pub const groupByNegativeInfRatio = lazy_relation_methods_mod.groupByNegativeInfRatio;
            pub const groupByNegativeInfRatioOn = lazy_relation_methods_mod.groupByNegativeInfRatioOn;
            pub const groupByFirstNaNIndex = lazy_relation_methods_mod.groupByFirstNaNIndex;
            pub const groupByFirstNaNIndexOn = lazy_relation_methods_mod.groupByFirstNaNIndexOn;
            pub const groupByLastNaNIndex = lazy_relation_methods_mod.groupByLastNaNIndex;
            pub const groupByLastNaNIndexOn = lazy_relation_methods_mod.groupByLastNaNIndexOn;
            pub const groupByFirstInfIndex = lazy_relation_methods_mod.groupByFirstInfIndex;
            pub const groupByFirstInfIndexOn = lazy_relation_methods_mod.groupByFirstInfIndexOn;
            pub const groupByLastInfIndex = lazy_relation_methods_mod.groupByLastInfIndex;
            pub const groupByLastInfIndexOn = lazy_relation_methods_mod.groupByLastInfIndexOn;
            pub const groupByFirstPositiveInfIndex = lazy_relation_methods_mod.groupByFirstPositiveInfIndex;
            pub const groupByFirstPositiveInfIndexOn = lazy_relation_methods_mod.groupByFirstPositiveInfIndexOn;
            pub const groupByLastPositiveInfIndex = lazy_relation_methods_mod.groupByLastPositiveInfIndex;
            pub const groupByLastPositiveInfIndexOn = lazy_relation_methods_mod.groupByLastPositiveInfIndexOn;
            pub const groupByFirstNegativeInfIndex = lazy_relation_methods_mod.groupByFirstNegativeInfIndex;
            pub const groupByFirstNegativeInfIndexOn = lazy_relation_methods_mod.groupByFirstNegativeInfIndexOn;
            pub const groupByLastNegativeInfIndex = lazy_relation_methods_mod.groupByLastNegativeInfIndex;
            pub const groupByLastNegativeInfIndexOn = lazy_relation_methods_mod.groupByLastNegativeInfIndexOn;
            pub const groupByFiniteCount = lazy_relation_methods_mod.groupByFiniteCount;
            pub const groupByFiniteCountOn = lazy_relation_methods_mod.groupByFiniteCountOn;
            pub const groupByFiniteRatio = lazy_relation_methods_mod.groupByFiniteRatio;
            pub const groupByFiniteRatioOn = lazy_relation_methods_mod.groupByFiniteRatioOn;
            pub const groupByFirstFiniteIndex = lazy_relation_methods_mod.groupByFirstFiniteIndex;
            pub const groupByFirstFiniteIndexOn = lazy_relation_methods_mod.groupByFirstFiniteIndexOn;
            pub const groupByLastFiniteIndex = lazy_relation_methods_mod.groupByLastFiniteIndex;
            pub const groupByLastFiniteIndexOn = lazy_relation_methods_mod.groupByLastFiniteIndexOn;
            pub const groupByNormalCount = lazy_relation_methods_mod.groupByNormalCount;
            pub const groupByNormalCountOn = lazy_relation_methods_mod.groupByNormalCountOn;
            pub const groupByNormalRatio = lazy_relation_methods_mod.groupByNormalRatio;
            pub const groupByNormalRatioOn = lazy_relation_methods_mod.groupByNormalRatioOn;
            pub const groupByFirstNormalIndex = lazy_relation_methods_mod.groupByFirstNormalIndex;
            pub const groupByFirstNormalIndexOn = lazy_relation_methods_mod.groupByFirstNormalIndexOn;
            pub const groupByLastNormalIndex = lazy_relation_methods_mod.groupByLastNormalIndex;
            pub const groupByLastNormalIndexOn = lazy_relation_methods_mod.groupByLastNormalIndexOn;
            pub const groupBySubnormalCount = lazy_relation_methods_mod.groupBySubnormalCount;
            pub const groupBySubnormalCountOn = lazy_relation_methods_mod.groupBySubnormalCountOn;
            pub const groupBySubnormalRatio = lazy_relation_methods_mod.groupBySubnormalRatio;
            pub const groupBySubnormalRatioOn = lazy_relation_methods_mod.groupBySubnormalRatioOn;
            pub const groupByFirstSubnormalIndex = lazy_relation_methods_mod.groupByFirstSubnormalIndex;
            pub const groupByFirstSubnormalIndexOn = lazy_relation_methods_mod.groupByFirstSubnormalIndexOn;
            pub const groupByLastSubnormalIndex = lazy_relation_methods_mod.groupByLastSubnormalIndex;
            pub const groupByLastSubnormalIndexOn = lazy_relation_methods_mod.groupByLastSubnormalIndexOn;
            pub const groupByNonFiniteCount = lazy_relation_methods_mod.groupByNonFiniteCount;
            pub const groupByNonFiniteCountOn = lazy_relation_methods_mod.groupByNonFiniteCountOn;
            pub const groupByNonFiniteRatio = lazy_relation_methods_mod.groupByNonFiniteRatio;
            pub const groupByNonFiniteRatioOn = lazy_relation_methods_mod.groupByNonFiniteRatioOn;
            pub const groupByFirstNonFiniteIndex = lazy_relation_methods_mod.groupByFirstNonFiniteIndex;
            pub const groupByFirstNonFiniteIndexOn = lazy_relation_methods_mod.groupByFirstNonFiniteIndexOn;
            pub const groupByLastNonFiniteIndex = lazy_relation_methods_mod.groupByLastNonFiniteIndex;
            pub const groupByLastNonFiniteIndexOn = lazy_relation_methods_mod.groupByLastNonFiniteIndexOn;
            pub const groupByZeroCount = lazy_relation_methods_mod.groupByZeroCount;
            pub const groupByZeroCountOn = lazy_relation_methods_mod.groupByZeroCountOn;
            pub const groupByZeroRatio = lazy_relation_methods_mod.groupByZeroRatio;
            pub const groupByZeroRatioOn = lazy_relation_methods_mod.groupByZeroRatioOn;
            pub const groupByFirstZeroIndex = lazy_relation_methods_mod.groupByFirstZeroIndex;
            pub const groupByFirstZeroIndexOn = lazy_relation_methods_mod.groupByFirstZeroIndexOn;
            pub const groupByLastZeroIndex = lazy_relation_methods_mod.groupByLastZeroIndex;
            pub const groupByLastZeroIndexOn = lazy_relation_methods_mod.groupByLastZeroIndexOn;
            pub const groupByPositiveZeroCount = lazy_relation_methods_mod.groupByPositiveZeroCount;
            pub const groupByPositiveZeroCountOn = lazy_relation_methods_mod.groupByPositiveZeroCountOn;
            pub const groupByPositiveZeroRatio = lazy_relation_methods_mod.groupByPositiveZeroRatio;
            pub const groupByPositiveZeroRatioOn = lazy_relation_methods_mod.groupByPositiveZeroRatioOn;
            pub const groupByNegativeZeroCount = lazy_relation_methods_mod.groupByNegativeZeroCount;
            pub const groupByNegativeZeroCountOn = lazy_relation_methods_mod.groupByNegativeZeroCountOn;
            pub const groupByNegativeZeroRatio = lazy_relation_methods_mod.groupByNegativeZeroRatio;
            pub const groupByNegativeZeroRatioOn = lazy_relation_methods_mod.groupByNegativeZeroRatioOn;
            pub const groupByFirstPositiveZeroIndex = lazy_relation_methods_mod.groupByFirstPositiveZeroIndex;
            pub const groupByFirstPositiveZeroIndexOn = lazy_relation_methods_mod.groupByFirstPositiveZeroIndexOn;
            pub const groupByLastPositiveZeroIndex = lazy_relation_methods_mod.groupByLastPositiveZeroIndex;
            pub const groupByLastPositiveZeroIndexOn = lazy_relation_methods_mod.groupByLastPositiveZeroIndexOn;
            pub const groupByFirstNegativeZeroIndex = lazy_relation_methods_mod.groupByFirstNegativeZeroIndex;
            pub const groupByFirstNegativeZeroIndexOn = lazy_relation_methods_mod.groupByFirstNegativeZeroIndexOn;
            pub const groupByLastNegativeZeroIndex = lazy_relation_methods_mod.groupByLastNegativeZeroIndex;
            pub const groupByLastNegativeZeroIndexOn = lazy_relation_methods_mod.groupByLastNegativeZeroIndexOn;
            pub const groupByNonZeroCount = lazy_relation_methods_mod.groupByNonZeroCount;
            pub const groupByNonZeroCountOn = lazy_relation_methods_mod.groupByNonZeroCountOn;
            pub const groupByNonZeroRatio = lazy_relation_methods_mod.groupByNonZeroRatio;
            pub const groupByNonZeroRatioOn = lazy_relation_methods_mod.groupByNonZeroRatioOn;
            pub const groupByFirstNonZeroIndex = lazy_relation_methods_mod.groupByFirstNonZeroIndex;
            pub const groupByFirstNonZeroIndexOn = lazy_relation_methods_mod.groupByFirstNonZeroIndexOn;
            pub const groupByLastNonZeroIndex = lazy_relation_methods_mod.groupByLastNonZeroIndex;
            pub const groupByLastNonZeroIndexOn = lazy_relation_methods_mod.groupByLastNonZeroIndexOn;
            pub const groupByPositiveCount = lazy_relation_methods_mod.groupByPositiveCount;
            pub const groupByPositiveCountOn = lazy_relation_methods_mod.groupByPositiveCountOn;
            pub const groupByPositiveRatio = lazy_relation_methods_mod.groupByPositiveRatio;
            pub const groupByPositiveRatioOn = lazy_relation_methods_mod.groupByPositiveRatioOn;
            pub const groupByFirstPositiveIndex = lazy_relation_methods_mod.groupByFirstPositiveIndex;
            pub const groupByFirstPositiveIndexOn = lazy_relation_methods_mod.groupByFirstPositiveIndexOn;
            pub const groupByLastPositiveIndex = lazy_relation_methods_mod.groupByLastPositiveIndex;
            pub const groupByLastPositiveIndexOn = lazy_relation_methods_mod.groupByLastPositiveIndexOn;
            pub const groupBySignBitCount = lazy_relation_methods_mod.groupBySignBitCount;
            pub const groupBySignBitCountOn = lazy_relation_methods_mod.groupBySignBitCountOn;
            pub const groupBySignBitRatio = lazy_relation_methods_mod.groupBySignBitRatio;
            pub const groupBySignBitRatioOn = lazy_relation_methods_mod.groupBySignBitRatioOn;
            pub const groupByFirstSignBitIndex = lazy_relation_methods_mod.groupByFirstSignBitIndex;
            pub const groupByFirstSignBitIndexOn = lazy_relation_methods_mod.groupByFirstSignBitIndexOn;
            pub const groupByLastSignBitIndex = lazy_relation_methods_mod.groupByLastSignBitIndex;
            pub const groupByLastSignBitIndexOn = lazy_relation_methods_mod.groupByLastSignBitIndexOn;
            pub const groupByNegativeCount = lazy_relation_methods_mod.groupByNegativeCount;
            pub const groupByNegativeCountOn = lazy_relation_methods_mod.groupByNegativeCountOn;
            pub const groupByNegativeRatio = lazy_relation_methods_mod.groupByNegativeRatio;
            pub const groupByNegativeRatioOn = lazy_relation_methods_mod.groupByNegativeRatioOn;
            pub const groupByFirstNegativeIndex = lazy_relation_methods_mod.groupByFirstNegativeIndex;
            pub const groupByFirstNegativeIndexOn = lazy_relation_methods_mod.groupByFirstNegativeIndexOn;
            pub const groupByLastNegativeIndex = lazy_relation_methods_mod.groupByLastNegativeIndex;
            pub const groupByLastNegativeIndexOn = lazy_relation_methods_mod.groupByLastNegativeIndexOn;
            pub const groupByArgMin = lazy_relation_methods_mod.groupByArgMin;
            pub const groupByArgMinOn = lazy_relation_methods_mod.groupByArgMinOn;
            pub const groupByArgMax = lazy_relation_methods_mod.groupByArgMax;
            pub const groupByArgMaxOn = lazy_relation_methods_mod.groupByArgMaxOn;
            pub const groupByArgmin = lazy_relation_methods_mod.groupByArgmin;
            pub const groupByArgminOn = lazy_relation_methods_mod.groupByArgminOn;
            pub const groupByArgmax = lazy_relation_methods_mod.groupByArgmax;
            pub const groupByArgmaxOn = lazy_relation_methods_mod.groupByArgmaxOn;
            pub const groupByStats = lazy_relation_methods_mod.groupByStats;
            pub const groupByStatsOn = lazy_relation_methods_mod.groupByStatsOn;
            pub const groupByProfile = lazy_relation_methods_mod.groupByProfile;
            pub const groupByProfileOn = lazy_relation_methods_mod.groupByProfileOn;
            pub const joinOn = lazy_relation_methods_mod.joinOn;
            pub const innerJoinOn = lazy_relation_methods_mod.innerJoinOn;
            pub const leftJoinOn = lazy_relation_methods_mod.leftJoinOn;
            pub const fullJoinOn = lazy_relation_methods_mod.fullJoinOn;
            pub const semiJoinOn = lazy_relation_methods_mod.semiJoinOn;
            pub const antiJoinOn = lazy_relation_methods_mod.antiJoinOn;
            pub const asofJoin = lazy_relation_methods_mod.asofJoin;
            pub const concatRows = lazy_relation_methods_mod.concatRows;
            pub const appendRows = lazy_relation_methods_mod.appendRows;
            pub const vstack = lazy_relation_methods_mod.vstack;
            pub const concatColumns = lazy_relation_methods_mod.concatColumns;
            pub const appendColumns = lazy_relation_methods_mod.appendColumns;
            pub const hstack = lazy_relation_methods_mod.hstack;
            pub const distinctRows = lazy_relation_methods_mod.distinctRows;
            pub const distinctRowsLast = lazy_relation_methods_mod.distinctRowsLast;
            pub const distinctRowsNone = lazy_relation_methods_mod.distinctRowsNone;
            pub const distinctOn = lazy_relation_methods_mod.distinctOn;
            pub const distinctOnLast = lazy_relation_methods_mod.distinctOnLast;
            pub const distinctOnNone = lazy_relation_methods_mod.distinctOnNone;
            pub const dropDuplicates = lazy_relation_methods_mod.dropDuplicates;
            pub const dropDuplicatesOn = lazy_relation_methods_mod.dropDuplicatesOn;
            pub const dropDuplicatesLast = lazy_relation_methods_mod.dropDuplicatesLast;
            pub const dropDuplicatesOnLast = lazy_relation_methods_mod.dropDuplicatesOnLast;
            pub const dropDuplicatesNone = lazy_relation_methods_mod.dropDuplicatesNone;
            pub const dropDuplicatesOnNone = lazy_relation_methods_mod.dropDuplicatesOnNone;
            pub const uniqueRows = lazy_relation_methods_mod.uniqueRows;
            pub fn filterColumnScalar(self: *DeviceLazyFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.filterColumnScalar(self, name, T, scalar, op);
            }

            pub fn filterColumnScalarWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.filterColumnScalarWithDeviceScalar(self, name, scalar, op);
            }

            pub fn dropColumnScalar(self: *DeviceLazyFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.dropColumnScalar(self, name, T, scalar, op);
            }

            pub fn dropColumnScalarWithDeviceScalar(self: *DeviceLazyFrame, name: []const u8, scalar: DeviceScalar, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.dropColumnScalarWithDeviceScalar(self, name, scalar, op);
            }

            pub fn sortBy(self: *DeviceLazyFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
                return lazy_sort_mod.sortBy(self, name, options_value);
            }

            pub fn sortByColumns(self: *DeviceLazyFrame, names: []const []const u8, options_values: []const DeviceSortOptions) DeviceDataError!void {
                return lazy_sort_mod.sortByColumns(self, names, options_values);
            }

            pub fn topKByColumns(self: *DeviceLazyFrame, names: []const []const u8, k: usize, options_values: []const DeviceSortOptions) DeviceDataError!void {
                return lazy_sort_mod.topKByColumns(self, names, k, options_values);
            }

            pub fn topKBy(self: *DeviceLazyFrame, name: []const u8, k: usize, options_value: DeviceSortOptions) DeviceDataError!void {
                return lazy_sort_mod.topKBy(self, name, k, options_value);
            }

            pub fn bottomKBy(self: *DeviceLazyFrame, name: []const u8, k: usize, options_value: DeviceSortOptions) DeviceDataError!void {
                return self.topKBy(name, k, options_value);
            }

            pub fn bottomKByColumns(self: *DeviceLazyFrame, names: []const []const u8, k: usize, options_values: []const DeviceSortOptions) DeviceDataError!void {
                return self.topKByColumns(names, k, options_values);
            }

            pub fn rankProfileBy(self: *DeviceLazyFrame, name: []const u8, output_prefix: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
                return lazy_sort_mod.rankProfileBy(self, name, output_prefix, options_value);
            }

            pub const rollingProfile = lazy_profile_methods_mod.rollingProfile;
            pub const rollingMomentProfile = lazy_profile_methods_mod.rollingMomentProfile;
            pub const rollingRangeProfile = lazy_profile_methods_mod.rollingRangeProfile;
            pub const rollingNormalizeProfile = lazy_profile_methods_mod.rollingNormalizeProfile;
            pub const expandingNormalizeProfile = lazy_profile_methods_mod.expandingNormalizeProfile;
            pub const rollingQuantileProfile = lazy_profile_methods_mod.rollingQuantileProfile;
            pub const expandingQuantileProfile = lazy_profile_methods_mod.expandingQuantileProfile;
            pub const rollingBoolProfile = lazy_profile_methods_mod.rollingBoolProfile;
            pub const rollingDrawdownProfile = lazy_profile_methods_mod.rollingDrawdownProfile;
            pub const rollingRobustProfile = lazy_profile_methods_mod.rollingRobustProfile;
            pub const rollingRankProfile = lazy_profile_methods_mod.rollingRankProfile;
            pub const lagProfile = lazy_profile_methods_mod.lagProfile;
            pub const leadProfile = lazy_profile_methods_mod.leadProfile;
            pub const clipProfile = lazy_profile_methods_mod.clipProfile;
            pub const rollingClipProfile = lazy_profile_methods_mod.rollingClipProfile;
            pub const expandingClipProfile = lazy_profile_methods_mod.expandingClipProfile;
            pub const thresholdProfile = lazy_profile_methods_mod.thresholdProfile;
            pub const rollingThresholdProfile = lazy_profile_methods_mod.rollingThresholdProfile;
            pub const expandingThresholdProfile = lazy_profile_methods_mod.expandingThresholdProfile;
            pub const expandingProfile = lazy_profile_methods_mod.expandingProfile;
            pub const expandingBoolProfile = lazy_profile_methods_mod.expandingBoolProfile;
            pub const expandingRankProfile = lazy_profile_methods_mod.expandingRankProfile;
            pub const expandingRobustProfile = lazy_profile_methods_mod.expandingRobustProfile;
            pub const expandingMomentProfile = lazy_profile_methods_mod.expandingMomentProfile;
            pub const standardizeProfile = lazy_profile_methods_mod.standardizeProfile;
            pub const robustProfile = lazy_profile_methods_mod.robustProfile;
            pub const drawdownProfile = lazy_profile_methods_mod.drawdownProfile;
            pub const extremaProfile = lazy_profile_methods_mod.extremaProfile;
            pub const trendProfile = lazy_profile_methods_mod.trendProfile;
            pub const rollingTrendProfile = lazy_profile_methods_mod.rollingTrendProfile;
            pub const expandingTrendProfile = lazy_profile_methods_mod.expandingTrendProfile;
            pub const changePointProfile = lazy_profile_methods_mod.changePointProfile;
            pub const rollingChangePointProfile = lazy_profile_methods_mod.rollingChangePointProfile;
            pub const expandingChangePointProfile = lazy_profile_methods_mod.expandingChangePointProfile;
            pub const signProfile = lazy_profile_methods_mod.signProfile;
            pub const rollingSignProfile = lazy_profile_methods_mod.rollingSignProfile;
            pub const expandingSignProfile = lazy_profile_methods_mod.expandingSignProfile;
            pub const crossoverProfile = lazy_profile_methods_mod.crossoverProfile;
            pub const rollingCrossoverProfile = lazy_profile_methods_mod.rollingCrossoverProfile;
            pub const expandingCrossoverProfile = lazy_profile_methods_mod.expandingCrossoverProfile;
            pub const bucketProfile = lazy_profile_methods_mod.bucketProfile;
            pub const emaProfile = lazy_profile_methods_mod.emaProfile;
            pub const linearFitProfile = lazy_profile_methods_mod.linearFitProfile;
            pub const errorProfile = lazy_profile_methods_mod.errorProfile;
            pub const rollingErrorProfile = lazy_profile_methods_mod.rollingErrorProfile;
            pub const expandingErrorProfile = lazy_profile_methods_mod.expandingErrorProfile;
            pub const classificationProfile = lazy_profile_methods_mod.classificationProfile;
            pub const rollingClassificationProfile = lazy_profile_methods_mod.rollingClassificationProfile;
            pub const expandingClassificationProfile = lazy_profile_methods_mod.expandingClassificationProfile;
            pub const boolTransitionProfile = lazy_profile_methods_mod.boolTransitionProfile;
            pub const rollingBoolTransitionProfile = lazy_profile_methods_mod.rollingBoolTransitionProfile;
            pub const expandingBoolTransitionProfile = lazy_profile_methods_mod.expandingBoolTransitionProfile;
            pub const rollingCorrelationProfile = lazy_profile_methods_mod.rollingCorrelationProfile;
            pub const expandingCorrelationProfile = lazy_profile_methods_mod.expandingCorrelationProfile;
            pub const expandingLinearFitProfile = lazy_profile_methods_mod.expandingLinearFitProfile;
            pub const rollingLinearFitProfile = lazy_profile_methods_mod.rollingLinearFitProfile;
            pub const validityProfile = lazy_profile_methods_mod.validityProfile;
            pub const rollingValidityProfile = lazy_profile_methods_mod.rollingValidityProfile;
            pub const expandingValidityProfile = lazy_profile_methods_mod.expandingValidityProfile;
            pub fn sliceRows(self: *DeviceLazyFrame, start: usize, stop: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .slice_rows = .{
                    .start = start,
                    .stop = stop,
                } });
            }

            pub fn sliceRowsLen(self: *DeviceLazyFrame, start: usize, length: usize) DeviceDataError!void {
                const stop = std.math.add(usize, start, length) catch return error.InvalidShape;
                return self.sliceRows(start, stop);
            }

            pub fn offset(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                return self.sliceRows(n, std.math.maxInt(usize));
            }

            pub fn sliceRowsSigned(self: *DeviceLazyFrame, start: isize, length: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .slice_rows_signed = .{
                    .start = start,
                    .length = length,
                } });
            }

            pub fn sliceSigned(self: *DeviceLazyFrame, start: isize, length: usize) DeviceDataError!void {
                return self.sliceRowsSigned(start, length);
            }

            pub fn dropRows(self: *DeviceLazyFrame, row_indices: []const usize) DeviceDataError!void {
                const owned = try self.allocator.dupe(usize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .drop_rows = owned });
            }

            pub fn dropRowsMode(self: *DeviceLazyFrame, row_indices: []const usize, mode: array_mod.IndexMode) DeviceDataError!void {
                const owned = try self.allocator.dupe(usize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .drop_rows_mode = .{
                    .row_indices = owned,
                    .mode = mode,
                } });
            }

            pub fn dropRowsSigned(self: *DeviceLazyFrame, row_indices: []const isize) DeviceDataError!void {
                const owned = try self.allocator.dupe(isize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .drop_rows_signed = owned });
            }

            pub fn dropRowsSignedMode(self: *DeviceLazyFrame, row_indices: []const isize, mode: array_mod.IndexMode) DeviceDataError!void {
                const owned = try self.allocator.dupe(isize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .drop_rows_signed_mode = .{
                    .row_indices = owned,
                    .mode = mode,
                } });
            }

            pub fn dropRowRange(self: *DeviceLazyFrame, start: usize, stop: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_row_range = .{
                    .start = start,
                    .stop = stop,
                } });
            }

            pub fn dropFirstRows(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                return self.dropRowRange(0, n);
            }

            pub fn dropLastRows(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .drop_last_rows = n });
            }

            pub fn slice(self: *DeviceLazyFrame, start: usize, len: usize) DeviceDataError!void {
                const stop = std.math.add(usize, start, len) catch return error.InvalidShape;
                return self.sliceRows(start, stop);
            }

            pub fn sliceRowsStep(self: *DeviceLazyFrame, start: usize, stop: usize, step: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .slice_rows_step = .{
                    .start = start,
                    .stop = stop,
                    .step = step,
                } });
            }

            pub fn sliceRowsSignedStep(self: *DeviceLazyFrame, start: isize, stop: isize, step: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .slice_rows_signed_step = .{
                    .start = start,
                    .stop = stop,
                    .step = step,
                } });
            }

            pub fn sliceSignedStep(self: *DeviceLazyFrame, start: isize, stop: isize, step: usize) DeviceDataError!void {
                return self.sliceRowsSignedStep(start, stop, step);
            }

            pub fn sliceStep(self: *DeviceLazyFrame, start: usize, len: usize, step: usize) DeviceDataError!void {
                const stop = std.math.add(usize, start, len) catch return error.InvalidShape;
                return self.sliceRowsStep(start, stop, step);
            }

            pub fn take(self: *DeviceLazyFrame, row_indices: []const usize) DeviceDataError!void {
                const owned = try self.allocator.dupe(usize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .take_rows = owned });
            }

            pub fn takeOptional(self: *DeviceLazyFrame, row_indices: []const ?usize) DeviceDataError!void {
                const owned = try self.allocator.dupe(?usize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .take_rows_optional = owned });
            }

            pub fn takeOptionalRows(self: *DeviceLazyFrame, row_indices: []const ?usize) DeviceDataError!void {
                return self.takeOptional(row_indices);
            }

            pub fn takeMode(self: *DeviceLazyFrame, row_indices: []const usize, mode: array_mod.IndexMode) DeviceDataError!void {
                const owned = try self.allocator.dupe(usize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .take_rows_mode = .{
                    .row_indices = owned,
                    .mode = mode,
                } });
            }

            pub fn takeSigned(self: *DeviceLazyFrame, row_indices: []const isize) DeviceDataError!void {
                const owned = try self.allocator.dupe(isize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .take_rows_signed = owned });
            }

            pub fn takeSignedMode(self: *DeviceLazyFrame, row_indices: []const isize, mode: array_mod.IndexMode) DeviceDataError!void {
                const owned = try self.allocator.dupe(isize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .take_rows_signed_mode = .{
                    .row_indices = owned,
                    .mode = mode,
                } });
            }

            pub fn takeByColumn(self: *DeviceLazyFrame, index_name: []const u8) DeviceDataError!void {
                const owned = try self.allocator.dupe(u8, index_name);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .take_rows_by_column = owned });
            }

            pub fn takeByColumnMode(self: *DeviceLazyFrame, index_name: []const u8, mode: array_mod.IndexMode) DeviceDataError!void {
                const owned = try self.allocator.dupe(u8, index_name);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .take_rows_by_column_mode = .{
                    .name = owned,
                    .mode = mode,
                } });
            }

            pub fn takeRowsByColumn(self: *DeviceLazyFrame, index_name: []const u8) DeviceDataError!void {
                return self.takeByColumn(index_name);
            }

            pub fn takeRowsByColumnMode(self: *DeviceLazyFrame, index_name: []const u8, mode: array_mod.IndexMode) DeviceDataError!void {
                return self.takeByColumnMode(index_name, mode);
            }

            pub fn dropRowsByColumn(self: *DeviceLazyFrame, index_name: []const u8) DeviceDataError!void {
                const owned = try self.allocator.dupe(u8, index_name);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .drop_rows_by_column = owned });
            }

            pub fn dropRowsByColumnMode(self: *DeviceLazyFrame, index_name: []const u8, mode: array_mod.IndexMode) DeviceDataError!void {
                const owned = try self.allocator.dupe(u8, index_name);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .drop_rows_by_column_mode = .{
                    .name = owned,
                    .mode = mode,
                } });
            }

            pub fn repeatRows(self: *DeviceLazyFrame, repeat_count: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .repeat_rows = repeat_count });
            }

            pub fn tileRows(self: *DeviceLazyFrame, tile_count: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .tile_rows = tile_count });
            }

            pub fn repeatRowsByColumn(self: *DeviceLazyFrame, count_name: []const u8) DeviceDataError!void {
                const owned = try self.allocator.dupe(u8, count_name);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .repeat_rows_by = owned });
            }

            pub fn sampleRows(self: *DeviceLazyFrame, count: usize, seed: u64) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .sample_rows = .{
                    .count = count,
                    .seed = seed,
                } });
            }

            pub fn shuffleRows(self: *DeviceLazyFrame, seed: u64) DeviceDataError!void {
                return self.sampleRowsFraction(1.0, seed);
            }

            pub fn sampleRowsFraction(self: *DeviceLazyFrame, fraction: f64, seed: u64) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .sample_rows_fraction = .{
                    .fraction = fraction,
                    .seed = seed,
                } });
            }

            pub fn sampleFrac(self: *DeviceLazyFrame, fraction: f64, seed: u64) DeviceDataError!void {
                return self.sampleRowsFraction(fraction, seed);
            }

            pub fn sampleRowsWithReplacement(self: *DeviceLazyFrame, count: usize, seed: u64) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .sample_rows_with_replacement = .{
                    .count = count,
                    .seed = seed,
                } });
            }

            pub fn sampleRowsFractionWithReplacement(self: *DeviceLazyFrame, fraction: f64, seed: u64) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .sample_rows_fraction_with_replacement = .{
                    .fraction = fraction,
                    .seed = seed,
                } });
            }

            pub fn sampleFracWithReplacement(self: *DeviceLazyFrame, fraction: f64, seed: u64) DeviceDataError!void {
                return self.sampleRowsFractionWithReplacement(fraction, seed);
            }

            pub fn strideRows(self: *DeviceLazyFrame, start: usize, step: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .stride_rows = .{
                    .start = start,
                    .step = step,
                } });
            }

            pub fn reverseRows(self: *DeviceLazyFrame) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .reverse_rows = {} });
            }

            pub fn rollRows(self: *DeviceLazyFrame, shift: isize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .roll_rows = shift });
            }

            pub fn shiftRows(self: *DeviceLazyFrame, shift: isize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .shift_rows = shift });
            }

            pub fn reverse(self: *DeviceLazyFrame) DeviceDataError!void {
                return self.reverseRows();
            }

            pub fn head(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .head = n });
            }

            pub fn limit(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                return self.head(n);
            }

            pub fn firstRow(self: *DeviceLazyFrame) DeviceDataError!void {
                return self.head(1);
            }

            pub fn tail(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .tail = n });
            }

            pub fn lastRow(self: *DeviceLazyFrame) DeviceDataError!void {
                return self.tail(1);
            }

            pub fn collect(self: DeviceLazyFrame) ParquetInteropError!DeviceDataFrame {
                return lazy_exec_mod.collect(DeviceDataFrame, DeviceLazyOp, self);
            }

            pub fn explain(self: DeviceLazyFrame, allocator: std.mem.Allocator) DeviceDataError![]u8 {
                return lazy_exec_mod.explain(DeviceLazyOp, self, allocator);
            }
        };

        fn deinitLazyOps(allocator: std.mem.Allocator, ops: *std.ArrayList(DeviceLazyOp)) void {
            for (ops.items) |*op| op.deinit(allocator);
            ops.deinit(allocator);
        }
    };
}
