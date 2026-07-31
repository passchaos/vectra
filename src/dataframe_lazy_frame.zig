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

            pub fn dropByNamePrefix(self: *DeviceLazyFrame, prefix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropByNamePrefix(self, prefix);
            }

            pub fn dropByNameSuffix(self: *DeviceLazyFrame, suffix: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropByNameSuffix(self, suffix);
            }

            pub fn dropByNameContains(self: *DeviceLazyFrame, needle: []const u8) DeviceDataError!void {
                return lazy_expr_mod.dropByNameContains(self, needle);
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

            pub fn withRowIndex(self: *DeviceLazyFrame, name: []const u8, offset: usize) DeviceDataError!void {
                return lazy_expr_mod.withRowIndex(self, name, offset);
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

            pub fn withColumnSqrt(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnSqrt(self, name, input_name);
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

            pub fn withColumnLog(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLog(self, name, input_name);
            }

            pub fn withColumnLog1p(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withColumnLog1p(self, name, input_name);
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

            pub fn withRowNaNCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNaNCount(self, names, output_name);
            }

            pub fn withRowInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowInfCount(self, names, output_name);
            }

            pub fn withRowPositiveInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPositiveInfCount(self, names, output_name);
            }

            pub fn withRowNegativeInfCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNegativeInfCount(self, names, output_name);
            }

            pub fn withRowZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowZeroCount(self, names, output_name);
            }

            pub fn withRowPositiveZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPositiveZeroCount(self, names, output_name);
            }

            pub fn withRowNegativeZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNegativeZeroCount(self, names, output_name);
            }

            pub fn withRowNonZeroCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNonZeroCount(self, names, output_name);
            }

            pub fn withRowPositiveCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowPositiveCount(self, names, output_name);
            }

            pub fn withRowSignBitCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSignBitCount(self, names, output_name);
            }

            pub fn withRowNegativeCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNegativeCount(self, names, output_name);
            }

            pub fn withRowFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowFiniteCount(self, names, output_name);
            }

            pub fn withRowNormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNormalCount(self, names, output_name);
            }

            pub fn withRowSubnormalCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowSubnormalCount(self, names, output_name);
            }

            pub fn withRowNonFiniteCount(self: *DeviceLazyFrame, names: []const []const u8, output_name: []const u8) DeviceDataError!void {
                return lazy_expr_mod.withRowNonFiniteCount(self, names, output_name);
            }

            pub fn withColumnCompare(self: *DeviceLazyFrame, name: []const u8, lhs_name: []const u8, rhs_name: []const u8, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnCompare(self, name, lhs_name, rhs_name, op);
            }

            pub fn withColumnCompareScalar(self: *DeviceLazyFrame, name: []const u8, input_name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.withColumnCompareScalar(self, name, input_name, T, scalar, op);
            }

            pub const groupByCount = lazy_relation_methods_mod.groupByCount;
            pub const groupByValue = lazy_relation_methods_mod.groupByValue;
            pub const groupBySum = lazy_relation_methods_mod.groupBySum;
            pub const groupByMin = lazy_relation_methods_mod.groupByMin;
            pub const groupByMax = lazy_relation_methods_mod.groupByMax;
            pub const groupByMean = lazy_relation_methods_mod.groupByMean;
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
            pub const distinctRows = lazy_relation_methods_mod.distinctRows;
            pub const distinctOn = lazy_relation_methods_mod.distinctOn;
            pub const dropDuplicates = lazy_relation_methods_mod.dropDuplicates;
            pub const dropDuplicatesOn = lazy_relation_methods_mod.dropDuplicatesOn;
            pub const uniqueRows = lazy_relation_methods_mod.uniqueRows;
            pub fn filterColumnScalar(self: *DeviceLazyFrame, name: []const u8, comptime T: type, scalar: T, op: DeviceColumnCompareOp) DeviceDataError!void {
                return lazy_expr_mod.filterColumnScalar(self, name, T, scalar, op);
            }

            pub fn sortBy(self: *DeviceLazyFrame, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
                return lazy_sort_mod.sortBy(self, name, options_value);
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

            pub fn dropRows(self: *DeviceLazyFrame, row_indices: []const usize) DeviceDataError!void {
                const owned = try self.allocator.dupe(usize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .drop_rows = owned });
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

            pub fn sliceStep(self: *DeviceLazyFrame, start: usize, len: usize, step: usize) DeviceDataError!void {
                const stop = std.math.add(usize, start, len) catch return error.InvalidShape;
                return self.sliceRowsStep(start, stop, step);
            }

            pub fn take(self: *DeviceLazyFrame, row_indices: []const usize) DeviceDataError!void {
                const owned = try self.allocator.dupe(usize, row_indices);
                errdefer self.allocator.free(owned);
                try self.ops.append(self.allocator, .{ .take_rows = owned });
            }

            pub fn sampleRows(self: *DeviceLazyFrame, count: usize, seed: u64) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .sample_rows = .{
                    .count = count,
                    .seed = seed,
                } });
            }

            pub fn sampleRowsWithReplacement(self: *DeviceLazyFrame, count: usize, seed: u64) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .sample_rows_with_replacement = .{
                    .count = count,
                    .seed = seed,
                } });
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

            pub fn reverse(self: *DeviceLazyFrame) DeviceDataError!void {
                return self.reverseRows();
            }

            pub fn head(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .head = n });
            }

            pub fn tail(self: *DeviceLazyFrame, n: usize) DeviceDataError!void {
                try self.ops.append(self.allocator, .{ .tail = n });
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
