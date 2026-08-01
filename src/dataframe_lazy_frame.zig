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
