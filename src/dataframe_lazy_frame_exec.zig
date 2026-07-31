//! Execution and optimization helpers for DeviceLazyFrame.
//!
//! This module owns collect/explain/optimization so dataframe_lazy_frame.zig
//! can focus on public lazy-plan construction methods.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_arrow_mod = @import("dataframe_arrow.zig");
const lazy_mod = @import("dataframe_lazy.zig");
const names_mod = @import("dataframe_names.zig");
const series_mod = @import("series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
pub const ParquetInteropError = dataframe_arrow_mod.ParquetInteropError;
const allNamesIn = names_mod.allNamesIn;
const planLazyScanPushdown = lazy_mod.planLazyScanPushdown;
const formatLazyScanPushdown = lazy_mod.formatLazyScanPushdown;
const formatLazyOp = lazy_mod.formatLazyOp;

// Let Zig infer the concrete ArrayList type at each call site.  Naming
// `std.ArrayList(DeviceLazyOp)` here forces the compiler to re-evaluate the
// large lazy-op union in this helper signature, which can exceed Zig 0.16's
// default comptime branch quota as new lazy expression tags are added.
fn deinitLazyOps(allocator: std.mem.Allocator, ops: anytype) void {
    for (ops.items) |*op| op.deinit(allocator);
    ops.deinit(allocator);
}

pub fn collect(comptime DeviceDataFrame: type, comptime DeviceLazyOp: type, self: anytype) ParquetInteropError!DeviceDataFrame {
    var optimized = try optimizedOps(DeviceLazyOp, self);
    defer deinitLazyOps(self.allocator, &optimized);
    var current = try collectSource(DeviceDataFrame, DeviceLazyOp, self, optimized.items);
    errdefer current.deinit();
    for (optimized.items) |op| {
        const next = switch (op) {
            .select => |names| try current.select(names),
            .select_column_indices => |indices| try current.selectByColumnIndices(indices),
            .select_column_range => |range| try current.selectColumnRange(range.start, range.stop),
            .select_last_columns => |n| try current.selectLastColumns(n),
            .drop_column_indices => |indices| try current.dropByColumnIndices(indices),
            .drop_column_range => |range| try current.dropColumnRange(range.start, range.stop),
            .drop_last_columns => |n| try current.dropLastColumns(n),
            .reverse_columns => try current.reverseColumns(),
            .sort_columns_by_name => |sort| try current.sortColumnsByName(sort.descending),
            .select_name_prefix => |pattern| try current.selectByNamePrefix(pattern.pattern),
            .select_name_suffix => |pattern| try current.selectByNameSuffix(pattern.pattern),
            .select_name_contains => |pattern| try current.selectByNameContains(pattern.pattern),
            .drop_name_prefix => |pattern| try current.dropByNamePrefix(pattern.pattern),
            .drop_name_suffix => |pattern| try current.dropByNameSuffix(pattern.pattern),
            .drop_name_contains => |pattern| try current.dropByNameContains(pattern.pattern),
            .select_dtypes => |dtypes| try current.selectByDTypes(dtypes),
            .select_dtype_class => |class| try current.selectByDTypeClass(class),
            .drop_dtypes => |dtypes| try current.dropByDTypes(dtypes),
            .drop_dtype_class => |class| try current.dropByDTypeClass(class),
            .select_nullable_columns => try current.selectNullableColumns(),
            .select_non_nullable_columns => try current.selectNonNullableColumns(),
            .select_columns_with_nulls => try current.selectColumnsWithNulls(),
            .select_columns_without_nulls => try current.selectColumnsWithoutNulls(),
            .drop_nullable_columns => try current.dropNullableColumns(),
            .drop_non_nullable_columns => try current.dropNonNullableColumns(),
            .drop_columns_with_nulls => try current.dropColumnsWithNulls(),
            .drop_columns_without_nulls => try current.dropColumnsWithoutNulls(),
            .select_columns_with_nans => try current.selectColumnsWithNaNs(),
            .select_columns_without_nans => try current.selectColumnsWithoutNaNs(),
            .drop_columns_with_nans => try current.dropColumnsWithNaNs(),
            .drop_columns_without_nans => try current.dropColumnsWithoutNaNs(),
            .select_columns_with_infs => try current.selectColumnsWithInfs(),
            .select_columns_without_infs => try current.selectColumnsWithoutInfs(),
            .drop_columns_with_infs => try current.dropColumnsWithInfs(),
            .drop_columns_without_infs => try current.dropColumnsWithoutInfs(),
            .select_columns_with_positive_infs => try current.selectColumnsWithPositiveInfs(),
            .select_columns_without_positive_infs => try current.selectColumnsWithoutPositiveInfs(),
            .drop_columns_with_positive_infs => try current.dropColumnsWithPositiveInfs(),
            .drop_columns_without_positive_infs => try current.dropColumnsWithoutPositiveInfs(),
            .select_columns_with_negative_infs => try current.selectColumnsWithNegativeInfs(),
            .select_columns_without_negative_infs => try current.selectColumnsWithoutNegativeInfs(),
            .drop_columns_with_negative_infs => try current.dropColumnsWithNegativeInfs(),
            .drop_columns_without_negative_infs => try current.dropColumnsWithoutNegativeInfs(),
            .select_columns_with_zeros => try current.selectColumnsWithZeros(),
            .select_columns_without_zeros => try current.selectColumnsWithoutZeros(),
            .drop_columns_with_zeros => try current.dropColumnsWithZeros(),
            .drop_columns_without_zeros => try current.dropColumnsWithoutZeros(),
            .select_columns_with_positive_zeros => try current.selectColumnsWithPositiveZeros(),
            .select_columns_without_positive_zeros => try current.selectColumnsWithoutPositiveZeros(),
            .drop_columns_with_positive_zeros => try current.dropColumnsWithPositiveZeros(),
            .drop_columns_without_positive_zeros => try current.dropColumnsWithoutPositiveZeros(),
            .select_columns_with_negative_zeros => try current.selectColumnsWithNegativeZeros(),
            .select_columns_without_negative_zeros => try current.selectColumnsWithoutNegativeZeros(),
            .drop_columns_with_negative_zeros => try current.dropColumnsWithNegativeZeros(),
            .drop_columns_without_negative_zeros => try current.dropColumnsWithoutNegativeZeros(),
            .select_columns_with_non_zeros => try current.selectColumnsWithNonZeros(),
            .select_columns_without_non_zeros => try current.selectColumnsWithoutNonZeros(),
            .drop_columns_with_non_zeros => try current.dropColumnsWithNonZeros(),
            .drop_columns_without_non_zeros => try current.dropColumnsWithoutNonZeros(),
            .select_columns_with_positives => try current.selectColumnsWithPositives(),
            .select_columns_without_positives => try current.selectColumnsWithoutPositives(),
            .drop_columns_with_positives => try current.dropColumnsWithPositives(),
            .drop_columns_without_positives => try current.dropColumnsWithoutPositives(),
            .select_columns_with_signbits => try current.selectColumnsWithSignBits(),
            .select_columns_without_signbits => try current.selectColumnsWithoutSignBits(),
            .drop_columns_with_signbits => try current.dropColumnsWithSignBits(),
            .drop_columns_without_signbits => try current.dropColumnsWithoutSignBits(),
            .select_columns_with_negatives => try current.selectColumnsWithNegatives(),
            .select_columns_without_negatives => try current.selectColumnsWithoutNegatives(),
            .drop_columns_with_negatives => try current.dropColumnsWithNegatives(),
            .drop_columns_without_negatives => try current.dropColumnsWithoutNegatives(),
            .select_columns_with_finites => try current.selectColumnsWithFinites(),
            .select_columns_without_finites => try current.selectColumnsWithoutFinites(),
            .drop_columns_with_finites => try current.dropColumnsWithFinites(),
            .drop_columns_without_finites => try current.dropColumnsWithoutFinites(),
            .select_columns_with_normals => try current.selectColumnsWithNormals(),
            .select_columns_without_normals => try current.selectColumnsWithoutNormals(),
            .drop_columns_with_normals => try current.dropColumnsWithNormals(),
            .drop_columns_without_normals => try current.dropColumnsWithoutNormals(),
            .select_columns_with_subnormals => try current.selectColumnsWithSubnormals(),
            .select_columns_without_subnormals => try current.selectColumnsWithoutSubnormals(),
            .drop_columns_with_subnormals => try current.dropColumnsWithSubnormals(),
            .drop_columns_without_subnormals => try current.dropColumnsWithoutSubnormals(),
            .select_columns_with_non_finites => try current.selectColumnsWithNonFinites(),
            .select_columns_without_non_finites => try current.selectColumnsWithoutNonFinites(),
            .drop_columns_with_non_finites => try current.dropColumnsWithNonFinites(),
            .drop_columns_without_non_finites => try current.dropColumnsWithoutNonFinites(),
            .with_row_index => |row_index| try current.withRowIndex(row_index.name, row_index.offset),
            .rename_column => |rename| try current.renameColumn(rename.old_name, rename.new_name),
            .rename_columns => |rename| try current.renameColumns(rename.old_names, rename.new_names),
            .add_column_name_prefix => |pattern| try current.addColumnNamePrefix(pattern.pattern),
            .add_column_name_suffix => |pattern| try current.addColumnNameSuffix(pattern.pattern),
            .move_column => |move| try current.moveColumn(move.name, move.target_index),
            .move_column_before => |move| try current.moveColumnBefore(move.name, move.anchor_name),
            .move_column_after => |move| try current.moveColumnAfter(move.name, move.anchor_name),
            .copy_column => |copy| try current.copyColumn(copy.source_name, copy.new_name),
            .copy_column_at => |copy| try current.copyColumnAt(copy.source_name, copy.new_name, copy.target_index),
            .copy_column_before => |copy| try current.copyColumnBefore(copy.source_name, copy.new_name, copy.anchor_name),
            .copy_column_after => |copy| try current.copyColumnAfter(copy.source_name, copy.new_name, copy.anchor_name),
            .drop_columns => |names| try current.dropColumns(names),
            .drop_nulls => |names| try current.dropNulls(names),
            .filter_nulls_column => |name| try current.filterNullsColumn(name),
            .drop_nans => |names| try current.dropNaNs(names),
            .filter_nans_column => |name| try current.filterNaNsColumn(name),
            .drop_infs => |names| try current.dropInfs(names),
            .filter_infs_column => |name| try current.filterInfsColumn(name),
            .drop_positive_infs => |names| try current.dropPositiveInfs(names),
            .filter_positive_infs_column => |name| try current.filterPositiveInfsColumn(name),
            .drop_negative_infs => |names| try current.dropNegativeInfs(names),
            .filter_negative_infs_column => |name| try current.filterNegativeInfsColumn(name),
            .drop_zeros => |names| try current.dropZeros(names),
            .filter_zeros_column => |name| try current.filterZerosColumn(name),
            .drop_positive_zeros => |names| try current.dropPositiveZeros(names),
            .filter_positive_zeros_column => |name| try current.filterPositiveZerosColumn(name),
            .drop_negative_zeros => |names| try current.dropNegativeZeros(names),
            .filter_negative_zeros_column => |name| try current.filterNegativeZerosColumn(name),
            .drop_non_zeros => |names| try current.dropNonZeros(names),
            .filter_non_zeros_column => |name| try current.filterNonZerosColumn(name),
            .drop_positives => |names| try current.dropPositives(names),
            .filter_positives_column => |name| try current.filterPositivesColumn(name),
            .drop_signbits => |names| try current.dropSignBits(names),
            .filter_signbits_column => |name| try current.filterSignBitsColumn(name),
            .drop_negatives => |names| try current.dropNegatives(names),
            .filter_negatives_column => |name| try current.filterNegativesColumn(name),
            .drop_finites => |names| try current.dropFinites(names),
            .filter_finites_column => |name| try current.filterFinitesColumn(name),
            .drop_normals => |names| try current.dropNormals(names),
            .filter_normals_column => |name| try current.filterNormalsColumn(name),
            .drop_subnormals => |names| try current.dropSubnormals(names),
            .filter_subnormals_column => |name| try current.filterSubnormalsColumn(name),
            .drop_non_finites => |names| try current.dropNonFinites(names),
            .filter_non_finites_column => |name| try current.filterNonFinitesColumn(name),
            .with_column_abs => |expr| blk: {
                var column_value = try current.unaryColumnAbs(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_neg => |expr| blk: {
                var column_value = try current.unaryColumnNeg(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_square => |expr| blk: {
                var column_value = try current.unaryColumnSquare(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_reciprocal => |expr| blk: {
                var column_value = try current.unaryColumnReciprocal(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_sign => |expr| blk: {
                var column_value = try current.unaryColumnSign(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_sqrt => |expr| blk: {
                var column_value = try current.unaryColumnSqrt(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_rsqrt => |expr| blk: {
                var column_value = try current.unaryColumnRsqrt(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_cbrt => |expr| blk: {
                var column_value = try current.unaryColumnCbrt(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_floor => |expr| blk: {
                var column_value = try current.unaryColumnFloor(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_ceil => |expr| blk: {
                var column_value = try current.unaryColumnCeil(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_round => |expr| blk: {
                var column_value = try current.unaryColumnRound(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_trunc => |expr| blk: {
                var column_value = try current.unaryColumnTrunc(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_deg2rad => |expr| blk: {
                var column_value = try current.unaryColumnDeg2rad(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_rad2deg => |expr| blk: {
                var column_value = try current.unaryColumnRad2deg(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_expit => |expr| blk: {
                var column_value = try current.unaryColumnExpit(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_logit => |expr| blk: {
                var column_value = try current.unaryColumnLogit(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_softplus => |expr| blk: {
                var column_value = try current.unaryColumnSoftplus(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_logsigmoid => |expr| blk: {
                var column_value = try current.unaryColumnLogsigmoid(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_relu => |expr| blk: {
                var column_value = try current.unaryColumnRelu(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_leaky_relu => |expr| blk: {
                var column_value = try current.unaryColumnLeakyReluWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_relu6 => |expr| blk: {
                var column_value = try current.unaryColumnRelu6(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_pow_scalar => |expr| blk: {
                var column_value = try current.unaryColumnPowWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_floor_div_scalar => |expr| blk: {
                var column_value = try current.unaryColumnFloorDivWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_mod_scalar => |expr| blk: {
                var column_value = try current.unaryColumnModWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_remainder_scalar => |expr| blk: {
                var column_value = try current.unaryColumnRemainderWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_log_add_exp_scalar => |expr| blk: {
                var column_value = try current.unaryColumnLogAddExpWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_log_add_exp2_scalar => |expr| blk: {
                var column_value = try current.unaryColumnLogAddExp2WithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_xlogy_scalar => |expr| blk: {
                var column_value = try current.unaryColumnXlogyWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_fmax_scalar => |expr| blk: {
                var column_value = try current.unaryColumnFmaxWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_fmin_scalar => |expr| blk: {
                var column_value = try current.unaryColumnFminWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_hypot_scalar => |expr| blk: {
                var column_value = try current.unaryColumnHypotWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_atan2_scalar => |expr| blk: {
                var column_value = try current.unaryColumnAtan2WithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_next_after_scalar => |expr| blk: {
                var column_value = try current.unaryColumnNextAfterWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_copysign_scalar => |expr| blk: {
                var column_value = try current.unaryColumnCopysignWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_heaviside_scalar => |expr| blk: {
                var column_value = try current.unaryColumnHeavisideWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_ldexp_scalar => |expr| blk: {
                var column_value = try current.unaryColumnLdexpScalar(expr.input_name, expr.exponent);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_threshold => |expr| blk: {
                var column_value = try current.unaryColumnThresholdWithDeviceScalars(expr.input_name, expr.lhs_scalar, expr.rhs_scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_hardtanh => |expr| blk: {
                var column_value = try current.unaryColumnHardtanhWithDeviceScalars(expr.input_name, expr.lhs_scalar, expr.rhs_scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_maximum_scalar => |expr| blk: {
                var column_value = try current.unaryColumnMaximumWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_minimum_scalar => |expr| blk: {
                var column_value = try current.unaryColumnMinimumWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_clip_min => |expr| blk: {
                var column_value = try current.unaryColumnClipMinWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_clip_max => |expr| blk: {
                var column_value = try current.unaryColumnClipMaxWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_hardshrink => |expr| blk: {
                var column_value = try current.unaryColumnHardshrinkWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_softshrink => |expr| blk: {
                var column_value = try current.unaryColumnSoftshrinkWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_tanhshrink => |expr| blk: {
                var column_value = try current.unaryColumnTanhshrink(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_elu => |expr| blk: {
                var column_value = try current.unaryColumnEluWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_celu => |expr| blk: {
                var column_value = try current.unaryColumnCeluWithDeviceScalar(expr.input_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_softsign => |expr| blk: {
                var column_value = try current.unaryColumnSoftsign(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_hardsigmoid => |expr| blk: {
                var column_value = try current.unaryColumnHardsigmoid(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_hardswish => |expr| blk: {
                var column_value = try current.unaryColumnHardswish(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_silu => |expr| blk: {
                var column_value = try current.unaryColumnSilu(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_swish => |expr| blk: {
                var column_value = try current.unaryColumnSwish(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_mish => |expr| blk: {
                var column_value = try current.unaryColumnMish(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_gelu => |expr| blk: {
                var column_value = try current.unaryColumnGelu(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_selu => |expr| blk: {
                var column_value = try current.unaryColumnSelu(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_exp => |expr| blk: {
                var column_value = try current.unaryColumnExp(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_exp2 => |expr| blk: {
                var column_value = try current.unaryColumnExp2(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_expm1 => |expr| blk: {
                var column_value = try current.unaryColumnExpm1(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_sin => |expr| blk: {
                var column_value = try current.unaryColumnSin(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_cos => |expr| blk: {
                var column_value = try current.unaryColumnCos(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_tan => |expr| blk: {
                var column_value = try current.unaryColumnTan(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_asin => |expr| blk: {
                var column_value = try current.unaryColumnAsin(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_acos => |expr| blk: {
                var column_value = try current.unaryColumnAcos(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_atan => |expr| blk: {
                var column_value = try current.unaryColumnAtan(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_sinh => |expr| blk: {
                var column_value = try current.unaryColumnSinh(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_cosh => |expr| blk: {
                var column_value = try current.unaryColumnCosh(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_tanh => |expr| blk: {
                var column_value = try current.unaryColumnTanh(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_asinh => |expr| blk: {
                var column_value = try current.unaryColumnAsinh(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_acosh => |expr| blk: {
                var column_value = try current.unaryColumnAcosh(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_atanh => |expr| blk: {
                var column_value = try current.unaryColumnAtanh(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_log => |expr| blk: {
                var column_value = try current.unaryColumnLog(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_log1p => |expr| blk: {
                var column_value = try current.unaryColumnLog1p(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_lgamma => |expr| blk: {
                var column_value = try current.unaryColumnLgamma(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_sinc => |expr| blk: {
                var column_value = try current.unaryColumnSinc(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_log2 => |expr| blk: {
                var column_value = try current.unaryColumnLog2(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_log10 => |expr| blk: {
                var column_value = try current.unaryColumnLog10(expr.input_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_binary => |expr| blk: {
                var column_value = try current.binaryColumns(expr.lhs_name, expr.rhs_name, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_scalar => |expr| blk: {
                var column_value = try current.binaryColumnScalarWithDeviceScalar(expr.input_name, expr.scalar, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_lerp_scalar => |expr| blk: {
                var column_value = try current.lerpColumnsWithDeviceScalar(expr.lhs_name, expr.rhs_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_addcmul_scalar => |expr| blk: {
                var column_value = try current.addcmulColumnsWithDeviceScalar(expr.base_name, expr.lhs_name, expr.rhs_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_addcdiv_scalar => |expr| blk: {
                var column_value = try current.addcdivColumnsWithDeviceScalar(expr.base_name, expr.lhs_name, expr.rhs_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_clip_array => |expr| blk: {
                var column_value = try current.clipArrayColumns(expr.input_name, expr.lhs_name, expr.rhs_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_where => |expr| blk: {
                var column_value = try current.whereColumns(expr.input_name, expr.lhs_name, expr.rhs_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_where_scalar => |expr| blk: {
                var column_value = try current.whereColumnWithDeviceScalar(expr.input_name, expr.mask_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_masked_put_scalar => |expr| blk: {
                var column_value = try current.maskedPutColumnWithDeviceScalar(expr.input_name, expr.mask_name, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_put_flat_scalar => |expr| blk: {
                var column_value = try current.putFlatColumnWithDeviceScalar(expr.input_name, expr.row_indices, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_put_flat => |expr| blk: {
                var column_value = try current.putFlatColumns(expr.input_name, expr.row_indices, expr.value_name);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_put_flat_scalar_mode => |expr| blk: {
                var column_value = try current.putFlatColumnModeWithDeviceScalar(expr.input_name, expr.row_indices, expr.scalar, expr.mode);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_put_flat_scalar_signed => |expr| blk: {
                var column_value = try current.putFlatColumnSignedWithDeviceScalar(expr.input_name, expr.row_indices, expr.scalar);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_isclose_scalar => |expr| blk: {
                var column_value = try current.iscloseColumnWithDeviceScalarsEqualNan(expr.input_name, expr.scalar, expr.rtol, expr.atol, expr.equal_nan);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_logical => |expr| blk: {
                var column_value = try current.logicalColumns(expr.lhs_name, expr.rhs_name, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_logical_scalar => |expr| blk: {
                var column_value = try current.logicalColumnScalar(expr.input_name, expr.scalar, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_literal => |expr| try current.withColumnLiteralScalar(expr.name, expr.scalar),
            .with_column_literal_at => |expr| try current.withColumnLiteralScalarAt(expr.name, expr.scalar, expr.target_index),
            .with_column_literal_before => |expr| try current.withColumnLiteralScalarBefore(expr.name, expr.scalar, expr.anchor_name),
            .with_column_literal_after => |expr| try current.withColumnLiteralScalarAfter(expr.name, expr.scalar, expr.anchor_name),
            .cast_column => |cast| try current.castColumn(cast.name, cast.dtype),
            .fill_null_column => |fill| try current.fillNullColumnWithScalar(fill.name, fill.scalar),
            .fill_nan_column => |fill| try current.fillNaNColumnWithScalar(fill.name, fill.scalar),
            .fill_inf_column => |fill| try current.fillInfColumnWithScalar(fill.name, fill.scalar),
            .fill_positive_inf_column => |fill| try current.fillPositiveInfColumnWithScalar(fill.name, fill.scalar),
            .fill_negative_inf_column => |fill| try current.fillNegativeInfColumnWithScalar(fill.name, fill.scalar),
            .fill_zero_column => |fill| try current.fillZeroColumnWithScalar(fill.name, fill.scalar),
            .fill_positive_zero_column => |fill| try current.fillPositiveZeroColumnWithScalar(fill.name, fill.scalar),
            .fill_negative_zero_column => |fill| try current.fillNegativeZeroColumnWithScalar(fill.name, fill.scalar),
            .fill_non_zero_column => |fill| try current.fillNonZeroColumnWithScalar(fill.name, fill.scalar),
            .fill_positive_column => |fill| try current.fillPositiveColumnWithScalar(fill.name, fill.scalar),
            .fill_signbit_column => |fill| try current.fillSignBitColumnWithScalar(fill.name, fill.scalar),
            .fill_negative_column => |fill| try current.fillNegativeColumnWithScalar(fill.name, fill.scalar),
            .fill_finite_column => |fill| try current.fillFiniteColumnWithScalar(fill.name, fill.scalar),
            .fill_normal_column => |fill| try current.fillNormalColumnWithScalar(fill.name, fill.scalar),
            .fill_subnormal_column => |fill| try current.fillSubnormalColumnWithScalar(fill.name, fill.scalar),
            .fill_non_finite_column => |fill| try current.fillNonFiniteColumnWithScalar(fill.name, fill.scalar),
            .coalesce_columns => |coalesce| try current.coalesceColumns(coalesce.primary_name, coalesce.fallback_name, coalesce.output_name),
            .is_null_column => |predicate| try current.isNullColumn(predicate.name, predicate.output_name),
            .is_valid_column => |predicate| try current.isValidColumn(predicate.name, predicate.output_name),
            .is_nan_column => |predicate| try current.isNanColumn(predicate.name, predicate.output_name),
            .is_zero_column => |predicate| try current.isZeroColumn(predicate.name, predicate.output_name),
            .is_positive_zero_column => |predicate| try current.isPositiveZeroColumn(predicate.name, predicate.output_name),
            .is_negative_zero_column => |predicate| try current.isNegativeZeroColumn(predicate.name, predicate.output_name),
            .is_non_zero_column => |predicate| try current.isNonZeroColumn(predicate.name, predicate.output_name),
            .is_positive_column => |predicate| try current.isPositiveColumn(predicate.name, predicate.output_name),
            .is_signbit_column => |predicate| try current.isSignBitColumn(predicate.name, predicate.output_name),
            .is_negative_column => |predicate| try current.isNegativeColumn(predicate.name, predicate.output_name),
            .is_finite_column => |predicate| try current.isFiniteColumn(predicate.name, predicate.output_name),
            .is_normal_column => |predicate| try current.isNormalColumn(predicate.name, predicate.output_name),
            .is_subnormal_column => |predicate| try current.isSubnormalColumn(predicate.name, predicate.output_name),
            .is_non_finite_column => |predicate| try current.isNonFiniteColumn(predicate.name, predicate.output_name),
            .is_inf_column => |predicate| try current.isInfColumn(predicate.name, predicate.output_name),
            .is_positive_inf_column => |predicate| try current.isPositiveInfColumn(predicate.name, predicate.output_name),
            .is_negative_inf_column => |predicate| try current.isNegativeInfColumn(predicate.name, predicate.output_name),
            .row_null_count => |row_count| try current.withRowNullCount(row_count.names, row_count.output_name),
            .row_valid_count => |row_count| try current.withRowValidCount(row_count.names, row_count.output_name),
            .row_nan_count => |row_count| try current.withRowNaNCount(row_count.names, row_count.output_name),
            .row_inf_count => |row_count| try current.withRowInfCount(row_count.names, row_count.output_name),
            .row_positive_inf_count => |row_count| try current.withRowPositiveInfCount(row_count.names, row_count.output_name),
            .row_negative_inf_count => |row_count| try current.withRowNegativeInfCount(row_count.names, row_count.output_name),
            .row_zero_count => |row_count| try current.withRowZeroCount(row_count.names, row_count.output_name),
            .row_positive_zero_count => |row_count| try current.withRowPositiveZeroCount(row_count.names, row_count.output_name),
            .row_negative_zero_count => |row_count| try current.withRowNegativeZeroCount(row_count.names, row_count.output_name),
            .row_non_zero_count => |row_count| try current.withRowNonZeroCount(row_count.names, row_count.output_name),
            .row_positive_count => |row_count| try current.withRowPositiveCount(row_count.names, row_count.output_name),
            .row_signbit_count => |row_count| try current.withRowSignBitCount(row_count.names, row_count.output_name),
            .row_negative_count => |row_count| try current.withRowNegativeCount(row_count.names, row_count.output_name),
            .row_finite_count => |row_count| try current.withRowFiniteCount(row_count.names, row_count.output_name),
            .row_normal_count => |row_count| try current.withRowNormalCount(row_count.names, row_count.output_name),
            .row_subnormal_count => |row_count| try current.withRowSubnormalCount(row_count.names, row_count.output_name),
            .row_non_finite_count => |row_count| try current.withRowNonFiniteCount(row_count.names, row_count.output_name),
            .with_column_compare => |expr| blk: {
                var column_value = try current.compareColumns(expr.lhs_name, expr.rhs_name, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_compare_scalar => |expr| blk: {
                var column_value = try current.compareColumnScalarWithDeviceScalar(expr.input_name, expr.scalar, expr.op);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .filter_mask => |mask| try current.filterColumnMask(mask),
            .filter_column => |name| try current.filterColumn(name),
            .filter_scalar => |filter_op| blk: {
                var mask = try current.compareColumnScalarWithDeviceScalar(filter_op.name, filter_op.scalar, filter_op.op);
                defer mask.deinit();
                break :blk try current.filterColumnMask(mask);
            },
            .group_by_count => |group| try current.groupByCount(group.key_name, group.output_name),
            .group_by_value => |group| switch (group.aggregation) {
                .sum => try current.groupBySum(group.key_name, group.value_name, group.output_name),
                .min => try current.groupByMin(group.key_name, group.value_name, group.output_name),
                .max => try current.groupByMax(group.key_name, group.value_name, group.output_name),
                .mean => try current.groupByMean(group.key_name, group.value_name, group.output_name),
            },
            .group_by_stats => |group| try current.groupByStats(group.key_name, group.value_name, group.output_prefix),
            .group_by_stats_on => |group| try current.groupByStatsOn(group.key_names, group.value_name, group.output_prefix),
            .group_by_profile => |group| try current.groupByProfile(group.key_name, group.value_name, group.output_prefix),
            .group_by_profile_on => |group| try current.groupByProfileOn(group.key_names, group.value_name, group.output_prefix),
            .join_on => |join| switch (join.kind) {
                .inner => try current.innerJoinOn(join.right, join.left_key_names, join.right_key_names, join.options),
                .left => try current.leftJoinOn(join.right, join.left_key_names, join.right_key_names, join.options),
                .full => try current.fullJoinOn(join.right, join.left_key_names, join.right_key_names, join.options),
                .semi => try current.semiJoinOn(join.right, join.left_key_names, join.right_key_names),
                .anti => try current.antiJoinOn(join.right, join.left_key_names, join.right_key_names),
            },
            .asof_join => |join| try current.asofJoin(join.right, join.left_key_name, join.right_key_name, join.options),
            .concat_rows => |right| try current.concatRows(right),
            .distinct_rows => try current.distinctRows(),
            .distinct_on => |names| try current.distinctOn(names),
            .sort_by => |sort| try current.sortBy(sort.name, sort.options),
            .top_k => |top| try current.topKBy(top.name, top.k, top.options),
            .rank_profile_by => |rank| try current.rankProfileBy(rank.name, rank.output_prefix, rank.options),
            .rolling_profile => |rolling| try current.rollingProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_moment_profile => |rolling| try current.rollingMomentProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_range_profile => |rolling| try current.rollingRangeProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_normalize_profile => |rolling| try current.rollingNormalizeProfile(rolling.name, rolling.output_prefix, rolling.options),
            .expanding_normalize_profile => |expanding| try current.expandingNormalizeProfile(expanding.name, expanding.output_prefix, expanding.options),
            .rolling_quantile_profile => |rolling| try current.rollingQuantileProfile(rolling.name, rolling.output_prefix, rolling.options),
            .expanding_quantile_profile => |expanding| try current.expandingQuantileProfile(expanding.name, expanding.output_prefix, expanding.options),
            .rolling_bool_profile => |rolling| try current.rollingBoolProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_drawdown_profile => |rolling| try current.rollingDrawdownProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_robust_profile => |rolling| try current.rollingRobustProfile(rolling.name, rolling.output_prefix, rolling.options),
            .rolling_rank_profile => |rolling| try current.rollingRankProfile(rolling.name, rolling.output_prefix, rolling.options),
            .lag_profile => |lag| try current.lagProfile(lag.name, lag.output_prefix, lag.options),
            .lead_profile => |lead| try current.leadProfile(lead.name, lead.output_prefix, lead.options),
            .clip_profile => |clip| try current.clipProfile(clip.name, clip.output_prefix, clip.options),
            .rolling_clip_profile => |clip| try current.rollingClipProfile(clip.name, clip.output_prefix, clip.clip_options, clip.options),
            .expanding_clip_profile => |clip| try current.expandingClipProfile(clip.name, clip.output_prefix, clip.clip_options, clip.options),
            .threshold_profile => |threshold| try current.thresholdProfile(threshold.name, threshold.output_prefix, threshold.options),
            .rolling_threshold_profile => |threshold| try current.rollingThresholdProfile(threshold.name, threshold.output_prefix, threshold.threshold, threshold.options),
            .expanding_threshold_profile => |threshold| try current.expandingThresholdProfile(threshold.name, threshold.output_prefix, threshold.threshold, threshold.options),
            .expanding_profile => |expanding| try current.expandingProfile(expanding.name, expanding.output_prefix, expanding.options),
            .expanding_bool_profile => |expanding| try current.expandingBoolProfile(expanding.name, expanding.output_prefix, expanding.options),
            .expanding_rank_profile => |expanding| try current.expandingRankProfile(expanding.name, expanding.output_prefix, expanding.options),
            .expanding_robust_profile => |expanding| try current.expandingRobustProfile(expanding.name, expanding.output_prefix, expanding.options),
            .expanding_moment_profile => |expanding| try current.expandingMomentProfile(expanding.name, expanding.output_prefix, expanding.options),
            .standardize_profile => |standardize| try current.standardizeProfile(standardize.name, standardize.output_prefix, standardize.options),
            .robust_profile => |robust| try current.robustProfile(robust.name, robust.output_prefix, robust.options),
            .drawdown_profile => |drawdown| try current.drawdownProfile(drawdown.name, drawdown.output_prefix, drawdown.options),
            .extrema_profile => |extrema| try current.extremaProfile(extrema.name, extrema.output_prefix, extrema.options),
            .trend_profile => |trend| try current.trendProfile(trend.name, trend.output_prefix, trend.options),
            .rolling_trend_profile => |trend| try current.rollingTrendProfile(trend.name, trend.output_prefix, trend.trend_options, trend.options),
            .expanding_trend_profile => |trend| try current.expandingTrendProfile(trend.name, trend.output_prefix, trend.trend_options, trend.options),
            .change_point_profile => |change| try current.changePointProfile(change.name, change.output_prefix, change.threshold, change.options),
            .rolling_change_point_profile => |change| try current.rollingChangePointProfile(change.name, change.output_prefix, change.threshold, change.change_options, change.options),
            .expanding_change_point_profile => |change| try current.expandingChangePointProfile(change.name, change.output_prefix, change.threshold, change.change_options, change.options),
            .sign_profile => |sign| try current.signProfile(sign.name, sign.output_prefix, sign.options),
            .rolling_sign_profile => |sign| try current.rollingSignProfile(sign.name, sign.output_prefix, sign.sign_options, sign.options),
            .expanding_sign_profile => |sign| try current.expandingSignProfile(sign.name, sign.output_prefix, sign.sign_options, sign.options),
            .crossover_profile => |cross| try current.crossoverProfile(cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.options),
            .rolling_crossover_profile => |cross| try current.rollingCrossoverProfile(cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.cross_options, cross.options),
            .expanding_crossover_profile => |cross| try current.expandingCrossoverProfile(cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.cross_options, cross.options),
            .bucket_profile => |bucket| try current.bucketProfile(bucket.name, bucket.output_prefix, bucket.options),
            .ema_profile => |ema| try current.emaProfile(ema.name, ema.output_prefix, ema.options),
            .linear_fit_profile => |fit| try current.linearFitProfile(fit.x_name, fit.y_name, fit.output_prefix, fit.options),
            .error_profile => |err| try current.errorProfile(err.actual_name, err.predicted_name, err.output_prefix),
            .rolling_error_profile => |err| try current.rollingErrorProfile(err.actual_name, err.predicted_name, err.output_prefix, err.options),
            .expanding_error_profile => |err| try current.expandingErrorProfile(err.actual_name, err.predicted_name, err.output_prefix, err.options),
            .classification_profile => |class| try current.classificationProfile(class.actual_name, class.predicted_name, class.output_prefix),
            .rolling_classification_profile => |class| try current.rollingClassificationProfile(class.actual_name, class.predicted_name, class.output_prefix, class.options),
            .expanding_classification_profile => |class| try current.expandingClassificationProfile(class.actual_name, class.predicted_name, class.output_prefix, class.options),
            .bool_transition_profile => |transition| try current.boolTransitionProfile(transition.name, transition.output_prefix, transition.options),
            .rolling_bool_transition_profile => |transition| try current.rollingBoolTransitionProfile(transition.name, transition.output_prefix, transition.transition_options, transition.options),
            .expanding_bool_transition_profile => |transition| try current.expandingBoolTransitionProfile(transition.name, transition.output_prefix, transition.transition_options, transition.options),
            .rolling_correlation_profile => |corr| try current.rollingCorrelationProfile(corr.x_name, corr.y_name, corr.output_prefix, corr.options),
            .expanding_correlation_profile => |corr| try current.expandingCorrelationProfile(corr.x_name, corr.y_name, corr.output_prefix, corr.options),
            .expanding_linear_fit_profile => |fit| try current.expandingLinearFitProfile(fit.x_name, fit.y_name, fit.output_prefix, fit.options),
            .rolling_linear_fit_profile => |fit| try current.rollingLinearFitProfile(fit.x_name, fit.y_name, fit.output_prefix, fit.options),
            .validity_profile => |validity| try current.validityProfile(validity.name, validity.output_prefix),
            .rolling_validity_profile => |validity| try current.rollingValidityProfile(validity.name, validity.output_prefix, validity.options),
            .expanding_validity_profile => |validity| try current.expandingValidityProfile(validity.name, validity.output_prefix, validity.options),
            .slice_rows => |slice| try current.sliceRows(slice.start, slice.stop),
            .drop_rows => |row_indices| try current.dropRows(row_indices),
            .drop_row_range => |range| try current.dropRowRange(range.start, range.stop),
            .drop_last_rows => |n| try current.dropLastRows(n),
            .slice_rows_step => |slice| try current.sliceRowsStep(slice.start, slice.stop, slice.step),
            .stride_rows => |stride| try current.strideRows(stride.start, stride.step),
            .take_rows => |row_indices| try current.take(row_indices),
            .take_rows_optional => |row_indices| try current.takeOptional(row_indices),
            .take_rows_mode => |take_mode| try current.takeMode(take_mode.row_indices, take_mode.mode),
            .take_rows_signed => |row_indices| try current.takeSigned(row_indices),
            .take_rows_signed_mode => |take_mode| try current.takeSignedMode(take_mode.row_indices, take_mode.mode),
            .take_rows_by_column => |name| try current.takeByColumn(name),
            .take_rows_by_column_mode => |take_mode| try current.takeByColumnMode(take_mode.name, take_mode.mode),
            .repeat_rows => |repeat_count| try current.repeatRows(repeat_count),
            .repeat_rows_by => |count_name| try current.repeatRowsByColumn(count_name),
            .sample_rows => |sample| try current.sampleRows(sample.count, sample.seed),
            .sample_rows_with_replacement => |sample| try current.sampleRowsWithReplacement(sample.count, sample.seed),
            .roll_rows => |shift| try current.rollRows(shift),
            .reverse_rows => try current.reverseRows(),
            .head => |n| try current.head(n),
            .tail => |n| try current.tail(n),
        };
        current.deinit();
        current = next;
    }
    return current;
}

pub fn explain(comptime DeviceLazyOp: type, self: anytype, allocator: std.mem.Allocator) DeviceDataError![]u8 {
    var optimized = try optimizedOps(DeviceLazyOp, self);
    defer deinitLazyOps(self.allocator, &optimized);
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    try aw.writer.print("DeviceLazyFrame(raw_ops={d}, optimized_ops={d}, source={s})\n", .{ self.ops.items.len, optimized.items.len, self.source.name() });
    if (self.source == .parquet_scan) {
        var pushdown = try planLazyScanPushdown(self.allocator, optimized.items);
        defer pushdown.deinit();
        try aw.writer.print("  scan_pushdown: ", .{});
        try formatLazyScanPushdown(&aw.writer, pushdown);
        try aw.writer.print("\n", .{});
    }
    for (optimized.items, 0..) |op, i| {
        try aw.writer.print("  {d}: ", .{i});
        try formatLazyOp(&aw.writer, op);
        try aw.writer.print("\n", .{});
    }
    return aw.toOwnedSlice();
}

fn optimizedOps(comptime DeviceLazyOp: type, self: anytype) DeviceDataError!std.ArrayList(DeviceLazyOp) {
    var optimized: std.ArrayList(DeviceLazyOp) = .empty;
    errdefer deinitLazyOps(self.allocator, &optimized);
    for (self.ops.items) |op| {
        switch (op) {
            .select => |names| {
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .select) {
                    const previous = optimized.items[optimized.items.len - 1].select;
                    if (allNamesIn(names, previous)) {
                        optimized.items[optimized.items.len - 1].deinit(self.allocator);
                        var cloned_op = try op.clone(self.allocator);
                        errdefer cloned_op.deinit(self.allocator);
                        optimized.items[optimized.items.len - 1] = cloned_op;
                        continue;
                    }
                }
            },
            .head => |n| {
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .sort_by) {
                    const sort = optimized.items[optimized.items.len - 1].sort_by;
                    const name = try self.allocator.dupe(u8, sort.name);
                    optimized.items[optimized.items.len - 1].deinit(self.allocator);
                    optimized.items[optimized.items.len - 1] = .{ .top_k = .{
                        .name = name,
                        .options = sort.options,
                        .k = n,
                    } };
                    continue;
                }
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .top_k) {
                    const top = optimized.items[optimized.items.len - 1].top_k;
                    optimized.items[optimized.items.len - 1] = .{ .top_k = .{
                        .name = top.name,
                        .options = top.options,
                        .k = @min(top.k, n),
                    } };
                    continue;
                }
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .head) {
                    const prev = optimized.items[optimized.items.len - 1].head;
                    optimized.items[optimized.items.len - 1] = .{ .head = @min(prev, n) };
                    continue;
                }
            },
            .tail => |n| {
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .tail) {
                    const prev = optimized.items[optimized.items.len - 1].tail;
                    optimized.items[optimized.items.len - 1] = .{ .tail = @min(prev, n) };
                    continue;
                }
            },
            else => {},
        }
        var cloned_op = try op.clone(self.allocator);
        errdefer cloned_op.deinit(self.allocator);
        try optimized.append(self.allocator, cloned_op);
    }
    return optimized;
}

fn collectSource(comptime DeviceDataFrame: type, comptime DeviceLazyOp: type, self: anytype, ops: []const DeviceLazyOp) ParquetInteropError!DeviceDataFrame {
    return switch (self.source) {
        .dataframe => |frame| try frame.clone(),
        .parquet_scan => |scan| blk: {
            var scan_plan = try scan.clone();
            defer scan_plan.deinit();

            var pushdown = try planLazyScanPushdown(self.allocator, ops);
            defer pushdown.deinit();
            if (pushdown.range_predicate) |predicate| {
                try scan_plan.whereRange(predicate.column, predicate.predicate);
            }
            if (pushdown.projection) |names| {
                try scan_plan.select(names);
            }

            break :blk try scan_plan.collect();
        },
    };
}
