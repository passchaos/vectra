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
            .select_name_glob => |pattern| try current.selectByNameGlob(pattern.pattern),
            .drop_name_prefix => |pattern| try current.dropByNamePrefix(pattern.pattern),
            .drop_name_suffix => |pattern| try current.dropByNameSuffix(pattern.pattern),
            .drop_name_contains => |pattern| try current.dropByNameContains(pattern.pattern),
            .drop_name_glob => |pattern| try current.dropByNameGlob(pattern.pattern),
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
            .strip_column_name_prefix => |pattern| try current.stripColumnNamePrefix(pattern.pattern),
            .strip_column_name_suffix => |pattern| try current.stripColumnNameSuffix(pattern.pattern),
            .replace_column_name_prefix => |replace| try current.replaceColumnNamePrefix(replace.old_pattern, replace.new_pattern),
            .replace_column_name_suffix => |replace| try current.replaceColumnNameSuffix(replace.old_pattern, replace.new_pattern),
            .move_column => |move| try current.moveColumn(move.name, move.target_index),
            .move_column_before => |move| try current.moveColumnBefore(move.name, move.anchor_name),
            .move_column_after => |move| try current.moveColumnAfter(move.name, move.anchor_name),
            .copy_column => |copy| try current.copyColumn(copy.source_name, copy.new_name),
            .copy_column_at => |copy| try current.copyColumnAt(copy.source_name, copy.new_name, copy.target_index),
            .copy_column_before => |copy| try current.copyColumnBefore(copy.source_name, copy.new_name, copy.anchor_name),
            .copy_column_after => |copy| try current.copyColumnAfter(copy.source_name, copy.new_name, copy.anchor_name),
            .drop_columns => |names| try current.dropColumns(names),
            .drop_nulls => |names| try current.dropNulls(names),
            .drop_all_nulls => |names| try current.dropAllNulls(names),
            .filter_all_nulls => |names| try current.filterAllNulls(names),
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
            .with_column_between => |expr| blk: {
                var column_value = try current.betweenColumnWithDeviceScalars(expr.input_name, expr.lower, expr.upper, expr.lower_inclusive, expr.upper_inclusive);
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
            .with_column_isin => |expr| blk: {
                var column_value = try current.isinColumns(expr.input_name, expr.test_name, expr.invert);
                defer column_value.deinit();
                break :blk try current.withColumn(expr.name, column_value);
            },
            .with_column_isin_values => |expr| blk: {
                var column_value = try current.isinColumnValuesWithDeviceColumn(expr.input_name, expr.values, expr.invert);
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
            .fill_null_forward_column => |name| try current.fillNullForwardColumn(name),
            .fill_null_backward_column => |name| try current.fillNullBackwardColumn(name),
            .null_if_column => |fill| try current.nullIfColumnScalar(fill.name, fill.scalar),
            .null_if_values_column => |null_if| try current.nullIfValuesColumnWithDeviceColumn(null_if.name, null_if.values),
            .null_if_nan_column => |name| try current.nullIfNaNColumn(name),
            .null_if_inf_column => |name| try current.nullIfInfColumn(name),
            .null_if_positive_inf_column => |name| try current.nullIfPositiveInfColumn(name),
            .null_if_negative_inf_column => |name| try current.nullIfNegativeInfColumn(name),
            .null_if_zero_column => |name| try current.nullIfZeroColumn(name),
            .null_if_positive_zero_column => |name| try current.nullIfPositiveZeroColumn(name),
            .null_if_negative_zero_column => |name| try current.nullIfNegativeZeroColumn(name),
            .null_if_non_zero_column => |name| try current.nullIfNonZeroColumn(name),
            .null_if_positive_column => |name| try current.nullIfPositiveColumn(name),
            .null_if_signbit_column => |name| try current.nullIfSignBitColumn(name),
            .null_if_negative_column => |name| try current.nullIfNegativeColumn(name),
            .null_if_finite_column => |name| try current.nullIfFiniteColumn(name),
            .null_if_normal_column => |name| try current.nullIfNormalColumn(name),
            .null_if_subnormal_column => |name| try current.nullIfSubnormalColumn(name),
            .null_if_non_finite_column => |name| try current.nullIfNonFiniteColumn(name),
            .coalesce_columns => |coalesce| try current.coalesceColumns(coalesce.primary_name, coalesce.fallback_name, coalesce.output_name),
            .coalesce_columns_many => |coalesce| try current.coalesceColumnsMany(coalesce.names, coalesce.output_name),
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
            .row_any_null => |row_count| try current.withRowAnyNull(row_count.names, row_count.output_name),
            .row_all_null => |row_count| try current.withRowAllNull(row_count.names, row_count.output_name),
            .row_any_valid => |row_count| try current.withRowAnyValid(row_count.names, row_count.output_name),
            .row_all_valid => |row_count| try current.withRowAllValid(row_count.names, row_count.output_name),
            .row_cumulative_any_null => |row_outputs| try current.withRowCumulativeAnyNull(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_null => |row_outputs| try current.withRowCumulativeAllNull(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_valid => |row_outputs| try current.withRowCumulativeAnyValid(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_valid => |row_outputs| try current.withRowCumulativeAllValid(row_outputs.names, row_outputs.output_names),
            .row_cumulative_null_count => |row_outputs| try current.withRowCumulativeNullCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_valid_count => |row_outputs| try current.withRowCumulativeValidCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_null_ratio => |row_outputs| try current.withRowCumulativeNullRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_valid_ratio => |row_outputs| try current.withRowCumulativeValidRatio(row_outputs.names, row_outputs.output_names),
            .row_null_ratio => |row_count| try current.withRowNullRatio(row_count.names, row_count.output_name),
            .row_valid_ratio => |row_count| try current.withRowValidRatio(row_count.names, row_count.output_name),
            .row_first_valid_index => |row_count| try current.withRowFirstValidIndex(row_count.names, row_count.output_name),
            .row_last_valid_index => |row_count| try current.withRowLastValidIndex(row_count.names, row_count.output_name),
            .row_first_null_index => |row_count| try current.withRowFirstNullIndex(row_count.names, row_count.output_name),
            .row_last_null_index => |row_count| try current.withRowLastNullIndex(row_count.names, row_count.output_name),
            .row_cumulative_first_valid_index => |row_outputs| try current.withRowCumulativeFirstValidIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_valid_index => |row_outputs| try current.withRowCumulativeLastValidIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_null_index => |row_outputs| try current.withRowCumulativeFirstNullIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_null_index => |row_outputs| try current.withRowCumulativeLastNullIndex(row_outputs.names, row_outputs.output_names),
            .row_pair_count => |row_paired| try current.withRowPairCount(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_weighted_mean => |row_weighted| try current.withRowWeightedMean(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_variance => |row_weighted| try current.withRowWeightedVariance(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name, row_weighted.correction),
            .row_weighted_stddev => |row_weighted| try current.withRowWeightedStddev(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name, row_weighted.correction),
            .row_weighted_covariance => |row_weighted| try current.withRowWeightedCovariance(row_weighted.lhs_names, row_weighted.rhs_names, row_weighted.weight_names, row_weighted.output_name, row_weighted.correction),
            .row_weighted_correlation => |row_weighted| try current.withRowWeightedCorrelation(row_weighted.lhs_names, row_weighted.rhs_names, row_weighted.weight_names, row_weighted.output_name, row_weighted.correction),
            .row_weighted_beta => |row_weighted| try current.withRowWeightedBeta(row_weighted.lhs_names, row_weighted.rhs_names, row_weighted.weight_names, row_weighted.output_name, row_weighted.correction),
            .row_weighted_quantile => |row_weighted| try current.withRowWeightedQuantile(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name, row_weighted.q),
            .row_weighted_median => |row_weighted| try current.withRowWeightedMedian(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_iqr => |row_weighted| try current.withRowWeightedIqr(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_mad => |row_weighted| try current.withRowWeightedMad(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_mode => |row_weighted| try current.withRowWeightedMode(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_mode_weight => |row_weighted| try current.withRowWeightedModeWeight(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_mode_ratio => |row_weighted| try current.withRowWeightedModeRatio(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_mode_margin => |row_weighted| try current.withRowWeightedModeMargin(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_mode_margin_ratio => |row_weighted| try current.withRowWeightedModeMarginRatio(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_entropy => |row_weighted| try current.withRowWeightedEntropy(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_gini_impurity => |row_weighted| try current.withRowWeightedGiniImpurity(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_perplexity => |row_weighted| try current.withRowWeightedPerplexity(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_inverse_simpson => |row_weighted| try current.withRowWeightedInverseSimpson(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_simpson_concentration => |row_weighted| try current.withRowWeightedSimpsonConcentration(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_weighted_evenness => |row_weighted| try current.withRowWeightedEvenness(row_weighted.value_names, row_weighted.weight_names, row_weighted.output_name),
            .row_dot => |row_paired| try current.withRowDot(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_cosine_similarity => |row_paired| try current.withRowCosineSimilarity(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_squared_euclidean_distance => |row_paired| try current.withRowSquaredEuclideanDistance(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_euclidean_distance => |row_paired| try current.withRowEuclideanDistance(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_manhattan_distance => |row_paired| try current.withRowManhattanDistance(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_chebyshev_distance => |row_paired| try current.withRowChebyshevDistance(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_canberra_distance => |row_paired| try current.withRowCanberraDistance(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_bray_curtis_distance => |row_paired| try current.withRowBrayCurtisDistance(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_mean_error => |row_paired| try current.withRowMeanError(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_mae => |row_paired| try current.withRowMae(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_mse => |row_paired| try current.withRowMse(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_rmse => |row_paired| try current.withRowRmse(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_mape => |row_paired| try current.withRowMape(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_smape => |row_paired| try current.withRowSmape(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_covariance => |row_paired| try current.withRowCovariance(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_correlation => |row_paired| try current.withRowCorrelation(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_beta => |row_paired| try current.withRowBeta(row_paired.value_names, row_paired.weight_names, row_paired.output_name),
            .row_argmin => |row_count| try current.withRowArgMin(row_count.names, row_count.output_name),
            .row_argmax => |row_count| try current.withRowArgMax(row_count.names, row_count.output_name),
            .row_cumulative_argmin => |row_outputs| try current.withRowCumulativeArgMin(row_outputs.names, row_outputs.output_names),
            .row_cumulative_argmax => |row_outputs| try current.withRowCumulativeArgMax(row_outputs.names, row_outputs.output_names),
            .row_quantile => |row_quantile| try current.withRowQuantile(row_quantile.names, row_quantile.output_name, row_quantile.q),
            .row_quantile_range => |row_quantile_range| try current.withRowQuantileRange(row_quantile_range.names, row_quantile_range.output_name, row_quantile_range.low_q, row_quantile_range.high_q),
            .row_trimmed_mean => |row_trimmed_mean| try current.withRowTrimmedMean(row_trimmed_mean.names, row_trimmed_mean.output_name, row_trimmed_mean.trim_fraction),
            .row_winsorized_mean => |row_winsorized_mean| try current.withRowWinsorizedMean(row_winsorized_mean.names, row_winsorized_mean.output_name, row_winsorized_mean.winsor_fraction),
            .row_median => |row_count| try current.withRowMedian(row_count.names, row_count.output_name),
            .row_iqr => |row_count| try current.withRowIqr(row_count.names, row_count.output_name),
            .row_interdecile_range => |row_count| try current.withRowInterdecileRange(row_count.names, row_count.output_name),
            .row_midhinge => |row_count| try current.withRowMidhinge(row_count.names, row_count.output_name),
            .row_trimean => |row_count| try current.withRowTrimean(row_count.names, row_count.output_name),
            .row_bowley_skewness => |row_count| try current.withRowBowleySkewness(row_count.names, row_count.output_name),
            .row_quartile_coeff_dispersion => |row_count| try current.withRowQuartileCoeffDispersion(row_count.names, row_count.output_name),
            .row_kelley_skewness => |row_count| try current.withRowKelleySkewness(row_count.names, row_count.output_name),
            .row_mad => |row_count| try current.withRowMad(row_count.names, row_count.output_name),
            .row_mode => |row_count| try current.withRowMode(row_count.names, row_count.output_name),
            .row_cumulative_mode => |row_outputs| try current.withRowCumulativeMode(row_outputs.names, row_outputs.output_names),
            .row_cumulative_mode_count => |row_outputs| try current.withRowCumulativeModeCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_mode_ratio => |row_outputs| try current.withRowCumulativeModeRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_mode_margin => |row_outputs| try current.withRowCumulativeModeMargin(row_outputs.names, row_outputs.output_names),
            .row_cumulative_mode_margin_ratio => |row_outputs| try current.withRowCumulativeModeMarginRatio(row_outputs.names, row_outputs.output_names),
            .row_entropy => |row_count| try current.withRowEntropy(row_count.names, row_count.output_name),
            .row_gini_impurity => |row_count| try current.withRowGiniImpurity(row_count.names, row_count.output_name),
            .row_perplexity => |row_count| try current.withRowPerplexity(row_count.names, row_count.output_name),
            .row_inverse_simpson => |row_count| try current.withRowInverseSimpson(row_count.names, row_count.output_name),
            .row_simpson_concentration => |row_count| try current.withRowSimpsonConcentration(row_count.names, row_count.output_name),
            .row_evenness => |row_count| try current.withRowEvenness(row_count.names, row_count.output_name),
            .row_mode_count => |row_count| try current.withRowModeCount(row_count.names, row_count.output_name),
            .row_mode_ratio => |row_count| try current.withRowModeRatio(row_count.names, row_count.output_name),
            .row_mode_margin => |row_count| try current.withRowModeMargin(row_count.names, row_count.output_name),
            .row_mode_margin_ratio => |row_count| try current.withRowModeMarginRatio(row_count.names, row_count.output_name),
            .row_count_distinct => |row_count| try current.withRowCountDistinct(row_count.names, row_count.output_name),
            .row_n_unique => |row_count| try current.withRowNUnique(row_count.names, row_count.output_name),
            .row_is_duplicated => |row_count| try current.withRowIsDuplicated(row_count.names, row_count.output_name),
            .row_is_unique => |row_count| try current.withRowIsUnique(row_count.names, row_count.output_name),
            .row_cumulative_distinct_count => |row_outputs| try current.withRowCumulativeDistinctCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_n_unique => |row_outputs| try current.withRowCumulativeNUnique(row_outputs.names, row_outputs.output_names),
            .row_sum => |row_count| try current.withRowSum(row_count.names, row_count.output_name),
            .row_mean => |row_count| try current.withRowMean(row_count.names, row_count.output_name),
            .row_logsumexp => |row_count| try current.withRowLogSumExp(row_count.names, row_count.output_name),
            .row_logmeanexp => |row_count| try current.withRowLogMeanExp(row_count.names, row_count.output_name),
            .row_centered => |row_outputs| try current.withRowCentered(row_outputs.names, row_outputs.output_names),
            .row_zscore => |row_outputs| try current.withRowZScore(row_outputs.names, row_outputs.output_names),
            .row_robust_zscore => |row_outputs| try current.withRowRobustZScore(row_outputs.names, row_outputs.output_names),
            .row_average_rank => |row_outputs| try current.withRowAverageRank(row_outputs.names, row_outputs.output_names),
            .row_ordinal_rank => |row_outputs| try current.withRowOrdinalRank(row_outputs.names, row_outputs.output_names),
            .row_dense_rank => |row_outputs| try current.withRowDenseRank(row_outputs.names, row_outputs.output_names),
            .row_competition_rank => |row_outputs| try current.withRowCompetitionRank(row_outputs.names, row_outputs.output_names),
            .row_percent_rank => |row_outputs| try current.withRowPercentRank(row_outputs.names, row_outputs.output_names),
            .row_cume_dist => |row_outputs| try current.withRowCumeDist(row_outputs.names, row_outputs.output_names),
            .row_cumulative_sum => |row_outputs| try current.withRowCumulativeSum(row_outputs.names, row_outputs.output_names),
            .row_cumulative_mean => |row_outputs| try current.withRowCumulativeMean(row_outputs.names, row_outputs.output_names),
            .row_cumulative_logsumexp => |row_outputs| try current.withRowCumulativeLogSumExp(row_outputs.names, row_outputs.output_names),
            .row_cumulative_logmeanexp => |row_outputs| try current.withRowCumulativeLogMeanExp(row_outputs.names, row_outputs.output_names),
            .row_cumulative_geometric_mean => |row_outputs| try current.withRowCumulativeGeometricMean(row_outputs.names, row_outputs.output_names),
            .row_cumulative_harmonic_mean => |row_outputs| try current.withRowCumulativeHarmonicMean(row_outputs.names, row_outputs.output_names),
            .row_cumulative_variance => |row_outputs| try current.withRowCumulativeVariance(row_outputs.names, row_outputs.output_names, row_outputs.correction),
            .row_cumulative_stddev => |row_outputs| try current.withRowCumulativeStddev(row_outputs.names, row_outputs.output_names, row_outputs.correction),
            .row_cumulative_sem => |row_outputs| try current.withRowCumulativeSem(row_outputs.names, row_outputs.output_names, row_outputs.correction),
            .row_cumulative_cv => |row_outputs| try current.withRowCumulativeCv(row_outputs.names, row_outputs.output_names, row_outputs.correction),
            .row_cumulative_fano => |row_outputs| try current.withRowCumulativeFano(row_outputs.names, row_outputs.output_names, row_outputs.correction),
            .row_cumulative_skewness => |row_outputs| try current.withRowCumulativeSkewness(row_outputs.names, row_outputs.output_names),
            .row_cumulative_kurtosis => |row_outputs| try current.withRowCumulativeKurtosis(row_outputs.names, row_outputs.output_names),
            .row_cumulative_rms => |row_outputs| try current.withRowCumulativeRms(row_outputs.names, row_outputs.output_names),
            .row_cumulative_mean_abs => |row_outputs| try current.withRowCumulativeMeanAbs(row_outputs.names, row_outputs.output_names),
            .row_cumulative_mean_square => |row_outputs| try current.withRowCumulativeMeanSquare(row_outputs.names, row_outputs.output_names),
            .row_cumulative_max_abs => |row_outputs| try current.withRowCumulativeMaxAbs(row_outputs.names, row_outputs.output_names),
            .row_cumulative_min_abs => |row_outputs| try current.withRowCumulativeMinAbs(row_outputs.names, row_outputs.output_names),
            .row_cumulative_l1_norm => |row_outputs| try current.withRowCumulativeL1Norm(row_outputs.names, row_outputs.output_names),
            .row_cumulative_l2_norm => |row_outputs| try current.withRowCumulativeL2Norm(row_outputs.names, row_outputs.output_names),
            .row_cumulative_product => |row_outputs| try current.withRowCumulativeProduct(row_outputs.names, row_outputs.output_names),
            .row_cumulative_max => |row_outputs| try current.withRowCumulativeMax(row_outputs.names, row_outputs.output_names),
            .row_cumulative_min => |row_outputs| try current.withRowCumulativeMin(row_outputs.names, row_outputs.output_names),
            .row_cumulative_range => |row_outputs| try current.withRowCumulativeRange(row_outputs.names, row_outputs.output_names),
            .row_iqr_outlier => |row_outputs| try current.withRowIqrOutlier(row_outputs.names, row_outputs.output_names),
            .row_tukey_winsorize => |row_outputs| try current.withRowTukeyWinsorize(row_outputs.names, row_outputs.output_names),
            .row_max_indicator => |row_outputs| try current.withRowMaxIndicator(row_outputs.names, row_outputs.output_names),
            .row_min_indicator => |row_outputs| try current.withRowMinIndicator(row_outputs.names, row_outputs.output_names),
            .row_minmax_scale => |row_outputs| try current.withRowMinMaxScale(row_outputs.names, row_outputs.output_names),
            .row_l2_normalize => |row_outputs| try current.withRowL2Normalize(row_outputs.names, row_outputs.output_names),
            .row_l1_normalize => |row_outputs| try current.withRowL1Normalize(row_outputs.names, row_outputs.output_names),
            .row_sum_normalize => |row_outputs| try current.withRowSumNormalize(row_outputs.names, row_outputs.output_names),
            .row_mean_normalize => |row_outputs| try current.withRowMeanNormalize(row_outputs.names, row_outputs.output_names),
            .row_max_abs_normalize => |row_outputs| try current.withRowMaxAbsNormalize(row_outputs.names, row_outputs.output_names),
            .row_softmax => |row_softmax| try current.withRowSoftmax(row_softmax.names, row_softmax.output_names),
            .row_log_softmax => |row_softmax| try current.withRowLogSoftmax(row_softmax.names, row_softmax.output_names),
            .row_softmin => |row_softmax| try current.withRowSoftmin(row_softmax.names, row_softmax.output_names),
            .row_log_softmin => |row_softmax| try current.withRowLogSoftmin(row_softmax.names, row_softmax.output_names),
            .row_softmax_entropy => |row_count| try current.withRowSoftmaxEntropy(row_count.names, row_count.output_name),
            .row_softmax_perplexity => |row_count| try current.withRowSoftmaxPerplexity(row_count.names, row_count.output_name),
            .row_softmax_confidence => |row_count| try current.withRowSoftmaxConfidence(row_count.names, row_count.output_name),
            .row_softmax_margin => |row_count| try current.withRowSoftmaxMargin(row_count.names, row_count.output_name),
            .row_softmax_evenness => |row_count| try current.withRowSoftmaxEvenness(row_count.names, row_count.output_name),
            .row_softmax_concentration => |row_count| try current.withRowSoftmaxConcentration(row_count.names, row_count.output_name),
            .row_softmax_normalized_hhi => |row_count| try current.withRowSoftmaxNormalizedHhi(row_count.names, row_count.output_name),
            .row_softmax_gini_impurity => |row_count| try current.withRowSoftmaxGiniImpurity(row_count.names, row_count.output_name),
            .row_softmax_inverse_simpson => |row_count| try current.withRowSoftmaxInverseSimpson(row_count.names, row_count.output_name),
            .row_softmax_simpson_evenness => |row_count| try current.withRowSoftmaxSimpsonEvenness(row_count.names, row_count.output_name),
            .row_logit_margin => |row_count| try current.withRowLogitMargin(row_count.names, row_count.output_name),
            .row_magnitude_skewness => |row_count| try current.withRowMagnitudeSkewness(row_count.names, row_count.output_name),
            .row_magnitude_kurtosis => |row_count| try current.withRowMagnitudeKurtosis(row_count.names, row_count.output_name),
            .row_geometric_mean => |row_count| try current.withRowGeometricMean(row_count.names, row_count.output_name),
            .row_magnitude_geometric_mean => |row_count| try current.withRowMagnitudeGeometricMean(row_count.names, row_count.output_name),
            .row_harmonic_mean => |row_count| try current.withRowHarmonicMean(row_count.names, row_count.output_name),
            .row_prod => |row_count| try current.withRowProd(row_count.names, row_count.output_name),
            .row_min => |row_count| try current.withRowMin(row_count.names, row_count.output_name),
            .row_max => |row_count| try current.withRowMax(row_count.names, row_count.output_name),
            .row_ptp => |row_count| try current.withRowPtp(row_count.names, row_count.output_name),
            .row_magnitude_ptp => |row_count| try current.withRowMagnitudePtp(row_count.names, row_count.output_name),
            .row_midrange => |row_count| try current.withRowMidrange(row_count.names, row_count.output_name),
            .row_magnitude_midrange => |row_count| try current.withRowMagnitudeMidrange(row_count.names, row_count.output_name),
            .row_range_coeff => |row_count| try current.withRowRangeCoeff(row_count.names, row_count.output_name),
            .row_magnitude_range_coeff => |row_count| try current.withRowMagnitudeRangeCoeff(row_count.names, row_count.output_name),
            .row_mean_abs => |row_count| try current.withRowMeanAbs(row_count.names, row_count.output_name),
            .row_hhi => |row_count| try current.withRowHhi(row_count.names, row_count.output_name),
            .row_magnitude_normalized_hhi => |row_count| try current.withRowMagnitudeNormalizedHhi(row_count.names, row_count.output_name),
            .row_magnitude_sparsity => |row_count| try current.withRowMagnitudeSparsity(row_count.names, row_count.output_name),
            .row_magnitude_inverse_simpson => |row_count| try current.withRowMagnitudeInverseSimpson(row_count.names, row_count.output_name),
            .row_magnitude_simpson_evenness => |row_count| try current.withRowMagnitudeSimpsonEvenness(row_count.names, row_count.output_name),
            .row_magnitude_dominance => |row_count| try current.withRowMagnitudeDominance(row_count.names, row_count.output_name),
            .row_magnitude_dominance_margin => |row_count| try current.withRowMagnitudeDominanceMargin(row_count.names, row_count.output_name),
            .row_magnitude_entropy => |row_count| try current.withRowMagnitudeEntropy(row_count.names, row_count.output_name),
            .row_magnitude_perplexity => |row_count| try current.withRowMagnitudePerplexity(row_count.names, row_count.output_name),
            .row_magnitude_evenness => |row_count| try current.withRowMagnitudeEvenness(row_count.names, row_count.output_name),
            .row_mean_abs_dev => |row_count| try current.withRowMeanAbsDev(row_count.names, row_count.output_name),
            .row_gini_mean_diff => |row_count| try current.withRowGiniMeanDiff(row_count.names, row_count.output_name),
            .row_gini_coefficient => |row_count| try current.withRowGiniCoefficient(row_count.names, row_count.output_name),
            .row_mean_abs_dev_ratio => |row_count| try current.withRowMeanAbsDevRatio(row_count.names, row_count.output_name),
            .row_rms => |row_count| try current.withRowRms(row_count.names, row_count.output_name),
            .row_l1_norm => |row_count| try current.withRowL1Norm(row_count.names, row_count.output_name),
            .row_l2_norm => |row_count| try current.withRowL2Norm(row_count.names, row_count.output_name),
            .row_variance => |row_dispersion| try current.withRowVariance(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_magnitude_variance => |row_dispersion| try current.withRowMagnitudeVariance(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_stddev => |row_dispersion| try current.withRowStddev(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_magnitude_stddev => |row_dispersion| try current.withRowMagnitudeStddev(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_sem => |row_dispersion| try current.withRowSem(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_magnitude_sem => |row_dispersion| try current.withRowMagnitudeSem(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_cv => |row_dispersion| try current.withRowCv(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_magnitude_cv => |row_dispersion| try current.withRowMagnitudeCv(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_magnitude_fano => |row_dispersion| try current.withRowMagnitudeFano(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_fano => |row_dispersion| try current.withRowFano(row_dispersion.names, row_dispersion.output_name, row_dispersion.correction),
            .row_skewness => |row_count| try current.withRowSkewness(row_count.names, row_count.output_name),
            .row_kurtosis => |row_count| try current.withRowKurtosis(row_count.names, row_count.output_name),
            .row_true_count => |row_count| try current.withRowTrueCount(row_count.names, row_count.output_name),
            .row_false_count => |row_count| try current.withRowFalseCount(row_count.names, row_count.output_name),
            .row_cumulative_true_count => |row_outputs| try current.withRowCumulativeTrueCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_false_count => |row_outputs| try current.withRowCumulativeFalseCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_true_ratio => |row_outputs| try current.withRowCumulativeTrueRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_false_ratio => |row_outputs| try current.withRowCumulativeFalseRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_positive_zero_count => |row_outputs| try current.withRowCumulativePositiveZeroCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_negative_zero_count => |row_outputs| try current.withRowCumulativeNegativeZeroCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_signbit_count => |row_outputs| try current.withRowCumulativeSignBitCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_positive_zero_ratio => |row_outputs| try current.withRowCumulativePositiveZeroRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_negative_zero_ratio => |row_outputs| try current.withRowCumulativeNegativeZeroRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_signbit_ratio => |row_outputs| try current.withRowCumulativeSignBitRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_nan_count => |row_outputs| try current.withRowCumulativeNaNCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_inf_count => |row_outputs| try current.withRowCumulativeInfCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_positive_inf_count => |row_outputs| try current.withRowCumulativePositiveInfCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_negative_inf_count => |row_outputs| try current.withRowCumulativeNegativeInfCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_finite_count => |row_outputs| try current.withRowCumulativeFiniteCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_normal_count => |row_outputs| try current.withRowCumulativeNormalCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_subnormal_count => |row_outputs| try current.withRowCumulativeSubnormalCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_non_finite_count => |row_outputs| try current.withRowCumulativeNonFiniteCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_nan_ratio => |row_outputs| try current.withRowCumulativeNaNRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_inf_ratio => |row_outputs| try current.withRowCumulativeInfRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_positive_inf_ratio => |row_outputs| try current.withRowCumulativePositiveInfRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_negative_inf_ratio => |row_outputs| try current.withRowCumulativeNegativeInfRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_finite_ratio => |row_outputs| try current.withRowCumulativeFiniteRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_normal_ratio => |row_outputs| try current.withRowCumulativeNormalRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_subnormal_ratio => |row_outputs| try current.withRowCumulativeSubnormalRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_non_finite_ratio => |row_outputs| try current.withRowCumulativeNonFiniteRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_zero => |row_outputs| try current.withRowCumulativeAnyZero(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_zero => |row_outputs| try current.withRowCumulativeAllZero(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_non_zero => |row_outputs| try current.withRowCumulativeAnyNonZero(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_non_zero => |row_outputs| try current.withRowCumulativeAllNonZero(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_positive_zero => |row_outputs| try current.withRowCumulativeAnyPositiveZero(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_positive_zero => |row_outputs| try current.withRowCumulativeAllPositiveZero(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_negative_zero => |row_outputs| try current.withRowCumulativeAnyNegativeZero(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_negative_zero => |row_outputs| try current.withRowCumulativeAllNegativeZero(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_positive => |row_outputs| try current.withRowCumulativeAnyPositive(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_positive => |row_outputs| try current.withRowCumulativeAllPositive(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_signbit => |row_outputs| try current.withRowCumulativeAnySignBit(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_signbit => |row_outputs| try current.withRowCumulativeAllSignBit(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_negative => |row_outputs| try current.withRowCumulativeAnyNegative(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_negative => |row_outputs| try current.withRowCumulativeAllNegative(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_nan => |row_outputs| try current.withRowCumulativeAnyNaN(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_nan => |row_outputs| try current.withRowCumulativeAllNaN(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_inf => |row_outputs| try current.withRowCumulativeAnyInf(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_inf => |row_outputs| try current.withRowCumulativeAllInf(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_positive_inf => |row_outputs| try current.withRowCumulativeAnyPositiveInf(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_positive_inf => |row_outputs| try current.withRowCumulativeAllPositiveInf(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_negative_inf => |row_outputs| try current.withRowCumulativeAnyNegativeInf(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_negative_inf => |row_outputs| try current.withRowCumulativeAllNegativeInf(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_finite => |row_outputs| try current.withRowCumulativeAnyFinite(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_finite => |row_outputs| try current.withRowCumulativeAllFinite(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_normal => |row_outputs| try current.withRowCumulativeAnyNormal(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_normal => |row_outputs| try current.withRowCumulativeAllNormal(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_subnormal => |row_outputs| try current.withRowCumulativeAnySubnormal(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_subnormal => |row_outputs| try current.withRowCumulativeAllSubnormal(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_non_finite => |row_outputs| try current.withRowCumulativeAnyNonFinite(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_non_finite => |row_outputs| try current.withRowCumulativeAllNonFinite(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_nan_index => |row_outputs| try current.withRowCumulativeFirstNaNIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_nan_index => |row_outputs| try current.withRowCumulativeLastNaNIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_inf_index => |row_outputs| try current.withRowCumulativeFirstInfIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_inf_index => |row_outputs| try current.withRowCumulativeLastInfIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_positive_inf_index => |row_outputs| try current.withRowCumulativeFirstPositiveInfIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_positive_inf_index => |row_outputs| try current.withRowCumulativeLastPositiveInfIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_negative_inf_index => |row_outputs| try current.withRowCumulativeFirstNegativeInfIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_negative_inf_index => |row_outputs| try current.withRowCumulativeLastNegativeInfIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_finite_index => |row_outputs| try current.withRowCumulativeFirstFiniteIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_finite_index => |row_outputs| try current.withRowCumulativeLastFiniteIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_normal_index => |row_outputs| try current.withRowCumulativeFirstNormalIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_normal_index => |row_outputs| try current.withRowCumulativeLastNormalIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_subnormal_index => |row_outputs| try current.withRowCumulativeFirstSubnormalIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_subnormal_index => |row_outputs| try current.withRowCumulativeLastSubnormalIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_non_finite_index => |row_outputs| try current.withRowCumulativeFirstNonFiniteIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_non_finite_index => |row_outputs| try current.withRowCumulativeLastNonFiniteIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_zero_count => |row_outputs| try current.withRowCumulativeZeroCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_zero_index => |row_outputs| try current.withRowCumulativeFirstZeroIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_zero_index => |row_outputs| try current.withRowCumulativeLastZeroIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_positive_zero_index => |row_outputs| try current.withRowCumulativeFirstPositiveZeroIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_positive_zero_index => |row_outputs| try current.withRowCumulativeLastPositiveZeroIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_negative_zero_index => |row_outputs| try current.withRowCumulativeFirstNegativeZeroIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_negative_zero_index => |row_outputs| try current.withRowCumulativeLastNegativeZeroIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_non_zero_count => |row_outputs| try current.withRowCumulativeNonZeroCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_non_zero_index => |row_outputs| try current.withRowCumulativeFirstNonZeroIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_non_zero_index => |row_outputs| try current.withRowCumulativeLastNonZeroIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_positive_index => |row_outputs| try current.withRowCumulativeFirstPositiveIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_positive_index => |row_outputs| try current.withRowCumulativeLastPositiveIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_signbit_index => |row_outputs| try current.withRowCumulativeFirstSignBitIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_signbit_index => |row_outputs| try current.withRowCumulativeLastSignBitIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_negative_index => |row_outputs| try current.withRowCumulativeFirstNegativeIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_negative_index => |row_outputs| try current.withRowCumulativeLastNegativeIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_positive_count => |row_outputs| try current.withRowCumulativePositiveCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_negative_count => |row_outputs| try current.withRowCumulativeNegativeCount(row_outputs.names, row_outputs.output_names),
            .row_cumulative_zero_ratio => |row_outputs| try current.withRowCumulativeZeroRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_non_zero_ratio => |row_outputs| try current.withRowCumulativeNonZeroRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_positive_ratio => |row_outputs| try current.withRowCumulativePositiveRatio(row_outputs.names, row_outputs.output_names),
            .row_cumulative_negative_ratio => |row_outputs| try current.withRowCumulativeNegativeRatio(row_outputs.names, row_outputs.output_names),
            .row_any_true => |row_count| try current.withRowAnyTrue(row_count.names, row_count.output_name),
            .row_all_true => |row_count| try current.withRowAllTrue(row_count.names, row_count.output_name),
            .row_any_false => |row_count| try current.withRowAnyFalse(row_count.names, row_count.output_name),
            .row_all_false => |row_count| try current.withRowAllFalse(row_count.names, row_count.output_name),
            .row_cumulative_any_true => |row_outputs| try current.withRowCumulativeAnyTrue(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_true => |row_outputs| try current.withRowCumulativeAllTrue(row_outputs.names, row_outputs.output_names),
            .row_cumulative_any_false => |row_outputs| try current.withRowCumulativeAnyFalse(row_outputs.names, row_outputs.output_names),
            .row_cumulative_all_false => |row_outputs| try current.withRowCumulativeAllFalse(row_outputs.names, row_outputs.output_names),
            .row_first_true_index => |row_count| try current.withRowFirstTrueIndex(row_count.names, row_count.output_name),
            .row_last_true_index => |row_count| try current.withRowLastTrueIndex(row_count.names, row_count.output_name),
            .row_first_false_index => |row_count| try current.withRowFirstFalseIndex(row_count.names, row_count.output_name),
            .row_last_false_index => |row_count| try current.withRowLastFalseIndex(row_count.names, row_count.output_name),
            .row_cumulative_first_true_index => |row_outputs| try current.withRowCumulativeFirstTrueIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_true_index => |row_outputs| try current.withRowCumulativeLastTrueIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_first_false_index => |row_outputs| try current.withRowCumulativeFirstFalseIndex(row_outputs.names, row_outputs.output_names),
            .row_cumulative_last_false_index => |row_outputs| try current.withRowCumulativeLastFalseIndex(row_outputs.names, row_outputs.output_names),
            .row_true_ratio => |row_count| try current.withRowTrueRatio(row_count.names, row_count.output_name),
            .row_false_ratio => |row_count| try current.withRowFalseRatio(row_count.names, row_count.output_name),
            .row_any_zero => |row_count| try current.withRowAnyZero(row_count.names, row_count.output_name),
            .row_all_zero => |row_count| try current.withRowAllZero(row_count.names, row_count.output_name),
            .row_any_non_zero => |row_count| try current.withRowAnyNonZero(row_count.names, row_count.output_name),
            .row_all_non_zero => |row_count| try current.withRowAllNonZero(row_count.names, row_count.output_name),
            .row_any_positive_zero => |row_count| try current.withRowAnyPositiveZero(row_count.names, row_count.output_name),
            .row_all_positive_zero => |row_count| try current.withRowAllPositiveZero(row_count.names, row_count.output_name),
            .row_any_negative_zero => |row_count| try current.withRowAnyNegativeZero(row_count.names, row_count.output_name),
            .row_all_negative_zero => |row_count| try current.withRowAllNegativeZero(row_count.names, row_count.output_name),
            .row_any_positive => |row_count| try current.withRowAnyPositive(row_count.names, row_count.output_name),
            .row_all_positive => |row_count| try current.withRowAllPositive(row_count.names, row_count.output_name),
            .row_any_signbit => |row_count| try current.withRowAnySignBit(row_count.names, row_count.output_name),
            .row_all_signbit => |row_count| try current.withRowAllSignBit(row_count.names, row_count.output_name),
            .row_any_negative => |row_count| try current.withRowAnyNegative(row_count.names, row_count.output_name),
            .row_all_negative => |row_count| try current.withRowAllNegative(row_count.names, row_count.output_name),
            .row_any_nan => |row_count| try current.withRowAnyNaN(row_count.names, row_count.output_name),
            .row_all_nan => |row_count| try current.withRowAllNaN(row_count.names, row_count.output_name),
            .row_any_inf => |row_count| try current.withRowAnyInf(row_count.names, row_count.output_name),
            .row_all_inf => |row_count| try current.withRowAllInf(row_count.names, row_count.output_name),
            .row_any_positive_inf => |row_count| try current.withRowAnyPositiveInf(row_count.names, row_count.output_name),
            .row_all_positive_inf => |row_count| try current.withRowAllPositiveInf(row_count.names, row_count.output_name),
            .row_any_negative_inf => |row_count| try current.withRowAnyNegativeInf(row_count.names, row_count.output_name),
            .row_all_negative_inf => |row_count| try current.withRowAllNegativeInf(row_count.names, row_count.output_name),
            .row_any_finite => |row_count| try current.withRowAnyFinite(row_count.names, row_count.output_name),
            .row_all_finite => |row_count| try current.withRowAllFinite(row_count.names, row_count.output_name),
            .row_any_normal => |row_count| try current.withRowAnyNormal(row_count.names, row_count.output_name),
            .row_all_normal => |row_count| try current.withRowAllNormal(row_count.names, row_count.output_name),
            .row_any_subnormal => |row_count| try current.withRowAnySubnormal(row_count.names, row_count.output_name),
            .row_all_subnormal => |row_count| try current.withRowAllSubnormal(row_count.names, row_count.output_name),
            .row_any_non_finite => |row_count| try current.withRowAnyNonFinite(row_count.names, row_count.output_name),
            .row_all_non_finite => |row_count| try current.withRowAllNonFinite(row_count.names, row_count.output_name),
            .row_nan_count => |row_count| try current.withRowNaNCount(row_count.names, row_count.output_name),
            .row_nan_ratio => |row_count| try current.withRowNaNRatio(row_count.names, row_count.output_name),
            .row_inf_count => |row_count| try current.withRowInfCount(row_count.names, row_count.output_name),
            .row_inf_ratio => |row_count| try current.withRowInfRatio(row_count.names, row_count.output_name),
            .row_positive_inf_count => |row_count| try current.withRowPositiveInfCount(row_count.names, row_count.output_name),
            .row_negative_inf_count => |row_count| try current.withRowNegativeInfCount(row_count.names, row_count.output_name),
            .row_positive_inf_ratio => |row_count| try current.withRowPositiveInfRatio(row_count.names, row_count.output_name),
            .row_negative_inf_ratio => |row_count| try current.withRowNegativeInfRatio(row_count.names, row_count.output_name),
            .row_zero_count => |row_count| try current.withRowZeroCount(row_count.names, row_count.output_name),
            .row_zero_ratio => |row_count| try current.withRowZeroRatio(row_count.names, row_count.output_name),
            .row_positive_zero_count => |row_count| try current.withRowPositiveZeroCount(row_count.names, row_count.output_name),
            .row_negative_zero_count => |row_count| try current.withRowNegativeZeroCount(row_count.names, row_count.output_name),
            .row_positive_zero_ratio => |row_count| try current.withRowPositiveZeroRatio(row_count.names, row_count.output_name),
            .row_negative_zero_ratio => |row_count| try current.withRowNegativeZeroRatio(row_count.names, row_count.output_name),
            .row_non_zero_count => |row_count| try current.withRowNonZeroCount(row_count.names, row_count.output_name),
            .row_non_zero_ratio => |row_count| try current.withRowNonZeroRatio(row_count.names, row_count.output_name),
            .row_first_nan_index => |row_count| try current.withRowFirstNaNIndex(row_count.names, row_count.output_name),
            .row_last_nan_index => |row_count| try current.withRowLastNaNIndex(row_count.names, row_count.output_name),
            .row_first_inf_index => |row_count| try current.withRowFirstInfIndex(row_count.names, row_count.output_name),
            .row_last_inf_index => |row_count| try current.withRowLastInfIndex(row_count.names, row_count.output_name),
            .row_first_positive_inf_index => |row_count| try current.withRowFirstPositiveInfIndex(row_count.names, row_count.output_name),
            .row_last_positive_inf_index => |row_count| try current.withRowLastPositiveInfIndex(row_count.names, row_count.output_name),
            .row_first_negative_inf_index => |row_count| try current.withRowFirstNegativeInfIndex(row_count.names, row_count.output_name),
            .row_last_negative_inf_index => |row_count| try current.withRowLastNegativeInfIndex(row_count.names, row_count.output_name),
            .row_first_finite_index => |row_count| try current.withRowFirstFiniteIndex(row_count.names, row_count.output_name),
            .row_last_finite_index => |row_count| try current.withRowLastFiniteIndex(row_count.names, row_count.output_name),
            .row_first_normal_index => |row_count| try current.withRowFirstNormalIndex(row_count.names, row_count.output_name),
            .row_last_normal_index => |row_count| try current.withRowLastNormalIndex(row_count.names, row_count.output_name),
            .row_first_subnormal_index => |row_count| try current.withRowFirstSubnormalIndex(row_count.names, row_count.output_name),
            .row_last_subnormal_index => |row_count| try current.withRowLastSubnormalIndex(row_count.names, row_count.output_name),
            .row_first_non_finite_index => |row_count| try current.withRowFirstNonFiniteIndex(row_count.names, row_count.output_name),
            .row_last_non_finite_index => |row_count| try current.withRowLastNonFiniteIndex(row_count.names, row_count.output_name),
            .row_first_positive_zero_index => |row_count| try current.withRowFirstPositiveZeroIndex(row_count.names, row_count.output_name),
            .row_last_positive_zero_index => |row_count| try current.withRowLastPositiveZeroIndex(row_count.names, row_count.output_name),
            .row_first_negative_zero_index => |row_count| try current.withRowFirstNegativeZeroIndex(row_count.names, row_count.output_name),
            .row_last_negative_zero_index => |row_count| try current.withRowLastNegativeZeroIndex(row_count.names, row_count.output_name),
            .row_first_signbit_index => |row_count| try current.withRowFirstSignBitIndex(row_count.names, row_count.output_name),
            .row_last_signbit_index => |row_count| try current.withRowLastSignBitIndex(row_count.names, row_count.output_name),
            .row_first_zero_index => |row_count| try current.withRowFirstZeroIndex(row_count.names, row_count.output_name),
            .row_last_zero_index => |row_count| try current.withRowLastZeroIndex(row_count.names, row_count.output_name),
            .row_first_non_zero_index => |row_count| try current.withRowFirstNonZeroIndex(row_count.names, row_count.output_name),
            .row_last_non_zero_index => |row_count| try current.withRowLastNonZeroIndex(row_count.names, row_count.output_name),
            .row_first_positive_index => |row_count| try current.withRowFirstPositiveIndex(row_count.names, row_count.output_name),
            .row_last_positive_index => |row_count| try current.withRowLastPositiveIndex(row_count.names, row_count.output_name),
            .row_first_negative_index => |row_count| try current.withRowFirstNegativeIndex(row_count.names, row_count.output_name),
            .row_last_negative_index => |row_count| try current.withRowLastNegativeIndex(row_count.names, row_count.output_name),
            .row_positive_count => |row_count| try current.withRowPositiveCount(row_count.names, row_count.output_name),
            .row_positive_ratio => |row_count| try current.withRowPositiveRatio(row_count.names, row_count.output_name),
            .row_signbit_count => |row_count| try current.withRowSignBitCount(row_count.names, row_count.output_name),
            .row_signbit_ratio => |row_count| try current.withRowSignBitRatio(row_count.names, row_count.output_name),
            .row_negative_count => |row_count| try current.withRowNegativeCount(row_count.names, row_count.output_name),
            .row_negative_ratio => |row_count| try current.withRowNegativeRatio(row_count.names, row_count.output_name),
            .row_finite_count => |row_count| try current.withRowFiniteCount(row_count.names, row_count.output_name),
            .row_finite_ratio => |row_count| try current.withRowFiniteRatio(row_count.names, row_count.output_name),
            .row_normal_count => |row_count| try current.withRowNormalCount(row_count.names, row_count.output_name),
            .row_normal_ratio => |row_count| try current.withRowNormalRatio(row_count.names, row_count.output_name),
            .row_subnormal_count => |row_count| try current.withRowSubnormalCount(row_count.names, row_count.output_name),
            .row_subnormal_ratio => |row_count| try current.withRowSubnormalRatio(row_count.names, row_count.output_name),
            .row_non_finite_count => |row_count| try current.withRowNonFiniteCount(row_count.names, row_count.output_name),
            .row_non_finite_ratio => |row_count| try current.withRowNonFiniteRatio(row_count.names, row_count.output_name),
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
            .filter_between_column => |range| if (range.keep_inside)
                try current.filterBetweenColumnWithDeviceScalars(range.name, range.lower, range.upper, range.lower_inclusive, range.upper_inclusive)
            else
                try current.filterOutsideColumnWithDeviceScalars(range.name, range.lower, range.upper, range.lower_inclusive, range.upper_inclusive),
            .filter_isin_column => |membership| if (membership.invert)
                try current.filterNotInColumn(membership.input_name, membership.test_name)
            else
                try current.filterIsInColumn(membership.input_name, membership.test_name),
            .filter_isin_values => |membership| blk: {
                var mask = try current.isinColumnValuesWithDeviceColumn(membership.input_name, membership.values, membership.invert);
                defer mask.deinit();
                break :blk try current.filterColumnMask(mask);
            },
            .drop_rows_by_mask_column => |name| try current.dropRowsByColumnMask(name),
            .where_indices_column => |predicate| try current.whereIndicesColumn(predicate.name, predicate.output_name),
            .filter_scalar => |filter_op| blk: {
                var mask = try current.compareColumnScalarWithDeviceScalar(filter_op.name, filter_op.scalar, filter_op.op);
                defer mask.deinit();
                break :blk if (filter_op.keep_matches)
                    try current.filterColumnMask(mask)
                else
                    try current.dropColumnMask(mask);
            },
            .group_by_count => |group| try current.groupByCount(group.key_name, group.output_name),
            .group_by_count_on => |group| try current.groupByCountOn(group.key_names, group.output_name),
            .group_by_value => |group| switch (group.aggregation) {
                .sum => try current.groupBySum(group.key_name, group.value_name, group.output_name),
                .prod => try current.groupByProd(group.key_name, group.value_name, group.output_name),
                .min => try current.groupByMin(group.key_name, group.value_name, group.output_name),
                .max => try current.groupByMax(group.key_name, group.value_name, group.output_name),
                .mean => try current.groupByMean(group.key_name, group.value_name, group.output_name),
                .first => try current.groupByFirst(group.key_name, group.value_name, group.output_name),
                .last => try current.groupByLast(group.key_name, group.value_name, group.output_name),
                .first_row => try current.groupByFirstRow(group.key_name, group.value_name, group.output_name),
                .last_row => try current.groupByLastRow(group.key_name, group.value_name, group.output_name),
                .nth => try current.groupByNth(group.key_name, group.value_name, group.output_name, group.index),
                .nth_row => try current.groupByNthRow(group.key_name, group.value_name, group.output_name, group.index),
                .nth_index => try current.groupByNthIndex(group.key_name, group.value_name, group.output_name, group.index),
                .nth_row_index => try current.groupByNthRowIndex(group.key_name, group.value_name, group.output_name, group.index),
                .n_unique => try current.groupByNUnique(group.key_name, group.value_name, group.output_name),
                .mode => try current.groupByMode(group.key_name, group.value_name, group.output_name),
                .mode_count => try current.groupByModeCount(group.key_name, group.value_name, group.output_name),
                .mode_ratio => try current.groupByModeRatio(group.key_name, group.value_name, group.output_name),
                .mode_margin => try current.groupByModeMargin(group.key_name, group.value_name, group.output_name),
                .mode_margin_ratio => try current.groupByModeMarginRatio(group.key_name, group.value_name, group.output_name),
                .entropy => try current.groupByEntropy(group.key_name, group.value_name, group.output_name),
                .gini_impurity => try current.groupByGiniImpurity(group.key_name, group.value_name, group.output_name),
                .perplexity => try current.groupByPerplexity(group.key_name, group.value_name, group.output_name),
                .inverse_simpson => try current.groupByInverseSimpson(group.key_name, group.value_name, group.output_name),
                .simpson_concentration => try current.groupBySimpsonConcentration(group.key_name, group.value_name, group.output_name),
                .evenness => try current.groupByEvenness(group.key_name, group.value_name, group.output_name),
                .gini_mean_diff => try current.groupByGiniMeanDiff(group.key_name, group.value_name, group.output_name),
                .gini_coefficient => try current.groupByGiniCoefficient(group.key_name, group.value_name, group.output_name),
                .mean_abs_dev => try current.groupByMeanAbsDev(group.key_name, group.value_name, group.output_name),
                .mean_abs_dev_ratio => try current.groupByMeanAbsDevRatio(group.key_name, group.value_name, group.output_name),
                .median => try current.groupByMedian(group.key_name, group.value_name, group.output_name),
                .quantile => try current.groupByQuantile(group.key_name, group.value_name, group.output_name, group.quantile),
                .iqr => try current.groupByIqr(group.key_name, group.value_name, group.output_name),
                .mad => try current.groupByMad(group.key_name, group.value_name, group.output_name),
                .trimmed_mean => try current.groupByTrimmedMean(group.key_name, group.value_name, group.output_name, group.quantile),
                .winsorized_mean => try current.groupByWinsorizedMean(group.key_name, group.value_name, group.output_name, group.quantile),
                .interdecile_range => try current.groupByInterdecileRange(group.key_name, group.value_name, group.output_name),
                .midhinge => try current.groupByMidhinge(group.key_name, group.value_name, group.output_name),
                .trimean => try current.groupByTrimean(group.key_name, group.value_name, group.output_name),
                .bowley_skewness => try current.groupByBowleySkewness(group.key_name, group.value_name, group.output_name),
                .quartile_coeff_dispersion => try current.groupByQuartileCoeffDispersion(group.key_name, group.value_name, group.output_name),
                .kelley_skewness => try current.groupByKelleySkewness(group.key_name, group.value_name, group.output_name),
                .variance => try current.groupByVariance(group.key_name, group.value_name, group.output_name),
                .stddev => try current.groupByStddev(group.key_name, group.value_name, group.output_name),
                .sem => try current.groupBySem(group.key_name, group.value_name, group.output_name),
                .cv => try current.groupByCv(group.key_name, group.value_name, group.output_name),
                .fano => try current.groupByFano(group.key_name, group.value_name, group.output_name),
                .skewness => try current.groupBySkewness(group.key_name, group.value_name, group.output_name),
                .kurtosis => try current.groupByKurtosis(group.key_name, group.value_name, group.output_name),
                .magnitude_variance => try current.groupByMagnitudeVariance(group.key_name, group.value_name, group.output_name),
                .magnitude_stddev => try current.groupByMagnitudeStddev(group.key_name, group.value_name, group.output_name),
                .magnitude_sem => try current.groupByMagnitudeSem(group.key_name, group.value_name, group.output_name),
                .magnitude_cv => try current.groupByMagnitudeCv(group.key_name, group.value_name, group.output_name),
                .magnitude_fano => try current.groupByMagnitudeFano(group.key_name, group.value_name, group.output_name),
                .magnitude_skewness => try current.groupByMagnitudeSkewness(group.key_name, group.value_name, group.output_name),
                .magnitude_kurtosis => try current.groupByMagnitudeKurtosis(group.key_name, group.value_name, group.output_name),
                .mean_abs => try current.groupByMeanAbs(group.key_name, group.value_name, group.output_name),
                .mean_square => try current.groupByMeanSquare(group.key_name, group.value_name, group.output_name),
                .rms => try current.groupByRms(group.key_name, group.value_name, group.output_name),
                .l1_norm => try current.groupByL1Norm(group.key_name, group.value_name, group.output_name),
                .l2_norm => try current.groupByL2Norm(group.key_name, group.value_name, group.output_name),
                .max_abs => try current.groupByMaxAbs(group.key_name, group.value_name, group.output_name),
                .min_abs => try current.groupByMinAbs(group.key_name, group.value_name, group.output_name),
                .hhi => try current.groupByHhi(group.key_name, group.value_name, group.output_name),
                .magnitude_normalized_hhi => try current.groupByMagnitudeNormalizedHhi(group.key_name, group.value_name, group.output_name),
                .magnitude_sparsity => try current.groupByMagnitudeSparsity(group.key_name, group.value_name, group.output_name),
                .magnitude_inverse_simpson => try current.groupByMagnitudeInverseSimpson(group.key_name, group.value_name, group.output_name),
                .magnitude_simpson_evenness => try current.groupByMagnitudeSimpsonEvenness(group.key_name, group.value_name, group.output_name),
                .magnitude_dominance => try current.groupByMagnitudeDominance(group.key_name, group.value_name, group.output_name),
                .magnitude_dominance_margin => try current.groupByMagnitudeDominanceMargin(group.key_name, group.value_name, group.output_name),
                .magnitude_entropy => try current.groupByMagnitudeEntropy(group.key_name, group.value_name, group.output_name),
                .magnitude_perplexity => try current.groupByMagnitudePerplexity(group.key_name, group.value_name, group.output_name),
                .magnitude_evenness => try current.groupByMagnitudeEvenness(group.key_name, group.value_name, group.output_name),
                .geometric_mean => try current.groupByGeometricMean(group.key_name, group.value_name, group.output_name),
                .harmonic_mean => try current.groupByHarmonicMean(group.key_name, group.value_name, group.output_name),
                .logsumexp => try current.groupByLogSumExp(group.key_name, group.value_name, group.output_name),
                .logmeanexp => try current.groupByLogMeanExp(group.key_name, group.value_name, group.output_name),
                .ptp => try current.groupByPtp(group.key_name, group.value_name, group.output_name),
                .midrange => try current.groupByMidrange(group.key_name, group.value_name, group.output_name),
                .range_coeff => try current.groupByRangeCoeff(group.key_name, group.value_name, group.output_name),
                .any => try current.groupByAny(group.key_name, group.value_name, group.output_name),
                .all => try current.groupByAll(group.key_name, group.value_name, group.output_name),
                .true_count => try current.groupByTrueCount(group.key_name, group.value_name, group.output_name),
                .false_count => try current.groupByFalseCount(group.key_name, group.value_name, group.output_name),
                .true_ratio => try current.groupByTrueRatio(group.key_name, group.value_name, group.output_name),
                .false_ratio => try current.groupByFalseRatio(group.key_name, group.value_name, group.output_name),
                .first_true_index => try current.groupByFirstTrueIndex(group.key_name, group.value_name, group.output_name),
                .last_true_index => try current.groupByLastTrueIndex(group.key_name, group.value_name, group.output_name),
                .first_false_index => try current.groupByFirstFalseIndex(group.key_name, group.value_name, group.output_name),
                .last_false_index => try current.groupByLastFalseIndex(group.key_name, group.value_name, group.output_name),
                .any_valid => try current.groupByAnyValid(group.key_name, group.value_name, group.output_name),
                .all_valid => try current.groupByAllValid(group.key_name, group.value_name, group.output_name),
                .any_null => try current.groupByAnyNull(group.key_name, group.value_name, group.output_name),
                .all_null => try current.groupByAllNull(group.key_name, group.value_name, group.output_name),
                .valid_count => try current.groupByValidCount(group.key_name, group.value_name, group.output_name),
                .null_count => try current.groupByNullCount(group.key_name, group.value_name, group.output_name),
                .valid_ratio => try current.groupByValidRatio(group.key_name, group.value_name, group.output_name),
                .null_ratio => try current.groupByNullRatio(group.key_name, group.value_name, group.output_name),
                .first_valid_index => try current.groupByFirstValidIndex(group.key_name, group.value_name, group.output_name),
                .last_valid_index => try current.groupByLastValidIndex(group.key_name, group.value_name, group.output_name),
                .first_null_index => try current.groupByFirstNullIndex(group.key_name, group.value_name, group.output_name),
                .last_null_index => try current.groupByLastNullIndex(group.key_name, group.value_name, group.output_name),
                .nan_count => try current.groupByNaNCount(group.key_name, group.value_name, group.output_name),
                .nan_ratio => try current.groupByNaNRatio(group.key_name, group.value_name, group.output_name),
                .inf_count => try current.groupByInfCount(group.key_name, group.value_name, group.output_name),
                .inf_ratio => try current.groupByInfRatio(group.key_name, group.value_name, group.output_name),
                .positive_inf_count => try current.groupByPositiveInfCount(group.key_name, group.value_name, group.output_name),
                .positive_inf_ratio => try current.groupByPositiveInfRatio(group.key_name, group.value_name, group.output_name),
                .negative_inf_count => try current.groupByNegativeInfCount(group.key_name, group.value_name, group.output_name),
                .negative_inf_ratio => try current.groupByNegativeInfRatio(group.key_name, group.value_name, group.output_name),
                .first_nan_index => try current.groupByFirstNaNIndex(group.key_name, group.value_name, group.output_name),
                .last_nan_index => try current.groupByLastNaNIndex(group.key_name, group.value_name, group.output_name),
                .first_inf_index => try current.groupByFirstInfIndex(group.key_name, group.value_name, group.output_name),
                .last_inf_index => try current.groupByLastInfIndex(group.key_name, group.value_name, group.output_name),
                .first_positive_inf_index => try current.groupByFirstPositiveInfIndex(group.key_name, group.value_name, group.output_name),
                .last_positive_inf_index => try current.groupByLastPositiveInfIndex(group.key_name, group.value_name, group.output_name),
                .first_negative_inf_index => try current.groupByFirstNegativeInfIndex(group.key_name, group.value_name, group.output_name),
                .last_negative_inf_index => try current.groupByLastNegativeInfIndex(group.key_name, group.value_name, group.output_name),
                .finite_count => try current.groupByFiniteCount(group.key_name, group.value_name, group.output_name),
                .finite_ratio => try current.groupByFiniteRatio(group.key_name, group.value_name, group.output_name),
                .first_finite_index => try current.groupByFirstFiniteIndex(group.key_name, group.value_name, group.output_name),
                .last_finite_index => try current.groupByLastFiniteIndex(group.key_name, group.value_name, group.output_name),
                .normal_count => try current.groupByNormalCount(group.key_name, group.value_name, group.output_name),
                .normal_ratio => try current.groupByNormalRatio(group.key_name, group.value_name, group.output_name),
                .first_normal_index => try current.groupByFirstNormalIndex(group.key_name, group.value_name, group.output_name),
                .last_normal_index => try current.groupByLastNormalIndex(group.key_name, group.value_name, group.output_name),
                .subnormal_count => try current.groupBySubnormalCount(group.key_name, group.value_name, group.output_name),
                .subnormal_ratio => try current.groupBySubnormalRatio(group.key_name, group.value_name, group.output_name),
                .first_subnormal_index => try current.groupByFirstSubnormalIndex(group.key_name, group.value_name, group.output_name),
                .last_subnormal_index => try current.groupByLastSubnormalIndex(group.key_name, group.value_name, group.output_name),
                .non_finite_count => try current.groupByNonFiniteCount(group.key_name, group.value_name, group.output_name),
                .non_finite_ratio => try current.groupByNonFiniteRatio(group.key_name, group.value_name, group.output_name),
                .first_non_finite_index => try current.groupByFirstNonFiniteIndex(group.key_name, group.value_name, group.output_name),
                .last_non_finite_index => try current.groupByLastNonFiniteIndex(group.key_name, group.value_name, group.output_name),
                .zero_count => try current.groupByZeroCount(group.key_name, group.value_name, group.output_name),
                .zero_ratio => try current.groupByZeroRatio(group.key_name, group.value_name, group.output_name),
                .first_zero_index => try current.groupByFirstZeroIndex(group.key_name, group.value_name, group.output_name),
                .last_zero_index => try current.groupByLastZeroIndex(group.key_name, group.value_name, group.output_name),
                .positive_zero_count => try current.groupByPositiveZeroCount(group.key_name, group.value_name, group.output_name),
                .positive_zero_ratio => try current.groupByPositiveZeroRatio(group.key_name, group.value_name, group.output_name),
                .negative_zero_count => try current.groupByNegativeZeroCount(group.key_name, group.value_name, group.output_name),
                .negative_zero_ratio => try current.groupByNegativeZeroRatio(group.key_name, group.value_name, group.output_name),
                .first_positive_zero_index => try current.groupByFirstPositiveZeroIndex(group.key_name, group.value_name, group.output_name),
                .last_positive_zero_index => try current.groupByLastPositiveZeroIndex(group.key_name, group.value_name, group.output_name),
                .first_negative_zero_index => try current.groupByFirstNegativeZeroIndex(group.key_name, group.value_name, group.output_name),
                .last_negative_zero_index => try current.groupByLastNegativeZeroIndex(group.key_name, group.value_name, group.output_name),
                .non_zero_count => try current.groupByNonZeroCount(group.key_name, group.value_name, group.output_name),
                .non_zero_ratio => try current.groupByNonZeroRatio(group.key_name, group.value_name, group.output_name),
                .first_non_zero_index => try current.groupByFirstNonZeroIndex(group.key_name, group.value_name, group.output_name),
                .last_non_zero_index => try current.groupByLastNonZeroIndex(group.key_name, group.value_name, group.output_name),
                .positive_count => try current.groupByPositiveCount(group.key_name, group.value_name, group.output_name),
                .positive_ratio => try current.groupByPositiveRatio(group.key_name, group.value_name, group.output_name),
                .first_positive_index => try current.groupByFirstPositiveIndex(group.key_name, group.value_name, group.output_name),
                .last_positive_index => try current.groupByLastPositiveIndex(group.key_name, group.value_name, group.output_name),
                .signbit_count => try current.groupBySignBitCount(group.key_name, group.value_name, group.output_name),
                .signbit_ratio => try current.groupBySignBitRatio(group.key_name, group.value_name, group.output_name),
                .first_signbit_index => try current.groupByFirstSignBitIndex(group.key_name, group.value_name, group.output_name),
                .last_signbit_index => try current.groupByLastSignBitIndex(group.key_name, group.value_name, group.output_name),
                .negative_count => try current.groupByNegativeCount(group.key_name, group.value_name, group.output_name),
                .negative_ratio => try current.groupByNegativeRatio(group.key_name, group.value_name, group.output_name),
                .first_negative_index => try current.groupByFirstNegativeIndex(group.key_name, group.value_name, group.output_name),
                .last_negative_index => try current.groupByLastNegativeIndex(group.key_name, group.value_name, group.output_name),
                .argmin => try current.groupByArgMin(group.key_name, group.value_name, group.output_name),
                .argmax => try current.groupByArgMax(group.key_name, group.value_name, group.output_name),
            },
            .group_by_value_on => |group| switch (group.aggregation) {
                .sum => try current.groupBySumOn(group.key_names, group.value_name, group.output_name),
                .prod => try current.groupByProdOn(group.key_names, group.value_name, group.output_name),
                .min => try current.groupByMinOn(group.key_names, group.value_name, group.output_name),
                .max => try current.groupByMaxOn(group.key_names, group.value_name, group.output_name),
                .mean => try current.groupByMeanOn(group.key_names, group.value_name, group.output_name),
                .first => try current.groupByFirstOn(group.key_names, group.value_name, group.output_name),
                .last => try current.groupByLastOn(group.key_names, group.value_name, group.output_name),
                .first_row => try current.groupByFirstRowOn(group.key_names, group.value_name, group.output_name),
                .last_row => try current.groupByLastRowOn(group.key_names, group.value_name, group.output_name),
                .nth => try current.groupByNthOn(group.key_names, group.value_name, group.output_name, group.index),
                .nth_row => try current.groupByNthRowOn(group.key_names, group.value_name, group.output_name, group.index),
                .nth_index => try current.groupByNthIndexOn(group.key_names, group.value_name, group.output_name, group.index),
                .nth_row_index => try current.groupByNthRowIndexOn(group.key_names, group.value_name, group.output_name, group.index),
                .n_unique => try current.groupByNUniqueOn(group.key_names, group.value_name, group.output_name),
                .mode => try current.groupByModeOn(group.key_names, group.value_name, group.output_name),
                .mode_count => try current.groupByModeCountOn(group.key_names, group.value_name, group.output_name),
                .mode_ratio => try current.groupByModeRatioOn(group.key_names, group.value_name, group.output_name),
                .mode_margin => try current.groupByModeMarginOn(group.key_names, group.value_name, group.output_name),
                .mode_margin_ratio => try current.groupByModeMarginRatioOn(group.key_names, group.value_name, group.output_name),
                .entropy => try current.groupByEntropyOn(group.key_names, group.value_name, group.output_name),
                .gini_impurity => try current.groupByGiniImpurityOn(group.key_names, group.value_name, group.output_name),
                .perplexity => try current.groupByPerplexityOn(group.key_names, group.value_name, group.output_name),
                .inverse_simpson => try current.groupByInverseSimpsonOn(group.key_names, group.value_name, group.output_name),
                .simpson_concentration => try current.groupBySimpsonConcentrationOn(group.key_names, group.value_name, group.output_name),
                .evenness => try current.groupByEvennessOn(group.key_names, group.value_name, group.output_name),
                .gini_mean_diff => try current.groupByGiniMeanDiffOn(group.key_names, group.value_name, group.output_name),
                .gini_coefficient => try current.groupByGiniCoefficientOn(group.key_names, group.value_name, group.output_name),
                .mean_abs_dev => try current.groupByMeanAbsDevOn(group.key_names, group.value_name, group.output_name),
                .mean_abs_dev_ratio => try current.groupByMeanAbsDevRatioOn(group.key_names, group.value_name, group.output_name),
                .median => try current.groupByMedianOn(group.key_names, group.value_name, group.output_name),
                .quantile => try current.groupByQuantileOn(group.key_names, group.value_name, group.output_name, group.quantile),
                .iqr => try current.groupByIqrOn(group.key_names, group.value_name, group.output_name),
                .mad => try current.groupByMadOn(group.key_names, group.value_name, group.output_name),
                .trimmed_mean => try current.groupByTrimmedMeanOn(group.key_names, group.value_name, group.output_name, group.quantile),
                .winsorized_mean => try current.groupByWinsorizedMeanOn(group.key_names, group.value_name, group.output_name, group.quantile),
                .interdecile_range => try current.groupByInterdecileRangeOn(group.key_names, group.value_name, group.output_name),
                .midhinge => try current.groupByMidhingeOn(group.key_names, group.value_name, group.output_name),
                .trimean => try current.groupByTrimeanOn(group.key_names, group.value_name, group.output_name),
                .bowley_skewness => try current.groupByBowleySkewnessOn(group.key_names, group.value_name, group.output_name),
                .quartile_coeff_dispersion => try current.groupByQuartileCoeffDispersionOn(group.key_names, group.value_name, group.output_name),
                .kelley_skewness => try current.groupByKelleySkewnessOn(group.key_names, group.value_name, group.output_name),
                .variance => try current.groupByVarianceOn(group.key_names, group.value_name, group.output_name),
                .stddev => try current.groupByStddevOn(group.key_names, group.value_name, group.output_name),
                .sem => try current.groupBySemOn(group.key_names, group.value_name, group.output_name),
                .cv => try current.groupByCvOn(group.key_names, group.value_name, group.output_name),
                .fano => try current.groupByFanoOn(group.key_names, group.value_name, group.output_name),
                .skewness => try current.groupBySkewnessOn(group.key_names, group.value_name, group.output_name),
                .kurtosis => try current.groupByKurtosisOn(group.key_names, group.value_name, group.output_name),
                .magnitude_variance => try current.groupByMagnitudeVarianceOn(group.key_names, group.value_name, group.output_name),
                .magnitude_stddev => try current.groupByMagnitudeStddevOn(group.key_names, group.value_name, group.output_name),
                .magnitude_sem => try current.groupByMagnitudeSemOn(group.key_names, group.value_name, group.output_name),
                .magnitude_cv => try current.groupByMagnitudeCvOn(group.key_names, group.value_name, group.output_name),
                .magnitude_fano => try current.groupByMagnitudeFanoOn(group.key_names, group.value_name, group.output_name),
                .magnitude_skewness => try current.groupByMagnitudeSkewnessOn(group.key_names, group.value_name, group.output_name),
                .magnitude_kurtosis => try current.groupByMagnitudeKurtosisOn(group.key_names, group.value_name, group.output_name),
                .mean_abs => try current.groupByMeanAbsOn(group.key_names, group.value_name, group.output_name),
                .mean_square => try current.groupByMeanSquareOn(group.key_names, group.value_name, group.output_name),
                .rms => try current.groupByRmsOn(group.key_names, group.value_name, group.output_name),
                .l1_norm => try current.groupByL1NormOn(group.key_names, group.value_name, group.output_name),
                .l2_norm => try current.groupByL2NormOn(group.key_names, group.value_name, group.output_name),
                .max_abs => try current.groupByMaxAbsOn(group.key_names, group.value_name, group.output_name),
                .min_abs => try current.groupByMinAbsOn(group.key_names, group.value_name, group.output_name),
                .hhi => try current.groupByHhiOn(group.key_names, group.value_name, group.output_name),
                .magnitude_normalized_hhi => try current.groupByMagnitudeNormalizedHhiOn(group.key_names, group.value_name, group.output_name),
                .magnitude_sparsity => try current.groupByMagnitudeSparsityOn(group.key_names, group.value_name, group.output_name),
                .magnitude_inverse_simpson => try current.groupByMagnitudeInverseSimpsonOn(group.key_names, group.value_name, group.output_name),
                .magnitude_simpson_evenness => try current.groupByMagnitudeSimpsonEvennessOn(group.key_names, group.value_name, group.output_name),
                .magnitude_dominance => try current.groupByMagnitudeDominanceOn(group.key_names, group.value_name, group.output_name),
                .magnitude_dominance_margin => try current.groupByMagnitudeDominanceMarginOn(group.key_names, group.value_name, group.output_name),
                .magnitude_entropy => try current.groupByMagnitudeEntropyOn(group.key_names, group.value_name, group.output_name),
                .magnitude_perplexity => try current.groupByMagnitudePerplexityOn(group.key_names, group.value_name, group.output_name),
                .magnitude_evenness => try current.groupByMagnitudeEvennessOn(group.key_names, group.value_name, group.output_name),
                .geometric_mean => try current.groupByGeometricMeanOn(group.key_names, group.value_name, group.output_name),
                .harmonic_mean => try current.groupByHarmonicMeanOn(group.key_names, group.value_name, group.output_name),
                .logsumexp => try current.groupByLogSumExpOn(group.key_names, group.value_name, group.output_name),
                .logmeanexp => try current.groupByLogMeanExpOn(group.key_names, group.value_name, group.output_name),
                .ptp => try current.groupByPtpOn(group.key_names, group.value_name, group.output_name),
                .midrange => try current.groupByMidrangeOn(group.key_names, group.value_name, group.output_name),
                .range_coeff => try current.groupByRangeCoeffOn(group.key_names, group.value_name, group.output_name),
                .any => try current.groupByAnyOn(group.key_names, group.value_name, group.output_name),
                .all => try current.groupByAllOn(group.key_names, group.value_name, group.output_name),
                .true_count => try current.groupByTrueCountOn(group.key_names, group.value_name, group.output_name),
                .false_count => try current.groupByFalseCountOn(group.key_names, group.value_name, group.output_name),
                .true_ratio => try current.groupByTrueRatioOn(group.key_names, group.value_name, group.output_name),
                .false_ratio => try current.groupByFalseRatioOn(group.key_names, group.value_name, group.output_name),
                .first_true_index => try current.groupByFirstTrueIndexOn(group.key_names, group.value_name, group.output_name),
                .last_true_index => try current.groupByLastTrueIndexOn(group.key_names, group.value_name, group.output_name),
                .first_false_index => try current.groupByFirstFalseIndexOn(group.key_names, group.value_name, group.output_name),
                .last_false_index => try current.groupByLastFalseIndexOn(group.key_names, group.value_name, group.output_name),
                .any_valid => try current.groupByAnyValidOn(group.key_names, group.value_name, group.output_name),
                .all_valid => try current.groupByAllValidOn(group.key_names, group.value_name, group.output_name),
                .any_null => try current.groupByAnyNullOn(group.key_names, group.value_name, group.output_name),
                .all_null => try current.groupByAllNullOn(group.key_names, group.value_name, group.output_name),
                .valid_count => try current.groupByValidCountOn(group.key_names, group.value_name, group.output_name),
                .null_count => try current.groupByNullCountOn(group.key_names, group.value_name, group.output_name),
                .valid_ratio => try current.groupByValidRatioOn(group.key_names, group.value_name, group.output_name),
                .null_ratio => try current.groupByNullRatioOn(group.key_names, group.value_name, group.output_name),
                .first_valid_index => try current.groupByFirstValidIndexOn(group.key_names, group.value_name, group.output_name),
                .last_valid_index => try current.groupByLastValidIndexOn(group.key_names, group.value_name, group.output_name),
                .first_null_index => try current.groupByFirstNullIndexOn(group.key_names, group.value_name, group.output_name),
                .last_null_index => try current.groupByLastNullIndexOn(group.key_names, group.value_name, group.output_name),
                .nan_count => try current.groupByNaNCountOn(group.key_names, group.value_name, group.output_name),
                .nan_ratio => try current.groupByNaNRatioOn(group.key_names, group.value_name, group.output_name),
                .inf_count => try current.groupByInfCountOn(group.key_names, group.value_name, group.output_name),
                .inf_ratio => try current.groupByInfRatioOn(group.key_names, group.value_name, group.output_name),
                .positive_inf_count => try current.groupByPositiveInfCountOn(group.key_names, group.value_name, group.output_name),
                .positive_inf_ratio => try current.groupByPositiveInfRatioOn(group.key_names, group.value_name, group.output_name),
                .negative_inf_count => try current.groupByNegativeInfCountOn(group.key_names, group.value_name, group.output_name),
                .negative_inf_ratio => try current.groupByNegativeInfRatioOn(group.key_names, group.value_name, group.output_name),
                .first_nan_index => try current.groupByFirstNaNIndexOn(group.key_names, group.value_name, group.output_name),
                .last_nan_index => try current.groupByLastNaNIndexOn(group.key_names, group.value_name, group.output_name),
                .first_inf_index => try current.groupByFirstInfIndexOn(group.key_names, group.value_name, group.output_name),
                .last_inf_index => try current.groupByLastInfIndexOn(group.key_names, group.value_name, group.output_name),
                .first_positive_inf_index => try current.groupByFirstPositiveInfIndexOn(group.key_names, group.value_name, group.output_name),
                .last_positive_inf_index => try current.groupByLastPositiveInfIndexOn(group.key_names, group.value_name, group.output_name),
                .first_negative_inf_index => try current.groupByFirstNegativeInfIndexOn(group.key_names, group.value_name, group.output_name),
                .last_negative_inf_index => try current.groupByLastNegativeInfIndexOn(group.key_names, group.value_name, group.output_name),
                .finite_count => try current.groupByFiniteCountOn(group.key_names, group.value_name, group.output_name),
                .finite_ratio => try current.groupByFiniteRatioOn(group.key_names, group.value_name, group.output_name),
                .first_finite_index => try current.groupByFirstFiniteIndexOn(group.key_names, group.value_name, group.output_name),
                .last_finite_index => try current.groupByLastFiniteIndexOn(group.key_names, group.value_name, group.output_name),
                .normal_count => try current.groupByNormalCountOn(group.key_names, group.value_name, group.output_name),
                .normal_ratio => try current.groupByNormalRatioOn(group.key_names, group.value_name, group.output_name),
                .first_normal_index => try current.groupByFirstNormalIndexOn(group.key_names, group.value_name, group.output_name),
                .last_normal_index => try current.groupByLastNormalIndexOn(group.key_names, group.value_name, group.output_name),
                .subnormal_count => try current.groupBySubnormalCountOn(group.key_names, group.value_name, group.output_name),
                .subnormal_ratio => try current.groupBySubnormalRatioOn(group.key_names, group.value_name, group.output_name),
                .first_subnormal_index => try current.groupByFirstSubnormalIndexOn(group.key_names, group.value_name, group.output_name),
                .last_subnormal_index => try current.groupByLastSubnormalIndexOn(group.key_names, group.value_name, group.output_name),
                .non_finite_count => try current.groupByNonFiniteCountOn(group.key_names, group.value_name, group.output_name),
                .non_finite_ratio => try current.groupByNonFiniteRatioOn(group.key_names, group.value_name, group.output_name),
                .first_non_finite_index => try current.groupByFirstNonFiniteIndexOn(group.key_names, group.value_name, group.output_name),
                .last_non_finite_index => try current.groupByLastNonFiniteIndexOn(group.key_names, group.value_name, group.output_name),
                .zero_count => try current.groupByZeroCountOn(group.key_names, group.value_name, group.output_name),
                .zero_ratio => try current.groupByZeroRatioOn(group.key_names, group.value_name, group.output_name),
                .first_zero_index => try current.groupByFirstZeroIndexOn(group.key_names, group.value_name, group.output_name),
                .last_zero_index => try current.groupByLastZeroIndexOn(group.key_names, group.value_name, group.output_name),
                .positive_zero_count => try current.groupByPositiveZeroCountOn(group.key_names, group.value_name, group.output_name),
                .positive_zero_ratio => try current.groupByPositiveZeroRatioOn(group.key_names, group.value_name, group.output_name),
                .negative_zero_count => try current.groupByNegativeZeroCountOn(group.key_names, group.value_name, group.output_name),
                .negative_zero_ratio => try current.groupByNegativeZeroRatioOn(group.key_names, group.value_name, group.output_name),
                .first_positive_zero_index => try current.groupByFirstPositiveZeroIndexOn(group.key_names, group.value_name, group.output_name),
                .last_positive_zero_index => try current.groupByLastPositiveZeroIndexOn(group.key_names, group.value_name, group.output_name),
                .first_negative_zero_index => try current.groupByFirstNegativeZeroIndexOn(group.key_names, group.value_name, group.output_name),
                .last_negative_zero_index => try current.groupByLastNegativeZeroIndexOn(group.key_names, group.value_name, group.output_name),
                .non_zero_count => try current.groupByNonZeroCountOn(group.key_names, group.value_name, group.output_name),
                .non_zero_ratio => try current.groupByNonZeroRatioOn(group.key_names, group.value_name, group.output_name),
                .first_non_zero_index => try current.groupByFirstNonZeroIndexOn(group.key_names, group.value_name, group.output_name),
                .last_non_zero_index => try current.groupByLastNonZeroIndexOn(group.key_names, group.value_name, group.output_name),
                .positive_count => try current.groupByPositiveCountOn(group.key_names, group.value_name, group.output_name),
                .positive_ratio => try current.groupByPositiveRatioOn(group.key_names, group.value_name, group.output_name),
                .first_positive_index => try current.groupByFirstPositiveIndexOn(group.key_names, group.value_name, group.output_name),
                .last_positive_index => try current.groupByLastPositiveIndexOn(group.key_names, group.value_name, group.output_name),
                .signbit_count => try current.groupBySignBitCountOn(group.key_names, group.value_name, group.output_name),
                .signbit_ratio => try current.groupBySignBitRatioOn(group.key_names, group.value_name, group.output_name),
                .first_signbit_index => try current.groupByFirstSignBitIndexOn(group.key_names, group.value_name, group.output_name),
                .last_signbit_index => try current.groupByLastSignBitIndexOn(group.key_names, group.value_name, group.output_name),
                .negative_count => try current.groupByNegativeCountOn(group.key_names, group.value_name, group.output_name),
                .negative_ratio => try current.groupByNegativeRatioOn(group.key_names, group.value_name, group.output_name),
                .first_negative_index => try current.groupByFirstNegativeIndexOn(group.key_names, group.value_name, group.output_name),
                .last_negative_index => try current.groupByLastNegativeIndexOn(group.key_names, group.value_name, group.output_name),
                .argmin => try current.groupByArgMinOn(group.key_names, group.value_name, group.output_name),
                .argmax => try current.groupByArgMaxOn(group.key_names, group.value_name, group.output_name),
            },
            .group_by_pair => |group| switch (group.aggregation) {
                .dot => try current.groupByDot(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .cosine_similarity => try current.groupByCosineSimilarity(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .squared_euclidean_distance => try current.groupBySquaredEuclideanDistance(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .euclidean_distance => try current.groupByEuclideanDistance(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .manhattan_distance => try current.groupByManhattanDistance(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .chebyshev_distance => try current.groupByChebyshevDistance(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .canberra_distance => try current.groupByCanberraDistance(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .bray_curtis_distance => try current.groupByBrayCurtisDistance(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .mean_error => try current.groupByMeanError(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .mae => try current.groupByMae(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .mse => try current.groupByMse(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .rmse => try current.groupByRmse(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .mape => try current.groupByMape(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .smape => try current.groupBySmape(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .pair_count => try current.groupByPairCount(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .covariance => try current.groupByCovariance(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .correlation => try current.groupByCorrelation(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
                .beta => try current.groupByBeta(group.key_name, group.lhs_name, group.rhs_name, group.output_name),
            },
            .group_by_pair_on => |group| switch (group.aggregation) {
                .dot => try current.groupByDotOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .cosine_similarity => try current.groupByCosineSimilarityOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .squared_euclidean_distance => try current.groupBySquaredEuclideanDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .euclidean_distance => try current.groupByEuclideanDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .manhattan_distance => try current.groupByManhattanDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .chebyshev_distance => try current.groupByChebyshevDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .canberra_distance => try current.groupByCanberraDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .bray_curtis_distance => try current.groupByBrayCurtisDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .mean_error => try current.groupByMeanErrorOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .mae => try current.groupByMaeOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .mse => try current.groupByMseOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .rmse => try current.groupByRmseOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .mape => try current.groupByMapeOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .smape => try current.groupBySmapeOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .pair_count => try current.groupByPairCountOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .covariance => try current.groupByCovarianceOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .correlation => try current.groupByCorrelationOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
                .beta => try current.groupByBetaOn(group.key_names, group.lhs_name, group.rhs_name, group.output_name),
            },
            .group_by_weighted => |group| switch (group.aggregation) {
                .weighted_mean => try current.groupByWeightedMean(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_variance => try current.groupByWeightedVariance(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_stddev => try current.groupByWeightedStddev(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_quantile => try current.groupByWeightedQuantile(group.key_name, group.value_name, group.weight_name, group.output_name, group.quantile),
                .weighted_median => try current.groupByWeightedMedian(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_iqr => try current.groupByWeightedIqr(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_mad => try current.groupByWeightedMad(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_mode => try current.groupByWeightedMode(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_mode_weight => try current.groupByWeightedModeWeight(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_mode_ratio => try current.groupByWeightedModeRatio(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_mode_margin => try current.groupByWeightedModeMargin(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_mode_margin_ratio => try current.groupByWeightedModeMarginRatio(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_entropy => try current.groupByWeightedEntropy(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_gini_impurity => try current.groupByWeightedGiniImpurity(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_perplexity => try current.groupByWeightedPerplexity(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_inverse_simpson => try current.groupByWeightedInverseSimpson(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_simpson_concentration => try current.groupByWeightedSimpsonConcentration(group.key_name, group.value_name, group.weight_name, group.output_name),
                .weighted_evenness => try current.groupByWeightedEvenness(group.key_name, group.value_name, group.weight_name, group.output_name),
            },
            .group_by_weighted_on => |group| switch (group.aggregation) {
                .weighted_mean => try current.groupByWeightedMeanOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_variance => try current.groupByWeightedVarianceOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_stddev => try current.groupByWeightedStddevOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_quantile => try current.groupByWeightedQuantileOn(group.key_names, group.value_name, group.weight_name, group.output_name, group.quantile),
                .weighted_median => try current.groupByWeightedMedianOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_iqr => try current.groupByWeightedIqrOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_mad => try current.groupByWeightedMadOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_mode => try current.groupByWeightedModeOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_mode_weight => try current.groupByWeightedModeWeightOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_mode_ratio => try current.groupByWeightedModeRatioOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_mode_margin => try current.groupByWeightedModeMarginOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_mode_margin_ratio => try current.groupByWeightedModeMarginRatioOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_entropy => try current.groupByWeightedEntropyOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_gini_impurity => try current.groupByWeightedGiniImpurityOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_perplexity => try current.groupByWeightedPerplexityOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_inverse_simpson => try current.groupByWeightedInverseSimpsonOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_simpson_concentration => try current.groupByWeightedSimpsonConcentrationOn(group.key_names, group.value_name, group.weight_name, group.output_name),
                .weighted_evenness => try current.groupByWeightedEvennessOn(group.key_names, group.value_name, group.weight_name, group.output_name),
            },
            .group_by_weighted_pair => |group| switch (group.aggregation) {
                .weighted_dot => try current.groupByWeightedDot(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_cosine_similarity => try current.groupByWeightedCosineSimilarity(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_squared_euclidean_distance => try current.groupByWeightedSquaredEuclideanDistance(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_euclidean_distance => try current.groupByWeightedEuclideanDistance(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_manhattan_distance => try current.groupByWeightedManhattanDistance(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_chebyshev_distance => try current.groupByWeightedChebyshevDistance(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_canberra_distance => try current.groupByWeightedCanberraDistance(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_bray_curtis_distance => try current.groupByWeightedBrayCurtisDistance(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_mean_error => try current.groupByWeightedMeanError(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_mae => try current.groupByWeightedMae(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_mse => try current.groupByWeightedMse(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_rmse => try current.groupByWeightedRmse(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_mape => try current.groupByWeightedMape(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_smape => try current.groupByWeightedSmape(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_covariance => try current.groupByWeightedCovariance(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name, group.correction),
                .weighted_correlation => try current.groupByWeightedCorrelation(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name, group.correction),
                .weighted_beta => try current.groupByWeightedBeta(group.key_name, group.lhs_name, group.rhs_name, group.weight_name, group.output_name, group.correction),
            },
            .group_by_weighted_pair_on => |group| switch (group.aggregation) {
                .weighted_dot => try current.groupByWeightedDotOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_cosine_similarity => try current.groupByWeightedCosineSimilarityOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_squared_euclidean_distance => try current.groupByWeightedSquaredEuclideanDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_euclidean_distance => try current.groupByWeightedEuclideanDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_manhattan_distance => try current.groupByWeightedManhattanDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_chebyshev_distance => try current.groupByWeightedChebyshevDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_canberra_distance => try current.groupByWeightedCanberraDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_bray_curtis_distance => try current.groupByWeightedBrayCurtisDistanceOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_mean_error => try current.groupByWeightedMeanErrorOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_mae => try current.groupByWeightedMaeOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_mse => try current.groupByWeightedMseOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_rmse => try current.groupByWeightedRmseOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_mape => try current.groupByWeightedMapeOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_smape => try current.groupByWeightedSmapeOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name),
                .weighted_covariance => try current.groupByWeightedCovarianceOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name, group.correction),
                .weighted_correlation => try current.groupByWeightedCorrelationOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name, group.correction),
                .weighted_beta => try current.groupByWeightedBetaOn(group.key_names, group.lhs_name, group.rhs_name, group.weight_name, group.output_name, group.correction),
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
            .concat_columns => |right| try current.concatColumns(right),
            .distinct_rows => try current.distinctRows(),
            .distinct_rows_last => try current.distinctRowsLast(),
            .distinct_rows_none => try current.distinctRowsNone(),
            .distinct_on => |names| try current.distinctOn(names),
            .distinct_on_last => |names| try current.distinctOnLast(names),
            .distinct_on_none => |names| try current.distinctOnNone(names),
            .sort_by => |sort| try current.sortBy(sort.name, sort.options),
            .sort_by_columns => |sort| try current.sortByColumns(sort.names, sort.options),
            .top_k => |top| try current.topKBy(top.name, top.k, top.options),
            .top_k_columns => |top| try current.topKByColumns(top.names, top.k, top.options),
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
            .slice_rows_signed => |slice| try current.sliceRowsSigned(slice.start, slice.length),
            .drop_rows => |row_indices| try current.dropRows(row_indices),
            .drop_rows_mode => |drop_mode| try current.dropRowsMode(drop_mode.row_indices, drop_mode.mode),
            .drop_rows_signed => |row_indices| try current.dropRowsSigned(row_indices),
            .drop_rows_signed_mode => |drop_mode| try current.dropRowsSignedMode(drop_mode.row_indices, drop_mode.mode),
            .drop_row_range => |range| try current.dropRowRange(range.start, range.stop),
            .drop_last_rows => |n| try current.dropLastRows(n),
            .slice_rows_step => |slice| try current.sliceRowsStep(slice.start, slice.stop, slice.step),
            .slice_rows_signed_step => |slice| try current.sliceRowsSignedStep(slice.start, slice.stop, slice.step),
            .stride_rows => |stride| try current.strideRows(stride.start, stride.step),
            .take_rows => |row_indices| try current.take(row_indices),
            .take_rows_optional => |row_indices| try current.takeOptional(row_indices),
            .take_rows_mode => |take_mode| try current.takeMode(take_mode.row_indices, take_mode.mode),
            .take_rows_signed => |row_indices| try current.takeSigned(row_indices),
            .take_rows_signed_mode => |take_mode| try current.takeSignedMode(take_mode.row_indices, take_mode.mode),
            .take_rows_by_column => |name| try current.takeByColumn(name),
            .take_rows_by_column_mode => |take_mode| try current.takeByColumnMode(take_mode.name, take_mode.mode),
            .drop_rows_by_column => |name| try current.dropRowsByColumn(name),
            .drop_rows_by_column_mode => |take_mode| try current.dropRowsByColumnMode(take_mode.name, take_mode.mode),
            .repeat_rows => |repeat_count| try current.repeatRows(repeat_count),
            .tile_rows => |tile_count| try current.tileRows(tile_count),
            .repeat_rows_by => |count_name| try current.repeatRowsByColumn(count_name),
            .sample_rows => |sample| try current.sampleRows(sample.count, sample.seed),
            .sample_rows_fraction => |sample| try current.sampleRowsFraction(sample.fraction, sample.seed),
            .sample_rows_with_replacement => |sample| try current.sampleRowsWithReplacement(sample.count, sample.seed),
            .sample_rows_fraction_with_replacement => |sample| try current.sampleRowsFractionWithReplacement(sample.fraction, sample.seed),
            .roll_rows => |shift| try current.rollRows(shift),
            .shift_rows => |shift| try current.shiftRows(shift),
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
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .sort_by_columns) {
                    const sort = optimized.items[optimized.items.len - 1].sort_by_columns;
                    const names = try names_mod.cloneNameList(self.allocator, sort.names);
                    errdefer names_mod.freeNameList(self.allocator, names);
                    const options = try self.allocator.dupe(std.meta.Elem(@TypeOf(sort.options)), sort.options);
                    errdefer self.allocator.free(options);
                    optimized.items[optimized.items.len - 1].deinit(self.allocator);
                    optimized.items[optimized.items.len - 1] = .{ .top_k_columns = .{
                        .names = names,
                        .options = options,
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
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .top_k_columns) {
                    const top = optimized.items[optimized.items.len - 1].top_k_columns;
                    optimized.items[optimized.items.len - 1] = .{ .top_k_columns = .{
                        .names = top.names,
                        .options = top.options,
                        .k = @min(top.k, n),
                    } };
                    continue;
                }
                if (optimized.items.len != 0 and optimized.items[optimized.items.len - 1] == .slice_rows) {
                    const slice = optimized.items[optimized.items.len - 1].slice_rows;
                    optimized.items[optimized.items.len - 1] = .{ .slice_rows = .{
                        .start = slice.start,
                        .stop = @min(slice.stop, std.math.add(usize, slice.start, n) catch std.math.maxInt(usize)),
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
