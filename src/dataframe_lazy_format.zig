//! Formatting helpers for lazy dataframe plans.

const std = @import("std");

pub fn formatLazyScanPushdown(writer: *std.Io.Writer, pushdown: anytype) std.Io.Writer.Error!void {
    var printed = false;
    if (pushdown.range_predicate) |predicate| {
        try writer.print("range={s}", .{predicate.column});
        printed = true;
    }
    if (pushdown.projection) |names| {
        if (printed) try writer.print(", ", .{});
        try writer.print("projection=[", .{});
        for (names, 0..) |name, i| {
            if (i != 0) try writer.print(",", .{});
            try writer.print("{s}", .{name});
        }
        try writer.print("]", .{});
        printed = true;
    }
    if (!printed) try writer.print("none", .{});
}

pub fn formatLazyOp(writer: *std.Io.Writer, op: anytype) std.Io.Writer.Error!void {
    switch (op) {
        .select => |names| {
            try writer.print("select[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .select_column_indices => |indices| {
            try writer.print("select_column_indices([", .{});
            for (indices, 0..) |index, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{d}", .{index});
            }
            try writer.print("])", .{});
        },
        .select_column_range => |range| try writer.print("select_column_range({d}..{d})", .{ range.start, range.stop }),
        .select_last_columns => |n| try writer.print("select_last_columns({d})", .{n}),
        .drop_column_indices => |indices| {
            try writer.print("drop_column_indices([", .{});
            for (indices, 0..) |index, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{d}", .{index});
            }
            try writer.print("])", .{});
        },
        .drop_column_range => |range| try writer.print("drop_column_range({d}..{d})", .{ range.start, range.stop }),
        .drop_last_columns => |n| try writer.print("drop_last_columns({d})", .{n}),
        .reverse_columns => try writer.print("reverse_columns", .{}),
        .sort_columns_by_name => |sort| try writer.print("sort_columns_by_name(desc={})", .{sort.descending}),
        .select_name_prefix => |pattern| try writer.print("select_name_prefix({s})", .{pattern.pattern}),
        .select_name_suffix => |pattern| try writer.print("select_name_suffix({s})", .{pattern.pattern}),
        .select_name_contains => |pattern| try writer.print("select_name_contains({s})", .{pattern.pattern}),
        .drop_name_prefix => |pattern| try writer.print("drop_name_prefix({s})", .{pattern.pattern}),
        .drop_name_suffix => |pattern| try writer.print("drop_name_suffix({s})", .{pattern.pattern}),
        .drop_name_contains => |pattern| try writer.print("drop_name_contains({s})", .{pattern.pattern}),
        .select_dtypes => |dtypes| {
            try writer.print("select_dtypes[", .{});
            for (dtypes, 0..) |dtype, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{dtype.name()});
            }
            try writer.print("]", .{});
        },
        .select_dtype_class => |class| try writer.print("select_dtype_class({s})", .{@tagName(class)}),
        .drop_dtypes => |dtypes| {
            try writer.print("drop_dtypes[", .{});
            for (dtypes, 0..) |dtype, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{dtype.name()});
            }
            try writer.print("]", .{});
        },
        .drop_dtype_class => |class| try writer.print("drop_dtype_class({s})", .{@tagName(class)}),
        .select_nullable_columns => try writer.print("select_nullable_columns", .{}),
        .select_non_nullable_columns => try writer.print("select_non_nullable_columns", .{}),
        .select_columns_with_nulls => try writer.print("select_columns_with_nulls", .{}),
        .select_columns_without_nulls => try writer.print("select_columns_without_nulls", .{}),
        .drop_nullable_columns => try writer.print("drop_nullable_columns", .{}),
        .drop_non_nullable_columns => try writer.print("drop_non_nullable_columns", .{}),
        .drop_columns_with_nulls => try writer.print("drop_columns_with_nulls", .{}),
        .drop_columns_without_nulls => try writer.print("drop_columns_without_nulls", .{}),
        .select_columns_with_nans => try writer.print("select_columns_with_nans", .{}),
        .select_columns_without_nans => try writer.print("select_columns_without_nans", .{}),
        .drop_columns_with_nans => try writer.print("drop_columns_with_nans", .{}),
        .drop_columns_without_nans => try writer.print("drop_columns_without_nans", .{}),
        .select_columns_with_infs => try writer.print("select_columns_with_infs", .{}),
        .select_columns_without_infs => try writer.print("select_columns_without_infs", .{}),
        .drop_columns_with_infs => try writer.print("drop_columns_with_infs", .{}),
        .drop_columns_without_infs => try writer.print("drop_columns_without_infs", .{}),
        .select_columns_with_positive_infs => try writer.print("select_columns_with_positive_infs", .{}),
        .select_columns_without_positive_infs => try writer.print("select_columns_without_positive_infs", .{}),
        .drop_columns_with_positive_infs => try writer.print("drop_columns_with_positive_infs", .{}),
        .drop_columns_without_positive_infs => try writer.print("drop_columns_without_positive_infs", .{}),
        .select_columns_with_negative_infs => try writer.print("select_columns_with_negative_infs", .{}),
        .select_columns_without_negative_infs => try writer.print("select_columns_without_negative_infs", .{}),
        .drop_columns_with_negative_infs => try writer.print("drop_columns_with_negative_infs", .{}),
        .drop_columns_without_negative_infs => try writer.print("drop_columns_without_negative_infs", .{}),
        .select_columns_with_zeros => try writer.print("select_columns_with_zeros", .{}),
        .select_columns_without_zeros => try writer.print("select_columns_without_zeros", .{}),
        .drop_columns_with_zeros => try writer.print("drop_columns_with_zeros", .{}),
        .drop_columns_without_zeros => try writer.print("drop_columns_without_zeros", .{}),
        .select_columns_with_positive_zeros => try writer.print("select_columns_with_positive_zeros", .{}),
        .select_columns_without_positive_zeros => try writer.print("select_columns_without_positive_zeros", .{}),
        .drop_columns_with_positive_zeros => try writer.print("drop_columns_with_positive_zeros", .{}),
        .drop_columns_without_positive_zeros => try writer.print("drop_columns_without_positive_zeros", .{}),
        .select_columns_with_negative_zeros => try writer.print("select_columns_with_negative_zeros", .{}),
        .select_columns_without_negative_zeros => try writer.print("select_columns_without_negative_zeros", .{}),
        .drop_columns_with_negative_zeros => try writer.print("drop_columns_with_negative_zeros", .{}),
        .drop_columns_without_negative_zeros => try writer.print("drop_columns_without_negative_zeros", .{}),
        .select_columns_with_non_zeros => try writer.print("select_columns_with_non_zeros", .{}),
        .select_columns_without_non_zeros => try writer.print("select_columns_without_non_zeros", .{}),
        .drop_columns_with_non_zeros => try writer.print("drop_columns_with_non_zeros", .{}),
        .drop_columns_without_non_zeros => try writer.print("drop_columns_without_non_zeros", .{}),
        .select_columns_with_positives => try writer.print("select_columns_with_positives", .{}),
        .select_columns_without_positives => try writer.print("select_columns_without_positives", .{}),
        .drop_columns_with_positives => try writer.print("drop_columns_with_positives", .{}),
        .drop_columns_without_positives => try writer.print("drop_columns_without_positives", .{}),
        .select_columns_with_signbits => try writer.print("select_columns_with_signbits", .{}),
        .select_columns_without_signbits => try writer.print("select_columns_without_signbits", .{}),
        .drop_columns_with_signbits => try writer.print("drop_columns_with_signbits", .{}),
        .drop_columns_without_signbits => try writer.print("drop_columns_without_signbits", .{}),
        .select_columns_with_negatives => try writer.print("select_columns_with_negatives", .{}),
        .select_columns_without_negatives => try writer.print("select_columns_without_negatives", .{}),
        .drop_columns_with_negatives => try writer.print("drop_columns_with_negatives", .{}),
        .drop_columns_without_negatives => try writer.print("drop_columns_without_negatives", .{}),
        .select_columns_with_finites => try writer.print("select_columns_with_finites", .{}),
        .select_columns_without_finites => try writer.print("select_columns_without_finites", .{}),
        .drop_columns_with_finites => try writer.print("drop_columns_with_finites", .{}),
        .drop_columns_without_finites => try writer.print("drop_columns_without_finites", .{}),
        .select_columns_with_normals => try writer.print("select_columns_with_normals", .{}),
        .select_columns_without_normals => try writer.print("select_columns_without_normals", .{}),
        .drop_columns_with_normals => try writer.print("drop_columns_with_normals", .{}),
        .drop_columns_without_normals => try writer.print("drop_columns_without_normals", .{}),
        .select_columns_with_subnormals => try writer.print("select_columns_with_subnormals", .{}),
        .select_columns_without_subnormals => try writer.print("select_columns_without_subnormals", .{}),
        .drop_columns_with_subnormals => try writer.print("drop_columns_with_subnormals", .{}),
        .drop_columns_without_subnormals => try writer.print("drop_columns_without_subnormals", .{}),
        .select_columns_with_non_finites => try writer.print("select_columns_with_non_finites", .{}),
        .select_columns_without_non_finites => try writer.print("select_columns_without_non_finites", .{}),
        .drop_columns_with_non_finites => try writer.print("drop_columns_with_non_finites", .{}),
        .drop_columns_without_non_finites => try writer.print("drop_columns_without_non_finites", .{}),
        .with_row_index => |row_index| try writer.print("with_row_index({s}, offset={d})", .{ row_index.name, row_index.offset }),
        .rename_column => |rename| try writer.print("rename_column({s}->{s})", .{ rename.old_name, rename.new_name }),
        .rename_columns => |rename| {
            try writer.print("rename_columns[", .{});
            for (rename.old_names, rename.new_names, 0..) |old_name, new_name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}->{s}", .{ old_name, new_name });
            }
            try writer.print("]", .{});
        },
        .add_column_name_prefix => |pattern| try writer.print("add_column_name_prefix({s})", .{pattern.pattern}),
        .add_column_name_suffix => |pattern| try writer.print("add_column_name_suffix({s})", .{pattern.pattern}),
        .move_column => |move| try writer.print("move_column({s} -> index={d})", .{ move.name, move.target_index }),
        .move_column_before => |move| try writer.print("move_column_before({s} before {s})", .{ move.name, move.anchor_name }),
        .move_column_after => |move| try writer.print("move_column_after({s} after {s})", .{ move.name, move.anchor_name }),
        .copy_column => |copy| try writer.print("copy_column({s}->{s})", .{ copy.source_name, copy.new_name }),
        .copy_column_at => |copy| try writer.print("copy_column_at({s}->{s}, index={d})", .{ copy.source_name, copy.new_name, copy.target_index }),
        .copy_column_before => |copy| try writer.print("copy_column_before({s}->{s} before {s})", .{ copy.source_name, copy.new_name, copy.anchor_name }),
        .copy_column_after => |copy| try writer.print("copy_column_after({s}->{s} after {s})", .{ copy.source_name, copy.new_name, copy.anchor_name }),
        .drop_columns => |names| {
            try writer.print("drop_columns[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .drop_nulls => |names| {
            try writer.print("drop_nulls[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_nulls_column => |name| try writer.print("filter_nulls_column({s})", .{name}),
        .drop_nans => |names| {
            try writer.print("drop_nans[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_nans_column => |name| try writer.print("filter_nans_column({s})", .{name}),
        .drop_infs => |names| {
            try writer.print("drop_infs[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_infs_column => |name| try writer.print("filter_infs_column({s})", .{name}),
        .drop_positive_infs => |names| {
            try writer.print("drop_positive_infs[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_positive_infs_column => |name| try writer.print("filter_positive_infs_column({s})", .{name}),
        .drop_negative_infs => |names| {
            try writer.print("drop_negative_infs[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_negative_infs_column => |name| try writer.print("filter_negative_infs_column({s})", .{name}),
        .drop_zeros => |names| {
            try writer.print("drop_zeros[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_zeros_column => |name| try writer.print("filter_zeros_column({s})", .{name}),
        .drop_positive_zeros => |names| {
            try writer.print("drop_positive_zeros[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_positive_zeros_column => |name| try writer.print("filter_positive_zeros_column({s})", .{name}),
        .drop_negative_zeros => |names| {
            try writer.print("drop_negative_zeros[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_negative_zeros_column => |name| try writer.print("filter_negative_zeros_column({s})", .{name}),
        .drop_non_zeros => |names| {
            try writer.print("drop_non_zeros[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_non_zeros_column => |name| try writer.print("filter_non_zeros_column({s})", .{name}),
        .drop_positives => |names| {
            try writer.print("drop_positives[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_positives_column => |name| try writer.print("filter_positives_column({s})", .{name}),
        .drop_signbits => |names| {
            try writer.print("drop_signbits[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_signbits_column => |name| try writer.print("filter_signbits_column({s})", .{name}),
        .drop_negatives => |names| {
            try writer.print("drop_negatives[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_negatives_column => |name| try writer.print("filter_negatives_column({s})", .{name}),
        .drop_finites => |names| {
            try writer.print("drop_finites[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_finites_column => |name| try writer.print("filter_finites_column({s})", .{name}),
        .drop_normals => |names| {
            try writer.print("drop_normals[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_normals_column => |name| try writer.print("filter_normals_column({s})", .{name}),
        .drop_subnormals => |names| {
            try writer.print("drop_subnormals[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_subnormals_column => |name| try writer.print("filter_subnormals_column({s})", .{name}),
        .drop_non_finites => |names| {
            try writer.print("drop_non_finites[", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]", .{});
        },
        .filter_non_finites_column => |name| try writer.print("filter_non_finites_column({s})", .{name}),
        .filter_mask => |mask| try writer.print("filter_mask(dtype={s}, rows={d})", .{ mask.dtype().name(), mask.len() }),
        .filter_column => |name| try writer.print("filter_column({s})", .{name}),
        .filter_scalar => |filter_op| try writer.print("filter_scalar({s}, op={s}, dtype={s})", .{ filter_op.name, @tagName(filter_op.op), @tagName(filter_op.scalar) }),
        .with_column_binary => |expr| try writer.print("with_column_binary({s}={s} {s} {s})", .{ expr.name, expr.lhs_name, @tagName(expr.op), expr.rhs_name }),
        .with_column_scalar => |expr| try writer.print("with_column_scalar({s}={s} {s} scalar:{s})", .{ expr.name, expr.input_name, @tagName(expr.op), @tagName(expr.scalar) }),
        .with_column_lerp_scalar => |expr| try writer.print("with_column_lerp_scalar({s}=lerp({s}, {s}, weight:{s}))", .{ expr.name, expr.lhs_name, expr.rhs_name, @tagName(expr.scalar) }),
        .with_column_isclose_scalar => |expr| try writer.print("with_column_isclose_scalar({s}=isclose({s}, scalar:{s}, rtol:{s}, atol:{s}, equal_nan={any}))", .{ expr.name, expr.input_name, @tagName(expr.scalar), @tagName(expr.rtol), @tagName(expr.atol), expr.equal_nan }),
        .with_column_logical => |expr| try writer.print("with_column_logical({s}=logical_{s}({s}, {s}))", .{ expr.name, @tagName(expr.op), expr.lhs_name, expr.rhs_name }),
        .with_column_logical_scalar => |expr| try writer.print("with_column_logical_scalar({s}=logical_{s}({s}, scalar:{any}))", .{ expr.name, @tagName(expr.op), expr.input_name, expr.scalar }),
        .with_column_literal => |expr| try writer.print("with_column_literal({s}=scalar:{s})", .{ expr.name, @tagName(expr.scalar) }),
        .with_column_literal_at => |expr| try writer.print("with_column_literal_at({s}=scalar:{s}, index={d})", .{ expr.name, @tagName(expr.scalar), expr.target_index }),
        .with_column_literal_before => |expr| try writer.print("with_column_literal_before({s}=scalar:{s} before {s})", .{ expr.name, @tagName(expr.scalar), expr.anchor_name }),
        .with_column_literal_after => |expr| try writer.print("with_column_literal_after({s}=scalar:{s} after {s})", .{ expr.name, @tagName(expr.scalar), expr.anchor_name }),
        .cast_column => |cast| try writer.print("cast_column({s}->{s})", .{ cast.name, cast.dtype.name() }),
        .fill_null_column => |fill| try writer.print("fill_null_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_nan_column => |fill| try writer.print("fill_nan_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_inf_column => |fill| try writer.print("fill_inf_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_positive_inf_column => |fill| try writer.print("fill_positive_inf_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_negative_inf_column => |fill| try writer.print("fill_negative_inf_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_zero_column => |fill| try writer.print("fill_zero_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_positive_zero_column => |fill| try writer.print("fill_positive_zero_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_negative_zero_column => |fill| try writer.print("fill_negative_zero_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_non_zero_column => |fill| try writer.print("fill_non_zero_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_positive_column => |fill| try writer.print("fill_positive_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_signbit_column => |fill| try writer.print("fill_signbit_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_negative_column => |fill| try writer.print("fill_negative_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_finite_column => |fill| try writer.print("fill_finite_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_normal_column => |fill| try writer.print("fill_normal_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_subnormal_column => |fill| try writer.print("fill_subnormal_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .fill_non_finite_column => |fill| try writer.print("fill_non_finite_column({s}=scalar:{s})", .{ fill.name, @tagName(fill.scalar) }),
        .coalesce_columns => |coalesce| try writer.print("coalesce_columns({s},{s}->{s})", .{ coalesce.primary_name, coalesce.fallback_name, coalesce.output_name }),
        .with_column_abs => |expr| try writer.print("with_column_abs({s}=abs({s}))", .{ expr.name, expr.input_name }),
        .with_column_neg => |expr| try writer.print("with_column_neg({s}=neg({s}))", .{ expr.name, expr.input_name }),
        .with_column_square => |expr| try writer.print("with_column_square({s}=square({s}))", .{ expr.name, expr.input_name }),
        .with_column_reciprocal => |expr| try writer.print("with_column_reciprocal({s}=reciprocal({s}))", .{ expr.name, expr.input_name }),
        .with_column_sign => |expr| try writer.print("with_column_sign({s}=sign({s}))", .{ expr.name, expr.input_name }),
        .with_column_sqrt => |expr| try writer.print("with_column_sqrt({s}=sqrt({s}))", .{ expr.name, expr.input_name }),
        .with_column_rsqrt => |expr| try writer.print("with_column_rsqrt({s}=rsqrt({s}))", .{ expr.name, expr.input_name }),
        .with_column_cbrt => |expr| try writer.print("with_column_cbrt({s}=cbrt({s}))", .{ expr.name, expr.input_name }),
        .with_column_floor => |expr| try writer.print("with_column_floor({s}=floor({s}))", .{ expr.name, expr.input_name }),
        .with_column_ceil => |expr| try writer.print("with_column_ceil({s}=ceil({s}))", .{ expr.name, expr.input_name }),
        .with_column_round => |expr| try writer.print("with_column_round({s}=round({s}))", .{ expr.name, expr.input_name }),
        .with_column_trunc => |expr| try writer.print("with_column_trunc({s}=trunc({s}))", .{ expr.name, expr.input_name }),
        .with_column_deg2rad => |expr| try writer.print("with_column_deg2rad({s}=deg2rad({s}))", .{ expr.name, expr.input_name }),
        .with_column_rad2deg => |expr| try writer.print("with_column_rad2deg({s}=rad2deg({s}))", .{ expr.name, expr.input_name }),
        .with_column_expit => |expr| try writer.print("with_column_expit({s}=expit({s}))", .{ expr.name, expr.input_name }),
        .with_column_logit => |expr| try writer.print("with_column_logit({s}=logit({s}))", .{ expr.name, expr.input_name }),
        .with_column_softplus => |expr| try writer.print("with_column_softplus({s}=softplus({s}))", .{ expr.name, expr.input_name }),
        .with_column_logsigmoid => |expr| try writer.print("with_column_logsigmoid({s}=logsigmoid({s}))", .{ expr.name, expr.input_name }),
        .with_column_relu => |expr| try writer.print("with_column_relu({s}=relu({s}))", .{ expr.name, expr.input_name }),
        .with_column_leaky_relu => |expr| try writer.print("with_column_leaky_relu({s}=leaky_relu({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_relu6 => |expr| try writer.print("with_column_relu6({s}=relu6({s}))", .{ expr.name, expr.input_name }),
        .with_column_pow_scalar => |expr| try writer.print("with_column_pow_scalar({s}=pow({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_floor_div_scalar => |expr| try writer.print("with_column_floor_div_scalar({s}=floor_div({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_mod_scalar => |expr| try writer.print("with_column_mod_scalar({s}=mod({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_remainder_scalar => |expr| try writer.print("with_column_remainder_scalar({s}=remainder({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_log_add_exp_scalar => |expr| try writer.print("with_column_log_add_exp_scalar({s}=log_add_exp({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_log_add_exp2_scalar => |expr| try writer.print("with_column_log_add_exp2_scalar({s}=log_add_exp2({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_xlogy_scalar => |expr| try writer.print("with_column_xlogy_scalar({s}=xlogy({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_fmax_scalar => |expr| try writer.print("with_column_fmax_scalar({s}=fmax({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_fmin_scalar => |expr| try writer.print("with_column_fmin_scalar({s}=fmin({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_hypot_scalar => |expr| try writer.print("with_column_hypot_scalar({s}=hypot({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_atan2_scalar => |expr| try writer.print("with_column_atan2_scalar({s}=atan2({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_next_after_scalar => |expr| try writer.print("with_column_next_after_scalar({s}=next_after({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_copysign_scalar => |expr| try writer.print("with_column_copysign_scalar({s}=copysign({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_heaviside_scalar => |expr| try writer.print("with_column_heaviside_scalar({s}=heaviside({s}, value_at_zero:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_ldexp_scalar => |expr| try writer.print("with_column_ldexp_scalar({s}=ldexp({s}, exponent:{d}))", .{ expr.name, expr.input_name, expr.exponent }),
        .with_column_threshold => |expr| try writer.print("with_column_threshold({s}=threshold({s}, threshold:{s}, replacement:{s}))", .{ expr.name, expr.input_name, @tagName(expr.lhs_scalar), @tagName(expr.rhs_scalar) }),
        .with_column_hardtanh => |expr| try writer.print("with_column_hardtanh({s}=hardtanh({s}, min:{s}, max:{s}))", .{ expr.name, expr.input_name, @tagName(expr.lhs_scalar), @tagName(expr.rhs_scalar) }),
        .with_column_maximum_scalar => |expr| try writer.print("with_column_maximum_scalar({s}=maximum({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_minimum_scalar => |expr| try writer.print("with_column_minimum_scalar({s}=minimum({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_clip_min => |expr| try writer.print("with_column_clip_min({s}=clip_min({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_clip_max => |expr| try writer.print("with_column_clip_max({s}=clip_max({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_hardshrink => |expr| try writer.print("with_column_hardshrink({s}=hardshrink({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_softshrink => |expr| try writer.print("with_column_softshrink({s}=softshrink({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_tanhshrink => |expr| try writer.print("with_column_tanhshrink({s}=tanhshrink({s}))", .{ expr.name, expr.input_name }),
        .with_column_elu => |expr| try writer.print("with_column_elu({s}=elu({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_celu => |expr| try writer.print("with_column_celu({s}=celu({s}, scalar:{s}))", .{ expr.name, expr.input_name, @tagName(expr.scalar) }),
        .with_column_softsign => |expr| try writer.print("with_column_softsign({s}=softsign({s}))", .{ expr.name, expr.input_name }),
        .with_column_hardsigmoid => |expr| try writer.print("with_column_hardsigmoid({s}=hardsigmoid({s}))", .{ expr.name, expr.input_name }),
        .with_column_hardswish => |expr| try writer.print("with_column_hardswish({s}=hardswish({s}))", .{ expr.name, expr.input_name }),
        .with_column_silu => |expr| try writer.print("with_column_silu({s}=silu({s}))", .{ expr.name, expr.input_name }),
        .with_column_swish => |expr| try writer.print("with_column_swish({s}=swish({s}))", .{ expr.name, expr.input_name }),
        .with_column_mish => |expr| try writer.print("with_column_mish({s}=mish({s}))", .{ expr.name, expr.input_name }),
        .with_column_gelu => |expr| try writer.print("with_column_gelu({s}=gelu({s}))", .{ expr.name, expr.input_name }),
        .with_column_selu => |expr| try writer.print("with_column_selu({s}=selu({s}))", .{ expr.name, expr.input_name }),
        .with_column_exp => |expr| try writer.print("with_column_exp({s}=exp({s}))", .{ expr.name, expr.input_name }),
        .with_column_exp2 => |expr| try writer.print("with_column_exp2({s}=exp2({s}))", .{ expr.name, expr.input_name }),
        .with_column_expm1 => |expr| try writer.print("with_column_expm1({s}=expm1({s}))", .{ expr.name, expr.input_name }),
        .with_column_sin => |expr| try writer.print("with_column_sin({s}=sin({s}))", .{ expr.name, expr.input_name }),
        .with_column_cos => |expr| try writer.print("with_column_cos({s}=cos({s}))", .{ expr.name, expr.input_name }),
        .with_column_tan => |expr| try writer.print("with_column_tan({s}=tan({s}))", .{ expr.name, expr.input_name }),
        .with_column_asin => |expr| try writer.print("with_column_asin({s}=asin({s}))", .{ expr.name, expr.input_name }),
        .with_column_acos => |expr| try writer.print("with_column_acos({s}=acos({s}))", .{ expr.name, expr.input_name }),
        .with_column_atan => |expr| try writer.print("with_column_atan({s}=atan({s}))", .{ expr.name, expr.input_name }),
        .with_column_sinh => |expr| try writer.print("with_column_sinh({s}=sinh({s}))", .{ expr.name, expr.input_name }),
        .with_column_cosh => |expr| try writer.print("with_column_cosh({s}=cosh({s}))", .{ expr.name, expr.input_name }),
        .with_column_tanh => |expr| try writer.print("with_column_tanh({s}=tanh({s}))", .{ expr.name, expr.input_name }),
        .with_column_asinh => |expr| try writer.print("with_column_asinh({s}=asinh({s}))", .{ expr.name, expr.input_name }),
        .with_column_acosh => |expr| try writer.print("with_column_acosh({s}=acosh({s}))", .{ expr.name, expr.input_name }),
        .with_column_atanh => |expr| try writer.print("with_column_atanh({s}=atanh({s}))", .{ expr.name, expr.input_name }),
        .with_column_log => |expr| try writer.print("with_column_log({s}=log({s}))", .{ expr.name, expr.input_name }),
        .with_column_log1p => |expr| try writer.print("with_column_log1p({s}=log1p({s}))", .{ expr.name, expr.input_name }),
        .with_column_lgamma => |expr| try writer.print("with_column_lgamma({s}=lgamma({s}))", .{ expr.name, expr.input_name }),
        .with_column_sinc => |expr| try writer.print("with_column_sinc({s}=sinc({s}))", .{ expr.name, expr.input_name }),
        .with_column_log2 => |expr| try writer.print("with_column_log2({s}=log2({s}))", .{ expr.name, expr.input_name }),
        .with_column_log10 => |expr| try writer.print("with_column_log10({s}=log10({s}))", .{ expr.name, expr.input_name }),
        .is_null_column => |predicate| try writer.print("is_null_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_valid_column => |predicate| try writer.print("is_valid_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_nan_column => |predicate| try writer.print("is_nan_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_zero_column => |predicate| try writer.print("is_zero_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_positive_zero_column => |predicate| try writer.print("is_positive_zero_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_negative_zero_column => |predicate| try writer.print("is_negative_zero_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_non_zero_column => |predicate| try writer.print("is_non_zero_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_positive_column => |predicate| try writer.print("is_positive_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_signbit_column => |predicate| try writer.print("is_signbit_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_negative_column => |predicate| try writer.print("is_negative_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_finite_column => |predicate| try writer.print("is_finite_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_normal_column => |predicate| try writer.print("is_normal_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_subnormal_column => |predicate| try writer.print("is_subnormal_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_non_finite_column => |predicate| try writer.print("is_non_finite_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_inf_column => |predicate| try writer.print("is_inf_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_positive_inf_column => |predicate| try writer.print("is_positive_inf_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .is_negative_inf_column => |predicate| try writer.print("is_negative_inf_column({s}->{s})", .{ predicate.name, predicate.output_name }),
        .row_null_count => |row_count| {
            try writer.print("row_null_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_valid_count => |row_count| {
            try writer.print("row_valid_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_nan_count => |row_count| {
            try writer.print("row_nan_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_inf_count => |row_count| {
            try writer.print("row_inf_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_positive_inf_count => |row_count| {
            try writer.print("row_positive_inf_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_negative_inf_count => |row_count| {
            try writer.print("row_negative_inf_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_zero_count => |row_count| {
            try writer.print("row_zero_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_positive_zero_count => |row_count| {
            try writer.print("row_positive_zero_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_negative_zero_count => |row_count| {
            try writer.print("row_negative_zero_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_non_zero_count => |row_count| {
            try writer.print("row_non_zero_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_positive_count => |row_count| {
            try writer.print("row_positive_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_signbit_count => |row_count| {
            try writer.print("row_signbit_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_negative_count => |row_count| {
            try writer.print("row_negative_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_finite_count => |row_count| {
            try writer.print("row_finite_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_normal_count => |row_count| {
            try writer.print("row_normal_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_subnormal_count => |row_count| {
            try writer.print("row_subnormal_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .row_non_finite_count => |row_count| {
            try writer.print("row_non_finite_count([", .{});
            for (row_count.names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("]->{s})", .{row_count.output_name});
        },
        .with_column_compare => |expr| try writer.print("with_column_compare({s}={s} {s} {s})", .{ expr.name, expr.lhs_name, @tagName(expr.op), expr.rhs_name }),
        .with_column_compare_scalar => |expr| try writer.print("with_column_compare_scalar({s}={s} {s} scalar:{s})", .{ expr.name, expr.input_name, @tagName(expr.op), @tagName(expr.scalar) }),
        .group_by_count => |group| try writer.print("group_by_count({s} -> {s})", .{ group.key_name, group.output_name }),
        .group_by_value => |group| try writer.print("group_by_{s}({s}, value={s} -> {s})", .{ @tagName(group.aggregation), group.key_name, group.value_name, group.output_name }),
        .group_by_stats => |group| try writer.print("group_by_stats({s}, value={s}, prefix={s})", .{ group.key_name, group.value_name, group.output_prefix }),
        .group_by_stats_on => |group| {
            try writer.print("group_by_stats_on([", .{});
            for (group.key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("], value={s}, prefix={s})", .{ group.value_name, group.output_prefix });
        },
        .group_by_profile => |group| try writer.print("group_by_profile({s}, value={s}, prefix={s})", .{ group.key_name, group.value_name, group.output_prefix }),
        .group_by_profile_on => |group| {
            try writer.print("group_by_profile_on([", .{});
            for (group.key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("], value={s}, prefix={s})", .{ group.value_name, group.output_prefix });
        },
        .join_on => |join| {
            try writer.print("{s}_join_on(left=[", .{@tagName(join.kind)});
            for (join.left_key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("], right=[", .{});
            for (join.right_key_names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("])", .{});
        },
        .asof_join => |join| try writer.print("asof_join({s}->{s}, strategy={s})", .{ join.left_key_name, join.right_key_name, @tagName(join.options.strategy) }),
        .concat_rows => |right| try writer.print("concat_rows(rows={d}, cols={d})", .{ right.height(), right.width() }),
        .distinct_rows => try writer.print("distinct_rows", .{}),
        .distinct_on => |names| {
            try writer.print("distinct_on([", .{});
            for (names, 0..) |name, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{s}", .{name});
            }
            try writer.print("])", .{});
        },
        .sort_by => |sort| try writer.print("sort_by({s}, desc={})", .{ sort.name, sort.options.descending }),
        .top_k => |top| try writer.print("top_k({s}, k={d}, desc={})", .{ top.name, top.k, top.options.descending }),
        .rank_profile_by => |rank| try writer.print("rank_profile_by({s}, prefix={s}, desc={})", .{ rank.name, rank.output_prefix, rank.options.descending }),
        .rolling_profile => |rolling| try writer.print("rolling_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_moment_profile => |rolling| try writer.print("rolling_moment_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_range_profile => |rolling| try writer.print("rolling_range_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_normalize_profile => |rolling| try writer.print("rolling_normalize_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .expanding_normalize_profile => |expanding| try writer.print("expanding_normalize_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .rolling_quantile_profile => |rolling| try writer.print("rolling_quantile_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .expanding_quantile_profile => |expanding| try writer.print("expanding_quantile_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .rolling_bool_profile => |rolling| try writer.print("rolling_bool_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_drawdown_profile => |rolling| try writer.print("rolling_drawdown_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_robust_profile => |rolling| try writer.print("rolling_robust_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .rolling_rank_profile => |rolling| try writer.print("rolling_rank_profile({s}, prefix={s}, window={d})", .{ rolling.name, rolling.output_prefix, rolling.options.window }),
        .lag_profile => |lag| try writer.print("lag_profile({s}, prefix={s}, periods={d})", .{ lag.name, lag.output_prefix, lag.options.periods }),
        .lead_profile => |lead| try writer.print("lead_profile({s}, prefix={s}, periods={d})", .{ lead.name, lead.output_prefix, lead.options.periods }),
        .clip_profile => |clip| try writer.print("clip_profile({s}, prefix={s}, [{d},{d}])", .{ clip.name, clip.output_prefix, clip.options.lower, clip.options.upper }),
        .rolling_clip_profile => |clip| try writer.print("rolling_clip_profile({s}, prefix={s}, [{d},{d}], window={d})", .{ clip.name, clip.output_prefix, clip.clip_options.lower, clip.clip_options.upper, clip.options.window }),
        .expanding_clip_profile => |clip| try writer.print("expanding_clip_profile({s}, prefix={s}, [{d},{d}], min_periods={d})", .{ clip.name, clip.output_prefix, clip.clip_options.lower, clip.clip_options.upper, clip.options.min_periods }),
        .threshold_profile => |threshold| try writer.print("threshold_profile({s}, prefix={s}, threshold={d})", .{ threshold.name, threshold.output_prefix, threshold.options.threshold }),
        .rolling_threshold_profile => |threshold| try writer.print("rolling_threshold_profile({s}, prefix={s}, threshold={d}, window={d})", .{ threshold.name, threshold.output_prefix, threshold.threshold, threshold.options.window }),
        .expanding_threshold_profile => |threshold| try writer.print("expanding_threshold_profile({s}, prefix={s}, threshold={d}, min_periods={d})", .{ threshold.name, threshold.output_prefix, threshold.threshold, threshold.options.min_periods }),
        .expanding_profile => |expanding| try writer.print("expanding_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .expanding_bool_profile => |expanding| try writer.print("expanding_bool_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .expanding_rank_profile => |expanding| try writer.print("expanding_rank_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .expanding_robust_profile => |expanding| try writer.print("expanding_robust_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .expanding_moment_profile => |expanding| try writer.print("expanding_moment_profile({s}, prefix={s}, min_periods={d})", .{ expanding.name, expanding.output_prefix, expanding.options.min_periods }),
        .standardize_profile => |standardize| try writer.print("standardize_profile({s}, prefix={s}, min_periods={d})", .{ standardize.name, standardize.output_prefix, standardize.options.min_periods }),
        .robust_profile => |robust| try writer.print("robust_profile({s}, prefix={s}, min_periods={d})", .{ robust.name, robust.output_prefix, robust.options.min_periods }),
        .drawdown_profile => |drawdown| try writer.print("drawdown_profile({s}, prefix={s}, min_periods={d})", .{ drawdown.name, drawdown.output_prefix, drawdown.options.min_periods }),
        .extrema_profile => |extrema| try writer.print("extrema_profile({s}, prefix={s}, min_periods={d})", .{ extrema.name, extrema.output_prefix, extrema.options.min_periods }),
        .trend_profile => |trend| try writer.print("trend_profile({s}, prefix={s}, periods={d})", .{ trend.name, trend.output_prefix, trend.options.periods }),
        .rolling_trend_profile => |trend| try writer.print("rolling_trend_profile({s}, prefix={s}, periods={d}, window={d})", .{ trend.name, trend.output_prefix, trend.trend_options.periods, trend.options.window }),
        .expanding_trend_profile => |trend| try writer.print("expanding_trend_profile({s}, prefix={s}, periods={d}, min_periods={d})", .{ trend.name, trend.output_prefix, trend.trend_options.periods, trend.options.min_periods }),
        .change_point_profile => |change| try writer.print("change_point_profile({s}, prefix={s}, threshold={d}, periods={d})", .{ change.name, change.output_prefix, change.threshold, change.options.periods }),
        .rolling_change_point_profile => |change| try writer.print("rolling_change_point_profile({s}, prefix={s}, threshold={d}, periods={d}, window={d})", .{ change.name, change.output_prefix, change.threshold, change.change_options.periods, change.options.window }),
        .expanding_change_point_profile => |change| try writer.print("expanding_change_point_profile({s}, prefix={s}, threshold={d}, periods={d}, min_periods={d})", .{ change.name, change.output_prefix, change.threshold, change.change_options.periods, change.options.min_periods }),
        .sign_profile => |sign| try writer.print("sign_profile({s}, prefix={s}, periods={d})", .{ sign.name, sign.output_prefix, sign.options.periods }),
        .rolling_sign_profile => |sign| try writer.print("rolling_sign_profile({s}, prefix={s}, periods={d}, window={d})", .{ sign.name, sign.output_prefix, sign.sign_options.periods, sign.options.window }),
        .expanding_sign_profile => |sign| try writer.print("expanding_sign_profile({s}, prefix={s}, periods={d}, min_periods={d})", .{ sign.name, sign.output_prefix, sign.sign_options.periods, sign.options.min_periods }),
        .crossover_profile => |cross| try writer.print("crossover_profile({s},{s}, prefix={s}, periods={d})", .{ cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.options.periods }),
        .rolling_crossover_profile => |cross| try writer.print("rolling_crossover_profile({s},{s}, prefix={s}, periods={d}, window={d})", .{ cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.cross_options.periods, cross.options.window }),
        .expanding_crossover_profile => |cross| try writer.print("expanding_crossover_profile({s},{s}, prefix={s}, periods={d}, min_periods={d})", .{ cross.lhs_name, cross.rhs_name, cross.output_prefix, cross.cross_options.periods, cross.options.min_periods }),
        .bucket_profile => |bucket| try writer.print("bucket_profile({s}, prefix={s}, buckets={d})", .{ bucket.name, bucket.output_prefix, bucket.options.buckets }),
        .ema_profile => |ema| try writer.print("ema_profile({s}, prefix={s}, alpha={d})", .{ ema.name, ema.output_prefix, ema.options.alpha }),
        .linear_fit_profile => |fit| try writer.print("linear_fit_profile({s}->{s}, prefix={s})", .{ fit.x_name, fit.y_name, fit.output_prefix }),
        .error_profile => |err| try writer.print("error_profile(actual={s}, predicted={s}, prefix={s})", .{ err.actual_name, err.predicted_name, err.output_prefix }),
        .rolling_error_profile => |err| try writer.print("rolling_error_profile(actual={s}, predicted={s}, prefix={s}, window={d})", .{ err.actual_name, err.predicted_name, err.output_prefix, err.options.window }),
        .expanding_error_profile => |err| try writer.print("expanding_error_profile(actual={s}, predicted={s}, prefix={s}, min_periods={d})", .{ err.actual_name, err.predicted_name, err.output_prefix, err.options.min_periods }),
        .classification_profile => |class| try writer.print("classification_profile(actual={s}, predicted={s}, prefix={s})", .{ class.actual_name, class.predicted_name, class.output_prefix }),
        .rolling_classification_profile => |class| try writer.print("rolling_classification_profile(actual={s}, predicted={s}, prefix={s}, window={d})", .{ class.actual_name, class.predicted_name, class.output_prefix, class.options.window }),
        .expanding_classification_profile => |class| try writer.print("expanding_classification_profile(actual={s}, predicted={s}, prefix={s}, min_periods={d})", .{ class.actual_name, class.predicted_name, class.output_prefix, class.options.min_periods }),
        .bool_transition_profile => |transition| try writer.print("bool_transition_profile({s}, prefix={s}, periods={d})", .{ transition.name, transition.output_prefix, transition.options.periods }),
        .rolling_bool_transition_profile => |transition| try writer.print("rolling_bool_transition_profile({s}, prefix={s}, periods={d}, window={d})", .{ transition.name, transition.output_prefix, transition.transition_options.periods, transition.options.window }),
        .expanding_bool_transition_profile => |transition| try writer.print("expanding_bool_transition_profile({s}, prefix={s}, periods={d}, min_periods={d})", .{ transition.name, transition.output_prefix, transition.transition_options.periods, transition.options.min_periods }),
        .rolling_correlation_profile => |corr| try writer.print("rolling_correlation_profile({s},{s}, prefix={s}, window={d})", .{ corr.x_name, corr.y_name, corr.output_prefix, corr.options.window }),
        .expanding_correlation_profile => |corr| try writer.print("expanding_correlation_profile({s},{s}, prefix={s}, min_periods={d})", .{ corr.x_name, corr.y_name, corr.output_prefix, corr.options.min_periods }),
        .expanding_linear_fit_profile => |fit| try writer.print("expanding_linear_fit_profile({s}->{s}, prefix={s}, min_periods={d})", .{ fit.x_name, fit.y_name, fit.output_prefix, fit.options.min_periods }),
        .rolling_linear_fit_profile => |fit| try writer.print("rolling_linear_fit_profile({s}->{s}, prefix={s}, window={d})", .{ fit.x_name, fit.y_name, fit.output_prefix, fit.options.window }),
        .validity_profile => |validity| try writer.print("validity_profile({s}, prefix={s})", .{ validity.name, validity.output_prefix }),
        .rolling_validity_profile => |validity| try writer.print("rolling_validity_profile({s}, prefix={s}, window={d})", .{ validity.name, validity.output_prefix, validity.options.window }),
        .expanding_validity_profile => |validity| try writer.print("expanding_validity_profile({s}, prefix={s}, min_periods={d})", .{ validity.name, validity.output_prefix, validity.options.min_periods }),
        .slice_rows => |slice| try writer.print("slice_rows({d}..{d})", .{ slice.start, slice.stop }),
        .drop_rows => |row_indices| {
            try writer.print("drop_rows([", .{});
            for (row_indices, 0..) |row_index, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{d}", .{row_index});
            }
            try writer.print("])", .{});
        },
        .drop_row_range => |range| try writer.print("drop_row_range({d}..{d})", .{ range.start, range.stop }),
        .drop_last_rows => |n| try writer.print("drop_last_rows({d})", .{n}),
        .slice_rows_step => |slice| try writer.print("slice_rows_step({d}..{d}, step={d})", .{ slice.start, slice.stop, slice.step }),
        .stride_rows => |stride| try writer.print("stride_rows(start={d}, step={d})", .{ stride.start, stride.step }),
        .take_rows => |row_indices| {
            try writer.print("take_rows([", .{});
            for (row_indices, 0..) |row_index, i| {
                if (i != 0) try writer.print(",", .{});
                try writer.print("{d}", .{row_index});
            }
            try writer.print("])", .{});
        },
        .sample_rows => |sample| try writer.print("sample_rows(count={d}, seed={d})", .{ sample.count, sample.seed }),
        .sample_rows_with_replacement => |sample| try writer.print("sample_rows_with_replacement(count={d}, seed={d})", .{ sample.count, sample.seed }),
        .reverse_rows => try writer.print("reverse_rows", .{}),
        .head => |n| try writer.print("head({d})", .{n}),
        .tail => |n| try writer.print("tail({d})", .{n}),
    }
}
