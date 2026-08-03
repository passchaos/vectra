//! Sorting helpers for tagged device columns.

const std = @import("std");
const array_mod = @import("../../array.zig");
const numeric_mod = @import("../../dataframe_numeric.zig");
const options_mod = @import("../../dataframe_options.zig");
const validity_core_mod = @import("../../dataframe_validity_core.zig");
const series_mod = @import("../../series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceSortOptions = options_mod.DeviceSortOptions;
const compareSortValues = numeric_mod.compareSortValues;
const validityValues = validity_core_mod.validityValues;

fn columnValue(self: anytype) switch (@typeInfo(@TypeOf(self))) {
    .pointer => |ptr| ptr.child,
    else => @TypeOf(self),
} {
    return switch (@typeInfo(@TypeOf(self))) {
        .pointer => self.*,
        else => self,
    };
}

pub fn argsort(self: anytype, allocator: std.mem.Allocator, options_value: DeviceSortOptions) DeviceDataError![]usize {
    return switch (columnValue(self)) {
        .bool => |typed| try argsortTypedColumn(bool, typed, allocator, options_value),
        .i8 => |typed| try argsortTypedColumn(i8, typed, allocator, options_value),
        .i16 => |typed| try argsortTypedColumn(i16, typed, allocator, options_value),
        .i32 => |typed| try argsortTypedColumn(i32, typed, allocator, options_value),
        .i64 => |typed| try argsortTypedColumn(i64, typed, allocator, options_value),
        .u8 => |typed| try argsortTypedColumn(u8, typed, allocator, options_value),
        .u16 => |typed| try argsortTypedColumn(u16, typed, allocator, options_value),
        .u32 => |typed| try argsortTypedColumn(u32, typed, allocator, options_value),
        .u64 => |typed| try argsortTypedColumn(u64, typed, allocator, options_value),
        .usize => |typed| try argsortTypedColumn(usize, typed, allocator, options_value),
        .isize => |typed| try argsortTypedColumn(isize, typed, allocator, options_value),
        .f16 => |typed| try argsortTypedColumn(f16, typed, allocator, options_value),
        .f32 => |typed| try argsortTypedColumn(f32, typed, allocator, options_value),
        .f64 => |typed| try argsortTypedColumn(f64, typed, allocator, options_value),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn argsortTypedColumn(comptime T: type, column: anytype, allocator: std.mem.Allocator, options_value: DeviceSortOptions) array_mod.ArrayError![]usize {
    const values = try column.values.toOwnedSlice(allocator);
    defer allocator.free(values);
    const maybe_validity = try validityValues(column, allocator);
    defer if (maybe_validity) |validity| allocator.free(validity);

    const order = try allocator.alloc(usize, values.len);
    for (order, 0..) |*slot, i| slot.* = i;

    const Ctx = struct {
        values: []const T,
        validity: ?[]const bool,
        options: DeviceSortOptions,

        fn isValid(ctx: @This(), index: usize) bool {
            return if (ctx.validity) |validity| validity[index] else true;
        }

        fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            const a_valid = ctx.isValid(a);
            const b_valid = ctx.isValid(b);
            if (a_valid != b_valid) {
                return switch (ctx.options.nulls) {
                    .first => !a_valid,
                    .last => a_valid,
                };
            }
            if (!a_valid and !b_valid) return a < b;

            const cmp = compareSortValues(T, ctx.values[a], ctx.values[b]);
            if (cmp == 0) return a < b;
            return if (ctx.options.descending) cmp > 0 else cmp < 0;
        }
    };

    std.sort.insertion(usize, order, Ctx{
        .values = values,
        .validity = maybe_validity,
        .options = options_value,
    }, Ctx.lessThan);
    return order;
}
