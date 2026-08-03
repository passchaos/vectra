//! Single-key/asof row-index builders for dataframe joins.

const std = @import("std");
const array_mod = @import("../../array.zig");
const asof_join_mod = @import("asof_join.zig");
const equi_typed_mod = @import("equi_join_typed.zig");

pub const KeyMatchError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
    IndexOutOfBounds,
};

pub const asofRightRowIndices = asof_join_mod.asofRightRowIndices;
pub const asofRightRowIndicesTyped = asof_join_mod.asofRightRowIndicesTyped;

pub const innerJoinRowIndicesTyped = equi_typed_mod.innerJoinRowIndicesTyped;
pub const leftJoinRowIndicesTyped = equi_typed_mod.leftJoinRowIndicesTyped;
pub const fullJoinRowIndicesTyped = equi_typed_mod.fullJoinRowIndicesTyped;
pub const semiAntiJoinRowIndicesTyped = equi_typed_mod.semiAntiJoinRowIndicesTyped;

pub fn innerJoinRowIndices(comptime JoinRowIndexPair: type, allocator: std.mem.Allocator, left: anytype, right: anytype) KeyMatchError!JoinRowIndexPair {
    return switch (left) {
        .bool => |typed| innerJoinRowIndicesTyped(bool, JoinRowIndexPair, allocator, typed, right.bool),
        .i8 => |typed| innerJoinRowIndicesTyped(i8, JoinRowIndexPair, allocator, typed, right.i8),
        .i16 => |typed| innerJoinRowIndicesTyped(i16, JoinRowIndexPair, allocator, typed, right.i16),
        .i32 => |typed| innerJoinRowIndicesTyped(i32, JoinRowIndexPair, allocator, typed, right.i32),
        .i64 => |typed| innerJoinRowIndicesTyped(i64, JoinRowIndexPair, allocator, typed, right.i64),
        .u8 => |typed| innerJoinRowIndicesTyped(u8, JoinRowIndexPair, allocator, typed, right.u8),
        .u16 => |typed| innerJoinRowIndicesTyped(u16, JoinRowIndexPair, allocator, typed, right.u16),
        .u32 => |typed| innerJoinRowIndicesTyped(u32, JoinRowIndexPair, allocator, typed, right.u32),
        .u64 => |typed| innerJoinRowIndicesTyped(u64, JoinRowIndexPair, allocator, typed, right.u64),
        .usize => |typed| innerJoinRowIndicesTyped(usize, JoinRowIndexPair, allocator, typed, right.usize),
        .isize => |typed| innerJoinRowIndicesTyped(isize, JoinRowIndexPair, allocator, typed, right.isize),
        .f16 => |typed| innerJoinRowIndicesTyped(f16, JoinRowIndexPair, allocator, typed, right.f16),
        .f32 => |typed| innerJoinRowIndicesTyped(f32, JoinRowIndexPair, allocator, typed, right.f32),
        .f64 => |typed| innerJoinRowIndicesTyped(f64, JoinRowIndexPair, allocator, typed, right.f64),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn leftJoinRowIndices(comptime JoinRowIndexPair: type, allocator: std.mem.Allocator, left: anytype, right: anytype) KeyMatchError!JoinRowIndexPair {
    return switch (left) {
        .bool => |typed| leftJoinRowIndicesTyped(bool, JoinRowIndexPair, allocator, typed, right.bool),
        .i8 => |typed| leftJoinRowIndicesTyped(i8, JoinRowIndexPair, allocator, typed, right.i8),
        .i16 => |typed| leftJoinRowIndicesTyped(i16, JoinRowIndexPair, allocator, typed, right.i16),
        .i32 => |typed| leftJoinRowIndicesTyped(i32, JoinRowIndexPair, allocator, typed, right.i32),
        .i64 => |typed| leftJoinRowIndicesTyped(i64, JoinRowIndexPair, allocator, typed, right.i64),
        .u8 => |typed| leftJoinRowIndicesTyped(u8, JoinRowIndexPair, allocator, typed, right.u8),
        .u16 => |typed| leftJoinRowIndicesTyped(u16, JoinRowIndexPair, allocator, typed, right.u16),
        .u32 => |typed| leftJoinRowIndicesTyped(u32, JoinRowIndexPair, allocator, typed, right.u32),
        .u64 => |typed| leftJoinRowIndicesTyped(u64, JoinRowIndexPair, allocator, typed, right.u64),
        .usize => |typed| leftJoinRowIndicesTyped(usize, JoinRowIndexPair, allocator, typed, right.usize),
        .isize => |typed| leftJoinRowIndicesTyped(isize, JoinRowIndexPair, allocator, typed, right.isize),
        .f16 => |typed| leftJoinRowIndicesTyped(f16, JoinRowIndexPair, allocator, typed, right.f16),
        .f32 => |typed| leftJoinRowIndicesTyped(f32, JoinRowIndexPair, allocator, typed, right.f32),
        .f64 => |typed| leftJoinRowIndicesTyped(f64, JoinRowIndexPair, allocator, typed, right.f64),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn fullJoinRowIndices(comptime JoinRowIndexPair: type, allocator: std.mem.Allocator, left: anytype, right: anytype) KeyMatchError!JoinRowIndexPair {
    return switch (left) {
        .bool => |typed| fullJoinRowIndicesTyped(bool, JoinRowIndexPair, allocator, typed, right.bool),
        .i8 => |typed| fullJoinRowIndicesTyped(i8, JoinRowIndexPair, allocator, typed, right.i8),
        .i16 => |typed| fullJoinRowIndicesTyped(i16, JoinRowIndexPair, allocator, typed, right.i16),
        .i32 => |typed| fullJoinRowIndicesTyped(i32, JoinRowIndexPair, allocator, typed, right.i32),
        .i64 => |typed| fullJoinRowIndicesTyped(i64, JoinRowIndexPair, allocator, typed, right.i64),
        .u8 => |typed| fullJoinRowIndicesTyped(u8, JoinRowIndexPair, allocator, typed, right.u8),
        .u16 => |typed| fullJoinRowIndicesTyped(u16, JoinRowIndexPair, allocator, typed, right.u16),
        .u32 => |typed| fullJoinRowIndicesTyped(u32, JoinRowIndexPair, allocator, typed, right.u32),
        .u64 => |typed| fullJoinRowIndicesTyped(u64, JoinRowIndexPair, allocator, typed, right.u64),
        .usize => |typed| fullJoinRowIndicesTyped(usize, JoinRowIndexPair, allocator, typed, right.usize),
        .isize => |typed| fullJoinRowIndicesTyped(isize, JoinRowIndexPair, allocator, typed, right.isize),
        .f16 => |typed| fullJoinRowIndicesTyped(f16, JoinRowIndexPair, allocator, typed, right.f16),
        .f32 => |typed| fullJoinRowIndicesTyped(f32, JoinRowIndexPair, allocator, typed, right.f32),
        .f64 => |typed| fullJoinRowIndicesTyped(f64, JoinRowIndexPair, allocator, typed, right.f64),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn semiAntiJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype, keep_matches: bool) KeyMatchError![]usize {
    return switch (left) {
        .bool => |typed| semiAntiJoinRowIndicesTyped(bool, allocator, typed, right.bool, keep_matches),
        .i8 => |typed| semiAntiJoinRowIndicesTyped(i8, allocator, typed, right.i8, keep_matches),
        .i16 => |typed| semiAntiJoinRowIndicesTyped(i16, allocator, typed, right.i16, keep_matches),
        .i32 => |typed| semiAntiJoinRowIndicesTyped(i32, allocator, typed, right.i32, keep_matches),
        .i64 => |typed| semiAntiJoinRowIndicesTyped(i64, allocator, typed, right.i64, keep_matches),
        .u8 => |typed| semiAntiJoinRowIndicesTyped(u8, allocator, typed, right.u8, keep_matches),
        .u16 => |typed| semiAntiJoinRowIndicesTyped(u16, allocator, typed, right.u16, keep_matches),
        .u32 => |typed| semiAntiJoinRowIndicesTyped(u32, allocator, typed, right.u32, keep_matches),
        .u64 => |typed| semiAntiJoinRowIndicesTyped(u64, allocator, typed, right.u64, keep_matches),
        .usize => |typed| semiAntiJoinRowIndicesTyped(usize, allocator, typed, right.usize, keep_matches),
        .isize => |typed| semiAntiJoinRowIndicesTyped(isize, allocator, typed, right.isize, keep_matches),
        .f16 => |typed| semiAntiJoinRowIndicesTyped(f16, allocator, typed, right.f16, keep_matches),
        .f32 => |typed| semiAntiJoinRowIndicesTyped(f32, allocator, typed, right.f32, keep_matches),
        .f64 => |typed| semiAntiJoinRowIndicesTyped(f64, allocator, typed, right.f64, keep_matches),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}
