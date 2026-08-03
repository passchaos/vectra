//! Dependency bookkeeping helpers for lazy Parquet scan pushdown.
//!
//! The planner in `../pushdown.zig` is intentionally a high-level switch over
//! lazy operations.  This module owns the repetitive rules for adding source
//! column requirements, marking derived outputs, and clearing literal-derived
//! pushdown metadata when a later operation overwrites a literal column.

const std = @import("std");
const names_mod = @import("../../../dataframe_names.zig");
const options_mod = @import("../../../dataframe_options.zig");

const appendOwnedNameUnique = names_mod.appendOwnedNameUnique;
const appendBorrowedNameUnique = names_mod.appendBorrowedNameUnique;
const nameInBorrowedList = names_mod.nameInBorrowedList;

pub fn addRowWeightedPairColumnOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    projection_blocked: *bool,
    row_weighted: anytype,
) std.mem.Allocator.Error!void {
    if (row_weighted.lhs_names.len == 0 or row_weighted.lhs_names.len != row_weighted.rhs_names.len or row_weighted.lhs_names.len != row_weighted.weight_names.len or row_weighted.output_names.len != row_weighted.lhs_names.len) {
        for (row_weighted.output_names) |output_name| {
            try markDerivedName(allocator, derived_names, literal_scalars, output_name);
        }
        projection_blocked.* = true;
        return;
    }
    for (row_weighted.lhs_names) |name| {
        if (!nameInBorrowedList(name, derived_names.items)) {
            try appendOwnedNameUnique(allocator, required_names, name);
        }
    }
    for (row_weighted.rhs_names) |name| {
        if (!nameInBorrowedList(name, derived_names.items)) {
            try appendOwnedNameUnique(allocator, required_names, name);
        }
    }
    for (row_weighted.weight_names) |name| {
        if (!nameInBorrowedList(name, derived_names.items)) {
            try appendOwnedNameUnique(allocator, required_names, name);
        }
    }
    for (row_weighted.output_names) |output_name| {
        try markDerivedName(allocator, derived_names, literal_scalars, output_name);
    }
}

pub fn markDerivedName(
    allocator: std.mem.Allocator,
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    name: []const u8,
) std.mem.Allocator.Error!void {
    try appendBorrowedNameUnique(allocator, derived_names, name);
    // A later derived expression may intentionally replace a prior literal
    // column with the same name.  Keep literal-based scan pushdown tied to the
    // currently visible lazy value rather than a stale earlier literal.
    _ = literal_scalars.remove(name);
}

pub fn addSourceNameRequirement(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: []const []const u8,
    name: []const u8,
) std.mem.Allocator.Error!void {
    if (!nameInBorrowedList(name, derived_names)) {
        try appendOwnedNameUnique(allocator, required_names, name);
    }
}

pub fn addSourceNameRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: []const []const u8,
    names: []const []const u8,
) std.mem.Allocator.Error!void {
    for (names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names, name);
    }
}

pub fn addUnaryColumnOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    output_name: []const u8,
    input_name: []const u8,
) std.mem.Allocator.Error!void {
    // Dependencies are evaluated before the output is marked as derived so an
    // in-place expression such as withColumnAbs("x", "x") still projects the
    // source column needed to compute the replacement.
    try addSourceNameRequirement(allocator, required_names, derived_names.items, input_name);
    try markDerivedName(allocator, derived_names, literal_scalars, output_name);
}

pub fn addBinaryColumnOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    output_name: []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
) std.mem.Allocator.Error!void {
    try addSourceNameRequirement(allocator, required_names, derived_names.items, lhs_name);
    try addSourceNameRequirement(allocator, required_names, derived_names.items, rhs_name);
    try markDerivedName(allocator, derived_names, literal_scalars, output_name);
}

pub fn addTernaryColumnOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    output_name: []const u8,
    first_name: []const u8,
    second_name: []const u8,
    third_name: []const u8,
) std.mem.Allocator.Error!void {
    try addSourceNameRequirement(allocator, required_names, derived_names.items, first_name);
    try addSourceNameRequirement(allocator, required_names, derived_names.items, second_name);
    try addSourceNameRequirement(allocator, required_names, derived_names.items, third_name);
    try markDerivedName(allocator, derived_names, literal_scalars, output_name);
}

pub fn addListColumnOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    output_name: []const u8,
    input_names: []const []const u8,
) std.mem.Allocator.Error!void {
    for (input_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    try markDerivedName(allocator, derived_names, literal_scalars, output_name);
}

pub fn markDerivedNames(
    allocator: std.mem.Allocator,
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    output_names: []const []const u8,
) std.mem.Allocator.Error!void {
    for (output_names) |output_name| {
        try markDerivedName(allocator, derived_names, literal_scalars, output_name);
    }
}

pub fn addRowSingleOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    projection_blocked: *bool,
    input_names: []const []const u8,
    output_name: []const u8,
) std.mem.Allocator.Error!bool {
    if (input_names.len == 0) {
        // Empty row-wise input means "all columns visible at this point". The
        // lightweight planner has no complete source schema or alias map, so
        // materialize instead of projecting an incomplete dependency set.
        try markDerivedName(allocator, derived_names, literal_scalars, output_name);
        projection_blocked.* = true;
        return false;
    }
    try addListColumnOutputRequirements(allocator, required_names, derived_names, literal_scalars, output_name, input_names);
    return true;
}

pub fn addRowMultiOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    projection_blocked: *bool,
    input_names: []const []const u8,
    output_names: []const []const u8,
) std.mem.Allocator.Error!bool {
    if (input_names.len == 0) {
        try markDerivedNames(allocator, derived_names, literal_scalars, output_names);
        projection_blocked.* = true;
        return false;
    }
    for (input_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    try markDerivedNames(allocator, derived_names, literal_scalars, output_names);
    return true;
}

pub fn addWeightedRowSingleOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    projection_blocked: *bool,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) std.mem.Allocator.Error!bool {
    if (value_names.len == 0 or weight_names.len == 0) {
        try markDerivedName(allocator, derived_names, literal_scalars, output_name);
        projection_blocked.* = true;
        return false;
    }
    for (value_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    for (weight_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    try markDerivedName(allocator, derived_names, literal_scalars, output_name);
    return true;
}

pub fn addWeightedRowMultiOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    projection_blocked: *bool,
    value_names: []const []const u8,
    weight_names: []const []const u8,
    output_names: []const []const u8,
) std.mem.Allocator.Error!bool {
    if (value_names.len == 0 or value_names.len != weight_names.len or output_names.len != value_names.len) {
        try markDerivedNames(allocator, derived_names, literal_scalars, output_names);
        projection_blocked.* = true;
        return false;
    }
    for (value_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    for (weight_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    try markDerivedNames(allocator, derived_names, literal_scalars, output_names);
    return true;
}

pub fn addWeightedPairRowSingleOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    projection_blocked: *bool,
    lhs_names: []const []const u8,
    rhs_names: []const []const u8,
    weight_names: []const []const u8,
    output_name: []const u8,
) std.mem.Allocator.Error!bool {
    if (lhs_names.len == 0 or lhs_names.len != rhs_names.len or lhs_names.len != weight_names.len) {
        try markDerivedName(allocator, derived_names, literal_scalars, output_name);
        projection_blocked.* = true;
        return false;
    }
    for (lhs_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    for (rhs_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    for (weight_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    try markDerivedName(allocator, derived_names, literal_scalars, output_name);
    return true;
}

pub fn addGroupedValueOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    projection_blocked: *bool,
    group_names: []const []const u8,
    value_name: []const u8,
    output_name: []const u8,
) std.mem.Allocator.Error!bool {
    if (group_names.len == 0) {
        try markDerivedName(allocator, derived_names, literal_scalars, output_name);
        projection_blocked.* = true;
        return false;
    }
    for (group_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    try addSourceNameRequirement(allocator, required_names, derived_names.items, value_name);
    try markDerivedName(allocator, derived_names, literal_scalars, output_name);
    return true;
}

pub fn addGroupedWeightedValueOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    projection_blocked: *bool,
    group_names: []const []const u8,
    value_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) std.mem.Allocator.Error!bool {
    if (group_names.len == 0) {
        try markDerivedName(allocator, derived_names, literal_scalars, output_name);
        projection_blocked.* = true;
        return false;
    }
    for (group_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    try addSourceNameRequirement(allocator, required_names, derived_names.items, value_name);
    try addSourceNameRequirement(allocator, required_names, derived_names.items, weight_name);
    try markDerivedName(allocator, derived_names, literal_scalars, output_name);
    return true;
}

pub fn addGroupedWeightedPairOutputRequirements(
    allocator: std.mem.Allocator,
    required_names: *std.ArrayList([]const u8),
    derived_names: *std.ArrayList([]const u8),
    literal_scalars: *std.StringHashMap(options_mod.DeviceScalar),
    projection_blocked: *bool,
    group_names: []const []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    weight_name: []const u8,
    output_name: []const u8,
) std.mem.Allocator.Error!bool {
    if (group_names.len == 0) {
        try markDerivedName(allocator, derived_names, literal_scalars, output_name);
        projection_blocked.* = true;
        return false;
    }
    for (group_names) |name| {
        try addSourceNameRequirement(allocator, required_names, derived_names.items, name);
    }
    try addSourceNameRequirement(allocator, required_names, derived_names.items, lhs_name);
    try addSourceNameRequirement(allocator, required_names, derived_names.items, rhs_name);
    try addSourceNameRequirement(allocator, required_names, derived_names.items, weight_name);
    try markDerivedName(allocator, derived_names, literal_scalars, output_name);
    return true;
}
