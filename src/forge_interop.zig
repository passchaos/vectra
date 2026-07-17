//! Data-only Forge interop boundary metadata.
//!
//! Vectra deliberately does not import or depend on Forge.  This manifest gives
//! downstream packages a stable, compile-time-readable summary of where Vectra's
//! Array layer fits in the Project-Z stack.

const std = @import("std");

pub const InteropBoundary = struct {
    schema_version: []const u8,
    producer: []const u8,
    consumer: []const u8,
    dependency_on_consumer: bool,
    vectra_surface: []const u8,
    forge_surface: []const u8,
    lowering_path: []const u8,
    notes: []const []const u8,
};

/// Vectra's data-only Forge interop boundary report.
///
/// The report is intentionally made from string/boolean literals only.  It is
/// safe for Forge, docs tooling, or release scripts to inspect without creating
/// a package dependency cycle.
pub const forge_array_interop_boundary = InteropBoundary{
    .schema_version = "vectra.forge_interop_boundary.v1",
    .producer = "vectra",
    .consumer = "forge",
    .dependency_on_consumer = false,
    .vectra_surface = "Array/NDArray values, dtype/device/layout metadata, host/device storage, and numerical array operations",
    .forge_surface = "Tensor ownership, differentiation, model/training/inference orchestration, and graph capture",
    .lowering_path = "Forge core op lowering should go through Forge IR -> Axiom dialect/runtime; Vectra remains an Array/data interop layer",
    .notes = &.{
        "No Forge imports are required or permitted for this Vectra manifest.",
        "Vectra Array metadata may be wrapped by higher-level frameworks without transferring training semantics into Vectra.",
        "Reusable compiler, tiling, and backend execution concerns belong below Forge in Axiom rather than inside Vectra.",
    },
};

pub fn forgeInteropBoundary() InteropBoundary {
    return forge_array_interop_boundary;
}

test "Forge interop boundary is data-only and dependency-free" {
    const boundary = forgeInteropBoundary();
    try std.testing.expectEqualStrings("vectra", boundary.producer);
    try std.testing.expectEqualStrings("forge", boundary.consumer);
    try std.testing.expect(!boundary.dependency_on_consumer);
    try std.testing.expect(std.mem.indexOf(u8, boundary.lowering_path, "Forge IR -> Axiom dialect/runtime") != null);
    try std.testing.expect(boundary.notes.len >= 3);
}
