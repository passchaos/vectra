//! Bounded NumPy/PyTorch-style einsum front-ends for Vectra arrays.
//!
//! This module intentionally implements a practical subset over existing Array
//! primitives instead of a separate contraction backend.  Fast paths route
//! matrix-style forms through Array.matmul so Axiom backend selection remains
//! centralized.

const std = @import("std");
const array_mod = @import("array.zig");

const ArrayError = array_mod.ArrayError;

pub fn einsumUnary(subscripts: []const u8, input: anytype) ArrayError!@TypeOf(input) {
    if (!input.device.isAvailable()) return error.InvalidDevice;
    const plan = try parseUnaryEinsum(subscripts, input.shape.len);
    if (plan.repeatedLabel()) |label| {
        var first_axis: ?usize = null;
        var second_axis: ?usize = null;
        for (plan.input[0..plan.input_len], 0..) |candidate, axis| {
            if (candidate != label) continue;
            if (first_axis == null) {
                first_axis = axis;
            } else {
                second_axis = axis;
                break;
            }
        }
        const axis1 = first_axis orelse return error.InvalidShape;
        const axis2 = second_axis orelse return error.InvalidShape;
        var current = try input.diagonalAxes(0, @intCast(axis1), @intCast(axis2));
        errdefer current.deinit();

        var diagonal_labels = [_]u8{0} ** max_einsum_rank;
        var diagonal_len: usize = 0;
        var counts = [_]u8{0} ** 256;
        for (plan.input[0..plan.input_len]) |candidate| counts[candidate] += 1;
        for (plan.input[0..plan.input_len]) |candidate| {
            if (counts[candidate] == 1) {
                diagonal_labels[diagonal_len] = candidate;
                diagonal_len += 1;
            }
        }
        diagonal_labels[diagonal_len] = label;
        diagonal_len += 1;

        return reduceAndPermuteOwned(current, diagonal_labels[0..diagonal_len], plan.out[0..plan.out_len]);
    }

    const current = try input.copy();
    return reduceAndPermuteOwned(current, plan.input[0..plan.input_len], plan.out[0..plan.out_len]);
}

pub fn einsum1(subscripts: []const u8, input: anytype) ArrayError!@TypeOf(input) {
    return einsumUnary(subscripts, input);
}

pub fn einsum3(subscripts: []const u8, a: anytype, b: @TypeOf(a), c: @TypeOf(a)) ArrayError!@TypeOf(a) {
    try requireSameDevice(a, b);
    try requireSameDevice(a, c);
    if (chainedMatmulLikeSubscripts(subscripts, a.shape.len, b.shape.len, c.shape.len)) {
        var ab = try a.matmul(b);
        defer ab.deinit();
        return ab.matmul(c);
    }
    return error.InvalidShape;
}

pub fn einsum(subscripts: []const u8, lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    // Bounded NumPy/PyTorch-style front-end syntax over existing Array
    // primitives.  This parser intentionally supports a bounded binary subset
    // rather than full NumPy syntax.  The dedicated fast paths handle common
    // batched matmul forms, including the PyTorch/NumPy spelling
    // `...ij,...jk->...ik`, by forwarding to Array.matmul so backend selection
    // still flows through Axiom instead of a special einsum backend.
    if (ellipsisBatchedMatmulLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) return lhs.matmul(rhs);
    if (ellipsisBatchedMatvecLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) {
        var rhs_expanded = try rhs.unsqueeze(-2);
        defer rhs_expanded.deinit();
        var product = try lhs.mul(rhs_expanded);
        defer product.deinit();
        return product.sum(-1, false);
    }
    if (ellipsisBatchedVecmatLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) {
        var lhs_expanded = try lhs.unsqueeze(-1);
        defer lhs_expanded.deinit();
        var product = try lhs_expanded.mul(rhs);
        defer product.deinit();
        return product.sum(-2, false);
    }
    if (ellipsisBatchedDotLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) {
        var product = try lhs.mul(rhs);
        defer product.deinit();
        return product.sum(-1, false);
    }
    if (batchedMatmulLikeSubscripts(subscripts, lhs.shape.len, rhs.shape.len)) return lhs.matmul(rhs);
    if (try parseSameLabelBinaryEinsum(subscripts, lhs.shape.len, rhs.shape.len)) |shared_plan| {
        const product = try lhs.mul(rhs);
        return reduceAndPermuteOwned(product, shared_plan.input[0..shared_plan.input_len], shared_plan.out[0..shared_plan.out_len]);
    }
    const plan = try parseBinaryEinsum(subscripts, lhs.shape.len, rhs.shape.len);
    if (plan.matmulLike()) return lhs.matmul(rhs);
    if (plan.matvecLike()) return lhs.matmul(rhs);
    if (plan.vecmatLike()) return lhs.matmul(rhs);
    if (plan.dotLike()) return lhs.dot(rhs);
    if (plan.outerLike()) return lhs.outer(rhs);

    var contracted = try lhs.contractAxes(rhs, plan.lhsAxes(), plan.rhsAxes());
    errdefer contracted.deinit();
    if (plan.outputIsDefault()) return contracted;
    const permuted = try contracted.permute(plan.permuteAxes());
    contracted.deinit();
    return permuted;
}

fn requireSameDevice(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!void {
    if (!lhs.device.sameDevice(rhs.device)) return error.InvalidDevice;
    if (!lhs.device.isAvailable()) return error.InvalidDevice;
}

fn reduceAndPermuteOwned(initial: anytype, source_labels: []const u8, out_labels: []const u8) ArrayError!@TypeOf(initial) {
    var current = initial;
    errdefer current.deinit();
    var axis = source_labels.len;
    while (axis > 0) {
        axis -= 1;
        if (findLabel(out_labels, source_labels[axis]) == null) {
            const next = try current.sum(@intCast(axis), false);
            current.deinit();
            current = next;
        }
    }

    var default_out = [_]u8{0} ** max_einsum_rank;
    var default_out_len: usize = 0;
    for (source_labels) |label| {
        if (findLabel(out_labels, label) != null) {
            default_out[default_out_len] = label;
            default_out_len += 1;
        }
    }
    if (out_labels.len == default_out_len and std.mem.eql(u8, out_labels, default_out[0..default_out_len])) return current;

    var permutation = [_]usize{0} ** max_einsum_rank;
    for (out_labels, 0..) |label, out_axis| {
        permutation[out_axis] = findLabel(default_out[0..default_out_len], label) orelse return error.InvalidShape;
    }
    const permuted = try current.permute(permutation[0..out_labels.len]);
    current.deinit();
    return permuted;
}

const UnaryEinsumPlan = struct {
    input: [max_einsum_rank]u8 = [_]u8{0} ** max_einsum_rank,
    out: [max_einsum_rank]u8 = [_]u8{0} ** max_einsum_rank,
    input_len: usize = 0,
    out_len: usize = 0,
    repeat_label: ?u8 = null,

    fn repeatedLabel(plan: UnaryEinsumPlan) ?u8 {
        return plan.repeat_label;
    }
};

fn parseUnaryEinsum(subscripts: []const u8, input_rank: usize) ArrayError!UnaryEinsumPlan {
    if (input_rank > max_einsum_rank) return error.InvalidShape;
    if (std.mem.indexOf(u8, subscripts, "...") != null) return error.InvalidShape;
    const explicit_output = std.mem.indexOf(u8, subscripts, "->");
    const arrow = explicit_output orelse subscripts.len;
    if (std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') != null) return error.InvalidShape;
    if (explicit_output != null and std.mem.indexOf(u8, subscripts[arrow + 2 ..], "->") != null) return error.InvalidShape;

    var plan: UnaryEinsumPlan = .{};
    plan.input_len = try parseUnaryEinsumInputLabels(subscripts[0..arrow], input_rank, plan.input[0..]);
    var counts = [_]u8{0} ** 256;
    for (plan.input[0..plan.input_len]) |label| counts[label] += 1;
    for (counts, 0..) |count, label_index| {
        if (count > 2) return error.InvalidShape;
        if (count == 2) {
            if (plan.repeat_label != null) return error.InvalidShape;
            plan.repeat_label = @intCast(label_index);
        }
    }

    if (explicit_output) |_| {
        plan.out_len = try parseEinsumLabels(subscripts[arrow + 2 ..], null, plan.out[0..]);
    } else {
        for (plan.input[0..plan.input_len]) |label| {
            if (counts[label] == 1) {
                plan.out[plan.out_len] = label;
                plan.out_len += 1;
            }
        }
    }

    var out_seen = [_]bool{false} ** 256;
    for (plan.out[0..plan.out_len]) |label| {
        if (out_seen[label]) return error.InvalidShape;
        if (counts[label] == 0) return error.InvalidShape;
        out_seen[label] = true;
    }
    if (plan.repeat_label != null) {
        return plan;
    }

    return plan;
}

fn parseUnaryEinsumInputLabels(segment: []const u8, expected_rank: usize, out: []u8) ArrayError!usize {
    if (segment.len != expected_rank or segment.len > out.len) return error.InvalidShape;
    for (segment, 0..) |label, index| {
        if (!std.ascii.isAlphabetic(label)) return error.InvalidShape;
        out[index] = label;
    }
    return segment.len;
}

fn chainedMatmulLikeSubscripts(subscripts: []const u8, a_rank: usize, b_rank: usize, c_rank: usize) bool {
    // Bounded three-operand coverage for common matrix-chain forms such as
    // `ij,jk,kl->il` and shared-prefix batches like
    // `abij,abjk,abkl->abil`.  It composes through Array.matmul so backend
    // policy and Axiom N-D batch lowering stay centralized.
    if (a_rank < 2 or a_rank != b_rank or b_rank != c_rank) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const first_comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const second_rel = std.mem.indexOfScalar(u8, subscripts[first_comma + 1 .. arrow], ',') orelse return false;
    const second_comma = first_comma + 1 + second_rel;
    if (std.mem.indexOfScalar(u8, subscripts[second_comma + 1 .. arrow], ',') != null) return false;
    const a_labels = subscripts[0..first_comma];
    const b_labels = subscripts[first_comma + 1 .. second_comma];
    const c_labels = subscripts[second_comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (a_labels.len != a_rank or b_labels.len != b_rank or c_labels.len != c_rank) return false;
    if (out.len != 0 and out.len != a_rank) return false;
    if (!allEinsumLabels(a_labels) or !allEinsumLabels(b_labels) or !allEinsumLabels(c_labels) or !allEinsumLabels(out)) return false;
    if (hasRepeatedLabels(a_labels) or hasRepeatedLabels(b_labels) or hasRepeatedLabels(c_labels) or hasRepeatedLabels(out)) return false;

    const batch_rank = a_rank - 2;
    if (!std.mem.eql(u8, a_labels[0..batch_rank], b_labels[0..batch_rank])) return false;
    if (!std.mem.eql(u8, a_labels[0..batch_rank], c_labels[0..batch_rank])) return false;
    const a_row = a_labels[batch_rank];
    const ab_contract = a_labels[batch_rank + 1];
    const b_col = b_labels[batch_rank + 1];
    const c_col = c_labels[batch_rank + 1];
    if (ab_contract != b_labels[batch_rank] or b_col != c_labels[batch_rank]) return false;
    if (a_row == b_col or a_row == c_col or c_col == ab_contract) return false;
    if (out.len == 0) return true;
    return std.mem.eql(u8, out[0..batch_rank], a_labels[0..batch_rank]) and
        out[batch_rank] == a_row and
        out[batch_rank + 1] == c_col;
}

fn batchedMatmulLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    // Fast-path explicit shared-prefix batched matmul spellings such as
    // `bij,bjk->bik` and `abij,abjk->abik`.  Routing these through `matmul`
    // preserves Axiom-backed N-D batch handling instead of falling back to the
    // generic contraction path, whose current binary parser intentionally does
    // not yet model shared non-contracted labels.
    if (lhs_rank < 3 or lhs_rank != rhs_rank) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (lhs.len != lhs_rank or rhs.len != rhs_rank) return false;
    if (out.len != 0 and out.len != lhs_rank) return false;
    if (!allEinsumLabels(lhs) or !allEinsumLabels(rhs) or !allEinsumLabels(out)) return false;
    if (hasRepeatedLabels(lhs) or hasRepeatedLabels(rhs) or hasRepeatedLabels(out)) return false;

    const batch_rank = lhs_rank - 2;
    if (!std.mem.eql(u8, lhs[0..batch_rank], rhs[0..batch_rank])) return false;
    if (lhs[lhs_rank - 1] != rhs[rhs_rank - 2]) return false;
    if (out.len == 0) return true;
    return std.mem.eql(u8, out[0..batch_rank], lhs[0..batch_rank]) and
        out[batch_rank] == lhs[lhs_rank - 2] and
        out[batch_rank + 1] == rhs[rhs_rank - 1];
}

fn ellipsisBatchedMatmulLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    if (lhs_rank < 2 or rhs_rank < 2) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (!std.mem.startsWith(u8, lhs, "...") or !std.mem.startsWith(u8, rhs, "...")) return false;
    const lhs_tail = lhs[3..];
    const rhs_tail = rhs[3..];
    if (lhs_tail.len != 2 or rhs_tail.len != 2) return false;
    if (!allEinsumLabels(lhs_tail) or !allEinsumLabels(rhs_tail)) return false;
    if (hasRepeatedLabels(lhs_tail) or hasRepeatedLabels(rhs_tail)) return false;
    if (lhs_tail[1] != rhs_tail[0]) return false;
    if (out.len == 0) return true;
    if (!std.mem.startsWith(u8, out, "...")) return false;
    const out_tail = out[3..];
    return out_tail.len == 2 and
        allEinsumLabels(out_tail) and
        !hasRepeatedLabels(out_tail) and
        out_tail[0] == lhs_tail[0] and
        out_tail[1] == rhs_tail[1];
}

fn ellipsisBatchedMatvecLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    if (lhs_rank < 1 or rhs_rank < 1) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (!std.mem.startsWith(u8, lhs, "...") or !std.mem.startsWith(u8, rhs, "...")) return false;
    const lhs_tail = lhs[3..];
    const rhs_tail = rhs[3..];
    if (!allEinsumLabels(lhs_tail) or !allEinsumLabels(rhs_tail)) return false;
    if (hasRepeatedLabels(lhs_tail) or hasRepeatedLabels(rhs_tail)) return false;
    if (!(lhs_tail.len == 2 and rhs_tail.len == 1 and lhs_tail[1] == rhs_tail[0])) return false;
    if (out.len == 0) return true;
    if (!std.mem.startsWith(u8, out, "...")) return false;
    const out_tail = out[3..];
    return out_tail.len == 1 and
        allEinsumLabels(out_tail) and
        out_tail[0] == lhs_tail[0];
}

fn ellipsisBatchedVecmatLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    if (lhs_rank < 1 or rhs_rank < 1) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (!std.mem.startsWith(u8, lhs, "...") or !std.mem.startsWith(u8, rhs, "...")) return false;
    const lhs_tail = lhs[3..];
    const rhs_tail = rhs[3..];
    if (!allEinsumLabels(lhs_tail) or !allEinsumLabels(rhs_tail)) return false;
    if (hasRepeatedLabels(lhs_tail) or hasRepeatedLabels(rhs_tail)) return false;
    if (!(lhs_tail.len == 1 and rhs_tail.len == 2 and lhs_tail[0] == rhs_tail[0])) return false;
    if (out.len == 0) return true;
    if (!std.mem.startsWith(u8, out, "...")) return false;
    const out_tail = out[3..];
    return out_tail.len == 1 and
        allEinsumLabels(out_tail) and
        out_tail[0] == rhs_tail[1];
}

fn ellipsisBatchedDotLikeSubscripts(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) bool {
    if (lhs_rank < 1 or rhs_rank < 1) return false;
    const arrow = std.mem.indexOf(u8, subscripts, "->") orelse subscripts.len;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return false;
    const lhs = subscripts[0..comma];
    const rhs = subscripts[comma + 1 .. arrow];
    const out = if (arrow == subscripts.len) "" else subscripts[arrow + 2 ..];
    if (!std.mem.startsWith(u8, lhs, "...") or !std.mem.startsWith(u8, rhs, "...")) return false;
    const lhs_tail = lhs[3..];
    const rhs_tail = rhs[3..];
    if (lhs_tail.len != 1 or rhs_tail.len != 1) return false;
    if (!allEinsumLabels(lhs_tail) or !allEinsumLabels(rhs_tail)) return false;
    if (lhs_tail[0] != rhs_tail[0]) return false;
    if (out.len == 0) return true;
    return std.mem.eql(u8, out, "...");
}

const max_einsum_rank = 16;

const SameLabelBinaryEinsumPlan = struct {
    input: [max_einsum_rank]u8 = [_]u8{0} ** max_einsum_rank,
    out: [max_einsum_rank]u8 = [_]u8{0} ** max_einsum_rank,
    input_len: usize = 0,
    out_len: usize = 0,
};

fn parseSameLabelBinaryEinsum(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) ArrayError!?SameLabelBinaryEinsumPlan {
    if (lhs_rank == 0 or lhs_rank > max_einsum_rank or lhs_rank != rhs_rank) return null;
    if (std.mem.indexOf(u8, subscripts, "...") != null) return null;
    const explicit_output = std.mem.indexOf(u8, subscripts, "->");
    const arrow = explicit_output orelse subscripts.len;
    if (explicit_output != null and std.mem.indexOf(u8, subscripts[arrow + 2 ..], "->") != null) return error.InvalidShape;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return null;
    if (std.mem.indexOfScalar(u8, subscripts[comma + 1 .. arrow], ',') != null) return null;
    const lhs_labels = subscripts[0..comma];
    const rhs_labels = subscripts[comma + 1 .. arrow];
    if (lhs_labels.len != lhs_rank or rhs_labels.len != rhs_rank) return null;
    if (!std.mem.eql(u8, lhs_labels, rhs_labels)) return null;
    if (!allEinsumLabels(lhs_labels) or hasRepeatedLabels(lhs_labels)) return null;

    var plan: SameLabelBinaryEinsumPlan = .{};
    @memcpy(plan.input[0..lhs_rank], lhs_labels);
    plan.input_len = lhs_rank;
    if (explicit_output) |_| {
        plan.out_len = try parseEinsumLabels(subscripts[arrow + 2 ..], null, plan.out[0..]);
    } else {
        plan.out_len = 0;
    }

    var out_seen = [_]bool{false} ** 256;
    for (plan.out[0..plan.out_len]) |label| {
        if (out_seen[label]) return error.InvalidShape;
        if (findLabel(lhs_labels, label) == null) return error.InvalidShape;
        out_seen[label] = true;
    }
    return plan;
}

const BinaryEinsumPlan = struct {
    lhs: [max_einsum_rank]u8 = [_]u8{0} ** max_einsum_rank,
    rhs: [max_einsum_rank]u8 = [_]u8{0} ** max_einsum_rank,
    out: [max_einsum_rank * 2]u8 = [_]u8{0} ** (max_einsum_rank * 2),
    default_out: [max_einsum_rank * 2]u8 = [_]u8{0} ** (max_einsum_rank * 2),
    lhs_contract_axes: [max_einsum_rank]usize = [_]usize{0} ** max_einsum_rank,
    rhs_contract_axes: [max_einsum_rank]usize = [_]usize{0} ** max_einsum_rank,
    permutation: [max_einsum_rank * 2]usize = [_]usize{0} ** (max_einsum_rank * 2),
    lhs_len: usize = 0,
    rhs_len: usize = 0,
    out_len: usize = 0,
    default_out_len: usize = 0,
    contract_len: usize = 0,

    fn lhsAxes(plan: *const BinaryEinsumPlan) []const usize {
        return plan.lhs_contract_axes[0..plan.contract_len];
    }

    fn rhsAxes(plan: *const BinaryEinsumPlan) []const usize {
        return plan.rhs_contract_axes[0..plan.contract_len];
    }

    fn permuteAxes(plan: *const BinaryEinsumPlan) []const usize {
        return plan.permutation[0..plan.out_len];
    }

    fn outputIsDefault(plan: BinaryEinsumPlan) bool {
        return plan.out_len == plan.default_out_len and std.mem.eql(u8, plan.out[0..plan.out_len], plan.default_out[0..plan.default_out_len]);
    }

    fn matmulLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 2 and plan.rhs_len == 2 and plan.contract_len == 1 and
            plan.lhs_contract_axes[0] == 1 and plan.rhs_contract_axes[0] == 0 and
            plan.outputIsDefault();
    }

    fn matvecLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 2 and plan.rhs_len == 1 and plan.contract_len == 1 and
            plan.lhs_contract_axes[0] == 1 and plan.rhs_contract_axes[0] == 0 and
            plan.outputIsDefault();
    }

    fn vecmatLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 1 and plan.rhs_len == 2 and plan.contract_len == 1 and
            plan.lhs_contract_axes[0] == 0 and plan.rhs_contract_axes[0] == 0 and
            plan.outputIsDefault();
    }

    fn dotLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 1 and plan.rhs_len == 1 and plan.contract_len == 1 and plan.out_len == 0;
    }

    fn outerLike(plan: BinaryEinsumPlan) bool {
        return plan.lhs_len == 1 and plan.rhs_len == 1 and plan.contract_len == 0 and plan.outputIsDefault();
    }
};

fn parseBinaryEinsum(subscripts: []const u8, lhs_rank: usize, rhs_rank: usize) ArrayError!BinaryEinsumPlan {
    if (lhs_rank > max_einsum_rank or rhs_rank > max_einsum_rank) return error.InvalidShape;
    if (std.mem.indexOf(u8, subscripts, "...") != null) return error.InvalidShape;
    const explicit_output = std.mem.indexOf(u8, subscripts, "->");
    const arrow = explicit_output orelse subscripts.len;
    if (explicit_output != null and std.mem.indexOf(u8, subscripts[arrow + 2 ..], "->") != null) return error.InvalidShape;
    const comma = std.mem.indexOfScalar(u8, subscripts[0..arrow], ',') orelse return error.InvalidShape;

    var plan: BinaryEinsumPlan = .{};
    plan.lhs_len = try parseEinsumLabels(subscripts[0..comma], lhs_rank, plan.lhs[0..]);
    plan.rhs_len = try parseEinsumLabels(subscripts[comma + 1 .. arrow], rhs_rank, plan.rhs[0..]);
    if (explicit_output) |_| {
        plan.out_len = try parseEinsumLabels(subscripts[arrow + 2 ..], null, plan.out[0..]);
    }

    var out_seen = [_]bool{false} ** 256;
    for (plan.out[0..plan.out_len]) |label| {
        if (out_seen[label]) return error.InvalidShape;
        out_seen[label] = true;
    }

    var default_seen = [_]bool{false} ** 256;
    for (plan.lhs[0..plan.lhs_len], 0..) |label, lhs_axis| {
        if (findLabel(plan.rhs[0..plan.rhs_len], label)) |rhs_axis| {
            if (out_seen[label]) return error.InvalidShape; // shared batch labels are a future extension.
            plan.lhs_contract_axes[plan.contract_len] = lhs_axis;
            plan.rhs_contract_axes[plan.contract_len] = rhs_axis;
            plan.contract_len += 1;
        } else {
            plan.default_out[plan.default_out_len] = label;
            default_seen[label] = true;
            plan.default_out_len += 1;
        }
    }
    for (plan.rhs[0..plan.rhs_len]) |label| {
        if (findLabel(plan.lhs[0..plan.lhs_len], label) == null) {
            plan.default_out[plan.default_out_len] = label;
            default_seen[label] = true;
            plan.default_out_len += 1;
        }
    }
    if (explicit_output == null) {
        @memcpy(plan.out[0..plan.default_out_len], plan.default_out[0..plan.default_out_len]);
        plan.out_len = plan.default_out_len;
    }
    if (plan.out_len != plan.default_out_len) return error.InvalidShape;
    for (plan.out[0..plan.out_len], 0..) |label, out_axis| {
        if (!default_seen[label]) return error.InvalidShape;
        plan.permutation[out_axis] = findLabel(plan.default_out[0..plan.default_out_len], label) orelse return error.InvalidShape;
    }
    return plan;
}

fn parseEinsumLabels(segment: []const u8, expected_rank: ?usize, out: []u8) ArrayError!usize {
    if (segment.len > out.len) return error.InvalidShape;
    if (expected_rank) |rank| {
        if (segment.len != rank) return error.InvalidShape;
    }
    var seen = [_]bool{false} ** 256;
    for (segment, 0..) |label, index| {
        if (!std.ascii.isAlphabetic(label)) return error.InvalidShape;
        if (seen[label]) return error.InvalidShape;
        seen[label] = true;
        out[index] = label;
    }
    return segment.len;
}

fn allEinsumLabels(segment: []const u8) bool {
    for (segment) |label| {
        if (!std.ascii.isAlphabetic(label)) return false;
    }
    return true;
}

fn hasRepeatedLabels(segment: []const u8) bool {
    var seen = [_]bool{false} ** 256;
    for (segment) |label| {
        if (seen[label]) return true;
        seen[label] = true;
    }
    return false;
}

fn findLabel(labels: []const u8, needle: u8) ?usize {
    for (labels, 0..) |label, index| {
        if (label == needle) return index;
    }
    return null;
}
