const std = @import("std");

pub const EmaMetrics = struct {
    allocator: std.mem.Allocator,
    ema_values: []f64,
    residuals: []f64,
    ratios: []f64,
    validity: []bool,

    pub fn deinit(self: *EmaMetrics) void {
        self.allocator.free(self.ema_values);
        self.allocator.free(self.residuals);
        self.allocator.free(self.ratios);
        self.allocator.free(self.validity);
        self.* = undefined;
    }
};

fn validate(values: []const f64, maybe_validity: ?[]const bool, alpha: f64, min_periods: usize) error{ InvalidShape, LengthMismatch }!void {
    if (alpha <= 0 or alpha > 1 or min_periods == 0) return error.InvalidShape;
    if (maybe_validity) |validity| {
        if (validity.len != values.len) return error.LengthMismatch;
    }
}

fn rowValid(maybe_validity: ?[]const bool, row: usize) bool {
    return if (maybe_validity) |mask| mask[row] else true;
}

pub fn emaProfile(
    allocator: std.mem.Allocator,
    values: []const f64,
    maybe_validity: ?[]const bool,
    alpha: f64,
    min_periods: usize,
) (std.mem.Allocator.Error || error{ InvalidShape, LengthMismatch })!EmaMetrics {
    try validate(values, maybe_validity, alpha, min_periods);

    const ema_values = try allocator.alloc(f64, values.len);
    errdefer allocator.free(ema_values);
    const residuals = try allocator.alloc(f64, values.len);
    errdefer allocator.free(residuals);
    const ratios = try allocator.alloc(f64, values.len);
    errdefer allocator.free(ratios);
    const validity = try allocator.alloc(bool, values.len);
    errdefer allocator.free(validity);

    var seen: usize = 0;
    var ema: f64 = 0;
    // Null observations do not update EMA state. This keeps sequence gaps from
    // biasing the smoother while preserving row-aligned nullable outputs.
    for (values, 0..) |x, row| {
        if (!rowValid(maybe_validity, row)) {
            ema_values[row] = 0;
            residuals[row] = 0;
            ratios[row] = 0;
            validity[row] = false;
            continue;
        }

        if (seen == 0) {
            ema = x;
        } else {
            ema = alpha * x + (1.0 - alpha) * ema;
        }
        seen += 1;

        const has_enough = seen >= min_periods;
        validity[row] = has_enough;
        if (has_enough) {
            ema_values[row] = ema;
            residuals[row] = x - ema;
            ratios[row] = if (ema == 0) std.math.nan(f64) else x / ema;
        } else {
            ema_values[row] = 0;
            residuals[row] = 0;
            ratios[row] = 0;
        }
    }

    return .{
        .allocator = allocator,
        .ema_values = ema_values,
        .residuals = residuals,
        .ratios = ratios,
        .validity = validity,
    };
}
