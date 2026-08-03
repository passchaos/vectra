const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;
const DeviceLazyFrame = vectra.DeviceLazyFrame;
const helpers = @import("test_helpers.zig");
const lazyCollectTable = helpers.lazyCollectTable;
const lazyQualityTable = helpers.lazyQualityTable;

fn expectApproxOrNan(expected: f64, actual: f64) !void {
    if (std.math.isNan(expected)) {
        try std.testing.expect(std.math.isNan(actual));
    } else {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
    }
}

fn expectF64SliceApproxOrNan(expected: []const f64, actual: []const f64) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_item, actual_item| {
        try expectApproxOrNan(expected_item, actual_item);
    }
}

fn expectF64ColumnApproxOrNanWithValidity(frame: anytype, allocator: std.mem.Allocator, name: []const u8, expected_values: []const f64, expected_validity: []const bool) !void {
    const column = try frame.column(name);
    try std.testing.expect(column.f64.nullable());
    const values = try column.f64.toOwnedSlice(allocator);
    defer allocator.free(values);
    const validity = try column.f64.validity.?.toOwnedSlice(allocator);
    defer allocator.free(validity);
    try expectF64SliceApproxOrNan(expected_values, values);
    try std.testing.expectEqualSlices(bool, expected_validity, validity);
}

test "device lazy frame collects plan operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try plan.withColumnAbs("sales_abs", "sales");
    try plan.withColumnNeg("sales_neg", "sales");
    try plan.withColumnSign("sales_neg_sign", "sales_neg");
    try plan.withColumnSquare("sales_square", "sales");
    try plan.withColumnReciprocal("sales_recip", "sales");
    try plan.withColumnSqrt("sales_sqrt", "sales");
    try plan.withColumnRsqrt("sales_rsqrt", "sales");
    try plan.withColumnCbrt("sales_cbrt", "sales");
    try plan.withColumnFloor("sales_recip_floor", "sales_recip");
    try plan.withColumnCeil("sales_recip_ceil", "sales_recip");
    try plan.withColumnRound("sales_recip_round", "sales_recip");
    try plan.withColumnTrunc("sales_recip_trunc", "sales_recip");
    try plan.withColumnDeg2rad("sales_deg2rad", "sales");
    try plan.withColumnRad2deg("sales_roundtrip_deg", "sales_deg2rad");
    try plan.withColumnExpit("sales_recip_expit", "sales_recip");
    try plan.withColumnLogit("sales_recip_logit", "sales_recip");
    try plan.withColumnSoftplus("sales_recip_softplus", "sales_recip");
    try plan.withColumnLogsigmoid("sales_recip_logsigmoid", "sales_recip");
    try plan.withColumnRelu("sales_neg_relu", "sales_neg");
    try plan.withColumnLeakyRelu("sales_neg_leaky_relu", "sales_neg", f64, 0.1);
    try plan.withColumnRelu6("sales_relu6", "sales");
    try plan.withColumnPowScalar("sales_pow2", "sales", f64, 2.0);
    try plan.withColumnFloorDivScalar("sales_floor_div2", "sales", f64, 2.0);
    try plan.withColumnModScalar("sales_mod2", "sales", f64, 2.0);
    try plan.withColumnRemainderScalar("sales_remainder2", "sales", f64, 2.0);
    try plan.withColumnLogAddExpScalar("sales_logaddexp0", "sales", f64, 0.0);
    try plan.withColumnLogAddExp2Scalar("sales_logaddexp2_0", "sales", f64, 0.0);
    try plan.withColumnXlogyScalar("sales_xlogy_e", "sales", f64, std.math.e);
    try plan.withColumnFmaxScalar("sales_fmax4", "sales", f64, 4.0);
    try plan.withColumnFminScalar("sales_fmin4", "sales", f64, 4.0);
    try plan.withColumnHypotScalar("sales_hypot4", "sales", f64, 4.0);
    try plan.withColumnAtan2Scalar("sales_atan2_4", "sales", f64, 4.0);
    try plan.withColumnNextAfterScalar("sales_next_after8", "sales", f64, 8.0);
    try plan.withColumnCopysignScalar("sales_copysign_neg", "sales", f64, -1.0);
    try plan.withColumnHeavisideScalar("sales_neg_heaviside", "sales_neg", f64, 0.25);
    try plan.withColumnLdexpScalar("sales_ldexp1", "sales", 1);
    try plan.withColumnLerpScalar("sales_lerp_x2", "sales", "sales_x2", f64, 0.25);
    try plan.withColumnAddcmulScalar("sales_addcmul", "sales", "sales_recip", "sales", f64, 2.0);
    try plan.withColumnAddcdivScalar("sales_addcdiv", "sales", "sales", "sales_x2", f64, 0.5);
    try plan.withColumnClipArray("sales_clipped", "sales", "sales_recip", "sales_addcdiv");
    try plan.withColumnIscloseScalar("sales_close5", "sales", f64, 5.0, 0.0, 0.1);
    try plan.withColumnLogicalNot("active_not", "active");
    try plan.withColumnWhereScalar("sales_when_active_not", "sales", "active_not", f64, -1.0);
    try plan.withColumnWhere("sales_where_active_not", "sales", "active_not", "sales_neg");
    try plan.withColumnMaskedPutScalar("sales_masked_active_not", "sales", "active_not", f64, -2.0);
    try plan.withColumnLogicalOr("active_or_not", "active", "active_not");
    try plan.withColumnThreshold("sales_neg_threshold", "sales_neg", f64, -6.0, 0.0);
    try plan.withColumnHardtanh("sales_neg_hardtanh", "sales_neg", f64, -6.0, -1.0);
    try plan.withColumnMaximumScalar("sales_neg_max", "sales_neg", f64, -6.0);
    try plan.withColumnMinimumScalar("sales_neg_min", "sales_neg", f64, -6.0);
    try plan.withColumnClipMin("sales_neg_clip_min", "sales_neg", f64, -6.0);
    try plan.withColumnClipMax("sales_neg_clip_max", "sales_neg", f64, -5.5);
    try plan.withColumnHardshrink("sales_neg_hardshrink", "sales_neg", f64, 6.0);
    try plan.withColumnSoftshrink("sales_neg_softshrink", "sales_neg", f64, 6.0);
    try plan.withColumnTanhshrink("sales_neg_tanhshrink", "sales_neg");
    try plan.withColumnElu("sales_neg_elu", "sales_neg", f64, 0.5);
    try plan.withColumnCelu("sales_neg_celu", "sales_neg", f64, 2.0);
    try plan.withColumnSoftsign("sales_neg_softsign", "sales_neg");
    try plan.withColumnHardsigmoid("sales_neg_hardsigmoid", "sales_neg");
    try plan.withColumnHardswish("sales_neg_hardswish", "sales_neg");
    try plan.withColumnSilu("sales_neg_silu", "sales_neg");
    try plan.withColumnSwish("sales_neg_swish", "sales_neg");
    try plan.withColumnMish("sales_neg_mish", "sales_neg");
    try plan.withColumnGelu("sales_neg_gelu", "sales_neg");
    try plan.withColumnSelu("sales_neg_selu", "sales_neg");
    try plan.withColumnExp("sales_exp", "sales");
    try plan.withColumnExp2("sales_exp2", "sales");
    try plan.withColumnExpm1("sales_expm1", "sales");
    try plan.withColumnSin("sales_sin", "sales");
    try plan.withColumnCos("sales_cos", "sales");
    try plan.withColumnTan("sales_tan", "sales");
    try plan.withColumnAsin("sales_recip_asin", "sales_recip");
    try plan.withColumnAcos("sales_recip_acos", "sales_recip");
    try plan.withColumnAtan("sales_recip_atan", "sales_recip");
    try plan.withColumnSinh("sales_sinh", "sales");
    try plan.withColumnCosh("sales_cosh", "sales");
    try plan.withColumnTanh("sales_tanh", "sales");
    try plan.withColumnAsinh("sales_asinh", "sales");
    try plan.withColumnAcosh("sales_acosh", "sales");
    try plan.withColumnAtanh("sales_recip_atanh", "sales_recip");
    try plan.withColumnLog("sales_log", "sales");
    try plan.withColumnLog1p("sales_log1p", "sales");
    try plan.withColumnLgamma("sales_lgamma", "sales");
    try plan.withColumnSinc("sales_sinc", "sales");
    try plan.withColumnLog2("sales_log2", "sales");
    try plan.withColumnLog10("sales_log10", "sales");
    try plan.withColumnCompareScalar("big_sale", "sales_x2", f64, 10.0, .gt);
    try plan.filterColumnScalar("sales", f64, 2.5, .gt);
    try plan.sortBy("sales", .{ .descending = true });
    try plan.select(&.{ "sales", "units", "sales_x2", "sales_abs", "sales_neg", "sales_neg_sign", "sales_square", "sales_recip", "sales_sqrt", "sales_rsqrt", "sales_cbrt", "sales_recip_floor", "sales_recip_ceil", "sales_recip_round", "sales_recip_trunc", "sales_deg2rad", "sales_roundtrip_deg", "sales_recip_expit", "sales_recip_logit", "sales_recip_softplus", "sales_recip_logsigmoid", "sales_neg_relu", "sales_neg_leaky_relu", "sales_relu6", "sales_pow2", "sales_floor_div2", "sales_mod2", "sales_remainder2", "sales_logaddexp0", "sales_logaddexp2_0", "sales_xlogy_e", "sales_fmax4", "sales_fmin4", "sales_hypot4", "sales_atan2_4", "sales_next_after8", "sales_copysign_neg", "sales_neg_heaviside", "sales_ldexp1", "sales_lerp_x2", "sales_addcmul", "sales_addcdiv", "sales_clipped", "sales_close5", "active_not", "sales_when_active_not", "sales_where_active_not", "sales_masked_active_not", "active_or_not", "sales_neg_threshold", "sales_neg_hardtanh", "sales_neg_max", "sales_neg_min", "sales_neg_clip_min", "sales_neg_clip_max", "sales_neg_hardshrink", "sales_neg_softshrink", "sales_neg_tanhshrink", "sales_neg_elu", "sales_neg_celu", "sales_neg_softsign", "sales_neg_hardsigmoid", "sales_neg_hardswish", "sales_neg_silu", "sales_neg_swish", "sales_neg_mish", "sales_neg_gelu", "sales_neg_selu", "sales_exp", "sales_exp2", "sales_expm1", "sales_sin", "sales_cos", "sales_tan", "sales_recip_asin", "sales_recip_acos", "sales_recip_atan", "sales_sinh", "sales_cosh", "sales_tanh", "sales_asinh", "sales_acosh", "sales_recip_atanh", "sales_log", "sales_log1p", "sales_lgamma", "sales_sinc", "sales_log2", "sales_log10", "big_sale", "active" });
    try plan.select(&.{ "sales", "units", "sales_x2", "sales_abs", "sales_neg", "sales_neg_sign", "sales_square", "sales_recip", "sales_sqrt", "sales_rsqrt", "sales_cbrt", "sales_recip_floor", "sales_recip_ceil", "sales_recip_round", "sales_recip_trunc", "sales_deg2rad", "sales_roundtrip_deg", "sales_recip_expit", "sales_recip_logit", "sales_recip_softplus", "sales_recip_logsigmoid", "sales_neg_relu", "sales_neg_leaky_relu", "sales_relu6", "sales_pow2", "sales_floor_div2", "sales_mod2", "sales_remainder2", "sales_logaddexp0", "sales_logaddexp2_0", "sales_xlogy_e", "sales_fmax4", "sales_fmin4", "sales_hypot4", "sales_atan2_4", "sales_next_after8", "sales_copysign_neg", "sales_neg_heaviside", "sales_ldexp1", "sales_lerp_x2", "sales_addcmul", "sales_addcdiv", "sales_clipped", "sales_close5", "active_not", "sales_when_active_not", "sales_where_active_not", "sales_masked_active_not", "active_or_not", "sales_neg_threshold", "sales_neg_hardtanh", "sales_neg_max", "sales_neg_min", "sales_neg_clip_min", "sales_neg_clip_max", "sales_neg_hardshrink", "sales_neg_softshrink", "sales_neg_tanhshrink", "sales_neg_elu", "sales_neg_celu", "sales_neg_softsign", "sales_neg_hardsigmoid", "sales_neg_hardswish", "sales_neg_silu", "sales_neg_swish", "sales_neg_mish", "sales_neg_gelu", "sales_neg_selu", "sales_exp", "sales_exp2", "sales_expm1", "sales_sin", "sales_cos", "sales_tan", "sales_recip_asin", "sales_recip_acos", "sales_recip_atan", "sales_sinh", "sales_cosh", "sales_tanh", "sales_asinh", "sales_acosh", "sales_recip_atanh", "sales_log", "sales_log1p", "sales_lgamma", "sales_sinc", "sales_log2", "sales_log10", "big_sale" });
    try plan.head(3);
    try plan.head(2);

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "raw_ops=94") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "optimized_ops=92") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_scalar(sales_x2") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_abs(sales_abs=abs(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_neg(sales_neg=neg(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_sign(sales_neg_sign=sign(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_square(sales_square=square(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_reciprocal(sales_recip=reciprocal(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_sqrt(sales_sqrt=sqrt(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_rsqrt(sales_rsqrt=rsqrt(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_cbrt(sales_cbrt=cbrt(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_floor(sales_recip_floor=floor(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_ceil(sales_recip_ceil=ceil(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_round(sales_recip_round=round(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_trunc(sales_recip_trunc=trunc(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_deg2rad(sales_deg2rad=deg2rad(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_rad2deg(sales_roundtrip_deg=rad2deg(sales_deg2rad))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_expit(sales_recip_expit=expit(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_logit(sales_recip_logit=logit(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_softplus(sales_recip_softplus=softplus(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_logsigmoid(sales_recip_logsigmoid=logsigmoid(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_relu(sales_neg_relu=relu(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_leaky_relu(sales_neg_leaky_relu=leaky_relu(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_relu6(sales_relu6=relu6(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_pow_scalar(sales_pow2=pow(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_floor_div_scalar(sales_floor_div2=floor_div(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_mod_scalar(sales_mod2=mod(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_remainder_scalar(sales_remainder2=remainder(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_log_add_exp_scalar(sales_logaddexp0=log_add_exp(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_log_add_exp2_scalar(sales_logaddexp2_0=log_add_exp2(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_xlogy_scalar(sales_xlogy_e=xlogy(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_fmax_scalar(sales_fmax4=fmax(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_fmin_scalar(sales_fmin4=fmin(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_hypot_scalar(sales_hypot4=hypot(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_atan2_scalar(sales_atan2_4=atan2(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_next_after_scalar(sales_next_after8=next_after(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_copysign_scalar(sales_copysign_neg=copysign(sales, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_heaviside_scalar(sales_neg_heaviside=heaviside(sales_neg, value_at_zero:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_ldexp_scalar(sales_ldexp1=ldexp(sales, exponent:1))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_lerp_scalar(sales_lerp_x2=lerp(sales, sales_x2, weight:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_addcmul_scalar(sales_addcmul=addcmul(sales, sales_recip, sales, value:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_addcdiv_scalar(sales_addcdiv=addcdiv(sales, sales, sales_x2, value:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_clip_array(sales_clipped=clip_array(sales, min:sales_recip, max:sales_addcdiv))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_isclose_scalar(sales_close5=isclose(sales, scalar:f64, rtol:f64, atol:f64, equal_nan=false))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_logical_scalar(active_not=logical_xor(active, scalar:true))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_where_scalar(sales_when_active_not=where(sales, mask:active_not, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_where(sales_where_active_not=where(sales, mask:active_not, other:sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_masked_put_scalar(sales_masked_active_not=masked_put(sales, mask:active_not, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_logical(active_or_not=logical_or(active, active_not))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_threshold(sales_neg_threshold=threshold(sales_neg, threshold:f64, replacement:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_hardtanh(sales_neg_hardtanh=hardtanh(sales_neg, min:f64, max:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_maximum_scalar(sales_neg_max=maximum(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_minimum_scalar(sales_neg_min=minimum(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_clip_min(sales_neg_clip_min=clip_min(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_clip_max(sales_neg_clip_max=clip_max(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_hardshrink(sales_neg_hardshrink=hardshrink(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_softshrink(sales_neg_softshrink=softshrink(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_tanhshrink(sales_neg_tanhshrink=tanhshrink(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_elu(sales_neg_elu=elu(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_celu(sales_neg_celu=celu(sales_neg, scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_softsign(sales_neg_softsign=softsign(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_hardsigmoid(sales_neg_hardsigmoid=hardsigmoid(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_hardswish(sales_neg_hardswish=hardswish(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_silu(sales_neg_silu=silu(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_swish(sales_neg_swish=swish(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_mish(sales_neg_mish=mish(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_gelu(sales_neg_gelu=gelu(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_selu(sales_neg_selu=selu(sales_neg))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_exp(sales_exp=exp(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_exp2(sales_exp2=exp2(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_expm1(sales_expm1=expm1(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_sin(sales_sin=sin(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_cos(sales_cos=cos(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_tan(sales_tan=tan(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_asin(sales_recip_asin=asin(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_acos(sales_recip_acos=acos(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_atan(sales_recip_atan=atan(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_sinh(sales_sinh=sinh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_cosh(sales_cosh=cosh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_tanh(sales_tanh=tanh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_asinh(sales_asinh=asinh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_acosh(sales_acosh=acosh(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_atanh(sales_recip_atanh=atanh(sales_recip))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_log(sales_log=log(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_log1p(sales_log1p=log1p(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_lgamma(sales_lgamma=lgamma(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_sinc(sales_sinc=sinc(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_log2(sales_log2=log2(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_log10(sales_log10=log10(sales))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_compare_scalar(big_sale") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "filter_scalar(sales") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.height());
    try std.testing.expectEqual(@as(usize, 90), result.width());
    const result_sales = try (try result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales);
    const result_units = try (try result.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(result_units);
    const result_sales_x2 = try (try result.column("sales_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_x2);
    const result_sales_abs = try (try result.column("sales_abs")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_abs);
    const result_sales_neg = try (try result.column("sales_neg")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg);
    const result_sales_neg_sign = try (try result.column("sales_neg_sign")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_sign);
    const result_sales_square = try (try result.column("sales_square")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_square);
    const result_sales_recip = try (try result.column("sales_recip")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip);
    const result_sales_sqrt = try (try result.column("sales_sqrt")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_sqrt);
    const result_sales_rsqrt = try (try result.column("sales_rsqrt")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_rsqrt);
    const result_sales_cbrt = try (try result.column("sales_cbrt")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_cbrt);
    const result_sales_recip_floor = try (try result.column("sales_recip_floor")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_floor);
    const result_sales_recip_ceil = try (try result.column("sales_recip_ceil")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_ceil);
    const result_sales_recip_round = try (try result.column("sales_recip_round")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_round);
    const result_sales_recip_trunc = try (try result.column("sales_recip_trunc")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_trunc);
    const result_sales_deg2rad = try (try result.column("sales_deg2rad")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_deg2rad);
    const result_sales_roundtrip_deg = try (try result.column("sales_roundtrip_deg")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_roundtrip_deg);
    const result_sales_recip_expit = try (try result.column("sales_recip_expit")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_expit);
    const result_sales_recip_logit = try (try result.column("sales_recip_logit")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_logit);
    const result_sales_recip_softplus = try (try result.column("sales_recip_softplus")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_softplus);
    const result_sales_recip_logsigmoid = try (try result.column("sales_recip_logsigmoid")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_logsigmoid);
    const result_sales_neg_relu = try (try result.column("sales_neg_relu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_relu);
    const result_sales_neg_leaky_relu = try (try result.column("sales_neg_leaky_relu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_leaky_relu);
    const result_sales_relu6 = try (try result.column("sales_relu6")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_relu6);
    const result_sales_pow2 = try (try result.column("sales_pow2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_pow2);
    const result_sales_floor_div2 = try (try result.column("sales_floor_div2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_floor_div2);
    const result_sales_mod2 = try (try result.column("sales_mod2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_mod2);
    const result_sales_remainder2 = try (try result.column("sales_remainder2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_remainder2);
    const result_sales_logaddexp0 = try (try result.column("sales_logaddexp0")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_logaddexp0);
    const result_sales_logaddexp2_0 = try (try result.column("sales_logaddexp2_0")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_logaddexp2_0);
    const result_sales_xlogy_e = try (try result.column("sales_xlogy_e")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_xlogy_e);
    const result_sales_fmax4 = try (try result.column("sales_fmax4")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_fmax4);
    const result_sales_fmin4 = try (try result.column("sales_fmin4")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_fmin4);
    const result_sales_hypot4 = try (try result.column("sales_hypot4")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_hypot4);
    const result_sales_atan2_4 = try (try result.column("sales_atan2_4")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_atan2_4);
    const result_sales_next_after8 = try (try result.column("sales_next_after8")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_next_after8);
    const result_sales_copysign_neg = try (try result.column("sales_copysign_neg")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_copysign_neg);
    const result_sales_neg_heaviside = try (try result.column("sales_neg_heaviside")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_heaviside);
    const result_sales_ldexp1 = try (try result.column("sales_ldexp1")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_ldexp1);
    const result_sales_lerp_x2 = try (try result.column("sales_lerp_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_lerp_x2);
    const result_sales_addcmul = try (try result.column("sales_addcmul")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_addcmul);
    const result_sales_addcdiv = try (try result.column("sales_addcdiv")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_addcdiv);
    const result_sales_clipped = try (try result.column("sales_clipped")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_clipped);
    const result_sales_close5 = try (try result.column("sales_close5")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_sales_close5);
    const result_active_not = try (try result.column("active_not")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_active_not);
    const result_sales_when_active_not = try (try result.column("sales_when_active_not")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_when_active_not);
    const result_sales_where_active_not = try (try result.column("sales_where_active_not")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_where_active_not);
    const result_sales_masked_active_not = try (try result.column("sales_masked_active_not")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_masked_active_not);
    const result_active_or_not = try (try result.column("active_or_not")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_active_or_not);
    const result_sales_neg_threshold = try (try result.column("sales_neg_threshold")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_threshold);
    const result_sales_neg_hardtanh = try (try result.column("sales_neg_hardtanh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_hardtanh);
    const result_sales_neg_max = try (try result.column("sales_neg_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_max);
    const result_sales_neg_min = try (try result.column("sales_neg_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_min);
    const result_sales_neg_clip_min = try (try result.column("sales_neg_clip_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_clip_min);
    const result_sales_neg_clip_max = try (try result.column("sales_neg_clip_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_clip_max);
    const result_sales_neg_hardshrink = try (try result.column("sales_neg_hardshrink")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_hardshrink);
    const result_sales_neg_softshrink = try (try result.column("sales_neg_softshrink")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_softshrink);
    const result_sales_neg_tanhshrink = try (try result.column("sales_neg_tanhshrink")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_tanhshrink);
    const result_sales_neg_elu = try (try result.column("sales_neg_elu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_elu);
    const result_sales_neg_celu = try (try result.column("sales_neg_celu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_celu);
    const result_sales_neg_softsign = try (try result.column("sales_neg_softsign")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_softsign);
    const result_sales_neg_hardsigmoid = try (try result.column("sales_neg_hardsigmoid")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_hardsigmoid);
    const result_sales_neg_hardswish = try (try result.column("sales_neg_hardswish")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_hardswish);
    const result_sales_neg_silu = try (try result.column("sales_neg_silu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_silu);
    const result_sales_neg_swish = try (try result.column("sales_neg_swish")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_swish);
    const result_sales_neg_mish = try (try result.column("sales_neg_mish")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_mish);
    const result_sales_neg_gelu = try (try result.column("sales_neg_gelu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_gelu);
    const result_sales_neg_selu = try (try result.column("sales_neg_selu")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_neg_selu);
    const result_sales_exp = try (try result.column("sales_exp")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_exp);
    const result_sales_exp2 = try (try result.column("sales_exp2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_exp2);
    const result_sales_expm1 = try (try result.column("sales_expm1")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_expm1);
    const result_sales_sin = try (try result.column("sales_sin")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_sin);
    const result_sales_cos = try (try result.column("sales_cos")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_cos);
    const result_sales_tan = try (try result.column("sales_tan")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_tan);
    const result_sales_recip_asin = try (try result.column("sales_recip_asin")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_asin);
    const result_sales_recip_acos = try (try result.column("sales_recip_acos")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_acos);
    const result_sales_recip_atan = try (try result.column("sales_recip_atan")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_atan);
    const result_sales_sinh = try (try result.column("sales_sinh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_sinh);
    const result_sales_cosh = try (try result.column("sales_cosh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_cosh);
    const result_sales_tanh = try (try result.column("sales_tanh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_tanh);
    const result_sales_asinh = try (try result.column("sales_asinh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_asinh);
    const result_sales_acosh = try (try result.column("sales_acosh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_acosh);
    const result_sales_recip_atanh = try (try result.column("sales_recip_atanh")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_recip_atanh);
    const result_sales_log = try (try result.column("sales_log")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_log);
    const result_sales_log1p = try (try result.column("sales_log1p")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_log1p);
    const result_sales_lgamma = try (try result.column("sales_lgamma")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_lgamma);
    const result_sales_sinc = try (try result.column("sales_sinc")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_sinc);
    const result_sales_log2 = try (try result.column("sales_log2")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_log2);
    const result_sales_log10 = try (try result.column("sales_log10")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales_log10);
    const result_big_sale = try (try result.column("big_sale")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_big_sale);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, result_sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3 }, result_units);
    try std.testing.expectEqualSlices(f64, &.{ 14.0, 10.0 }, result_sales_x2);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, result_sales_abs);
    try std.testing.expectEqualSlices(f64, &.{ -7.0, -5.0 }, result_sales_neg);
    try std.testing.expectEqualSlices(f64, &.{ -1.0, -1.0 }, result_sales_neg_sign);
    try std.testing.expectEqualSlices(f64, &.{ 49.0, 25.0 }, result_sales_square);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 7.0), result_sales_recip[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), result_sales_recip[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 7.0)), result_sales_sqrt[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 5.0)), result_sales_sqrt[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / std.math.sqrt(@as(f64, 7.0)), result_sales_rsqrt[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / std.math.sqrt(@as(f64, 5.0)), result_sales_rsqrt[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cbrt(@as(f64, 7.0)), result_sales_cbrt[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cbrt(@as(f64, 5.0)), result_sales_cbrt[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_recip_floor);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0 }, result_sales_recip_ceil);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_recip_round);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_recip_trunc);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0) * std.math.pi / @as(f64, 180.0), result_sales_deg2rad[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) * std.math.pi / @as(f64, 180.0), result_sales_deg2rad[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), result_sales_roundtrip_deg[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result_sales_roundtrip_deg[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(-@as(f64, 1.0 / 7.0))), result_sales_recip_expit[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(-@as(f64, 0.2))), result_sales_recip_expit[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, (@as(f64, 1.0 / 7.0)) / (@as(f64, 1.0) - @as(f64, 1.0 / 7.0))), result_sales_recip_logit[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 0.2) / @as(f64, 0.8)), result_sales_recip_logit[1], 1e-12);
    try std.testing.expectApproxEqAbs(@max(@as(f64, 1.0 / 7.0), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 1.0 / 7.0)))), result_sales_recip_softplus[0], 1e-12);
    try std.testing.expectApproxEqAbs(@max(@as(f64, 0.2), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.2)))), result_sales_recip_softplus[1], 1e-12);
    try std.testing.expectApproxEqAbs(-(@max(-@as(f64, 1.0 / 7.0), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 1.0 / 7.0))))), result_sales_recip_logsigmoid[0], 1e-12);
    try std.testing.expectApproxEqAbs(-(@max(-@as(f64, 0.2), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, 0.2))))), result_sales_recip_logsigmoid[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_neg_relu);
    try std.testing.expectApproxEqAbs(@as(f64, -0.7), result_sales_neg_leaky_relu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), result_sales_neg_leaky_relu[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 6.0, 5.0 }, result_sales_relu6);
    try std.testing.expectEqualSlices(f64, &.{ 49.0, 25.0 }, result_sales_pow2);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 2.0 }, result_sales_floor_div2);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0 }, result_sales_mod2);
    try std.testing.expectEqualSlices(f64, result_sales_mod2, result_sales_remainder2);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0) + std.math.log1p(std.math.exp(@as(f64, -7.0))), result_sales_logaddexp0[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + std.math.log1p(std.math.exp(@as(f64, -5.0))), result_sales_logaddexp0[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0) + std.math.log2(@as(f64, 1.0) + std.math.pow(f64, 2.0, -@as(f64, 7.0))), result_sales_logaddexp2_0[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + std.math.log2(@as(f64, 1.0) + std.math.pow(f64, 2.0, -@as(f64, 5.0))), result_sales_logaddexp2_0[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), result_sales_xlogy_e[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result_sales_xlogy_e[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, result_sales_fmax4);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 4.0 }, result_sales_fmin4);
    try std.testing.expectApproxEqAbs(std.math.hypot(@as(f64, 7.0), @as(f64, 4.0)), result_sales_hypot4[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.hypot(@as(f64, 5.0), @as(f64, 4.0)), result_sales_hypot4[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 7.0), @as(f64, 4.0)), result_sales_atan2_4[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 5.0), @as(f64, 4.0)), result_sales_atan2_4[1], 1e-12);
    try std.testing.expectEqual(std.math.nextAfter(f64, @as(f64, 7.0), @as(f64, 8.0)), result_sales_next_after8[0]);
    try std.testing.expectEqual(std.math.nextAfter(f64, @as(f64, 5.0), @as(f64, 8.0)), result_sales_next_after8[1]);
    try std.testing.expectEqualSlices(f64, &.{ -7.0, -5.0 }, result_sales_copysign_neg);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_neg_heaviside);
    try std.testing.expectEqualSlices(f64, &.{ 14.0, 10.0 }, result_sales_ldexp1);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0) + (@as(f64, 14.0) - @as(f64, 7.0)) * @as(f64, 0.25), result_sales_lerp_x2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + (@as(f64, 10.0) - @as(f64, 5.0)) * @as(f64, 0.25), result_sales_lerp_x2[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result_sales_addcmul[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), result_sales_addcmul[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.25), result_sales_addcdiv[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.25), result_sales_addcdiv[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, result_sales_clipped);
    try std.testing.expectEqualSlices(bool, &.{ false, true }, result_sales_close5);
    try std.testing.expectEqualSlices(bool, &.{ false, false }, result_active_not);
    try std.testing.expectEqualSlices(f64, &.{ -1.0, -1.0 }, result_sales_when_active_not);
    try std.testing.expectEqualSlices(f64, &.{ -7.0, -5.0 }, result_sales_where_active_not);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, result_sales_masked_active_not);
    try std.testing.expectEqualSlices(bool, &.{ true, true }, result_active_or_not);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, -5.0 }, result_sales_neg_threshold);
    try std.testing.expectEqualSlices(f64, &.{ -6.0, -5.0 }, result_sales_neg_hardtanh);
    try std.testing.expectEqualSlices(f64, &.{ -6.0, -5.0 }, result_sales_neg_max);
    try std.testing.expectEqualSlices(f64, &.{ -7.0, -6.0 }, result_sales_neg_min);
    try std.testing.expectEqualSlices(f64, &.{ -6.0, -5.0 }, result_sales_neg_clip_min);
    try std.testing.expectEqualSlices(f64, &.{ -7.0, -5.5 }, result_sales_neg_clip_max);
    try std.testing.expectEqualSlices(f64, &.{ -7.0, 0.0 }, result_sales_neg_hardshrink);
    try std.testing.expectEqualSlices(f64, &.{ -1.0, 0.0 }, result_sales_neg_softshrink);
    try std.testing.expectApproxEqAbs(@as(f64, -7.0) - std.math.tanh(@as(f64, -7.0)), result_sales_neg_tanhshrink[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) - std.math.tanh(@as(f64, -5.0)), result_sales_neg_tanhshrink[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) * std.math.expm1(@as(f64, -7.0)), result_sales_neg_elu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5) * std.math.expm1(@as(f64, -5.0)), result_sales_neg_elu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) * std.math.expm1(@as(f64, -7.0) / @as(f64, 2.0)), result_sales_neg_celu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) * std.math.expm1(@as(f64, -5.0) / @as(f64, 2.0)), result_sales_neg_celu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -7.0) / @as(f64, 8.0), result_sales_neg_softsign[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) / @as(f64, 6.0), result_sales_neg_softsign[1], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0 }, result_sales_neg_hardsigmoid);
    try std.testing.expectEqualSlices(f64, &.{ -0.0, -0.0 }, result_sales_neg_hardswish);
    try std.testing.expectApproxEqAbs(@as(f64, -7.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, 7.0))), result_sales_neg_silu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, 5.0))), result_sales_neg_silu[1], 1e-12);
    try std.testing.expectApproxEqAbs(result_sales_neg_silu[0], result_sales_neg_swish[0], 1e-12);
    try std.testing.expectApproxEqAbs(result_sales_neg_silu[1], result_sales_neg_swish[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -7.0) * std.math.tanh(@max(@as(f64, -7.0), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, -7.0))))), result_sales_neg_mish[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) * std.math.tanh(@max(@as(f64, -5.0), @as(f64, 0.0)) + std.math.log1p(std.math.exp(-@abs(@as(f64, -5.0))))), result_sales_neg_mish[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -7.0) * @as(f64, 0.5) * (@as(f64, 1.0) + std.math.tanh(@sqrt(@as(f64, 2.0) / std.math.pi) * (@as(f64, -7.0) + @as(f64, 0.044715) * @as(f64, -343.0)))), result_sales_neg_gelu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -5.0) * @as(f64, 0.5) * (@as(f64, 1.0) + std.math.tanh(@sqrt(@as(f64, 2.0) / std.math.pi) * (@as(f64, -5.0) + @as(f64, 0.044715) * @as(f64, -125.0)))), result_sales_neg_gelu[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0507009873554805) * @as(f64, 1.6732632423543772) * std.math.expm1(@as(f64, -7.0)), result_sales_neg_selu[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0507009873554805) * @as(f64, 1.6732632423543772) * std.math.expm1(@as(f64, -5.0)), result_sales_neg_selu[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f64, 7.0)), result_sales_exp[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f64, 5.0)), result_sales_exp[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp2(@as(f64, 7.0)), result_sales_exp2[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp2(@as(f64, 5.0)), result_sales_exp2[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.expm1(@as(f64, 7.0)), result_sales_expm1[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.expm1(@as(f64, 5.0)), result_sales_expm1[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(@as(f64, 7.0)), result_sales_sin[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(@as(f64, 5.0)), result_sales_sin[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cos(@as(f64, 7.0)), result_sales_cos[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cos(@as(f64, 5.0)), result_sales_cos[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tan(@as(f64, 7.0)), result_sales_tan[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tan(@as(f64, 5.0)), result_sales_tan[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asin(@as(f64, 1.0 / 7.0)), result_sales_recip_asin[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asin(@as(f64, 0.2)), result_sales_recip_asin[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acos(@as(f64, 1.0 / 7.0)), result_sales_recip_acos[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acos(@as(f64, 0.2)), result_sales_recip_acos[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan(@as(f64, 1.0 / 7.0)), result_sales_recip_atan[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atan(@as(f64, 0.2)), result_sales_recip_atan[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sinh(@as(f64, 7.0)), result_sales_sinh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sinh(@as(f64, 5.0)), result_sales_sinh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cosh(@as(f64, 7.0)), result_sales_cosh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.cosh(@as(f64, 5.0)), result_sales_cosh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f64, 7.0)), result_sales_tanh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f64, 5.0)), result_sales_tanh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asinh(@as(f64, 7.0)), result_sales_asinh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.asinh(@as(f64, 5.0)), result_sales_asinh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acosh(@as(f64, 7.0)), result_sales_acosh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.acosh(@as(f64, 5.0)), result_sales_acosh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atanh(@as(f64, 1.0 / 7.0)), result_sales_recip_atanh[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.atanh(@as(f64, 0.2)), result_sales_recip_atanh[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 7.0)), result_sales_log[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 5.0)), result_sales_log[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log1p(@as(f64, 7.0)), result_sales_log1p[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log1p(@as(f64, 5.0)), result_sales_log1p[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.lgamma(f64, @as(f64, 7.0)), result_sales_lgamma[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.lgamma(f64, @as(f64, 5.0)), result_sales_lgamma[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(std.math.pi * @as(f64, 7.0)) / (std.math.pi * @as(f64, 7.0)), result_sales_sinc[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sin(std.math.pi * @as(f64, 5.0)) / (std.math.pi * @as(f64, 5.0)), result_sales_sinc[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log2(@as(f64, 7.0)), result_sales_log2[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log2(@as(f64, 5.0)), result_sales_log2[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log10(@as(f64, 7.0)), result_sales_log10[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log10(@as(f64, 5.0)), result_sales_log10[1], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, result_big_sale);
}

test "device lazy frame derives between predicate columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnBetween("sales_between_closed", "sales", f64, 3.0, 5.0);
    try plan.withColumnBetweenExclusive("sales_between_open", "sales", f64, 2.0, 7.0);
    try plan.withColumnBetweenLeftClosed("sales_between_left", "sales", f64, 2.0, 7.0);
    try plan.withColumnBetweenRightClosed("sales_between_right", "sales", f64, 2.0, 7.0);
    try plan.withColumnNotBetween("sales_not_between", "sales", f64, 3.0, 5.0);
    try plan.withColumnOutside("sales_outside", "sales", f64, 2.0, 7.0);
    try plan.select(&.{ "sales_between_closed", "sales_between_open", "sales_between_left", "sales_between_right", "sales_not_between", "sales_outside" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_between(sales_between_closed=between(sales, lower:f64, upper:f64, lower_inclusive=true, upper_inclusive=true))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_between(sales_between_open=between(sales, lower:f64, upper:f64, lower_inclusive=false, upper_inclusive=false))") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_logical_scalar(sales_not_between=logical_xor(sales_not_between, scalar:true))") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 6), result.width());
    const closed = try (try result.column("sales_between_closed")).bool.toOwnedSlice(gpa);
    defer gpa.free(closed);
    const open = try (try result.column("sales_between_open")).bool.toOwnedSlice(gpa);
    defer gpa.free(open);
    const left = try (try result.column("sales_between_left")).bool.toOwnedSlice(gpa);
    defer gpa.free(left);
    const right = try (try result.column("sales_between_right")).bool.toOwnedSlice(gpa);
    defer gpa.free(right);
    const not_between = try (try result.column("sales_not_between")).bool.toOwnedSlice(gpa);
    defer gpa.free(not_between);
    const outside = try (try result.column("sales_outside")).bool.toOwnedSlice(gpa);
    defer gpa.free(outside);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, closed);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, open);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, left);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, right);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, not_between);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, outside);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withColumnBetween("bad_between", "active", f64, 0.0, 1.0);
    try std.testing.expectError(error.TypeUnsupported, invalid_plan.collect());
}

test "device lazy frame filters by named boolean columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnCompareScalar("big_sale", "sales", f64, 4.0, .gt);
    try plan.filterColumn("big_sale");
    try plan.select(&.{ "sales", "big_sale" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "filter_column(big_sale)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.height());
    try std.testing.expectEqual(@as(usize, 2), result.width());
    const result_sales = try (try result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(result_sales);
    const result_big_sale = try (try result.column("big_sale")).bool.toOwnedSlice(gpa);
    defer gpa.free(result_big_sale);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 7.0 }, result_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, true }, result_big_sale);

    var source_bool_plan = try DeviceLazyFrame.init(gpa, table);
    defer source_bool_plan.deinit();
    try source_bool_plan.filterColumn("active");
    try source_bool_plan.select(&.{ "sales", "active" });
    var active_result = try source_bool_plan.collect();
    defer active_result.deinit();
    const active_sales = try (try active_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(active_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0, 7.0 }, active_sales);

    var range_plan = try DeviceLazyFrame.init(gpa, table);
    defer range_plan.deinit();
    try range_plan.filterBetweenColumn("sales", f64, 3.0, 5.0);
    try range_plan.select(&.{"sales"});
    const range_explain = try range_plan.explain(gpa);
    defer gpa.free(range_explain);
    try std.testing.expect(std.mem.indexOf(u8, range_explain, "filter_between_column(sales, lower:f64, upper:f64, lower_inclusive=true, upper_inclusive=true, keep_inside=true)") != null);
    var range_result = try range_plan.collect();
    defer range_result.deinit();
    const range_sales = try (try range_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(range_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, range_sales);

    var outside_plan = try DeviceLazyFrame.init(gpa, table);
    defer outside_plan.deinit();
    try outside_plan.filterOutsideColumn("sales", f64, 3.0, 5.0);
    try outside_plan.select(&.{"sales"});
    const outside_explain = try outside_plan.explain(gpa);
    defer gpa.free(outside_explain);
    try std.testing.expect(std.mem.indexOf(u8, outside_explain, "filter_between_column(sales, lower:f64, upper:f64, lower_inclusive=true, upper_inclusive=true, keep_inside=false)") != null);
    var outside_result = try outside_plan.collect();
    defer outside_result.deinit();
    const outside_sales = try (try outside_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(outside_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 7.0 }, outside_sales);

    var invalid_range_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_range_plan.deinit();
    try invalid_range_plan.filterBetweenColumn("active", f64, 0.0, 1.0);
    try std.testing.expectError(error.TypeUnsupported, invalid_range_plan.collect());

    var scalar_filter_plan = try DeviceLazyFrame.init(gpa, table);
    defer scalar_filter_plan.deinit();
    try scalar_filter_plan.filterColumnScalar("sales", f64, 2.5, .gt);
    try scalar_filter_plan.select(&.{"sales"});
    const scalar_filter_explain = try scalar_filter_plan.explain(gpa);
    defer gpa.free(scalar_filter_explain);
    try std.testing.expect(std.mem.indexOf(u8, scalar_filter_explain, "filter_scalar(sales, op=gt, dtype=f64, keep_matches=true)") != null);
    var scalar_filter_result = try scalar_filter_plan.collect();
    defer scalar_filter_result.deinit();
    const scalar_filter_sales = try (try scalar_filter_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(scalar_filter_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0, 7.0 }, scalar_filter_sales);

    var scalar_drop_plan = try DeviceLazyFrame.init(gpa, table);
    defer scalar_drop_plan.deinit();
    try scalar_drop_plan.dropColumnScalar("sales", f64, 2.5, .gt);
    try scalar_drop_plan.select(&.{"sales"});
    const scalar_drop_explain = try scalar_drop_plan.explain(gpa);
    defer gpa.free(scalar_drop_explain);
    try std.testing.expect(std.mem.indexOf(u8, scalar_drop_explain, "filter_scalar(sales, op=gt, dtype=f64, keep_matches=false)") != null);
    var scalar_drop_result = try scalar_drop_plan.collect();
    defer scalar_drop_result.deinit();
    const scalar_drop_sales = try (try scalar_drop_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(scalar_drop_sales);
    try std.testing.expectEqualSlices(f64, &.{2.0}, scalar_drop_sales);

    var invalid_scalar_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_scalar_plan.deinit();
    try invalid_scalar_plan.dropColumnScalar("active", f64, 0.0, .gt);
    try std.testing.expectError(error.TypeUnsupported, invalid_scalar_plan.collect());

    var where_indices_plan = try DeviceLazyFrame.init(gpa, table);
    defer where_indices_plan.deinit();
    try where_indices_plan.whereIndicesColumn("active", "active_row");
    const where_indices_explain = try where_indices_plan.explain(gpa);
    defer gpa.free(where_indices_explain);
    try std.testing.expect(std.mem.indexOf(u8, where_indices_explain, "where_indices_column(active->active_row)") != null);
    var where_indices_result = try where_indices_plan.collect();
    defer where_indices_result.deinit();
    try std.testing.expectEqual(@as(usize, 1), where_indices_result.width());
    const active_rows = try (try where_indices_result.column("active_row")).usize.toOwnedSlice(gpa);
    defer gpa.free(active_rows);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3 }, active_rows);

    var isin_plan = try DeviceLazyFrame.init(gpa, table);
    defer isin_plan.deinit();
    try isin_plan.withColumnLiteral("needle", f64, 3.0);
    try isin_plan.withColumnIsIn("sales_isin", "sales", "needle");
    try isin_plan.withColumnIsInInverted("sales_notin", "sales", "needle");
    try isin_plan.select(&.{ "sales", "sales_isin", "sales_notin" });
    const isin_explain = try isin_plan.explain(gpa);
    defer gpa.free(isin_explain);
    try std.testing.expect(std.mem.indexOf(u8, isin_explain, "with_column_isin(sales_isin=isin(sales, test:needle, invert=false))") != null);
    try std.testing.expect(std.mem.indexOf(u8, isin_explain, "with_column_isin(sales_notin=isin(sales, test:needle, invert=true))") != null);
    var isin_result = try isin_plan.collect();
    defer isin_result.deinit();
    const sales_isin = try (try isin_result.column("sales_isin")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_isin);
    const sales_notin = try (try isin_result.column("sales_notin")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_notin);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, sales_isin);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, sales_notin);

    var isin_values_plan = try DeviceLazyFrame.init(gpa, table);
    defer isin_values_plan.deinit();
    try isin_values_plan.withColumnIsInValues("sales_isin_values", "sales", f64, &.{ 3.0, 7.0 });
    try isin_values_plan.select(&.{ "sales", "sales_isin_values" });
    const isin_values_explain = try isin_values_plan.explain(gpa);
    defer gpa.free(isin_values_explain);
    try std.testing.expect(std.mem.indexOf(u8, isin_values_explain, "with_column_isin_values(sales_isin_values=isin(sales, values_dtype=f64, values_len=2, invert=false))") != null);
    var isin_values_result = try isin_values_plan.collect();
    defer isin_values_result.deinit();
    const sales_isin_values = try (try isin_values_result.column("sales_isin_values")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_isin_values);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, sales_isin_values);

    var filter_isin_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_isin_plan.deinit();
    try filter_isin_plan.withColumnLiteral("needle", f64, 3.0);
    try filter_isin_plan.filterIsInColumn("sales", "needle");
    try filter_isin_plan.select(&.{"sales"});
    const filter_isin_explain = try filter_isin_plan.explain(gpa);
    defer gpa.free(filter_isin_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_isin_explain, "filter_isin_column(sales, test:needle, invert=false)") != null);
    var filter_isin_result = try filter_isin_plan.collect();
    defer filter_isin_result.deinit();
    const filter_isin_sales = try (try filter_isin_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(filter_isin_sales);
    try std.testing.expectEqualSlices(f64, &.{3.0}, filter_isin_sales);

    var filter_notin_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_notin_plan.deinit();
    try filter_notin_plan.withColumnLiteral("needle", f64, 3.0);
    try filter_notin_plan.filterNotInColumn("sales", "needle");
    try filter_notin_plan.select(&.{"sales"});
    const filter_notin_explain = try filter_notin_plan.explain(gpa);
    defer gpa.free(filter_notin_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_notin_explain, "filter_isin_column(sales, test:needle, invert=true)") != null);
    var filter_notin_result = try filter_notin_plan.collect();
    defer filter_notin_result.deinit();
    const filter_notin_sales = try (try filter_notin_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(filter_notin_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0, 7.0 }, filter_notin_sales);

    var filter_values_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_values_plan.deinit();
    try filter_values_plan.filterIsInValues("sales", f64, &.{ 3.0, 7.0 });
    try filter_values_plan.select(&.{"sales"});
    const filter_values_explain = try filter_values_plan.explain(gpa);
    defer gpa.free(filter_values_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_values_explain, "filter_isin_values(sales, values_dtype=f64, values_len=2, invert=false)") != null);
    var filter_values_result = try filter_values_plan.collect();
    defer filter_values_result.deinit();
    const filter_values_sales = try (try filter_values_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(filter_values_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 7.0 }, filter_values_sales);

    var filter_not_values_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_not_values_plan.deinit();
    try filter_not_values_plan.filterNotInValues("sales", f64, &.{ 3.0, 7.0 });
    try filter_not_values_plan.select(&.{"sales"});
    var filter_not_values_result = try filter_not_values_plan.collect();
    defer filter_not_values_result.deinit();
    const filter_not_values_sales = try (try filter_not_values_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(filter_not_values_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, filter_not_values_sales);

    var drop_values_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_values_plan.deinit();
    try drop_values_plan.dropIsInValues("sales", f64, &.{ 3.0, 7.0 });
    try drop_values_plan.select(&.{"sales"});
    var drop_values_result = try drop_values_plan.collect();
    defer drop_values_result.deinit();
    const drop_values_sales = try (try drop_values_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(drop_values_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, drop_values_sales);

    var drop_isin_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_isin_plan.deinit();
    try drop_isin_plan.withColumnLiteral("needle", f64, 3.0);
    try drop_isin_plan.dropIsInColumn("sales", "needle");
    try drop_isin_plan.select(&.{"sales"});
    var drop_isin_result = try drop_isin_plan.collect();
    defer drop_isin_result.deinit();
    const drop_isin_sales = try (try drop_isin_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(drop_isin_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0, 7.0 }, drop_isin_sales);

    var drop_notin_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_notin_plan.deinit();
    try drop_notin_plan.withColumnLiteral("needle", f64, 3.0);
    try drop_notin_plan.dropNotInColumn("sales", "needle");
    try drop_notin_plan.select(&.{"sales"});
    var drop_notin_result = try drop_notin_plan.collect();
    defer drop_notin_result.deinit();
    const drop_notin_sales = try (try drop_notin_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(drop_notin_sales);
    try std.testing.expectEqualSlices(f64, &.{3.0}, drop_notin_sales);

    var invalid_isin_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_isin_plan.deinit();
    try invalid_isin_plan.filterIsInColumn("sales", "active");
    try std.testing.expectError(error.TypeUnsupported, invalid_isin_plan.collect());

    var drop_mask_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_mask_plan.deinit();
    try drop_mask_plan.dropRowsByColumnMask("active");
    try drop_mask_plan.select(&.{ "sales", "active" });
    const drop_mask_explain = try drop_mask_plan.explain(gpa);
    defer gpa.free(drop_mask_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_mask_explain, "drop_rows_by_mask_column(active)") != null);
    var drop_mask_result = try drop_mask_plan.collect();
    defer drop_mask_result.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_mask_result.height());
    const drop_mask_sales = try (try drop_mask_result.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(drop_mask_sales);
    try std.testing.expectEqualSlices(f64, &.{3.0}, drop_mask_sales);
}

test "device lazy frame selects and drops columns by dtype" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var positional_plan = try DeviceLazyFrame.init(gpa, table);
    defer positional_plan.deinit();
    try positional_plan.selectByColumnIndices(&.{ 2, 0 });
    const positional_explain = try positional_plan.explain(gpa);
    defer gpa.free(positional_explain);
    try std.testing.expect(std.mem.indexOf(u8, positional_explain, "select_column_indices([2,0])") != null);
    var positional = try positional_plan.collect();
    defer positional.deinit();
    try std.testing.expectEqual(@as(usize, 2), positional.width());
    try std.testing.expectEqual(@as(?usize, 0), positional.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), positional.columnIndex("sales"));

    var range_plan = try DeviceLazyFrame.init(gpa, table);
    defer range_plan.deinit();
    try range_plan.selectColumnRange(1, 3);
    const range_explain = try range_plan.explain(gpa);
    defer gpa.free(range_explain);
    try std.testing.expect(std.mem.indexOf(u8, range_explain, "select_column_range(1..3)") != null);
    var range = try range_plan.collect();
    defer range.deinit();
    try std.testing.expectEqual(@as(usize, 2), range.width());
    try std.testing.expectEqual(@as(?usize, 0), range.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), range.columnIndex("active"));

    var first_plan = try DeviceLazyFrame.init(gpa, table);
    defer first_plan.deinit();
    try first_plan.selectFirstColumns(2);
    const first_explain = try first_plan.explain(gpa);
    defer gpa.free(first_explain);
    try std.testing.expect(std.mem.indexOf(u8, first_explain, "select_column_range(0..2)") != null);
    var first = try first_plan.collect();
    defer first.deinit();
    try std.testing.expectEqual(@as(?usize, 0), first.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), first.columnIndex("units"));

    var last_plan = try DeviceLazyFrame.init(gpa, table);
    defer last_plan.deinit();
    try last_plan.selectLastColumns(2);
    const last_explain = try last_plan.explain(gpa);
    defer gpa.free(last_explain);
    try std.testing.expect(std.mem.indexOf(u8, last_explain, "select_last_columns(2)") != null);
    var last = try last_plan.collect();
    defer last.deinit();
    try std.testing.expectEqual(@as(?usize, 0), last.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), last.columnIndex("active"));

    var drop_positional_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_positional_plan.deinit();
    try drop_positional_plan.dropByColumnIndices(&.{1});
    const drop_positional_explain = try drop_positional_plan.explain(gpa);
    defer gpa.free(drop_positional_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_positional_explain, "drop_column_indices([1])") != null);
    var drop_positional = try drop_positional_plan.collect();
    defer drop_positional.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_positional.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_positional.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_positional.columnIndex("active"));

    var drop_range_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_range_plan.deinit();
    try drop_range_plan.dropColumnRange(1, 3);
    const drop_range_explain = try drop_range_plan.explain(gpa);
    defer gpa.free(drop_range_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_range_explain, "drop_column_range(1..3)") != null);
    var drop_range = try drop_range_plan.collect();
    defer drop_range.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_range.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_range.columnIndex("sales"));

    var drop_first_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_first_plan.deinit();
    try drop_first_plan.dropFirstColumns(1);
    const drop_first_explain = try drop_first_plan.explain(gpa);
    defer gpa.free(drop_first_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_first_explain, "drop_column_range(0..1)") != null);
    var drop_first = try drop_first_plan.collect();
    defer drop_first.deinit();
    try std.testing.expectEqual(@as(?usize, 0), drop_first.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), drop_first.columnIndex("active"));

    var drop_last_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_last_plan.deinit();
    try drop_last_plan.dropLastColumns(1);
    const drop_last_explain = try drop_last_plan.explain(gpa);
    defer gpa.free(drop_last_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_last_explain, "drop_last_columns(1)") != null);
    var drop_last = try drop_last_plan.collect();
    defer drop_last.deinit();
    try std.testing.expectEqual(@as(?usize, 0), drop_last.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_last.columnIndex("units"));

    var select_except_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_except_plan.deinit();
    try select_except_plan.selectExcept(&.{"units"});
    const select_except_explain = try select_except_plan.explain(gpa);
    defer gpa.free(select_except_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_except_explain, "drop_columns[units]") != null);
    var select_except = try select_except_plan.collect();
    defer select_except.deinit();
    try std.testing.expectEqual(@as(usize, 2), select_except.width());
    try std.testing.expectEqual(@as(?usize, 0), select_except.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), select_except.columnIndex("active"));

    var reverse_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer reverse_columns_plan.deinit();
    try reverse_columns_plan.reverseColumns();
    const reverse_columns_explain = try reverse_columns_plan.explain(gpa);
    defer gpa.free(reverse_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, reverse_columns_explain, "reverse_columns") != null);
    var reversed_columns = try reverse_columns_plan.collect();
    defer reversed_columns.deinit();
    try std.testing.expectEqual(@as(?usize, 0), reversed_columns.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), reversed_columns.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 2), reversed_columns.columnIndex("sales"));

    var sort_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer sort_columns_plan.deinit();
    try sort_columns_plan.sortColumnsByName(false);
    const sort_columns_explain = try sort_columns_plan.explain(gpa);
    defer gpa.free(sort_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, sort_columns_explain, "sort_columns_by_name(desc=false)") != null);
    var sorted_columns = try sort_columns_plan.collect();
    defer sorted_columns.deinit();
    try std.testing.expectEqual(@as(?usize, 0), sorted_columns.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), sorted_columns.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), sorted_columns.columnIndex("units"));

    var sort_columns_desc_plan = try DeviceLazyFrame.init(gpa, table);
    defer sort_columns_desc_plan.deinit();
    try sort_columns_desc_plan.sortColumnsByName(true);
    var sorted_columns_desc = try sort_columns_desc_plan.collect();
    defer sorted_columns_desc.deinit();
    try std.testing.expectEqual(@as(?usize, 0), sorted_columns_desc.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), sorted_columns_desc.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), sorted_columns_desc.columnIndex("active"));

    var invalid_positional_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_positional_plan.deinit();
    try invalid_positional_plan.selectByColumnIndices(&.{3});
    try std.testing.expectError(error.IndexOutOfBounds, invalid_positional_plan.collect());

    var numeric_plan = try DeviceLazyFrame.init(gpa, table);
    defer numeric_plan.deinit();
    try numeric_plan.selectNumeric();

    const numeric_explain = try numeric_plan.explain(gpa);
    defer gpa.free(numeric_explain);
    try std.testing.expect(std.mem.indexOf(u8, numeric_explain, "select_dtype_class(numeric)") != null);

    var numeric = try numeric_plan.collect();
    defer numeric.deinit();
    try std.testing.expectEqual(@as(usize, 2), numeric.width());
    try std.testing.expectEqual(@as(?usize, 0), numeric.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), numeric.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, null), numeric.columnIndex("active"));

    var exact_plan = try DeviceLazyFrame.init(gpa, table);
    defer exact_plan.deinit();
    try exact_plan.selectByDTypes(&.{ .bool, .f64 });
    const exact_explain = try exact_plan.explain(gpa);
    defer gpa.free(exact_explain);
    try std.testing.expect(std.mem.indexOf(u8, exact_explain, "select_dtypes[bool,f64]") != null);

    var exact = try exact_plan.collect();
    defer exact.deinit();
    try std.testing.expectEqual(@as(usize, 2), exact.width());
    try std.testing.expectEqual(@as(?usize, 0), exact.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), exact.columnIndex("active"));

    var empty_plan = try DeviceLazyFrame.init(gpa, table);
    defer empty_plan.deinit();
    try empty_plan.selectFloat();
    try empty_plan.selectInteger();
    var empty = try empty_plan.collect();
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty.width());
    try std.testing.expectEqual(table.height(), empty.height());

    var drop_numeric_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_numeric_plan.deinit();
    try drop_numeric_plan.dropNumeric();
    const drop_numeric_explain = try drop_numeric_plan.explain(gpa);
    defer gpa.free(drop_numeric_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_numeric_explain, "drop_dtype_class(numeric)") != null);

    var non_numeric = try drop_numeric_plan.collect();
    defer non_numeric.deinit();
    try std.testing.expectEqual(@as(usize, 1), non_numeric.width());
    try std.testing.expectEqual(@as(?usize, 0), non_numeric.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), non_numeric.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), non_numeric.columnIndex("units"));

    var drop_exact_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_exact_plan.deinit();
    try drop_exact_plan.dropByDTypes(&.{ .bool, .f64 });
    const drop_exact_explain = try drop_exact_plan.explain(gpa);
    defer gpa.free(drop_exact_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_exact_explain, "drop_dtypes[bool,f64]") != null);

    var drop_exact = try drop_exact_plan.collect();
    defer drop_exact.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_exact.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_exact.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, null), drop_exact.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), drop_exact.columnIndex("active"));

    var drop_all_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_all_plan.deinit();
    try drop_all_plan.dropByDTypes(&.{ .f64, .i64, .bool });
    var drop_all = try drop_all_plan.collect();
    defer drop_all.deinit();
    try std.testing.expectEqual(@as(usize, 0), drop_all.width());
    try std.testing.expectEqual(table.height(), drop_all.height());
}

test "device lazy frame selects and drops columns by nullability" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0 }, .cpu);
    defer sales.deinit();
    var audited_units = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3 }, &.{ true, true, true }, .cpu);
    defer audited_units.deinit();
    var quality = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.8, 0.0, 0.9 }, &.{ true, false, true }, .cpu);
    defer quality.deinit();
    var active = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer active.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "audited_units", .data = audited_units },
        .{ .name = "quality", .data = quality },
        .{ .name = "active", .data = active },
    });
    defer table.deinit();

    var nullable_plan = try DeviceLazyFrame.init(gpa, table);
    defer nullable_plan.deinit();
    try nullable_plan.selectNullableColumns();
    const nullable_explain = try nullable_plan.explain(gpa);
    defer gpa.free(nullable_explain);
    try std.testing.expect(std.mem.indexOf(u8, nullable_explain, "select_nullable_columns") != null);
    var nullable = try nullable_plan.collect();
    defer nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), nullable.columnIndex("audited_units"));
    try std.testing.expectEqual(@as(?usize, 1), nullable.columnIndex("quality"));

    var non_nullable_plan = try DeviceLazyFrame.init(gpa, table);
    defer non_nullable_plan.deinit();
    try non_nullable_plan.selectNonNullableColumns();
    const non_nullable_explain = try non_nullable_plan.explain(gpa);
    defer gpa.free(non_nullable_explain);
    try std.testing.expect(std.mem.indexOf(u8, non_nullable_explain, "select_non_nullable_columns") != null);
    var non_nullable = try non_nullable_plan.collect();
    defer non_nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), non_nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), non_nullable.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), non_nullable.columnIndex("active"));

    var with_nulls_plan = try DeviceLazyFrame.init(gpa, table);
    defer with_nulls_plan.deinit();
    try with_nulls_plan.selectColumnsWithNulls();
    const with_nulls_explain = try with_nulls_plan.explain(gpa);
    defer gpa.free(with_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, with_nulls_explain, "select_columns_with_nulls") != null);
    var with_nulls = try with_nulls_plan.collect();
    defer with_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 1), with_nulls.width());
    try std.testing.expectEqual(@as(?usize, 0), with_nulls.columnIndex("quality"));
    const quality_values = try (try with_nulls.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality_values);
    try std.testing.expectEqualSlices(f64, &.{ 0.8, 0.0, 0.9 }, quality_values);

    var without_nulls_plan = try DeviceLazyFrame.init(gpa, table);
    defer without_nulls_plan.deinit();
    try without_nulls_plan.selectColumnsWithoutNulls();
    const without_nulls_explain = try without_nulls_plan.explain(gpa);
    defer gpa.free(without_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, without_nulls_explain, "select_columns_without_nulls") != null);
    var without_nulls = try without_nulls_plan.collect();
    defer without_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 3), without_nulls.width());
    try std.testing.expectEqual(@as(?usize, 0), without_nulls.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), without_nulls.columnIndex("audited_units"));
    try std.testing.expectEqual(@as(?usize, 2), without_nulls.columnIndex("active"));

    var drop_nullable_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_nullable_plan.deinit();
    try drop_nullable_plan.dropNullableColumns();
    const drop_nullable_explain = try drop_nullable_plan.explain(gpa);
    defer gpa.free(drop_nullable_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_nullable_explain, "drop_nullable_columns") != null);
    var drop_nullable = try drop_nullable_plan.collect();
    defer drop_nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_nullable.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_nullable.columnIndex("active"));

    var drop_non_nullable_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_nullable_plan.deinit();
    try drop_non_nullable_plan.dropNonNullableColumns();
    const drop_non_nullable_explain = try drop_non_nullable_plan.explain(gpa);
    defer gpa.free(drop_non_nullable_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_nullable_explain, "drop_non_nullable_columns") != null);
    var drop_non_nullable = try drop_non_nullable_plan.collect();
    defer drop_non_nullable.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_non_nullable.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_nullable.columnIndex("audited_units"));
    try std.testing.expectEqual(@as(?usize, 1), drop_non_nullable.columnIndex("quality"));

    var drop_with_nulls_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_with_nulls_plan.deinit();
    try drop_with_nulls_plan.dropColumnsWithNulls();
    const drop_with_nulls_explain = try drop_with_nulls_plan.explain(gpa);
    defer gpa.free(drop_with_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_with_nulls_explain, "drop_columns_with_nulls") != null);
    var drop_with_nulls = try drop_with_nulls_plan.collect();
    defer drop_with_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 3), drop_with_nulls.width());
    try std.testing.expectEqual(@as(?usize, null), drop_with_nulls.columnIndex("quality"));

    var drop_without_nulls_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_without_nulls_plan.deinit();
    try drop_without_nulls_plan.dropColumnsWithoutNulls();
    const drop_without_nulls_explain = try drop_without_nulls_plan.explain(gpa);
    defer gpa.free(drop_without_nulls_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_without_nulls_explain, "drop_columns_without_nulls") != null);
    var drop_without_nulls = try drop_without_nulls_plan.collect();
    defer drop_without_nulls.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_without_nulls.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_without_nulls.columnIndex("quality"));
}

test "device lazy frame selects and drops columns by name pattern" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer prefix_plan.deinit();
    try prefix_plan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try prefix_plan.selectByNamePrefix("sales");

    const prefix_explain = try prefix_plan.explain(gpa);
    defer gpa.free(prefix_explain);
    try std.testing.expect(std.mem.indexOf(u8, prefix_explain, "select_name_prefix(sales)") != null);

    var prefixed = try prefix_plan.collect();
    defer prefixed.deinit();
    try std.testing.expectEqual(@as(usize, 2), prefixed.width());
    try std.testing.expectEqual(@as(?usize, 0), prefixed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), prefixed.columnIndex("sales_x2"));
    const sales_x2 = try (try prefixed.column("sales_x2")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_x2);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 6.0, 10.0, 14.0 }, sales_x2);

    var suffix_plan = try DeviceLazyFrame.init(gpa, table);
    defer suffix_plan.deinit();
    try suffix_plan.selectByNameSuffix("s");

    const suffix_explain = try suffix_plan.explain(gpa);
    defer gpa.free(suffix_explain);
    try std.testing.expect(std.mem.indexOf(u8, suffix_explain, "select_name_suffix(s)") != null);

    var suffixed = try suffix_plan.collect();
    defer suffixed.deinit();
    try std.testing.expectEqual(@as(usize, 2), suffixed.width());
    try std.testing.expectEqual(@as(?usize, 0), suffixed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), suffixed.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, null), suffixed.columnIndex("active"));

    var contains_plan = try DeviceLazyFrame.init(gpa, table);
    defer contains_plan.deinit();
    try contains_plan.selectByNameContains("ct");

    const contains_explain = try contains_plan.explain(gpa);
    defer gpa.free(contains_explain);
    try std.testing.expect(std.mem.indexOf(u8, contains_explain, "select_name_contains(ct)") != null);

    var contained = try contains_plan.collect();
    defer contained.deinit();
    try std.testing.expectEqual(@as(usize, 1), contained.width());
    try std.testing.expectEqual(@as(?usize, 0), contained.columnIndex("active"));
    const active = try (try contained.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(active);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, active);

    var glob_plan = try DeviceLazyFrame.init(gpa, table);
    defer glob_plan.deinit();
    try glob_plan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try glob_plan.selectByNameGlob("sales*");

    const glob_explain = try glob_plan.explain(gpa);
    defer gpa.free(glob_explain);
    try std.testing.expect(std.mem.indexOf(u8, glob_explain, "select_name_glob(sales*)") != null);

    var globbed = try glob_plan.collect();
    defer globbed.deinit();
    try std.testing.expectEqual(@as(usize, 2), globbed.width());
    try std.testing.expectEqual(@as(?usize, 0), globbed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), globbed.columnIndex("sales_x2"));

    var empty_plan = try DeviceLazyFrame.init(gpa, table);
    defer empty_plan.deinit();
    try empty_plan.selectByNameContains("missing");
    var empty = try empty_plan.collect();
    defer empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty.width());
    try std.testing.expectEqual(table.height(), empty.height());

    var drop_prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_prefix_plan.deinit();
    try drop_prefix_plan.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try drop_prefix_plan.dropByNamePrefix("sales");

    const drop_prefix_explain = try drop_prefix_plan.explain(gpa);
    defer gpa.free(drop_prefix_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_prefix_explain, "drop_name_prefix(sales)") != null);

    var drop_prefixed = try drop_prefix_plan.collect();
    defer drop_prefixed.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_prefixed.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_prefixed.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), drop_prefixed.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), drop_prefixed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), drop_prefixed.columnIndex("sales_x2"));

    var drop_suffix_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_suffix_plan.deinit();
    try drop_suffix_plan.dropByNameSuffix("s");

    const drop_suffix_explain = try drop_suffix_plan.explain(gpa);
    defer gpa.free(drop_suffix_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_suffix_explain, "drop_name_suffix(s)") != null);

    var drop_suffixed = try drop_suffix_plan.collect();
    defer drop_suffixed.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_suffixed.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_suffixed.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), drop_suffixed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), drop_suffixed.columnIndex("units"));

    var drop_contains_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_contains_plan.deinit();
    try drop_contains_plan.dropByNameContains("ct");

    const drop_contains_explain = try drop_contains_plan.explain(gpa);
    defer gpa.free(drop_contains_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_contains_explain, "drop_name_contains(ct)") != null);

    var drop_contained = try drop_contains_plan.collect();
    defer drop_contained.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_contained.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_contained.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), drop_contained.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, null), drop_contained.columnIndex("active"));

    var drop_glob_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_glob_plan.deinit();
    try drop_glob_plan.dropByNameGlob("*s");

    const drop_glob_explain = try drop_glob_plan.explain(gpa);
    defer gpa.free(drop_glob_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_glob_explain, "drop_name_glob(*s)") != null);

    var drop_globbed = try drop_glob_plan.collect();
    defer drop_globbed.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_globbed.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_globbed.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, null), drop_globbed.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), drop_globbed.columnIndex("units"));

    var drop_empty_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_empty_plan.deinit();
    try drop_empty_plan.dropByNamePrefix("");
    var drop_empty = try drop_empty_plan.collect();
    defer drop_empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), drop_empty.width());
    try std.testing.expectEqual(table.height(), drop_empty.height());
}

test "device lazy frame casts columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.castColumn("units", .f64);
    try plan.castColumn("active", .i8);
    try plan.select(&.{ "units", "active" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "cast_column(units->f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "cast_column(active->i8)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.width());
    try std.testing.expectEqual(vectra.DeviceDType.f64, try result.columnDType("units"));
    try std.testing.expectEqual(vectra.DeviceDType.i8, try result.columnDType("active"));
    const units = try (try result.column("units")).f64.toOwnedSlice(gpa);
    defer gpa.free(units);
    const active = try (try result.column("active")).i8.toOwnedSlice(gpa);
    defer gpa.free(active);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0, 4.0 }, units);
    try std.testing.expectEqualSlices(i8, &.{ 1, 0, 1, 1 }, active);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.castColumn("missing", .f64);
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame fills nullable columns" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnFillNull("quality_filled_copy", "quality", f64, -2.0);
    try plan.withColumnFillNullForward("quality_ffill", "quality");
    try plan.withColumnFillNullBackward("quality_bfill", "quality");
    try plan.withColumnNullIf("quality_without_three", "quality", f64, 3.0);
    try plan.withColumnNullIfValues("quality_without_values", "quality", f64, &.{ 1.0, 4.0 });
    try plan.fillNullColumn("quality", f64, -1.0);

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "fill_null_column(quality=scalar:f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "copy_column(quality->quality_filled_copy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "fill_null_column(quality_filled_copy=scalar:f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "fill_null_forward_column(quality_ffill)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "fill_null_backward_column(quality_bfill)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "copy_column(quality->quality_without_three)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "null_if_column(quality_without_three=scalar:f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "null_if_values_column(quality_without_values, values_dtype=f64, values_len=2)") != null);

    var filled = try plan.collect();
    defer filled.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try filled.column("quality")).nullCount());
    const quality = try (try filled.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, -1.0, 3.0, 4.0 }, quality);
    try std.testing.expectEqual(@as(usize, 0), (try filled.column("quality_filled_copy")).nullCount());
    const quality_filled_copy = try (try filled.column("quality_filled_copy")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality_filled_copy);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, -2.0, 3.0, 4.0 }, quality_filled_copy);
    const quality_ffill = try (try filled.column("quality_ffill")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality_ffill);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 3.0, 4.0 }, quality_ffill);
    const quality_bfill = try (try filled.column("quality_bfill")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality_bfill);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 3.0, 3.0, 4.0 }, quality_bfill);
    const quality_without_three = try (try filled.column("quality_without_three")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality_without_three);
    const quality_without_three_validity = try (try filled.column("quality_without_three")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(quality_without_three_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0, 4.0 }, quality_without_three);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, quality_without_three_validity);
    const quality_without_values_validity = try (try filled.column("quality_without_values")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(quality_without_values_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, quality_without_values_validity);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.fillNullColumn("quality", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, invalid_plan.collect());
}

test "device lazy frame coalesces nullable columns" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnLiteral("fallback_quality", f64, 9.0);
    try plan.coalesceColumns("quality", "fallback_quality", "quality_coalesced");
    try plan.withColumnLiteral("secondary_quality", f64, 10.0);
    try plan.coalesceColumnsMany(&.{ "quality", "fallback_quality", "secondary_quality" }, "quality_coalesced_many");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal(fallback_quality=scalar:f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "coalesce_columns(quality,fallback_quality->quality_coalesced)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "coalesce_columns_many([quality,fallback_quality,secondary_quality]->quality_coalesced_many)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try result.column("quality_coalesced")).nullCount());
    const values = try (try result.column("quality_coalesced")).f64.toOwnedSlice(gpa);
    defer gpa.free(values);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 9.0, 3.0, 4.0 }, values);
    try std.testing.expectEqual(@as(usize, 0), (try result.column("quality_coalesced_many")).nullCount());
    const many_values = try (try result.column("quality_coalesced_many")).f64.toOwnedSlice(gpa);
    defer gpa.free(many_values);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 9.0, 3.0, 4.0 }, many_values);

    var mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer mismatch_plan.deinit();
    try mismatch_plan.withColumnLiteral("fallback_i64", i64, 9);
    try mismatch_plan.coalesceColumnsMany(&.{ "quality", "fallback_i64" }, "bad");
    try std.testing.expectError(error.TypeMismatch, mismatch_plan.collect());

    var empty_plan = try DeviceLazyFrame.init(gpa, table);
    defer empty_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, empty_plan.coalesceFirstValidColumns(&.{}, "bad_empty"));
}

test "device lazy frame derives null predicate columns" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.isNullColumn("quality", "quality_is_null");
    try plan.isValidColumn("quality", "quality_is_valid");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_null_column(quality->quality_is_null)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_valid_column(quality->quality_is_valid)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 3), result.width());
    const is_null = try (try result.column("quality_is_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(is_null);
    const is_valid = try (try result.column("quality_is_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(is_valid);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, is_null);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, is_valid);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isNullColumn("missing", "missing_is_null");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame derives row null and valid count columns" {
    const gpa = std.testing.allocator;

    var sales = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, 5.0, 7.0 }, .cpu);
    defer sales.deinit();
    var quality = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.8, 0.0, 0.9, 1.0 }, &.{ true, false, true, true }, .cpu);
    defer quality.deinit();
    var flag = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, true, false }, &.{ true, true, false, false }, .cpu);
    defer flag.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = sales },
        .{ .name = "quality", .data = quality },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowValidCount(&.{}, "row_valids_all");
    try plan.withRowNullCount(&.{ "quality", "flag" }, "row_nulls");
    try plan.withRowAnyNull(&.{ "sales", "quality", "flag" }, "row_any_null");
    try plan.withRowAllValid(&.{ "sales", "quality", "flag" }, "row_all_valid");
    try plan.withRowAnyValid(&.{"flag"}, "row_any_valid_flag");
    try plan.withRowAllNull(&.{"flag"}, "row_all_null_flag");
    try plan.withRowCumulativeValidCount(&.{ "sales", "quality", "flag" }, &.{ "sales_cum_valid", "quality_cum_valid", "flag_cum_valid" });
    try plan.withRowCumulativeNullCount(&.{ "sales", "quality", "flag" }, &.{ "sales_cum_null", "quality_cum_null", "flag_cum_null" });
    try plan.withRowCumulativeAnyNull(&.{ "sales", "quality", "flag" }, &.{ "sales_cum_any_null", "quality_cum_any_null", "flag_cum_any_null" });
    try plan.withRowPrefixAllValid(&.{ "sales", "quality", "flag" }, &.{ "sales_prefix_all_valid", "quality_prefix_all_valid", "flag_prefix_all_valid" });
    try plan.withRowCumulativeValidRatio(&.{ "sales", "quality", "flag" }, &.{ "sales_cum_valid_ratio", "quality_cum_valid_ratio", "flag_cum_valid_ratio" });
    try plan.withRowCumulativeNullRatio(&.{ "sales", "quality", "flag" }, &.{ "sales_cum_null_ratio", "quality_cum_null_ratio", "flag_cum_null_ratio" });
    try plan.withRowCumulativeFirstValidIndex(&.{ "sales", "quality", "flag" }, &.{ "sales_first_valid", "quality_first_valid", "flag_first_valid" });
    try plan.withRowCumulativeLastNullIndex(&.{ "sales", "quality", "flag" }, &.{ "sales_last_null", "quality_last_null", "flag_last_null" });
    try plan.withRowNullRatio(&.{ "quality", "flag" }, "row_null_ratio");
    try plan.withRowValidRatio(&.{ "quality", "flag" }, "row_valid_ratio");
    try plan.withRowTrueCount(&.{"flag"}, "row_true_count");
    try plan.withRowFalseCount(&.{"flag"}, "row_false_count");
    try plan.withRowCumulativeTrueCount(&.{"flag"}, &.{"flag_cum_true"});
    try plan.withRowCumulativeFalseCount(&.{"flag"}, &.{"flag_cum_false"});
    try plan.withRowCumulativeTrueRatio(&.{"flag"}, &.{"flag_cum_true_ratio"});
    try plan.withRowCumulativeFalseRatio(&.{"flag"}, &.{"flag_cum_false_ratio"});
    try plan.withRowCumulativeAnyTrue(&.{"flag"}, &.{"flag_cum_any_true"});
    try plan.withRowCumulativeAllTrue(&.{"flag"}, &.{"flag_cum_all_true"});
    try plan.withRowCumulativeFirstTrueIndex(&.{"flag"}, &.{"flag_first_true"});
    try plan.withRowCumulativeLastFalseIndex(&.{"flag"}, &.{"flag_last_false"});
    try plan.withRowAnyTrue(&.{"flag"}, "row_any_true");
    try plan.withRowAllTrue(&.{"flag"}, "row_all_true");
    try plan.withRowAnyFalse(&.{"flag"}, "row_any_false");
    try plan.withRowAllFalse(&.{"flag"}, "row_all_false");
    try plan.withRowTrueRatio(&.{"flag"}, "row_true_ratio");
    try plan.withRowFalseRatio(&.{"flag"}, "row_false_ratio");
    try plan.select(&.{ "row_nulls", "row_valids_all", "row_any_null", "row_all_valid", "row_any_valid_flag", "row_all_null_flag", "sales_cum_valid", "quality_cum_valid", "flag_cum_valid", "sales_cum_null", "quality_cum_null", "flag_cum_null", "sales_cum_any_null", "quality_cum_any_null", "flag_cum_any_null", "sales_prefix_all_valid", "quality_prefix_all_valid", "flag_prefix_all_valid", "sales_cum_valid_ratio", "quality_cum_valid_ratio", "flag_cum_valid_ratio", "sales_cum_null_ratio", "quality_cum_null_ratio", "flag_cum_null_ratio", "sales_first_valid", "quality_first_valid", "flag_first_valid", "sales_last_null", "quality_last_null", "flag_last_null", "row_null_ratio", "row_valid_ratio", "row_true_count", "row_false_count", "flag_cum_true", "flag_cum_false", "flag_cum_true_ratio", "flag_cum_false_ratio", "flag_cum_any_true", "flag_cum_all_true", "flag_first_true", "flag_last_false", "row_any_true", "row_all_true", "row_any_false", "row_all_false", "row_true_ratio", "row_false_ratio" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_null_count([quality,flag]->row_nulls)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_valid_count([]->row_valids_all)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_null([sales,quality,flag]->row_any_null)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_all_valid([sales,quality,flag]->row_all_valid)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_valid([flag]->row_any_valid_flag)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_all_null([flag]->row_all_null_flag)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_valid_count([sales,quality,flag]->[sales_cum_valid,quality_cum_valid,flag_cum_valid])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_null_count([sales,quality,flag]->[sales_cum_null,quality_cum_null,flag_cum_null])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_any_null([sales,quality,flag]->[sales_cum_any_null,quality_cum_any_null,flag_cum_any_null])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_all_valid([sales,quality,flag]->[sales_prefix_all_valid,quality_prefix_all_valid,flag_prefix_all_valid])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_valid_ratio([sales,quality,flag]->[sales_cum_valid_ratio,quality_cum_valid_ratio,flag_cum_valid_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_null_ratio([sales,quality,flag]->[sales_cum_null_ratio,quality_cum_null_ratio,flag_cum_null_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_first_valid_index([sales,quality,flag]->[sales_first_valid,quality_first_valid,flag_first_valid])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_last_null_index([sales,quality,flag]->[sales_last_null,quality_last_null,flag_last_null])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_null_ratio([quality,flag]->row_null_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_valid_ratio([quality,flag]->row_valid_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_true_count([flag]->row_true_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_false_count([flag]->row_false_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_true_count([flag]->[flag_cum_true])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_false_count([flag]->[flag_cum_false])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_true_ratio([flag]->[flag_cum_true_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_false_ratio([flag]->[flag_cum_false_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_any_true([flag]->[flag_cum_any_true])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_all_true([flag]->[flag_cum_all_true])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_first_true_index([flag]->[flag_first_true])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_last_false_index([flag]->[flag_last_false])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_true([flag]->row_any_true)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_all_true([flag]->row_all_true)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_false([flag]->row_any_false)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_all_false([flag]->row_all_false)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_true_ratio([flag]->row_true_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_false_ratio([flag]->row_false_ratio)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 48), result.width());
    const row_nulls = try (try result.column("row_nulls")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_nulls);
    const row_valids_all = try (try result.column("row_valids_all")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_valids_all);
    const row_any_null = try (try result.column("row_any_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_null);
    const row_all_valid = try (try result.column("row_all_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_valid);
    const row_any_valid_flag = try (try result.column("row_any_valid_flag")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_valid_flag);
    const row_all_null_flag = try (try result.column("row_all_null_flag")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_null_flag);
    const sales_cum_valid = try (try result.column("sales_cum_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_valid);
    const quality_cum_valid = try (try result.column("quality_cum_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(quality_cum_valid);
    const flag_cum_valid = try (try result.column("flag_cum_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_valid);
    const sales_cum_null = try (try result.column("sales_cum_null")).i64.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_null);
    const quality_cum_null = try (try result.column("quality_cum_null")).i64.toOwnedSlice(gpa);
    defer gpa.free(quality_cum_null);
    const flag_cum_null = try (try result.column("flag_cum_null")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_null);
    const sales_cum_any_null = try (try result.column("sales_cum_any_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_any_null);
    const quality_cum_any_null = try (try result.column("quality_cum_any_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(quality_cum_any_null);
    const flag_cum_any_null = try (try result.column("flag_cum_any_null")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_any_null);
    const sales_prefix_all_valid = try (try result.column("sales_prefix_all_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(sales_prefix_all_valid);
    const quality_prefix_all_valid = try (try result.column("quality_prefix_all_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(quality_prefix_all_valid);
    const flag_prefix_all_valid = try (try result.column("flag_prefix_all_valid")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_all_valid);
    const sales_cum_valid_ratio = try (try result.column("sales_cum_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_valid_ratio);
    const quality_cum_valid_ratio = try (try result.column("quality_cum_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality_cum_valid_ratio);
    const flag_cum_valid_ratio = try (try result.column("flag_cum_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_valid_ratio);
    const sales_cum_null_ratio = try (try result.column("sales_cum_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales_cum_null_ratio);
    const quality_cum_null_ratio = try (try result.column("quality_cum_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality_cum_null_ratio);
    const flag_cum_null_ratio = try (try result.column("flag_cum_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_null_ratio);
    const sales_first_valid = try (try result.column("sales_first_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(sales_first_valid);
    const quality_first_valid = try (try result.column("quality_first_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(quality_first_valid);
    const flag_first_valid = try (try result.column("flag_first_valid")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_first_valid);
    const sales_last_null_column = try result.column("sales_last_null");
    const sales_last_null = try sales_last_null_column.i64.toOwnedSlice(gpa);
    defer gpa.free(sales_last_null);
    const sales_last_null_validity = try sales_last_null_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(sales_last_null_validity);
    const quality_last_null = try (try result.column("quality_last_null")).i64.toOwnedSlice(gpa);
    defer gpa.free(quality_last_null);
    const flag_last_null = try (try result.column("flag_last_null")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_last_null);
    const row_null_ratio = try (try result.column("row_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_null_ratio);
    const row_valid_ratio = try (try result.column("row_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_valid_ratio);
    const row_true_count = try (try result.column("row_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_true_count);
    const row_false_count = try (try result.column("row_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_false_count);
    const flag_cum_true = try (try result.column("flag_cum_true")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_true);
    const flag_cum_false = try (try result.column("flag_cum_false")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_false);
    const flag_cum_true_ratio = try (try result.column("flag_cum_true_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_true_ratio);
    const flag_cum_false_ratio = try (try result.column("flag_cum_false_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_false_ratio);
    const flag_cum_any_true_column = try result.column("flag_cum_any_true");
    try std.testing.expect(flag_cum_any_true_column.bool.nullable());
    const flag_cum_any_true = try flag_cum_any_true_column.bool.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_any_true);
    const flag_cum_any_true_validity = try flag_cum_any_true_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_any_true_validity);
    const flag_cum_all_true = try (try result.column("flag_cum_all_true")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_all_true);
    const flag_first_true = try (try result.column("flag_first_true")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_first_true);
    const flag_last_false = try (try result.column("flag_last_false")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_last_false);
    const row_any_true_column = try result.column("row_any_true");
    try std.testing.expect(row_any_true_column.bool.nullable());
    const row_any_true = try row_any_true_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_true);
    const row_any_true_validity = try row_any_true_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_any_true_validity);
    const row_all_true_column = try result.column("row_all_true");
    try std.testing.expect(row_all_true_column.bool.nullable());
    const row_all_true = try row_all_true_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_true);
    const row_all_true_validity = try row_all_true_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_all_true_validity);
    const row_any_false_column = try result.column("row_any_false");
    try std.testing.expect(row_any_false_column.bool.nullable());
    const row_any_false = try row_any_false_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_false);
    const row_any_false_validity = try row_any_false_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_any_false_validity);
    const row_all_false_column = try result.column("row_all_false");
    try std.testing.expect(row_all_false_column.bool.nullable());
    const row_all_false = try row_all_false_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_false);
    const row_all_false_validity = try row_all_false_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_all_false_validity);
    const row_true_ratio_column = try result.column("row_true_ratio");
    try std.testing.expect(row_true_ratio_column.f64.nullable());
    const row_true_ratio = try row_true_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_true_ratio);
    const row_true_ratio_validity = try row_true_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_true_ratio_validity);
    const row_false_ratio_column = try result.column("row_false_ratio");
    try std.testing.expect(row_false_ratio_column.f64.nullable());
    const row_false_ratio = try row_false_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_false_ratio);
    const row_false_ratio_validity = try row_false_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_false_ratio_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, row_nulls);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2, 2, 2 }, row_valids_all);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, row_any_null);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, row_all_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, row_any_valid_flag);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, row_all_null_flag);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 1 }, sales_cum_valid);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 2, 2 }, quality_cum_valid);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2, 2, 2 }, flag_cum_valid);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, sales_cum_null);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, quality_cum_null);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, flag_cum_null);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, sales_cum_any_null);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, quality_cum_any_null);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, flag_cum_any_null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, sales_prefix_all_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, quality_prefix_all_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, flag_prefix_all_valid);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 1.0 }, sales_cum_valid_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.5, 1.0, 1.0 }, quality_cum_valid_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0 }, flag_cum_valid_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, sales_cum_null_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.0, 0.0 }, quality_cum_null_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 }, flag_cum_null_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, sales_first_valid);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, quality_first_valid);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, flag_first_valid);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, sales_last_null);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, sales_last_null_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, quality_last_null);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2 }, flag_last_null);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.5, 0.5 }, row_null_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.5, 0.5, 0.5 }, row_valid_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, row_true_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, row_false_count);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, flag_cum_true);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, flag_cum_false);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 0.0 }, flag_cum_true_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0, 0.0 }, flag_cum_false_ratio);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, flag_cum_any_true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, flag_cum_any_true_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, flag_cum_all_true);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, flag_first_true);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, flag_last_false);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, row_any_true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, row_any_true_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, row_all_true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, row_all_true_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, row_any_false);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, row_any_false_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, row_all_false);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, row_all_false_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 0.0 }, row_true_ratio);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, row_true_ratio_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0, 0.0 }, row_false_ratio);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false }, row_false_ratio_validity);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowNullCount(&.{ "quality", "missing" }, "bad_count");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());

    var invalid_cumulative_count_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_count_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_count_plan.withRowPrefixValidCount(&.{"sales"}, &.{ "sales_cum_valid", "extra_cum_valid" }));

    var invalid_cumulative_reduction_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_reduction_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_reduction_plan.withRowPrefixAnyNull(&.{"sales"}, &.{ "sales_any_null", "extra_any_null" }));

    var invalid_cumulative_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_index_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_index_plan.withRowPrefixFirstValidIndex(&.{"sales"}, &.{ "sales_first_valid", "extra_first_valid" }));

    var invalid_cumulative_ratio_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_ratio_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_ratio_plan.withRowPrefixValidRatio(&.{"sales"}, &.{ "sales_cum_valid_ratio", "extra_cum_valid_ratio" }));

    var invalid_bool_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_bool_plan.deinit();
    try invalid_bool_plan.withRowTrueCount(&.{"sales"}, "bad_bool_count");
    try std.testing.expectError(error.TypeMismatch, invalid_bool_plan.collect());

    var invalid_cumulative_bool_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_bool_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_bool_plan.withRowPrefixTrueCount(&.{"flag"}, &.{ "flag_cum_true", "extra_cum_true" }));

    var invalid_cumulative_bool_ratio_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_bool_ratio_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_bool_ratio_plan.withRowPrefixTrueRatio(&.{"flag"}, &.{ "flag_cum_true_ratio", "extra_cum_true_ratio" }));

    var invalid_cumulative_bool_reduction_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_bool_reduction_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_bool_reduction_plan.withRowPrefixAnyTrue(&.{"flag"}, &.{ "flag_cum_any_true", "extra_cum_any_true" }));

    var invalid_cumulative_bool_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_bool_index_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_bool_index_plan.withRowPrefixFirstTrueIndex(&.{"flag"}, &.{ "flag_first_true", "extra_first_true" }));

    var invalid_bool_reduction_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_bool_reduction_plan.deinit();
    try invalid_bool_reduction_plan.withRowAnyTrue(&.{"sales"}, "bad_bool_reduce");
    try std.testing.expectError(error.TypeMismatch, invalid_bool_reduction_plan.collect());

    var invalid_bool_ratio_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_bool_ratio_plan.deinit();
    try invalid_bool_ratio_plan.withRowTrueRatio(&.{"sales"}, "bad_bool_ratio");
    try std.testing.expectError(error.TypeMismatch, invalid_bool_ratio_plan.collect());
}

test "device lazy frame derives row validity match index columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var c = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, true, false }, &.{ false, false, true, true }, .cpu);
    defer c.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "c", .data = c },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowFirstValidIndex(&.{ "a", "b", "c" }, "first_valid");
    try plan.withRowLastValidIndex(&.{ "a", "b", "c" }, "last_valid");
    try plan.withRowFirstNullIndex(&.{ "a", "b", "c" }, "first_null");
    try plan.withRowLastNullIndex(&.{ "a", "b", "c" }, "last_null");
    try plan.select(&.{ "first_valid", "last_valid", "first_null", "last_null" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_first_valid_index([a,b,c]->first_valid)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_last_valid_index([a,b,c]->last_valid)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_first_null_index([a,b,c]->first_null)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_last_null_index([a,b,c]->last_null)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 4), result.width());
    const first_valid_column = try result.column("first_valid");
    try std.testing.expect(first_valid_column.i64.nullable());
    const first_valid = try first_valid_column.i64.toOwnedSlice(gpa);
    defer gpa.free(first_valid);
    const first_valid_validity = try first_valid_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(first_valid_validity);
    const last_valid_column = try result.column("last_valid");
    try std.testing.expect(last_valid_column.i64.nullable());
    const last_valid = try last_valid_column.i64.toOwnedSlice(gpa);
    defer gpa.free(last_valid);
    const last_valid_validity = try last_valid_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(last_valid_validity);
    const first_null_column = try result.column("first_null");
    try std.testing.expect(first_null_column.i64.nullable());
    const first_null = try first_null_column.i64.toOwnedSlice(gpa);
    defer gpa.free(first_null);
    const first_null_validity = try first_null_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(first_null_validity);
    const last_null_column = try result.column("last_null");
    try std.testing.expect(last_null_column.i64.nullable());
    const last_null = try last_null_column.i64.toOwnedSlice(gpa);
    defer gpa.free(last_null);
    const last_null_validity = try last_null_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(last_null_validity);

    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 0 }, first_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, first_valid_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 2 }, last_valid);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, last_valid_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, first_null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, first_null_validity);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 1, 0 }, last_null);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, last_null_validity);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowFirstNullIndex(&.{ "a", "missing" }, "bad_null_index");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame derives row numeric reduction columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true, false }, .cpu);
    defer flag.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "flag", .data = flag },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowArgMin(&.{ "a", "b" }, "row_argmin");
    try plan.withRowArgMax(&.{ "a", "b" }, "row_argmax");
    try plan.withRowCumulativeArgMin(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cum_argmin", "row_b_cum_argmin", "row_wa_cum_argmin", "row_wb_cum_argmin" });
    try plan.withRowCumulativeArgMax(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cum_argmax", "row_b_cum_argmax", "row_wa_cum_argmax", "row_wb_cum_argmax" });
    try plan.withRowQuantile(&.{ "a", "b" }, "row_quantile", 0.25);
    try plan.withRowQuantileRange(&.{ "a", "b" }, "row_quantile_range", 0.2, 0.8);
    try plan.withRowTrimmedMean(&.{ "a", "b" }, "row_trimmed_mean", 0.25);
    try plan.withRowWinsorizedMean(&.{ "a", "b" }, "row_winsorized_mean", 0.25);
    try plan.withRowMedian(&.{ "a", "b" }, "row_median");
    try plan.withRowIqr(&.{ "a", "b" }, "row_iqr");
    try plan.withRowInterdecileRange(&.{ "a", "b" }, "row_idr");
    try plan.withRowMidhinge(&.{ "a", "b" }, "row_midhinge");
    try plan.withRowTrimean(&.{ "a", "b" }, "row_trimean");
    try plan.withRowBowleySkewness(&.{ "a", "b" }, "row_bowley");
    try plan.withRowQuartileCoeffDispersion(&.{ "a", "b" }, "row_qcd");
    try plan.withRowKelleySkewness(&.{ "a", "b" }, "row_kelley");
    try plan.withRowMad(&.{ "a", "b" }, "row_mad");
    try plan.withRowMode(&.{ "a", "b" }, "row_mode");
    try plan.withRowEntropy(&.{ "a", "b", "wa" }, "row_entropy");
    try plan.withRowGiniImpurity(&.{ "a", "b", "wa" }, "row_gini");
    try plan.withRowPerplexity(&.{ "a", "b", "wa" }, "row_perplexity");
    try plan.withRowInverseSimpson(&.{ "a", "b", "wa" }, "row_inverse_simpson");
    try plan.withRowSimpsonConcentration(&.{ "a", "b", "wa" }, "row_concentration");
    try plan.withRowEvenness(&.{ "a", "b", "wa" }, "row_evenness");
    try plan.withRowModeCount(&.{ "a", "b", "wa" }, "row_mode_count");
    try plan.withRowModeRatio(&.{ "a", "b", "wa" }, "row_mode_ratio");
    try plan.withRowModeMargin(&.{ "a", "b", "wa" }, "row_mode_margin");
    try plan.withRowModeMarginRatio(&.{ "a", "b", "wa" }, "row_mode_margin_ratio");
    try plan.withRowPairCount(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_pair_count");
    try plan.withRowWeightedPairWeightSum(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_pair_weight_sum");
    try plan.withRowWeightedPairPositiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_pair_positive_count");
    try plan.withRowWeightedPairEffectiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_pair_effective_n");
    try plan.withRowWeightedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mean");
    try plan.withRowWeightedSum(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_sum");
    try plan.withRowCumulativeWeightedSum(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumsum", "b_row_weighted_cumsum" });
    try plan.withRowWeightedWeightSum(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_weight_sum");
    try plan.withRowWeightedPositiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_positive_count");
    try plan.withRowWeightedEffectiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_effective_n");
    try plan.withRowWeightedMeanSq(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mean_square");
    try plan.withRowWeightedRMS(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_rms");
    try plan.withRowWeightedMeanAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mean_abs");
    try plan.withRowWeightedL1(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_l1");
    try plan.withRowWeightedL2Norm(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_l2");
    try plan.withRowWeightedMinimum(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_min");
    try plan.withRowWeightedMaximum(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_max");
    try plan.withRowWeightedMaximumAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_max_abs");
    try plan.withRowWeightedMinimumAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_min_abs");
    try plan.withRowWeightedRange(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_range");
    try plan.withRowWeightedMidrange(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_midrange");
    try plan.withRowWeightedRangeCoefficient(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_range_coeff");
    try plan.withRowWeightedProduct(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_product");
    try plan.withRowWeightedGeoMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_geo");
    try plan.withRowWeightedHarmonicMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_harmonic");
    try plan.withRowWeightedLogsumexp(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_logsumexp");
    try plan.withRowWeightedLogMeanExp(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_logmeanexp");
    try plan.withRowWeightedQuantile(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_quantile", 0.9);
    try plan.withRowWeightedMedian(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_median");
    try plan.withRowWeightedIqr(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_iqr");
    try plan.withRowWeightedMad(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mad");
    try plan.withRowWeightedTrimmedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_trimmed", 0.25);
    try plan.withRowWeightedWinsorizedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_winsorized", 0.25);
    try plan.withRowWeightedInterdecileRange(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_idr");
    try plan.withRowWeightedMidhinge(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_midhinge");
    try plan.withRowWeightedTrimean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_trimean");
    try plan.withRowWeightedBowleySkewness(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_bowley");
    try plan.withRowWeightedQcd(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_qcd");
    try plan.withRowWeightedKelleySkewness(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_kelley");
    try plan.withRowWeightedMode(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode");
    try plan.withRowWeightedModeWeight(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode_weight");
    try plan.withRowWeightedModeRatio(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode_ratio");
    try plan.withRowWeightedModeMargin(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode_margin");
    try plan.withRowWeightedModeMarginRatio(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_mode_margin_ratio");
    try plan.withRowWeightedEntropy(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_entropy");
    try plan.withRowWeightedGiniImpurity(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_gini");
    try plan.withRowWeightedPerplexity(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_perplexity");
    try plan.withRowWeightedInverseSimpson(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_inverse");
    try plan.withRowWeightedSimpsonConcentration(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_concentration");
    try plan.withRowWeightedEvenness(&.{ "a", "b", "wa" }, &.{ "wb", "wa", "wb" }, "row_weighted_evenness");
    try plan.withRowWeightedMeanAbsDev(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mean_abs_dev");
    try plan.withRowWeightedMadRatio(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mad_ratio");
    try plan.withRowWeightedGiniMeanDiff(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_gini_mean_diff");
    try plan.withRowWeightedGiniCoeff(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_gini_coeff");
    try plan.withRowWeightedVariance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_variance", 0.0);
    try plan.withRowWeightedStddev(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_stddev", 0.0);
    try plan.withRowWeightedSEM(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_sem", 0.0);
    try plan.withRowWeightedCV(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_cv", 0.0);
    try plan.withRowWeightedFano(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_fano", 0.0);
    try plan.withRowWeightedSkew(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_skew");
    try plan.withRowWeightedKurtosis(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_kurt");
    try plan.withRowWeightedCovariance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_covariance", 0.0);
    try plan.withRowWeightedCorrelation(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_correlation", 0.0);
    try plan.withRowWeightedBeta(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_beta", 0.0);
    try plan.withRowWeightedDot(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_dot");
    try plan.withRowWeightedCosine(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_cosine");
    try plan.withRowWeightedSquaredDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_sqdist");
    try plan.withRowWeightedL2Distance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_euclidean");
    try plan.withRowWeightedL1Distance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_manhattan");
    try plan.withRowWeightedChebyshevDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_chebyshev");
    try plan.withRowWeightedCanberraDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_canberra");
    try plan.withRowWeightedBrayCurtisDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_bray");
    try plan.withRowWeightedBias(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_bias");
    try plan.withRowWeightedMAE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_mae");
    try plan.withRowWeightedMSE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_mse");
    try plan.withRowWeightedRMSE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_rmse");
    try plan.withRowWeightedMAPE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_mape");
    try plan.withRowWeightedSMAPE(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_smape");
    try plan.withRowDot(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_dot");
    try plan.withRowCosineSimilarity(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_cosine");
    try plan.withRowSquaredEuclideanDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_sqdist");
    try plan.withRowEuclideanDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_euclidean");
    try plan.withRowManhattanDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_manhattan");
    try plan.withRowChebyshevDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_chebyshev");
    try plan.withRowCanberraDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_canberra");
    try plan.withRowBrayCurtisDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_bray");
    try plan.withRowMeanError(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_mean_error");
    try plan.withRowMae(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_mae");
    try plan.withRowMse(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_mse");
    try plan.withRowRmse(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_rmse");
    try plan.withRowMape(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_mape");
    try plan.withRowSmape(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_smape");
    try plan.withRowCovariance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_covariance");
    try plan.withRowCorrelation(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_correlation");
    try plan.withRowBeta(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_beta");
    try plan.withRowCountDistinct(&.{ "a", "b" }, "row_distinct");
    try plan.withRowNUnique(&.{ "a", "b" }, "row_unique");
    try plan.withRowCumulativeMode(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummode", "row_b_cummode", "row_wa_cummode", "row_wb_cummode" });
    try plan.withRowCumulativeModeCount(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummode_count", "row_b_cummode_count", "row_wa_cummode_count", "row_wb_cummode_count" });
    try plan.withRowCumulativeModeRatio(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummode_ratio", "row_b_cummode_ratio", "row_wa_cummode_ratio", "row_wb_cummode_ratio" });
    try plan.withRowCumulativeModeMargin(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummode_margin", "row_b_cummode_margin", "row_wa_cummode_margin", "row_wb_cummode_margin" });
    try plan.withRowCumulativeModeMarginRatio(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummode_margin_ratio", "row_b_cummode_margin_ratio", "row_wa_cummode_margin_ratio", "row_wb_cummode_margin_ratio" });
    try plan.withRowCumulativeDistinctCount(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumdistinct", "row_b_cumdistinct", "row_wa_cumdistinct", "row_wb_cumdistinct" });
    try plan.withRowCumulativeNUnique(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumunique", "row_b_cumunique", "row_wa_cumunique", "row_wb_cumunique" });
    try plan.withRowSum(&.{ "a", "b" }, "row_sum");
    try plan.withRowMean(&.{ "a", "b" }, "row_mean");
    try plan.withRowLogSumExp(&.{ "a", "b" }, "row_logsumexp");
    try plan.withRowLogMeanExp(&.{ "a", "b" }, "row_logmeanexp");
    try plan.withRowCentered(&.{ "a", "b" }, &.{ "row_a_centered", "row_b_centered" });
    try plan.withRowZScore(&.{ "a", "b" }, &.{ "row_a_zscore", "row_b_zscore" });
    try plan.withRowDenseRank(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_dense_rank", "row_b_dense_rank", "row_wa_dense_rank", "row_wb_dense_rank" });
    try plan.withRowOrdinalRank(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_ordinal_rank", "row_b_ordinal_rank", "row_wa_ordinal_rank", "row_wb_ordinal_rank" });
    try plan.withRowAverageRank(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_average_rank", "row_b_average_rank", "row_wa_average_rank", "row_wb_average_rank" });
    try plan.withRowCompetitionRank(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_competition_rank", "row_b_competition_rank", "row_wa_competition_rank", "row_wb_competition_rank" });
    try plan.withRowPercentRank(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_percent_rank", "row_b_percent_rank", "row_wa_percent_rank", "row_wb_percent_rank" });
    try plan.withRowCumeDist(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cume", "row_b_cume", "row_wa_cume", "row_wb_cume" });
    try plan.withRowCumulativeSum(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumsum", "row_b_cumsum", "row_wa_cumsum", "row_wb_cumsum" });
    try plan.withRowCumulativeMean(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummean", "row_b_cummean", "row_wa_cummean", "row_wb_cummean" });
    try plan.withRowCumulativeLogSumExp(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumlse", "row_b_cumlse", "row_wa_cumlse", "row_wb_cumlse" });
    try plan.withRowCumulativeLogMeanExp(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumlme", "row_b_cumlme", "row_wa_cumlme", "row_wb_cumlme" });
    try plan.withRowCumulativeGeometricMean(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumgeo", "row_b_cumgeo", "row_wa_cumgeo", "row_wb_cumgeo" });
    try plan.withRowCumulativeHarmonicMean(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumharm", "row_b_cumharm", "row_wa_cumharm", "row_wb_cumharm" });
    try plan.withRowCumulativeVariance(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumvar", "row_b_cumvar", "row_wa_cumvar", "row_wb_cumvar" }, 0.0);
    try plan.withRowCumulativeStddev(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumstd", "row_b_cumstd", "row_wa_cumstd", "row_wb_cumstd" }, 0.0);
    try plan.withRowCumulativeSem(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumsem", "row_b_cumsem", "row_wa_cumsem", "row_wb_cumsem" }, 0.0);
    try plan.withRowCumulativeCv(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumcv", "row_b_cumcv", "row_wa_cumcv", "row_wb_cumcv" }, 0.0);
    try plan.withRowCumulativeFano(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumfano", "row_b_cumfano", "row_wa_cumfano", "row_wb_cumfano" }, 0.0);
    try plan.withRowCumulativeSkewness(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumskew", "row_b_cumskew", "row_wa_cumskew", "row_wb_cumskew" });
    try plan.withRowCumulativeKurtosis(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumkurt", "row_b_cumkurt", "row_wa_cumkurt", "row_wb_cumkurt" });
    try plan.withRowCumulativeRms(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumrms", "row_b_cumrms", "row_wa_cumrms", "row_wb_cumrms" });
    try plan.withRowCumulativeMeanAbs(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummeanabs", "row_b_cummeanabs", "row_wa_cummeanabs", "row_wb_cummeanabs" });
    try plan.withRowPrefixMeanSquare(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummeansq", "row_b_cummeansq", "row_wa_cummeansq", "row_wb_cummeansq" });
    try plan.withRowCumulativeMaxAbs(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummaxabs", "row_b_cummaxabs", "row_wa_cummaxabs", "row_wb_cummaxabs" });
    try plan.withRowPrefixMinAbs(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumminabs", "row_b_cumminabs", "row_wa_cumminabs", "row_wb_cumminabs" });
    try plan.withRowCumulativeL1Norm(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cuml1", "row_b_cuml1", "row_wa_cuml1", "row_wb_cuml1" });
    try plan.withRowCumulativeL2Norm(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cuml2", "row_b_cuml2", "row_wa_cuml2", "row_wb_cuml2" });
    try plan.withRowCumulativeProduct(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumprod", "row_b_cumprod", "row_wa_cumprod", "row_wb_cumprod" });
    try plan.withRowCumulativeMax(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummax", "row_b_cummax", "row_wa_cummax", "row_wb_cummax" });
    try plan.withRowCumulativeMin(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cummin", "row_b_cummin", "row_wa_cummin", "row_wb_cummin" });
    try plan.withRowCumulativeRange(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_cumrange", "row_b_cumrange", "row_wa_cumrange", "row_wb_cumrange" });
    try plan.withRowRobustZScore(&.{ "a", "b" }, &.{ "row_a_robust_zscore", "row_b_robust_zscore" });
    try plan.withRowIqrOutlier(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_iqr_outlier", "row_b_iqr_outlier", "row_wa_iqr_outlier", "row_wb_iqr_outlier" });
    try plan.withRowTukeyWinsorize(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_tukey_winsor", "row_b_tukey_winsor", "row_wa_tukey_winsor", "row_wb_tukey_winsor" });
    try plan.withRowMaxIndicator(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_is_max", "row_b_is_max", "row_wa_is_max", "row_wb_is_max" });
    try plan.withRowMinIndicator(&.{ "a", "b", "wa", "wb" }, &.{ "row_a_is_min", "row_b_is_min", "row_wa_is_min", "row_wb_is_min" });
    try plan.withRowMinMaxScale(&.{ "a", "b" }, &.{ "row_a_minmax", "row_b_minmax" });
    try plan.withRowL2Normalize(&.{ "a", "b" }, &.{ "row_a_l2_unit", "row_b_l2_unit" });
    try plan.withRowL1Normalize(&.{ "a", "b" }, &.{ "row_a_l1_unit", "row_b_l1_unit" });
    try plan.withRowSumNormalize(&.{ "a", "b" }, &.{ "row_a_share", "row_b_share" });
    try plan.withRowMeanNormalize(&.{ "a", "b" }, &.{ "row_a_mean_ratio", "row_b_mean_ratio" });
    try plan.withRowMaxAbsNormalize(&.{ "a", "b" }, &.{ "row_a_maxabs", "row_b_maxabs" });
    try plan.withRowSoftmax(&.{ "a", "b" }, &.{ "row_a_softmax", "row_b_softmax" });
    try plan.withRowLogSoftmax(&.{ "a", "b" }, &.{ "row_a_log_softmax", "row_b_log_softmax" });
    try plan.withRowSoftmin(&.{ "a", "b" }, &.{ "row_a_softmin", "row_b_softmin" });
    try plan.withRowLogSoftmin(&.{ "a", "b" }, &.{ "row_a_log_softmin", "row_b_log_softmin" });
    try plan.withRowSoftmaxEntropy(&.{ "a", "b" }, "row_softmax_entropy");
    try plan.withRowSoftmaxPerplexity(&.{ "a", "b" }, "row_softmax_perplexity");
    try plan.withRowSoftmaxConfidence(&.{ "a", "b" }, "row_softmax_confidence");
    try plan.withRowSoftmaxMargin(&.{ "a", "b" }, "row_softmax_margin");
    try plan.withRowSoftmaxEvenness(&.{ "a", "b" }, "row_softmax_evenness");
    try plan.withRowSoftmaxConcentration(&.{ "a", "b" }, "row_softmax_concentration");
    try plan.withRowSoftmaxNormalizedHhi(&.{ "a", "b" }, "row_softmax_normalized_hhi");
    try plan.withRowSoftmaxGiniImpurity(&.{ "a", "b" }, "row_softmax_gini");
    try plan.withRowSoftmaxInverseSimpson(&.{ "a", "b" }, "row_softmax_inverse");
    try plan.withRowSoftmaxSimpsonEvenness(&.{ "a", "b" }, "row_softmax_simpson_evenness");
    try plan.withRowLogitMargin(&.{ "a", "b" }, "row_logit_margin");
    try plan.withRowGeometricMean(&.{ "a", "b" }, "row_geo");
    try plan.withRowMagnitudeGeometricMean(&.{ "a", "b" }, "row_magnitude_geo");
    try plan.withRowHarmonicMean(&.{ "a", "b" }, "row_harm");
    try plan.withRowSkewness(&.{ "a", "b" }, "row_skew");
    try plan.withRowMagnitudeSkewness(&.{ "a", "b" }, "row_magnitude_skew");
    try plan.withRowKurtosis(&.{ "a", "b" }, "row_kurt");
    try plan.withRowMagnitudeKurtosis(&.{ "a", "b" }, "row_magnitude_kurt");
    try plan.withRowProd(&.{ "a", "b" }, "row_prod");
    try plan.withRowMin(&.{ "a", "b" }, "row_min");
    try plan.withRowMax(&.{ "a", "b" }, "row_max");
    try plan.withRowPtp(&.{ "a", "b" }, "row_ptp");
    try plan.withRowMagnitudePtp(&.{ "a", "b" }, "row_magnitude_ptp");
    try plan.withRowMidrange(&.{ "a", "b" }, "row_midrange");
    try plan.withRowMagnitudeMidrange(&.{ "a", "b" }, "row_magnitude_midrange");
    try plan.withRowRangeCoeff(&.{ "a", "b" }, "row_range_coeff");
    try plan.withRowMagnitudeRangeCoeff(&.{ "a", "b" }, "row_magnitude_range_coeff");
    try plan.withRowMeanAbs(&.{ "a", "b" }, "row_mean_abs");
    try plan.withRowHhi(&.{ "a", "b" }, "row_hhi");
    try plan.withRowMagnitudeNormalizedHhi(&.{ "a", "b" }, "row_magnitude_normalized_hhi");
    try plan.withRowMagnitudeSparsity(&.{ "a", "b" }, "row_magnitude_sparsity");
    try plan.withRowMagnitudeInverseSimpson(&.{ "a", "b" }, "row_magnitude_inverse");
    try plan.withRowMagnitudeSimpsonEvenness(&.{ "a", "b" }, "row_magnitude_simpson_evenness");
    try plan.withRowMagnitudeDominance(&.{ "a", "b" }, "row_magnitude_dominance");
    try plan.withRowMagnitudeDominanceMargin(&.{ "a", "b" }, "row_magnitude_margin");
    try plan.withRowMagnitudeEntropy(&.{ "a", "b" }, "row_magnitude_entropy");
    try plan.withRowMagnitudePerplexity(&.{ "a", "b" }, "row_magnitude_perplexity");
    try plan.withRowMagnitudeEvenness(&.{ "a", "b" }, "row_magnitude_evenness");
    try plan.withRowMeanAbsDev(&.{ "a", "b" }, "row_mean_abs_dev");
    try plan.withRowGiniMeanDiff(&.{ "a", "b" }, "row_gini_mean_diff");
    try plan.withRowGiniCoefficient(&.{ "a", "b" }, "row_gini_coeff");
    try plan.withRowMeanAbsDevRatio(&.{ "a", "b" }, "row_mad_ratio");
    try plan.withRowRms(&.{ "a", "b" }, "row_rms");
    try plan.withRowL1Norm(&.{ "a", "b" }, "row_l1");
    try plan.withRowL2Norm(&.{ "a", "b" }, "row_l2");
    try plan.withRowVariance(&.{ "a", "b" }, "row_variance", 0.0);
    try plan.withRowMagnitudeVariance(&.{ "a", "b" }, "row_magnitude_variance", 0.0);
    try plan.withRowStddev(&.{ "a", "b" }, "row_stddev", 1.0);
    try plan.withRowMagnitudeStddev(&.{ "a", "b" }, "row_magnitude_stddev", 0.0);
    try plan.withRowSem(&.{ "a", "b" }, "row_sem", 1.0);
    try plan.withRowMagnitudeSem(&.{ "a", "b" }, "row_magnitude_sem", 0.0);
    try plan.withRowCv(&.{ "a", "b" }, "row_cv", 0.0);
    try plan.withRowMagnitudeCv(&.{ "a", "b" }, "row_magnitude_cv", 0.0);
    try plan.withRowMagnitudeFano(&.{ "a", "b" }, "row_magnitude_fano", 0.0);
    try plan.withRowFano(&.{ "a", "b" }, "row_fano", 0.0);
    try plan.select(&.{ "row_argmin", "row_argmax", "row_a_cum_argmin", "row_b_cum_argmin", "row_wa_cum_argmin", "row_wb_cum_argmin", "row_a_cum_argmax", "row_b_cum_argmax", "row_wa_cum_argmax", "row_wb_cum_argmax", "row_quantile", "row_quantile_range", "row_trimmed_mean", "row_winsorized_mean", "row_median", "row_iqr", "row_idr", "row_midhinge", "row_trimean", "row_bowley", "row_qcd", "row_kelley", "row_mad", "row_mode", "row_entropy", "row_gini", "row_perplexity", "row_inverse_simpson", "row_concentration", "row_evenness", "row_mode_count", "row_mode_ratio", "row_mode_margin", "row_mode_margin_ratio", "row_pair_count", "row_weighted_pair_weight_sum", "row_weighted_pair_positive_count", "row_weighted_pair_effective_n", "row_weighted_mean", "row_weighted_sum", "a_row_weighted_cumsum", "b_row_weighted_cumsum", "row_weighted_weight_sum", "row_weighted_positive_count", "row_weighted_effective_n", "row_weighted_mean_square", "row_weighted_rms", "row_weighted_mean_abs", "row_weighted_l1", "row_weighted_l2", "row_weighted_min", "row_weighted_max", "row_weighted_max_abs", "row_weighted_min_abs", "row_weighted_range", "row_weighted_midrange", "row_weighted_range_coeff", "row_weighted_product", "row_weighted_geo", "row_weighted_harmonic", "row_weighted_logsumexp", "row_weighted_logmeanexp", "row_weighted_quantile", "row_weighted_median", "row_weighted_iqr", "row_weighted_mad", "row_weighted_trimmed", "row_weighted_winsorized", "row_weighted_idr", "row_weighted_midhinge", "row_weighted_trimean", "row_weighted_bowley", "row_weighted_qcd", "row_weighted_kelley", "row_weighted_mode", "row_weighted_mode_weight", "row_weighted_mode_ratio", "row_weighted_mode_margin", "row_weighted_mode_margin_ratio", "row_weighted_entropy", "row_weighted_gini", "row_weighted_perplexity", "row_weighted_inverse", "row_weighted_concentration", "row_weighted_evenness", "row_weighted_mean_abs_dev", "row_weighted_mad_ratio", "row_weighted_gini_mean_diff", "row_weighted_gini_coeff", "row_weighted_variance", "row_weighted_stddev", "row_weighted_sem", "row_weighted_cv", "row_weighted_fano", "row_weighted_skew", "row_weighted_kurt", "row_weighted_covariance", "row_weighted_correlation", "row_weighted_beta", "row_weighted_dot", "row_weighted_cosine", "row_weighted_sqdist", "row_weighted_euclidean", "row_weighted_manhattan", "row_weighted_chebyshev", "row_weighted_canberra", "row_weighted_bray", "row_weighted_bias", "row_weighted_mae", "row_weighted_mse", "row_weighted_rmse", "row_weighted_mape", "row_weighted_smape", "row_dot", "row_cosine", "row_sqdist", "row_euclidean", "row_manhattan", "row_chebyshev", "row_canberra", "row_bray", "row_mean_error", "row_mae", "row_mse", "row_rmse", "row_mape", "row_smape", "row_covariance", "row_correlation", "row_beta", "row_distinct", "row_unique", "row_a_cummode", "row_b_cummode", "row_wa_cummode", "row_wb_cummode", "row_a_cummode_count", "row_b_cummode_count", "row_wa_cummode_count", "row_wb_cummode_count", "row_a_cummode_ratio", "row_b_cummode_ratio", "row_wa_cummode_ratio", "row_wb_cummode_ratio", "row_a_cummode_margin", "row_b_cummode_margin", "row_wa_cummode_margin", "row_wb_cummode_margin", "row_a_cummode_margin_ratio", "row_b_cummode_margin_ratio", "row_wa_cummode_margin_ratio", "row_wb_cummode_margin_ratio", "row_a_cumdistinct", "row_b_cumdistinct", "row_wa_cumdistinct", "row_wb_cumdistinct", "row_a_cumunique", "row_b_cumunique", "row_wa_cumunique", "row_wb_cumunique", "row_sum", "row_mean", "row_logsumexp", "row_logmeanexp", "row_a_centered", "row_b_centered", "row_a_zscore", "row_b_zscore", "row_a_dense_rank", "row_b_dense_rank", "row_wa_dense_rank", "row_wb_dense_rank", "row_a_ordinal_rank", "row_b_ordinal_rank", "row_wa_ordinal_rank", "row_wb_ordinal_rank", "row_a_average_rank", "row_b_average_rank", "row_wa_average_rank", "row_wb_average_rank", "row_a_competition_rank", "row_b_competition_rank", "row_wa_competition_rank", "row_wb_competition_rank", "row_a_percent_rank", "row_b_percent_rank", "row_wa_percent_rank", "row_wb_percent_rank", "row_a_cume", "row_b_cume", "row_wa_cume", "row_wb_cume", "row_a_cumsum", "row_b_cumsum", "row_wa_cumsum", "row_wb_cumsum", "row_a_cummean", "row_b_cummean", "row_wa_cummean", "row_wb_cummean", "row_a_cumlse", "row_b_cumlse", "row_wa_cumlse", "row_wb_cumlse", "row_a_cumlme", "row_b_cumlme", "row_wa_cumlme", "row_wb_cumlme", "row_a_cumgeo", "row_b_cumgeo", "row_wa_cumgeo", "row_wb_cumgeo", "row_a_cumharm", "row_b_cumharm", "row_wa_cumharm", "row_wb_cumharm", "row_a_cumvar", "row_b_cumvar", "row_wa_cumvar", "row_wb_cumvar", "row_a_cumstd", "row_b_cumstd", "row_wa_cumstd", "row_wb_cumstd", "row_a_cumsem", "row_b_cumsem", "row_wa_cumsem", "row_wb_cumsem", "row_a_cumcv", "row_b_cumcv", "row_wa_cumcv", "row_wb_cumcv", "row_a_cumfano", "row_b_cumfano", "row_wa_cumfano", "row_wb_cumfano", "row_a_cumskew", "row_b_cumskew", "row_wa_cumskew", "row_wb_cumskew", "row_a_cumkurt", "row_b_cumkurt", "row_wa_cumkurt", "row_wb_cumkurt", "row_a_cumrms", "row_b_cumrms", "row_wa_cumrms", "row_wb_cumrms", "row_a_cummeanabs", "row_b_cummeanabs", "row_wa_cummeanabs", "row_wb_cummeanabs", "row_a_cummeansq", "row_b_cummeansq", "row_wa_cummeansq", "row_wb_cummeansq", "row_a_cummaxabs", "row_b_cummaxabs", "row_wa_cummaxabs", "row_wb_cummaxabs", "row_a_cumminabs", "row_b_cumminabs", "row_wa_cumminabs", "row_wb_cumminabs", "row_a_cuml1", "row_b_cuml1", "row_wa_cuml1", "row_wb_cuml1", "row_a_cuml2", "row_b_cuml2", "row_wa_cuml2", "row_wb_cuml2", "row_a_cumprod", "row_b_cumprod", "row_wa_cumprod", "row_wb_cumprod", "row_a_cummax", "row_b_cummax", "row_wa_cummax", "row_wb_cummax", "row_a_cummin", "row_b_cummin", "row_wa_cummin", "row_wb_cummin", "row_a_cumrange", "row_b_cumrange", "row_wa_cumrange", "row_wb_cumrange", "row_a_robust_zscore", "row_b_robust_zscore", "row_a_iqr_outlier", "row_b_iqr_outlier", "row_wa_iqr_outlier", "row_wb_iqr_outlier", "row_a_tukey_winsor", "row_b_tukey_winsor", "row_wa_tukey_winsor", "row_wb_tukey_winsor", "row_a_is_max", "row_b_is_max", "row_wa_is_max", "row_wb_is_max", "row_a_is_min", "row_b_is_min", "row_wa_is_min", "row_wb_is_min", "row_a_minmax", "row_b_minmax", "row_a_l2_unit", "row_b_l2_unit", "row_a_l1_unit", "row_b_l1_unit", "row_a_share", "row_b_share", "row_a_mean_ratio", "row_b_mean_ratio", "row_a_maxabs", "row_b_maxabs", "row_a_softmax", "row_b_softmax", "row_a_log_softmax", "row_b_log_softmax", "row_a_softmin", "row_b_softmin", "row_a_log_softmin", "row_b_log_softmin", "row_softmax_entropy", "row_softmax_perplexity", "row_softmax_confidence", "row_softmax_margin", "row_softmax_evenness", "row_softmax_concentration", "row_softmax_normalized_hhi", "row_softmax_gini", "row_softmax_inverse", "row_softmax_simpson_evenness", "row_logit_margin", "row_geo", "row_magnitude_geo", "row_harm", "row_skew", "row_magnitude_skew", "row_kurt", "row_magnitude_kurt", "row_prod", "row_min", "row_max", "row_ptp", "row_magnitude_ptp", "row_midrange", "row_magnitude_midrange", "row_range_coeff", "row_magnitude_range_coeff", "row_mean_abs", "row_hhi", "row_magnitude_normalized_hhi", "row_magnitude_sparsity", "row_magnitude_inverse", "row_magnitude_simpson_evenness", "row_magnitude_dominance", "row_magnitude_margin", "row_magnitude_entropy", "row_magnitude_perplexity", "row_magnitude_evenness", "row_mean_abs_dev", "row_gini_mean_diff", "row_gini_coeff", "row_mad_ratio", "row_rms", "row_l1", "row_l2", "row_variance", "row_magnitude_variance", "row_stddev", "row_magnitude_stddev", "row_sem", "row_magnitude_sem", "row_cv", "row_magnitude_cv", "row_magnitude_fano", "row_fano" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_argmin([a,b]->row_argmin)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_argmax([a,b]->row_argmax)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_argmin([a,b,wa,wb]->[row_a_cum_argmin,row_b_cum_argmin,row_wa_cum_argmin,row_wb_cum_argmin])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_argmax([a,b,wa,wb]->[row_a_cum_argmax,row_b_cum_argmax,row_wa_cum_argmax,row_wb_cum_argmax])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_quantile([a,b]->row_quantile, q=0.25)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_quantile_range([a,b]->row_quantile_range, low_q=0.2, high_q=0.8)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_trimmed_mean([a,b]->row_trimmed_mean, trim_fraction=0.25)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_winsorized_mean([a,b]->row_winsorized_mean, winsor_fraction=0.25)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_median([a,b]->row_median)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_iqr([a,b]->row_iqr)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_interdecile_range([a,b]->row_idr)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_midhinge([a,b]->row_midhinge)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_trimean([a,b]->row_trimean)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_bowley_skewness([a,b]->row_bowley)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_quartile_coeff_dispersion([a,b]->row_qcd)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_kelley_skewness([a,b]->row_kelley)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mad([a,b]->row_mad)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mode([a,b]->row_mode)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_entropy([a,b,wa]->row_entropy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_gini_impurity([a,b,wa]->row_gini)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_perplexity([a,b,wa]->row_perplexity)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_inverse_simpson([a,b,wa]->row_inverse_simpson)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_simpson_concentration([a,b,wa]->row_concentration)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_evenness([a,b,wa]->row_evenness)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mode_count([a,b,wa]->row_mode_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mode_ratio([a,b,wa]->row_mode_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mode_margin([a,b,wa]->row_mode_margin)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mode_margin_ratio([a,b,wa]->row_mode_margin_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_pair_count(lhs=[a,b], rhs=[wa,wb]->row_pair_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_pair_weight_sum(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_pair_weight_sum)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_pair_positive_count(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_pair_positive_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_pair_effective_n(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_pair_effective_n)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mean(values=[a,b], weights=[wa,wb]->row_weighted_mean)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_sum(values=[a,b], weights=[wa,wb]->row_weighted_sum)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_sum(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumsum,b_row_weighted_cumsum])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_weight_sum(values=[a,b], weights=[wa,wb]->row_weighted_weight_sum)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_positive_count(values=[a,b], weights=[wa,wb]->row_weighted_positive_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_effective_n(values=[a,b], weights=[wa,wb]->row_weighted_effective_n)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mean_square(values=[a,b], weights=[wa,wb]->row_weighted_mean_square)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_rms(values=[a,b], weights=[wa,wb]->row_weighted_rms)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mean_abs(values=[a,b], weights=[wa,wb]->row_weighted_mean_abs)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_l1_norm(values=[a,b], weights=[wa,wb]->row_weighted_l1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_l2_norm(values=[a,b], weights=[wa,wb]->row_weighted_l2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_min(values=[a,b], weights=[wa,wb]->row_weighted_min)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_max(values=[a,b], weights=[wa,wb]->row_weighted_max)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_max_abs(values=[a,b], weights=[wa,wb]->row_weighted_max_abs)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_min_abs(values=[a,b], weights=[wa,wb]->row_weighted_min_abs)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_range(values=[a,b], weights=[wa,wb]->row_weighted_range)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_midrange(values=[a,b], weights=[wa,wb]->row_weighted_midrange)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_range_coeff(values=[a,b], weights=[wa,wb]->row_weighted_range_coeff)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_product(values=[a,b], weights=[wa,wb]->row_weighted_product)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_geometric_mean(values=[a,b], weights=[wa,wb]->row_weighted_geo)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_harmonic_mean(values=[a,b], weights=[wa,wb]->row_weighted_harmonic)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_logsumexp(values=[a,b], weights=[wa,wb]->row_weighted_logsumexp)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_logmeanexp(values=[a,b], weights=[wa,wb]->row_weighted_logmeanexp)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_quantile(values=[a,b], weights=[wa,wb]->row_weighted_quantile, q=0.9)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_median(values=[a,b], weights=[wa,wb]->row_weighted_median)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_iqr(values=[a,b], weights=[wa,wb]->row_weighted_iqr)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mad(values=[a,b], weights=[wa,wb]->row_weighted_mad)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_trimmed_mean(values=[a,b], weights=[wa,wb]->row_weighted_trimmed, trim_fraction=0.25)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_winsorized_mean(values=[a,b], weights=[wa,wb]->row_weighted_winsorized, winsor_fraction=0.25)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_interdecile_range(values=[a,b], weights=[wa,wb]->row_weighted_idr)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_midhinge(values=[a,b], weights=[wa,wb]->row_weighted_midhinge)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_trimean(values=[a,b], weights=[wa,wb]->row_weighted_trimean)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_bowley_skewness(values=[a,b], weights=[wa,wb]->row_weighted_bowley)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_quartile_coeff_dispersion(values=[a,b], weights=[wa,wb]->row_weighted_qcd)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_kelley_skewness(values=[a,b], weights=[wa,wb]->row_weighted_kelley)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mode(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_mode)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mode_weight(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_mode_weight)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mode_ratio(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_mode_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mode_margin(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_mode_margin)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mode_margin_ratio(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_mode_margin_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_entropy(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_entropy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_gini_impurity(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_gini)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_perplexity(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_perplexity)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_inverse_simpson(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_inverse)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_simpson_concentration(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_concentration)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_evenness(values=[a,b,wa], weights=[wb,wa,wb]->row_weighted_evenness)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mean_abs_dev(values=[a,b], weights=[wa,wb]->row_weighted_mean_abs_dev)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mean_abs_dev_ratio(values=[a,b], weights=[wa,wb]->row_weighted_mad_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_gini_mean_diff(values=[a,b], weights=[wa,wb]->row_weighted_gini_mean_diff)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_gini_coefficient(values=[a,b], weights=[wa,wb]->row_weighted_gini_coeff)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_variance(values=[a,b], weights=[wa,wb]->row_weighted_variance, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_stddev(values=[a,b], weights=[wa,wb]->row_weighted_stddev, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_sem(values=[a,b], weights=[wa,wb]->row_weighted_sem, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_cv(values=[a,b], weights=[wa,wb]->row_weighted_cv, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_fano(values=[a,b], weights=[wa,wb]->row_weighted_fano, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_skewness(values=[a,b], weights=[wa,wb]->row_weighted_skew)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_kurtosis(values=[a,b], weights=[wa,wb]->row_weighted_kurt)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_covariance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_covariance, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_correlation(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_correlation, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_beta(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_beta, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_dot(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_dot)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_cosine_similarity(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_cosine)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_squared_euclidean_distance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_sqdist)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_euclidean_distance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_euclidean)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_manhattan_distance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_manhattan)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_chebyshev_distance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_chebyshev)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_canberra_distance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_canberra)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_bray_curtis_distance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_bray)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mean_error(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_bias)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mae(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_mae)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mse(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_mse)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_rmse(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_rmse)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mape(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_mape)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_smape(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_smape)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_dot(lhs=[a,b], rhs=[wa,wb]->row_dot)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cosine_similarity(lhs=[a,b], rhs=[wa,wb]->row_cosine)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_squared_euclidean_distance(lhs=[a,b], rhs=[wa,wb]->row_sqdist)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_euclidean_distance(lhs=[a,b], rhs=[wa,wb]->row_euclidean)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_manhattan_distance(lhs=[a,b], rhs=[wa,wb]->row_manhattan)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_chebyshev_distance(lhs=[a,b], rhs=[wa,wb]->row_chebyshev)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_canberra_distance(lhs=[a,b], rhs=[wa,wb]->row_canberra)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_bray_curtis_distance(lhs=[a,b], rhs=[wa,wb]->row_bray)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mean_error(actual=[a,b], predicted=[wa,wb]->row_mean_error)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mae(lhs=[a,b], rhs=[wa,wb]->row_mae)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mse(lhs=[a,b], rhs=[wa,wb]->row_mse)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_rmse(lhs=[a,b], rhs=[wa,wb]->row_rmse)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mape(actual=[a,b], predicted=[wa,wb]->row_mape)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_smape(actual=[a,b], predicted=[wa,wb]->row_smape)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_covariance(lhs=[a,b], rhs=[wa,wb]->row_covariance)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_correlation(lhs=[a,b], rhs=[wa,wb]->row_correlation)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_beta(lhs=[a,b], rhs=[wa,wb]->row_beta)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_count_distinct([a,b]->row_distinct)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_n_unique([a,b]->row_unique)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_mode([a,b,wa,wb]->[row_a_cummode,row_b_cummode,row_wa_cummode,row_wb_cummode])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_mode_count([a,b,wa,wb]->[row_a_cummode_count,row_b_cummode_count,row_wa_cummode_count,row_wb_cummode_count])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_mode_ratio([a,b,wa,wb]->[row_a_cummode_ratio,row_b_cummode_ratio,row_wa_cummode_ratio,row_wb_cummode_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_mode_margin([a,b,wa,wb]->[row_a_cummode_margin,row_b_cummode_margin,row_wa_cummode_margin,row_wb_cummode_margin])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_mode_margin_ratio([a,b,wa,wb]->[row_a_cummode_margin_ratio,row_b_cummode_margin_ratio,row_wa_cummode_margin_ratio,row_wb_cummode_margin_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_distinct_count([a,b,wa,wb]->[row_a_cumdistinct,row_b_cumdistinct,row_wa_cumdistinct,row_wb_cumdistinct])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_n_unique([a,b,wa,wb]->[row_a_cumunique,row_b_cumunique,row_wa_cumunique,row_wb_cumunique])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_sum([a,b]->row_sum)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mean([a,b]->row_mean)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_logsumexp([a,b]->row_logsumexp)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_logmeanexp([a,b]->row_logmeanexp)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_centered([a,b]->[row_a_centered,row_b_centered])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_zscore([a,b]->[row_a_zscore,row_b_zscore])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_dense_rank([a,b,wa,wb]->[row_a_dense_rank,row_b_dense_rank,row_wa_dense_rank,row_wb_dense_rank])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_ordinal_rank([a,b,wa,wb]->[row_a_ordinal_rank,row_b_ordinal_rank,row_wa_ordinal_rank,row_wb_ordinal_rank])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_average_rank([a,b,wa,wb]->[row_a_average_rank,row_b_average_rank,row_wa_average_rank,row_wb_average_rank])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_competition_rank([a,b,wa,wb]->[row_a_competition_rank,row_b_competition_rank,row_wa_competition_rank,row_wb_competition_rank])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_percent_rank([a,b,wa,wb]->[row_a_percent_rank,row_b_percent_rank,row_wa_percent_rank,row_wb_percent_rank])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cume_dist([a,b,wa,wb]->[row_a_cume,row_b_cume,row_wa_cume,row_wb_cume])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_sum([a,b,wa,wb]->[row_a_cumsum,row_b_cumsum,row_wa_cumsum,row_wb_cumsum])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_mean([a,b,wa,wb]->[row_a_cummean,row_b_cummean,row_wa_cummean,row_wb_cummean])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_logsumexp([a,b,wa,wb]->[row_a_cumlse,row_b_cumlse,row_wa_cumlse,row_wb_cumlse])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_logmeanexp([a,b,wa,wb]->[row_a_cumlme,row_b_cumlme,row_wa_cumlme,row_wb_cumlme])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_geometric_mean([a,b,wa,wb]->[row_a_cumgeo,row_b_cumgeo,row_wa_cumgeo,row_wb_cumgeo])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_harmonic_mean([a,b,wa,wb]->[row_a_cumharm,row_b_cumharm,row_wa_cumharm,row_wb_cumharm])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_variance([a,b,wa,wb]->[row_a_cumvar,row_b_cumvar,row_wa_cumvar,row_wb_cumvar], correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_stddev([a,b,wa,wb]->[row_a_cumstd,row_b_cumstd,row_wa_cumstd,row_wb_cumstd], correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_sem([a,b,wa,wb]->[row_a_cumsem,row_b_cumsem,row_wa_cumsem,row_wb_cumsem], correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_cv([a,b,wa,wb]->[row_a_cumcv,row_b_cumcv,row_wa_cumcv,row_wb_cumcv], correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_fano([a,b,wa,wb]->[row_a_cumfano,row_b_cumfano,row_wa_cumfano,row_wb_cumfano], correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_skewness([a,b,wa,wb]->[row_a_cumskew,row_b_cumskew,row_wa_cumskew,row_wb_cumskew])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_kurtosis([a,b,wa,wb]->[row_a_cumkurt,row_b_cumkurt,row_wa_cumkurt,row_wb_cumkurt])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_rms([a,b,wa,wb]->[row_a_cumrms,row_b_cumrms,row_wa_cumrms,row_wb_cumrms])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_mean_abs([a,b,wa,wb]->[row_a_cummeanabs,row_b_cummeanabs,row_wa_cummeanabs,row_wb_cummeanabs])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_mean_square([a,b,wa,wb]->[row_a_cummeansq,row_b_cummeansq,row_wa_cummeansq,row_wb_cummeansq])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_max_abs([a,b,wa,wb]->[row_a_cummaxabs,row_b_cummaxabs,row_wa_cummaxabs,row_wb_cummaxabs])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_min_abs([a,b,wa,wb]->[row_a_cumminabs,row_b_cumminabs,row_wa_cumminabs,row_wb_cumminabs])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_l1_norm([a,b,wa,wb]->[row_a_cuml1,row_b_cuml1,row_wa_cuml1,row_wb_cuml1])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_l2_norm([a,b,wa,wb]->[row_a_cuml2,row_b_cuml2,row_wa_cuml2,row_wb_cuml2])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_product([a,b,wa,wb]->[row_a_cumprod,row_b_cumprod,row_wa_cumprod,row_wb_cumprod])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_max([a,b,wa,wb]->[row_a_cummax,row_b_cummax,row_wa_cummax,row_wb_cummax])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_min([a,b,wa,wb]->[row_a_cummin,row_b_cummin,row_wa_cummin,row_wb_cummin])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_range([a,b,wa,wb]->[row_a_cumrange,row_b_cumrange,row_wa_cumrange,row_wb_cumrange])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_robust_zscore([a,b]->[row_a_robust_zscore,row_b_robust_zscore])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_iqr_outlier([a,b,wa,wb]->[row_a_iqr_outlier,row_b_iqr_outlier,row_wa_iqr_outlier,row_wb_iqr_outlier])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_tukey_winsorize([a,b,wa,wb]->[row_a_tukey_winsor,row_b_tukey_winsor,row_wa_tukey_winsor,row_wb_tukey_winsor])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_max_indicator([a,b,wa,wb]->[row_a_is_max,row_b_is_max,row_wa_is_max,row_wb_is_max])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_min_indicator([a,b,wa,wb]->[row_a_is_min,row_b_is_min,row_wa_is_min,row_wb_is_min])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_minmax_scale([a,b]->[row_a_minmax,row_b_minmax])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_l2_normalize([a,b]->[row_a_l2_unit,row_b_l2_unit])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_l1_normalize([a,b]->[row_a_l1_unit,row_b_l1_unit])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_sum_normalize([a,b]->[row_a_share,row_b_share])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mean_normalize([a,b]->[row_a_mean_ratio,row_b_mean_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_max_abs_normalize([a,b]->[row_a_maxabs,row_b_maxabs])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax([a,b]->[row_a_softmax,row_b_softmax])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_log_softmax([a,b]->[row_a_log_softmax,row_b_log_softmax])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmin([a,b]->[row_a_softmin,row_b_softmin])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_log_softmin([a,b]->[row_a_log_softmin,row_b_log_softmin])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_entropy([a,b]->row_softmax_entropy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_perplexity([a,b]->row_softmax_perplexity)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_confidence([a,b]->row_softmax_confidence)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_margin([a,b]->row_softmax_margin)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_evenness([a,b]->row_softmax_evenness)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_concentration([a,b]->row_softmax_concentration)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_normalized_hhi([a,b]->row_softmax_normalized_hhi)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_gini_impurity([a,b]->row_softmax_gini)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_inverse_simpson([a,b]->row_softmax_inverse)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_softmax_simpson_evenness([a,b]->row_softmax_simpson_evenness)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_logit_margin([a,b]->row_logit_margin)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_geometric_mean([a,b]->row_geo)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_geometric_mean([a,b]->row_magnitude_geo)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_harmonic_mean([a,b]->row_harm)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_skewness([a,b]->row_skew)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_skewness([a,b]->row_magnitude_skew)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_kurtosis([a,b]->row_kurt)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_kurtosis([a,b]->row_magnitude_kurt)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_prod([a,b]->row_prod)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_min([a,b]->row_min)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_max([a,b]->row_max)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_ptp([a,b]->row_ptp)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_ptp([a,b]->row_magnitude_ptp)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_midrange([a,b]->row_midrange)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_midrange([a,b]->row_magnitude_midrange)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_range_coeff([a,b]->row_range_coeff)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_range_coeff([a,b]->row_magnitude_range_coeff)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mean_abs([a,b]->row_mean_abs)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_hhi([a,b]->row_hhi)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_normalized_hhi([a,b]->row_magnitude_normalized_hhi)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_sparsity([a,b]->row_magnitude_sparsity)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_inverse_simpson([a,b]->row_magnitude_inverse)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_simpson_evenness([a,b]->row_magnitude_simpson_evenness)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_dominance([a,b]->row_magnitude_dominance)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_dominance_margin([a,b]->row_magnitude_margin)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_entropy([a,b]->row_magnitude_entropy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_perplexity([a,b]->row_magnitude_perplexity)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_evenness([a,b]->row_magnitude_evenness)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mean_abs_dev([a,b]->row_mean_abs_dev)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_gini_mean_diff([a,b]->row_gini_mean_diff)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_gini_coefficient([a,b]->row_gini_coeff)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mean_abs_dev_ratio([a,b]->row_mad_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_rms([a,b]->row_rms)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_l1_norm([a,b]->row_l1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_l2_norm([a,b]->row_l2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_variance([a,b]->row_variance, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_variance([a,b]->row_magnitude_variance, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_stddev([a,b]->row_stddev, correction=1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_stddev([a,b]->row_magnitude_stddev, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_sem([a,b]->row_sem, correction=1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_sem([a,b]->row_magnitude_sem, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cv([a,b]->row_cv, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_cv([a,b]->row_magnitude_cv, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_magnitude_fano([a,b]->row_magnitude_fano, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_fano([a,b]->row_fano, correction=0)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 381), result.width());
    const row_argmin_column = try result.column("row_argmin");
    try std.testing.expect(row_argmin_column.i64.nullable());
    const row_argmin = try row_argmin_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_argmin);
    const row_argmin_validity = try row_argmin_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_argmin_validity);
    const row_argmax_column = try result.column("row_argmax");
    try std.testing.expect(row_argmax_column.i64.nullable());
    const row_argmax = try row_argmax_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_argmax);
    const row_argmax_validity = try row_argmax_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_argmax_validity);
    const row_b_cum_argmin_column = try result.column("row_b_cum_argmin");
    try std.testing.expect(row_b_cum_argmin_column.i64.nullable());
    const row_b_cum_argmin = try row_b_cum_argmin_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cum_argmin);
    const row_b_cum_argmin_validity = try row_b_cum_argmin_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cum_argmin_validity);
    const row_wb_cum_argmax_column = try result.column("row_wb_cum_argmax");
    try std.testing.expect(row_wb_cum_argmax_column.i64.nullable());
    const row_wb_cum_argmax = try row_wb_cum_argmax_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cum_argmax);
    const row_wb_cum_argmax_validity = try row_wb_cum_argmax_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cum_argmax_validity);
    const row_quantile_column = try result.column("row_quantile");
    try std.testing.expect(row_quantile_column.f64.nullable());
    const row_quantile = try row_quantile_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_quantile);
    const row_quantile_validity = try row_quantile_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_quantile_validity);
    const row_quantile_range_column = try result.column("row_quantile_range");
    try std.testing.expect(row_quantile_range_column.f64.nullable());
    const row_quantile_range = try row_quantile_range_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_quantile_range);
    const row_quantile_range_validity = try row_quantile_range_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_quantile_range_validity);
    const row_trimmed_mean_column = try result.column("row_trimmed_mean");
    try std.testing.expect(row_trimmed_mean_column.f64.nullable());
    const row_trimmed_mean = try row_trimmed_mean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_trimmed_mean);
    const row_trimmed_mean_validity = try row_trimmed_mean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_trimmed_mean_validity);
    const row_winsorized_mean_column = try result.column("row_winsorized_mean");
    try std.testing.expect(row_winsorized_mean_column.f64.nullable());
    const row_winsorized_mean = try row_winsorized_mean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_winsorized_mean);
    const row_winsorized_mean_validity = try row_winsorized_mean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_winsorized_mean_validity);
    const row_median_column = try result.column("row_median");
    try std.testing.expect(row_median_column.f64.nullable());
    const row_median = try row_median_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_median);
    const row_median_validity = try row_median_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_median_validity);
    const row_iqr_column = try result.column("row_iqr");
    try std.testing.expect(row_iqr_column.f64.nullable());
    const row_iqr = try row_iqr_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_iqr);
    const row_iqr_validity = try row_iqr_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_iqr_validity);
    const row_idr_column = try result.column("row_idr");
    try std.testing.expect(row_idr_column.f64.nullable());
    const row_idr = try row_idr_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_idr);
    const row_idr_validity = try row_idr_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_idr_validity);
    const row_trimean_column = try result.column("row_trimean");
    try std.testing.expect(row_trimean_column.f64.nullable());
    const row_trimean = try row_trimean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_trimean);
    const row_trimean_validity = try row_trimean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_trimean_validity);
    const row_bowley_column = try result.column("row_bowley");
    try std.testing.expect(row_bowley_column.f64.nullable());
    const row_bowley = try row_bowley_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_bowley);
    const row_bowley_validity = try row_bowley_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_bowley_validity);
    const row_qcd_column = try result.column("row_qcd");
    try std.testing.expect(row_qcd_column.f64.nullable());
    const row_qcd = try row_qcd_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_qcd);
    const row_qcd_validity = try row_qcd_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_qcd_validity);
    const row_kelley_column = try result.column("row_kelley");
    try std.testing.expect(row_kelley_column.f64.nullable());
    const row_kelley = try row_kelley_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_kelley);
    const row_kelley_validity = try row_kelley_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_kelley_validity);
    const row_mad_column = try result.column("row_mad");
    try std.testing.expect(row_mad_column.f64.nullable());
    const row_mad = try row_mad_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mad);
    const row_mad_validity = try row_mad_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mad_validity);
    const row_mode_column = try result.column("row_mode");
    try std.testing.expect(row_mode_column.f64.nullable());
    const row_mode = try row_mode_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mode);
    const row_mode_validity = try row_mode_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mode_validity);
    const row_entropy_column = try result.column("row_entropy");
    try std.testing.expect(row_entropy_column.f64.nullable());
    const row_entropy = try row_entropy_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_entropy);
    const row_entropy_validity = try row_entropy_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_entropy_validity);
    const row_gini_column = try result.column("row_gini");
    try std.testing.expect(row_gini_column.f64.nullable());
    const row_gini = try row_gini_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_gini);
    const row_gini_validity = try row_gini_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_gini_validity);
    const row_perplexity_column = try result.column("row_perplexity");
    try std.testing.expect(row_perplexity_column.f64.nullable());
    const row_perplexity = try row_perplexity_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_perplexity);
    const row_perplexity_validity = try row_perplexity_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_perplexity_validity);
    const row_inverse_simpson_column = try result.column("row_inverse_simpson");
    try std.testing.expect(row_inverse_simpson_column.f64.nullable());
    const row_inverse_simpson = try row_inverse_simpson_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_inverse_simpson);
    const row_inverse_simpson_validity = try row_inverse_simpson_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_inverse_simpson_validity);
    const row_concentration = try (try result.column("row_concentration")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_concentration);
    const row_evenness = try (try result.column("row_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_evenness);
    const row_mode_count_column = try result.column("row_mode_count");
    try std.testing.expect(row_mode_count_column.i64.nullable());
    const row_mode_count = try row_mode_count_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_mode_count);
    const row_mode_count_validity = try row_mode_count_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mode_count_validity);
    const row_mode_ratio_column = try result.column("row_mode_ratio");
    try std.testing.expect(row_mode_ratio_column.f64.nullable());
    const row_mode_ratio = try row_mode_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mode_ratio);
    const row_mode_ratio_validity = try row_mode_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mode_ratio_validity);
    const row_mode_margin = try (try result.column("row_mode_margin")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_mode_margin);
    const row_pair_count = try (try result.column("row_pair_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_pair_count);
    const row_weighted_pair_weight_sum = try (try result.column("row_weighted_pair_weight_sum")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_pair_weight_sum);
    const row_weighted_pair_positive_count = try (try result.column("row_weighted_pair_positive_count")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_pair_positive_count);
    const row_weighted_pair_effective_n = try (try result.column("row_weighted_pair_effective_n")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_pair_effective_n);
    const row_weighted_mean_column = try result.column("row_weighted_mean");
    try std.testing.expect(row_weighted_mean_column.f64.nullable());
    const row_weighted_mean = try row_weighted_mean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean);
    const row_weighted_mean_validity = try row_weighted_mean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean_validity);
    const row_weighted_sum_column = try result.column("row_weighted_sum");
    try std.testing.expect(row_weighted_sum_column.f64.nullable());
    const row_weighted_sum = try row_weighted_sum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_sum);
    const row_weighted_sum_validity = try row_weighted_sum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_sum_validity);
    const a_row_weighted_cumsum_column = try result.column("a_row_weighted_cumsum");
    try std.testing.expect(a_row_weighted_cumsum_column.f64.nullable());
    const a_row_weighted_cumsum = try a_row_weighted_cumsum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(a_row_weighted_cumsum);
    const a_row_weighted_cumsum_validity = try a_row_weighted_cumsum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(a_row_weighted_cumsum_validity);
    const b_row_weighted_cumsum_column = try result.column("b_row_weighted_cumsum");
    try std.testing.expect(b_row_weighted_cumsum_column.f64.nullable());
    const b_row_weighted_cumsum = try b_row_weighted_cumsum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(b_row_weighted_cumsum);
    const b_row_weighted_cumsum_validity = try b_row_weighted_cumsum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(b_row_weighted_cumsum_validity);
    const row_weighted_weight_sum_column = try result.column("row_weighted_weight_sum");
    try std.testing.expect(row_weighted_weight_sum_column.f64.nullable());
    const row_weighted_weight_sum = try row_weighted_weight_sum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_weight_sum);
    const row_weighted_weight_sum_validity = try row_weighted_weight_sum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_weight_sum_validity);
    const row_weighted_positive_count_column = try result.column("row_weighted_positive_count");
    try std.testing.expect(row_weighted_positive_count_column.f64.nullable());
    const row_weighted_positive_count = try row_weighted_positive_count_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_positive_count);
    const row_weighted_positive_count_validity = try row_weighted_positive_count_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_positive_count_validity);
    const row_weighted_effective_n_column = try result.column("row_weighted_effective_n");
    try std.testing.expect(row_weighted_effective_n_column.f64.nullable());
    const row_weighted_effective_n = try row_weighted_effective_n_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_effective_n);
    const row_weighted_effective_n_validity = try row_weighted_effective_n_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_effective_n_validity);
    const row_weighted_mean_square_column = try result.column("row_weighted_mean_square");
    try std.testing.expect(row_weighted_mean_square_column.f64.nullable());
    const row_weighted_mean_square = try row_weighted_mean_square_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean_square);
    const row_weighted_mean_square_validity = try row_weighted_mean_square_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean_square_validity);
    const row_weighted_rms_column = try result.column("row_weighted_rms");
    try std.testing.expect(row_weighted_rms_column.f64.nullable());
    const row_weighted_rms = try row_weighted_rms_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_rms);
    const row_weighted_rms_validity = try row_weighted_rms_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_rms_validity);
    const row_weighted_mean_abs_column = try result.column("row_weighted_mean_abs");
    try std.testing.expect(row_weighted_mean_abs_column.f64.nullable());
    const row_weighted_mean_abs = try row_weighted_mean_abs_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean_abs);
    const row_weighted_mean_abs_validity = try row_weighted_mean_abs_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean_abs_validity);
    const row_weighted_l1_column = try result.column("row_weighted_l1");
    try std.testing.expect(row_weighted_l1_column.f64.nullable());
    const row_weighted_l1 = try row_weighted_l1_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_l1);
    const row_weighted_l1_validity = try row_weighted_l1_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_l1_validity);
    const row_weighted_l2_column = try result.column("row_weighted_l2");
    try std.testing.expect(row_weighted_l2_column.f64.nullable());
    const row_weighted_l2 = try row_weighted_l2_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_l2);
    const row_weighted_l2_validity = try row_weighted_l2_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_l2_validity);
    const row_weighted_min = try (try result.column("row_weighted_min")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_min);
    const row_weighted_max = try (try result.column("row_weighted_max")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_max);
    const row_weighted_max_abs = try (try result.column("row_weighted_max_abs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_max_abs);
    const row_weighted_min_abs = try (try result.column("row_weighted_min_abs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_min_abs);
    const row_weighted_range_column = try result.column("row_weighted_range");
    try std.testing.expect(row_weighted_range_column.f64.nullable());
    const row_weighted_range = try row_weighted_range_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_range);
    const row_weighted_range_validity = try row_weighted_range_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_range_validity);
    const row_weighted_midrange = try (try result.column("row_weighted_midrange")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_midrange);
    const row_weighted_range_coeff = try (try result.column("row_weighted_range_coeff")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_range_coeff);
    const row_weighted_product = try (try result.column("row_weighted_product")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_product);
    const row_weighted_geo = try (try result.column("row_weighted_geo")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_geo);
    const row_weighted_harmonic = try (try result.column("row_weighted_harmonic")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_harmonic);
    const row_weighted_logsumexp = try (try result.column("row_weighted_logsumexp")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_logsumexp);
    const row_weighted_logmeanexp = try (try result.column("row_weighted_logmeanexp")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_logmeanexp);
    const row_weighted_quantile_column = try result.column("row_weighted_quantile");
    try std.testing.expect(row_weighted_quantile_column.f64.nullable());
    const row_weighted_quantile = try row_weighted_quantile_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_quantile);
    const row_weighted_quantile_validity = try row_weighted_quantile_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_quantile_validity);
    const row_weighted_median_column = try result.column("row_weighted_median");
    try std.testing.expect(row_weighted_median_column.f64.nullable());
    const row_weighted_median = try row_weighted_median_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_median);
    const row_weighted_median_validity = try row_weighted_median_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_median_validity);
    const row_weighted_iqr_column = try result.column("row_weighted_iqr");
    try std.testing.expect(row_weighted_iqr_column.f64.nullable());
    const row_weighted_iqr = try row_weighted_iqr_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_iqr);
    const row_weighted_iqr_validity = try row_weighted_iqr_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_iqr_validity);
    const row_weighted_mad_column = try result.column("row_weighted_mad");
    try std.testing.expect(row_weighted_mad_column.f64.nullable());
    const row_weighted_mad = try row_weighted_mad_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mad);
    const row_weighted_mad_validity = try row_weighted_mad_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mad_validity);
    const row_weighted_trimmed_column = try result.column("row_weighted_trimmed");
    try std.testing.expect(row_weighted_trimmed_column.f64.nullable());
    const row_weighted_trimmed = try row_weighted_trimmed_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_trimmed);
    const row_weighted_trimmed_validity = try row_weighted_trimmed_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_trimmed_validity);
    const row_weighted_winsorized_column = try result.column("row_weighted_winsorized");
    try std.testing.expect(row_weighted_winsorized_column.f64.nullable());
    const row_weighted_winsorized = try row_weighted_winsorized_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_winsorized);
    const row_weighted_winsorized_validity = try row_weighted_winsorized_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_winsorized_validity);
    const row_weighted_idr_column = try result.column("row_weighted_idr");
    try std.testing.expect(row_weighted_idr_column.f64.nullable());
    const row_weighted_idr = try row_weighted_idr_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_idr);
    const row_weighted_idr_validity = try row_weighted_idr_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_idr_validity);
    const row_weighted_midhinge_column = try result.column("row_weighted_midhinge");
    try std.testing.expect(row_weighted_midhinge_column.f64.nullable());
    const row_weighted_midhinge = try row_weighted_midhinge_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_midhinge);
    const row_weighted_midhinge_validity = try row_weighted_midhinge_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_midhinge_validity);
    const row_weighted_trimean_column = try result.column("row_weighted_trimean");
    try std.testing.expect(row_weighted_trimean_column.f64.nullable());
    const row_weighted_trimean = try row_weighted_trimean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_trimean);
    const row_weighted_trimean_validity = try row_weighted_trimean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_trimean_validity);
    const row_weighted_bowley_column = try result.column("row_weighted_bowley");
    try std.testing.expect(row_weighted_bowley_column.f64.nullable());
    const row_weighted_bowley = try row_weighted_bowley_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_bowley);
    const row_weighted_bowley_validity = try row_weighted_bowley_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_bowley_validity);
    const row_weighted_qcd_column = try result.column("row_weighted_qcd");
    try std.testing.expect(row_weighted_qcd_column.f64.nullable());
    const row_weighted_qcd = try row_weighted_qcd_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_qcd);
    const row_weighted_qcd_validity = try row_weighted_qcd_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_qcd_validity);
    const row_weighted_kelley_column = try result.column("row_weighted_kelley");
    try std.testing.expect(row_weighted_kelley_column.f64.nullable());
    const row_weighted_kelley = try row_weighted_kelley_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_kelley);
    const row_weighted_kelley_validity = try row_weighted_kelley_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_kelley_validity);
    const row_weighted_mode_column = try result.column("row_weighted_mode");
    try std.testing.expect(row_weighted_mode_column.f64.nullable());
    const row_weighted_mode = try row_weighted_mode_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode);
    const row_weighted_mode_validity = try row_weighted_mode_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode_validity);
    const row_weighted_mode_weight = try (try result.column("row_weighted_mode_weight")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode_weight);
    const row_weighted_mode_ratio = try (try result.column("row_weighted_mode_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode_ratio);
    const row_weighted_mode_margin = try (try result.column("row_weighted_mode_margin")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mode_margin);
    const row_weighted_entropy = try (try result.column("row_weighted_entropy")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_entropy);
    const row_weighted_gini = try (try result.column("row_weighted_gini")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_gini);
    const row_weighted_perplexity = try (try result.column("row_weighted_perplexity")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_perplexity);
    const row_weighted_inverse = try (try result.column("row_weighted_inverse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_inverse);
    const row_weighted_concentration = try (try result.column("row_weighted_concentration")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_concentration);
    const row_weighted_evenness = try (try result.column("row_weighted_evenness")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_evenness);
    const row_weighted_mean_abs_dev_column = try result.column("row_weighted_mean_abs_dev");
    try std.testing.expect(row_weighted_mean_abs_dev_column.f64.nullable());
    const row_weighted_mean_abs_dev = try row_weighted_mean_abs_dev_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean_abs_dev);
    const row_weighted_mean_abs_dev_validity = try row_weighted_mean_abs_dev_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean_abs_dev_validity);
    const row_weighted_mad_ratio_column = try result.column("row_weighted_mad_ratio");
    try std.testing.expect(row_weighted_mad_ratio_column.f64.nullable());
    const row_weighted_mad_ratio = try row_weighted_mad_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mad_ratio);
    const row_weighted_mad_ratio_validity = try row_weighted_mad_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mad_ratio_validity);
    const row_weighted_gini_mean_diff_column = try result.column("row_weighted_gini_mean_diff");
    try std.testing.expect(row_weighted_gini_mean_diff_column.f64.nullable());
    const row_weighted_gini_mean_diff = try row_weighted_gini_mean_diff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_gini_mean_diff);
    const row_weighted_gini_mean_diff_validity = try row_weighted_gini_mean_diff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_gini_mean_diff_validity);
    const row_weighted_gini_coeff_column = try result.column("row_weighted_gini_coeff");
    try std.testing.expect(row_weighted_gini_coeff_column.f64.nullable());
    const row_weighted_gini_coeff = try row_weighted_gini_coeff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_gini_coeff);
    const row_weighted_gini_coeff_validity = try row_weighted_gini_coeff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_gini_coeff_validity);
    const row_weighted_variance_column = try result.column("row_weighted_variance");
    try std.testing.expect(row_weighted_variance_column.f64.nullable());
    const row_weighted_variance = try row_weighted_variance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_variance);
    const row_weighted_variance_validity = try row_weighted_variance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_variance_validity);
    const row_weighted_stddev_column = try result.column("row_weighted_stddev");
    try std.testing.expect(row_weighted_stddev_column.f64.nullable());
    const row_weighted_stddev = try row_weighted_stddev_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_stddev);
    const row_weighted_stddev_validity = try row_weighted_stddev_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_stddev_validity);
    const row_weighted_sem_column = try result.column("row_weighted_sem");
    try std.testing.expect(row_weighted_sem_column.f64.nullable());
    const row_weighted_sem = try row_weighted_sem_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_sem);
    const row_weighted_sem_validity = try row_weighted_sem_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_sem_validity);
    const row_weighted_cv_column = try result.column("row_weighted_cv");
    try std.testing.expect(row_weighted_cv_column.f64.nullable());
    const row_weighted_cv = try row_weighted_cv_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_cv);
    const row_weighted_cv_validity = try row_weighted_cv_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_cv_validity);
    const row_weighted_fano_column = try result.column("row_weighted_fano");
    try std.testing.expect(row_weighted_fano_column.f64.nullable());
    const row_weighted_fano = try row_weighted_fano_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_fano);
    const row_weighted_fano_validity = try row_weighted_fano_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_fano_validity);
    const row_weighted_skew = try (try result.column("row_weighted_skew")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_skew);
    const row_weighted_kurt = try (try result.column("row_weighted_kurt")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_kurt);
    const row_weighted_covariance_column = try result.column("row_weighted_covariance");
    try std.testing.expect(row_weighted_covariance_column.f64.nullable());
    const row_weighted_covariance = try row_weighted_covariance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_covariance);
    const row_weighted_covariance_validity = try row_weighted_covariance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_covariance_validity);
    const row_weighted_correlation_column = try result.column("row_weighted_correlation");
    try std.testing.expect(row_weighted_correlation_column.f64.nullable());
    const row_weighted_correlation = try row_weighted_correlation_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_correlation);
    const row_weighted_correlation_validity = try row_weighted_correlation_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_correlation_validity);
    const row_weighted_beta_column = try result.column("row_weighted_beta");
    try std.testing.expect(row_weighted_beta_column.f64.nullable());
    const row_weighted_beta = try row_weighted_beta_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_beta);
    const row_weighted_beta_validity = try row_weighted_beta_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_beta_validity);
    const row_weighted_dot = try (try result.column("row_weighted_dot")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_dot);
    const row_weighted_cosine = try (try result.column("row_weighted_cosine")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_cosine);
    const row_weighted_sqdist = try (try result.column("row_weighted_sqdist")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_sqdist);
    const row_weighted_euclidean = try (try result.column("row_weighted_euclidean")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_euclidean);
    const row_weighted_manhattan = try (try result.column("row_weighted_manhattan")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_manhattan);
    const row_weighted_chebyshev = try (try result.column("row_weighted_chebyshev")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_chebyshev);
    const row_weighted_canberra = try (try result.column("row_weighted_canberra")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_canberra);
    const row_weighted_bray = try (try result.column("row_weighted_bray")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_bray);
    const row_weighted_bias = try (try result.column("row_weighted_bias")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_bias);
    const row_weighted_mae = try (try result.column("row_weighted_mae")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mae);
    const row_weighted_mse = try (try result.column("row_weighted_mse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mse);
    const row_weighted_rmse = try (try result.column("row_weighted_rmse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_rmse);
    const row_weighted_mape = try (try result.column("row_weighted_mape")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mape);
    const row_weighted_smape = try (try result.column("row_weighted_smape")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_smape);
    const row_dot_column = try result.column("row_dot");
    try std.testing.expect(row_dot_column.f64.nullable());
    const row_dot = try row_dot_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_dot);
    const row_dot_validity = try row_dot_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_dot_validity);
    const row_cosine_column = try result.column("row_cosine");
    try std.testing.expect(row_cosine_column.f64.nullable());
    const row_cosine = try row_cosine_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_cosine);
    const row_cosine_validity = try row_cosine_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_cosine_validity);
    const row_sqdist = try (try result.column("row_sqdist")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_sqdist);
    const row_sqdist_validity = try (try result.column("row_sqdist")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_sqdist_validity);
    const row_euclidean = try (try result.column("row_euclidean")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_euclidean);
    const row_manhattan = try (try result.column("row_manhattan")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_manhattan);
    const row_chebyshev = try (try result.column("row_chebyshev")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_chebyshev);
    const row_canberra = try (try result.column("row_canberra")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_canberra);
    const row_bray = try (try result.column("row_bray")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_bray);
    const row_mean_error_column = try result.column("row_mean_error");
    try std.testing.expect(row_mean_error_column.f64.nullable());
    const row_mean_error = try row_mean_error_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mean_error);
    const row_mean_error_validity = try row_mean_error_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mean_error_validity);
    const row_mae = try (try result.column("row_mae")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_mae);
    const row_mse = try (try result.column("row_mse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_mse);
    const row_rmse = try (try result.column("row_rmse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_rmse);
    const row_mape_column = try result.column("row_mape");
    try std.testing.expect(row_mape_column.f64.nullable());
    const row_mape = try row_mape_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mape);
    const row_mape_validity = try row_mape_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mape_validity);
    const row_smape_column = try result.column("row_smape");
    try std.testing.expect(row_smape_column.f64.nullable());
    const row_smape = try row_smape_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_smape);
    const row_smape_validity = try row_smape_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_smape_validity);
    const row_covariance_column = try result.column("row_covariance");
    try std.testing.expect(row_covariance_column.f64.nullable());
    const row_covariance = try row_covariance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_covariance);
    const row_covariance_validity = try row_covariance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_covariance_validity);
    const row_correlation_column = try result.column("row_correlation");
    try std.testing.expect(row_correlation_column.f64.nullable());
    const row_correlation = try row_correlation_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_correlation);
    const row_correlation_validity = try row_correlation_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_correlation_validity);
    const row_beta_column = try result.column("row_beta");
    try std.testing.expect(row_beta_column.f64.nullable());
    const row_beta = try row_beta_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_beta);
    const row_beta_validity = try row_beta_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_beta_validity);
    const row_distinct = try (try result.column("row_distinct")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_distinct);
    const row_unique = try (try result.column("row_unique")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_unique);
    const row_b_cummode = try (try result.column("row_b_cummode")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummode);
    const row_wb_cummode = try (try result.column("row_wb_cummode")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummode);
    const row_b_cummode_count = try (try result.column("row_b_cummode_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummode_count);
    const row_wb_cummode_ratio = try (try result.column("row_wb_cummode_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummode_ratio);
    const row_b_cummode_margin = try (try result.column("row_b_cummode_margin")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummode_margin);
    const row_wb_cummode_margin_ratio = try (try result.column("row_wb_cummode_margin_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummode_margin_ratio);
    const row_b_cumdistinct = try (try result.column("row_b_cumdistinct")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumdistinct);
    const row_wb_cumunique = try (try result.column("row_wb_cumunique")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumunique);
    const row_sum_column = try result.column("row_sum");
    try std.testing.expect(row_sum_column.f64.nullable());
    const row_sum = try row_sum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_sum);
    const row_sum_validity = try row_sum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_sum_validity);
    const row_mean_column = try result.column("row_mean");
    try std.testing.expect(row_mean_column.f64.nullable());
    const row_mean = try row_mean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mean);
    const row_mean_validity = try row_mean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mean_validity);
    const row_logsumexp_column = try result.column("row_logsumexp");
    try std.testing.expect(row_logsumexp_column.f64.nullable());
    const row_logsumexp = try row_logsumexp_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_logsumexp);
    const row_logsumexp_validity = try row_logsumexp_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_logsumexp_validity);
    const row_logmeanexp_column = try result.column("row_logmeanexp");
    try std.testing.expect(row_logmeanexp_column.f64.nullable());
    const row_logmeanexp = try row_logmeanexp_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_logmeanexp);
    const row_logmeanexp_validity = try row_logmeanexp_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_logmeanexp_validity);
    const row_a_centered_column = try result.column("row_a_centered");
    try std.testing.expect(row_a_centered_column.f64.nullable());
    const row_a_centered = try row_a_centered_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_centered);
    const row_a_centered_validity = try row_a_centered_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_centered_validity);
    const row_b_centered_column = try result.column("row_b_centered");
    try std.testing.expect(row_b_centered_column.f64.nullable());
    const row_b_centered = try row_b_centered_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_centered);
    const row_b_centered_validity = try row_b_centered_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_centered_validity);
    const row_a_zscore_column = try result.column("row_a_zscore");
    try std.testing.expect(row_a_zscore_column.f64.nullable());
    const row_a_zscore = try row_a_zscore_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_zscore);
    const row_a_zscore_validity = try row_a_zscore_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_zscore_validity);
    const row_b_zscore_column = try result.column("row_b_zscore");
    try std.testing.expect(row_b_zscore_column.f64.nullable());
    const row_b_zscore = try row_b_zscore_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_zscore);
    const row_b_zscore_validity = try row_b_zscore_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_zscore_validity);
    const row_a_dense_rank_column = try result.column("row_a_dense_rank");
    try std.testing.expect(row_a_dense_rank_column.i64.nullable());
    const row_a_dense_rank = try row_a_dense_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_a_dense_rank);
    const row_a_dense_rank_validity = try row_a_dense_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_dense_rank_validity);
    const row_b_dense_rank_column = try result.column("row_b_dense_rank");
    try std.testing.expect(row_b_dense_rank_column.i64.nullable());
    const row_b_dense_rank = try row_b_dense_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_dense_rank);
    const row_b_dense_rank_validity = try row_b_dense_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_dense_rank_validity);
    const row_a_ordinal_rank_column = try result.column("row_a_ordinal_rank");
    try std.testing.expect(row_a_ordinal_rank_column.i64.nullable());
    const row_a_ordinal_rank = try row_a_ordinal_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_a_ordinal_rank);
    const row_a_ordinal_rank_validity = try row_a_ordinal_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_ordinal_rank_validity);
    const row_b_ordinal_rank_column = try result.column("row_b_ordinal_rank");
    try std.testing.expect(row_b_ordinal_rank_column.i64.nullable());
    const row_b_ordinal_rank = try row_b_ordinal_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_ordinal_rank);
    const row_b_ordinal_rank_validity = try row_b_ordinal_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_ordinal_rank_validity);
    const row_a_average_rank_column = try result.column("row_a_average_rank");
    try std.testing.expect(row_a_average_rank_column.f64.nullable());
    const row_a_average_rank = try row_a_average_rank_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_average_rank);
    const row_a_average_rank_validity = try row_a_average_rank_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_average_rank_validity);
    const row_b_average_rank_column = try result.column("row_b_average_rank");
    try std.testing.expect(row_b_average_rank_column.f64.nullable());
    const row_b_average_rank = try row_b_average_rank_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_average_rank);
    const row_b_average_rank_validity = try row_b_average_rank_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_average_rank_validity);
    const row_a_competition_rank_column = try result.column("row_a_competition_rank");
    try std.testing.expect(row_a_competition_rank_column.i64.nullable());
    const row_a_competition_rank = try row_a_competition_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_a_competition_rank);
    const row_a_competition_rank_validity = try row_a_competition_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_competition_rank_validity);
    const row_b_competition_rank_column = try result.column("row_b_competition_rank");
    try std.testing.expect(row_b_competition_rank_column.i64.nullable());
    const row_b_competition_rank = try row_b_competition_rank_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_b_competition_rank);
    const row_b_competition_rank_validity = try row_b_competition_rank_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_competition_rank_validity);
    const row_a_percent_rank_column = try result.column("row_a_percent_rank");
    try std.testing.expect(row_a_percent_rank_column.f64.nullable());
    const row_a_percent_rank = try row_a_percent_rank_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_percent_rank);
    const row_a_percent_rank_validity = try row_a_percent_rank_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_percent_rank_validity);
    const row_b_percent_rank_column = try result.column("row_b_percent_rank");
    try std.testing.expect(row_b_percent_rank_column.f64.nullable());
    const row_b_percent_rank = try row_b_percent_rank_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_percent_rank);
    const row_b_percent_rank_validity = try row_b_percent_rank_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_percent_rank_validity);
    const row_a_cume_column = try result.column("row_a_cume");
    try std.testing.expect(row_a_cume_column.f64.nullable());
    const row_a_cume = try row_a_cume_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cume);
    const row_a_cume_validity = try row_a_cume_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cume_validity);
    const row_b_cume_column = try result.column("row_b_cume");
    try std.testing.expect(row_b_cume_column.f64.nullable());
    const row_b_cume = try row_b_cume_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cume);
    const row_b_cume_validity = try row_b_cume_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cume_validity);
    const row_a_cumsum_column = try result.column("row_a_cumsum");
    try std.testing.expect(row_a_cumsum_column.f64.nullable());
    const row_a_cumsum = try row_a_cumsum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumsum);
    const row_a_cumsum_validity = try row_a_cumsum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumsum_validity);
    const row_b_cumsum_column = try result.column("row_b_cumsum");
    try std.testing.expect(row_b_cumsum_column.f64.nullable());
    const row_b_cumsum = try row_b_cumsum_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumsum);
    const row_b_cumsum_validity = try row_b_cumsum_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumsum_validity);
    const row_a_cummean_column = try result.column("row_a_cummean");
    try std.testing.expect(row_a_cummean_column.f64.nullable());
    const row_a_cummean = try row_a_cummean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummean);
    const row_a_cummean_validity = try row_a_cummean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummean_validity);
    const row_b_cummean_column = try result.column("row_b_cummean");
    try std.testing.expect(row_b_cummean_column.f64.nullable());
    const row_b_cummean = try row_b_cummean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummean);
    const row_b_cummean_validity = try row_b_cummean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummean_validity);
    const row_b_cumlse = try (try result.column("row_b_cumlse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumlse);
    const row_wb_cumlse = try (try result.column("row_wb_cumlse")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumlse);
    const row_wb_cumlme = try (try result.column("row_wb_cumlme")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumlme);
    const row_wb_cumgeo = try (try result.column("row_wb_cumgeo")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumgeo);
    const row_wb_cumharm = try (try result.column("row_wb_cumharm")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumharm);
    const row_b_cumvar_column = try result.column("row_b_cumvar");
    try std.testing.expect(row_b_cumvar_column.f64.nullable());
    const row_b_cumvar = try row_b_cumvar_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumvar);
    const row_b_cumvar_validity = try row_b_cumvar_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumvar_validity);
    const row_wb_cumvar = try (try result.column("row_wb_cumvar")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumvar);
    const row_b_cumstd_column = try result.column("row_b_cumstd");
    try std.testing.expect(row_b_cumstd_column.f64.nullable());
    const row_b_cumstd = try row_b_cumstd_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumstd);
    const row_b_cumstd_validity = try row_b_cumstd_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumstd_validity);
    const row_wb_cumstd = try (try result.column("row_wb_cumstd")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumstd);
    const row_wb_cumsem = try (try result.column("row_wb_cumsem")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumsem);
    const row_wb_cumcv = try (try result.column("row_wb_cumcv")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumcv);
    const row_wb_cumfano = try (try result.column("row_wb_cumfano")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumfano);
    const row_wb_cumskew = try (try result.column("row_wb_cumskew")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumskew);
    const row_wb_cumkurt = try (try result.column("row_wb_cumkurt")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumkurt);
    const row_b_cumrms = try (try result.column("row_b_cumrms")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumrms);
    const row_wb_cumrms = try (try result.column("row_wb_cumrms")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumrms);
    const row_b_cummeanabs_column = try result.column("row_b_cummeanabs");
    try std.testing.expect(row_b_cummeanabs_column.f64.nullable());
    const row_b_cummeanabs = try row_b_cummeanabs_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummeanabs);
    const row_b_cummeanabs_validity = try row_b_cummeanabs_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummeanabs_validity);
    const row_wb_cummeanabs = try (try result.column("row_wb_cummeanabs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummeanabs);
    const row_b_cummeansq = try (try result.column("row_b_cummeansq")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummeansq);
    const row_wb_cummeansq = try (try result.column("row_wb_cummeansq")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummeansq);
    const row_b_cummaxabs_column = try result.column("row_b_cummaxabs");
    try std.testing.expect(row_b_cummaxabs_column.f64.nullable());
    const row_b_cummaxabs = try row_b_cummaxabs_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummaxabs);
    const row_b_cummaxabs_validity = try row_b_cummaxabs_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummaxabs_validity);
    const row_wb_cummaxabs = try (try result.column("row_wb_cummaxabs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cummaxabs);
    const row_wb_cumminabs = try (try result.column("row_wb_cumminabs")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cumminabs);
    const row_wb_cuml1 = try (try result.column("row_wb_cuml1")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cuml1);
    const row_wb_cuml2 = try (try result.column("row_wb_cuml2")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_wb_cuml2);
    const row_a_cumprod_column = try result.column("row_a_cumprod");
    try std.testing.expect(row_a_cumprod_column.f64.nullable());
    const row_a_cumprod = try row_a_cumprod_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumprod);
    const row_a_cumprod_validity = try row_a_cumprod_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumprod_validity);
    const row_b_cumprod_column = try result.column("row_b_cumprod");
    try std.testing.expect(row_b_cumprod_column.f64.nullable());
    const row_b_cumprod = try row_b_cumprod_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumprod);
    const row_b_cumprod_validity = try row_b_cumprod_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumprod_validity);
    const row_a_cummax_column = try result.column("row_a_cummax");
    try std.testing.expect(row_a_cummax_column.f64.nullable());
    const row_a_cummax = try row_a_cummax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummax);
    const row_a_cummax_validity = try row_a_cummax_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummax_validity);
    const row_b_cummax_column = try result.column("row_b_cummax");
    try std.testing.expect(row_b_cummax_column.f64.nullable());
    const row_b_cummax = try row_b_cummax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummax);
    const row_b_cummax_validity = try row_b_cummax_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummax_validity);
    const row_a_cummin_column = try result.column("row_a_cummin");
    try std.testing.expect(row_a_cummin_column.f64.nullable());
    const row_a_cummin = try row_a_cummin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummin);
    const row_a_cummin_validity = try row_a_cummin_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cummin_validity);
    const row_b_cummin_column = try result.column("row_b_cummin");
    try std.testing.expect(row_b_cummin_column.f64.nullable());
    const row_b_cummin = try row_b_cummin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummin);
    const row_b_cummin_validity = try row_b_cummin_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cummin_validity);
    const row_a_cumrange_column = try result.column("row_a_cumrange");
    try std.testing.expect(row_a_cumrange_column.f64.nullable());
    const row_a_cumrange = try row_a_cumrange_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumrange);
    const row_a_cumrange_validity = try row_a_cumrange_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_cumrange_validity);
    const row_b_cumrange_column = try result.column("row_b_cumrange");
    try std.testing.expect(row_b_cumrange_column.f64.nullable());
    const row_b_cumrange = try row_b_cumrange_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumrange);
    const row_b_cumrange_validity = try row_b_cumrange_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_cumrange_validity);
    const row_a_robust_zscore_column = try result.column("row_a_robust_zscore");
    try std.testing.expect(row_a_robust_zscore_column.f64.nullable());
    const row_a_robust_zscore = try row_a_robust_zscore_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_robust_zscore);
    const row_a_robust_zscore_validity = try row_a_robust_zscore_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_robust_zscore_validity);
    const row_b_robust_zscore_column = try result.column("row_b_robust_zscore");
    try std.testing.expect(row_b_robust_zscore_column.f64.nullable());
    const row_b_robust_zscore = try row_b_robust_zscore_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_robust_zscore);
    const row_b_robust_zscore_validity = try row_b_robust_zscore_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_robust_zscore_validity);
    const row_b_iqr_outlier_column = try result.column("row_b_iqr_outlier");
    try std.testing.expect(row_b_iqr_outlier_column.bool.nullable());
    const row_b_iqr_outlier = try row_b_iqr_outlier_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_b_iqr_outlier);
    const row_b_iqr_outlier_validity = try row_b_iqr_outlier_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_iqr_outlier_validity);
    const row_b_tukey_winsor_column = try result.column("row_b_tukey_winsor");
    try std.testing.expect(row_b_tukey_winsor_column.f64.nullable());
    const row_b_tukey_winsor = try row_b_tukey_winsor_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_tukey_winsor);
    const row_b_tukey_winsor_validity = try row_b_tukey_winsor_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_tukey_winsor_validity);
    const row_a_is_max_column = try result.column("row_a_is_max");
    try std.testing.expect(row_a_is_max_column.bool.nullable());
    const row_a_is_max = try row_a_is_max_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_a_is_max);
    const row_a_is_max_validity = try row_a_is_max_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_is_max_validity);
    const row_b_is_max_column = try result.column("row_b_is_max");
    try std.testing.expect(row_b_is_max_column.bool.nullable());
    const row_b_is_max = try row_b_is_max_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_b_is_max);
    const row_b_is_max_validity = try row_b_is_max_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_is_max_validity);
    const row_a_is_min_column = try result.column("row_a_is_min");
    try std.testing.expect(row_a_is_min_column.bool.nullable());
    const row_a_is_min = try row_a_is_min_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_a_is_min);
    const row_a_is_min_validity = try row_a_is_min_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_is_min_validity);
    const row_b_is_min_column = try result.column("row_b_is_min");
    try std.testing.expect(row_b_is_min_column.bool.nullable());
    const row_b_is_min = try row_b_is_min_column.bool.toOwnedSlice(gpa);
    defer gpa.free(row_b_is_min);
    const row_b_is_min_validity = try row_b_is_min_column.bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_is_min_validity);
    const row_a_minmax_column = try result.column("row_a_minmax");
    try std.testing.expect(row_a_minmax_column.f64.nullable());
    const row_a_minmax = try row_a_minmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_minmax);
    const row_b_minmax_column = try result.column("row_b_minmax");
    try std.testing.expect(row_b_minmax_column.f64.nullable());
    const row_b_minmax = try row_b_minmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_minmax);
    const row_a_l2_column = try result.column("row_a_l2_unit");
    try std.testing.expect(row_a_l2_column.f64.nullable());
    const row_a_l2 = try row_a_l2_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_l2);
    const row_b_l2_column = try result.column("row_b_l2_unit");
    try std.testing.expect(row_b_l2_column.f64.nullable());
    const row_b_l2 = try row_b_l2_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_l2);
    const row_a_l1_column = try result.column("row_a_l1_unit");
    try std.testing.expect(row_a_l1_column.f64.nullable());
    const row_a_l1 = try row_a_l1_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_l1);
    const row_b_l1_column = try result.column("row_b_l1_unit");
    try std.testing.expect(row_b_l1_column.f64.nullable());
    const row_b_l1 = try row_b_l1_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_l1);
    const row_a_share_column = try result.column("row_a_share");
    try std.testing.expect(row_a_share_column.f64.nullable());
    const row_a_share = try row_a_share_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_share);
    const row_b_share_column = try result.column("row_b_share");
    try std.testing.expect(row_b_share_column.f64.nullable());
    const row_b_share = try row_b_share_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_share);
    const row_a_mean_ratio_column = try result.column("row_a_mean_ratio");
    try std.testing.expect(row_a_mean_ratio_column.f64.nullable());
    const row_a_mean_ratio = try row_a_mean_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_mean_ratio);
    const row_a_mean_ratio_validity = try row_a_mean_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_mean_ratio_validity);
    const row_b_mean_ratio_column = try result.column("row_b_mean_ratio");
    try std.testing.expect(row_b_mean_ratio_column.f64.nullable());
    const row_b_mean_ratio = try row_b_mean_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_mean_ratio);
    const row_b_mean_ratio_validity = try row_b_mean_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_mean_ratio_validity);
    const row_a_maxabs_column = try result.column("row_a_maxabs");
    try std.testing.expect(row_a_maxabs_column.f64.nullable());
    const row_a_maxabs = try row_a_maxabs_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_maxabs);
    const row_b_maxabs_column = try result.column("row_b_maxabs");
    try std.testing.expect(row_b_maxabs_column.f64.nullable());
    const row_b_maxabs = try row_b_maxabs_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_maxabs);
    const row_a_softmax_column = try result.column("row_a_softmax");
    try std.testing.expect(row_a_softmax_column.f64.nullable());
    const row_a_softmax = try row_a_softmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_softmax);
    const row_a_softmax_validity = try row_a_softmax_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_a_softmax_validity);
    const row_b_softmax_column = try result.column("row_b_softmax");
    try std.testing.expect(row_b_softmax_column.f64.nullable());
    const row_b_softmax = try row_b_softmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_softmax);
    const row_b_softmax_validity = try row_b_softmax_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_b_softmax_validity);
    const row_a_log_softmax_column = try result.column("row_a_log_softmax");
    try std.testing.expect(row_a_log_softmax_column.f64.nullable());
    const row_a_log_softmax = try row_a_log_softmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_log_softmax);
    const row_b_log_softmax_column = try result.column("row_b_log_softmax");
    try std.testing.expect(row_b_log_softmax_column.f64.nullable());
    const row_b_log_softmax = try row_b_log_softmax_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_log_softmax);
    const row_a_softmin_column = try result.column("row_a_softmin");
    try std.testing.expect(row_a_softmin_column.f64.nullable());
    const row_a_softmin = try row_a_softmin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_softmin);
    const row_b_softmin_column = try result.column("row_b_softmin");
    try std.testing.expect(row_b_softmin_column.f64.nullable());
    const row_b_softmin = try row_b_softmin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_softmin);
    const row_a_log_softmin_column = try result.column("row_a_log_softmin");
    try std.testing.expect(row_a_log_softmin_column.f64.nullable());
    const row_a_log_softmin = try row_a_log_softmin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_a_log_softmin);
    const row_b_log_softmin_column = try result.column("row_b_log_softmin");
    try std.testing.expect(row_b_log_softmin_column.f64.nullable());
    const row_b_log_softmin = try row_b_log_softmin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_b_log_softmin);
    const row_softmax_entropy_column = try result.column("row_softmax_entropy");
    try std.testing.expect(row_softmax_entropy_column.f64.nullable());
    const row_softmax_entropy = try row_softmax_entropy_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_entropy);
    const row_softmax_entropy_validity = try row_softmax_entropy_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_entropy_validity);
    const row_softmax_perplexity_column = try result.column("row_softmax_perplexity");
    try std.testing.expect(row_softmax_perplexity_column.f64.nullable());
    const row_softmax_perplexity = try row_softmax_perplexity_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_perplexity);
    const row_softmax_confidence_column = try result.column("row_softmax_confidence");
    try std.testing.expect(row_softmax_confidence_column.f64.nullable());
    const row_softmax_confidence = try row_softmax_confidence_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_confidence);
    const row_softmax_margin_column = try result.column("row_softmax_margin");
    try std.testing.expect(row_softmax_margin_column.f64.nullable());
    const row_softmax_margin = try row_softmax_margin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_margin);
    const row_softmax_evenness_column = try result.column("row_softmax_evenness");
    try std.testing.expect(row_softmax_evenness_column.f64.nullable());
    const row_softmax_evenness = try row_softmax_evenness_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_evenness);
    const row_softmax_concentration_column = try result.column("row_softmax_concentration");
    try std.testing.expect(row_softmax_concentration_column.f64.nullable());
    const row_softmax_concentration = try row_softmax_concentration_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_concentration);
    const row_softmax_normalized_hhi_column = try result.column("row_softmax_normalized_hhi");
    try std.testing.expect(row_softmax_normalized_hhi_column.f64.nullable());
    const row_softmax_normalized_hhi = try row_softmax_normalized_hhi_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_normalized_hhi);
    const row_softmax_gini_column = try result.column("row_softmax_gini");
    try std.testing.expect(row_softmax_gini_column.f64.nullable());
    const row_softmax_gini = try row_softmax_gini_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_gini);
    const row_softmax_inverse_column = try result.column("row_softmax_inverse");
    try std.testing.expect(row_softmax_inverse_column.f64.nullable());
    const row_softmax_inverse = try row_softmax_inverse_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_inverse);
    const row_softmax_simpson_evenness_column = try result.column("row_softmax_simpson_evenness");
    try std.testing.expect(row_softmax_simpson_evenness_column.f64.nullable());
    const row_softmax_simpson_evenness = try row_softmax_simpson_evenness_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_softmax_simpson_evenness);
    const row_logit_margin_column = try result.column("row_logit_margin");
    try std.testing.expect(row_logit_margin_column.f64.nullable());
    const row_logit_margin = try row_logit_margin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_logit_margin);
    const row_geo_column = try result.column("row_geo");
    try std.testing.expect(row_geo_column.f64.nullable());
    const row_geo = try row_geo_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_geo);
    const row_geo_validity = try row_geo_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_geo_validity);
    const row_magnitude_geo_column = try result.column("row_magnitude_geo");
    try std.testing.expect(row_magnitude_geo_column.f64.nullable());
    const row_magnitude_geo = try row_magnitude_geo_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_geo);
    const row_magnitude_geo_validity = try row_magnitude_geo_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_geo_validity);
    const row_harm_column = try result.column("row_harm");
    try std.testing.expect(row_harm_column.f64.nullable());
    const row_harm = try row_harm_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_harm);
    const row_harm_validity = try row_harm_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_harm_validity);
    const row_skew_column = try result.column("row_skew");
    try std.testing.expect(row_skew_column.f64.nullable());
    const row_skew = try row_skew_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_skew);
    const row_skew_validity = try row_skew_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_skew_validity);
    const row_magnitude_skew = try (try result.column("row_magnitude_skew")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_skew);
    const row_kurt_column = try result.column("row_kurt");
    try std.testing.expect(row_kurt_column.f64.nullable());
    const row_kurt = try row_kurt_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_kurt);
    const row_kurt_validity = try row_kurt_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_kurt_validity);
    const row_magnitude_kurt = try (try result.column("row_magnitude_kurt")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_kurt);
    const row_prod_column = try result.column("row_prod");
    try std.testing.expect(row_prod_column.f64.nullable());
    const row_prod = try row_prod_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_prod);
    const row_prod_validity = try row_prod_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_prod_validity);
    const row_min_column = try result.column("row_min");
    try std.testing.expect(row_min_column.f64.nullable());
    const row_min = try row_min_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_min);
    const row_min_validity = try row_min_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_min_validity);
    const row_max_column = try result.column("row_max");
    try std.testing.expect(row_max_column.f64.nullable());
    const row_max = try row_max_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_max);
    const row_max_validity = try row_max_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_max_validity);
    const row_ptp_column = try result.column("row_ptp");
    try std.testing.expect(row_ptp_column.f64.nullable());
    const row_ptp = try row_ptp_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_ptp);
    const row_ptp_validity = try row_ptp_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_ptp_validity);
    const row_magnitude_ptp_column = try result.column("row_magnitude_ptp");
    try std.testing.expect(row_magnitude_ptp_column.f64.nullable());
    const row_magnitude_ptp = try row_magnitude_ptp_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_ptp);
    const row_magnitude_ptp_validity = try row_magnitude_ptp_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_ptp_validity);
    const row_magnitude_midrange = try (try result.column("row_magnitude_midrange")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_midrange);
    const row_range_coeff_column = try result.column("row_range_coeff");
    try std.testing.expect(row_range_coeff_column.f64.nullable());
    const row_range_coeff = try row_range_coeff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_range_coeff);
    const row_range_coeff_validity = try row_range_coeff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_range_coeff_validity);
    const row_magnitude_range_coeff_column = try result.column("row_magnitude_range_coeff");
    try std.testing.expect(row_magnitude_range_coeff_column.f64.nullable());
    const row_magnitude_range_coeff = try row_magnitude_range_coeff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_range_coeff);
    const row_magnitude_range_coeff_validity = try row_magnitude_range_coeff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_range_coeff_validity);
    const row_mean_abs_column = try result.column("row_mean_abs");
    try std.testing.expect(row_mean_abs_column.f64.nullable());
    const row_mean_abs = try row_mean_abs_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mean_abs);
    const row_mean_abs_validity = try row_mean_abs_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mean_abs_validity);
    const row_hhi_column = try result.column("row_hhi");
    try std.testing.expect(row_hhi_column.f64.nullable());
    const row_hhi = try row_hhi_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_hhi);
    const row_hhi_validity = try row_hhi_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_hhi_validity);
    const row_magnitude_normalized_hhi_column = try result.column("row_magnitude_normalized_hhi");
    try std.testing.expect(row_magnitude_normalized_hhi_column.f64.nullable());
    const row_magnitude_normalized_hhi = try row_magnitude_normalized_hhi_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_normalized_hhi);
    const row_magnitude_normalized_hhi_validity = try row_magnitude_normalized_hhi_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_normalized_hhi_validity);
    const row_magnitude_sparsity_column = try result.column("row_magnitude_sparsity");
    try std.testing.expect(row_magnitude_sparsity_column.f64.nullable());
    const row_magnitude_sparsity = try row_magnitude_sparsity_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_sparsity);
    const row_magnitude_sparsity_validity = try row_magnitude_sparsity_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_sparsity_validity);
    const row_magnitude_inverse_column = try result.column("row_magnitude_inverse");
    try std.testing.expect(row_magnitude_inverse_column.f64.nullable());
    const row_magnitude_inverse = try row_magnitude_inverse_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_inverse);
    const row_magnitude_inverse_validity = try row_magnitude_inverse_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_inverse_validity);
    const row_magnitude_simpson_evenness_column = try result.column("row_magnitude_simpson_evenness");
    try std.testing.expect(row_magnitude_simpson_evenness_column.f64.nullable());
    const row_magnitude_simpson_evenness = try row_magnitude_simpson_evenness_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_simpson_evenness);
    const row_magnitude_simpson_evenness_validity = try row_magnitude_simpson_evenness_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_simpson_evenness_validity);
    const row_magnitude_dominance_column = try result.column("row_magnitude_dominance");
    try std.testing.expect(row_magnitude_dominance_column.f64.nullable());
    const row_magnitude_dominance = try row_magnitude_dominance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_dominance);
    const row_magnitude_dominance_validity = try row_magnitude_dominance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_dominance_validity);
    const row_magnitude_margin_column = try result.column("row_magnitude_margin");
    try std.testing.expect(row_magnitude_margin_column.f64.nullable());
    const row_magnitude_margin = try row_magnitude_margin_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_margin);
    const row_magnitude_margin_validity = try row_magnitude_margin_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_margin_validity);
    const row_magnitude_entropy_column = try result.column("row_magnitude_entropy");
    try std.testing.expect(row_magnitude_entropy_column.f64.nullable());
    const row_magnitude_entropy = try row_magnitude_entropy_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_entropy);
    const row_magnitude_entropy_validity = try row_magnitude_entropy_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_entropy_validity);
    const row_magnitude_perplexity_column = try result.column("row_magnitude_perplexity");
    try std.testing.expect(row_magnitude_perplexity_column.f64.nullable());
    const row_magnitude_perplexity = try row_magnitude_perplexity_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_perplexity);
    const row_magnitude_perplexity_validity = try row_magnitude_perplexity_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_perplexity_validity);
    const row_magnitude_evenness_column = try result.column("row_magnitude_evenness");
    try std.testing.expect(row_magnitude_evenness_column.f64.nullable());
    const row_magnitude_evenness = try row_magnitude_evenness_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_evenness);
    const row_magnitude_evenness_validity = try row_magnitude_evenness_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_evenness_validity);
    const row_mean_abs_dev_column = try result.column("row_mean_abs_dev");
    try std.testing.expect(row_mean_abs_dev_column.f64.nullable());
    const row_mean_abs_dev = try row_mean_abs_dev_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mean_abs_dev);
    const row_mean_abs_dev_validity = try row_mean_abs_dev_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mean_abs_dev_validity);
    const row_gini_mean_diff_column = try result.column("row_gini_mean_diff");
    try std.testing.expect(row_gini_mean_diff_column.f64.nullable());
    const row_gini_mean_diff = try row_gini_mean_diff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_gini_mean_diff);
    const row_gini_mean_diff_validity = try row_gini_mean_diff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_gini_mean_diff_validity);
    const row_gini_coeff_column = try result.column("row_gini_coeff");
    try std.testing.expect(row_gini_coeff_column.f64.nullable());
    const row_gini_coeff = try row_gini_coeff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_gini_coeff);
    const row_gini_coeff_validity = try row_gini_coeff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_gini_coeff_validity);
    const row_mad_ratio_column = try result.column("row_mad_ratio");
    try std.testing.expect(row_mad_ratio_column.f64.nullable());
    const row_mad_ratio = try row_mad_ratio_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_mad_ratio);
    const row_mad_ratio_validity = try row_mad_ratio_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_mad_ratio_validity);
    const row_rms_column = try result.column("row_rms");
    try std.testing.expect(row_rms_column.f64.nullable());
    const row_rms = try row_rms_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_rms);
    const row_rms_validity = try row_rms_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_rms_validity);
    const row_l1_column = try result.column("row_l1");
    try std.testing.expect(row_l1_column.f64.nullable());
    const row_l1 = try row_l1_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_l1);
    const row_l1_validity = try row_l1_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_l1_validity);
    const row_l2_column = try result.column("row_l2");
    try std.testing.expect(row_l2_column.f64.nullable());
    const row_l2 = try row_l2_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_l2);
    const row_l2_validity = try row_l2_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_l2_validity);
    const row_variance_column = try result.column("row_variance");
    try std.testing.expect(row_variance_column.f64.nullable());
    const row_variance = try row_variance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_variance);
    const row_variance_validity = try row_variance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_variance_validity);
    const row_magnitude_variance_column = try result.column("row_magnitude_variance");
    try std.testing.expect(row_magnitude_variance_column.f64.nullable());
    const row_magnitude_variance = try row_magnitude_variance_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_variance);
    const row_magnitude_variance_validity = try row_magnitude_variance_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_variance_validity);
    const row_stddev_column = try result.column("row_stddev");
    try std.testing.expect(row_stddev_column.f64.nullable());
    const row_stddev = try row_stddev_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_stddev);
    const row_stddev_validity = try row_stddev_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_stddev_validity);
    const row_magnitude_stddev_column = try result.column("row_magnitude_stddev");
    try std.testing.expect(row_magnitude_stddev_column.f64.nullable());
    const row_magnitude_stddev = try row_magnitude_stddev_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_stddev);
    const row_magnitude_stddev_validity = try row_magnitude_stddev_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_stddev_validity);
    const row_sem_column = try result.column("row_sem");
    try std.testing.expect(row_sem_column.f64.nullable());
    const row_sem = try row_sem_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_sem);
    const row_sem_validity = try row_sem_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_sem_validity);
    const row_magnitude_sem_column = try result.column("row_magnitude_sem");
    try std.testing.expect(row_magnitude_sem_column.f64.nullable());
    const row_magnitude_sem = try row_magnitude_sem_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_sem);
    const row_magnitude_sem_validity = try row_magnitude_sem_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_sem_validity);
    const row_cv_column = try result.column("row_cv");
    try std.testing.expect(row_cv_column.f64.nullable());
    const row_cv = try row_cv_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_cv);
    const row_cv_validity = try row_cv_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_cv_validity);
    const row_magnitude_cv_column = try result.column("row_magnitude_cv");
    try std.testing.expect(row_magnitude_cv_column.f64.nullable());
    const row_magnitude_cv = try row_magnitude_cv_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_cv);
    const row_magnitude_cv_validity = try row_magnitude_cv_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_cv_validity);
    const row_magnitude_fano_column = try result.column("row_magnitude_fano");
    try std.testing.expect(row_magnitude_fano_column.f64.nullable());
    const row_magnitude_fano = try row_magnitude_fano_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_fano);
    const row_magnitude_fano_validity = try row_magnitude_fano_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_magnitude_fano_validity);
    const row_fano_column = try result.column("row_fano");
    try std.testing.expect(row_fano_column.f64.nullable());
    const row_fano = try row_fano_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_fano);
    const row_fano_validity = try row_fano_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_fano_validity);

    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, row_argmin);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_argmin_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1 }, row_argmax);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_argmax_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, row_b_cum_argmin);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_b_cum_argmin_validity);
    try std.testing.expectEqualSlices(i64, &.{ 3, 1, 3, 1 }, row_wb_cum_argmax);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_wb_cum_argmax_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 13.0 }, row_quantile);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_quantile_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_quantile_range[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_quantile_range[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_quantile_range[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 21.6), row_quantile_range[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_quantile_range_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_trimmed_mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_trimmed_mean[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_trimmed_mean[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 22.0), row_trimmed_mean[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_trimmed_mean_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_winsorized_mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_winsorized_mean[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_winsorized_mean[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 22.0), row_winsorized_mean[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_winsorized_mean_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_median);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_median_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_iqr);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_iqr_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_idr[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_idr[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_idr[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 28.8), row_idr[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_idr_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_trimean);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_trimean_validity);
    try std.testing.expect(std.math.isNan(row_bowley[0]));
    try std.testing.expect(std.math.isNan(row_bowley[1]));
    try std.testing.expectEqual(@as(f64, 0.0), row_bowley[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_bowley[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_bowley_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_qcd[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_qcd[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_qcd[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 22.0), row_qcd[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_qcd_validity);
    try std.testing.expect(std.math.isNan(row_kelley[0]));
    try std.testing.expect(std.math.isNan(row_kelley[1]));
    try std.testing.expectEqual(@as(f64, 0.0), row_kelley[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_kelley[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_kelley_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_mad);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mad_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_mode);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mode_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_entropy[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, @as(f64, 2.0)), row_entropy[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_entropy[2], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))), row_entropy[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_entropy_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_gini[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), row_gini[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_gini[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 9.0), row_gini[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_gini_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_perplexity[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), row_perplexity[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_perplexity[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0)))), row_perplexity[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_perplexity_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_inverse_simpson[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), row_inverse_simpson[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_inverse_simpson[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8), row_inverse_simpson[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_inverse_simpson_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_concentration[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), row_concentration[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_concentration[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), row_concentration[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_evenness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_evenness[2], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))) / std.math.log(f64, std.math.e, @as(f64, 2.0)), row_evenness[3], 1e-12);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 2 }, row_mode_count);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_mode_count_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_mode_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), row_mode_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_mode_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), row_mode_ratio[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_mode_ratio_validity);
    try std.testing.expectEqualSlices(i64, &.{ 2, 0, 1, 1 }, row_mode_margin);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 2 }, row_pair_count);
    try expectF64SliceApproxOrNan(&.{ 1.0, 1.0, 0.0, 5.0 }, row_weighted_pair_weight_sum);
    try expectF64SliceApproxOrNan(&.{ 1.0, 1.0, 0.0, 2.0 }, row_weighted_pair_positive_count);
    try expectF64SliceApproxOrNan(&.{ 1.0, 1.0, 0.0, 25.0 / 17.0 }, row_weighted_pair_effective_n);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_weighted_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 56.0 / 5.0), row_weighted_mean[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mean_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 56.0 }, row_weighted_sum);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_sum_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 0.0, 0.0, 16.0 }, a_row_weighted_cumsum);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, a_row_weighted_cumsum_validity);
    try expectF64SliceApproxOrNan(&.{ 0.0, 20.0, 0.0, 56.0 }, b_row_weighted_cumsum);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, b_row_weighted_cumsum_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 1.0, 0.0, 5.0 }, row_weighted_weight_sum);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_weight_sum_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 1.0, 0.0, 2.0 }, row_weighted_positive_count);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_positive_count_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 1.0, 0.0, 25.0 / 17.0 }, row_weighted_effective_n);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_effective_n_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 400.0, 0.0, 1664.0 / 5.0 }, row_weighted_mean_square);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mean_square_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, std.math.sqrt(@as(f64, 1664.0 / 5.0)) }, row_weighted_rms);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_rms_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 56.0 / 5.0 }, row_weighted_mean_abs);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mean_abs_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 56.0 }, row_weighted_l1);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_l1_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, std.math.sqrt(@as(f64, 1664.0)) }, row_weighted_l2);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_l2_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_min);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 40.0 }, row_weighted_max);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 40.0 }, row_weighted_max_abs);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_min_abs);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 36.0 }, row_weighted_range);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_range_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 22.0 }, row_weighted_midrange);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 9.0 / 11.0 }, row_weighted_range_coeff);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, std.math.exp(4.0 * std.math.log(f64, std.math.e, @as(f64, 4.0)) + std.math.log(f64, std.math.e, @as(f64, 40.0))) }, row_weighted_product);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, std.math.exp((4.0 * std.math.log(f64, std.math.e, @as(f64, 4.0)) + std.math.log(f64, std.math.e, @as(f64, 40.0))) / 5.0) }, row_weighted_geo);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 5.0 / (4.0 / 4.0 + 1.0 / 40.0) }, row_weighted_harmonic);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 40.0 + std.math.log1p(@as(f64, 4.0) * std.math.exp(@as(f64, -36.0))) }, row_weighted_logsumexp);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 40.0 + std.math.log1p(@as(f64, 4.0) * std.math.exp(@as(f64, -36.0))) - std.math.log(f64, std.math.e, @as(f64, 5.0)) }, row_weighted_logmeanexp);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 40.0 }, row_weighted_quantile);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_quantile_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_median);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_median_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, row_weighted_iqr);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_iqr_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, row_weighted_mad);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mad_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_trimmed);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_trimmed_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_winsorized);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_winsorized_validity);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 36.0 }, row_weighted_idr);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_idr_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_midhinge);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_midhinge_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_trimean);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_trimean_validity);
    try expectF64SliceApproxOrNan(&.{ std.math.nan(f64), std.math.nan(f64), 0.0, std.math.nan(f64) }, row_weighted_bowley);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_bowley_validity);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 0.0 }, row_weighted_qcd);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_qcd_validity);
    try expectF64SliceApproxOrNan(&.{ std.math.nan(f64), std.math.nan(f64), 0.0, 1.0 }, row_weighted_kelley);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_kelley_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 3.0, 40.0 }, row_weighted_mode);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_weighted_mode_validity);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 2.0, 5.0, 4.0 }, row_weighted_mode_weight);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_mode_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), row_weighted_mode_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_mode_ratio[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), row_weighted_mode_ratio[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 1.0, 5.0, 2.0 }, row_weighted_mode_margin);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_entropy[0], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))), row_weighted_entropy[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_entropy[2], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))), row_weighted_entropy[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_gini[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 9.0), row_weighted_gini[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_gini[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 9.0), row_weighted_gini[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_perplexity[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0)))), row_weighted_perplexity[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_perplexity[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0)))), row_weighted_perplexity[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_inverse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8), row_weighted_inverse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_inverse[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.8), row_weighted_inverse[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_concentration[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), row_weighted_concentration[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_concentration[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 9.0), row_weighted_concentration[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))) / std.math.log(f64, std.math.e, @as(f64, 2.0)), row_weighted_evenness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_evenness[2], 1e-12);
    try std.testing.expectApproxEqAbs(-(@as(f64, 2.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 2.0 / 3.0)) + @as(f64, 1.0 / 3.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 3.0))) / std.math.log(f64, std.math.e, @as(f64, 2.0)), row_weighted_evenness[3], 1e-12);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 11.52 }, row_weighted_mean_abs_dev);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mean_abs_dev_validity);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 36.0 / 35.0 }, row_weighted_mad_ratio);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mad_ratio_validity);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 36.0 }, row_weighted_gini_mean_diff);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_gini_mean_diff_validity);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 45.0 / 28.0 }, row_weighted_gini_coeff);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_gini_coeff_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_variance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 207.36), row_weighted_variance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_variance_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_stddev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 207.36)), row_weighted_stddev[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_stddev_validity);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, std.math.sqrt(@as(f64, 207.36 / 5.0)) }, row_weighted_sem);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_sem_validity);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 9.0 / 7.0 }, row_weighted_cv);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_cv_validity);
    try expectF64SliceApproxOrNan(&.{ 0.0, 0.0, 0.0, 648.0 / 35.0 }, row_weighted_fano);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_fano_validity);
    const lazy_row_weighted_skew3 = std.math.sqrt(@as(f64, 5.0)) * @as(f64, 22394.88) / std.math.pow(f64, @as(f64, 1036.8), 1.5);
    try expectF64SliceApproxOrNan(&.{ std.math.nan(f64), std.math.nan(f64), 0.0, lazy_row_weighted_skew3 }, row_weighted_skew);
    const lazy_row_weighted_kurt3 = @as(f64, 5.0) * @as(f64, 698720.256) / (@as(f64, 1036.8) * @as(f64, 1036.8)) - 3.0;
    try expectF64SliceApproxOrNan(&.{ std.math.nan(f64), std.math.nan(f64), 0.0, lazy_row_weighted_kurt3 }, row_weighted_kurt);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_covariance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_covariance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_covariance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -17.28), row_weighted_covariance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_covariance_validity);
    try std.testing.expect(std.math.isNan(row_weighted_correlation[0]));
    try std.testing.expect(std.math.isNan(row_weighted_correlation[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_correlation[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), row_weighted_correlation[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_correlation_validity);
    try std.testing.expect(std.math.isNan(row_weighted_beta[0]));
    try std.testing.expect(std.math.isNan(row_weighted_beta[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_beta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 12.0), row_weighted_beta[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_beta_validity);
    try expectF64SliceApproxOrNan(&.{ 1.0, 20.0, 0.0, 104.0 }, row_weighted_dot);
    try expectF64SliceApproxOrNan(&.{ 1.0, 1.0, 0.0, 104.0 / (std.math.sqrt(@as(f64, 1664.0)) * std.math.sqrt(@as(f64, 65.0))) }, row_weighted_cosine);
    try expectF64SliceApproxOrNan(&.{ 0.0, 361.0, 0.0, 1521.0 }, row_weighted_sqdist);
    try expectF64SliceApproxOrNan(&.{ 0.0, 19.0, 0.0, 39.0 }, row_weighted_euclidean);
    try expectF64SliceApproxOrNan(&.{ 0.0, 19.0, 0.0, 39.0 }, row_weighted_manhattan);
    try expectF64SliceApproxOrNan(&.{ 0.0, 19.0, 0.0, 39.0 }, row_weighted_chebyshev);
    try expectF64SliceApproxOrNan(&.{ 0.0, 19.0 / 21.0, 0.0, 39.0 / 41.0 }, row_weighted_canberra);
    try expectF64SliceApproxOrNan(&.{ 0.0, 19.0 / 21.0, 0.0, 39.0 / 73.0 }, row_weighted_bray);
    try expectF64SliceApproxOrNan(&.{ 0.0, 19.0, 0.0, 39.0 / 5.0 }, row_weighted_bias);
    try expectF64SliceApproxOrNan(&.{ 0.0, 19.0, 0.0, 39.0 / 5.0 }, row_weighted_mae);
    try expectF64SliceApproxOrNan(&.{ 0.0, 361.0, 0.0, 1521.0 / 5.0 }, row_weighted_mse);
    try expectF64SliceApproxOrNan(&.{ 0.0, 19.0, 0.0, std.math.sqrt(@as(f64, 1521.0 / 5.0)) }, row_weighted_rmse);
    try expectF64SliceApproxOrNan(&.{ 0.0, 19.0 / 20.0, 0.0, 39.0 / 200.0 }, row_weighted_mape);
    try expectF64SliceApproxOrNan(&.{ 0.0, 38.0 / 21.0, 0.0, 78.0 / 205.0 }, row_weighted_smape);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 56.0 }, row_dot);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_dot_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_cosine[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_cosine[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_cosine[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 56.0) / (std.math.sqrt(@as(f64, 1616.0)) * std.math.sqrt(@as(f64, 17.0))), row_cosine[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_cosine_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 361.0, 0.0, 1521.0 }, row_sqdist);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_sqdist_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_euclidean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0), row_euclidean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_euclidean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 39.0), row_euclidean[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 19.0, 0.0, 39.0 }, row_manhattan);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 19.0, 0.0, 39.0 }, row_chebyshev);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_canberra[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0 / 21.0), row_canberra[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_canberra[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 39.0 / 41.0), row_canberra[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_bray[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0 / 21.0), row_bray[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_bray[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 39.0 / 49.0), row_bray[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 19.0, 0.0, 19.5 }, row_mean_error);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mean_error_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 19.0, 0.0, 19.5 }, row_mae);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 361.0, 0.0, 760.5 }, row_mse);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_rmse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0), row_rmse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_rmse[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 760.5)), row_rmse[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_mape[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0 / 20.0), row_mape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_mape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 39.0 / 80.0), row_mape[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mape_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_smape[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 38.0 / 21.0), row_smape[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_smape[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 39.0 / 41.0), row_smape[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_smape_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_covariance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_covariance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_covariance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -27.0), row_covariance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_covariance_validity);
    try std.testing.expect(std.math.isNan(row_correlation[0]));
    try std.testing.expect(std.math.isNan(row_correlation[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_correlation[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), row_correlation[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_correlation_validity);
    try std.testing.expect(std.math.isNan(row_beta[0]));
    try std.testing.expect(std.math.isNan(row_beta[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_beta[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0 / 12.0), row_beta[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_beta_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 2 }, row_distinct);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 2 }, row_unique);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_b_cummode);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 3.0, 4.0 }, row_wb_cummode);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 1 }, row_b_cummode_count);
    try std.testing.expectEqualSlices(f64, &.{ 2.0 / 3.0, 1.0 / 3.0, 0.5, 0.5 }, row_wb_cummode_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 0 }, row_b_cummode_margin);
    try std.testing.expectEqualSlices(f64, &.{ 1.0 / 3.0, 0.0, 0.0, 0.25 }, row_wb_cummode_margin_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 2 }, row_b_cumdistinct);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3, 2, 3 }, row_wb_cumunique);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 44.0 }, row_sum);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_sum_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_mean);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mean_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_logsumexp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_logsumexp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_logsumexp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) + std.math.log1p(std.math.exp(@as(f64, -36.0))), row_logsumexp[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_logsumexp_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_logmeanexp[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_logmeanexp[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_logmeanexp[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) + std.math.log1p(std.math.exp(@as(f64, -36.0))) - std.math.ln2, row_logmeanexp[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_logmeanexp_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, -18.0 }, row_a_centered);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_b_centered);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_centered_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_centered_validity);
    try std.testing.expect(std.math.isNan(row_a_zscore[0]));
    try std.testing.expect(std.math.isNan(row_b_zscore[1]));
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), row_a_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_zscore_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_zscore_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 2 }, row_a_dense_rank);
    try std.testing.expectEqualSlices(i64, &.{ 0, 3, 0, 3 }, row_b_dense_rank);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_dense_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_dense_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 2 }, row_a_ordinal_rank);
    try std.testing.expectEqualSlices(i64, &.{ 0, 3, 0, 4 }, row_b_ordinal_rank);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_ordinal_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_ordinal_rank_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), row_a_average_rank[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), row_a_average_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), row_b_average_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), row_b_average_rank[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_average_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_average_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 2 }, row_a_competition_rank);
    try std.testing.expectEqualSlices(i64, &.{ 0, 3, 0, 4 }, row_b_competition_rank);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_competition_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_competition_rank_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_a_percent_rank[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), row_a_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_percent_rank[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_percent_rank_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_percent_rank_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), row_a_cume[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), row_a_cume[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_cume[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_cume[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cume_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cume_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cumsum);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 44.0 }, row_b_cumsum);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cumsum_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumsum_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cummean);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 22.0 }, row_b_cummean);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cummean_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummean_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_b_cumlse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) + std.math.log1p(std.math.exp(@as(f64, -36.0))), row_b_cumlse[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0) + std.math.log1p(@as(f64, 2.0) * std.math.exp(@as(f64, -1.0))), row_wb_cumlse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0) + std.math.log1p(std.math.exp(@as(f64, -18.0)) + std.math.exp(@as(f64, -19.0))), row_wb_cumlse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0) + std.math.log1p(std.math.exp(@as(f64, -2.0))), row_wb_cumlse[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) + std.math.log1p(@as(f64, 2.0) * std.math.exp(@as(f64, -36.0)) + std.math.exp(@as(f64, -39.0))), row_wb_cumlse[3], 1e-12);
    try std.testing.expectApproxEqAbs(row_wb_cumlse[0] - std.math.log(f64, std.math.e, 3.0), row_wb_cumlme[0], 1e-12);
    try std.testing.expectApproxEqAbs(row_wb_cumlse[1] - std.math.log(f64, std.math.e, 3.0), row_wb_cumlme[1], 1e-12);
    try std.testing.expectApproxEqAbs(row_wb_cumlse[2] - std.math.ln2, row_wb_cumlme[2], 1e-12);
    try std.testing.expectApproxEqAbs(row_wb_cumlse[3] - std.math.log(f64, std.math.e, 4.0), row_wb_cumlme[3], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pow(f64, 2.0, 1.0 / 3.0), row_wb_cumgeo[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pow(f64, 40.0, 1.0 / 3.0), row_wb_cumgeo[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 15.0)), row_wb_cumgeo[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.pow(f64, 640.0, 0.25), row_wb_cumgeo[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2), row_wb_cumharm[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 60.0 / 31.0), row_wb_cumharm[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0 / 8.0), row_wb_cumharm[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 160.0 / 61.0), row_wb_cumharm[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 324.0 }, row_b_cumvar);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 9.0), row_wb_cumvar[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 686.0 / 9.0), row_wb_cumvar[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_wb_cumvar[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4131.0 / 16.0), row_wb_cumvar[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumvar_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_b_cumstd);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0 / 9.0)), row_wb_cumstd[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 686.0 / 9.0)), row_wb_cumstd[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_wb_cumstd[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 4131.0 / 16.0)), row_wb_cumstd[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumstd_validity);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0 / 9.0)) / std.math.sqrt(@as(f64, 3.0)), row_wb_cumsem[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 686.0 / 9.0)) / std.math.sqrt(@as(f64, 3.0)), row_wb_cumsem[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / std.math.sqrt(@as(f64, 2.0)), row_wb_cumsem[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 4131.0 / 16.0)) / 2.0, row_wb_cumsem[3], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0 / 9.0)) / @as(f64, 4.0 / 3.0), row_wb_cumcv[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 686.0 / 9.0)) / @as(f64, 23.0 / 3.0), row_wb_cumcv[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), row_wb_cumcv[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 4131.0 / 16.0)) / @as(f64, 12.25), row_wb_cumcv[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), row_wb_cumfano[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 686.0 / 69.0), row_wb_cumfano[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), row_wb_cumfano[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4131.0 / 196.0), row_wb_cumfano[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7071067811865479), row_wb_cumskew[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7001554400787792), row_wb_cumskew[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_wb_cumskew[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.134257375254061), row_wb_cumskew[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), row_wb_cumkurt[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), row_wb_cumkurt[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), row_wb_cumkurt[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.6812479530664843), row_wb_cumkurt[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, std.math.sqrt(@as(f64, 808.0)) }, row_b_cumrms);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 2.0)), row_wb_cumrms[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 135.0)), row_wb_cumrms[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 17.0)), row_wb_cumrms[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 1633.0 / 4.0)), row_wb_cumrms[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 22.0 }, row_b_cummeanabs);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummeanabs_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 3.0), row_wb_cummeanabs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 23.0 / 3.0), row_wb_cummeanabs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), row_wb_cummeanabs[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 49.0 / 4.0), row_wb_cummeanabs[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 400.0, 0.0, 808.0 }, row_b_cummeansq);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), row_wb_cummeansq[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 135.0), row_wb_cummeansq[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 17.0), row_wb_cummeansq[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1633.0 / 4.0), row_wb_cummeansq[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 40.0 }, row_b_cummaxabs);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummaxabs_validity);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 20.0, 5.0, 40.0 }, row_wb_cummaxabs);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 3.0, 1.0 }, row_wb_cumminabs);
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 23.0, 8.0, 49.0 }, row_wb_cuml1);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 6.0)), row_wb_cuml2[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 405.0)), row_wb_cuml2[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 34.0)), row_wb_cuml2[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 1633.0)), row_wb_cuml2[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cumprod);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 160.0 }, row_b_cumprod);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cumprod_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumprod_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cummax);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 40.0 }, row_b_cummax);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cummax_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummax_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 4.0 }, row_a_cummin);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 4.0 }, row_b_cummin);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cummin_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cummin_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, row_a_cumrange);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 36.0 }, row_b_cumrange);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_cumrange_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_cumrange_validity);
    try std.testing.expect(std.math.isNan(row_a_robust_zscore[0]));
    try std.testing.expect(std.math.isNan(row_b_robust_zscore[1]));
    try std.testing.expectApproxEqAbs(-@as(f64, 0.6744897501960817), row_a_robust_zscore[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6744897501960817), row_b_robust_zscore[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_robust_zscore_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_robust_zscore_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, row_b_iqr_outlier);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_iqr_outlier_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 20.0, 0.0, 27.625 }, row_b_tukey_winsor);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_tukey_winsor_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, row_a_is_max);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_is_max);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_is_max_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_is_max_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, row_a_is_min);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, row_b_is_min);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_is_min_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_is_min_validity);
    try std.testing.expect(std.math.isNan(row_a_minmax[0]));
    try std.testing.expect(std.math.isNan(row_b_minmax[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_a_minmax[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_minmax[3], 1e-12);
    const lazy_row3_l2_norm = std.math.sqrt(@as(f64, 1616.0));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_l2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_l2[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0) / lazy_row3_l2_norm, row_a_l2[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 40.0) / lazy_row3_l2_norm, row_b_l2[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_l1[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_l1[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 11.0), row_a_l1[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 11.0), row_b_l1[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_share[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_share[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 11.0), row_a_share[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 11.0), row_b_share[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_mean_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_mean_ratio[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 11.0), row_a_mean_ratio[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0 / 11.0), row_b_mean_ratio[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_mean_ratio_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_mean_ratio_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_maxabs[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_maxabs[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), row_a_maxabs[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_maxabs[3], 1e-12);
    const lazy_row3_a_softmax = std.math.exp(@as(f64, -36.0)) / (@as(f64, 1.0) + std.math.exp(@as(f64, -36.0)));
    const lazy_row3_b_softmax = @as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, -36.0)));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_softmax[0], 1e-12);
    try std.testing.expectApproxEqAbs(lazy_row3_a_softmax, row_a_softmax[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_softmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(lazy_row3_b_softmax, row_b_softmax[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_a_softmax_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, row_b_softmax_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_a_log_softmax[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, lazy_row3_a_softmax), row_a_log_softmax[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_b_log_softmax[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, lazy_row3_b_softmax), row_b_log_softmax[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_a_softmin[0], 1e-12);
    try std.testing.expectApproxEqAbs(lazy_row3_b_softmax, row_a_softmin[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_b_softmin[1], 1e-12);
    try std.testing.expectApproxEqAbs(lazy_row3_a_softmax, row_b_softmin[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_a_log_softmin[0], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, lazy_row3_b_softmax), row_a_log_softmin[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_b_log_softmin[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.log(f64, std.math.e, lazy_row3_a_softmax), row_b_log_softmin[3], 1e-12);
    const lazy_row3_softmax_entropy = -(lazy_row3_a_softmax * std.math.log(f64, std.math.e, lazy_row3_a_softmax) + lazy_row3_b_softmax * std.math.log(f64, std.math.e, lazy_row3_b_softmax));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_softmax_entropy[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_softmax_entropy[1], 1e-12);
    try std.testing.expectApproxEqAbs(lazy_row3_softmax_entropy, row_softmax_entropy[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_softmax_entropy_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_perplexity[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_perplexity[1], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.exp(lazy_row3_softmax_entropy), row_softmax_perplexity[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_confidence[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_confidence[1], 1e-12);
    try std.testing.expectApproxEqAbs(lazy_row3_b_softmax, row_softmax_confidence[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_margin[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_margin[1], 1e-12);
    try std.testing.expectApproxEqAbs(lazy_row3_b_softmax - lazy_row3_a_softmax, row_softmax_margin[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_evenness[1], 1e-12);
    try std.testing.expectApproxEqAbs(lazy_row3_softmax_entropy / std.math.ln2, row_softmax_evenness[3], 1e-12);
    const lazy_row3_concentration = lazy_row3_a_softmax * lazy_row3_a_softmax + lazy_row3_b_softmax * lazy_row3_b_softmax;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_concentration[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_concentration[1], 1e-12);
    try std.testing.expectApproxEqAbs(lazy_row3_concentration, row_softmax_concentration[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_normalized_hhi[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_normalized_hhi[1], 1e-12);
    try std.testing.expectApproxEqAbs((lazy_row3_concentration - 0.5) / 0.5, row_softmax_normalized_hhi[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_softmax_gini[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_softmax_gini[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) - lazy_row3_concentration, row_softmax_gini[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_inverse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_inverse[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / lazy_row3_concentration, row_softmax_inverse[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_simpson_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_softmax_simpson_evenness[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0) / (lazy_row3_concentration * 2.0), row_softmax_simpson_evenness[3], 1e-12);
    try std.testing.expect(std.math.isPositiveInf(row_logit_margin[0]));
    try std.testing.expect(std.math.isPositiveInf(row_logit_margin[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 36.0), row_logit_margin[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_geo[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_geo[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_geo[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 160.0)), row_geo[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_geo_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_geo[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_magnitude_geo[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_geo[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 160.0)), row_magnitude_geo[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_geo_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_harm[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_harm[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_harm[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 80.0 / 11.0), row_harm[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_harm_validity);
    try std.testing.expect(std.math.isNan(row_skew[0]));
    try std.testing.expect(std.math.isNan(row_skew[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_skew[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_skew[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_skew_validity);
    try std.testing.expect(std.math.isNan(row_magnitude_skew[0]));
    try std.testing.expect(std.math.isNan(row_magnitude_skew[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_skew[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_skew[3], 1e-12);
    try std.testing.expect(std.math.isNan(row_kurt[0]));
    try std.testing.expect(std.math.isNan(row_kurt[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_kurt[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), row_kurt[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_kurt_validity);
    try std.testing.expect(std.math.isNan(row_magnitude_kurt[0]));
    try std.testing.expect(std.math.isNan(row_magnitude_kurt[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_kurt[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), row_magnitude_kurt[3], 1e-12);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 160.0 }, row_prod);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_prod_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_min);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_min_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 40.0 }, row_max);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_max_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 36.0 }, row_ptp);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_ptp_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 36.0 }, row_magnitude_ptp);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_ptp_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_magnitude_midrange);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_range_coeff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_range_coeff[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_range_coeff[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_range_coeff[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_range_coeff_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_range_coeff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_range_coeff[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_range_coeff[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_magnitude_range_coeff[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_range_coeff_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_mean_abs);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mean_abs_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_hhi[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_hhi[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_hhi[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 101.0 / 121.0), row_hhi[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_hhi_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_normalized_hhi[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_normalized_hhi[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_normalized_hhi[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 81.0 / 121.0), row_magnitude_normalized_hhi[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_normalized_hhi_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_sparsity[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_sparsity[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_sparsity[2]);
    try std.testing.expectApproxEqAbs((std.math.sqrt(@as(f64, 2.0)) - @as(f64, 11.0) / std.math.sqrt(@as(f64, 101.0))) / (std.math.sqrt(@as(f64, 2.0)) - 1.0), row_magnitude_sparsity[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_sparsity_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_inverse[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_inverse[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_inverse[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 121.0 / 101.0), row_magnitude_inverse[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_inverse_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_simpson_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_simpson_evenness[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_simpson_evenness[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 121.0 / 202.0), row_magnitude_simpson_evenness[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_simpson_evenness_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_dominance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_dominance[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_dominance[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / 11.0), row_magnitude_dominance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_dominance_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_margin[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_margin[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_margin[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_magnitude_margin[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_margin_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_entropy[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_entropy[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_entropy[2]);
    try std.testing.expectApproxEqAbs(-(@as(f64, 1.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 11.0)) + @as(f64, 10.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 10.0 / 11.0))), row_magnitude_entropy[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_entropy_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_perplexity[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_perplexity[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_perplexity[2]);
    try std.testing.expectApproxEqAbs(std.math.exp(-(@as(f64, 1.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 11.0)) + @as(f64, 10.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 10.0 / 11.0)))), row_magnitude_perplexity[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_perplexity_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_evenness[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_magnitude_evenness[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_magnitude_evenness[2]);
    try std.testing.expectApproxEqAbs(-(@as(f64, 1.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 11.0)) + @as(f64, 10.0 / 11.0) * std.math.log(f64, std.math.e, @as(f64, 10.0 / 11.0))) / std.math.log(f64, std.math.e, @as(f64, 2.0)), row_magnitude_evenness[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_evenness_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 18.0 }, row_mean_abs_dev);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mean_abs_dev_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 36.0 }, row_gini_mean_diff);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_gini_mean_diff_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_gini_coeff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_gini_coeff[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_gini_coeff[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_gini_coeff[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_gini_coeff_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_mad_ratio[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_mad_ratio[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_mad_ratio[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_mad_ratio[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mad_ratio_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_rms[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_rms[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_rms[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 808.0)), row_rms[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_rms_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 44.0 }, row_l1);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_l1_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_l2[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_l2[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_l2[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 1616.0)), row_l2[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_l2_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_variance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 324.0), row_variance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_variance_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_variance[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_variance[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_variance[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 324.0), row_magnitude_variance[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_variance_validity);
    try std.testing.expect(std.math.isNan(row_stddev[0]));
    try std.testing.expect(std.math.isNan(row_stddev[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 648.0)), row_stddev[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_stddev_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_stddev[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_stddev[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_stddev[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18.0), row_magnitude_stddev[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_stddev_validity);
    try std.testing.expect(std.math.isNan(row_sem[0]));
    try std.testing.expect(std.math.isNan(row_sem[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_sem[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18.0), row_sem[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_sem_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_sem[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_sem[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_magnitude_sem[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18.0 / std.math.sqrt(@as(f64, 2.0))), row_magnitude_sem[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_magnitude_sem_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_cv[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_cv[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_cv[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18.0 / 22.0), row_cv[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_cv_validity);

    var invalid_pair_count_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_pair_count_plan.deinit();
    try invalid_pair_count_plan.withRowPairCount(&.{"a"}, &.{ "wa", "wb" }, "bad_row_pair_count");
    try std.testing.expectError(error.LengthMismatch, invalid_pair_count_plan.collect());

    var invalid_weighted_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_weighted_plan.deinit();
    try invalid_weighted_plan.withRowWeightedMean(&.{"a"}, &.{ "wa", "wb" }, "bad_row_weighted_mean");
    try std.testing.expectError(error.LengthMismatch, invalid_weighted_plan.collect());

    var invalid_weighted_quantile_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_weighted_quantile_plan.deinit();
    try invalid_weighted_quantile_plan.withRowWeightedQuantile(&.{"a"}, &.{ "wa", "wb" }, "bad_row_weighted_quantile", 0.5);
    try std.testing.expectError(error.LengthMismatch, invalid_weighted_quantile_plan.collect());

    var invalid_weighted_quantile_q_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_weighted_quantile_q_plan.deinit();
    try invalid_weighted_quantile_q_plan.withRowWeightedQuantile(&.{ "a", "b" }, &.{ "wa", "wb" }, "bad_row_weighted_quantile", 1.5);
    try std.testing.expectError(error.InvalidShape, invalid_weighted_quantile_q_plan.collect());

    var invalid_weighted_trimmed_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_weighted_trimmed_plan.deinit();
    try std.testing.expectError(error.InvalidShape, invalid_weighted_trimmed_plan.withRowWeightedTrimmedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "bad_row_weighted_trimmed", 0.5));

    var invalid_weighted_winsorized_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_weighted_winsorized_plan.deinit();
    try std.testing.expectError(error.InvalidShape, invalid_weighted_winsorized_plan.withRowWeightedWinsorizedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "bad_row_weighted_winsorized", -0.01));

    var invalid_weighted_variance_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_weighted_variance_plan.deinit();
    try invalid_weighted_variance_plan.withRowWeightedVariance(&.{"a"}, &.{ "wa", "wb" }, "bad_row_weighted_variance", 0.0);
    try std.testing.expectError(error.LengthMismatch, invalid_weighted_variance_plan.collect());

    var invalid_weighted_correction_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_weighted_correction_plan.deinit();
    try invalid_weighted_correction_plan.withRowWeightedVariance(&.{ "a", "b" }, &.{ "wa", "wb" }, "bad_row_weighted_variance", -1.0);
    try std.testing.expectError(error.InvalidShape, invalid_weighted_correction_plan.collect());

    var invalid_weighted_covariance_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_weighted_covariance_plan.deinit();
    try invalid_weighted_covariance_plan.withRowWeightedCovariance(&.{"a"}, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "bad_row_weighted_covariance", 0.0);
    try std.testing.expectError(error.LengthMismatch, invalid_weighted_covariance_plan.collect());

    var invalid_weighted_covariance_correction_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_weighted_covariance_correction_plan.deinit();
    try invalid_weighted_covariance_correction_plan.withRowWeightedCovariance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "bad_row_weighted_covariance", -1.0);
    try std.testing.expectError(error.InvalidShape, invalid_weighted_covariance_correction_plan.collect());

    var invalid_dot_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_dot_plan.deinit();
    try invalid_dot_plan.withRowDot(&.{"a"}, &.{ "wa", "wb" }, "bad_row_dot");
    try std.testing.expectError(error.LengthMismatch, invalid_dot_plan.collect());

    var invalid_covariance_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_covariance_plan.deinit();
    try invalid_covariance_plan.withRowCovariance(&.{"a"}, &.{ "wa", "wb" }, "bad_row_covariance");
    try std.testing.expectError(error.LengthMismatch, invalid_covariance_plan.collect());

    var invalid_quantile_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_quantile_plan.deinit();
    try invalid_quantile_plan.withRowQuantile(&.{ "a", "b" }, "bad_row_quantile", 1.5);
    try std.testing.expectError(error.InvalidShape, invalid_quantile_plan.collect());

    var invalid_quantile_range_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_quantile_range_plan.deinit();
    try invalid_quantile_range_plan.withRowQuantileRange(&.{ "a", "b" }, "bad_row_quantile_range", 0.8, 0.2);
    try std.testing.expectError(error.InvalidShape, invalid_quantile_range_plan.collect());

    var invalid_trimmed_mean_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_trimmed_mean_plan.deinit();
    try invalid_trimmed_mean_plan.withRowTrimmedMean(&.{ "a", "b" }, "bad_row_trimmed_mean", 0.5);
    try std.testing.expectError(error.InvalidShape, invalid_trimmed_mean_plan.collect());

    var invalid_winsorized_mean_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_winsorized_mean_plan.deinit();
    try invalid_winsorized_mean_plan.withRowWinsorizedMean(&.{ "a", "b" }, "bad_row_winsorized_mean", 0.5);
    try std.testing.expectError(error.InvalidShape, invalid_winsorized_mean_plan.collect());

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowSum(&.{"flag"}, "bad_row_sum");
    try std.testing.expectError(error.TypeMismatch, invalid_plan.collect());

    var invalid_mean_ratio_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_mean_ratio_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_mean_ratio_plan.withRowMeanRatio(&.{"a"}, &.{ "bad_a_mean_ratio", "extra_mean_ratio" }));

    var invalid_average_rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_average_rank_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_average_rank_plan.withRowAvgRanks(&.{"a"}, &.{ "bad_a_average_rank", "extra_average_rank" }));

    var invalid_ordinal_rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_ordinal_rank_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_ordinal_rank_plan.withRowOrdinalRanks(&.{"a"}, &.{ "bad_a_ordinal_rank", "extra_ordinal_rank" }));

    var invalid_dense_rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_dense_rank_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_dense_rank_plan.withRowDenseRanks(&.{"a"}, &.{ "bad_a_dense_rank", "extra_dense_rank" }));

    var invalid_competition_rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_competition_rank_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_competition_rank_plan.withRowMinRank(&.{"a"}, &.{ "bad_a_min_rank", "extra_min_rank" }));

    var invalid_percent_rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_percent_rank_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_percent_rank_plan.withRowPercentileRank(&.{"a"}, &.{ "bad_a_percent_rank", "extra_percent_rank" }));

    var invalid_cume_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cume_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cume_plan.withRowCumeDistribution(&.{"a"}, &.{ "bad_a_cume", "extra_cume" }));

    var invalid_cummode_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cummode_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cummode_plan.withRowPrefixMode(&.{"a"}, &.{ "bad_a_cummode", "extra_cummode" }));

    var invalid_cummode_margin_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cummode_margin_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cummode_margin_plan.withRowPrefixModeMargin(&.{"a"}, &.{ "bad_a_cummode_margin", "extra_cummode_margin" }));

    var invalid_cummode_count_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cummode_count_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cummode_count_plan.withRowPrefixModeCount(&.{"a"}, &.{ "bad_a_cummode_count", "extra_cummode_count" }));

    var invalid_cumdistinct_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumdistinct_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumdistinct_plan.withRowPrefixNUnique(&.{"a"}, &.{ "bad_a_cumunique", "extra_cumunique" }));

    var invalid_cumsum_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumsum_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumsum_plan.withRowCumSum(&.{"a"}, &.{ "bad_a_cumsum", "extra_cumsum" }));

    var invalid_cummean_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cummean_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cummean_plan.withRowCumAvg(&.{"a"}, &.{ "bad_a_cummean", "extra_cummean" }));

    var invalid_cumlse_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumlse_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumlse_plan.withRowPrefixLogsumexp(&.{"a"}, &.{ "bad_a_cumlse", "extra_cumlse" }));

    var invalid_cumgeo_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumgeo_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumgeo_plan.withRowPrefixGeoMean(&.{"a"}, &.{ "bad_a_cumgeo", "extra_cumgeo" }));

    var invalid_cumvar_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumvar_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumvar_plan.withRowCumVar(&.{"a"}, &.{ "bad_a_cumvar", "extra_cumvar" }, 0.0));

    var invalid_cumstd_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumstd_plan.deinit();
    try invalid_cumstd_plan.withRowPrefixStd(&.{"a"}, &.{"bad_a_cumstd"}, -1.0);
    try std.testing.expectError(error.InvalidShape, invalid_cumstd_plan.collect());

    var invalid_cumfano_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumfano_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumfano_plan.withRowPrefixFano(&.{"a"}, &.{ "bad_a_cumfano", "extra_cumfano" }, 0.0));

    var invalid_cumskew_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumskew_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumskew_plan.withRowPrefixSkew(&.{"a"}, &.{ "bad_a_cumskew", "extra_cumskew" }));

    var invalid_cummeansq_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cummeansq_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cummeansq_plan.withRowPrefixMeanSquared(&.{"a"}, &.{ "bad_a_cummeansq", "extra_cummeansq" }));

    var invalid_cummaxabs_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cummaxabs_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cummaxabs_plan.withRowPrefixLInfNorm(&.{"a"}, &.{ "bad_a_cummaxabs", "extra_cummaxabs" }));

    var invalid_cuml2_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cuml2_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cuml2_plan.withRowCumL2Norm(&.{"a"}, &.{ "bad_a_cuml2", "extra_cuml2" }));

    var invalid_cumprod_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumprod_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumprod_plan.withRowCumProd(&.{"a"}, &.{ "bad_a_cumprod", "extra_cumprod" }));

    var invalid_cummax_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cummax_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cummax_plan.withRowCumMax(&.{"a"}, &.{ "bad_a_cummax", "extra_cummax" }));

    var invalid_cummin_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cummin_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cummin_plan.withRowCumMin(&.{"a"}, &.{ "bad_a_cummin", "extra_cummin" }));

    var invalid_cumarg_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumarg_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumarg_plan.withRowPrefixArgMin(&.{"a"}, &.{ "bad_a_cum_argmin", "extra_cum_argmin" }));

    var invalid_cumrange_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumrange_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumrange_plan.withRowCumPtp(&.{"a"}, &.{ "bad_a_cumrange", "extra_cumrange" }));

    var invalid_robust_zscore_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_robust_zscore_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_robust_zscore_plan.withRowMadZscore(&.{"a"}, &.{ "bad_a_mad_zscore", "extra_mad_zscore" }));

    var invalid_iqr_outlier_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_iqr_outlier_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_iqr_outlier_plan.withRowTukeyOutlier(&.{"a"}, &.{ "bad_a_iqr_outlier", "extra_iqr_outlier" }));

    var invalid_tukey_winsor_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_tukey_winsor_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_tukey_winsor_plan.withRowIqrWinsorize(&.{"a"}, &.{ "bad_a_tukey_winsor", "extra_tukey_winsor" }));

    var invalid_max_indicator_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_max_indicator_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_max_indicator_plan.withRowIsMax(&.{"a"}, &.{ "bad_a_is_max", "extra_is_max" }));

    var invalid_min_indicator_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_min_indicator_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_min_indicator_plan.withRowIsMin(&.{"a"}, &.{ "bad_a_is_min", "extra_is_min" }));

    var invalid_correction_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_correction_plan.deinit();
    try invalid_correction_plan.withRowVariance(&.{ "a", "b" }, "bad_row_variance", -1.0);
    try std.testing.expectError(error.InvalidShape, invalid_correction_plan.collect());
}

test "device lazy frame derives row cumulative weighted support columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowPrefixWeightedAvg(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummean", "b_row_weighted_cummean" });
    try plan.withRowCumWeightedMeanSq(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummeansq", "b_row_weighted_cummeansq" });
    try plan.withRowCumulativeWeightedRMS(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumrms", "b_row_weighted_cumrms" });
    try plan.withRowCumulativeWeightedMeanAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummeanabs", "b_row_weighted_cummeanabs" });
    try plan.withRowCumWeightedL1(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuml1", "b_row_weighted_cuml1" });
    try plan.withRowPrefixWeightedL2Norm(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuml2", "b_row_weighted_cuml2" });
    try plan.withRowCumulativeWeightedWeightSum(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_weight_sum", "b_row_weighted_cum_weight_sum" });
    try plan.withRowCumWeightedPositiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_positive_count", "b_row_weighted_cum_positive_count" });
    try plan.withRowPrefixWeightedEffectiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_effective_n", "b_row_weighted_cum_effective_n" });
    try plan.select(&.{
        "a_row_weighted_cummean",
        "b_row_weighted_cummean",
        "a_row_weighted_cummeansq",
        "b_row_weighted_cummeansq",
        "a_row_weighted_cumrms",
        "b_row_weighted_cumrms",
        "a_row_weighted_cummeanabs",
        "b_row_weighted_cummeanabs",
        "a_row_weighted_cuml1",
        "b_row_weighted_cuml1",
        "a_row_weighted_cuml2",
        "b_row_weighted_cuml2",
        "a_row_weighted_cum_weight_sum",
        "b_row_weighted_cum_weight_sum",
        "a_row_weighted_cum_positive_count",
        "b_row_weighted_cum_positive_count",
        "a_row_weighted_cum_effective_n",
        "b_row_weighted_cum_effective_n",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mean(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummean,b_row_weighted_cummean])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mean_square(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummeansq,b_row_weighted_cummeansq])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_rms(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumrms,b_row_weighted_cumrms])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mean_abs(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummeanabs,b_row_weighted_cummeanabs])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_l1_norm(values=[a,b], weights=[wa,wb]->[a_row_weighted_cuml1,b_row_weighted_cuml1])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_l2_norm(values=[a,b], weights=[wa,wb]->[a_row_weighted_cuml2,b_row_weighted_cuml2])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_weight_sum(values=[a,b], weights=[wa,wb]->[a_row_weighted_cum_weight_sum,b_row_weighted_cum_weight_sum])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_positive_count(values=[a,b], weights=[wa,wb]->[a_row_weighted_cum_positive_count,b_row_weighted_cum_positive_count])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_effective_n(values=[a,b], weights=[wa,wb]->[a_row_weighted_cum_effective_n,b_row_weighted_cum_effective_n])") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 18), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummean", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummean", &.{ 0.0, 20.0, 0.0, 56.0 / 5.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummeansq", &.{ 1.0, 0.0, 0.0, 16.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummeansq", &.{ 0.0, 400.0, 0.0, 1664.0 / 5.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumrms", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumrms", &.{ 0.0, 20.0, 0.0, std.math.sqrt(@as(f64, 1664.0 / 5.0)) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummeanabs", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummeanabs", &.{ 0.0, 20.0, 0.0, 56.0 / 5.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cuml1", &.{ 1.0, 0.0, 0.0, 16.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cuml1", &.{ 0.0, 20.0, 0.0, 56.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cuml2", &.{ 1.0, 0.0, 0.0, 8.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cuml2", &.{ 0.0, 20.0, 0.0, std.math.sqrt(@as(f64, 1664.0)) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cum_weight_sum", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cum_weight_sum", &.{ 0.0, 1.0, 0.0, 5.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cum_positive_count", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cum_positive_count", &.{ 0.0, 1.0, 0.0, 2.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cum_effective_n", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cum_effective_n", &.{ 0.0, 1.0, 0.0, 25.0 / 17.0 }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedMeanSquare(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cummeansq", "extra_row_weighted_cummeansq" });
    try std.testing.expectError(error.LengthMismatch, invalid_plan.collect());
}

test "device lazy frame derives row cumulative weighted extrema columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowCumulativeWeightedMin(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummin", "b_row_weighted_cummin" });
    try plan.withRowPrefixWeightedMax(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummax", "b_row_weighted_cummax" });
    try plan.withRowCumWeightedMaxAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummaxabs", "b_row_weighted_cummaxabs" });
    try plan.withRowPrefixWeightedMinAbs(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumminabs", "b_row_weighted_cumminabs" });
    try plan.withRowCumulativeWeightedRange(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumrange", "b_row_weighted_cumrange" });
    try plan.withRowCumWeightedMidrange(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummidrange", "b_row_weighted_cummidrange" });
    try plan.withRowPrefixWeightedRangeCoeff(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumrange_coeff", "b_row_weighted_cumrange_coeff" });
    try plan.select(&.{
        "a_row_weighted_cummin",
        "b_row_weighted_cummin",
        "a_row_weighted_cummax",
        "b_row_weighted_cummax",
        "a_row_weighted_cummaxabs",
        "b_row_weighted_cummaxabs",
        "a_row_weighted_cumminabs",
        "b_row_weighted_cumminabs",
        "a_row_weighted_cumrange",
        "b_row_weighted_cumrange",
        "a_row_weighted_cummidrange",
        "b_row_weighted_cummidrange",
        "a_row_weighted_cumrange_coeff",
        "b_row_weighted_cumrange_coeff",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_min(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummin,b_row_weighted_cummin])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_max(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummax,b_row_weighted_cummax])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_max_abs(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummaxabs,b_row_weighted_cummaxabs])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_min_abs(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumminabs,b_row_weighted_cumminabs])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_range(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumrange,b_row_weighted_cumrange])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_midrange(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummidrange,b_row_weighted_cummidrange])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_range_coeff(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumrange_coeff,b_row_weighted_cumrange_coeff])") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 14), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummin", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummin", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummax", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummax", &.{ 0.0, 20.0, 0.0, 40.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummaxabs", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummaxabs", &.{ 0.0, 20.0, 0.0, 40.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumminabs", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumminabs", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumrange", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumrange", &.{ 0.0, 0.0, 0.0, 36.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummidrange", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummidrange", &.{ 0.0, 20.0, 0.0, 22.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumrange_coeff", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumrange_coeff", &.{ 0.0, 0.0, 0.0, 9.0 / 11.0 }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedRange(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cumrange", "extra_row_weighted_cumrange" });
    try std.testing.expectError(error.LengthMismatch, invalid_plan.collect());
}

test "device lazy frame derives row cumulative weighted log summary columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowCumWeightedProd(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumprod", "b_row_weighted_cumprod" });
    try plan.withRowPrefixWeightedGeoMean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumgeo", "b_row_weighted_cumgeo" });
    try plan.withRowCumWeightedHarmonicMean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumharmonic", "b_row_weighted_cumharmonic" });
    try plan.withRowPrefixWeightedLogSumExp(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumlogsumexp", "b_row_weighted_cumlogsumexp" });
    try plan.withRowCumWeightedLogmeanexp(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumlogmeanexp", "b_row_weighted_cumlogmeanexp" });
    try plan.select(&.{
        "a_row_weighted_cumprod",
        "b_row_weighted_cumprod",
        "a_row_weighted_cumgeo",
        "b_row_weighted_cumgeo",
        "a_row_weighted_cumharmonic",
        "b_row_weighted_cumharmonic",
        "a_row_weighted_cumlogsumexp",
        "b_row_weighted_cumlogsumexp",
        "a_row_weighted_cumlogmeanexp",
        "b_row_weighted_cumlogmeanexp",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_product(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumprod,b_row_weighted_cumprod])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_geometric_mean(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumgeo,b_row_weighted_cumgeo])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_harmonic_mean(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumharmonic,b_row_weighted_cumharmonic])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_logsumexp(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumlogsumexp,b_row_weighted_cumlogsumexp])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_logmeanexp(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumlogmeanexp,b_row_weighted_cumlogmeanexp])") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 10), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumprod", &.{ 1.0, 0.0, 0.0, 256.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumprod", &.{ 0.0, 20.0, 0.0, std.math.exp(4.0 * std.math.log(f64, std.math.e, @as(f64, 4.0)) + std.math.log(f64, std.math.e, @as(f64, 40.0))) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumgeo", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumgeo", &.{ 0.0, 20.0, 0.0, std.math.exp((4.0 * std.math.log(f64, std.math.e, @as(f64, 4.0)) + std.math.log(f64, std.math.e, @as(f64, 40.0))) / 5.0) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumharmonic", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumharmonic", &.{ 0.0, 20.0, 0.0, 5.0 / (4.0 / 4.0 + 1.0 / 40.0) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumlogsumexp", &.{ 1.0, 0.0, 0.0, 4.0 + std.math.log(f64, std.math.e, @as(f64, 4.0)) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumlogsumexp", &.{ 0.0, 20.0, 0.0, 40.0 + std.math.log1p(@as(f64, 4.0) * std.math.exp(@as(f64, -36.0))) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumlogmeanexp", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumlogmeanexp", &.{ 0.0, 20.0, 0.0, 40.0 + std.math.log1p(@as(f64, 4.0) * std.math.exp(@as(f64, -36.0))) - std.math.log(f64, std.math.e, @as(f64, 5.0)) }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedProduct(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cumprod", "extra_row_weighted_cumprod" });
    try std.testing.expectError(error.LengthMismatch, invalid_plan.collect());
}

test "device lazy frame derives row cumulative weighted dispersion columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowCumWeightedVar(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumvar", "b_row_weighted_cumvar" }, 0.0);
    try plan.withRowPrefixWeightedStd(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumstd", "b_row_weighted_cumstd" }, 0.0);
    try plan.withRowCumulativeWeightedSem(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumsem", "b_row_weighted_cumsem" }, 0.0);
    try plan.withRowCumWeightedCV(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcv", "b_row_weighted_cumcv" }, 0.0);
    try plan.withRowPrefixWeightedFano(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumfano", "b_row_weighted_cumfano" }, 0.0);
    try plan.select(&.{
        "a_row_weighted_cumvar",
        "b_row_weighted_cumvar",
        "a_row_weighted_cumstd",
        "b_row_weighted_cumstd",
        "a_row_weighted_cumsem",
        "b_row_weighted_cumsem",
        "a_row_weighted_cumcv",
        "b_row_weighted_cumcv",
        "a_row_weighted_cumfano",
        "b_row_weighted_cumfano",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_variance(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumvar,b_row_weighted_cumvar], correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_stddev(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumstd,b_row_weighted_cumstd], correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_sem(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumsem,b_row_weighted_cumsem], correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_cv(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumcv,b_row_weighted_cumcv], correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_fano(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumfano,b_row_weighted_cumfano], correction=0)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 10), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumvar", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumvar", &.{ 0.0, 0.0, 0.0, 207.36 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumstd", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumstd", &.{ 0.0, 0.0, 0.0, 14.4 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumsem", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumsem", &.{ 0.0, 0.0, 0.0, std.math.sqrt(@as(f64, 207.36 / 5.0)) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumcv", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumcv", &.{ 0.0, 0.0, 0.0, 9.0 / 7.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumfano", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumfano", &.{ 0.0, 0.0, 0.0, 648.0 / 35.0 }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedVariance(&.{"a"}, &.{"wa"}, &.{"a_row_weighted_cumvar"}, -1.0);
    try std.testing.expectError(error.InvalidShape, invalid_plan.collect());
}

test "device lazy frame derives row cumulative weighted shape columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowCumWeightedSkew(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumskew", "b_row_weighted_cumskew" });
    try plan.withRowPrefixWeightedKurtosis(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumkurt", "b_row_weighted_cumkurt" });
    try plan.select(&.{
        "a_row_weighted_cumskew",
        "b_row_weighted_cumskew",
        "a_row_weighted_cumkurt",
        "b_row_weighted_cumkurt",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_skewness(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumskew,b_row_weighted_cumskew])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_kurtosis(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumkurt,b_row_weighted_cumkurt])") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 4), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumskew", &.{ std.math.nan(f64), 0.0, 0.0, std.math.nan(f64) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumskew", &.{ 0.0, std.math.nan(f64), 0.0, 1.5 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumkurt", &.{ std.math.nan(f64), 0.0, 0.0, std.math.nan(f64) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumkurt", &.{ 0.0, std.math.nan(f64), 0.0, 0.25 }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedSkewness(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cumskew", "extra_row_weighted_cumskew" });
    try std.testing.expectError(error.LengthMismatch, invalid_plan.collect());
}

test "device lazy frame derives row cumulative weighted quantile columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowPrefixWeightedQuantile(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumquantile", "b_row_weighted_cumquantile" }, 0.9);
    try plan.withRowCumWeightedMedian(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummedian", "b_row_weighted_cummedian" });
    try plan.withRowPrefixWeightedIQR(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumiqr", "b_row_weighted_cumiqr" });
    try plan.withRowPrefixWeightedMAD(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummad", "b_row_weighted_cummad" });
    try plan.withRowPrefixWeightedTrimmedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumtrimmed", "b_row_weighted_cumtrimmed" }, 0.25);
    try plan.withRowCumWeightedWinsorizedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumwinsor", "b_row_weighted_cumwinsor" }, 0.25);
    try plan.select(&.{
        "a_row_weighted_cumquantile",
        "b_row_weighted_cumquantile",
        "a_row_weighted_cummedian",
        "b_row_weighted_cummedian",
        "a_row_weighted_cumiqr",
        "b_row_weighted_cumiqr",
        "a_row_weighted_cummad",
        "b_row_weighted_cummad",
        "a_row_weighted_cumtrimmed",
        "b_row_weighted_cumtrimmed",
        "a_row_weighted_cumwinsor",
        "b_row_weighted_cumwinsor",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_quantile(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumquantile,b_row_weighted_cumquantile], q=0.9)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_median(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummedian,b_row_weighted_cummedian])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_iqr(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumiqr,b_row_weighted_cumiqr])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mad(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummad,b_row_weighted_cummad])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_trimmed_mean(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumtrimmed,b_row_weighted_cumtrimmed], trim_fraction=0.25)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_winsorized_mean(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumwinsor,b_row_weighted_cumwinsor], winsor_fraction=0.25)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 12), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumquantile", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumquantile", &.{ 0.0, 20.0, 0.0, 40.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummedian", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummedian", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumiqr", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumiqr", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummad", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummad", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumtrimmed", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumtrimmed", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumwinsor", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumwinsor", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedTrimmedMean(&.{"a"}, &.{"wa"}, &.{"a_row_weighted_cumtrimmed"}, 0.5);
    try std.testing.expectError(error.InvalidShape, invalid_plan.collect());
}

test "device lazy frame derives row cumulative weighted percentile-shape columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowCumWeightedIDR(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumidr", "b_row_weighted_cumidr" });
    try plan.withRowPrefixWeightedMidhinge(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummidhinge", "b_row_weighted_cummidhinge" });
    try plan.withRowCumWeightedTrimean(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumtrimean", "b_row_weighted_cumtrimean" });
    try plan.withRowPrefixWeightedBowleySkew(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumbowley", "b_row_weighted_cumbowley" });
    try plan.withRowCumWeightedQCD(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumqcd", "b_row_weighted_cumqcd" });
    try plan.withRowPrefixWeightedKelleySkew(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumkelley", "b_row_weighted_cumkelley" });
    try plan.select(&.{
        "a_row_weighted_cumidr",
        "b_row_weighted_cumidr",
        "a_row_weighted_cummidhinge",
        "b_row_weighted_cummidhinge",
        "a_row_weighted_cumtrimean",
        "b_row_weighted_cumtrimean",
        "a_row_weighted_cumbowley",
        "b_row_weighted_cumbowley",
        "a_row_weighted_cumqcd",
        "b_row_weighted_cumqcd",
        "a_row_weighted_cumkelley",
        "b_row_weighted_cumkelley",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_interdecile_range(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumidr,b_row_weighted_cumidr])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_midhinge(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummidhinge,b_row_weighted_cummidhinge])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_trimean(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumtrimean,b_row_weighted_cumtrimean])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_bowley_skewness(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumbowley,b_row_weighted_cumbowley])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_quartile_coeff_dispersion(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumqcd,b_row_weighted_cumqcd])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_kelley_skewness(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumkelley,b_row_weighted_cumkelley])") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 12), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumidr", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumidr", &.{ 0.0, 0.0, 0.0, 36.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummidhinge", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummidhinge", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumtrimean", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumtrimean", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumbowley", &.{ std.math.nan(f64), 0.0, 0.0, std.math.nan(f64) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumbowley", &.{ 0.0, std.math.nan(f64), 0.0, std.math.nan(f64) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumqcd", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumqcd", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumkelley", &.{ std.math.nan(f64), 0.0, 0.0, std.math.nan(f64) }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumkelley", &.{ 0.0, std.math.nan(f64), 0.0, 1.0 }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedMidhinge(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cummidhinge", "extra_row_weighted_cummidhinge" });
    try std.testing.expectError(error.LengthMismatch, invalid_plan.collect());
}

test "device lazy frame derives row cumulative weighted mode columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowPrefixWeightedMode(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode", "b_row_weighted_cummode" });
    try plan.withRowCumWeightedModeWeight(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode_weight", "b_row_weighted_cummode_weight" });
    try plan.withRowPrefixWeightedModeRatio(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode_ratio", "b_row_weighted_cummode_ratio" });
    try plan.withRowCumWeightedModeMargin(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode_margin", "b_row_weighted_cummode_margin" });
    try plan.withRowPrefixWeightedModeMarginRatio(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummode_margin_ratio", "b_row_weighted_cummode_margin_ratio" });
    try plan.select(&.{
        "a_row_weighted_cummode",
        "b_row_weighted_cummode",
        "a_row_weighted_cummode_weight",
        "b_row_weighted_cummode_weight",
        "a_row_weighted_cummode_ratio",
        "b_row_weighted_cummode_ratio",
        "a_row_weighted_cummode_margin",
        "b_row_weighted_cummode_margin",
        "a_row_weighted_cummode_margin_ratio",
        "b_row_weighted_cummode_margin_ratio",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mode(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummode,b_row_weighted_cummode])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mode_weight(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummode_weight,b_row_weighted_cummode_weight])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mode_ratio(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummode_ratio,b_row_weighted_cummode_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mode_margin(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummode_margin,b_row_weighted_cummode_margin])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mode_margin_ratio(values=[a,b], weights=[wa,wb]->[a_row_weighted_cummode_margin_ratio,b_row_weighted_cummode_margin_ratio])") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 10), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummode", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummode", &.{ 0.0, 20.0, 0.0, 4.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummode_weight", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummode_weight", &.{ 0.0, 1.0, 0.0, 4.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummode_ratio", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummode_ratio", &.{ 0.0, 1.0, 0.0, 4.0 / 5.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummode_margin", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummode_margin", &.{ 0.0, 1.0, 0.0, 3.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummode_margin_ratio", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummode_margin_ratio", &.{ 0.0, 1.0, 0.0, 3.0 / 5.0 }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedModeWeight(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cummode_weight", "extra_row_weighted_cummode_weight" });
    try std.testing.expectError(error.LengthMismatch, invalid_plan.collect());
}

test "device lazy frame derives row cumulative weighted pair support columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowCumulativeWeightedPairWeightSum(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_pair_cum_weight_sum", "b_row_weighted_pair_cum_weight_sum" });
    try plan.withRowCumWeightedPairPositiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_pair_cum_positive_count", "b_row_weighted_pair_cum_positive_count" });
    try plan.withRowPrefixWeightedPairEffectiveCount(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_pair_cum_effective_n", "b_row_weighted_pair_cum_effective_n" });
    try plan.withRowPrefixWeightedDot(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumdot", "b_row_weighted_cumdot" });
    try plan.withRowCumWeightedCosine(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcosine", "b_row_weighted_cumcosine" });
    try plan.withRowCumWeightedSquaredDistance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumsqdist", "b_row_weighted_cumsqdist" });
    try plan.withRowPrefixWeightedL2Distance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuml2", "b_row_weighted_cuml2" });
    try plan.withRowPrefixWeightedL1Distance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuml1dist", "b_row_weighted_cuml1dist" });
    try plan.withRowCumWeightedChebyshevDistance(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumchebyshev", "b_row_weighted_cumchebyshev" });
    try plan.withRowCumWeightedCanberraDistance(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcanberra", "b_row_weighted_cumcanberra" });
    try plan.withRowCumWeightedBrayCurtisDistance(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumbray", "b_row_weighted_cumbray" });
    try plan.withRowPrefixWeightedBias(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumbias", "b_row_weighted_cumbias" });
    try plan.withRowPrefixWeightedMAE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummae", "b_row_weighted_cummae" });
    try plan.withRowPrefixWeightedMSE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummse", "b_row_weighted_cummse" });
    try plan.withRowPrefixWeightedRMSE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumrmse", "b_row_weighted_cumrmse" });
    try plan.withRowPrefixWeightedMAPE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cummape", "b_row_weighted_cummape" });
    try plan.withRowPrefixWeightedSMAPE(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumsmape", "b_row_weighted_cumsmape" });
    try plan.withRowPrefixWeightedCov(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcov", "b_row_weighted_cumcov" }, 0.0);
    try plan.select(&.{
        "a_row_weighted_pair_cum_weight_sum",
        "b_row_weighted_pair_cum_weight_sum",
        "a_row_weighted_pair_cum_positive_count",
        "b_row_weighted_pair_cum_positive_count",
        "a_row_weighted_pair_cum_effective_n",
        "b_row_weighted_pair_cum_effective_n",
        "a_row_weighted_cumdot",
        "b_row_weighted_cumdot",
        "a_row_weighted_cumcosine",
        "b_row_weighted_cumcosine",
        "a_row_weighted_cumsqdist",
        "b_row_weighted_cumsqdist",
        "a_row_weighted_cuml2",
        "b_row_weighted_cuml2",
        "a_row_weighted_cuml1dist",
        "b_row_weighted_cuml1dist",
        "a_row_weighted_cumchebyshev",
        "b_row_weighted_cumchebyshev",
        "a_row_weighted_cumcanberra",
        "b_row_weighted_cumcanberra",
        "a_row_weighted_cumbray",
        "b_row_weighted_cumbray",
        "a_row_weighted_cumbias",
        "b_row_weighted_cumbias",
        "a_row_weighted_cummae",
        "b_row_weighted_cummae",
        "a_row_weighted_cummse",
        "b_row_weighted_cummse",
        "a_row_weighted_cumrmse",
        "b_row_weighted_cumrmse",
        "a_row_weighted_cummape",
        "b_row_weighted_cummape",
        "a_row_weighted_cumsmape",
        "b_row_weighted_cumsmape",
        "a_row_weighted_cumcov",
        "b_row_weighted_cumcov",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_pair_weight_sum(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->[a_row_weighted_pair_cum_weight_sum,b_row_weighted_pair_cum_weight_sum])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_pair_positive_count(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->[a_row_weighted_pair_cum_positive_count,b_row_weighted_pair_cum_positive_count])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_pair_effective_n(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->[a_row_weighted_pair_cum_effective_n,b_row_weighted_pair_cum_effective_n])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_dot(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->[a_row_weighted_cumdot,b_row_weighted_cumdot])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_cosine_similarity(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->[a_row_weighted_cumcosine,b_row_weighted_cumcosine])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_squared_euclidean_distance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->[a_row_weighted_cumsqdist,b_row_weighted_cumsqdist])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_euclidean_distance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->[a_row_weighted_cuml2,b_row_weighted_cuml2])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_manhattan_distance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->[a_row_weighted_cuml1dist,b_row_weighted_cuml1dist])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_chebyshev_distance(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cumchebyshev,b_row_weighted_cumchebyshev])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_canberra_distance(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cumcanberra,b_row_weighted_cumcanberra])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_bray_curtis_distance(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cumbray,b_row_weighted_cumbray])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mean_error(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cumbias,b_row_weighted_cumbias])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mae(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cummae,b_row_weighted_cummae])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mse(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cummse,b_row_weighted_cummse])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_rmse(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cumrmse,b_row_weighted_cumrmse])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mape(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cummape,b_row_weighted_cummape])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_smape(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cumsmape,b_row_weighted_cumsmape])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_covariance(lhs=[a,b], rhs=[wb,wa], weights=[wa,wb]->[a_row_weighted_cumcov,b_row_weighted_cumcov], correction=0)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 36), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_pair_cum_weight_sum", &.{ 1.0, 0.0, 0.0, 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_pair_cum_weight_sum", &.{ 0.0, 1.0, 0.0, 5.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_pair_cum_positive_count", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_pair_cum_positive_count", &.{ 0.0, 1.0, 0.0, 2.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_pair_cum_effective_n", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_pair_cum_effective_n", &.{ 0.0, 1.0, 0.0, 25.0 / 17.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumdot", &.{ 1.0, 0.0, 0.0, 64.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumdot", &.{ 0.0, 20.0, 0.0, 104.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumcosine", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumcosine", &.{ 0.0, 1.0, 0.0, 104.0 / std.math.sqrt(@as(f64, 1664.0 * 65.0)) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumsqdist", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumsqdist", &.{ 0.0, 361.0, 0.0, 1521.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cuml2", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cuml2", &.{ 0.0, 19.0, 0.0, 39.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cuml1dist", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cuml1dist", &.{ 0.0, 19.0, 0.0, 39.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumchebyshev", &.{ 1.0, 0.0, 0.0, 3.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumchebyshev", &.{ 0.0, 18.0, 0.0, 36.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumcanberra", &.{ 1.0 / 3.0, 0.0, 0.0, 12.0 / 5.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumcanberra", &.{ 0.0, 9.0 / 11.0, 0.0, 177.0 / 55.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumbray", &.{ 1.0 / 3.0, 0.0, 0.0, 3.0 / 5.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumbray", &.{ 0.0, 9.0 / 11.0, 0.0, 3.0 / 4.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumbias", &.{ -1.0, 0.0, 0.0, 3.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumbias", &.{ 0.0, 18.0, 0.0, 48.0 / 5.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummae", &.{ 1.0, 0.0, 0.0, 3.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummae", &.{ 0.0, 18.0, 0.0, 48.0 / 5.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummse", &.{ 1.0, 0.0, 0.0, 9.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummse", &.{ 0.0, 324.0, 0.0, 1332.0 / 5.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumrmse", &.{ 1.0, 0.0, 0.0, 3.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumrmse", &.{ 0.0, 18.0, 0.0, std.math.sqrt(@as(f64, 1332.0 / 5.0)) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cummape", &.{ 1.0, 0.0, 0.0, 3.0 / 4.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cummape", &.{ 0.0, 9.0 / 10.0, 0.0, 39.0 / 50.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumsmape", &.{ 2.0 / 3.0, 0.0, 0.0, 6.0 / 5.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumsmape", &.{ 0.0, 18.0 / 11.0, 0.0, 354.0 / 275.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumcov", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumcov", &.{ 0.0, 0.0, 0.0, 432.0 / 25.0 }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedPairWeightSum(&.{"a"}, &.{"wa"}, &.{"wa"}, &.{ "a_row_weighted_pair_cum_weight_sum", "extra_row_weighted_pair_cum_weight_sum" });
    try std.testing.expectError(error.LengthMismatch, invalid_plan.collect());

    var invalid_cov_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cov_plan.deinit();
    try invalid_cov_plan.withRowCumulativeWeightedCovariance(&.{ "a", "b" }, &.{ "wb", "wa" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumcov", "b_row_weighted_cumcov" }, -1.0);
    try std.testing.expectError(error.InvalidShape, invalid_cov_plan.collect());
}

test "device lazy frame derives row cumulative weighted distribution columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowCumWeightedEntropy(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumentropy", "b_row_weighted_cumentropy" });
    try plan.withRowPrefixWeightedGini(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumgini", "b_row_weighted_cumgini" });
    try plan.withRowCumWeightedPerplexity(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumperplexity", "b_row_weighted_cumperplexity" });
    try plan.withRowPrefixWeightedInverseSimpson(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cuminverse", "b_row_weighted_cuminverse" });
    try plan.withRowCumWeightedConcentration(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumconcentration", "b_row_weighted_cumconcentration" });
    try plan.withRowPrefixWeightedEvenness(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cumevenness", "b_row_weighted_cumevenness" });
    try plan.select(&.{
        "a_row_weighted_cumentropy",
        "b_row_weighted_cumentropy",
        "a_row_weighted_cumgini",
        "b_row_weighted_cumgini",
        "a_row_weighted_cumperplexity",
        "b_row_weighted_cumperplexity",
        "a_row_weighted_cuminverse",
        "b_row_weighted_cuminverse",
        "a_row_weighted_cumconcentration",
        "b_row_weighted_cumconcentration",
        "a_row_weighted_cumevenness",
        "b_row_weighted_cumevenness",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_entropy(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumentropy,b_row_weighted_cumentropy])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_gini_impurity(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumgini,b_row_weighted_cumgini])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_perplexity(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumperplexity,b_row_weighted_cumperplexity])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_inverse_simpson(values=[a,b], weights=[wa,wb]->[a_row_weighted_cuminverse,b_row_weighted_cuminverse])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_simpson_concentration(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumconcentration,b_row_weighted_cumconcentration])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_evenness(values=[a,b], weights=[wa,wb]->[a_row_weighted_cumevenness,b_row_weighted_cumevenness])") != null);

    const weighted_prefix_entropy = -(@as(f64, 4.0 / 5.0) * std.math.log(f64, std.math.e, @as(f64, 4.0 / 5.0)) + @as(f64, 1.0 / 5.0) * std.math.log(f64, std.math.e, @as(f64, 1.0 / 5.0)));
    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 12), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumentropy", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumentropy", &.{ 0.0, 0.0, 0.0, weighted_prefix_entropy }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumgini", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumgini", &.{ 0.0, 0.0, 0.0, 8.0 / 25.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumperplexity", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumperplexity", &.{ 0.0, 1.0, 0.0, std.math.exp(weighted_prefix_entropy) }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cuminverse", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cuminverse", &.{ 0.0, 1.0, 0.0, 25.0 / 17.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumconcentration", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumconcentration", &.{ 0.0, 1.0, 0.0, 17.0 / 25.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cumevenness", &.{ 1.0, 0.0, 0.0, 1.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cumevenness", &.{ 0.0, 1.0, 0.0, weighted_prefix_entropy / std.math.log(f64, std.math.e, @as(f64, 2.0)) }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedEntropy(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cumentropy", "extra_row_weighted_cumentropy" });
    try std.testing.expectError(error.LengthMismatch, invalid_plan.collect());
}

test "device lazy frame derives row cumulative weighted inequality columns" {
    const gpa = std.testing.allocator;

    var a = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, &.{ true, false, false, true }, .cpu);
    defer a.deinit();
    var b = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, true, false, true }, .cpu);
    defer b.deinit();
    var weight_a = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer weight_a.deinit();
    var weight_b = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 1.0, 5.0, 1.0 }, .cpu);
    defer weight_b.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = a },
        .{ .name = "b", .data = b },
        .{ .name = "wa", .data = weight_a },
        .{ .name = "wb", .data = weight_b },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowCumWeightedMeanAbsDev(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_mean_abs_dev", "b_row_weighted_cum_mean_abs_dev" });
    try plan.withRowPrefixWeightedMadRatio(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_mad_ratio", "b_row_weighted_cum_mad_ratio" });
    try plan.withRowCumWeightedGiniMeanDiff(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_gini_mean_diff", "b_row_weighted_cum_gini_mean_diff" });
    try plan.withRowPrefixWeightedGiniCoeff(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "a_row_weighted_cum_gini_coeff", "b_row_weighted_cum_gini_coeff" });
    try plan.select(&.{
        "a_row_weighted_cum_mean_abs_dev",
        "b_row_weighted_cum_mean_abs_dev",
        "a_row_weighted_cum_mad_ratio",
        "b_row_weighted_cum_mad_ratio",
        "a_row_weighted_cum_gini_mean_diff",
        "b_row_weighted_cum_gini_mean_diff",
        "a_row_weighted_cum_gini_coeff",
        "b_row_weighted_cum_gini_coeff",
    });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mean_abs_dev(values=[a,b], weights=[wa,wb]->[a_row_weighted_cum_mean_abs_dev,b_row_weighted_cum_mean_abs_dev])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_mean_abs_dev_ratio(values=[a,b], weights=[wa,wb]->[a_row_weighted_cum_mad_ratio,b_row_weighted_cum_mad_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_gini_mean_diff(values=[a,b], weights=[wa,wb]->[a_row_weighted_cum_gini_mean_diff,b_row_weighted_cum_gini_mean_diff])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_weighted_gini_coefficient(values=[a,b], weights=[wa,wb]->[a_row_weighted_cum_gini_coeff,b_row_weighted_cum_gini_coeff])") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 8), result.width());
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cum_mean_abs_dev", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cum_mean_abs_dev", &.{ 0.0, 0.0, 0.0, 288.0 / 25.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cum_mad_ratio", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cum_mad_ratio", &.{ 0.0, 0.0, 0.0, 36.0 / 35.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cum_gini_mean_diff", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cum_gini_mean_diff", &.{ 0.0, 0.0, 0.0, 36.0 }, &.{ false, true, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "a_row_weighted_cum_gini_coeff", &.{ 0.0, 0.0, 0.0, 0.0 }, &.{ true, false, false, true });
    try expectF64ColumnApproxOrNanWithValidity(result, gpa, "b_row_weighted_cum_gini_coeff", &.{ 0.0, 0.0, 0.0, 45.0 / 28.0 }, &.{ false, true, false, true });

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowCumulativeWeightedMeanAbsDev(&.{"a"}, &.{"wa"}, &.{ "a_row_weighted_cum_mean_abs_dev", "extra_row_weighted_cum_mean_abs_dev" });
    try std.testing.expectError(error.LengthMismatch, invalid_plan.collect());
}

test "device lazy frame derives row boolean match index columns" {
    const gpa = std.testing.allocator;

    var signal_a = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ false, true, false, true }, &.{ true, true, true, false }, .cpu);
    defer signal_a.deinit();
    var signal_b = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ true, false, false, false }, &.{ true, false, true, true }, .cpu);
    defer signal_b.deinit();
    var signal_c = try DeviceColumn.fromSliceWithValidity(bool, gpa, &.{ false, true, false, true }, &.{ false, true, true, true }, .cpu);
    defer signal_c.deinit();
    var metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0, 4.0 }, .cpu);
    defer metric.deinit();

    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "a", .data = signal_a },
        .{ .name = "b", .data = signal_b },
        .{ .name = "c", .data = signal_c },
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withRowFirstTrueIndex(&.{ "a", "b", "c" }, "first_true");
    try plan.withRowLastTrueIndex(&.{ "a", "b", "c" }, "last_true");
    try plan.withRowFirstFalseIndex(&.{ "a", "b", "c" }, "first_false");
    try plan.withRowLastFalseIndex(&.{ "a", "b", "c" }, "last_false");
    try plan.select(&.{ "first_true", "last_true", "first_false", "last_false" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_first_true_index([a,b,c]->first_true)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_last_true_index([a,b,c]->last_true)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_first_false_index([a,b,c]->first_false)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_last_false_index([a,b,c]->last_false)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 4), result.width());
    const first_true_column = try result.column("first_true");
    try std.testing.expect(first_true_column.i64.nullable());
    const first_true = try first_true_column.i64.toOwnedSlice(gpa);
    defer gpa.free(first_true);
    const first_true_validity = try first_true_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(first_true_validity);
    const last_true_column = try result.column("last_true");
    try std.testing.expect(last_true_column.i64.nullable());
    const last_true = try last_true_column.i64.toOwnedSlice(gpa);
    defer gpa.free(last_true);
    const last_true_validity = try last_true_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(last_true_validity);
    const first_false_column = try result.column("first_false");
    try std.testing.expect(first_false_column.i64.nullable());
    const first_false = try first_false_column.i64.toOwnedSlice(gpa);
    defer gpa.free(first_false);
    const first_false_validity = try first_false_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(first_false_validity);
    const last_false_column = try result.column("last_false");
    try std.testing.expect(last_false_column.i64.nullable());
    const last_false = try last_false_column.i64.toOwnedSlice(gpa);
    defer gpa.free(last_false);
    const last_false_validity = try last_false_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(last_false_validity);

    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 2 }, first_true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, first_true_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 0, 2 }, last_true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, last_true_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1 }, first_false);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, first_false_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2, 1 }, last_false);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, last_false_validity);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.withRowFirstTrueIndex(&.{"metric"}, "bad_bool_index");
    try std.testing.expectError(error.TypeMismatch, invalid_plan.collect());
}

test "device lazy frame derives zero predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.0, -0.0, 3.0, std.math.nan(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 0, 5, 0, -7, 9, 0 }, .cpu);
    defer id.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ false, true, false, true, true, false }, .cpu);
    defer flag.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.isZeroColumn("metric", "metric_is_zero");
    try plan.isNonZeroColumn("metric", "metric_is_non_zero");
    try plan.isZeroColumn("id", "id_is_zero");
    try plan.isNonZeroColumn("flag", "flag_is_non_zero");
    try plan.withRowZeroCount(&.{ "metric", "id", "flag" }, "row_zero_count");
    try plan.withRowNonZeroCount(&.{ "metric", "id", "flag" }, "row_non_zero_count");
    try plan.withRowZeroRatio(&.{ "metric", "id", "flag" }, "row_zero_ratio");
    try plan.withRowNonZeroRatio(&.{ "metric", "id", "flag" }, "row_non_zero_ratio");
    try plan.withRowAnyZero(&.{ "metric", "id", "flag" }, "row_any_zero");
    try plan.withRowAllNonZero(&.{ "metric", "id", "flag" }, "row_all_non_zero");
    try plan.withRowFirstZeroIndex(&.{ "metric", "id", "flag" }, "row_first_zero_index");
    try plan.withRowLastNonZeroIndex(&.{ "metric", "id", "flag" }, "row_last_nonzero_index");
    try plan.withRowCumulativeFirstZeroIndex(&.{ "metric", "id", "flag" }, &.{ "metric_cum_first_zero", "id_cum_first_zero", "flag_cum_first_zero" });
    try plan.withRowPrefixLastNonZeroIndex(&.{ "metric", "id", "flag" }, &.{ "metric_prefix_last_nonzero", "id_prefix_last_nonzero", "flag_prefix_last_nonzero" });
    try plan.withRowCumulativeZeroCount(&.{ "metric", "id", "flag" }, &.{ "metric_cum_zero", "id_cum_zero", "flag_cum_zero" });
    try plan.withRowPrefixNonZeroRatio(&.{ "metric", "id", "flag" }, &.{ "metric_cum_nonzero", "id_cum_nonzero", "flag_cum_nonzero" });
    try plan.withRowCumulativeAnyZero(&.{ "metric", "id", "flag" }, &.{ "metric_cum_any_zero", "id_cum_any_zero", "flag_cum_any_zero" });
    try plan.withRowPrefixAllNonZero(&.{ "metric", "id", "flag" }, &.{ "metric_prefix_all_nonzero", "id_prefix_all_nonzero", "flag_prefix_all_nonzero" });
    try plan.select(&.{ "metric_is_zero", "metric_is_non_zero", "id_is_zero", "flag_is_non_zero", "row_zero_count", "row_non_zero_count", "row_zero_ratio", "row_non_zero_ratio", "row_any_zero", "row_all_non_zero", "row_first_zero_index", "row_last_nonzero_index", "metric_cum_first_zero", "id_cum_first_zero", "flag_cum_first_zero", "metric_prefix_last_nonzero", "id_prefix_last_nonzero", "flag_prefix_last_nonzero", "metric_cum_zero", "id_cum_zero", "flag_cum_zero", "metric_cum_nonzero", "id_cum_nonzero", "flag_cum_nonzero", "metric_cum_any_zero", "flag_cum_any_zero", "flag_prefix_all_nonzero" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_zero_column(metric->metric_is_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_non_zero_column(metric->metric_is_non_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_zero_count([metric,id,flag]->row_zero_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_non_zero_count([metric,id,flag]->row_non_zero_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_zero_ratio([metric,id,flag]->row_zero_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_non_zero_ratio([metric,id,flag]->row_non_zero_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_zero([metric,id,flag]->row_any_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_all_non_zero([metric,id,flag]->row_all_non_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_first_zero_index([metric,id,flag]->row_first_zero_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_last_non_zero_index([metric,id,flag]->row_last_nonzero_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_first_zero_index([metric,id,flag]->[metric_cum_first_zero,id_cum_first_zero,flag_cum_first_zero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_last_non_zero_index([metric,id,flag]->[metric_prefix_last_nonzero,id_prefix_last_nonzero,flag_prefix_last_nonzero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_zero_count([metric,id,flag]->[metric_cum_zero,id_cum_zero,flag_cum_zero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_non_zero_ratio([metric,id,flag]->[metric_cum_nonzero,id_cum_nonzero,flag_cum_nonzero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_any_zero([metric,id,flag]->[metric_cum_any_zero,id_cum_any_zero,flag_cum_any_zero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_all_non_zero([metric,id,flag]->[metric_prefix_all_nonzero,id_prefix_all_nonzero,flag_prefix_all_nonzero])") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 27), result.width());
    const metric_is_zero = try (try result.column("metric_is_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_zero);
    const metric_is_non_zero = try (try result.column("metric_is_non_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_non_zero);
    const id_is_zero = try (try result.column("id_is_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_zero);
    const flag_is_non_zero = try (try result.column("flag_is_non_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_is_non_zero);
    const row_zero_count = try (try result.column("row_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_zero_count);
    const row_non_zero_count = try (try result.column("row_non_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_non_zero_count);
    const row_zero_ratio = try (try result.column("row_zero_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_zero_ratio);
    const row_non_zero_ratio = try (try result.column("row_non_zero_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_non_zero_ratio);
    const row_any_zero = try (try result.column("row_any_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_zero);
    const row_all_non_zero = try (try result.column("row_all_non_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_non_zero);
    const row_first_zero_column = try result.column("row_first_zero_index");
    try std.testing.expect(row_first_zero_column.i64.nullable());
    const row_first_zero_index = try row_first_zero_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_zero_index);
    const row_first_zero_validity = try row_first_zero_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_zero_validity);
    const row_last_nonzero_column = try result.column("row_last_nonzero_index");
    try std.testing.expect(row_last_nonzero_column.i64.nullable());
    const row_last_nonzero_index = try row_last_nonzero_column.i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_nonzero_index);
    const row_last_nonzero_validity = try row_last_nonzero_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_nonzero_validity);
    const metric_cum_first_zero_column = try result.column("metric_cum_first_zero");
    try std.testing.expect(metric_cum_first_zero_column.i64.nullable());
    const metric_cum_first_zero = try metric_cum_first_zero_column.i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_zero);
    const metric_cum_first_zero_validity = try metric_cum_first_zero_column.i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_zero_validity);
    const flag_cum_first_zero = try (try result.column("flag_cum_first_zero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_zero);
    const flag_cum_first_zero_validity = try (try result.column("flag_cum_first_zero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_zero_validity);
    const id_prefix_last_nonzero = try (try result.column("id_prefix_last_nonzero")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_nonzero);
    const id_prefix_last_nonzero_validity = try (try result.column("id_prefix_last_nonzero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_nonzero_validity);
    const flag_prefix_last_nonzero = try (try result.column("flag_prefix_last_nonzero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_nonzero);
    const flag_prefix_last_nonzero_validity = try (try result.column("flag_prefix_last_nonzero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_nonzero_validity);
    const id_cum_zero = try (try result.column("id_cum_zero")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_zero);
    const flag_cum_zero = try (try result.column("flag_cum_zero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_zero);
    const metric_cum_nonzero = try (try result.column("metric_cum_nonzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_nonzero);
    const flag_cum_nonzero = try (try result.column("flag_cum_nonzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_nonzero);
    const metric_cum_any_zero = try (try result.column("metric_cum_any_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_zero);
    const metric_cum_any_zero_validity = try (try result.column("metric_cum_any_zero")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_zero_validity);
    const flag_cum_any_zero = try (try result.column("flag_cum_any_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_any_zero);
    const flag_prefix_all_nonzero = try (try result.column("flag_prefix_all_nonzero")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_all_nonzero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false }, metric_is_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true, false }, metric_is_non_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false, false, true }, id_is_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, false }, flag_is_non_zero);
    try std.testing.expectEqualSlices(i64, &.{ 3, 1, 2, 0, 0, 2 }, row_zero_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 1, 3, 3, 0 }, row_non_zero_count);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0, 1.0 }, row_zero_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 0.0 }, row_non_zero_ratio);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false, true }, row_any_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, true, false }, row_all_non_zero);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 1 }, row_first_zero_index);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false, true }, row_first_zero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 0, 2, 2, 0 }, row_last_nonzero_index);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, row_last_nonzero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0 }, metric_cum_first_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false }, metric_cum_first_zero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 1 }, flag_cum_first_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false, true }, flag_cum_first_zero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1, 1, 0 }, id_prefix_last_nonzero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, id_prefix_last_nonzero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 0, 2, 2, 0 }, flag_prefix_last_nonzero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, false }, flag_prefix_last_nonzero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 0, 0, 1 }, id_cum_zero);
    try std.testing.expectEqualSlices(i64, &.{ 3, 1, 2, 0, 0, 2 }, flag_cum_zero);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 1.0, 1.0, 1.0, 0.0 }, metric_cum_nonzero);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 0.0 }, flag_cum_nonzero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false }, metric_cum_any_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false }, metric_cum_any_zero_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false, false, true }, flag_cum_any_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, true, false }, flag_prefix_all_nonzero);

    var filter_zero_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_zero_plan.deinit();
    try filter_zero_plan.filterZerosColumn("metric");
    const filter_zero_explain = try filter_zero_plan.explain(gpa);
    defer gpa.free(filter_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_zero_explain, "filter_zeros_column(metric)") != null);
    var filtered_zero_rows = try filter_zero_plan.collect();
    defer filtered_zero_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered_zero_rows.height());
    const filtered_zero_metric = try (try filtered_zero_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_zero_metric);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, -0.0 }, filtered_zero_metric);

    var drop_non_zero_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_zero_plan.deinit();
    try drop_non_zero_plan.dropNonZerosColumn("metric");
    const drop_non_zero_explain = try drop_non_zero_plan.explain(gpa);
    defer gpa.free(drop_non_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_zero_explain, "drop_non_zeros[metric]") != null);
    var dropped_non_zero_rows = try drop_non_zero_plan.collect();
    defer dropped_non_zero_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped_non_zero_rows.height());
    const dropped_non_zero_metric = try (try dropped_non_zero_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_non_zero_metric);
    const dropped_non_zero_validity = try (try dropped_non_zero_rows.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_non_zero_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, -0.0, -2.0 }, dropped_non_zero_metric);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, dropped_non_zero_validity);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isZeroColumn("missing", "missing_is_zero");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());

    var invalid_any_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_any_plan.deinit();
    try invalid_any_plan.withRowAnyZero(&.{"missing"}, "bad_any_zero");
    try std.testing.expectError(error.ColumnNotFound, invalid_any_plan.collect());

    var invalid_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_index_plan.deinit();
    try invalid_index_plan.withRowFirstZeroIndex(&.{"missing"}, "bad_zero_index");
    try std.testing.expectError(error.ColumnNotFound, invalid_index_plan.collect());

    var invalid_cumulative_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_index_plan.deinit();
    try invalid_cumulative_index_plan.withRowCumulativeFirstZeroIndex(&.{"missing"}, &.{"bad_first_zero"});
    try std.testing.expectError(error.ColumnNotFound, invalid_cumulative_index_plan.collect());

    var invalid_cumulative_index_length_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_index_length_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_index_length_plan.withRowPrefixLastNonZeroIndex(&.{"metric"}, &.{ "metric_last_nonzero", "extra_last_nonzero" }));
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_index_length_plan.withRowPrefixAllNonZero(&.{"metric"}, &.{ "metric_all_nonzero", "extra_all_nonzero" }));
}

test "device lazy frame derives sign predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -2.0, -0.0, 0.0, 3.0, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 9.0 }, &.{ true, true, true, true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ -3, 0, 4, -5, 6, 0, -7, 8 }, .cpu);
    defer id.deinit();
    var unsigned = try DeviceColumn.fromSlice(u64, gpa, &.{ 0, 2, 0, 5, 0, 9, 11, 0 }, .cpu);
    defer unsigned.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ false, true, false, true, true, false, true, false }, .cpu);
    defer flag.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
        .{ .name = "unsigned", .data = unsigned },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.isPositiveColumn("metric", "metric_is_positive");
    try plan.isNegativeColumn("metric", "metric_is_negative");
    try plan.isSignBitColumn("metric", "metric_signbit");
    try plan.isSignBitColumn("id", "id_signbit");
    try plan.isPositiveZeroColumn("metric", "metric_is_positive_zero");
    try plan.isNegativeZeroColumn("metric", "metric_is_negative_zero");
    try plan.isPositiveColumn("id", "id_is_positive");
    try plan.isNegativeColumn("unsigned", "unsigned_is_negative");
    try plan.isPositiveColumn("flag", "flag_is_positive");
    try plan.withRowPositiveZeroCount(&.{ "metric", "id", "unsigned", "flag" }, "row_positive_zero_count");
    try plan.withRowNegativeZeroCount(&.{ "metric", "id", "unsigned", "flag" }, "row_negative_zero_count");
    try plan.withRowPositiveZeroRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_positive_zero_ratio");
    try plan.withRowNegativeZeroRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_negative_zero_ratio");
    try plan.withRowPositiveCount(&.{ "metric", "id", "unsigned", "flag" }, "row_positive_count");
    try plan.withRowSignBitCount(&.{ "metric", "id", "unsigned", "flag" }, "row_signbit_count");
    try plan.withRowNegativeCount(&.{ "metric", "id", "unsigned", "flag" }, "row_negative_count");
    try plan.withRowPositiveRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_positive_ratio");
    try plan.withRowSignBitRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_signbit_ratio");
    try plan.withRowNegativeRatio(&.{ "metric", "id", "unsigned", "flag" }, "row_negative_ratio");
    try plan.withRowAnyPositive(&.{ "metric", "id", "unsigned", "flag" }, "row_any_positive");
    try plan.withRowAnySignBit(&.{ "metric", "id", "unsigned", "flag" }, "row_any_signbit");
    try plan.withRowAnyPositiveZero(&.{ "metric", "id", "unsigned", "flag" }, "row_any_positive_zero");
    try plan.withRowAnyNegativeZero(&.{ "metric", "id", "unsigned", "flag" }, "row_any_negative_zero");
    try plan.withRowFirstPositiveZeroIndex(&.{ "metric", "id", "unsigned", "flag" }, "row_first_positive_zero_index");
    try plan.withRowLastSignBitIndex(&.{ "metric", "id", "unsigned", "flag" }, "row_last_signbit_index");
    try plan.withRowFirstPositiveIndex(&.{ "metric", "id", "unsigned", "flag" }, "row_first_positive_index");
    try plan.withRowLastNegativeIndex(&.{ "metric", "id", "unsigned", "flag" }, "row_last_negative_index");
    try plan.withRowCumulativePositiveCount(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_positive", "id_cum_positive", "unsigned_cum_positive", "flag_cum_positive" });
    try plan.withRowCumulativeFirstPositiveIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_first_positive", "id_cum_first_positive", "unsigned_cum_first_positive", "flag_cum_first_positive" });
    try plan.withRowPrefixLastSignBitIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_prefix_last_signbit", "id_prefix_last_signbit", "unsigned_prefix_last_signbit", "flag_prefix_last_signbit" });
    try plan.withRowCumulativeLastNegativeIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_last_negative", "id_cum_last_negative", "unsigned_cum_last_negative", "flag_cum_last_negative" });
    try plan.withRowPrefixNegativeRatio(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_negative", "id_cum_negative", "unsigned_cum_negative", "flag_cum_negative" });
    try plan.withRowCumulativePositiveZeroCount(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_poszero", "id_cum_poszero", "unsigned_cum_poszero", "flag_cum_poszero" });
    try plan.withRowCumulativeFirstPositiveZeroIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_first_poszero", "id_cum_first_poszero", "unsigned_cum_first_poszero", "flag_cum_first_poszero" });
    try plan.withRowPrefixLastNegativeZeroIndex(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_prefix_last_negzero", "id_prefix_last_negzero", "unsigned_prefix_last_negzero", "flag_prefix_last_negzero" });
    try plan.withRowPrefixNegativeZeroRatio(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_negzero", "id_cum_negzero", "unsigned_cum_negzero", "flag_cum_negzero" });
    try plan.withRowCumulativeSignBitRatio(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_signbit", "id_cum_signbit", "unsigned_cum_signbit", "flag_cum_signbit" });
    try plan.withRowCumulativeAnyPositive(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_any_positive", "id_cum_any_positive", "unsigned_cum_any_positive", "flag_cum_any_positive" });
    try plan.withRowPrefixAllSignBit(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_prefix_all_signbit", "id_prefix_all_signbit", "unsigned_prefix_all_signbit", "flag_prefix_all_signbit" });
    try plan.withRowCumulativeAnyPositiveZero(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_cum_any_poszero", "id_cum_any_poszero", "unsigned_cum_any_poszero", "flag_cum_any_poszero" });
    try plan.withRowPrefixAllNegative(&.{ "metric", "id", "unsigned", "flag" }, &.{ "metric_prefix_all_negative", "id_prefix_all_negative", "unsigned_prefix_all_negative", "flag_prefix_all_negative" });
    try plan.select(&.{ "metric_is_positive", "metric_is_negative", "metric_signbit", "id_signbit", "metric_is_positive_zero", "metric_is_negative_zero", "id_is_positive", "unsigned_is_negative", "flag_is_positive", "row_positive_zero_count", "row_negative_zero_count", "row_positive_zero_ratio", "row_negative_zero_ratio", "row_positive_count", "row_signbit_count", "row_negative_count", "row_positive_ratio", "row_signbit_ratio", "row_negative_ratio", "row_any_positive", "row_any_signbit", "row_any_positive_zero", "row_any_negative_zero", "row_first_positive_zero_index", "row_last_signbit_index", "row_first_positive_index", "row_last_negative_index", "metric_cum_positive", "id_cum_positive", "unsigned_cum_positive", "flag_cum_positive", "metric_cum_first_positive", "id_cum_first_positive", "unsigned_cum_first_positive", "flag_cum_first_positive", "metric_prefix_last_signbit", "id_prefix_last_signbit", "unsigned_prefix_last_signbit", "flag_prefix_last_signbit", "metric_cum_last_negative", "id_cum_last_negative", "unsigned_cum_last_negative", "flag_cum_last_negative", "metric_cum_negative", "id_cum_negative", "unsigned_cum_negative", "flag_cum_negative", "metric_cum_poszero", "id_cum_poszero", "unsigned_cum_poszero", "flag_cum_poszero", "metric_cum_first_poszero", "id_cum_first_poszero", "unsigned_cum_first_poszero", "flag_cum_first_poszero", "metric_prefix_last_negzero", "id_prefix_last_negzero", "unsigned_prefix_last_negzero", "flag_prefix_last_negzero", "metric_cum_negzero", "id_cum_negzero", "unsigned_cum_negzero", "flag_cum_negzero", "metric_cum_signbit", "id_cum_signbit", "unsigned_cum_signbit", "flag_cum_signbit", "metric_cum_any_positive", "flag_cum_any_positive", "metric_prefix_all_signbit", "flag_prefix_all_signbit", "flag_cum_any_poszero", "flag_prefix_all_negative" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_positive_column(metric->metric_is_positive)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_negative_column(metric->metric_is_negative)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_signbit_column(metric->metric_signbit)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_signbit_column(id->id_signbit)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_positive_zero_column(metric->metric_is_positive_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_negative_zero_column(metric->metric_is_negative_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_positive_zero_count([metric,id,unsigned,flag]->row_positive_zero_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_negative_zero_count([metric,id,unsigned,flag]->row_negative_zero_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_positive_zero_ratio([metric,id,unsigned,flag]->row_positive_zero_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_negative_zero_ratio([metric,id,unsigned,flag]->row_negative_zero_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_positive_count([metric,id,unsigned,flag]->row_positive_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_signbit_count([metric,id,unsigned,flag]->row_signbit_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_negative_count([metric,id,unsigned,flag]->row_negative_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_positive_ratio([metric,id,unsigned,flag]->row_positive_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_signbit_ratio([metric,id,unsigned,flag]->row_signbit_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_negative_ratio([metric,id,unsigned,flag]->row_negative_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_positive([metric,id,unsigned,flag]->row_any_positive)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_signbit([metric,id,unsigned,flag]->row_any_signbit)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_positive_zero([metric,id,unsigned,flag]->row_any_positive_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_negative_zero([metric,id,unsigned,flag]->row_any_negative_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_first_positive_zero_index([metric,id,unsigned,flag]->row_first_positive_zero_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_last_signbit_index([metric,id,unsigned,flag]->row_last_signbit_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_first_positive_index([metric,id,unsigned,flag]->row_first_positive_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_last_negative_index([metric,id,unsigned,flag]->row_last_negative_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_positive_count([metric,id,unsigned,flag]->[metric_cum_positive,id_cum_positive,unsigned_cum_positive,flag_cum_positive])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_first_positive_index([metric,id,unsigned,flag]->[metric_cum_first_positive,id_cum_first_positive,unsigned_cum_first_positive,flag_cum_first_positive])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_last_signbit_index([metric,id,unsigned,flag]->[metric_prefix_last_signbit,id_prefix_last_signbit,unsigned_prefix_last_signbit,flag_prefix_last_signbit])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_last_negative_index([metric,id,unsigned,flag]->[metric_cum_last_negative,id_cum_last_negative,unsigned_cum_last_negative,flag_cum_last_negative])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_negative_ratio([metric,id,unsigned,flag]->[metric_cum_negative,id_cum_negative,unsigned_cum_negative,flag_cum_negative])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_positive_zero_count([metric,id,unsigned,flag]->[metric_cum_poszero,id_cum_poszero,unsigned_cum_poszero,flag_cum_poszero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_first_positive_zero_index([metric,id,unsigned,flag]->[metric_cum_first_poszero,id_cum_first_poszero,unsigned_cum_first_poszero,flag_cum_first_poszero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_last_negative_zero_index([metric,id,unsigned,flag]->[metric_prefix_last_negzero,id_prefix_last_negzero,unsigned_prefix_last_negzero,flag_prefix_last_negzero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_negative_zero_ratio([metric,id,unsigned,flag]->[metric_cum_negzero,id_cum_negzero,unsigned_cum_negzero,flag_cum_negzero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_signbit_ratio([metric,id,unsigned,flag]->[metric_cum_signbit,id_cum_signbit,unsigned_cum_signbit,flag_cum_signbit])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_any_positive([metric,id,unsigned,flag]->[metric_cum_any_positive,id_cum_any_positive,unsigned_cum_any_positive,flag_cum_any_positive])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_all_signbit([metric,id,unsigned,flag]->[metric_prefix_all_signbit,id_prefix_all_signbit,unsigned_prefix_all_signbit,flag_prefix_all_signbit])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_any_positive_zero([metric,id,unsigned,flag]->[metric_cum_any_poszero,id_cum_any_poszero,unsigned_cum_any_poszero,flag_cum_any_poszero])") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_cumulative_all_negative([metric,id,unsigned,flag]->[metric_prefix_all_negative,id_prefix_all_negative,unsigned_prefix_all_negative,flag_prefix_all_negative])") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 73), result.width());
    const metric_is_positive = try (try result.column("metric_is_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_positive);
    const metric_is_negative = try (try result.column("metric_is_negative")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_negative);
    const metric_signbit = try (try result.column("metric_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_signbit);
    const id_signbit = try (try result.column("id_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_signbit);
    const metric_is_positive_zero = try (try result.column("metric_is_positive_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_positive_zero);
    const metric_is_negative_zero = try (try result.column("metric_is_negative_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_negative_zero);
    const id_is_positive = try (try result.column("id_is_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_positive);
    const unsigned_is_negative = try (try result.column("unsigned_is_negative")).bool.toOwnedSlice(gpa);
    defer gpa.free(unsigned_is_negative);
    const flag_is_positive = try (try result.column("flag_is_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_is_positive);
    const row_positive_zero_count = try (try result.column("row_positive_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_zero_count);
    const row_negative_zero_count = try (try result.column("row_negative_zero_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_zero_count);
    const row_positive_zero_ratio = try (try result.column("row_positive_zero_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_zero_ratio);
    const row_negative_zero_ratio = try (try result.column("row_negative_zero_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_zero_ratio);
    const row_positive_count = try (try result.column("row_positive_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_count);
    const row_signbit_count = try (try result.column("row_signbit_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_signbit_count);
    const row_negative_count = try (try result.column("row_negative_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_count);
    const row_positive_ratio = try (try result.column("row_positive_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_ratio);
    const row_signbit_ratio = try (try result.column("row_signbit_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_signbit_ratio);
    const row_negative_ratio = try (try result.column("row_negative_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_ratio);
    const row_any_positive = try (try result.column("row_any_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_positive);
    const row_any_signbit = try (try result.column("row_any_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_signbit);
    const row_any_positive_zero = try (try result.column("row_any_positive_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_positive_zero);
    const row_any_negative_zero = try (try result.column("row_any_negative_zero")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_negative_zero);
    const row_first_positive_zero = try (try result.column("row_first_positive_zero_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_zero);
    const row_first_positive_zero_validity = try (try result.column("row_first_positive_zero_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_zero_validity);
    const row_last_signbit = try (try result.column("row_last_signbit_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_signbit);
    const row_last_signbit_validity = try (try result.column("row_last_signbit_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_signbit_validity);
    const row_first_positive = try (try result.column("row_first_positive_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive);
    const row_first_positive_validity = try (try result.column("row_first_positive_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_validity);
    const row_last_negative = try (try result.column("row_last_negative_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_negative);
    const row_last_negative_validity = try (try result.column("row_last_negative_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_negative_validity);
    const metric_cum_positive = try (try result.column("metric_cum_positive")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_positive);
    const unsigned_cum_positive = try (try result.column("unsigned_cum_positive")).i64.toOwnedSlice(gpa);
    defer gpa.free(unsigned_cum_positive);
    const metric_cum_first_positive = try (try result.column("metric_cum_first_positive")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_positive);
    const metric_cum_first_positive_validity = try (try result.column("metric_cum_first_positive")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_positive_validity);
    const flag_cum_first_positive = try (try result.column("flag_cum_first_positive")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_positive);
    const flag_cum_first_positive_validity = try (try result.column("flag_cum_first_positive")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_positive_validity);
    const id_prefix_last_signbit = try (try result.column("id_prefix_last_signbit")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_signbit);
    const id_prefix_last_signbit_validity = try (try result.column("id_prefix_last_signbit")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_signbit_validity);
    const flag_prefix_last_signbit = try (try result.column("flag_prefix_last_signbit")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_signbit);
    const flag_prefix_last_signbit_validity = try (try result.column("flag_prefix_last_signbit")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_signbit_validity);
    const metric_cum_last_negative = try (try result.column("metric_cum_last_negative")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_last_negative);
    const metric_cum_last_negative_validity = try (try result.column("metric_cum_last_negative")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_last_negative_validity);
    const flag_cum_last_negative = try (try result.column("flag_cum_last_negative")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_last_negative);
    const flag_cum_last_negative_validity = try (try result.column("flag_cum_last_negative")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_last_negative_validity);
    const id_cum_negative = try (try result.column("id_cum_negative")).f64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_negative);
    const flag_cum_negative = try (try result.column("flag_cum_negative")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_negative);
    const metric_cum_poszero = try (try result.column("metric_cum_poszero")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_poszero);
    const flag_cum_poszero = try (try result.column("flag_cum_poszero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_poszero);
    const metric_cum_first_poszero = try (try result.column("metric_cum_first_poszero")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_poszero);
    const metric_cum_first_poszero_validity = try (try result.column("metric_cum_first_poszero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_poszero_validity);
    const flag_cum_first_poszero = try (try result.column("flag_cum_first_poszero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_poszero);
    const flag_cum_first_poszero_validity = try (try result.column("flag_cum_first_poszero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_first_poszero_validity);
    const metric_prefix_last_negzero = try (try result.column("metric_prefix_last_negzero")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_last_negzero);
    const metric_prefix_last_negzero_validity = try (try result.column("metric_prefix_last_negzero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_last_negzero_validity);
    const flag_prefix_last_negzero = try (try result.column("flag_prefix_last_negzero")).i64.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_negzero);
    const flag_prefix_last_negzero_validity = try (try result.column("flag_prefix_last_negzero")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_last_negzero_validity);
    const metric_cum_negzero = try (try result.column("metric_cum_negzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_negzero);
    const flag_cum_negzero = try (try result.column("flag_cum_negzero")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_negzero);
    const metric_cum_signbit = try (try result.column("metric_cum_signbit")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_signbit);
    const flag_cum_signbit = try (try result.column("flag_cum_signbit")).f64.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_signbit);
    const metric_cum_any_positive = try (try result.column("metric_cum_any_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_positive);
    const metric_cum_any_positive_validity = try (try result.column("metric_cum_any_positive")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_positive_validity);
    const flag_cum_any_positive = try (try result.column("flag_cum_any_positive")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_any_positive);
    const metric_prefix_all_signbit = try (try result.column("metric_prefix_all_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_all_signbit);
    const metric_prefix_all_signbit_validity = try (try result.column("metric_prefix_all_signbit")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_all_signbit_validity);
    const flag_prefix_all_signbit = try (try result.column("flag_prefix_all_signbit")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_all_signbit);
    const flag_cum_any_poszero = try (try result.column("flag_cum_any_poszero")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_cum_any_poszero);
    const flag_prefix_all_negative = try (try result.column("flag_prefix_all_negative")).bool.toOwnedSlice(gpa);
    defer gpa.free(flag_prefix_all_negative);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false, true, false, false }, metric_is_positive);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false, false, true, false }, metric_is_negative);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false, true, false }, metric_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, false, false, true, false }, id_signbit);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, metric_is_positive_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false, false, false, false }, metric_is_negative_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true, false, false, true }, id_is_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false, false, false }, unsigned_is_negative);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false, false, false }, flag_is_positive);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0, 0 }, row_positive_zero_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 0, 0, 0, 0 }, row_negative_zero_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0 }, row_positive_zero_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, row_negative_zero_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 2, 1, 2, 1, 1 }, row_positive_count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 0, 1, 0, 0, 2, 0 }, row_signbit_count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 0, 0, 1, 0, 0, 2, 0 }, row_negative_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.25, 0.25, 0.5, 0.25, 0.5, 0.25, 1.0 / 3.0 }, row_positive_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.25, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0 }, row_signbit_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0 }, row_negative_ratio);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true, true }, row_any_positive);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, false, true, false }, row_any_signbit);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, row_any_positive_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false, false, false, false }, row_any_negative_zero);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, row_first_positive_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, row_first_positive_zero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, row_last_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, false, true, false }, row_last_signbit_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 1, 0, 1, 0, 2, 1 }, row_first_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true, true }, row_first_positive_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, row_last_negative);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, false, false, true, false }, row_last_negative_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 1, 0, 1, 0, 0 }, metric_cum_positive);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 2, 1, 2, 1, 1 }, unsigned_cum_positive);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, metric_cum_first_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false, true, false, false }, metric_cum_first_positive_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 1, 0, 1, 0, 2, 1 }, flag_cum_first_positive);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true, true }, flag_cum_first_positive_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, id_prefix_last_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, false, true, false }, id_prefix_last_signbit_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, flag_prefix_last_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false, false, true, false }, flag_prefix_last_signbit_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, metric_cum_last_negative);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false, false, true, false }, metric_cum_last_negative_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0, 0, 1, 0 }, flag_cum_last_negative);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, false, false, true, false }, flag_cum_last_negative_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0 }, id_cum_negative);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0 }, flag_cum_negative);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0, 0 }, metric_cum_poszero);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0, 0, 0, 0 }, flag_cum_poszero);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, metric_cum_first_poszero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, metric_cum_first_poszero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, flag_cum_first_poszero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, flag_cum_first_poszero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, metric_prefix_last_negzero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false, false, false, false }, metric_prefix_last_negzero_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, flag_prefix_last_negzero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false, false, false, false }, flag_prefix_last_negzero_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, metric_cum_negzero);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, flag_cum_negzero);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 }, metric_cum_signbit);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.25, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0 }, flag_cum_signbit);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false, true, false, false }, metric_cum_any_positive);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true, true, false }, metric_cum_any_positive_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true, true, true, true, true }, flag_cum_any_positive);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false, true, false }, metric_prefix_all_signbit);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, true, true, false }, metric_prefix_all_signbit_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false, false, false }, flag_prefix_all_signbit);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false, false, false, false }, flag_cum_any_poszero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, false, false, false }, flag_prefix_all_negative);

    var filter_positive_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_positive_plan.deinit();
    try filter_positive_plan.filterPositivesColumn("metric");
    const filter_positive_explain = try filter_positive_plan.explain(gpa);
    defer gpa.free(filter_positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_positive_explain, "filter_positives_column(metric)") != null);
    var filtered_positive_rows = try filter_positive_plan.collect();
    defer filtered_positive_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered_positive_rows.height());
    const filtered_positive_metric = try (try filtered_positive_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_positive_metric);
    try std.testing.expectEqual(@as(f64, 3.0), filtered_positive_metric[0]);
    try std.testing.expect(std.math.isPositiveInf(filtered_positive_metric[1]));

    var filter_signbit_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_signbit_plan.deinit();
    try filter_signbit_plan.filterSignBitsColumn("metric");
    const filter_signbit_explain = try filter_signbit_plan.explain(gpa);
    defer gpa.free(filter_signbit_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_signbit_explain, "filter_signbits_column(metric)") != null);
    var filtered_signbit_rows = try filter_signbit_plan.collect();
    defer filtered_signbit_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), filtered_signbit_rows.height());
    const filtered_signbit_metric = try (try filtered_signbit_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_signbit_metric);
    try std.testing.expectEqual(@as(f64, -2.0), filtered_signbit_metric[0]);
    try std.testing.expectEqual(@as(f64, -0.0), filtered_signbit_metric[1]);
    try std.testing.expect(std.math.isNegativeInf(filtered_signbit_metric[2]));

    var filter_positive_zero_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_positive_zero_plan.deinit();
    try filter_positive_zero_plan.filterPositiveZerosColumn("metric");
    const filter_positive_zero_explain = try filter_positive_zero_plan.explain(gpa);
    defer gpa.free(filter_positive_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_positive_zero_explain, "filter_positive_zeros_column(metric)") != null);
    var filtered_positive_zero_rows = try filter_positive_zero_plan.collect();
    defer filtered_positive_zero_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_positive_zero_rows.height());
    const filtered_positive_zero_metric = try (try filtered_positive_zero_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_positive_zero_metric);
    try std.testing.expectEqual(@as(f64, 0.0), filtered_positive_zero_metric[0]);

    var drop_negative_zero_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_negative_zero_plan.deinit();
    try drop_negative_zero_plan.dropNegativeZerosColumn("metric");
    const drop_negative_zero_explain = try drop_negative_zero_plan.explain(gpa);
    defer gpa.free(drop_negative_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_negative_zero_explain, "drop_negative_zeros[metric]") != null);
    var dropped_negative_zero_rows = try drop_negative_zero_plan.collect();
    defer dropped_negative_zero_rows.deinit();
    try std.testing.expectEqual(@as(usize, 7), dropped_negative_zero_rows.height());
    const dropped_negative_zero_metric = try (try dropped_negative_zero_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_negative_zero_metric);
    try std.testing.expectEqual(@as(f64, -2.0), dropped_negative_zero_metric[0]);
    try std.testing.expectEqual(@as(f64, 0.0), dropped_negative_zero_metric[1]);
    try std.testing.expectEqual(@as(f64, 3.0), dropped_negative_zero_metric[2]);
    try std.testing.expect(std.math.isNan(dropped_negative_zero_metric[3]));
    try std.testing.expect(std.math.isPositiveInf(dropped_negative_zero_metric[4]));
    try std.testing.expect(std.math.isNegativeInf(dropped_negative_zero_metric[5]));
    try std.testing.expectEqual(@as(f64, 9.0), dropped_negative_zero_metric[6]);

    var drop_negative_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_negative_plan.deinit();
    try drop_negative_plan.dropNegativesColumn("id");
    const drop_negative_explain = try drop_negative_plan.explain(gpa);
    defer gpa.free(drop_negative_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_negative_explain, "drop_negatives[id]") != null);
    var dropped_negative_rows = try drop_negative_plan.collect();
    defer dropped_negative_rows.deinit();
    try std.testing.expectEqual(@as(usize, 5), dropped_negative_rows.height());
    const dropped_negative_id = try (try dropped_negative_rows.column("id")).i64.toOwnedSlice(gpa);
    defer gpa.free(dropped_negative_id);
    try std.testing.expectEqualSlices(i64, &.{ 0, 4, 6, 0, 8 }, dropped_negative_id);

    var invalid_prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_prefix_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_prefix_plan.withRowPrefixNegativeCount(&.{"metric"}, &.{ "metric_cum_negative", "extra_cum_negative" }));
    try std.testing.expectError(error.LengthMismatch, invalid_prefix_plan.withRowPrefixSignBitRatio(&.{"metric"}, &.{ "metric_cum_signbit", "extra_cum_signbit" }));
    try std.testing.expectError(error.LengthMismatch, invalid_prefix_plan.withRowPrefixAllSignBit(&.{"metric"}, &.{ "metric_all_signbit", "extra_all_signbit" }));

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isPositiveColumn("missing", "missing_is_positive");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());

    var invalid_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_index_plan.deinit();
    try invalid_index_plan.withRowFirstPositiveIndex(&.{"missing"}, "bad_positive_index");
    try std.testing.expectError(error.ColumnNotFound, invalid_index_plan.collect());

    var invalid_signbit_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_signbit_index_plan.deinit();
    try invalid_signbit_index_plan.withRowFirstSignBitIndex(&.{"missing"}, "bad_signbit_index");
    try std.testing.expectError(error.ColumnNotFound, invalid_signbit_index_plan.collect());

    var invalid_cumulative_signed_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_signed_index_plan.deinit();
    try invalid_cumulative_signed_index_plan.withRowCumulativeFirstPositiveIndex(&.{"missing"}, &.{"bad_positive_index"});
    try std.testing.expectError(error.ColumnNotFound, invalid_cumulative_signed_index_plan.collect());

    var invalid_cumulative_signed_index_length_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_signed_index_length_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_signed_index_length_plan.withRowPrefixLastSignBitIndex(&.{"metric"}, &.{ "metric_last_signbit", "extra_last_signbit" }));

    var invalid_cumulative_signed_zero_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_signed_zero_index_plan.deinit();
    try invalid_cumulative_signed_zero_index_plan.withRowCumulativeFirstPositiveZeroIndex(&.{"missing"}, &.{"bad_poszero_index"});
    try std.testing.expectError(error.ColumnNotFound, invalid_cumulative_signed_zero_index_plan.collect());

    var invalid_cumulative_signed_zero_index_length_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_cumulative_signed_zero_index_length_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_cumulative_signed_zero_index_length_plan.withRowPrefixLastNegativeZeroIndex(&.{"metric"}, &.{ "metric_last_negzero", "extra_last_negzero" }));

    var invalid_filter_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_filter_plan.deinit();
    try invalid_filter_plan.filterNegativesColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_filter_plan.collect());

    var invalid_signed_zero_filter_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_signed_zero_filter_plan.deinit();
    try invalid_signed_zero_filter_plan.filterPositiveZerosColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_signed_zero_filter_plan.collect());

    var invalid_signbit_filter_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_signbit_filter_plan.deinit();
    try invalid_signbit_filter_plan.dropSignBitsColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_signbit_filter_plan.collect());
}

test "device lazy frame derives NaN and finite predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.nan(f64), std.math.inf(f64), 7.0 }, &.{ true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40 }, .cpu);
    defer id.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.isNanColumn("metric", "metric_is_nan");
    try plan.isFiniteColumn("metric", "metric_is_finite");
    try plan.isNonFiniteColumn("metric", "metric_is_non_finite");
    try plan.isInfColumn("metric", "metric_is_inf");
    try plan.isFiniteColumn("id", "id_is_finite");
    try plan.isNonFiniteColumn("id", "id_is_non_finite");
    try plan.select(&.{ "metric_is_nan", "metric_is_finite", "metric_is_non_finite", "metric_is_inf", "id_is_finite", "id_is_non_finite" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_nan_column(metric->metric_is_nan)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_finite_column(metric->metric_is_finite)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_non_finite_column(metric->metric_is_non_finite)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_inf_column(metric->metric_is_inf)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 6), result.width());
    const metric_is_nan = try (try result.column("metric_is_nan")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_nan);
    const metric_is_finite = try (try result.column("metric_is_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_finite);
    const metric_is_non_finite = try (try result.column("metric_is_non_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_non_finite);
    const metric_is_inf = try (try result.column("metric_is_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_inf);
    const id_is_finite = try (try result.column("id_is_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_finite);
    const id_is_non_finite = try (try result.column("id_is_non_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_non_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, metric_is_nan);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, metric_is_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, metric_is_non_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, metric_is_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, id_is_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, id_is_non_finite);

    var fill_nan_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_nan_plan.deinit();
    try fill_nan_plan.withColumnFillNaN("metric_no_nan", "metric", f64, -2.0);
    try fill_nan_plan.withColumnNullIfNaN("metric_nan_null", "metric");
    try fill_nan_plan.fillNaNColumn("metric", f64, -1.0);
    const fill_nan_explain = try fill_nan_plan.explain(gpa);
    defer gpa.free(fill_nan_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_nan_explain, "copy_column(metric->metric_no_nan)") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_nan_explain, "fill_nan_column(metric_no_nan=scalar:f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_nan_explain, "copy_column(metric->metric_nan_null)") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_nan_explain, "null_if_nan_column(metric_nan_null)") != null);
    try std.testing.expect(std.mem.indexOf(u8, fill_nan_explain, "fill_nan_column(metric=scalar:f64)") != null);
    var filled_nan = try fill_nan_plan.collect();
    defer filled_nan.deinit();
    const filled_metric = try (try filled_nan.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_metric);
    const filled_metric_validity = try (try filled_nan.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_metric_validity);
    try std.testing.expectEqual(@as(f64, 1.0), filled_metric[0]);
    try std.testing.expectEqual(@as(f64, -1.0), filled_metric[1]);
    try std.testing.expect(std.math.isInf(filled_metric[2]));
    try std.testing.expectEqual(@as(f64, 7.0), filled_metric[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, filled_metric_validity);
    const metric_no_nan = try (try filled_nan.column("metric_no_nan")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_no_nan);
    const metric_no_nan_validity = try (try filled_nan.column("metric_no_nan")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_no_nan_validity);
    try std.testing.expectEqual(@as(f64, 1.0), metric_no_nan[0]);
    try std.testing.expectEqual(@as(f64, -2.0), metric_no_nan[1]);
    try std.testing.expect(std.math.isInf(metric_no_nan[2]));
    try std.testing.expectEqual(@as(f64, 7.0), metric_no_nan[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, metric_no_nan_validity);
    const metric_nan_null_validity = try (try filled_nan.column("metric_nan_null")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_nan_null_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, metric_nan_null_validity);

    var fill_nan_mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_nan_mismatch_plan.deinit();
    try fill_nan_mismatch_plan.fillNaNColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, fill_nan_mismatch_plan.collect());

    var fill_inf_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_inf_plan.deinit();
    try fill_inf_plan.fillInfColumn("metric", f64, -9.0);
    const fill_inf_explain = try fill_inf_plan.explain(gpa);
    defer gpa.free(fill_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_inf_explain, "fill_inf_column(metric=scalar:f64)") != null);
    var filled_inf = try fill_inf_plan.collect();
    defer filled_inf.deinit();
    const filled_inf_metric = try (try filled_inf.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_inf_metric);
    const filled_inf_validity = try (try filled_inf.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_inf_validity);
    try std.testing.expectEqual(@as(f64, 1.0), filled_inf_metric[0]);
    try std.testing.expect(std.math.isNan(filled_inf_metric[1]));
    try std.testing.expectEqual(@as(f64, -9.0), filled_inf_metric[2]);
    try std.testing.expectEqual(@as(f64, 7.0), filled_inf_metric[3]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, filled_inf_validity);

    var fill_inf_mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_inf_mismatch_plan.deinit();
    try fill_inf_mismatch_plan.fillInfColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, fill_inf_mismatch_plan.collect());

    var fill_non_finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_non_finite_plan.deinit();
    try fill_non_finite_plan.fillNonFiniteColumn("metric", f64, -5.0);
    const fill_non_finite_explain = try fill_non_finite_plan.explain(gpa);
    defer gpa.free(fill_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, fill_non_finite_explain, "fill_non_finite_column(metric=scalar:f64)") != null);
    var filled_non_finite = try fill_non_finite_plan.collect();
    defer filled_non_finite.deinit();
    const filled_non_finite_metric = try (try filled_non_finite.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_non_finite_metric);
    const filled_non_finite_validity = try (try filled_non_finite.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filled_non_finite_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, -5.0, -5.0, 7.0 }, filled_non_finite_metric);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, false }, filled_non_finite_validity);

    var fill_non_finite_mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer fill_non_finite_mismatch_plan.deinit();
    try fill_non_finite_mismatch_plan.fillNonFiniteColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, fill_non_finite_mismatch_plan.collect());

    var select_nan_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_nan_columns_plan.deinit();
    try select_nan_columns_plan.selectColumnsWithNaNs();
    const select_nan_columns_explain = try select_nan_columns_plan.explain(gpa);
    defer gpa.free(select_nan_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_nan_columns_explain, "select_columns_with_nans") != null);
    var nan_columns = try select_nan_columns_plan.collect();
    defer nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), nan_columns.columnIndex("metric"));

    var select_non_nan_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_non_nan_columns_plan.deinit();
    try select_non_nan_columns_plan.selectColumnsWithoutNaNs();
    const select_non_nan_columns_explain = try select_non_nan_columns_plan.explain(gpa);
    defer gpa.free(select_non_nan_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_non_nan_columns_explain, "select_columns_without_nans") != null);
    var non_nan_columns = try select_non_nan_columns_plan.collect();
    defer non_nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), non_nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), non_nan_columns.columnIndex("id"));

    var drop_nan_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_nan_columns_plan.deinit();
    try drop_nan_columns_plan.dropColumnsWithNaNs();
    const drop_nan_columns_explain = try drop_nan_columns_plan.explain(gpa);
    defer gpa.free(drop_nan_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_nan_columns_explain, "drop_columns_with_nans") != null);
    var drop_nan_columns = try drop_nan_columns_plan.collect();
    defer drop_nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_nan_columns.columnIndex("id"));

    var drop_non_nan_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_nan_columns_plan.deinit();
    try drop_non_nan_columns_plan.dropColumnsWithoutNaNs();
    const drop_non_nan_columns_explain = try drop_non_nan_columns_plan.explain(gpa);
    defer gpa.free(drop_non_nan_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_nan_columns_explain, "drop_columns_without_nans") != null);
    var drop_non_nan_columns = try drop_non_nan_columns_plan.collect();
    defer drop_non_nan_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_non_nan_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_nan_columns.columnIndex("metric"));

    var select_inf_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_inf_columns_plan.deinit();
    try select_inf_columns_plan.selectColumnsWithInfs();
    const select_inf_columns_explain = try select_inf_columns_plan.explain(gpa);
    defer gpa.free(select_inf_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_inf_columns_explain, "select_columns_with_infs") != null);
    var inf_columns = try select_inf_columns_plan.collect();
    defer inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), inf_columns.columnIndex("metric"));

    var select_non_inf_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_non_inf_columns_plan.deinit();
    try select_non_inf_columns_plan.selectColumnsWithoutInfs();
    const select_non_inf_columns_explain = try select_non_inf_columns_plan.explain(gpa);
    defer gpa.free(select_non_inf_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_non_inf_columns_explain, "select_columns_without_infs") != null);
    var non_inf_columns = try select_non_inf_columns_plan.collect();
    defer non_inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), non_inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), non_inf_columns.columnIndex("id"));

    var drop_inf_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_inf_columns_plan.deinit();
    try drop_inf_columns_plan.dropColumnsWithInfs();
    const drop_inf_columns_explain = try drop_inf_columns_plan.explain(gpa);
    defer gpa.free(drop_inf_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_inf_columns_explain, "drop_columns_with_infs") != null);
    var drop_inf_columns = try drop_inf_columns_plan.collect();
    defer drop_inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_inf_columns.columnIndex("id"));

    var drop_non_inf_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_inf_columns_plan.deinit();
    try drop_non_inf_columns_plan.dropColumnsWithoutInfs();
    const drop_non_inf_columns_explain = try drop_non_inf_columns_plan.explain(gpa);
    defer gpa.free(drop_non_inf_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_inf_columns_explain, "drop_columns_without_infs") != null);
    var drop_non_inf_columns = try drop_non_inf_columns_plan.collect();
    defer drop_non_inf_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_non_inf_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_inf_columns.columnIndex("metric"));

    var select_non_finite_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_non_finite_columns_plan.deinit();
    try select_non_finite_columns_plan.selectColumnsWithNonFinites();
    const select_non_finite_columns_explain = try select_non_finite_columns_plan.explain(gpa);
    defer gpa.free(select_non_finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_non_finite_columns_explain, "select_columns_with_non_finites") != null);
    var non_finite_columns = try select_non_finite_columns_plan.collect();
    defer non_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), non_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), non_finite_columns.columnIndex("metric"));

    var select_finite_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_finite_columns_plan.deinit();
    try select_finite_columns_plan.selectColumnsWithoutNonFinites();
    const select_finite_columns_explain = try select_finite_columns_plan.explain(gpa);
    defer gpa.free(select_finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_finite_columns_explain, "select_columns_without_non_finites") != null);
    var finite_columns = try select_finite_columns_plan.collect();
    defer finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), finite_columns.columnIndex("id"));

    var drop_non_finite_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_finite_columns_plan.deinit();
    try drop_non_finite_columns_plan.dropColumnsWithNonFinites();
    const drop_non_finite_columns_explain = try drop_non_finite_columns_plan.explain(gpa);
    defer gpa.free(drop_non_finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_finite_columns_explain, "drop_columns_with_non_finites") != null);
    var drop_non_finite_columns = try drop_non_finite_columns_plan.collect();
    defer drop_non_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_non_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_non_finite_columns.columnIndex("id"));

    var drop_finite_columns_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_finite_columns_plan.deinit();
    try drop_finite_columns_plan.dropColumnsWithoutNonFinites();
    const drop_finite_columns_explain = try drop_finite_columns_plan.explain(gpa);
    defer gpa.free(drop_finite_columns_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_finite_columns_explain, "drop_columns_without_non_finites") != null);
    var drop_finite_columns = try drop_finite_columns_plan.collect();
    defer drop_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_finite_columns.columnIndex("metric"));

    var drop_nan_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_nan_plan.deinit();
    try drop_nan_plan.dropNaNsColumn("metric");
    const drop_nan_explain = try drop_nan_plan.explain(gpa);
    defer gpa.free(drop_nan_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_nan_explain, "drop_nans[metric]") != null);
    var dropped_nan_rows = try drop_nan_plan.collect();
    defer dropped_nan_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped_nan_rows.height());
    const dropped_nan_metric = try (try dropped_nan_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_nan_metric);
    try std.testing.expect(!std.math.isNan(dropped_nan_metric[0]));
    try std.testing.expect(std.math.isInf(dropped_nan_metric[1]));
    try std.testing.expectEqual(@as(f64, 7.0), dropped_nan_metric[2]);

    var filter_nan_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_nan_plan.deinit();
    try filter_nan_plan.filterNaNsColumn("metric");
    const filter_nan_explain = try filter_nan_plan.explain(gpa);
    defer gpa.free(filter_nan_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_nan_explain, "filter_nans_column(metric)") != null);
    var filtered_nan_rows = try filter_nan_plan.collect();
    defer filtered_nan_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_nan_rows.height());
    const filtered_nan_metric = try (try filtered_nan_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_nan_metric);
    try std.testing.expect(std.math.isNan(filtered_nan_metric[0]));

    var drop_inf_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_inf_plan.deinit();
    try drop_inf_plan.dropInfsColumn("metric");
    const drop_inf_explain = try drop_inf_plan.explain(gpa);
    defer gpa.free(drop_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_inf_explain, "drop_infs[metric]") != null);
    var dropped_inf_rows = try drop_inf_plan.collect();
    defer dropped_inf_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped_inf_rows.height());
    const dropped_inf_metric = try (try dropped_inf_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_inf_metric);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_inf_metric[0]);
    try std.testing.expect(std.math.isNan(dropped_inf_metric[1]));
    try std.testing.expectEqual(@as(f64, 7.0), dropped_inf_metric[2]);

    var filter_inf_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_inf_plan.deinit();
    try filter_inf_plan.filterInfsColumn("metric");
    const filter_inf_explain = try filter_inf_plan.explain(gpa);
    defer gpa.free(filter_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_inf_explain, "filter_infs_column(metric)") != null);
    var filtered_inf_rows = try filter_inf_plan.collect();
    defer filtered_inf_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_inf_rows.height());
    const filtered_inf_metric = try (try filtered_inf_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_inf_metric);
    try std.testing.expect(std.math.isInf(filtered_inf_metric[0]));

    var drop_finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_finite_plan.deinit();
    try drop_finite_plan.dropFinitesColumn("metric");
    const drop_finite_explain = try drop_finite_plan.explain(gpa);
    defer gpa.free(drop_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_finite_explain, "drop_finites[metric]") != null);
    var dropped_finite_rows = try drop_finite_plan.collect();
    defer dropped_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped_finite_rows.height());
    const dropped_finite_metric = try (try dropped_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_finite_metric);
    const dropped_finite_validity = try (try dropped_finite_rows.column("metric")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(dropped_finite_validity);
    try std.testing.expect(std.math.isNan(dropped_finite_metric[0]));
    try std.testing.expect(std.math.isInf(dropped_finite_metric[1]));
    try std.testing.expectEqual(@as(f64, 7.0), dropped_finite_metric[2]);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false }, dropped_finite_validity);

    var filter_finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_finite_plan.deinit();
    try filter_finite_plan.filterFinitesColumn("metric");
    const filter_finite_explain = try filter_finite_plan.explain(gpa);
    defer gpa.free(filter_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_finite_explain, "filter_finites_column(metric)") != null);
    var filtered_finite_rows = try filter_finite_plan.collect();
    defer filtered_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_finite_rows.height());
    const filtered_finite_metric = try (try filtered_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_finite_metric);
    try std.testing.expectEqual(@as(f64, 1.0), filtered_finite_metric[0]);

    var drop_non_finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_non_finite_plan.deinit();
    try drop_non_finite_plan.dropNonFinitesColumn("metric");
    const drop_non_finite_explain = try drop_non_finite_plan.explain(gpa);
    defer gpa.free(drop_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_non_finite_explain, "drop_non_finites[metric]") != null);
    var dropped_non_finite_rows = try drop_non_finite_plan.collect();
    defer dropped_non_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), dropped_non_finite_rows.height());
    const dropped_non_finite_metric = try (try dropped_non_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_non_finite_metric);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_non_finite_metric[0]);
    try std.testing.expectEqual(@as(f64, 7.0), dropped_non_finite_metric[1]);

    var filter_non_finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_non_finite_plan.deinit();
    try filter_non_finite_plan.filterNonFinitesColumn("metric");
    const filter_non_finite_explain = try filter_non_finite_plan.explain(gpa);
    defer gpa.free(filter_non_finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_non_finite_explain, "filter_non_finites_column(metric)") != null);
    var filtered_non_finite_rows = try filter_non_finite_plan.collect();
    defer filtered_non_finite_rows.deinit();
    try std.testing.expectEqual(@as(usize, 2), filtered_non_finite_rows.height());
    const filtered_non_finite_metric = try (try filtered_non_finite_rows.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_non_finite_metric);
    try std.testing.expect(std.math.isNan(filtered_non_finite_metric[0]));
    try std.testing.expect(std.math.isInf(filtered_non_finite_metric[1]));

    var row_special_plan = try DeviceLazyFrame.init(gpa, table);
    defer row_special_plan.deinit();
    try row_special_plan.withRowNaNCount(&.{ "metric", "id" }, "row_nan_count");
    try row_special_plan.withRowInfCount(&.{}, "row_inf_count");
    try row_special_plan.withRowFiniteCount(&.{ "metric", "id" }, "row_finite_count");
    try row_special_plan.withRowNonFiniteCount(&.{}, "row_non_finite_count");
    try row_special_plan.withRowNaNRatio(&.{ "metric", "id" }, "row_nan_ratio");
    try row_special_plan.withRowInfRatio(&.{ "metric", "id" }, "row_inf_ratio");
    try row_special_plan.withRowFiniteRatio(&.{ "metric", "id" }, "row_finite_ratio");
    try row_special_plan.withRowNonFiniteRatio(&.{ "metric", "id" }, "row_non_finite_ratio");
    try row_special_plan.withRowAnyNaN(&.{ "metric", "id" }, "row_any_nan");
    try row_special_plan.withRowAllFinite(&.{ "metric", "id" }, "row_all_finite");
    try row_special_plan.withRowFirstNaNIndex(&.{ "metric", "id" }, "row_first_nan_index");
    try row_special_plan.withRowLastInfIndex(&.{ "metric", "id" }, "row_last_inf_index");
    try row_special_plan.withRowFirstFiniteIndex(&.{ "metric", "id" }, "row_first_finite_index");
    try row_special_plan.withRowLastNonFiniteIndex(&.{ "metric", "id" }, "row_last_non_finite_index");
    try row_special_plan.withRowCumulativeFirstNaNIndex(&.{ "id", "metric" }, &.{ "id_cum_first_nan", "metric_cum_first_nan" });
    try row_special_plan.withRowPrefixLastInfIndex(&.{ "id", "metric" }, &.{ "id_prefix_last_inf", "metric_prefix_last_inf" });
    try row_special_plan.withRowCumulativeFirstFiniteIndex(&.{ "metric", "id" }, &.{ "metric_cum_first_finite", "id_cum_first_finite" });
    try row_special_plan.withRowCumulativeAnyNaN(&.{ "id", "metric" }, &.{ "id_cum_any_nan", "metric_cum_any_nan" });
    try row_special_plan.withRowPrefixAllFinite(&.{ "metric", "id" }, &.{ "metric_prefix_all_finite", "id_prefix_all_finite" });
    try row_special_plan.withRowPrefixLastNonFiniteIndex(&.{ "metric", "id" }, &.{ "metric_prefix_last_non_finite", "id_prefix_last_non_finite" });
    try row_special_plan.withRowCumulativeNaNCount(&.{ "metric", "id" }, &.{ "metric_cum_nan", "id_cum_nan" });
    try row_special_plan.withRowPrefixInfCount(&.{ "metric", "id" }, &.{ "metric_cum_inf", "id_cum_inf" });
    try row_special_plan.withRowCumulativeFiniteRatio(&.{ "metric", "id" }, &.{ "metric_cum_finite_ratio", "id_cum_finite_ratio" });
    try row_special_plan.withRowPrefixNonFiniteRatio(&.{ "metric", "id" }, &.{ "metric_cum_non_finite_ratio", "id_cum_non_finite_ratio" });
    try row_special_plan.select(&.{ "row_nan_count", "row_inf_count", "row_finite_count", "row_non_finite_count", "row_nan_ratio", "row_inf_ratio", "row_finite_ratio", "row_non_finite_ratio", "row_any_nan", "row_all_finite", "row_first_nan_index", "row_last_inf_index", "row_first_finite_index", "row_last_non_finite_index", "id_cum_first_nan", "metric_cum_first_nan", "id_prefix_last_inf", "metric_prefix_last_inf", "metric_cum_first_finite", "id_cum_first_finite", "metric_cum_any_nan", "id_prefix_all_finite", "metric_prefix_last_non_finite", "id_prefix_last_non_finite", "metric_cum_nan", "id_cum_nan", "metric_cum_inf", "id_cum_inf", "metric_cum_finite_ratio", "id_cum_finite_ratio", "metric_cum_non_finite_ratio", "id_cum_non_finite_ratio" });
    const row_special_explain = try row_special_plan.explain(gpa);
    defer gpa.free(row_special_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_nan_count([metric,id]->row_nan_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_inf_count([]->row_inf_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_finite_count([metric,id]->row_finite_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_non_finite_count([]->row_non_finite_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_nan_ratio([metric,id]->row_nan_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_inf_ratio([metric,id]->row_inf_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_finite_ratio([metric,id]->row_finite_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_non_finite_ratio([metric,id]->row_non_finite_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_any_nan([metric,id]->row_any_nan)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_all_finite([metric,id]->row_all_finite)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_first_nan_index([metric,id]->row_first_nan_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_last_inf_index([metric,id]->row_last_inf_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_first_finite_index([metric,id]->row_first_finite_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_last_non_finite_index([metric,id]->row_last_non_finite_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_first_nan_index([id,metric]->[id_cum_first_nan,metric_cum_first_nan])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_last_inf_index([id,metric]->[id_prefix_last_inf,metric_prefix_last_inf])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_first_finite_index([metric,id]->[metric_cum_first_finite,id_cum_first_finite])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_any_nan([id,metric]->[id_cum_any_nan,metric_cum_any_nan])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_all_finite([metric,id]->[metric_prefix_all_finite,id_prefix_all_finite])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_last_non_finite_index([metric,id]->[metric_prefix_last_non_finite,id_prefix_last_non_finite])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_nan_count([metric,id]->[metric_cum_nan,id_cum_nan])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_inf_count([metric,id]->[metric_cum_inf,id_cum_inf])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_finite_ratio([metric,id]->[metric_cum_finite_ratio,id_cum_finite_ratio])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_special_explain, "row_cumulative_non_finite_ratio([metric,id]->[metric_cum_non_finite_ratio,id_cum_non_finite_ratio])") != null);
    var row_special = try row_special_plan.collect();
    defer row_special.deinit();
    const row_nan_count = try (try row_special.column("row_nan_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_nan_count);
    const row_inf_count = try (try row_special.column("row_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_inf_count);
    const row_finite_count = try (try row_special.column("row_finite_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_finite_count);
    const row_non_finite_count = try (try row_special.column("row_non_finite_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_non_finite_count);
    const row_nan_ratio = try (try row_special.column("row_nan_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_nan_ratio);
    const row_inf_ratio = try (try row_special.column("row_inf_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_inf_ratio);
    const row_finite_ratio = try (try row_special.column("row_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_finite_ratio);
    const row_non_finite_ratio = try (try row_special.column("row_non_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_non_finite_ratio);
    const row_any_nan = try (try row_special.column("row_any_nan")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_nan);
    const row_all_finite = try (try row_special.column("row_all_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_all_finite);
    const row_first_nan = try (try row_special.column("row_first_nan_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_nan);
    const row_first_nan_validity = try (try row_special.column("row_first_nan_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_nan_validity);
    const row_last_inf = try (try row_special.column("row_last_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_inf);
    const row_last_inf_validity = try (try row_special.column("row_last_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_inf_validity);
    const row_first_finite = try (try row_special.column("row_first_finite_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_finite);
    const row_first_finite_validity = try (try row_special.column("row_first_finite_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_finite_validity);
    const row_last_non_finite = try (try row_special.column("row_last_non_finite_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_non_finite);
    const row_last_non_finite_validity = try (try row_special.column("row_last_non_finite_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_non_finite_validity);
    const metric_cum_first_nan = try (try row_special.column("metric_cum_first_nan")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_nan);
    const metric_cum_first_nan_validity = try (try row_special.column("metric_cum_first_nan")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_first_nan_validity);
    const metric_prefix_last_inf = try (try row_special.column("metric_prefix_last_inf")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_last_inf);
    const metric_prefix_last_inf_validity = try (try row_special.column("metric_prefix_last_inf")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_prefix_last_inf_validity);
    const id_cum_first_finite = try (try row_special.column("id_cum_first_finite")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_first_finite);
    const id_cum_first_finite_validity = try (try row_special.column("id_cum_first_finite")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_cum_first_finite_validity);
    const metric_cum_any_nan = try (try row_special.column("metric_cum_any_nan")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_nan);
    const metric_cum_any_nan_validity = try (try row_special.column("metric_cum_any_nan")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_any_nan_validity);
    const id_prefix_all_finite = try (try row_special.column("id_prefix_all_finite")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_all_finite);
    const id_prefix_all_finite_validity = try (try row_special.column("id_prefix_all_finite")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_all_finite_validity);
    const id_prefix_last_non_finite = try (try row_special.column("id_prefix_last_non_finite")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_non_finite);
    const id_prefix_last_non_finite_validity = try (try row_special.column("id_prefix_last_non_finite")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_prefix_last_non_finite_validity);
    const id_cum_nan = try (try row_special.column("id_cum_nan")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_nan);
    const id_cum_inf = try (try row_special.column("id_cum_inf")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_inf);
    const id_cum_finite_ratio = try (try row_special.column("id_cum_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_finite_ratio);
    const metric_cum_non_finite_ratio = try (try row_special.column("metric_cum_non_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_non_finite_ratio);
    const id_cum_non_finite_ratio = try (try row_special.column("id_cum_non_finite_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_non_finite_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, row_nan_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0 }, row_inf_count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, row_finite_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 0 }, row_non_finite_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.0, 0.0 }, row_nan_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.5, 0.0 }, row_inf_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.5, 0.5, 1.0 }, row_finite_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.5, 0.0 }, row_non_finite_ratio);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, row_any_nan);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, row_all_finite);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, row_first_nan);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, row_first_nan_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, row_last_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, row_last_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, row_first_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, row_first_finite_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, row_last_non_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, row_last_non_finite_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, metric_cum_first_nan);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, metric_cum_first_nan_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0 }, metric_prefix_last_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false }, metric_prefix_last_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, id_cum_first_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, id_cum_first_finite_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, metric_cum_any_nan);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, metric_cum_any_nan_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true }, id_prefix_all_finite);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, id_prefix_all_finite_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, id_prefix_last_non_finite);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, id_prefix_last_non_finite_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, id_cum_nan);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0 }, id_cum_inf);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.5, 0.5, 0.5 }, id_cum_finite_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 1.0, 0.0 }, metric_cum_non_finite_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.5, 0.0 }, id_cum_non_finite_ratio);

    var signed_inf_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 5.0 }, &.{ true, true, true, true, false }, .cpu);
    defer signed_inf_metric.deinit();
    var signed_inf_peer = try DeviceColumn.fromSlice(f64, gpa, &.{ -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64) }, .cpu);
    defer signed_inf_peer.deinit();
    var signed_inf_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40, 50 }, .cpu);
    defer signed_inf_id.deinit();
    var signed_inf_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = signed_inf_metric },
        .{ .name = "peer", .data = signed_inf_peer },
        .{ .name = "id", .data = signed_inf_id },
    });
    defer signed_inf_table.deinit();

    var signed_inf_plan = try DeviceLazyFrame.init(gpa, signed_inf_table);
    defer signed_inf_plan.deinit();
    try signed_inf_plan.withRowFirstPositiveInfIndex(&.{ "metric", "peer", "id" }, "row_first_positive_inf_index");
    try signed_inf_plan.withRowLastPositiveInfIndex(&.{ "metric", "peer", "id" }, "row_last_positive_inf_index");
    try signed_inf_plan.withRowFirstNegativeInfIndex(&.{ "metric", "peer", "id" }, "row_first_negative_inf_index");
    try signed_inf_plan.withRowLastNegativeInfIndex(&.{ "metric", "peer", "id" }, "row_last_negative_inf_index");
    try signed_inf_plan.withRowCumulativeFirstPositiveInfIndex(&.{ "metric", "peer" }, &.{ "metric_cum_first_positive_inf", "peer_cum_first_positive_inf" });
    try signed_inf_plan.withRowPrefixLastNegativeInfIndex(&.{ "metric", "peer" }, &.{ "metric_prefix_last_negative_inf", "peer_prefix_last_negative_inf" });
    try signed_inf_plan.withRowAnyPositiveInf(&.{ "metric", "peer" }, "row_any_positive_inf");
    try signed_inf_plan.withRowPrefixAllNegativeInf(&.{ "metric", "peer" }, &.{ "metric_prefix_all_negative_inf", "peer_prefix_all_negative_inf" });
    try signed_inf_plan.select(&.{ "row_first_positive_inf_index", "row_last_positive_inf_index", "row_first_negative_inf_index", "row_last_negative_inf_index", "peer_cum_first_positive_inf", "peer_prefix_last_negative_inf", "row_any_positive_inf", "peer_prefix_all_negative_inf" });
    const signed_inf_explain = try signed_inf_plan.explain(gpa);
    defer gpa.free(signed_inf_explain);
    try std.testing.expect(std.mem.indexOf(u8, signed_inf_explain, "row_first_positive_inf_index([metric,peer,id]->row_first_positive_inf_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, signed_inf_explain, "row_last_positive_inf_index([metric,peer,id]->row_last_positive_inf_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, signed_inf_explain, "row_first_negative_inf_index([metric,peer,id]->row_first_negative_inf_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, signed_inf_explain, "row_last_negative_inf_index([metric,peer,id]->row_last_negative_inf_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, signed_inf_explain, "row_cumulative_first_positive_inf_index([metric,peer]->[metric_cum_first_positive_inf,peer_cum_first_positive_inf])") != null);
    try std.testing.expect(std.mem.indexOf(u8, signed_inf_explain, "row_cumulative_last_negative_inf_index([metric,peer]->[metric_prefix_last_negative_inf,peer_prefix_last_negative_inf])") != null);
    try std.testing.expect(std.mem.indexOf(u8, signed_inf_explain, "row_any_positive_inf([metric,peer]->row_any_positive_inf)") != null);
    try std.testing.expect(std.mem.indexOf(u8, signed_inf_explain, "row_cumulative_all_negative_inf([metric,peer]->[metric_prefix_all_negative_inf,peer_prefix_all_negative_inf])") != null);
    var signed_inf = try signed_inf_plan.collect();
    defer signed_inf.deinit();
    try std.testing.expectEqual(@as(usize, 8), signed_inf.width());
    const row_first_positive_inf = try (try signed_inf.column("row_first_positive_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_inf);
    const row_first_positive_inf_validity = try (try signed_inf.column("row_first_positive_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_positive_inf_validity);
    const row_last_positive_inf = try (try signed_inf.column("row_last_positive_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_positive_inf);
    const row_last_positive_inf_validity = try (try signed_inf.column("row_last_positive_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_positive_inf_validity);
    const row_first_negative_inf = try (try signed_inf.column("row_first_negative_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_negative_inf);
    const row_first_negative_inf_validity = try (try signed_inf.column("row_first_negative_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_negative_inf_validity);
    const row_last_negative_inf = try (try signed_inf.column("row_last_negative_inf_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_negative_inf);
    const row_last_negative_inf_validity = try (try signed_inf.column("row_last_negative_inf_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_negative_inf_validity);
    const peer_cum_first_positive_inf = try (try signed_inf.column("peer_cum_first_positive_inf")).i64.toOwnedSlice(gpa);
    defer gpa.free(peer_cum_first_positive_inf);
    const peer_cum_first_positive_inf_validity = try (try signed_inf.column("peer_cum_first_positive_inf")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(peer_cum_first_positive_inf_validity);
    const peer_prefix_last_negative_inf = try (try signed_inf.column("peer_prefix_last_negative_inf")).i64.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_last_negative_inf);
    const peer_prefix_last_negative_inf_validity = try (try signed_inf.column("peer_prefix_last_negative_inf")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_last_negative_inf_validity);
    const row_any_positive_inf = try (try signed_inf.column("row_any_positive_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_positive_inf);
    const peer_prefix_all_negative_inf = try (try signed_inf.column("peer_prefix_all_negative_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_all_negative_inf);
    const peer_prefix_all_negative_inf_validity = try (try signed_inf.column("peer_prefix_all_negative_inf")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_all_negative_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 1 }, row_first_positive_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, true }, row_first_positive_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 1 }, row_last_positive_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, true }, row_last_positive_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 1, 0 }, row_first_negative_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true, false }, row_first_negative_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1, 1, 0 }, row_last_negative_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true, false }, row_last_negative_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 1 }, peer_cum_first_positive_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, true }, peer_cum_first_positive_inf_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1, 1, 0 }, peer_prefix_last_negative_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true, false }, peer_prefix_last_negative_inf_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, true }, row_any_positive_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, peer_prefix_all_negative_inf);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true }, peer_prefix_all_negative_inf_validity);

    var invalid_positive_inf_index_plan = try DeviceLazyFrame.init(gpa, signed_inf_table);
    defer invalid_positive_inf_index_plan.deinit();
    try invalid_positive_inf_index_plan.withRowFirstPositiveInfIndex(&.{"missing"}, "bad_positive_inf_index");
    try std.testing.expectError(error.ColumnNotFound, invalid_positive_inf_index_plan.collect());

    var invalid_filter_finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_filter_finite_plan.deinit();
    try invalid_filter_finite_plan.filterFinitesColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_filter_finite_plan.collect());

    var invalid_row_count_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_row_count_plan.deinit();
    try invalid_row_count_plan.withRowInfCount(&.{"missing"}, "bad_count");
    try std.testing.expectError(error.ColumnNotFound, invalid_row_count_plan.collect());

    var invalid_row_ratio_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_row_ratio_plan.deinit();
    try invalid_row_ratio_plan.withRowInfRatio(&.{"missing"}, "bad_ratio");
    try std.testing.expectError(error.ColumnNotFound, invalid_row_ratio_plan.collect());

    var invalid_row_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_row_index_plan.deinit();
    try invalid_row_index_plan.withRowFirstNaNIndex(&.{"missing"}, "bad_nan_index");
    try std.testing.expectError(error.ColumnNotFound, invalid_row_index_plan.collect());

    var invalid_finite_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_finite_index_plan.deinit();
    try invalid_finite_index_plan.withRowFirstFiniteIndex(&.{"missing"}, "bad_finite_index");
    try std.testing.expectError(error.ColumnNotFound, invalid_finite_index_plan.collect());

    var invalid_prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_prefix_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_prefix_plan.withRowPrefixNonFiniteRatio(&.{"metric"}, &.{ "metric_cum_non_finite", "extra_cum_non_finite" }));
    try std.testing.expectError(error.LengthMismatch, invalid_prefix_plan.withRowPrefixLastNonFiniteIndex(&.{"metric"}, &.{ "metric_last_non_finite", "extra_last_non_finite" }));
    try std.testing.expectError(error.LengthMismatch, invalid_prefix_plan.withRowPrefixAllFinite(&.{"metric"}, &.{ "metric_all_finite", "extra_all_finite" }));

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isFiniteColumn("missing", "missing_is_finite");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame selects zero columns" {
    const gpa = std.testing.allocator;

    var zero_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, -0.0, 0.0 }, .cpu);
    defer zero_metric.deinit();
    var mixed_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, 4.0, std.math.nan(f64) }, .cpu);
    defer mixed_metric.deinit();
    var non_zero_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, std.math.nan(f64), std.math.inf(f64) }, .cpu);
    defer non_zero_metric.deinit();
    var null_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.0, 0.0, 0.0 }, &.{ false, false, false }, .cpu);
    defer null_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 0, 5, 0 }, .cpu);
    defer id.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ false, true, false }, .cpu);
    defer flag.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "zero_metric", .data = zero_metric },
        .{ .name = "mixed_metric", .data = mixed_metric },
        .{ .name = "non_zero_metric", .data = non_zero_metric },
        .{ .name = "null_metric", .data = null_metric },
        .{ .name = "id", .data = id },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var select_zeros_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_zeros_plan.deinit();
    try select_zeros_plan.selectColumnsWithZeros();
    const select_zeros_explain = try select_zeros_plan.explain(gpa);
    defer gpa.free(select_zeros_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_zeros_explain, "select_columns_with_zeros") != null);
    var zero_columns = try select_zeros_plan.collect();
    defer zero_columns.deinit();
    try std.testing.expectEqual(@as(usize, 4), zero_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), zero_columns.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 1), zero_columns.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), zero_columns.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 3), zero_columns.columnIndex("flag"));

    var select_non_zeros_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_non_zeros_plan.deinit();
    try select_non_zeros_plan.selectColumnsWithNonZeros();
    const select_non_zeros_explain = try select_non_zeros_plan.explain(gpa);
    defer gpa.free(select_non_zeros_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_non_zeros_explain, "select_columns_with_non_zeros") != null);
    var non_zero_columns = try select_non_zeros_plan.collect();
    defer non_zero_columns.deinit();
    try std.testing.expectEqual(@as(usize, 4), non_zero_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), non_zero_columns.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 1), non_zero_columns.columnIndex("non_zero_metric"));
    try std.testing.expectEqual(@as(?usize, 2), non_zero_columns.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 3), non_zero_columns.columnIndex("flag"));

    var drop_without_zeros_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_without_zeros_plan.deinit();
    try drop_without_zeros_plan.dropColumnsWithoutZeros();
    const drop_without_zeros_explain = try drop_without_zeros_plan.explain(gpa);
    defer gpa.free(drop_without_zeros_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_without_zeros_explain, "drop_columns_without_zeros") != null);
    var only_zero_columns = try drop_without_zeros_plan.collect();
    defer only_zero_columns.deinit();
    try std.testing.expectEqual(@as(usize, 4), only_zero_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), only_zero_columns.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 1), only_zero_columns.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), only_zero_columns.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 3), only_zero_columns.columnIndex("flag"));

    var select_positive_zeros_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_positive_zeros_plan.deinit();
    try select_positive_zeros_plan.selectColumnsWithPositiveZeros();
    const select_positive_zeros_explain = try select_positive_zeros_plan.explain(gpa);
    defer gpa.free(select_positive_zeros_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_positive_zeros_explain, "select_columns_with_positive_zeros") != null);
    var positive_zero_columns = try select_positive_zeros_plan.collect();
    defer positive_zero_columns.deinit();
    try std.testing.expectEqual(@as(usize, 2), positive_zero_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), positive_zero_columns.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 1), positive_zero_columns.columnIndex("mixed_metric"));

    var select_negative_zeros_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_negative_zeros_plan.deinit();
    try select_negative_zeros_plan.selectColumnsWithNegativeZeros();
    const select_negative_zeros_explain = try select_negative_zeros_plan.explain(gpa);
    defer gpa.free(select_negative_zeros_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_negative_zeros_explain, "select_columns_with_negative_zeros") != null);
    var negative_zero_columns = try select_negative_zeros_plan.collect();
    defer negative_zero_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), negative_zero_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), negative_zero_columns.columnIndex("zero_metric"));
}

test "device lazy frame selects sign columns" {
    const gpa = std.testing.allocator;

    var positive_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, std.math.inf(f64), 3.0 }, .cpu);
    defer positive_metric.deinit();
    var negative_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ -1.0, -std.math.inf(f64), -3.0 }, .cpu);
    defer negative_metric.deinit();
    var mixed_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ -1.0, 0.0, 2.0 }, .cpu);
    defer mixed_metric.deinit();
    var zero_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, -0.0, 0.0 }, .cpu);
    defer zero_metric.deinit();
    var null_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -1.0, 2.0, -3.0 }, &.{ false, false, false }, .cpu);
    defer null_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ -1, 0, 3 }, .cpu);
    defer id.deinit();
    var unsigned = try DeviceColumn.fromSlice(u64, gpa, &.{ 0, 4, 0 }, .cpu);
    defer unsigned.deinit();
    var flag = try DeviceColumn.fromSlice(bool, gpa, &.{ true, false, true }, .cpu);
    defer flag.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "positive_metric", .data = positive_metric },
        .{ .name = "negative_metric", .data = negative_metric },
        .{ .name = "mixed_metric", .data = mixed_metric },
        .{ .name = "zero_metric", .data = zero_metric },
        .{ .name = "null_metric", .data = null_metric },
        .{ .name = "id", .data = id },
        .{ .name = "unsigned", .data = unsigned },
        .{ .name = "flag", .data = flag },
    });
    defer table.deinit();

    var select_positives_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_positives_plan.deinit();
    try select_positives_plan.selectColumnsWithPositives();
    const select_positives_explain = try select_positives_plan.explain(gpa);
    defer gpa.free(select_positives_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_positives_explain, "select_columns_with_positives") != null);
    var positive_columns = try select_positives_plan.collect();
    defer positive_columns.deinit();
    try std.testing.expectEqual(@as(usize, 4), positive_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), positive_columns.columnIndex("positive_metric"));
    try std.testing.expectEqual(@as(?usize, 1), positive_columns.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), positive_columns.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 3), positive_columns.columnIndex("unsigned"));

    var select_signbits_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_signbits_plan.deinit();
    try select_signbits_plan.selectColumnsWithSignBits();
    const select_signbits_explain = try select_signbits_plan.explain(gpa);
    defer gpa.free(select_signbits_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_signbits_explain, "select_columns_with_signbits") != null);
    var signbit_columns = try select_signbits_plan.collect();
    defer signbit_columns.deinit();
    try std.testing.expectEqual(@as(usize, 4), signbit_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), signbit_columns.columnIndex("negative_metric"));
    try std.testing.expectEqual(@as(?usize, 1), signbit_columns.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), signbit_columns.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 3), signbit_columns.columnIndex("id"));

    var select_negatives_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_negatives_plan.deinit();
    try select_negatives_plan.selectColumnsWithNegatives();
    const select_negatives_explain = try select_negatives_plan.explain(gpa);
    defer gpa.free(select_negatives_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_negatives_explain, "select_columns_with_negatives") != null);
    var negative_columns = try select_negatives_plan.collect();
    defer negative_columns.deinit();
    try std.testing.expectEqual(@as(usize, 3), negative_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), negative_columns.columnIndex("negative_metric"));
    try std.testing.expectEqual(@as(?usize, 1), negative_columns.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), negative_columns.columnIndex("id"));

    var drop_without_negatives_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_without_negatives_plan.deinit();
    try drop_without_negatives_plan.dropColumnsWithoutNegatives();
    const drop_without_negatives_explain = try drop_without_negatives_plan.explain(gpa);
    defer gpa.free(drop_without_negatives_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_without_negatives_explain, "drop_columns_without_negatives") != null);
    var only_negative_columns = try drop_without_negatives_plan.collect();
    defer only_negative_columns.deinit();
    try std.testing.expectEqual(@as(usize, 3), only_negative_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), only_negative_columns.columnIndex("negative_metric"));
    try std.testing.expectEqual(@as(?usize, 1), only_negative_columns.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), only_negative_columns.columnIndex("id"));
}

test "device lazy frame selects finite columns" {
    const gpa = std.testing.allocator;

    var finite_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0 }, .cpu);
    defer finite_metric.deinit();
    var mixed_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.nan(f64), 4.0, std.math.inf(f64) }, .cpu);
    defer mixed_metric.deinit();
    var non_finite_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, .cpu);
    defer non_finite_metric.deinit();
    var null_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 8.0, 9.0, 10.0 }, &.{ false, false, false }, .cpu);
    defer null_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30 }, .cpu);
    defer id.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "finite_metric", .data = finite_metric },
        .{ .name = "mixed_metric", .data = mixed_metric },
        .{ .name = "non_finite_metric", .data = non_finite_metric },
        .{ .name = "null_metric", .data = null_metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var select_finites_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_finites_plan.deinit();
    try select_finites_plan.selectColumnsWithFinites();
    const select_finites_explain = try select_finites_plan.explain(gpa);
    defer gpa.free(select_finites_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_finites_explain, "select_columns_with_finites") != null);
    var finite_columns = try select_finites_plan.collect();
    defer finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 3), finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), finite_columns.columnIndex("finite_metric"));
    try std.testing.expectEqual(@as(?usize, 1), finite_columns.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), finite_columns.columnIndex("id"));

    var select_without_finites_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_without_finites_plan.deinit();
    try select_without_finites_plan.selectColumnsWithoutFinites();
    const select_without_finites_explain = try select_without_finites_plan.explain(gpa);
    defer gpa.free(select_without_finites_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_without_finites_explain, "select_columns_without_finites") != null);
    var non_finite_columns = try select_without_finites_plan.collect();
    defer non_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 2), non_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), non_finite_columns.columnIndex("non_finite_metric"));
    try std.testing.expectEqual(@as(?usize, 1), non_finite_columns.columnIndex("null_metric"));

    var drop_with_finites_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_with_finites_plan.deinit();
    try drop_with_finites_plan.dropColumnsWithFinites();
    const drop_with_finites_explain = try drop_with_finites_plan.explain(gpa);
    defer gpa.free(drop_with_finites_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_with_finites_explain, "drop_columns_with_finites") != null);
    var drop_finite_columns = try drop_with_finites_plan.collect();
    defer drop_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 2), drop_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_finite_columns.columnIndex("non_finite_metric"));
    try std.testing.expectEqual(@as(?usize, 1), drop_finite_columns.columnIndex("null_metric"));

    var drop_without_finites_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_without_finites_plan.deinit();
    try drop_without_finites_plan.dropColumnsWithoutFinites();
    const drop_without_finites_explain = try drop_without_finites_plan.explain(gpa);
    defer gpa.free(drop_without_finites_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_without_finites_explain, "drop_columns_without_finites") != null);
    var only_finite_columns = try drop_without_finites_plan.collect();
    defer only_finite_columns.deinit();
    try std.testing.expectEqual(@as(usize, 3), only_finite_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), only_finite_columns.columnIndex("finite_metric"));
    try std.testing.expectEqual(@as(?usize, 1), only_finite_columns.columnIndex("mixed_metric"));
    try std.testing.expectEqual(@as(?usize, 2), only_finite_columns.columnIndex("id"));
}

test "device lazy frame derives signed Inf predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 9.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40, 50 }, .cpu);
    defer id.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.isPositiveInfColumn("metric", "metric_is_pos_inf");
    try plan.isNegativeInfColumn("metric", "metric_is_neg_inf");
    try plan.isPositiveInfColumn("id", "id_is_pos_inf");
    try plan.select(&.{ "metric_is_pos_inf", "metric_is_neg_inf", "id_is_pos_inf" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_positive_inf_column(metric->metric_is_pos_inf)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_negative_inf_column(metric->metric_is_neg_inf)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 3), result.width());
    const metric_is_pos_inf = try (try result.column("metric_is_pos_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_pos_inf);
    const metric_is_neg_inf = try (try result.column("metric_is_neg_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_neg_inf);
    const id_is_pos_inf = try (try result.column("id_is_pos_inf")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_pos_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, metric_is_pos_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, metric_is_neg_inf);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, id_is_pos_inf);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isPositiveInfColumn("missing", "missing_is_pos_inf");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame derives normal predicate columns" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 0.0, std.math.floatTrueMin(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40, 50 }, .cpu);
    defer id.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.isNormalColumn("metric", "metric_is_normal");
    try plan.isSubnormalColumn("metric", "metric_is_subnormal");
    try plan.isNormalColumn("id", "id_is_normal");
    try plan.isSubnormalColumn("id", "id_is_subnormal");
    try plan.select(&.{ "metric_is_normal", "metric_is_subnormal", "id_is_normal", "id_is_subnormal" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_normal_column(metric->metric_is_normal)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_subnormal_column(metric->metric_is_subnormal)") != null);

    var result = try plan.collect();
    defer result.deinit();
    const metric_is_normal = try (try result.column("metric_is_normal")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_normal);
    const metric_is_subnormal = try (try result.column("metric_is_subnormal")).bool.toOwnedSlice(gpa);
    defer gpa.free(metric_is_subnormal);
    const id_is_normal = try (try result.column("id_is_normal")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_normal);
    const id_is_subnormal = try (try result.column("id_is_subnormal")).bool.toOwnedSlice(gpa);
    defer gpa.free(id_is_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false }, metric_is_normal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, false }, metric_is_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, id_is_normal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false }, id_is_subnormal);

    var row_normal_plan = try DeviceLazyFrame.init(gpa, table);
    defer row_normal_plan.deinit();
    try row_normal_plan.withRowNormalCount(&.{ "metric", "id" }, "row_normal_count");
    try row_normal_plan.withRowNormalRatio(&.{ "metric", "id" }, "row_normal_ratio");
    try row_normal_plan.withRowCumulativeNormalCount(&.{ "metric", "id" }, &.{ "metric_cum_normal", "id_cum_normal" });
    try row_normal_plan.withRowCumulativeFirstNormalIndex(&.{ "metric", "id" }, &.{ "metric_cum_first_normal", "id_cum_first_normal" });
    try row_normal_plan.withRowAnyNormal(&.{ "metric", "id" }, "row_any_normal");
    try row_normal_plan.select(&.{ "row_normal_count", "row_normal_ratio", "metric_cum_normal", "id_cum_normal", "id_cum_first_normal", "row_any_normal" });
    const row_normal_explain = try row_normal_plan.explain(gpa);
    defer gpa.free(row_normal_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_normal_explain, "row_normal_count([metric,id]->row_normal_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_normal_explain, "row_normal_ratio([metric,id]->row_normal_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_normal_explain, "row_cumulative_normal_count([metric,id]->[metric_cum_normal,id_cum_normal])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_normal_explain, "row_cumulative_first_normal_index([metric,id]->[metric_cum_first_normal,id_cum_first_normal])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_normal_explain, "row_any_normal([metric,id]->row_any_normal)") != null);
    var row_normal = try row_normal_plan.collect();
    defer row_normal.deinit();
    const row_normal_count = try (try row_normal.column("row_normal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_normal_count);
    const row_normal_ratio = try (try row_normal.column("row_normal_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_normal_ratio);
    const id_cum_normal = try (try row_normal.column("id_cum_normal")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_normal);
    const id_cum_first_normal = try (try row_normal.column("id_cum_first_normal")).i64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_first_normal);
    const id_cum_first_normal_validity = try (try row_normal.column("id_cum_first_normal")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(id_cum_first_normal_validity);
    const row_any_normal = try (try row_normal.column("row_any_normal")).bool.toOwnedSlice(gpa);
    defer gpa.free(row_any_normal);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0, 0 }, row_normal_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.0, 0.0 }, row_normal_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0, 0 }, id_cum_normal);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0 }, id_cum_first_normal);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false }, id_cum_first_normal_validity);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false, false }, row_any_normal);

    var row_subnormal_plan = try DeviceLazyFrame.init(gpa, table);
    defer row_subnormal_plan.deinit();
    try row_subnormal_plan.withRowSubnormalCount(&.{ "metric", "id" }, "row_subnormal_count");
    try row_subnormal_plan.withRowSubnormalRatio(&.{ "metric", "id" }, "row_subnormal_ratio");
    try row_subnormal_plan.withRowPrefixSubnormalRatio(&.{ "metric", "id" }, &.{ "metric_cum_subnormal", "id_cum_subnormal" });
    try row_subnormal_plan.select(&.{ "row_subnormal_count", "row_subnormal_ratio", "metric_cum_subnormal", "id_cum_subnormal" });
    const row_subnormal_explain = try row_subnormal_plan.explain(gpa);
    defer gpa.free(row_subnormal_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_subnormal_explain, "row_subnormal_count([metric,id]->row_subnormal_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_subnormal_explain, "row_subnormal_ratio([metric,id]->row_subnormal_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_subnormal_explain, "row_cumulative_subnormal_ratio([metric,id]->[metric_cum_subnormal,id_cum_subnormal])") != null);
    var row_subnormal = try row_subnormal_plan.collect();
    defer row_subnormal.deinit();
    const row_subnormal_count = try (try row_subnormal.column("row_subnormal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_subnormal_count);
    const row_subnormal_ratio = try (try row_subnormal.column("row_subnormal_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_subnormal_ratio);
    const metric_cum_subnormal = try (try row_subnormal.column("metric_cum_subnormal")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_subnormal);
    const id_cum_subnormal = try (try row_subnormal.column("id_cum_subnormal")).f64.toOwnedSlice(gpa);
    defer gpa.free(id_cum_subnormal);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0 }, row_subnormal_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.5, 0.0, 0.0 }, row_subnormal_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 1.0, 0.0, 0.0 }, metric_cum_subnormal);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.5, 0.0, 0.0 }, id_cum_subnormal);

    var index_metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, 0.0, std.math.floatTrueMin(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, false }, .cpu);
    defer index_metric.deinit();
    var index_peer = try DeviceColumn.fromSlice(f64, gpa, &.{ 2.0, 3.0, std.math.floatTrueMin(f64), 4.0, std.math.floatTrueMin(f64) }, .cpu);
    defer index_peer.deinit();
    var index_id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30, 40, 50 }, .cpu);
    defer index_id.deinit();
    var index_table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = index_metric },
        .{ .name = "peer", .data = index_peer },
        .{ .name = "id", .data = index_id },
    });
    defer index_table.deinit();

    var index_plan = try DeviceLazyFrame.init(gpa, index_table);
    defer index_plan.deinit();
    try index_plan.withRowFirstNormalIndex(&.{ "metric", "peer", "id" }, "row_first_normal_index");
    try index_plan.withRowLastNormalIndex(&.{ "metric", "peer", "id" }, "row_last_normal_index");
    try index_plan.withRowFirstSubnormalIndex(&.{ "metric", "peer", "id" }, "row_first_subnormal_index");
    try index_plan.withRowLastSubnormalIndex(&.{ "metric", "peer", "id" }, "row_last_subnormal_index");
    try index_plan.withRowPrefixLastSubnormalIndex(&.{ "metric", "peer" }, &.{ "metric_prefix_last_subnormal", "peer_prefix_last_subnormal" });
    try index_plan.withRowPrefixAnySubnormal(&.{ "metric", "peer" }, &.{ "metric_prefix_any_subnormal", "peer_prefix_any_subnormal" });
    try index_plan.select(&.{ "row_first_normal_index", "row_last_normal_index", "row_first_subnormal_index", "row_last_subnormal_index", "peer_prefix_last_subnormal", "peer_prefix_any_subnormal" });
    const index_explain = try index_plan.explain(gpa);
    defer gpa.free(index_explain);
    try std.testing.expect(std.mem.indexOf(u8, index_explain, "row_first_normal_index([metric,peer,id]->row_first_normal_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, index_explain, "row_last_normal_index([metric,peer,id]->row_last_normal_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, index_explain, "row_first_subnormal_index([metric,peer,id]->row_first_subnormal_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, index_explain, "row_last_subnormal_index([metric,peer,id]->row_last_subnormal_index)") != null);
    try std.testing.expect(std.mem.indexOf(u8, index_explain, "row_cumulative_last_subnormal_index([metric,peer]->[metric_prefix_last_subnormal,peer_prefix_last_subnormal])") != null);
    try std.testing.expect(std.mem.indexOf(u8, index_explain, "row_cumulative_any_subnormal([metric,peer]->[metric_prefix_any_subnormal,peer_prefix_any_subnormal])") != null);
    var index_result = try index_plan.collect();
    defer index_result.deinit();
    try std.testing.expectEqual(@as(usize, 6), index_result.width());
    const row_first_normal = try (try index_result.column("row_first_normal_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_normal);
    const row_first_normal_validity = try (try index_result.column("row_first_normal_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_normal_validity);
    const row_last_normal = try (try index_result.column("row_last_normal_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_normal);
    const row_last_normal_validity = try (try index_result.column("row_last_normal_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_normal_validity);
    const row_first_subnormal = try (try index_result.column("row_first_subnormal_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_first_subnormal);
    const row_first_subnormal_validity = try (try index_result.column("row_first_subnormal_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_first_subnormal_validity);
    const row_last_subnormal = try (try index_result.column("row_last_subnormal_index")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_last_subnormal);
    const row_last_subnormal_validity = try (try index_result.column("row_last_subnormal_index")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_last_subnormal_validity);
    const peer_prefix_last_subnormal = try (try index_result.column("peer_prefix_last_subnormal")).i64.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_last_subnormal);
    const peer_prefix_last_subnormal_validity = try (try index_result.column("peer_prefix_last_subnormal")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_last_subnormal_validity);
    const peer_prefix_any_subnormal = try (try index_result.column("peer_prefix_any_subnormal")).bool.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_any_subnormal);
    const peer_prefix_any_subnormal_validity = try (try index_result.column("peer_prefix_any_subnormal")).bool.validity.?.toOwnedSlice(gpa);
    defer gpa.free(peer_prefix_any_subnormal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 1, 0 }, row_first_normal);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false }, row_first_normal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 1, 0 }, row_last_normal);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true, false }, row_last_normal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 1 }, row_first_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true }, row_first_subnormal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 1 }, row_last_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true }, row_last_subnormal_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 1 }, peer_prefix_last_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true }, peer_prefix_last_subnormal_validity);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, false, true }, peer_prefix_any_subnormal);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true }, peer_prefix_any_subnormal_validity);

    var drop_normal_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_normal_plan.deinit();
    try drop_normal_plan.dropNormalsColumn("metric");
    const drop_normal_explain = try drop_normal_plan.explain(gpa);
    defer gpa.free(drop_normal_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_normal_explain, "drop_normals[metric]") != null);
    var dropped_normal = try drop_normal_plan.collect();
    defer dropped_normal.deinit();
    try std.testing.expectEqual(@as(usize, 4), dropped_normal.height());
    const dropped_normal_metric = try (try dropped_normal.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_normal_metric);
    try std.testing.expectEqual(@as(f64, 0.0), dropped_normal_metric[0]);
    try std.testing.expectEqual(@as(f64, std.math.floatTrueMin(f64)), dropped_normal_metric[1]);
    try std.testing.expect(std.math.isPositiveInf(dropped_normal_metric[2]));
    try std.testing.expectEqual(@as(f64, -2.0), dropped_normal_metric[3]);

    var filter_normal_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_normal_plan.deinit();
    try filter_normal_plan.filterNormalsColumn("metric");
    const filter_normal_explain = try filter_normal_plan.explain(gpa);
    defer gpa.free(filter_normal_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_normal_explain, "filter_normals_column(metric)") != null);
    var filtered_normal = try filter_normal_plan.collect();
    defer filtered_normal.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_normal.height());
    const filtered_normal_metric = try (try filtered_normal.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_normal_metric);
    try std.testing.expectEqual(@as(f64, 1.0), filtered_normal_metric[0]);

    var drop_subnormal_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_subnormal_plan.deinit();
    try drop_subnormal_plan.dropSubnormalsColumn("metric");
    const drop_subnormal_explain = try drop_subnormal_plan.explain(gpa);
    defer gpa.free(drop_subnormal_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_subnormal_explain, "drop_subnormals[metric]") != null);
    var dropped_subnormal = try drop_subnormal_plan.collect();
    defer dropped_subnormal.deinit();
    try std.testing.expectEqual(@as(usize, 4), dropped_subnormal.height());
    const dropped_subnormal_metric = try (try dropped_subnormal.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_subnormal_metric);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_subnormal_metric[0]);
    try std.testing.expectEqual(@as(f64, 0.0), dropped_subnormal_metric[1]);
    try std.testing.expect(std.math.isPositiveInf(dropped_subnormal_metric[2]));
    try std.testing.expectEqual(@as(f64, -2.0), dropped_subnormal_metric[3]);

    var filter_subnormal_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_subnormal_plan.deinit();
    try filter_subnormal_plan.filterSubnormalsColumn("metric");
    const filter_subnormal_explain = try filter_subnormal_plan.explain(gpa);
    defer gpa.free(filter_subnormal_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_subnormal_explain, "filter_subnormals_column(metric)") != null);
    var filtered_subnormal = try filter_subnormal_plan.collect();
    defer filtered_subnormal.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_subnormal.height());
    const filtered_subnormal_metric = try (try filtered_subnormal.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_subnormal_metric);
    try std.testing.expectEqual(@as(f64, std.math.floatTrueMin(f64)), filtered_subnormal_metric[0]);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isNormalColumn("missing", "missing_is_normal");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());

    var invalid_subnormal_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_subnormal_plan.deinit();
    try invalid_subnormal_plan.isSubnormalColumn("missing", "missing_is_subnormal");
    try std.testing.expectError(error.ColumnNotFound, invalid_subnormal_plan.collect());

    var invalid_count_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_count_plan.deinit();
    try invalid_count_plan.withRowNormalCount(&.{"missing"}, "bad_count");
    try std.testing.expectError(error.ColumnNotFound, invalid_count_plan.collect());

    var invalid_ratio_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_ratio_plan.deinit();
    try invalid_ratio_plan.withRowNormalRatio(&.{"missing"}, "bad_ratio");
    try std.testing.expectError(error.ColumnNotFound, invalid_ratio_plan.collect());

    var invalid_subnormal_count_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_subnormal_count_plan.deinit();
    try invalid_subnormal_count_plan.withRowSubnormalCount(&.{"missing"}, "bad_subnormal_count");
    try std.testing.expectError(error.ColumnNotFound, invalid_subnormal_count_plan.collect());

    var invalid_normal_index_plan = try DeviceLazyFrame.init(gpa, index_table);
    defer invalid_normal_index_plan.deinit();
    try invalid_normal_index_plan.withRowFirstNormalIndex(&.{"missing"}, "bad_normal_index");
    try std.testing.expectError(error.ColumnNotFound, invalid_normal_index_plan.collect());

    var invalid_subnormal_index_plan = try DeviceLazyFrame.init(gpa, index_table);
    defer invalid_subnormal_index_plan.deinit();
    try invalid_subnormal_index_plan.withRowFirstSubnormalIndex(&.{"missing"}, "bad_subnormal_index");
    try std.testing.expectError(error.ColumnNotFound, invalid_subnormal_index_plan.collect());

    var invalid_filter_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_filter_plan.deinit();
    try invalid_filter_plan.filterNormalsColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_filter_plan.collect());

    var invalid_subnormal_filter_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_subnormal_filter_plan.deinit();
    try invalid_subnormal_filter_plan.filterSubnormalsColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_subnormal_filter_plan.collect());
}

test "device lazy frame selects normal columns" {
    const gpa = std.testing.allocator;

    var normal_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, 2.0, 3.0 }, .cpu);
    defer normal_metric.deinit();
    var zero_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 0.0, -0.0, 0.0 }, .cpu);
    defer zero_metric.deinit();
    var mixed_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.floatTrueMin(f64), -4.0, std.math.nan(f64) }, .cpu);
    defer mixed_metric.deinit();
    var special_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.inf(f64), std.math.nan(f64), 0.0 }, .cpu);
    defer special_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30 }, .cpu);
    defer id.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "normal_metric", .data = normal_metric },
        .{ .name = "zero_metric", .data = zero_metric },
        .{ .name = "mixed_metric", .data = mixed_metric },
        .{ .name = "special_metric", .data = special_metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var select_normals_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_normals_plan.deinit();
    try select_normals_plan.selectColumnsWithNormals();
    const select_normals_explain = try select_normals_plan.explain(gpa);
    defer gpa.free(select_normals_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_normals_explain, "select_columns_with_normals") != null);
    var normal_columns = try select_normals_plan.collect();
    defer normal_columns.deinit();
    try std.testing.expectEqual(@as(usize, 2), normal_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), normal_columns.columnIndex("normal_metric"));
    try std.testing.expectEqual(@as(?usize, 1), normal_columns.columnIndex("mixed_metric"));

    var drop_without_normals_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_without_normals_plan.deinit();
    try drop_without_normals_plan.dropColumnsWithoutNormals();
    const drop_without_normals_explain = try drop_without_normals_plan.explain(gpa);
    defer gpa.free(drop_without_normals_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_without_normals_explain, "drop_columns_without_normals") != null);
    var only_normal_columns = try drop_without_normals_plan.collect();
    defer only_normal_columns.deinit();
    try std.testing.expectEqual(@as(usize, 2), only_normal_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), only_normal_columns.columnIndex("normal_metric"));
    try std.testing.expectEqual(@as(?usize, 1), only_normal_columns.columnIndex("mixed_metric"));

    var drop_with_normals_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_with_normals_plan.deinit();
    try drop_with_normals_plan.dropColumnsWithNormals();
    const drop_with_normals_explain = try drop_with_normals_plan.explain(gpa);
    defer gpa.free(drop_with_normals_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_with_normals_explain, "drop_columns_with_normals") != null);
    var non_normal_columns = try drop_with_normals_plan.collect();
    defer non_normal_columns.deinit();
    try std.testing.expectEqual(@as(usize, 3), non_normal_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), non_normal_columns.columnIndex("zero_metric"));
    try std.testing.expectEqual(@as(?usize, 1), non_normal_columns.columnIndex("special_metric"));
    try std.testing.expectEqual(@as(?usize, 2), non_normal_columns.columnIndex("id"));

    var select_subnormals_plan = try DeviceLazyFrame.init(gpa, table);
    defer select_subnormals_plan.deinit();
    try select_subnormals_plan.selectColumnsWithSubnormals();
    const select_subnormals_explain = try select_subnormals_plan.explain(gpa);
    defer gpa.free(select_subnormals_explain);
    try std.testing.expect(std.mem.indexOf(u8, select_subnormals_explain, "select_columns_with_subnormals") != null);
    var subnormal_columns = try select_subnormals_plan.collect();
    defer subnormal_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), subnormal_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), subnormal_columns.columnIndex("mixed_metric"));

    var drop_without_subnormals_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_without_subnormals_plan.deinit();
    try drop_without_subnormals_plan.dropColumnsWithoutSubnormals();
    const drop_without_subnormals_explain = try drop_without_subnormals_plan.explain(gpa);
    defer gpa.free(drop_without_subnormals_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_without_subnormals_explain, "drop_columns_without_subnormals") != null);
    var only_subnormal_columns = try drop_without_subnormals_plan.collect();
    defer only_subnormal_columns.deinit();
    try std.testing.expectEqual(@as(usize, 1), only_subnormal_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), only_subnormal_columns.columnIndex("mixed_metric"));
}

test "device lazy frame selects signed Inf columns" {
    const gpa = std.testing.allocator;

    var pos_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 1.0, std.math.inf(f64), 2.0 }, .cpu);
    defer pos_metric.deinit();
    var neg_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 3.0, -std.math.inf(f64), 4.0 }, .cpu);
    defer neg_metric.deinit();
    var both_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ std.math.inf(f64), -std.math.inf(f64), 5.0 }, .cpu);
    defer both_metric.deinit();
    var finite_metric = try DeviceColumn.fromSlice(f64, gpa, &.{ 6.0, 7.0, 8.0 }, .cpu);
    defer finite_metric.deinit();
    var id = try DeviceColumn.fromSlice(i64, gpa, &.{ 10, 20, 30 }, .cpu);
    defer id.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "pos_metric", .data = pos_metric },
        .{ .name = "neg_metric", .data = neg_metric },
        .{ .name = "both_metric", .data = both_metric },
        .{ .name = "finite_metric", .data = finite_metric },
        .{ .name = "id", .data = id },
    });
    defer table.deinit();

    var positive_plan = try DeviceLazyFrame.init(gpa, table);
    defer positive_plan.deinit();
    try positive_plan.selectColumnsWithPositiveInfs();
    const positive_explain = try positive_plan.explain(gpa);
    defer gpa.free(positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, positive_explain, "select_columns_with_positive_infs") != null);
    var positive_columns = try positive_plan.collect();
    defer positive_columns.deinit();
    try std.testing.expectEqual(@as(usize, 2), positive_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), positive_columns.columnIndex("pos_metric"));
    try std.testing.expectEqual(@as(?usize, 1), positive_columns.columnIndex("both_metric"));

    var negative_plan = try DeviceLazyFrame.init(gpa, table);
    defer negative_plan.deinit();
    try negative_plan.selectColumnsWithNegativeInfs();
    const negative_explain = try negative_plan.explain(gpa);
    defer gpa.free(negative_explain);
    try std.testing.expect(std.mem.indexOf(u8, negative_explain, "select_columns_with_negative_infs") != null);
    var negative_columns = try negative_plan.collect();
    defer negative_columns.deinit();
    try std.testing.expectEqual(@as(usize, 2), negative_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), negative_columns.columnIndex("neg_metric"));
    try std.testing.expectEqual(@as(?usize, 1), negative_columns.columnIndex("both_metric"));

    var drop_without_positive_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_without_positive_plan.deinit();
    try drop_without_positive_plan.dropColumnsWithoutPositiveInfs();
    const drop_without_positive_explain = try drop_without_positive_plan.explain(gpa);
    defer gpa.free(drop_without_positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_without_positive_explain, "drop_columns_without_positive_infs") != null);
    var only_positive_columns = try drop_without_positive_plan.collect();
    defer only_positive_columns.deinit();
    try std.testing.expectEqual(@as(usize, 2), only_positive_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), only_positive_columns.columnIndex("pos_metric"));
    try std.testing.expectEqual(@as(?usize, 1), only_positive_columns.columnIndex("both_metric"));

    var drop_with_negative_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_with_negative_plan.deinit();
    try drop_with_negative_plan.dropColumnsWithNegativeInfs();
    const drop_with_negative_explain = try drop_with_negative_plan.explain(gpa);
    defer gpa.free(drop_with_negative_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_with_negative_explain, "drop_columns_with_negative_infs") != null);
    var no_negative_columns = try drop_with_negative_plan.collect();
    defer no_negative_columns.deinit();
    try std.testing.expectEqual(@as(usize, 3), no_negative_columns.width());
    try std.testing.expectEqual(@as(?usize, 0), no_negative_columns.columnIndex("pos_metric"));
    try std.testing.expectEqual(@as(?usize, 1), no_negative_columns.columnIndex("finite_metric"));
    try std.testing.expectEqual(@as(?usize, 2), no_negative_columns.columnIndex("id"));
}

test "device lazy frame fills signed Inf values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 9.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var positive_plan = try DeviceLazyFrame.init(gpa, table);
    defer positive_plan.deinit();
    try positive_plan.fillPositiveInfColumn("metric", f64, 100.0);
    const positive_explain = try positive_plan.explain(gpa);
    defer gpa.free(positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, positive_explain, "fill_positive_inf_column(metric=scalar:f64)") != null);
    var filled_positive = try positive_plan.collect();
    defer filled_positive.deinit();
    const positive_values = try (try filled_positive.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(positive_values);
    try std.testing.expectEqual(@as(f64, 1.0), positive_values[0]);
    try std.testing.expectEqual(@as(f64, 100.0), positive_values[1]);
    try std.testing.expect(std.math.isNegativeInf(positive_values[2]));
    try std.testing.expect(std.math.isNan(positive_values[3]));
    try std.testing.expectEqual(@as(f64, 9.0), positive_values[4]);

    var negative_plan = try DeviceLazyFrame.init(gpa, table);
    defer negative_plan.deinit();
    try negative_plan.fillNegativeInfColumn("metric", f64, -100.0);
    const negative_explain = try negative_plan.explain(gpa);
    defer gpa.free(negative_explain);
    try std.testing.expect(std.mem.indexOf(u8, negative_explain, "fill_negative_inf_column(metric=scalar:f64)") != null);
    var filled_negative = try negative_plan.collect();
    defer filled_negative.deinit();
    const negative_values = try (try filled_negative.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(negative_values);
    try std.testing.expectEqual(@as(f64, 1.0), negative_values[0]);
    try std.testing.expect(std.math.isPositiveInf(negative_values[1]));
    try std.testing.expectEqual(@as(f64, -100.0), negative_values[2]);
    try std.testing.expect(std.math.isNan(negative_values[3]));
    try std.testing.expectEqual(@as(f64, 9.0), negative_values[4]);

    var mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer mismatch_plan.deinit();
    try mismatch_plan.fillPositiveInfColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, mismatch_plan.collect());
}

test "device lazy frame fills zero values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 0.0, -0.0, 3.0, std.math.nan(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var zero_plan = try DeviceLazyFrame.init(gpa, table);
    defer zero_plan.deinit();
    try zero_plan.fillZeroColumn("metric", f64, 42.0);
    const zero_explain = try zero_plan.explain(gpa);
    defer gpa.free(zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, zero_explain, "fill_zero_column(metric=scalar:f64)") != null);
    var filled_zero = try zero_plan.collect();
    defer filled_zero.deinit();
    const zero_values = try (try filled_zero.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(zero_values);
    try std.testing.expectEqual(@as(f64, 42.0), zero_values[0]);
    try std.testing.expectEqual(@as(f64, 42.0), zero_values[1]);
    try std.testing.expectEqual(@as(f64, 3.0), zero_values[2]);
    try std.testing.expect(std.math.isNan(zero_values[3]));
    try std.testing.expect(std.math.isPositiveInf(zero_values[4]));
    try std.testing.expectEqual(@as(f64, -2.0), zero_values[5]);

    var non_zero_plan = try DeviceLazyFrame.init(gpa, table);
    defer non_zero_plan.deinit();
    try non_zero_plan.fillNonZeroColumn("metric", f64, -7.0);
    const non_zero_explain = try non_zero_plan.explain(gpa);
    defer gpa.free(non_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, non_zero_explain, "fill_non_zero_column(metric=scalar:f64)") != null);
    var filled_non_zero = try non_zero_plan.collect();
    defer filled_non_zero.deinit();
    const non_zero_values = try (try filled_non_zero.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(non_zero_values);
    try std.testing.expectEqual(@as(f64, 0.0), non_zero_values[0]);
    try std.testing.expectEqual(@as(f64, -0.0), non_zero_values[1]);
    try std.testing.expectEqual(@as(f64, -7.0), non_zero_values[2]);
    try std.testing.expectEqual(@as(f64, -7.0), non_zero_values[3]);
    try std.testing.expectEqual(@as(f64, -7.0), non_zero_values[4]);
    try std.testing.expectEqual(@as(f64, -2.0), non_zero_values[5]);

    var mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer mismatch_plan.deinit();
    try mismatch_plan.fillZeroColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, mismatch_plan.collect());
}

test "device lazy frame fills sign values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ -2.0, -0.0, 0.0, 3.0, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 9.0 }, &.{ true, true, true, true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var positive_plan = try DeviceLazyFrame.init(gpa, table);
    defer positive_plan.deinit();
    try positive_plan.fillPositiveColumn("metric", f64, 42.0);
    const positive_explain = try positive_plan.explain(gpa);
    defer gpa.free(positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, positive_explain, "fill_positive_column(metric=scalar:f64)") != null);
    var filled_positive = try positive_plan.collect();
    defer filled_positive.deinit();
    const positive_values = try (try filled_positive.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(positive_values);
    try std.testing.expectEqual(@as(f64, -2.0), positive_values[0]);
    try std.testing.expectEqual(@as(f64, -0.0), positive_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), positive_values[2]);
    try std.testing.expectEqual(@as(f64, 42.0), positive_values[3]);
    try std.testing.expect(std.math.isNan(positive_values[4]));
    try std.testing.expectEqual(@as(f64, 42.0), positive_values[5]);
    try std.testing.expect(std.math.isNegativeInf(positive_values[6]));
    try std.testing.expectEqual(@as(f64, 9.0), positive_values[7]);

    var signbit_plan = try DeviceLazyFrame.init(gpa, table);
    defer signbit_plan.deinit();
    try signbit_plan.fillSignBitColumn("metric", f64, -42.0);
    const signbit_explain = try signbit_plan.explain(gpa);
    defer gpa.free(signbit_explain);
    try std.testing.expect(std.mem.indexOf(u8, signbit_explain, "fill_signbit_column(metric=scalar:f64)") != null);
    var filled_signbit = try signbit_plan.collect();
    defer filled_signbit.deinit();
    const signbit_values = try (try filled_signbit.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(signbit_values);
    try std.testing.expectEqual(@as(f64, -42.0), signbit_values[0]);
    try std.testing.expectEqual(@as(f64, -42.0), signbit_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), signbit_values[2]);
    try std.testing.expectEqual(@as(f64, 3.0), signbit_values[3]);
    try std.testing.expect(std.math.isNan(signbit_values[4]));
    try std.testing.expect(std.math.isPositiveInf(signbit_values[5]));
    try std.testing.expectEqual(@as(f64, -42.0), signbit_values[6]);
    try std.testing.expectEqual(@as(f64, 9.0), signbit_values[7]);

    var negative_plan = try DeviceLazyFrame.init(gpa, table);
    defer negative_plan.deinit();
    try negative_plan.fillNegativeColumn("metric", f64, 7.0);
    const negative_explain = try negative_plan.explain(gpa);
    defer gpa.free(negative_explain);
    try std.testing.expect(std.mem.indexOf(u8, negative_explain, "fill_negative_column(metric=scalar:f64)") != null);
    var filled_negative = try negative_plan.collect();
    defer filled_negative.deinit();
    const negative_values = try (try filled_negative.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(negative_values);
    try std.testing.expectEqual(@as(f64, 7.0), negative_values[0]);
    try std.testing.expectEqual(@as(f64, -0.0), negative_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), negative_values[2]);
    try std.testing.expectEqual(@as(f64, 3.0), negative_values[3]);
    try std.testing.expect(std.math.isNan(negative_values[4]));
    try std.testing.expect(std.math.isPositiveInf(negative_values[5]));
    try std.testing.expectEqual(@as(f64, 7.0), negative_values[6]);
    try std.testing.expectEqual(@as(f64, 9.0), negative_values[7]);

    var positive_zero_plan = try DeviceLazyFrame.init(gpa, table);
    defer positive_zero_plan.deinit();
    try positive_zero_plan.fillPositiveZeroColumn("metric", f64, 11.0);
    const positive_zero_explain = try positive_zero_plan.explain(gpa);
    defer gpa.free(positive_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, positive_zero_explain, "fill_positive_zero_column(metric=scalar:f64)") != null);
    var filled_positive_zero = try positive_zero_plan.collect();
    defer filled_positive_zero.deinit();
    const positive_zero_values = try (try filled_positive_zero.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(positive_zero_values);
    try std.testing.expectEqual(@as(f64, -2.0), positive_zero_values[0]);
    try std.testing.expectEqual(@as(f64, -0.0), positive_zero_values[1]);
    try std.testing.expectEqual(@as(f64, 11.0), positive_zero_values[2]);
    try std.testing.expectEqual(@as(f64, 3.0), positive_zero_values[3]);
    try std.testing.expect(std.math.isNan(positive_zero_values[4]));
    try std.testing.expect(std.math.isPositiveInf(positive_zero_values[5]));
    try std.testing.expect(std.math.isNegativeInf(positive_zero_values[6]));
    try std.testing.expectEqual(@as(f64, 9.0), positive_zero_values[7]);

    var negative_zero_plan = try DeviceLazyFrame.init(gpa, table);
    defer negative_zero_plan.deinit();
    try negative_zero_plan.fillNegativeZeroColumn("metric", f64, -11.0);
    const negative_zero_explain = try negative_zero_plan.explain(gpa);
    defer gpa.free(negative_zero_explain);
    try std.testing.expect(std.mem.indexOf(u8, negative_zero_explain, "fill_negative_zero_column(metric=scalar:f64)") != null);
    var filled_negative_zero = try negative_zero_plan.collect();
    defer filled_negative_zero.deinit();
    const negative_zero_values = try (try filled_negative_zero.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(negative_zero_values);
    try std.testing.expectEqual(@as(f64, -2.0), negative_zero_values[0]);
    try std.testing.expectEqual(@as(f64, -11.0), negative_zero_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), negative_zero_values[2]);
    try std.testing.expectEqual(@as(f64, 3.0), negative_zero_values[3]);
    try std.testing.expect(std.math.isNan(negative_zero_values[4]));
    try std.testing.expect(std.math.isPositiveInf(negative_zero_values[5]));
    try std.testing.expect(std.math.isNegativeInf(negative_zero_values[6]));
    try std.testing.expectEqual(@as(f64, 9.0), negative_zero_values[7]);

    var mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer mismatch_plan.deinit();
    try mismatch_plan.fillSignBitColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, mismatch_plan.collect());
}

test "device lazy frame fills finite values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.floatTrueMin(f64), 0.0, std.math.nan(f64), std.math.inf(f64), -2.0 }, &.{ true, true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var finite_plan = try DeviceLazyFrame.init(gpa, table);
    defer finite_plan.deinit();
    try finite_plan.fillFiniteColumn("metric", f64, 42.0);
    const finite_explain = try finite_plan.explain(gpa);
    defer gpa.free(finite_explain);
    try std.testing.expect(std.mem.indexOf(u8, finite_explain, "fill_finite_column(metric=scalar:f64)") != null);
    var filled_finite = try finite_plan.collect();
    defer filled_finite.deinit();
    const filled_values = try (try filled_finite.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_values);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[0]);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[1]);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[2]);
    try std.testing.expect(std.math.isNan(filled_values[3]));
    try std.testing.expect(std.math.isPositiveInf(filled_values[4]));
    try std.testing.expectEqual(@as(f64, -2.0), filled_values[5]);

    var mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer mismatch_plan.deinit();
    try mismatch_plan.fillFiniteColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, mismatch_plan.collect());
}

test "device lazy frame fills normal values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.floatTrueMin(f64), 0.0, std.math.nan(f64), -2.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var normal_plan = try DeviceLazyFrame.init(gpa, table);
    defer normal_plan.deinit();
    try normal_plan.fillNormalColumn("metric", f64, 42.0);
    const normal_explain = try normal_plan.explain(gpa);
    defer gpa.free(normal_explain);
    try std.testing.expect(std.mem.indexOf(u8, normal_explain, "fill_normal_column(metric=scalar:f64)") != null);
    var filled_normal = try normal_plan.collect();
    defer filled_normal.deinit();
    const filled_values = try (try filled_normal.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_values);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[0]);
    try std.testing.expectEqual(@as(f64, std.math.floatTrueMin(f64)), filled_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), filled_values[2]);
    try std.testing.expect(std.math.isNan(filled_values[3]));
    try std.testing.expectEqual(@as(f64, -2.0), filled_values[4]);

    var mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer mismatch_plan.deinit();
    try mismatch_plan.fillNormalColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, mismatch_plan.collect());
}

test "device lazy frame fills subnormal values" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.floatTrueMin(f64), 0.0, std.math.nan(f64), -2.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var subnormal_plan = try DeviceLazyFrame.init(gpa, table);
    defer subnormal_plan.deinit();
    try subnormal_plan.fillSubnormalColumn("metric", f64, 42.0);
    const subnormal_explain = try subnormal_plan.explain(gpa);
    defer gpa.free(subnormal_explain);
    try std.testing.expect(std.mem.indexOf(u8, subnormal_explain, "fill_subnormal_column(metric=scalar:f64)") != null);
    var filled_subnormal = try subnormal_plan.collect();
    defer filled_subnormal.deinit();
    const filled_values = try (try filled_subnormal.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filled_values);
    try std.testing.expectEqual(@as(f64, 1.0), filled_values[0]);
    try std.testing.expectEqual(@as(f64, 42.0), filled_values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), filled_values[2]);
    try std.testing.expect(std.math.isNan(filled_values[3]));
    try std.testing.expectEqual(@as(f64, -2.0), filled_values[4]);

    var mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer mismatch_plan.deinit();
    try mismatch_plan.fillSubnormalColumn("metric", i64, 0);
    try std.testing.expectError(error.TypeUnsupported, mismatch_plan.collect());
}

test "device lazy frame filters signed Inf rows" {
    const gpa = std.testing.allocator;

    var metric = try DeviceColumn.fromSliceWithValidity(f64, gpa, &.{ 1.0, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 9.0 }, &.{ true, true, true, true, false }, .cpu);
    defer metric.deinit();

    var table = try vectra.DeviceDataFrame.init(gpa, &.{
        .{ .name = "metric", .data = metric },
    });
    defer table.deinit();

    var drop_positive_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_positive_plan.deinit();
    try drop_positive_plan.dropPositiveInfsColumn("metric");
    const drop_positive_explain = try drop_positive_plan.explain(gpa);
    defer gpa.free(drop_positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_positive_explain, "drop_positive_infs[metric]") != null);
    var dropped_positive = try drop_positive_plan.collect();
    defer dropped_positive.deinit();
    try std.testing.expectEqual(@as(usize, 4), dropped_positive.height());
    const dropped_positive_values = try (try dropped_positive.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_positive_values);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_positive_values[0]);
    try std.testing.expect(std.math.isNegativeInf(dropped_positive_values[1]));
    try std.testing.expect(std.math.isNan(dropped_positive_values[2]));
    try std.testing.expectEqual(@as(f64, 9.0), dropped_positive_values[3]);

    var filter_positive_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_positive_plan.deinit();
    try filter_positive_plan.filterPositiveInfsColumn("metric");
    const filter_positive_explain = try filter_positive_plan.explain(gpa);
    defer gpa.free(filter_positive_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_positive_explain, "filter_positive_infs_column(metric)") != null);
    var filtered_positive = try filter_positive_plan.collect();
    defer filtered_positive.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_positive.height());
    const filtered_positive_values = try (try filtered_positive.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_positive_values);
    try std.testing.expect(std.math.isPositiveInf(filtered_positive_values[0]));

    var drop_negative_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_negative_plan.deinit();
    try drop_negative_plan.dropNegativeInfsColumn("metric");
    const drop_negative_explain = try drop_negative_plan.explain(gpa);
    defer gpa.free(drop_negative_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_negative_explain, "drop_negative_infs[metric]") != null);
    var dropped_negative = try drop_negative_plan.collect();
    defer dropped_negative.deinit();
    try std.testing.expectEqual(@as(usize, 4), dropped_negative.height());
    const dropped_negative_values = try (try dropped_negative.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_negative_values);
    try std.testing.expectEqual(@as(f64, 1.0), dropped_negative_values[0]);
    try std.testing.expect(std.math.isPositiveInf(dropped_negative_values[1]));
    try std.testing.expect(std.math.isNan(dropped_negative_values[2]));
    try std.testing.expectEqual(@as(f64, 9.0), dropped_negative_values[3]);

    var filter_negative_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_negative_plan.deinit();
    try filter_negative_plan.filterNegativeInfsColumn("metric");
    const filter_negative_explain = try filter_negative_plan.explain(gpa);
    defer gpa.free(filter_negative_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_negative_explain, "filter_negative_infs_column(metric)") != null);
    var filtered_negative = try filter_negative_plan.collect();
    defer filtered_negative.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered_negative.height());
    const filtered_negative_values = try (try filtered_negative.column("metric")).f64.toOwnedSlice(gpa);
    defer gpa.free(filtered_negative_values);
    try std.testing.expect(std.math.isNegativeInf(filtered_negative_values[0]));

    var row_signed_plan = try DeviceLazyFrame.init(gpa, table);
    defer row_signed_plan.deinit();
    try row_signed_plan.withRowPositiveInfCount(&.{}, "row_positive_inf_count");
    try row_signed_plan.withRowNegativeInfCount(&.{"metric"}, "row_negative_inf_count");
    try row_signed_plan.withRowPositiveInfRatio(&.{"metric"}, "row_positive_inf_ratio");
    try row_signed_plan.withRowNegativeInfRatio(&.{"metric"}, "row_negative_inf_ratio");
    try row_signed_plan.withRowPrefixPositiveInfCount(&.{"metric"}, &.{"metric_cum_pos_inf"});
    try row_signed_plan.withRowCumulativeNegativeInfRatio(&.{"metric"}, &.{"metric_cum_neg_inf"});
    try row_signed_plan.select(&.{ "row_positive_inf_count", "row_negative_inf_count", "row_positive_inf_ratio", "row_negative_inf_ratio", "metric_cum_pos_inf", "metric_cum_neg_inf" });
    const row_signed_explain = try row_signed_plan.explain(gpa);
    defer gpa.free(row_signed_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_positive_inf_count([]->row_positive_inf_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_negative_inf_count([metric]->row_negative_inf_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_positive_inf_ratio([metric]->row_positive_inf_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_negative_inf_ratio([metric]->row_negative_inf_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_cumulative_positive_inf_count([metric]->[metric_cum_pos_inf])") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_cumulative_negative_inf_ratio([metric]->[metric_cum_neg_inf])") != null);
    var row_signed = try row_signed_plan.collect();
    defer row_signed.deinit();
    const row_positive_inf_count = try (try row_signed.column("row_positive_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_inf_count);
    const row_negative_inf_count = try (try row_signed.column("row_negative_inf_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_inf_count);
    const row_positive_inf_ratio = try (try row_signed.column("row_positive_inf_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_positive_inf_ratio);
    const row_negative_inf_ratio = try (try row_signed.column("row_negative_inf_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_negative_inf_ratio);
    const metric_cum_pos_inf = try (try row_signed.column("metric_cum_pos_inf")).i64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_pos_inf);
    const metric_cum_neg_inf = try (try row_signed.column("metric_cum_neg_inf")).f64.toOwnedSlice(gpa);
    defer gpa.free(metric_cum_neg_inf);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 0 }, row_positive_inf_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0 }, row_negative_inf_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0, 0.0, 0.0 }, row_positive_inf_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 1.0, 0.0, 0.0 }, row_negative_inf_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 0 }, metric_cum_pos_inf);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 1.0, 0.0, 0.0 }, metric_cum_neg_inf);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.dropNegativeInfsColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());

    var invalid_count_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_count_plan.deinit();
    try invalid_count_plan.withRowPositiveInfCount(&.{"missing"}, "bad_count");
    try std.testing.expectError(error.ColumnNotFound, invalid_count_plan.collect());

    var invalid_ratio_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_ratio_plan.deinit();
    try invalid_ratio_plan.withRowPositiveInfRatio(&.{"missing"}, "bad_ratio");
    try std.testing.expectError(error.ColumnNotFound, invalid_ratio_plan.collect());

    var invalid_prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_prefix_plan.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid_prefix_plan.withRowPrefixNegativeInfRatio(&.{"metric"}, &.{ "metric_cum_neg_inf", "extra_cum_neg_inf" }));
}

test "device lazy frame drops null rows" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.dropNullsColumn("quality");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "drop_nulls[quality]") != null);

    var dropped = try plan.collect();
    defer dropped.deinit();
    try std.testing.expectEqual(@as(usize, 3), dropped.height());
    const quality = try (try dropped.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 3.0, 4.0 }, quality);
    try std.testing.expectEqual(@as(usize, 0), (try dropped.column("quality")).nullCount());

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.dropNullsColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame filters null rows" {
    const gpa = std.testing.allocator;
    var table = try lazyQualityTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.filterNullsColumn("quality");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "filter_nulls_column(quality)") != null);

    var filtered = try plan.collect();
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered.height());
    const quality = try (try filtered.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality);
    const validity = try (try filtered.column("quality")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(validity);
    try std.testing.expectEqualSlices(f64, &.{2.0}, quality);
    try std.testing.expectEqualSlices(bool, &.{false}, validity);

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.filterNullsColumn("missing");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());
}

test "device lazy frame filters all-null rows" {
    const gpa = std.testing.allocator;
    var left = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 1, 2, 3, 4 }, &.{ false, true, false, true }, .cpu);
    defer left.deinit();
    var right = try DeviceColumn.fromSliceWithValidity(i64, gpa, &.{ 10, 20, 30, 40 }, &.{ false, false, true, true }, .cpu);
    defer right.deinit();
    var table = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "left", .data = left },
        .{ .name = "right", .data = right },
    });
    defer table.deinit();

    var drop_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_plan.deinit();
    try drop_plan.dropAllNulls(&.{ "left", "right" });
    const drop_explain = try drop_plan.explain(gpa);
    defer gpa.free(drop_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_explain, "drop_all_nulls[left,right]") != null);
    var dropped = try drop_plan.collect();
    defer dropped.deinit();
    const dropped_left = try (try dropped.column("left")).i64.toOwnedSlice(gpa);
    defer gpa.free(dropped_left);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3, 4 }, dropped_left);

    var filter_plan = try DeviceLazyFrame.init(gpa, table);
    defer filter_plan.deinit();
    try filter_plan.filterAllNulls(&.{ "left", "right" });
    const filter_explain = try filter_plan.explain(gpa);
    defer gpa.free(filter_explain);
    try std.testing.expect(std.mem.indexOf(u8, filter_explain, "filter_all_nulls[left,right]") != null);
    var filtered = try filter_plan.collect();
    defer filtered.deinit();
    try std.testing.expectEqual(@as(usize, 1), filtered.height());
    const filtered_right_validity = try (try filtered.column("right")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(filtered_right_validity);
    try std.testing.expectEqualSlices(bool, &.{false}, filtered_right_validity);
}

test "device lazy frame renames and drops columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnLiteral("segment", i32, 42);
    try plan.withColumnLiteral("always_true", bool, true);
    try plan.withRowIndex("row_nr", 100);
    try plan.renameColumn("sales", "revenue");
    try plan.dropColumn("active");
    try plan.select(&.{ "row_nr", "segment", "always_true", "revenue", "units" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal(segment=scalar:i32)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal(always_true=scalar:bool)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_row_index(row_nr, offset=100)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "rename_column(sales->revenue)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "drop_columns[active]") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 4), result.height());
    try std.testing.expectEqual(@as(usize, 5), result.width());
    try std.testing.expectEqual(@as(?usize, 0), result.columnIndex("row_nr"));
    try std.testing.expectEqual(@as(?usize, 1), result.columnIndex("segment"));
    try std.testing.expectEqual(@as(?usize, 2), result.columnIndex("always_true"));
    try std.testing.expectEqual(@as(?usize, 3), result.columnIndex("revenue"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, null), result.columnIndex("active"));
    const row_nr = try (try result.column("row_nr")).usize.toOwnedSlice(gpa);
    defer gpa.free(row_nr);
    const segment = try (try result.column("segment")).i32.toOwnedSlice(gpa);
    defer gpa.free(segment);
    const always_true = try (try result.column("always_true")).bool.toOwnedSlice(gpa);
    defer gpa.free(always_true);
    const revenue = try (try result.column("revenue")).f64.toOwnedSlice(gpa);
    defer gpa.free(revenue);
    const units = try (try result.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(units);
    try std.testing.expectEqualSlices(usize, &.{ 100, 101, 102, 103 }, row_nr);
    try std.testing.expectEqualSlices(i32, &.{ 42, 42, 42, 42 }, segment);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, always_true);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, revenue);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, units);

    var drop_many_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_many_plan.deinit();
    try drop_many_plan.dropColumns(&.{ "units", "active" });
    var drop_many = try drop_many_plan.collect();
    defer drop_many.deinit();
    try std.testing.expectEqual(@as(usize, 1), drop_many.width());
    try std.testing.expectEqual(@as(?usize, 0), drop_many.columnIndex("sales"));

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.renameColumn("sales", "units");
    try std.testing.expectError(error.InvalidShape, invalid_plan.collect());

    var rename_many_plan = try DeviceLazyFrame.init(gpa, table);
    defer rename_many_plan.deinit();
    try rename_many_plan.renameColumns(&.{ "sales", "units" }, &.{ "revenue", "quantity" });
    const rename_many_explained = try rename_many_plan.explain(gpa);
    defer gpa.free(rename_many_explained);
    try std.testing.expect(std.mem.indexOf(u8, rename_many_explained, "rename_columns[sales->revenue,units->quantity]") != null);
    var renamed_many = try rename_many_plan.collect();
    defer renamed_many.deinit();
    try std.testing.expectEqual(@as(?usize, 0), renamed_many.columnIndex("revenue"));
    try std.testing.expectEqual(@as(?usize, 1), renamed_many.columnIndex("quantity"));
    try std.testing.expectEqual(@as(?usize, 2), renamed_many.columnIndex("active"));

    var prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer prefix_plan.deinit();
    try prefix_plan.addColumnNamePrefix("src_");
    const prefix_explained = try prefix_plan.explain(gpa);
    defer gpa.free(prefix_explained);
    try std.testing.expect(std.mem.indexOf(u8, prefix_explained, "add_column_name_prefix(src_)") != null);
    var prefixed = try prefix_plan.collect();
    defer prefixed.deinit();
    try std.testing.expectEqual(@as(?usize, 0), prefixed.columnIndex("src_sales"));
    try std.testing.expectEqual(@as(?usize, 1), prefixed.columnIndex("src_units"));
    try std.testing.expectEqual(@as(?usize, 2), prefixed.columnIndex("src_active"));

    var suffix_plan = try DeviceLazyFrame.init(gpa, table);
    defer suffix_plan.deinit();
    try suffix_plan.addColumnNameSuffix("_raw");
    const suffix_explained = try suffix_plan.explain(gpa);
    defer gpa.free(suffix_explained);
    try std.testing.expect(std.mem.indexOf(u8, suffix_explained, "add_column_name_suffix(_raw)") != null);
    var suffixed = try suffix_plan.collect();
    defer suffixed.deinit();
    try std.testing.expectEqual(@as(?usize, 0), suffixed.columnIndex("sales_raw"));
    try std.testing.expectEqual(@as(?usize, 1), suffixed.columnIndex("units_raw"));
    try std.testing.expectEqual(@as(?usize, 2), suffixed.columnIndex("active_raw"));

    var strip_prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer strip_prefix_plan.deinit();
    try strip_prefix_plan.addColumnNamePrefix("src_");
    try strip_prefix_plan.stripColumnNamePrefix("src_");
    const strip_prefix_explained = try strip_prefix_plan.explain(gpa);
    defer gpa.free(strip_prefix_explained);
    try std.testing.expect(std.mem.indexOf(u8, strip_prefix_explained, "strip_column_name_prefix(src_)") != null);
    var stripped_prefix = try strip_prefix_plan.collect();
    defer stripped_prefix.deinit();
    try std.testing.expectEqual(@as(?usize, 0), stripped_prefix.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), stripped_prefix.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 2), stripped_prefix.columnIndex("active"));

    var strip_suffix_plan = try DeviceLazyFrame.init(gpa, table);
    defer strip_suffix_plan.deinit();
    try strip_suffix_plan.addColumnNameSuffix("_raw");
    try strip_suffix_plan.stripColumnNameSuffix("_raw");
    const strip_suffix_explained = try strip_suffix_plan.explain(gpa);
    defer gpa.free(strip_suffix_explained);
    try std.testing.expect(std.mem.indexOf(u8, strip_suffix_explained, "strip_column_name_suffix(_raw)") != null);
    var stripped_suffix = try strip_suffix_plan.collect();
    defer stripped_suffix.deinit();
    try std.testing.expectEqual(@as(?usize, 0), stripped_suffix.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), stripped_suffix.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 2), stripped_suffix.columnIndex("active"));

    var replace_prefix_plan = try DeviceLazyFrame.init(gpa, table);
    defer replace_prefix_plan.deinit();
    try replace_prefix_plan.addColumnNamePrefix("src_");
    try replace_prefix_plan.replaceColumnNamePrefix("src_", "raw_");
    const replace_prefix_explained = try replace_prefix_plan.explain(gpa);
    defer gpa.free(replace_prefix_explained);
    try std.testing.expect(std.mem.indexOf(u8, replace_prefix_explained, "replace_column_name_prefix(src_->raw_)") != null);
    var replaced_prefix = try replace_prefix_plan.collect();
    defer replaced_prefix.deinit();
    try std.testing.expectEqual(@as(?usize, 0), replaced_prefix.columnIndex("raw_sales"));
    try std.testing.expectEqual(@as(?usize, 1), replaced_prefix.columnIndex("raw_units"));
    try std.testing.expectEqual(@as(?usize, 2), replaced_prefix.columnIndex("raw_active"));

    var replace_suffix_plan = try DeviceLazyFrame.init(gpa, table);
    defer replace_suffix_plan.deinit();
    try replace_suffix_plan.addColumnNameSuffix("_raw");
    try replace_suffix_plan.replaceColumnNameSuffix("_raw", "_clean");
    const replace_suffix_explained = try replace_suffix_plan.explain(gpa);
    defer gpa.free(replace_suffix_explained);
    try std.testing.expect(std.mem.indexOf(u8, replace_suffix_explained, "replace_column_name_suffix(_raw->_clean)") != null);
    var replaced_suffix = try replace_suffix_plan.collect();
    defer replaced_suffix.deinit();
    try std.testing.expectEqual(@as(?usize, 0), replaced_suffix.columnIndex("sales_clean"));
    try std.testing.expectEqual(@as(?usize, 1), replaced_suffix.columnIndex("units_clean"));
    try std.testing.expectEqual(@as(?usize, 2), replaced_suffix.columnIndex("active_clean"));

    var invalid_many_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_many_plan.deinit();
    try invalid_many_plan.renameColumns(&.{ "sales", "units" }, &.{ "revenue", "revenue" });
    try std.testing.expectError(error.InvalidShape, invalid_many_plan.collect());

    var invalid_index_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_index_plan.deinit();
    try invalid_index_plan.withRowIndex("sales", 0);
    try std.testing.expectError(error.InvalidShape, invalid_index_plan.collect());

    var replace_literal_plan = try DeviceLazyFrame.init(gpa, table);
    defer replace_literal_plan.deinit();
    try replace_literal_plan.withColumnLiteral("sales", f64, 1.0);
    try replace_literal_plan.select(&.{"sales"});
    var replaced_literal = try replace_literal_plan.collect();
    defer replaced_literal.deinit();
    const replaced_sales = try (try replaced_literal.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(replaced_sales);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 1.0 }, replaced_sales);
}

test "device lazy frame places literal columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.withColumnLiteralAt("segment", i32, 42, 0);
    try plan.withColumnLiteralBefore("rank", i16, 5, "units");
    try plan.withColumnLiteralAfter("score", f32, 1.5, "active");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal_at(segment=scalar:i32, index=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal_before(rank=scalar:i16 before units)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal_after(score=scalar:f32 after active)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 6), result.width());
    try std.testing.expectEqual(@as(?usize, 0), result.columnIndex("segment"));
    try std.testing.expectEqual(@as(?usize, 1), result.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), result.columnIndex("rank"));
    try std.testing.expectEqual(@as(?usize, 3), result.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 4), result.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 5), result.columnIndex("score"));
    const segment = try (try result.column("segment")).i32.toOwnedSlice(gpa);
    defer gpa.free(segment);
    const score = try (try result.column("score")).f32.toOwnedSlice(gpa);
    defer gpa.free(score);
    try std.testing.expectEqualSlices(i32, &.{ 42, 42, 42, 42 }, segment);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, 1.5, 1.5, 1.5 }, score);

    var replace_plan = try DeviceLazyFrame.init(gpa, table);
    defer replace_plan.deinit();
    try replace_plan.withColumnLiteralAt("sales", f64, 9.0, 2);
    var replaced = try replace_plan.collect();
    defer replaced.deinit();
    try std.testing.expectEqual(@as(usize, 3), replaced.width());
    try std.testing.expectEqual(@as(?usize, 0), replaced.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), replaced.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 2), replaced.columnIndex("sales"));
    const replaced_sales = try (try replaced.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(replaced_sales);
    try std.testing.expectEqualSlices(f64, &.{ 9.0, 9.0, 9.0, 9.0 }, replaced_sales);

    var missing_anchor_plan = try DeviceLazyFrame.init(gpa, table);
    defer missing_anchor_plan.deinit();
    try missing_anchor_plan.withColumnLiteralBefore("bad", i8, 1, "missing");
    try std.testing.expectError(error.ColumnNotFound, missing_anchor_plan.collect());

    var bounds_plan = try DeviceLazyFrame.init(gpa, table);
    defer bounds_plan.deinit();
    try bounds_plan.withColumnLiteralAt("bad", i8, 1, table.width() + 1);
    try std.testing.expectError(error.IndexOutOfBounds, bounds_plan.collect());
}

test "device lazy frame moves columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var move_plan = try DeviceLazyFrame.init(gpa, table);
    defer move_plan.deinit();
    try move_plan.moveColumn("active", 0);

    const move_explain = try move_plan.explain(gpa);
    defer gpa.free(move_explain);
    try std.testing.expect(std.mem.indexOf(u8, move_explain, "move_column(active -> index=0)") != null);

    var moved = try move_plan.collect();
    defer moved.deinit();
    try std.testing.expectEqual(@as(usize, 3), moved.width());
    try std.testing.expectEqual(@as(?usize, 0), moved.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 1), moved.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), moved.columnIndex("units"));
    const moved_active = try (try moved.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(moved_active);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, moved_active);

    var before_plan = try DeviceLazyFrame.init(gpa, table);
    defer before_plan.deinit();
    try before_plan.moveColumnBefore("units", "sales");
    const before_explain = try before_plan.explain(gpa);
    defer gpa.free(before_explain);
    try std.testing.expect(std.mem.indexOf(u8, before_explain, "move_column_before(units before sales)") != null);

    var before = try before_plan.collect();
    defer before.deinit();
    try std.testing.expectEqual(@as(?usize, 0), before.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), before.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 2), before.columnIndex("active"));

    var after_plan = try DeviceLazyFrame.init(gpa, table);
    defer after_plan.deinit();
    try after_plan.moveColumnAfter("sales", "active");
    const after_explain = try after_plan.explain(gpa);
    defer gpa.free(after_explain);
    try std.testing.expect(std.mem.indexOf(u8, after_explain, "move_column_after(sales after active)") != null);

    var after = try after_plan.collect();
    defer after.deinit();
    try std.testing.expectEqual(@as(?usize, 0), after.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 1), after.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 2), after.columnIndex("sales"));
    const after_sales = try (try after.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(after_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, after_sales);

    var missing_plan = try DeviceLazyFrame.init(gpa, table);
    defer missing_plan.deinit();
    try missing_plan.moveColumn("missing", 0);
    try std.testing.expectError(error.ColumnNotFound, missing_plan.collect());

    var bounds_plan = try DeviceLazyFrame.init(gpa, table);
    defer bounds_plan.deinit();
    try bounds_plan.moveColumn("sales", table.width());
    try std.testing.expectError(error.IndexOutOfBounds, bounds_plan.collect());
}

test "device lazy frame copies columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.copyColumn("sales", "sales_copy");
    try plan.copyColumnBefore("active", "active_copy", "units");
    try plan.copyColumnAfter("units", "units_after", "active");

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "copy_column(sales->sales_copy)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "copy_column_before(active->active_copy before units)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "copy_column_after(units->units_after after active)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 6), result.width());
    try std.testing.expectEqual(@as(?usize, 0), result.columnIndex("sales"));
    try std.testing.expectEqual(@as(?usize, 1), result.columnIndex("active_copy"));
    try std.testing.expectEqual(@as(?usize, 2), result.columnIndex("units"));
    try std.testing.expectEqual(@as(?usize, 3), result.columnIndex("active"));
    try std.testing.expectEqual(@as(?usize, 4), result.columnIndex("units_after"));
    try std.testing.expectEqual(@as(?usize, 5), result.columnIndex("sales_copy"));
    const copied_sales = try (try result.column("sales_copy")).f64.toOwnedSlice(gpa);
    defer gpa.free(copied_sales);
    const copied_active = try (try result.column("active_copy")).bool.toOwnedSlice(gpa);
    defer gpa.free(copied_active);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0 }, copied_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, copied_active);

    var at_plan = try DeviceLazyFrame.init(gpa, table);
    defer at_plan.deinit();
    try at_plan.copyColumnAt("units", "units_first", 0);
    const at_explained = try at_plan.explain(gpa);
    defer gpa.free(at_explained);
    try std.testing.expect(std.mem.indexOf(u8, at_explained, "copy_column_at(units->units_first, index=0)") != null);

    var at_result = try at_plan.collect();
    defer at_result.deinit();
    try std.testing.expectEqual(@as(?usize, 0), at_result.columnIndex("units_first"));
    try std.testing.expectEqual(@as(?usize, 1), at_result.columnIndex("sales"));
    const units_first = try (try at_result.column("units_first")).i64.toOwnedSlice(gpa);
    defer gpa.free(units_first);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, units_first);

    var missing_plan = try DeviceLazyFrame.init(gpa, table);
    defer missing_plan.deinit();
    try missing_plan.copyColumn("missing", "copy");
    try std.testing.expectError(error.ColumnNotFound, missing_plan.collect());

    var bounds_plan = try DeviceLazyFrame.init(gpa, table);
    defer bounds_plan.deinit();
    try bounds_plan.copyColumnAt("sales", "copy", table.width() + 1);
    try std.testing.expectError(error.IndexOutOfBounds, bounds_plan.collect());
}

test "device lazy frame collects topk operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var topk_plan = try DeviceLazyFrame.init(gpa, table);
    defer topk_plan.deinit();
    try topk_plan.sortBy("sales", .{ .descending = true });
    try topk_plan.head(2);
    const topk_explain = try topk_plan.explain(gpa);
    defer gpa.free(topk_explain);
    try std.testing.expect(std.mem.indexOf(u8, topk_explain, "top_k(sales, k=2") != null);
    var topk = try topk_plan.collect();
    defer topk.deinit();
    const topk_sales = try (try topk.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(topk_sales);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, topk_sales);

    var bottomk_plan = try DeviceLazyFrame.init(gpa, table);
    defer bottomk_plan.deinit();
    try bottomk_plan.bottomKBy("sales", 2, .{});
    const bottomk_explain = try bottomk_plan.explain(gpa);
    defer gpa.free(bottomk_explain);
    try std.testing.expect(std.mem.indexOf(u8, bottomk_explain, "top_k(sales, k=2, desc=false)") != null);
    var bottomk = try bottomk_plan.collect();
    defer bottomk.deinit();
    const bottomk_sales = try (try bottomk.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(bottomk_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, bottomk_sales);
}

test "device lazy frame sorts by multiple columns" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.sortByColumns(&.{ "active", "units" }, &.{ .{ .descending = true }, .{ .descending = true } });
    try plan.select(&.{ "sales", "active", "units" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "sort_by_columns[active:desc=true,units:desc=true]") != null);

    var sorted = try plan.collect();
    defer sorted.deinit();
    const sales = try (try sorted.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales);
    const units = try (try sorted.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(units);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0, 2.0, 3.0 }, sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3, 1, 2 }, units);

    var invalid = try DeviceLazyFrame.init(gpa, table);
    defer invalid.deinit();
    try std.testing.expectError(error.LengthMismatch, invalid.sortByColumns(&.{"active"}, &.{ .{ .descending = true }, .{ .descending = true } }));

    var optimized_topk_plan = try DeviceLazyFrame.init(gpa, table);
    defer optimized_topk_plan.deinit();
    try optimized_topk_plan.sortByColumns(&.{ "active", "units" }, &.{ .{ .descending = true }, .{ .descending = true } });
    try optimized_topk_plan.head(2);
    const optimized_topk_explain = try optimized_topk_plan.explain(gpa);
    defer gpa.free(optimized_topk_explain);
    try std.testing.expect(std.mem.indexOf(u8, optimized_topk_explain, "top_k_columns(k=2)[active:desc=true,units:desc=true]") != null);
}

test "device lazy frame collects multi-key topk operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();

    var plan = try DeviceLazyFrame.init(gpa, table);
    defer plan.deinit();
    try plan.topKByColumns(&.{ "active", "units" }, 2, &.{ .{ .descending = true }, .{ .descending = true } });
    try plan.select(&.{ "sales", "active", "units" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "top_k_columns(k=2)[active:desc=true,units:desc=true]") != null);

    var topk = try plan.collect();
    defer topk.deinit();
    const sales = try (try topk.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales);
    const units = try (try topk.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(units);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0 }, sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3 }, units);

    var bottom_plan = try DeviceLazyFrame.init(gpa, table);
    defer bottom_plan.deinit();
    try bottom_plan.bottomKByColumns(&.{ "active", "units" }, 2, &.{ .{ .descending = false }, .{ .descending = false } });
    try bottom_plan.select(&.{ "sales", "active", "units" });

    const bottom_explained = try bottom_plan.explain(gpa);
    defer gpa.free(bottom_explained);
    try std.testing.expect(std.mem.indexOf(u8, bottom_explained, "top_k_columns(k=2)[active:desc=false,units:desc=false]") != null);

    var bottom = try bottom_plan.collect();
    defer bottom.deinit();
    const bottom_sales = try (try bottom.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(bottom_sales);
    const bottom_units = try (try bottom.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(bottom_units);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 2.0 }, bottom_sales);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1 }, bottom_units);
}

test "device lazy frame collects row slice operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var slice_plan = try DeviceLazyFrame.init(gpa, table);
    defer slice_plan.deinit();
    try slice_plan.sliceRows(1, 3);
    try slice_plan.select(&.{"sales"});

    const slice_explain = try slice_plan.explain(gpa);
    defer gpa.free(slice_explain);
    try std.testing.expect(std.mem.indexOf(u8, slice_explain, "slice_rows(1..3)") != null);

    var sliced = try slice_plan.collect();
    defer sliced.deinit();
    try std.testing.expectEqual(@as(usize, 2), sliced.height());
    try std.testing.expectEqual(@as(usize, 1), sliced.width());
    const sliced_sales = try (try sliced.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sliced_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, sliced_sales);

    var limit_plan = try DeviceLazyFrame.init(gpa, table);
    defer limit_plan.deinit();
    try limit_plan.limit(2);
    try limit_plan.select(&.{"sales"});
    var limited = try limit_plan.collect();
    defer limited.deinit();
    const limited_sales = try (try limited.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(limited_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, limited_sales);

    var first_plan = try DeviceLazyFrame.init(gpa, table);
    defer first_plan.deinit();
    try first_plan.firstRow();
    try first_plan.select(&.{"sales"});
    var first = try first_plan.collect();
    defer first.deinit();
    const first_sales = try (try first.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(first_sales);
    try std.testing.expectEqualSlices(f64, &.{2.0}, first_sales);

    var last_plan = try DeviceLazyFrame.init(gpa, table);
    defer last_plan.deinit();
    try last_plan.lastRow();
    try last_plan.select(&.{"sales"});
    var last = try last_plan.collect();
    defer last.deinit();
    const last_sales = try (try last.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(last_sales);
    try std.testing.expectEqualSlices(f64, &.{7.0}, last_sales);

    var offset_plan = try DeviceLazyFrame.init(gpa, table);
    defer offset_plan.deinit();
    try offset_plan.offset(2);
    try offset_plan.select(&.{"sales"});
    var offset = try offset_plan.collect();
    defer offset.deinit();
    const offset_sales = try (try offset.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(offset_sales);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 7.0 }, offset_sales);

    var offset_limit_plan = try DeviceLazyFrame.init(gpa, table);
    defer offset_limit_plan.deinit();
    try offset_limit_plan.offset(1);
    try offset_limit_plan.limit(2);
    try offset_limit_plan.select(&.{"sales"});
    const offset_limit_explain = try offset_limit_plan.explain(gpa);
    defer gpa.free(offset_limit_explain);
    try std.testing.expect(std.mem.indexOf(u8, offset_limit_explain, "slice_rows(1..3)") != null);
    var offset_limited = try offset_limit_plan.collect();
    defer offset_limited.deinit();
    const offset_limited_sales = try (try offset_limited.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(offset_limited_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, offset_limited_sales);

    var slice_len_plan = try DeviceLazyFrame.init(gpa, table);
    defer slice_len_plan.deinit();
    try slice_len_plan.sliceRowsLen(1, 2);
    try slice_len_plan.select(&.{"sales"});
    var slice_len = try slice_len_plan.collect();
    defer slice_len.deinit();
    const slice_len_sales = try (try slice_len.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(slice_len_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 5.0 }, slice_len_sales);

    var signed_slice_plan = try DeviceLazyFrame.init(gpa, table);
    defer signed_slice_plan.deinit();
    try signed_slice_plan.sliceRowsSigned(-2, 2);
    try signed_slice_plan.select(&.{ "sales", "units" });
    const signed_slice_explain = try signed_slice_plan.explain(gpa);
    defer gpa.free(signed_slice_explain);
    try std.testing.expect(std.mem.indexOf(u8, signed_slice_explain, "slice_rows_signed(start=-2, len=2)") != null);
    var signed_sliced = try signed_slice_plan.collect();
    defer signed_sliced.deinit();
    const signed_sliced_sales = try (try signed_sliced.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(signed_sliced_sales);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 7.0 }, signed_sliced_sales);

    var invalid_signed_slice_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_signed_slice_plan.deinit();
    try invalid_signed_slice_plan.sliceRowsSigned(-1, 2);
    try std.testing.expectError(error.IndexOutOfBounds, invalid_signed_slice_plan.collect());

    var signed_step_plan = try DeviceLazyFrame.init(gpa, table);
    defer signed_step_plan.deinit();
    try signed_step_plan.sliceRowsSignedStep(-4, 4, 2);
    try signed_step_plan.select(&.{"sales"});
    const signed_step_explain = try signed_step_plan.explain(gpa);
    defer gpa.free(signed_step_explain);
    try std.testing.expect(std.mem.indexOf(u8, signed_step_explain, "slice_rows_signed_step(-4..4, step=2)") != null);
    var signed_stepped = try signed_step_plan.collect();
    defer signed_stepped.deinit();
    const signed_stepped_sales = try (try signed_stepped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(signed_stepped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, signed_stepped_sales);

    var invalid_signed_step_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_signed_step_plan.deinit();
    try invalid_signed_step_plan.sliceRowsSignedStep(-4, 4, 0);
    try std.testing.expectError(error.InvalidShape, invalid_signed_step_plan.collect());

    var len_plan = try DeviceLazyFrame.init(gpa, table);
    defer len_plan.deinit();
    try len_plan.slice(2, 8);
    var len_sliced = try len_plan.collect();
    defer len_sliced.deinit();
    const len_sliced_sales = try (try len_sliced.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(len_sliced_sales);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 7.0 }, len_sliced_sales);

    var drop_rows_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_rows_plan.deinit();
    try drop_rows_plan.dropRows(&.{ 1, 1 });
    try drop_rows_plan.select(&.{ "sales", "units" });
    const drop_rows_explain = try drop_rows_plan.explain(gpa);
    defer gpa.free(drop_rows_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_rows_explain, "drop_rows([1,1])") != null);
    var rows_dropped = try drop_rows_plan.collect();
    defer rows_dropped.deinit();
    try std.testing.expectEqual(@as(usize, 3), rows_dropped.height());
    try std.testing.expectEqual(@as(usize, 2), rows_dropped.width());
    const rows_dropped_sales = try (try rows_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(rows_dropped_sales);
    const rows_dropped_units = try (try rows_dropped.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(rows_dropped_units);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0, 7.0 }, rows_dropped_sales);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 4 }, rows_dropped_units);

    var drop_mode_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_mode_plan.deinit();
    try drop_mode_plan.dropRowsMode(&.{5}, .wrap);
    try drop_mode_plan.dropRowsSignedMode(&.{-9}, .clip);
    try drop_mode_plan.dropRowsSigned(&.{-1});
    try drop_mode_plan.select(&.{ "sales", "units" });
    const drop_mode_explain = try drop_mode_plan.explain(gpa);
    defer gpa.free(drop_mode_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_mode_explain, "drop_rows_mode([5], mode:wrap)") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_mode_explain, "drop_rows_signed_mode([-9], mode:clip)") != null);
    try std.testing.expect(std.mem.indexOf(u8, drop_mode_explain, "drop_rows_signed([-1])") != null);
    var dropped_mode = try drop_mode_plan.collect();
    defer dropped_mode.deinit();
    const dropped_mode_sales = try (try dropped_mode.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_mode_sales);
    try std.testing.expectEqualSlices(f64, &.{5.0}, dropped_mode_sales);

    var invalid_drop_rows_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_drop_rows_plan.deinit();
    try invalid_drop_rows_plan.dropRows(&.{table.height()});
    try std.testing.expectError(error.IndexOutOfBounds, invalid_drop_rows_plan.collect());

    var drop_range_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_range_plan.deinit();
    try drop_range_plan.dropRowRange(1, 3);
    try drop_range_plan.select(&.{ "sales", "units" });
    const drop_range_explain = try drop_range_plan.explain(gpa);
    defer gpa.free(drop_range_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_range_explain, "drop_row_range(1..3)") != null);
    var range_dropped = try drop_range_plan.collect();
    defer range_dropped.deinit();
    const range_dropped_sales = try (try range_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(range_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 7.0 }, range_dropped_sales);

    var drop_first_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_first_plan.deinit();
    try drop_first_plan.dropFirstRows(2);
    try drop_first_plan.select(&.{"sales"});
    const drop_first_explain = try drop_first_plan.explain(gpa);
    defer gpa.free(drop_first_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_first_explain, "drop_row_range(0..2)") != null);
    var first_dropped = try drop_first_plan.collect();
    defer first_dropped.deinit();
    const first_dropped_sales = try (try first_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(first_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 7.0 }, first_dropped_sales);

    var drop_last_plan = try DeviceLazyFrame.init(gpa, table);
    defer drop_last_plan.deinit();
    try drop_last_plan.dropLastRows(1);
    try drop_last_plan.select(&.{"sales"});
    const drop_last_explain = try drop_last_plan.explain(gpa);
    defer gpa.free(drop_last_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_last_explain, "drop_last_rows(1)") != null);
    var last_dropped = try drop_last_plan.collect();
    defer last_dropped.deinit();
    const last_dropped_sales = try (try last_dropped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(last_dropped_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0 }, last_dropped_sales);

    var step_plan = try DeviceLazyFrame.init(gpa, table);
    defer step_plan.deinit();
    try step_plan.sliceRowsStep(0, table.height(), 2);
    try step_plan.select(&.{ "sales", "units" });
    const step_explain = try step_plan.explain(gpa);
    defer gpa.free(step_explain);
    try std.testing.expect(std.mem.indexOf(u8, step_explain, "slice_rows_step(0..4, step=2)") != null);
    var stepped = try step_plan.collect();
    defer stepped.deinit();
    try std.testing.expectEqual(@as(usize, 2), stepped.height());
    try std.testing.expectEqual(@as(usize, 2), stepped.width());
    const stepped_sales = try (try stepped.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(stepped_sales);
    const stepped_units = try (try stepped.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(stepped_units);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, stepped_sales);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3 }, stepped_units);

    var invalid_step_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_step_plan.deinit();
    try invalid_step_plan.sliceRowsStep(0, table.height(), 0);
    try std.testing.expectError(error.InvalidShape, invalid_step_plan.collect());

    var stride_plan = try DeviceLazyFrame.init(gpa, table);
    defer stride_plan.deinit();
    try stride_plan.strideRows(0, 2);
    try stride_plan.select(&.{ "sales", "units" });
    const stride_explain = try stride_plan.explain(gpa);
    defer gpa.free(stride_explain);
    try std.testing.expect(std.mem.indexOf(u8, stride_explain, "stride_rows(start=0, step=2)") != null);
    var strided = try stride_plan.collect();
    defer strided.deinit();
    try std.testing.expectEqual(@as(usize, 2), strided.height());
    try std.testing.expectEqual(@as(usize, 2), strided.width());
    const strided_sales = try (try strided.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(strided_sales);
    const strided_units = try (try strided.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(strided_units);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, strided_sales);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3 }, strided_units);

    var empty_stride_plan = try DeviceLazyFrame.init(gpa, table);
    defer empty_stride_plan.deinit();
    try empty_stride_plan.strideRows(table.height(), 1);
    var empty_stride = try empty_stride_plan.collect();
    defer empty_stride.deinit();
    try std.testing.expectEqual(@as(usize, 0), empty_stride.height());
    try std.testing.expectEqual(table.width(), empty_stride.width());

    var invalid_stride_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_stride_plan.deinit();
    try invalid_stride_plan.strideRows(0, 0);
    try std.testing.expectError(error.InvalidShape, invalid_stride_plan.collect());

    var sample_plan = try DeviceLazyFrame.init(gpa, table);
    defer sample_plan.deinit();
    try sample_plan.sampleRows(2, 1234);
    try sample_plan.select(&.{ "sales", "units" });
    const sample_explain = try sample_plan.explain(gpa);
    defer gpa.free(sample_explain);
    try std.testing.expect(std.mem.indexOf(u8, sample_explain, "sample_rows(count=2, seed=1234)") != null);
    var sampled = try sample_plan.collect();
    defer sampled.deinit();
    try std.testing.expectEqual(@as(usize, 2), sampled.height());
    try std.testing.expectEqual(@as(usize, 2), sampled.width());

    var sample_again_plan = try DeviceLazyFrame.init(gpa, table);
    defer sample_again_plan.deinit();
    try sample_again_plan.sampleRows(2, 1234);
    try sample_again_plan.select(&.{ "sales", "units" });
    var sampled_again = try sample_again_plan.collect();
    defer sampled_again.deinit();
    const sampled_sales = try (try sampled.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sampled_sales);
    const sampled_again_sales = try (try sampled_again.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sampled_again_sales);
    try std.testing.expectEqualSlices(f64, sampled_sales, sampled_again_sales);

    var shuffle_plan = try DeviceLazyFrame.init(gpa, table);
    defer shuffle_plan.deinit();
    try shuffle_plan.shuffleRows(1234);
    try shuffle_plan.select(&.{ "sales", "units" });
    const shuffle_explain = try shuffle_plan.explain(gpa);
    defer gpa.free(shuffle_explain);
    try std.testing.expect(std.mem.indexOf(u8, shuffle_explain, "sample_rows_fraction(fraction=1, seed=1234)") != null);
    var shuffled = try shuffle_plan.collect();
    defer shuffled.deinit();
    try std.testing.expectEqual(table.height(), shuffled.height());

    var sample_fraction_plan = try DeviceLazyFrame.init(gpa, table);
    defer sample_fraction_plan.deinit();
    try sample_fraction_plan.sampleRowsFraction(0.5, 1234);
    try sample_fraction_plan.select(&.{ "sales", "units" });
    const sample_fraction_explain = try sample_fraction_plan.explain(gpa);
    defer gpa.free(sample_fraction_explain);
    try std.testing.expect(std.mem.indexOf(u8, sample_fraction_explain, "sample_rows_fraction(fraction=0.5, seed=1234)") != null);
    var sampled_fraction = try sample_fraction_plan.collect();
    defer sampled_fraction.deinit();
    try std.testing.expectEqual(@as(usize, 2), sampled_fraction.height());

    var sample_frac_alias_plan = try DeviceLazyFrame.init(gpa, table);
    defer sample_frac_alias_plan.deinit();
    try sample_frac_alias_plan.sampleFrac(0.5, 1234);
    try sample_frac_alias_plan.select(&.{ "sales", "units" });
    var sampled_frac_alias = try sample_frac_alias_plan.collect();
    defer sampled_frac_alias.deinit();
    try std.testing.expectEqual(@as(usize, 2), sampled_frac_alias.height());

    var invalid_sample_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_sample_plan.deinit();
    try invalid_sample_plan.sampleRows(table.height() + 1, 1234);
    try std.testing.expectError(error.InvalidShape, invalid_sample_plan.collect());

    var replacement_plan = try DeviceLazyFrame.init(gpa, table);
    defer replacement_plan.deinit();
    try replacement_plan.sampleRowsWithReplacement(table.height() + 2, 4321);
    try replacement_plan.select(&.{ "sales", "units" });
    const replacement_explain = try replacement_plan.explain(gpa);
    defer gpa.free(replacement_explain);
    try std.testing.expect(std.mem.indexOf(u8, replacement_explain, "sample_rows_with_replacement(count=6, seed=4321)") != null);
    var sampled_replacement = try replacement_plan.collect();
    defer sampled_replacement.deinit();
    try std.testing.expectEqual(@as(usize, 6), sampled_replacement.height());
    try std.testing.expectEqual(@as(usize, 2), sampled_replacement.width());

    var replacement_again_plan = try DeviceLazyFrame.init(gpa, table);
    defer replacement_again_plan.deinit();
    try replacement_again_plan.sampleRowsWithReplacement(table.height() + 2, 4321);
    try replacement_again_plan.select(&.{ "sales", "units" });
    var sampled_replacement_again = try replacement_again_plan.collect();
    defer sampled_replacement_again.deinit();
    const replacement_sales = try (try sampled_replacement.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(replacement_sales);
    const replacement_again_sales = try (try sampled_replacement_again.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(replacement_again_sales);
    try std.testing.expectEqualSlices(f64, replacement_sales, replacement_again_sales);

    var replacement_fraction_plan = try DeviceLazyFrame.init(gpa, table);
    defer replacement_fraction_plan.deinit();
    try replacement_fraction_plan.sampleRowsFractionWithReplacement(1.5, 4321);
    try replacement_fraction_plan.select(&.{ "sales", "units" });
    const replacement_fraction_explain = try replacement_fraction_plan.explain(gpa);
    defer gpa.free(replacement_fraction_explain);
    try std.testing.expect(std.mem.indexOf(u8, replacement_fraction_explain, "sample_rows_fraction_with_replacement(fraction=1.5, seed=4321)") != null);
    var sampled_fraction_replacement = try replacement_fraction_plan.collect();
    defer sampled_fraction_replacement.deinit();
    try std.testing.expectEqual(@as(usize, 6), sampled_fraction_replacement.height());

    var replacement_frac_alias_plan = try DeviceLazyFrame.init(gpa, table);
    defer replacement_frac_alias_plan.deinit();
    try replacement_frac_alias_plan.sampleFracWithReplacement(1.5, 4321);
    try replacement_frac_alias_plan.select(&.{ "sales", "units" });
    var sampled_frac_alias_replacement = try replacement_frac_alias_plan.collect();
    defer sampled_frac_alias_replacement.deinit();
    try std.testing.expectEqual(@as(usize, 6), sampled_frac_alias_replacement.height());

    var put_flat_plan = try DeviceLazyFrame.init(gpa, table);
    defer put_flat_plan.deinit();
    try put_flat_plan.withColumnPutFlatScalar("sales_put", "sales", &.{ 1, 3 }, f64, 9.0);
    try put_flat_plan.withColumnNeg("sales_put_values_source", "sales");
    try put_flat_plan.withColumnPutFlat("sales_put_values", "sales", &.{ 0, 2, 1, 3 }, "sales_put_values_source");
    try put_flat_plan.withColumnPutFlatScalarSigned("sales_put_signed", "sales", &.{-1}, f64, -5.0);
    try put_flat_plan.withColumnPutFlatScalarMode("sales_put_wrap", "sales", &.{5}, f64, 11.0, .wrap);
    try put_flat_plan.select(&.{ "sales", "sales_put", "sales_put_values", "sales_put_signed", "sales_put_wrap" });
    const put_flat_explain = try put_flat_plan.explain(gpa);
    defer gpa.free(put_flat_explain);
    try std.testing.expect(std.mem.indexOf(u8, put_flat_explain, "with_column_put_flat_scalar(sales_put=put_flat(sales, indices=[1,3], scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, put_flat_explain, "with_column_put_flat(sales_put_values=put_flat(sales, indices=[0,2,1,3], values:sales_put_values_source))") != null);
    try std.testing.expect(std.mem.indexOf(u8, put_flat_explain, "with_column_put_flat_scalar_signed(sales_put_signed=put_flat_signed(sales, indices=[-1], scalar:f64))") != null);
    try std.testing.expect(std.mem.indexOf(u8, put_flat_explain, "with_column_put_flat_scalar_mode(sales_put_wrap=put_flat(sales, indices=[5], scalar:f64, mode:wrap))") != null);
    var put_flat_result = try put_flat_plan.collect();
    defer put_flat_result.deinit();
    const put_flat_sales = try (try put_flat_result.column("sales_put")).f64.toOwnedSlice(gpa);
    defer gpa.free(put_flat_sales);
    const put_flat_value_sales = try (try put_flat_result.column("sales_put_values")).f64.toOwnedSlice(gpa);
    defer gpa.free(put_flat_value_sales);
    const put_flat_signed_sales = try (try put_flat_result.column("sales_put_signed")).f64.toOwnedSlice(gpa);
    defer gpa.free(put_flat_signed_sales);
    const put_flat_wrap_sales = try (try put_flat_result.column("sales_put_wrap")).f64.toOwnedSlice(gpa);
    defer gpa.free(put_flat_wrap_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 9.0, 5.0, 9.0 }, put_flat_sales);
    try std.testing.expectEqualSlices(f64, &.{ -2.0, -5.0, -3.0, -7.0 }, put_flat_value_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, -5.0 }, put_flat_signed_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 11.0, 5.0, 7.0 }, put_flat_wrap_sales);

    var repeat_plan = try DeviceLazyFrame.init(gpa, table);
    defer repeat_plan.deinit();
    try repeat_plan.repeatRows(2);
    try repeat_plan.select(&.{ "sales", "units" });
    const repeat_explain = try repeat_plan.explain(gpa);
    defer gpa.free(repeat_explain);
    try std.testing.expect(std.mem.indexOf(u8, repeat_explain, "repeat_rows(2)") != null);
    var repeated_lazy = try repeat_plan.collect();
    defer repeated_lazy.deinit();
    try std.testing.expectEqual(@as(usize, 8), repeated_lazy.height());
    const repeated_lazy_sales = try (try repeated_lazy.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(repeated_lazy_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 2.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0 }, repeated_lazy_sales);

    var tile_plan = try DeviceLazyFrame.init(gpa, table);
    defer tile_plan.deinit();
    try tile_plan.tileRows(2);
    try tile_plan.select(&.{ "sales", "units" });
    const tile_explain = try tile_plan.explain(gpa);
    defer gpa.free(tile_explain);
    try std.testing.expect(std.mem.indexOf(u8, tile_explain, "tile_rows(2)") != null);
    var tiled_lazy = try tile_plan.collect();
    defer tiled_lazy.deinit();
    try std.testing.expectEqual(@as(usize, 8), tiled_lazy.height());
    const tiled_lazy_sales = try (try tiled_lazy.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(tiled_lazy_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 5.0, 7.0, 2.0, 3.0, 5.0, 7.0 }, tiled_lazy_sales);

    var repeat_by_plan = try DeviceLazyFrame.init(gpa, table);
    defer repeat_by_plan.deinit();
    try repeat_by_plan.repeatRowsByColumn("units");
    try repeat_by_plan.select(&.{ "sales", "units" });
    const repeat_by_explain = try repeat_by_plan.explain(gpa);
    defer gpa.free(repeat_by_explain);
    try std.testing.expect(std.mem.indexOf(u8, repeat_by_explain, "repeat_rows_by(units)") != null);
    var repeated_by_lazy = try repeat_by_plan.collect();
    defer repeated_by_lazy.deinit();
    try std.testing.expectEqual(@as(usize, 10), repeated_by_lazy.height());
    const repeated_by_lazy_sales = try (try repeated_by_lazy.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(repeated_by_lazy_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0 }, repeated_by_lazy_sales);

    var take_plan = try DeviceLazyFrame.init(gpa, table);
    defer take_plan.deinit();
    try take_plan.take(&.{ 3, 1, 1 });
    try take_plan.select(&.{ "sales", "units" });
    const take_explain = try take_plan.explain(gpa);
    defer gpa.free(take_explain);
    try std.testing.expect(std.mem.indexOf(u8, take_explain, "take_rows([3,1,1])") != null);
    var taken = try take_plan.collect();
    defer taken.deinit();
    try std.testing.expectEqual(@as(usize, 3), taken.height());
    try std.testing.expectEqual(@as(usize, 2), taken.width());
    const taken_sales = try (try taken.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_sales);
    const taken_units = try (try taken.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(taken_units);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 3.0, 3.0 }, taken_sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 2, 2 }, taken_units);

    var invalid_take_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_take_plan.deinit();
    try invalid_take_plan.take(&.{4});
    try std.testing.expectError(error.IndexOutOfBounds, invalid_take_plan.collect());

    var optional_take_plan = try DeviceLazyFrame.init(gpa, table);
    defer optional_take_plan.deinit();
    try optional_take_plan.takeOptional(&.{ 2, null, 1 });
    try optional_take_plan.select(&.{ "sales", "units" });
    const optional_take_explain = try optional_take_plan.explain(gpa);
    defer gpa.free(optional_take_explain);
    try std.testing.expect(std.mem.indexOf(u8, optional_take_explain, "take_rows_optional([2,null,1])") != null);
    var taken_optional = try optional_take_plan.collect();
    defer taken_optional.deinit();
    try std.testing.expectEqual(@as(usize, 3), taken_optional.height());
    const taken_optional_sales = try (try taken_optional.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_optional_sales);
    const taken_optional_sales_validity = try (try taken_optional.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(taken_optional_sales_validity);
    const taken_optional_units = try (try taken_optional.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(taken_optional_units);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 0.0, 3.0 }, taken_optional_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, taken_optional_sales_validity);
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 2 }, taken_optional_units);

    var row_pick = try DeviceColumn.fromSliceWithValidity(isize, gpa, &.{ 2, 0, -1, 1 }, &.{ true, false, true, true }, .cpu);
    defer row_pick.deinit();
    var take_by_source = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = (try table.column("sales")).* },
        .{ .name = "units", .data = (try table.column("units")).* },
        .{ .name = "row_pick", .data = row_pick },
    });
    defer take_by_source.deinit();
    var take_by_plan = try DeviceLazyFrame.init(gpa, take_by_source);
    defer take_by_plan.deinit();
    try take_by_plan.takeByColumn("row_pick");
    try take_by_plan.select(&.{ "sales", "units" });
    const take_by_explain = try take_by_plan.explain(gpa);
    defer gpa.free(take_by_explain);
    try std.testing.expect(std.mem.indexOf(u8, take_by_explain, "take_rows_by_column(row_pick)") != null);
    var taken_by = try take_by_plan.collect();
    defer taken_by.deinit();
    const taken_by_sales = try (try taken_by.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_by_sales);
    const taken_by_sales_validity = try (try taken_by.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(taken_by_sales_validity);
    const taken_by_units = try (try taken_by.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(taken_by_units);
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 0.0, 7.0, 3.0 }, taken_by_sales);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, true }, taken_by_sales_validity);
    try std.testing.expectEqualSlices(i64, &.{ 3, 0, 4, 2 }, taken_by_units);

    var row_pick_wrap = try DeviceColumn.fromSlice(usize, gpa, &.{ 5, 0, 3, 6 }, .cpu);
    defer row_pick_wrap.deinit();
    var take_by_wrap_source = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = (try table.column("sales")).* },
        .{ .name = "row_pick", .data = row_pick_wrap },
    });
    defer take_by_wrap_source.deinit();
    var take_by_wrap_plan = try DeviceLazyFrame.init(gpa, take_by_wrap_source);
    defer take_by_wrap_plan.deinit();
    try take_by_wrap_plan.takeByColumnMode("row_pick", .wrap);
    try take_by_wrap_plan.select(&.{"sales"});
    const take_by_wrap_explain = try take_by_wrap_plan.explain(gpa);
    defer gpa.free(take_by_wrap_explain);
    try std.testing.expect(std.mem.indexOf(u8, take_by_wrap_explain, "take_rows_by_column_mode(row_pick, mode:wrap)") != null);
    var taken_by_wrap = try take_by_wrap_plan.collect();
    defer taken_by_wrap.deinit();
    const taken_by_wrap_sales = try (try taken_by_wrap.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_by_wrap_sales);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 2.0, 7.0, 5.0 }, taken_by_wrap_sales);

    var drop_pick = try DeviceColumn.fromSliceWithValidity(isize, gpa, &.{ 1, -1, 0, 2 }, &.{ true, false, true, true }, .cpu);
    defer drop_pick.deinit();
    var drop_by_source = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = (try table.column("sales")).* },
        .{ .name = "units", .data = (try table.column("units")).* },
        .{ .name = "drop_pick", .data = drop_pick },
    });
    defer drop_by_source.deinit();
    var drop_by_plan = try DeviceLazyFrame.init(gpa, drop_by_source);
    defer drop_by_plan.deinit();
    try drop_by_plan.dropRowsByColumn("drop_pick");
    try drop_by_plan.select(&.{ "sales", "units" });
    const drop_by_explain = try drop_by_plan.explain(gpa);
    defer gpa.free(drop_by_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_by_explain, "drop_rows_by_column(drop_pick)") != null);
    var dropped_by = try drop_by_plan.collect();
    defer dropped_by.deinit();
    const dropped_by_sales = try (try dropped_by.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_by_sales);
    try std.testing.expectEqualSlices(f64, &.{7.0}, dropped_by_sales);

    var drop_pick_wrap = try DeviceColumn.fromSlice(usize, gpa, &.{ 5, 5, 5, 5 }, .cpu);
    defer drop_pick_wrap.deinit();
    var drop_by_wrap_source = try DeviceDataFrame.init(gpa, &.{
        .{ .name = "sales", .data = (try table.column("sales")).* },
        .{ .name = "drop_pick", .data = drop_pick_wrap },
    });
    defer drop_by_wrap_source.deinit();
    var drop_by_wrap_plan = try DeviceLazyFrame.init(gpa, drop_by_wrap_source);
    defer drop_by_wrap_plan.deinit();
    try drop_by_wrap_plan.dropRowsByColumnMode("drop_pick", .wrap);
    try drop_by_wrap_plan.select(&.{"sales"});
    const drop_by_wrap_explain = try drop_by_wrap_plan.explain(gpa);
    defer gpa.free(drop_by_wrap_explain);
    try std.testing.expect(std.mem.indexOf(u8, drop_by_wrap_explain, "drop_rows_by_column_mode(drop_pick, mode:wrap)") != null);
    var dropped_by_wrap = try drop_by_wrap_plan.collect();
    defer dropped_by_wrap.deinit();
    const dropped_by_wrap_sales = try (try dropped_by_wrap.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(dropped_by_wrap_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0, 7.0 }, dropped_by_wrap_sales);

    var take_mode_plan = try DeviceLazyFrame.init(gpa, table);
    defer take_mode_plan.deinit();
    try take_mode_plan.takeMode(&.{ 5, 0 }, .wrap);
    try take_mode_plan.takeSignedMode(&.{ -9, 9 }, .clip);
    try take_mode_plan.takeSigned(&.{ -1, 0 });
    try take_mode_plan.select(&.{ "sales", "units" });
    const take_mode_explain = try take_mode_plan.explain(gpa);
    defer gpa.free(take_mode_explain);
    try std.testing.expect(std.mem.indexOf(u8, take_mode_explain, "take_rows_mode([5,0], mode:wrap)") != null);
    try std.testing.expect(std.mem.indexOf(u8, take_mode_explain, "take_rows_signed_mode([-9,9], mode:clip)") != null);
    try std.testing.expect(std.mem.indexOf(u8, take_mode_explain, "take_rows_signed([-1,0])") != null);
    var taken_mode = try take_mode_plan.collect();
    defer taken_mode.deinit();
    try std.testing.expectEqual(@as(usize, 2), taken_mode.height());
    const taken_mode_sales = try (try taken_mode.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(taken_mode_sales);
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 3.0 }, taken_mode_sales);
}

test "device lazy frame reverses rows" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var reverse_plan = try DeviceLazyFrame.init(gpa, table);
    defer reverse_plan.deinit();
    try reverse_plan.reverseRows();
    try reverse_plan.select(&.{ "sales", "units", "active" });

    const reverse_explain = try reverse_plan.explain(gpa);
    defer gpa.free(reverse_explain);
    try std.testing.expect(std.mem.indexOf(u8, reverse_explain, "reverse_rows") != null);

    var reversed = try reverse_plan.collect();
    defer reversed.deinit();
    try std.testing.expectEqual(@as(usize, 4), reversed.height());
    try std.testing.expectEqual(@as(usize, 3), reversed.width());
    const sales = try (try reversed.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(sales);
    const units = try (try reversed.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(units);
    const active = try (try reversed.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(active);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 5.0, 3.0, 2.0 }, sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3, 2, 1 }, units);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, active);

    var roll_plan = try DeviceLazyFrame.init(gpa, table);
    defer roll_plan.deinit();
    try roll_plan.rollRows(1);
    try roll_plan.select(&.{ "sales", "units", "active" });
    const roll_explain = try roll_plan.explain(gpa);
    defer gpa.free(roll_explain);
    try std.testing.expect(std.mem.indexOf(u8, roll_explain, "roll_rows(1)") != null);
    var rolled = try roll_plan.collect();
    defer rolled.deinit();
    const rolled_sales = try (try rolled.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(rolled_sales);
    const rolled_units = try (try rolled.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(rolled_units);
    const rolled_active = try (try rolled.column("active")).bool.toOwnedSlice(gpa);
    defer gpa.free(rolled_active);
    try std.testing.expectEqualSlices(f64, &.{ 7.0, 2.0, 3.0, 5.0 }, rolled_sales);
    try std.testing.expectEqualSlices(i64, &.{ 4, 1, 2, 3 }, rolled_units);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, rolled_active);

    var shift_plan = try DeviceLazyFrame.init(gpa, table);
    defer shift_plan.deinit();
    try shift_plan.shiftRows(1);
    try shift_plan.select(&.{ "sales", "units", "active" });
    const shift_explain = try shift_plan.explain(gpa);
    defer gpa.free(shift_explain);
    try std.testing.expect(std.mem.indexOf(u8, shift_explain, "shift_rows(1)") != null);
    var shifted = try shift_plan.collect();
    defer shifted.deinit();
    const shifted_sales = try (try shifted.column("sales")).f64.toOwnedSlice(gpa);
    defer gpa.free(shifted_sales);
    const shifted_sales_validity = try (try shifted.column("sales")).f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(shifted_sales_validity);
    const shifted_units = try (try shifted.column("units")).i64.toOwnedSlice(gpa);
    defer gpa.free(shifted_units);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 2.0, 3.0, 5.0 }, shifted_sales);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, shifted_sales_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3 }, shifted_units);
}

test "device lazy frame collects rank operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer rank_plan.deinit();
    try rank_plan.rankProfileBy("sales", "sales_rank", .{ .descending = true });
    try rank_plan.select(&.{ "sales", "sales_rank_ordinal_rank", "sales_rank_percent_rank", "sales_rank_cume_dist" });
    const rank_explain = try rank_plan.explain(gpa);
    defer gpa.free(rank_explain);
    try std.testing.expect(std.mem.indexOf(u8, rank_explain, "rank_profile_by(sales") != null);
    var ranked = try rank_plan.collect();
    defer ranked.deinit();
    try std.testing.expectEqual(@as(usize, 4), ranked.height());
    try std.testing.expectEqual(@as(usize, 4), ranked.width());
    const ranked_ordinal = try (try ranked.column("sales_rank_ordinal_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(ranked_ordinal);
    const ranked_percent = try (try ranked.column("sales_rank_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(ranked_percent);
    const ranked_cume = try (try ranked.column("sales_rank_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(ranked_cume);
    try std.testing.expectEqualSlices(i64, &.{ 4, 3, 2, 1 }, ranked_ordinal);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ranked_percent[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), ranked_percent[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), ranked_percent[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), ranked_percent[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ranked_cume[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), ranked_cume[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), ranked_cume[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), ranked_cume[3], 1e-12);
}

test "device lazy frame collects rolling rank operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var rolling_rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer rolling_rank_plan.deinit();
    try rolling_rank_plan.rollingRankProfile("sales", "sales_roll", .{ .window = 2, .min_periods = 2, .descending = true });
    try rolling_rank_plan.select(&.{ "sales", "sales_roll_rolling_rank_count", "sales_roll_rolling_rank", "sales_roll_rolling_percent_rank", "sales_roll_rolling_cume_dist" });
    const rolling_rank_explain = try rolling_rank_plan.explain(gpa);
    defer gpa.free(rolling_rank_explain);
    try std.testing.expect(std.mem.indexOf(u8, rolling_rank_explain, "rolling_rank_profile(sales") != null);
    var rolling_ranked = try rolling_rank_plan.collect();
    defer rolling_ranked.deinit();
    try std.testing.expectEqual(@as(usize, 4), rolling_ranked.height());
    try std.testing.expectEqual(@as(usize, 5), rolling_ranked.width());
    const lazy_rolling_rank_count = try (try rolling_ranked.column("sales_roll_rolling_rank_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_rank_count);
    const lazy_rolling_rank = try (try rolling_ranked.column("sales_roll_rolling_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_rank);
    const lazy_rolling_percent_rank = try (try rolling_ranked.column("sales_roll_rolling_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_percent_rank);
    const lazy_rolling_cume_dist = try (try rolling_ranked.column("sales_roll_rolling_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_cume_dist);
    const lazy_rolling_rank_validity = try (try rolling_ranked.column("sales_roll_rolling_rank")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_rolling_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2, 2 }, lazy_rolling_rank_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_rolling_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_rolling_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_rolling_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_rolling_cume_dist[3], 1e-12);
}

test "device lazy frame collects expanding rank operations" {
    const gpa = std.testing.allocator;
    var table = try lazyCollectTable(gpa);
    defer table.deinit();
    var expanding_rank_plan = try DeviceLazyFrame.init(gpa, table);
    defer expanding_rank_plan.deinit();
    try expanding_rank_plan.expandingRankProfile("sales", "sales_expand", .{ .min_periods = 2, .descending = true });
    try expanding_rank_plan.select(&.{ "sales", "sales_expand_expanding_rank_count", "sales_expand_expanding_rank", "sales_expand_expanding_percent_rank", "sales_expand_expanding_cume_dist" });
    const expanding_rank_explain = try expanding_rank_plan.explain(gpa);
    defer gpa.free(expanding_rank_explain);
    try std.testing.expect(std.mem.indexOf(u8, expanding_rank_explain, "expanding_rank_profile(sales") != null);
    var expanding_ranked = try expanding_rank_plan.collect();
    defer expanding_ranked.deinit();
    try std.testing.expectEqual(@as(usize, 4), expanding_ranked.height());
    try std.testing.expectEqual(@as(usize, 5), expanding_ranked.width());
    const lazy_expanding_rank_count = try (try expanding_ranked.column("sales_expand_expanding_rank_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_rank_count);
    const lazy_expanding_rank = try (try expanding_ranked.column("sales_expand_expanding_rank")).i64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_rank);
    const lazy_expanding_percent_rank = try (try expanding_ranked.column("sales_expand_expanding_percent_rank")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_percent_rank);
    const lazy_expanding_cume_dist = try (try expanding_ranked.column("sales_expand_expanding_cume_dist")).f64.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_cume_dist);
    const lazy_expanding_rank_validity = try (try expanding_ranked.column("sales_expand_expanding_rank")).i64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(lazy_expanding_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4 }, lazy_expanding_rank_count);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, lazy_expanding_rank_validity);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 1 }, lazy_expanding_rank);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_percent_rank[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_percent_rank[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lazy_expanding_percent_rank[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lazy_expanding_cume_dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), lazy_expanding_cume_dist[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lazy_expanding_cume_dist[3], 1e-12);
}
