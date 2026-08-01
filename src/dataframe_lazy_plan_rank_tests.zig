const std = @import("std");
const vectra = @import("vectra");

const DeviceColumn = vectra.DeviceColumn;
const DeviceDataFrame = vectra.DeviceDataFrame;
const DeviceLazyFrame = vectra.DeviceLazyFrame;
const helpers = @import("dataframe_lazy_test_helpers.zig");
const lazyCollectTable = helpers.lazyCollectTable;
const lazyQualityTable = helpers.lazyQualityTable;

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
    try plan.withColumnLogicalXorScalar("active_not", "active", true);
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
    try plan.fillNullColumn("quality", f64, -1.0);

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "fill_null_column(quality=scalar:f64)") != null);

    var filled = try plan.collect();
    defer filled.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try filled.column("quality")).nullCount());
    const quality = try (try filled.column("quality")).f64.toOwnedSlice(gpa);
    defer gpa.free(quality);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, -1.0, 3.0, 4.0 }, quality);

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

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "with_column_literal(fallback_quality=scalar:f64)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "coalesce_columns(quality,fallback_quality->quality_coalesced)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try result.column("quality_coalesced")).nullCount());
    const values = try (try result.column("quality_coalesced")).f64.toOwnedSlice(gpa);
    defer gpa.free(values);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 9.0, 3.0, 4.0 }, values);

    var mismatch_plan = try DeviceLazyFrame.init(gpa, table);
    defer mismatch_plan.deinit();
    try mismatch_plan.withColumnLiteral("fallback_i64", i64, 9);
    try mismatch_plan.coalesceColumns("quality", "fallback_i64", "bad");
    try std.testing.expectError(error.TypeMismatch, mismatch_plan.collect());
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
    try plan.withRowNullRatio(&.{ "quality", "flag" }, "row_null_ratio");
    try plan.withRowValidRatio(&.{ "quality", "flag" }, "row_valid_ratio");
    try plan.withRowTrueCount(&.{"flag"}, "row_true_count");
    try plan.withRowFalseCount(&.{"flag"}, "row_false_count");
    try plan.withRowAnyTrue(&.{"flag"}, "row_any_true");
    try plan.withRowAllTrue(&.{"flag"}, "row_all_true");
    try plan.withRowAnyFalse(&.{"flag"}, "row_any_false");
    try plan.withRowAllFalse(&.{"flag"}, "row_all_false");
    try plan.withRowTrueRatio(&.{"flag"}, "row_true_ratio");
    try plan.withRowFalseRatio(&.{"flag"}, "row_false_ratio");
    try plan.select(&.{ "row_nulls", "row_valids_all", "row_null_ratio", "row_valid_ratio", "row_true_count", "row_false_count", "row_any_true", "row_all_true", "row_any_false", "row_all_false", "row_true_ratio", "row_false_ratio" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_null_count([quality,flag]->row_nulls)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_valid_count([]->row_valids_all)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_null_ratio([quality,flag]->row_null_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_valid_ratio([quality,flag]->row_valid_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_true_count([flag]->row_true_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_false_count([flag]->row_false_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_true([flag]->row_any_true)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_all_true([flag]->row_all_true)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_any_false([flag]->row_any_false)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_all_false([flag]->row_all_false)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_true_ratio([flag]->row_true_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_false_ratio([flag]->row_false_ratio)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 12), result.width());
    const row_nulls = try (try result.column("row_nulls")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_nulls);
    const row_valids_all = try (try result.column("row_valids_all")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_valids_all);
    const row_null_ratio = try (try result.column("row_null_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_null_ratio);
    const row_valid_ratio = try (try result.column("row_valid_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_valid_ratio);
    const row_true_count = try (try result.column("row_true_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_true_count);
    const row_false_count = try (try result.column("row_false_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_false_count);
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
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.5, 0.5 }, row_null_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.5, 0.5, 0.5 }, row_valid_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0 }, row_true_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, row_false_count);
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

    var invalid_bool_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_bool_plan.deinit();
    try invalid_bool_plan.withRowTrueCount(&.{"sales"}, "bad_bool_count");
    try std.testing.expectError(error.TypeMismatch, invalid_bool_plan.collect());

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
    try plan.withRowWeightedMean(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mean");
    try plan.withRowWeightedQuantile(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_quantile", 0.9);
    try plan.withRowWeightedMedian(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_median");
    try plan.withRowWeightedIqr(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_iqr");
    try plan.withRowWeightedMad(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_mad");
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
    try plan.withRowWeightedVariance(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_variance", 0.0);
    try plan.withRowWeightedStddev(&.{ "a", "b" }, &.{ "wa", "wb" }, "row_weighted_stddev", 0.0);
    try plan.withRowWeightedCovariance(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_covariance", 0.0);
    try plan.withRowWeightedCorrelation(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_correlation", 0.0);
    try plan.withRowWeightedBeta(&.{ "a", "b" }, &.{ "wa", "wb" }, &.{ "wa", "wb" }, "row_weighted_beta", 0.0);
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
    try plan.withRowSum(&.{ "a", "b" }, "row_sum");
    try plan.withRowMean(&.{ "a", "b" }, "row_mean");
    try plan.withRowGeometricMean(&.{ "a", "b" }, "row_geo");
    try plan.withRowHarmonicMean(&.{ "a", "b" }, "row_harm");
    try plan.withRowSkewness(&.{ "a", "b" }, "row_skew");
    try plan.withRowKurtosis(&.{ "a", "b" }, "row_kurt");
    try plan.withRowProd(&.{ "a", "b" }, "row_prod");
    try plan.withRowMin(&.{ "a", "b" }, "row_min");
    try plan.withRowMax(&.{ "a", "b" }, "row_max");
    try plan.withRowPtp(&.{ "a", "b" }, "row_ptp");
    try plan.withRowMidrange(&.{ "a", "b" }, "row_midrange");
    try plan.withRowRangeCoeff(&.{ "a", "b" }, "row_range_coeff");
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
    try plan.select(&.{ "row_argmin", "row_argmax", "row_quantile", "row_quantile_range", "row_trimmed_mean", "row_winsorized_mean", "row_median", "row_iqr", "row_idr", "row_midhinge", "row_trimean", "row_bowley", "row_qcd", "row_kelley", "row_mad", "row_mode", "row_entropy", "row_gini", "row_perplexity", "row_inverse_simpson", "row_concentration", "row_evenness", "row_mode_count", "row_mode_ratio", "row_mode_margin", "row_mode_margin_ratio", "row_pair_count", "row_weighted_mean", "row_weighted_quantile", "row_weighted_median", "row_weighted_iqr", "row_weighted_mad", "row_weighted_mode", "row_weighted_mode_weight", "row_weighted_mode_ratio", "row_weighted_mode_margin", "row_weighted_mode_margin_ratio", "row_weighted_entropy", "row_weighted_gini", "row_weighted_perplexity", "row_weighted_inverse", "row_weighted_concentration", "row_weighted_evenness", "row_weighted_variance", "row_weighted_stddev", "row_weighted_covariance", "row_weighted_correlation", "row_weighted_beta", "row_dot", "row_cosine", "row_sqdist", "row_euclidean", "row_manhattan", "row_chebyshev", "row_canberra", "row_bray", "row_mean_error", "row_mae", "row_mse", "row_rmse", "row_mape", "row_smape", "row_covariance", "row_correlation", "row_beta", "row_distinct", "row_unique", "row_sum", "row_mean", "row_geo", "row_harm", "row_skew", "row_kurt", "row_prod", "row_min", "row_max", "row_ptp", "row_midrange", "row_range_coeff", "row_mean_abs", "row_hhi", "row_magnitude_normalized_hhi", "row_magnitude_sparsity", "row_magnitude_inverse", "row_magnitude_simpson_evenness", "row_magnitude_dominance", "row_magnitude_margin", "row_magnitude_entropy", "row_magnitude_perplexity", "row_magnitude_evenness", "row_mean_abs_dev", "row_gini_mean_diff", "row_gini_coeff", "row_mad_ratio", "row_rms", "row_l1", "row_l2", "row_variance", "row_magnitude_variance", "row_stddev", "row_magnitude_stddev", "row_sem", "row_magnitude_sem", "row_cv", "row_magnitude_cv", "row_magnitude_fano", "row_fano" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_argmin([a,b]->row_argmin)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_argmax([a,b]->row_argmax)") != null);
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
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mean(values=[a,b], weights=[wa,wb]->row_weighted_mean)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_quantile(values=[a,b], weights=[wa,wb]->row_weighted_quantile, q=0.9)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_median(values=[a,b], weights=[wa,wb]->row_weighted_median)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_iqr(values=[a,b], weights=[wa,wb]->row_weighted_iqr)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_mad(values=[a,b], weights=[wa,wb]->row_weighted_mad)") != null);
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
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_variance(values=[a,b], weights=[wa,wb]->row_weighted_variance, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_stddev(values=[a,b], weights=[wa,wb]->row_weighted_stddev, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_covariance(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_covariance, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_correlation(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_correlation, correction=0)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_weighted_beta(lhs=[a,b], rhs=[wa,wb], weights=[wa,wb]->row_weighted_beta, correction=0)") != null);
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
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_sum([a,b]->row_sum)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_mean([a,b]->row_mean)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_geometric_mean([a,b]->row_geo)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_harmonic_mean([a,b]->row_harm)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_skewness([a,b]->row_skew)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_kurtosis([a,b]->row_kurt)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_prod([a,b]->row_prod)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_min([a,b]->row_min)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_max([a,b]->row_max)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_ptp([a,b]->row_ptp)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_range_coeff([a,b]->row_range_coeff)") != null);
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
    try std.testing.expectEqual(@as(usize, 107), result.width());
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
    const row_weighted_mean_column = try result.column("row_weighted_mean");
    try std.testing.expect(row_weighted_mean_column.f64.nullable());
    const row_weighted_mean = try row_weighted_mean_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean);
    const row_weighted_mean_validity = try row_weighted_mean_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_weighted_mean_validity);
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
    const row_geo_column = try result.column("row_geo");
    try std.testing.expect(row_geo_column.f64.nullable());
    const row_geo = try row_geo_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_geo);
    const row_geo_validity = try row_geo_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_geo_validity);
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
    const row_kurt_column = try result.column("row_kurt");
    try std.testing.expect(row_kurt_column.f64.nullable());
    const row_kurt = try row_kurt_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_kurt);
    const row_kurt_validity = try row_kurt_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_kurt_validity);
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
    const row_range_coeff_column = try result.column("row_range_coeff");
    try std.testing.expect(row_range_coeff_column.f64.nullable());
    const row_range_coeff = try row_range_coeff_column.f64.toOwnedSlice(gpa);
    defer gpa.free(row_range_coeff);
    const row_range_coeff_validity = try row_range_coeff_column.f64.validity.?.toOwnedSlice(gpa);
    defer gpa.free(row_range_coeff_validity);
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
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_weighted_mean[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_weighted_mean[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_weighted_mean[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 56.0 / 5.0), row_weighted_mean[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mean_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 40.0 }, row_weighted_quantile);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_quantile_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_weighted_median);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_median_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, row_weighted_iqr);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_iqr_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 0.0 }, row_weighted_mad);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_weighted_mad_validity);
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
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 44.0 }, row_sum);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_sum_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 22.0 }, row_mean);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_mean_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), row_geo[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), row_geo[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_geo[2], 1e-12);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f64, 160.0)), row_geo[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_geo_validity);
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
    try std.testing.expect(std.math.isNan(row_kurt[0]));
    try std.testing.expect(std.math.isNan(row_kurt[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_kurt[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -2.0), row_kurt[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_kurt_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 160.0 }, row_prod);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_prod_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 4.0 }, row_min);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_min_validity);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 20.0, 0.0, 40.0 }, row_max);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_max_validity);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.0, 36.0 }, row_ptp);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_ptp_validity);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_range_coeff[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), row_range_coeff[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), row_range_coeff[2]);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0 / 11.0), row_range_coeff[3], 1e-12);
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, true }, row_range_coeff_validity);
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

    var invalid_correction_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_correction_plan.deinit();
    try invalid_correction_plan.withRowVariance(&.{ "a", "b" }, "bad_row_variance", -1.0);
    try std.testing.expectError(error.InvalidShape, invalid_correction_plan.collect());
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
    try plan.select(&.{ "metric_is_zero", "metric_is_non_zero", "id_is_zero", "flag_is_non_zero", "row_zero_count", "row_non_zero_count", "row_zero_ratio", "row_non_zero_ratio" });

    const explained = try plan.explain(gpa);
    defer gpa.free(explained);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_zero_column(metric->metric_is_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "is_non_zero_column(metric->metric_is_non_zero)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_zero_count([metric,id,flag]->row_zero_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_non_zero_count([metric,id,flag]->row_non_zero_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_zero_ratio([metric,id,flag]->row_zero_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, explained, "row_non_zero_ratio([metric,id,flag]->row_non_zero_ratio)") != null);

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 8), result.width());
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
    try std.testing.expectEqualSlices(bool, &.{ true, true, false, false, false, false }, metric_is_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true, true, false }, metric_is_non_zero);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false, false, true }, id_is_zero);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, true, false }, flag_is_non_zero);
    try std.testing.expectEqualSlices(i64, &.{ 3, 1, 2, 0, 0, 2 }, row_zero_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 1, 3, 3, 0 }, row_non_zero_count);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0, 1.0 }, row_zero_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 0.0 }, row_non_zero_ratio);

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
    try plan.select(&.{ "metric_is_positive", "metric_is_negative", "metric_signbit", "id_signbit", "metric_is_positive_zero", "metric_is_negative_zero", "id_is_positive", "unsigned_is_negative", "flag_is_positive", "row_positive_zero_count", "row_negative_zero_count", "row_positive_zero_ratio", "row_negative_zero_ratio", "row_positive_count", "row_signbit_count", "row_negative_count", "row_positive_ratio", "row_signbit_ratio", "row_negative_ratio" });

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

    var result = try plan.collect();
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 19), result.width());
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

    var invalid_plan = try DeviceLazyFrame.init(gpa, table);
    defer invalid_plan.deinit();
    try invalid_plan.isPositiveColumn("missing", "missing_is_positive");
    try std.testing.expectError(error.ColumnNotFound, invalid_plan.collect());

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
    try fill_nan_plan.fillNaNColumn("metric", f64, -1.0);
    const fill_nan_explain = try fill_nan_plan.explain(gpa);
    defer gpa.free(fill_nan_explain);
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
    try row_special_plan.select(&.{ "row_nan_count", "row_inf_count", "row_finite_count", "row_non_finite_count", "row_nan_ratio", "row_inf_ratio", "row_finite_ratio", "row_non_finite_ratio" });
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
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0 }, row_nan_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0 }, row_inf_count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 1, 1 }, row_finite_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 1, 0 }, row_non_finite_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.0, 0.0 }, row_nan_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.5, 0.0 }, row_inf_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 0.5, 0.5, 1.0 }, row_finite_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.5, 0.5, 0.0 }, row_non_finite_ratio);

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
    try row_normal_plan.select(&.{ "row_normal_count", "row_normal_ratio" });
    const row_normal_explain = try row_normal_plan.explain(gpa);
    defer gpa.free(row_normal_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_normal_explain, "row_normal_count([metric,id]->row_normal_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_normal_explain, "row_normal_ratio([metric,id]->row_normal_ratio)") != null);
    var row_normal = try row_normal_plan.collect();
    defer row_normal.deinit();
    const row_normal_count = try (try row_normal.column("row_normal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_normal_count);
    const row_normal_ratio = try (try row_normal.column("row_normal_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_normal_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0, 0 }, row_normal_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.0, 0.0, 0.0, 0.0 }, row_normal_ratio);

    var row_subnormal_plan = try DeviceLazyFrame.init(gpa, table);
    defer row_subnormal_plan.deinit();
    try row_subnormal_plan.withRowSubnormalCount(&.{ "metric", "id" }, "row_subnormal_count");
    try row_subnormal_plan.withRowSubnormalRatio(&.{ "metric", "id" }, "row_subnormal_ratio");
    try row_subnormal_plan.select(&.{ "row_subnormal_count", "row_subnormal_ratio" });
    const row_subnormal_explain = try row_subnormal_plan.explain(gpa);
    defer gpa.free(row_subnormal_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_subnormal_explain, "row_subnormal_count([metric,id]->row_subnormal_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_subnormal_explain, "row_subnormal_ratio([metric,id]->row_subnormal_ratio)") != null);
    var row_subnormal = try row_subnormal_plan.collect();
    defer row_subnormal.deinit();
    const row_subnormal_count = try (try row_subnormal.column("row_subnormal_count")).i64.toOwnedSlice(gpa);
    defer gpa.free(row_subnormal_count);
    const row_subnormal_ratio = try (try row_subnormal.column("row_subnormal_ratio")).f64.toOwnedSlice(gpa);
    defer gpa.free(row_subnormal_ratio);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0 }, row_subnormal_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 0.5, 0.0, 0.0 }, row_subnormal_ratio);

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
    try row_signed_plan.select(&.{ "row_positive_inf_count", "row_negative_inf_count", "row_positive_inf_ratio", "row_negative_inf_ratio" });
    const row_signed_explain = try row_signed_plan.explain(gpa);
    defer gpa.free(row_signed_explain);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_positive_inf_count([]->row_positive_inf_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_negative_inf_count([metric]->row_negative_inf_count)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_positive_inf_ratio([metric]->row_positive_inf_ratio)") != null);
    try std.testing.expect(std.mem.indexOf(u8, row_signed_explain, "row_negative_inf_ratio([metric]->row_negative_inf_ratio)") != null);
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
    try std.testing.expectEqualSlices(i64, &.{ 0, 1, 0, 0, 0 }, row_positive_inf_count);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 1, 0, 0 }, row_negative_inf_count);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 1.0, 0.0, 0.0, 0.0 }, row_positive_inf_ratio);
    try std.testing.expectEqualSlices(f64, &.{ 0.0, 0.0, 1.0, 0.0, 0.0 }, row_negative_inf_ratio);

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
