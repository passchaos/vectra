//! Public facade for lazy dataframe plan formatting and scan pushdown helpers.

const format_mod = @import("dataframe_lazy_format.zig");
const pushdown_mod = @import("dataframe_lazy_pushdown.zig");

pub const LazyScanPushdown = pushdown_mod.LazyScanPushdown;
pub const formatLazyOp = format_mod.formatLazyOp;
pub const formatLazyScanPushdown = format_mod.formatLazyScanPushdown;
pub const planLazyScanPushdown = pushdown_mod.planLazyScanPushdown;
