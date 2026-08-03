//! Eager DeviceDataFrame profile method wrappers.
//!
//! These methods dispatch to the focused profile implementation modules. Keeping
//! them outside `dataframe.zig` keeps the public facade small while preserving
//! `frame.rollingProfile(...)` style method syntax through aliases.

const validity_methods_mod = @import("profile/validity_methods.zig");
const name_methods_mod = @import("profile/name_methods.zig");
const pair_methods_mod = @import("profile/pair_methods.zig");

pub const rollingProfile = name_methods_mod.rollingProfile;
pub const rollingMomentProfile = name_methods_mod.rollingMomentProfile;
pub const rollingRangeProfile = name_methods_mod.rollingRangeProfile;
pub const rollingNormalizeProfile = name_methods_mod.rollingNormalizeProfile;
pub const expandingNormalizeProfile = name_methods_mod.expandingNormalizeProfile;
pub const rollingQuantileProfile = name_methods_mod.rollingQuantileProfile;
pub const expandingQuantileProfile = name_methods_mod.expandingQuantileProfile;
pub const rollingBoolProfile = name_methods_mod.rollingBoolProfile;
pub const rollingDrawdownProfile = name_methods_mod.rollingDrawdownProfile;
pub const rollingRobustProfile = name_methods_mod.rollingRobustProfile;
pub const rollingRankProfile = name_methods_mod.rollingRankProfile;
pub const lagProfile = name_methods_mod.lagProfile;
pub const leadProfile = name_methods_mod.leadProfile;
pub const clipProfile = name_methods_mod.clipProfile;
pub const rollingClipProfile = name_methods_mod.rollingClipProfile;
pub const expandingClipProfile = name_methods_mod.expandingClipProfile;
pub const thresholdProfile = name_methods_mod.thresholdProfile;
pub const rollingThresholdProfile = name_methods_mod.rollingThresholdProfile;
pub const expandingThresholdProfile = name_methods_mod.expandingThresholdProfile;
pub const expandingProfile = name_methods_mod.expandingProfile;
pub const expandingBoolProfile = name_methods_mod.expandingBoolProfile;
pub const expandingRankProfile = name_methods_mod.expandingRankProfile;
pub const expandingRobustProfile = name_methods_mod.expandingRobustProfile;
pub const expandingMomentProfile = name_methods_mod.expandingMomentProfile;
pub const standardizeProfile = name_methods_mod.standardizeProfile;
pub const robustProfile = name_methods_mod.robustProfile;
pub const drawdownProfile = name_methods_mod.drawdownProfile;
pub const extremaProfile = name_methods_mod.extremaProfile;
pub const trendProfile = name_methods_mod.trendProfile;
pub const changePointProfile = name_methods_mod.changePointProfile;
pub const rollingChangePointProfile = name_methods_mod.rollingChangePointProfile;
pub const expandingChangePointProfile = name_methods_mod.expandingChangePointProfile;
pub const rollingTrendProfile = name_methods_mod.rollingTrendProfile;
pub const expandingTrendProfile = name_methods_mod.expandingTrendProfile;
pub const signProfile = name_methods_mod.signProfile;
pub const rollingSignProfile = name_methods_mod.rollingSignProfile;
pub const expandingSignProfile = name_methods_mod.expandingSignProfile;
pub const crossoverProfile = pair_methods_mod.crossoverProfile;
pub const rollingCrossoverProfile = pair_methods_mod.rollingCrossoverProfile;
pub const expandingCrossoverProfile = pair_methods_mod.expandingCrossoverProfile;
pub const bucketProfile = pair_methods_mod.bucketProfile;
pub const emaProfile = pair_methods_mod.emaProfile;
pub const linearFitProfile = pair_methods_mod.linearFitProfile;
pub const errorProfile = pair_methods_mod.errorProfile;
pub const rollingErrorProfile = pair_methods_mod.rollingErrorProfile;
pub const expandingErrorProfile = pair_methods_mod.expandingErrorProfile;
pub const classificationProfile = pair_methods_mod.classificationProfile;
pub const rollingClassificationProfile = pair_methods_mod.rollingClassificationProfile;
pub const expandingClassificationProfile = pair_methods_mod.expandingClassificationProfile;
pub const boolTransitionProfile = pair_methods_mod.boolTransitionProfile;
pub const rollingBoolTransitionProfile = pair_methods_mod.rollingBoolTransitionProfile;
pub const expandingBoolTransitionProfile = pair_methods_mod.expandingBoolTransitionProfile;
pub const rollingCorrelationProfile = pair_methods_mod.rollingCorrelationProfile;
pub const expandingCorrelationProfile = pair_methods_mod.expandingCorrelationProfile;
pub const expandingLinearFitProfile = pair_methods_mod.expandingLinearFitProfile;
pub const rollingLinearFitProfile = pair_methods_mod.rollingLinearFitProfile;

pub const validityProfile = validity_methods_mod.validityProfile;
pub const rollingValidityProfile = validity_methods_mod.rollingValidityProfile;
pub const expandingValidityProfile = validity_methods_mod.expandingValidityProfile;
