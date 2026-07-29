//! Join key validation helpers.

const keys_mod = @import("dataframe_keys.zig");

pub fn validateSingleJoinKeys(left: anytype, right: anytype, left_key_name: []const u8, right_key_name: []const u8) keys_mod.KeyMatchError!void {
    if (!left.device.sameDevice(right.device)) return error.InvalidDevice;
    const left_key = try left.column(left_key_name);
    const right_key = try right.column(right_key_name);
    if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
}

pub fn validateMultiJoinKeys(
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!void {
    if (!left.device.sameDevice(right.device)) return error.InvalidDevice;
    if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
    for (left_key_names, right_key_names) |left_name, right_name| {
        const left_key = try left.column(left_name);
        const right_key = try right.column(right_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
    }
}
