//! Arrow extension metadata used for Vectra-only fixed-width dataframe dtypes.
//!
//! Arrow has no standard logical BFloat16 or complex scalar type.  Boltha does
//! provide the Arrow extension metadata keys and fixed-size-binary storage, so
//! Vectra records these dtypes as little-endian fixed-size binary payloads with
//! explicit extension names.  Readers that do not understand the extensions
//! still see valid Arrow fixed-size-binary arrays, while Vectra can recover the
//! original dtype losslessly.

const std = @import("std");
const array_mod = @import("../../array.zig");
const boltha = @import("boltha");

pub const ExtensionSpec = struct {
    dtype: array_mod.DType,
    name: []const u8,
    metadata: []const u8,
    byte_width: i32,
};

pub const bf16: ExtensionSpec = .{
    .dtype = .bf16,
    .name = "vectra.bfloat16",
    .metadata = "storage=fixed_size_binary:2;layout=le:u16",
    .byte_width = 2,
};

pub const complex64: ExtensionSpec = .{
    .dtype = .c64,
    .name = "vectra.complex64",
    .metadata = "storage=fixed_size_binary:8;layout=le:f32,re;le:f32,im",
    .byte_width = 8,
};

pub const complex128: ExtensionSpec = .{
    .dtype = .c128,
    .name = "vectra.complex128",
    .metadata = "storage=fixed_size_binary:16;layout=le:f64,re;le:f64,im",
    .byte_width = 16,
};

pub fn forDType(dtype: array_mod.DType) ?ExtensionSpec {
    return switch (dtype) {
        .bf16 => bf16,
        .c64 => complex64,
        .c128 => complex128,
        else => null,
    };
}

pub fn dtypeFromField(field: boltha.arrow.Field) ?array_mod.DType {
    const name = field.extensionTypeName() orelse return null;
    if (field.data_type != .fixed_size_binary) return null;
    const byte_width = field.data_type.fixed_size_binary;
    inline for (.{ bf16, complex64, complex128 }) |spec| {
        if (byte_width == spec.byte_width and std.mem.eql(u8, name, spec.name)) return spec.dtype;
    }
    return null;
}
