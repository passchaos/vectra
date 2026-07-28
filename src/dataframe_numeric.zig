pub fn castToF64(comptime T: type, value: T) f64 {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => @floatCast(value),
        .int, .comptime_int => @floatFromInt(value),
        else => @compileError("numeric dataframe profile requires numeric values"),
    };
}
