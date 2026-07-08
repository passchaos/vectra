const std = @import("std");
const array_mod = @import("array.zig");

pub const DataError = error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    InvalidCsv,
    EmptyDataFrame,
    UnsupportedType,
} || std.mem.Allocator.Error || std.Io.Writer.Error;

fn isNumeric(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .float, .comptime_int, .comptime_float => true,
        else => false,
    };
}

fn castValue(comptime T: type, value: anytype) T {
    const V = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .float => switch (@typeInfo(V)) {
            .float, .comptime_float => @floatCast(value),
            .int, .comptime_int => @floatFromInt(value),
            else => @compileError("cannot cast to float"),
        },
        .int => switch (@typeInfo(V)) {
            .int, .comptime_int => @intCast(value),
            .float, .comptime_float => @intFromFloat(value),
            else => @compileError("cannot cast to int"),
        },
        else => @compileError("unsupported cast target"),
    };
}

pub fn Series(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        name: []const u8,
        data: []T,

        pub fn init(allocator: std.mem.Allocator, name: []const u8, values: []const T) DataError!Self {
            return .{
                .allocator = allocator,
                .name = try allocator.dupe(u8, name),
                .data = try allocator.dupe(T, values),
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.name);
            self.allocator.free(self.data);
            self.* = undefined;
        }

        pub fn len(self: Self) usize {
            return self.data.len;
        }

        pub fn get(self: Self, index: usize) T {
            return self.data[index];
        }

        pub fn set(self: *Self, index: usize, value: T) void {
            self.data[index] = value;
        }

        pub fn clone(self: Self) DataError!Self {
            return Self.init(self.allocator, self.name, self.data);
        }

        pub fn head(self: Self, n: usize) DataError!Self {
            return Self.init(self.allocator, self.name, self.data[0..@min(n, self.data.len)]);
        }

        pub fn tail(self: Self, n: usize) DataError!Self {
            const count = @min(n, self.data.len);
            return Self.init(self.allocator, self.name, self.data[self.data.len - count ..]);
        }

        pub fn toArray(self: Self) array_mod.ArrayError!array_mod.Array(T) {
            return array_mod.Array(T).fromSlice(self.allocator, self.data, &.{self.data.len});
        }

        pub fn toNDArray(self: Self) array_mod.ArrayError!array_mod.NDArray(T) {
            return self.toArray();
        }

        fn map(self: Self, comptime op: fn (T) T) DataError!Self {
            const out = try Self.init(self.allocator, self.name, self.data);
            for (out.data) |*v| v.* = op(v.*);
            return out;
        }

        fn binaryScalar(self: Self, scalar: T, comptime op: fn (T, T) T) DataError!Self {
            const out = try Self.init(self.allocator, self.name, self.data);
            for (out.data) |*v| v.* = op(v.*, scalar);
            return out;
        }

        fn opAdd(a: T, b: T) T {
            return a + b;
        }
        fn opSub(a: T, b: T) T {
            return a - b;
        }
        fn opMul(a: T, b: T) T {
            return a * b;
        }
        fn opDiv(a: T, b: T) T {
            return a / b;
        }

        pub fn addScalar(self: Self, scalar: T) DataError!Self {
            if (comptime !isNumeric(T)) @compileError("addScalar requires a numeric Series");
            return self.binaryScalar(scalar, opAdd);
        }

        pub fn subScalar(self: Self, scalar: T) DataError!Self {
            if (comptime !isNumeric(T)) @compileError("subScalar requires a numeric Series");
            return self.binaryScalar(scalar, opSub);
        }

        pub fn mulScalar(self: Self, scalar: T) DataError!Self {
            if (comptime !isNumeric(T)) @compileError("mulScalar requires a numeric Series");
            return self.binaryScalar(scalar, opMul);
        }

        pub fn divScalar(self: Self, scalar: T) DataError!Self {
            if (comptime !isNumeric(T)) @compileError("divScalar requires a numeric Series");
            return self.binaryScalar(scalar, opDiv);
        }

        pub fn sum(self: Self) T {
            if (comptime !isNumeric(T)) @compileError("sum requires a numeric Series");
            var total: T = 0;
            for (self.data) |v| total += v;
            return total;
        }

        pub fn mean(self: Self) f64 {
            if (comptime !isNumeric(T)) @compileError("mean requires a numeric Series");
            if (self.data.len == 0) return std.math.nan(f64);
            var total: f64 = 0;
            for (self.data) |v| total += castValue(f64, v);
            return total / @as(f64, @floatFromInt(self.data.len));
        }

        pub fn min(self: Self) ?T {
            if (comptime !isNumeric(T)) @compileError("min requires a numeric Series");
            if (self.data.len == 0) return null;
            var m = self.data[0];
            for (self.data[1..]) |v| {
                if (v < m) m = v;
            }
            return m;
        }

        pub fn max(self: Self) ?T {
            if (comptime !isNumeric(T)) @compileError("max requires a numeric Series");
            if (self.data.len == 0) return null;
            var m = self.data[0];
            for (self.data[1..]) |v| {
                if (v > m) m = v;
            }
            return m;
        }

        pub fn sort(self: Self) DataError!Self {
            const out = try self.clone();
            std.sort.insertion(T, out.data, {}, struct {
                fn lessThan(_: void, a: T, b: T) bool {
                    return a < b;
                }
            }.lessThan);
            return out;
        }

        pub fn filter(self: Self, mask: []const bool) DataError!Self {
            if (mask.len != self.data.len) return error.LengthMismatch;
            var count: usize = 0;
            for (mask) |keep| {
                if (keep) count += 1;
            }
            var values = try self.allocator.alloc(T, count);
            defer self.allocator.free(values);
            var write: usize = 0;
            for (self.data, mask) |v, keep| {
                if (keep) {
                    values[write] = v;
                    write += 1;
                }
            }
            return Self.init(self.allocator, self.name, values);
        }

        pub fn print(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("Series({s}, len={}): [", .{ self.name, self.data.len });
            const limit = @min(self.data.len, 12);
            for (self.data[0..limit], 0..) |v, i| {
                if (i != 0) try writer.print(", ", .{});
                try writer.print("{}", .{v});
            }
            if (self.data.len > limit) try writer.print(", ...", .{});
            try writer.print("]", .{});
        }
    };
}

test "series basics" {
    const gpa = std.testing.allocator;
    var s = try Series(f64).init(gpa, "x", &.{ 1, 2, 3 });
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 3), s.len());
    try std.testing.expectApproxEqAbs(@as(f64, 2), s.mean(), 1e-12);
    var shifted = try s.addScalar(10);
    defer shifted.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 11, 12, 13 }, shifted.data);
}
