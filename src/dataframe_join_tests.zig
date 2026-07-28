test {
    _ = @import("dataframe_join_eager_key_tests.zig");
    _ = @import("dataframe_join_eager_nullable_tests.zig");
    _ = @import("dataframe_join_stack_distinct_tests.zig");
    _ = @import("dataframe_join_lazy_tests.zig");
}
