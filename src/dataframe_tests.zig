test {
    _ = @import("dataframe_eager_tests.zig");
    _ = @import("dataframe_sort_profile_tests.zig");
    _ = @import("dataframe_profile_tests.zig");
    _ = @import("dataframe_lazy_collect_tests.zig");
    _ = @import("dataframe_join_tests.zig");
    _ = @import("dataframe_parquet_tests.zig");
}
