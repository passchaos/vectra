test {
    _ = @import("dataframe_eager_tests.zig");
    _ = @import("dataframe_sort_profile_tests.zig");
    _ = @import("dataframe/profile/tests.zig");
    _ = @import("dataframe/lazy/collect_tests.zig");
    _ = @import("dataframe/join/tests.zig");
    _ = @import("dataframe_parquet_tests.zig");
}
