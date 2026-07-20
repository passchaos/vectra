const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    _ = b.option(bool, "axiom-cuda", "Compatibility flag: Axiom CUDA wrapping is always compiled in") orelse true;
    _ = b.option(bool, "axiom-cuda-dispatch", "Compatibility flag: supported CUDA dispatch always uses Axiom") orelse true;
    _ = b.option(bool, "axiom-cpu-dispatch", "Compatibility flag: supported CPU dispatch always uses Axiom CPU lowering") orelse true;
    const enable_axiom_cuda = true;
    const enable_axiom_cuda_dispatch = true;
    const enable_axiom_cpu_dispatch = true;
    const axiom_cuda_expect = b.option([]const u8, "axiom-cuda-expect", "Optional Axiom CUDA smoke status expectation: disabled, skipped, ran, or failed");
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const veyra_dep = b.dependency("veyra", .{
        .target = target,
        .optimize = optimize,
    });
    const veyra_mod = veyra_dep.module("veyra");
    const alea_dep = b.dependency("alea", .{
        .target = target,
        .optimize = optimize,
    });
    const alea_mod = alea_dep.module("alea");
    const axiom_dep = b.dependency("axiom", .{
        .target = target,
        .optimize = optimize,
    });
    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_axiom_cuda", enable_axiom_cuda);
    build_options.addOption(bool, "enable_axiom_cuda_dispatch", enable_axiom_cuda_dispatch);
    build_options.addOption(bool, "enable_axiom_cpu_dispatch", enable_axiom_cpu_dispatch);

    const mod = b.addModule("vectra", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
        .link_libc = enable_axiom_cuda,
        .imports = &.{
            .{ .name = "veyra", .module = veyra_mod },
            .{ .name = "alea", .module = alea_mod },
        },
    });
    mod.addOptions("vectra_build_options", build_options);
    mod.addImport("axiom", axiom_dep.module("axiom"));
    if (target.result.os.tag == .macos) {
        mod.linkSystemLibrary("objc", .{});
        mod.linkFramework("Metal", .{});
        mod.linkFramework("Foundation", .{});
    }

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const exe = b.addExecutable(.{
        .name = "vectra",
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "vectra" is the name you will use in your source code to
                // import this module (e.g. `@import("vectra")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "vectra", .module = mod },
            },
        }),
    });

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const bench_exe = b.addExecutable(.{
        .name = "vectra-array-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/bench_array_perf.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const bench_cmd = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run Array performance smoke benchmark");
    bench_step.dependOn(&bench_cmd.step);

    const api_boundary_audit_cmd = b.addSystemCommand(&.{ "python3", "tools/api_boundary_audit.py" });
    const api_boundary_audit_step = b.step("api-boundary-audit", "Check that Vectra keeps Array API boundaries and leaves Tensor/autograd to Forge");
    api_boundary_audit_step.dependOn(&api_boundary_audit_cmd.step);

    const array_api_coverage_audit_cmd = b.addSystemCommand(&.{ "python3", "tools/array_api_coverage_audit.py" });
    const array_api_coverage_audit_step = b.step("array-api-coverage-audit", "Audit NumPy/PyTorch-style dense Array API coverage with autograd out of scope");
    array_api_coverage_audit_step.dependOn(&array_api_coverage_audit_cmd.step);

    const dtype_promotion_smoke_exe = b.addExecutable(.{
        .name = "vectra-dtype-promotion-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/dtype_promotion_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const dtype_promotion_smoke_cmd = b.addRunArtifact(dtype_promotion_smoke_exe);
    const dtype_promotion_smoke_step = b.step("dtype-promotion-smoke", "Run representative NumPy/PyTorch-style dtype promotion smoke");
    dtype_promotion_smoke_step.dependOn(&dtype_promotion_smoke_cmd.step);

    const einsum_smoke_exe = b.addExecutable(.{
        .name = "vectra-einsum-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/einsum_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const einsum_smoke_cmd = b.addRunArtifact(einsum_smoke_exe);
    const einsum_smoke_step = b.step("einsum-smoke", "Run bounded NumPy/PyTorch-style einsum syntax smoke");
    einsum_smoke_step.dependOn(&einsum_smoke_cmd.step);

    const contraction_smoke_exe = b.addExecutable(.{
        .name = "vectra-contraction-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/contraction_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const contraction_smoke_cmd = b.addRunArtifact(contraction_smoke_exe);
    const contraction_smoke_step = b.step("contraction-smoke", "Run NumPy/PyTorch-style tensordot/contractAxes smoke");
    contraction_smoke_step.dependOn(&contraction_smoke_cmd.step);

    const indexing_smoke_exe = b.addExecutable(.{
        .name = "vectra-indexing-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/indexing_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const indexing_smoke_cmd = b.addRunArtifact(indexing_smoke_exe);
    const indexing_smoke_step = b.step("indexing-smoke", "Run NumPy/PyTorch-style gather/scatter/where indexing smoke");
    indexing_smoke_step.dependOn(&indexing_smoke_cmd.step);

    const shape_view_smoke_exe = b.addExecutable(.{
        .name = "vectra-shape-view-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/shape_view_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const shape_view_smoke_cmd = b.addRunArtifact(shape_view_smoke_exe);
    const shape_view_smoke_step = b.step("shape-view-smoke", "Run NumPy/PyTorch-style shape/view/broadcast smoke");
    shape_view_smoke_step.dependOn(&shape_view_smoke_cmd.step);

    const examples_step = b.step("examples", "Run Vectra usage examples");

    const basic_array_example_exe = b.addExecutable(.{
        .name = "vectra-example-basic-array",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/basic_array.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const basic_array_example_cmd = b.addRunArtifact(basic_array_example_exe);
    const basic_array_example_step = b.step("example-basic-array", "Run basic Array/broadcast/reduction usage example");
    basic_array_example_step.dependOn(&basic_array_example_cmd.step);
    examples_step.dependOn(&basic_array_example_cmd.step);

    const axiom_backend_policy_example_exe = b.addExecutable(.{
        .name = "vectra-example-axiom-backend-policy",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/axiom_backend_policy.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_backend_policy_example_cmd = b.addRunArtifact(axiom_backend_policy_example_exe);
    const axiom_backend_policy_example_step = b.step("example-axiom-backend-policy", "Run unified Axiom backend policy usage example");
    axiom_backend_policy_example_step.dependOn(&axiom_backend_policy_example_cmd.step);
    examples_step.dependOn(&axiom_backend_policy_example_cmd.step);

    const axiom_cuda_bridge_example_exe = b.addExecutable(.{
        .name = "vectra-example-axiom-cuda-bridge",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/axiom_cuda_bridge.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_cuda_bridge_example_cmd = b.addRunArtifact(axiom_cuda_bridge_example_exe);
    const axiom_cuda_bridge_example_step = b.step("example-axiom-cuda-bridge", "Run explicit Axiom CUDA bridge usage example");
    axiom_cuda_bridge_example_step.dependOn(&axiom_cuda_bridge_example_cmd.step);
    examples_step.dependOn(&axiom_cuda_bridge_example_cmd.step);

    const large_matmul_add_example_exe = b.addExecutable(.{
        .name = "vectra-example-large-matmul-add",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/large_matmul_add.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const large_matmul_add_example_cmd = b.addRunArtifact(large_matmul_add_example_exe);
    if (b.args) |args| {
        large_matmul_add_example_cmd.addArgs(args);
    }
    const large_matmul_add_example_step = b.step("example-large-matmul-add", "Run large random GEMM-plus-add CPU/CUDA usage example (dry-run by default)");
    large_matmul_add_example_step.dependOn(&large_matmul_add_example_cmd.step);
    examples_step.dependOn(&large_matmul_add_example_cmd.step);

    const large_matmul_add_smoke_cmd = b.addRunArtifact(large_matmul_add_example_exe);
    large_matmul_add_smoke_cmd.addArgs(&.{ "--smoke", "--backend=both" });
    const large_matmul_add_smoke_step = b.step("example-large-matmul-add-smoke", "Run tiny executable smoke for the large GEMM-plus-add example");
    large_matmul_add_smoke_step.dependOn(&large_matmul_add_smoke_cmd.step);

    const matmul_add_compare_smoke_cmd = b.addSystemCommand(&.{
        "python3",
        "tools/bench_matmul_add_compare.py",
        "--smoke",
        "--m=64",
        "--n=64",
        "--k=64",
        "--warmup=1",
        "--iters=2",
        "--op=matmul_add",
        "--skip-torch-compile",
        "--repeat=2",
        "--max-ratio=2.0",
    });
    const matmul_add_compare_smoke_step = b.step("bench-matmul-add-compare-smoke", "Run quick Vectra/Axiom vs PyTorch CUDA matmul+add ratio gate");
    matmul_add_compare_smoke_step.dependOn(&matmul_add_compare_smoke_cmd.step);

    const matmul_add_compare_production_cmd = b.addSystemCommand(&.{
        "python3",
        "tools/bench_matmul_add_compare.py",
        "--execute",
        "--m=16384",
        "--n=4096",
        "--k=4096",
        "--warmup=3",
        "--iters=5",
        "--baseline=torch_addmm",
        "--max-ratio=1.10",
    });
    const matmul_add_compare_production_step = b.step("bench-matmul-add-compare-production", "Run production Vectra/Axiom vs PyTorch CUDA matmul+add ratio gate");
    matmul_add_compare_production_step.dependOn(&matmul_add_compare_production_cmd.step);

    const matmul_add_compare_compile_cmd = b.addSystemCommand(&.{
        "python3",
        "tools/bench_matmul_add_compare.py",
        "--execute",
        "--m=16384",
        "--n=4096",
        "--k=4096",
        "--warmup=3",
        "--iters=5",
        "--baseline=torch_compile",
        "--max-ratio=1.10",
    });
    const matmul_add_compare_compile_step = b.step("bench-matmul-add-compare-production-compile", "Run production Vectra/Axiom vs torch.compile CUDA matmul+add ratio gate");
    matmul_add_compare_compile_step.dependOn(&matmul_add_compare_compile_cmd.step);

    const matmul_add_compare_bf16_large_cmd = b.addSystemCommand(&.{
        "python3",
        "tools/bench_matmul_add_compare.py",
        "--smoke",
        "--m=2048",
        "--n=2048",
        "--k=2048",
        "--warmup=2",
        "--iters=3",
        "--dtype=bf16",
        "--op=matmul_add",
        "--skip-torch-compile",
        "--repeat=2",
        "--max-ratio=1.10",
        "--max-first-error=0.01",
        "--max-checksum-error=64.0",
    });
    const matmul_add_compare_bf16_large_step = b.step("bench-matmul-add-compare-bf16-large", "Run repeated BF16 CUDA matmulAdd vs PyTorch ratio gate");
    matmul_add_compare_bf16_large_step.dependOn(&matmul_add_compare_bf16_large_cmd.step);

    const matmul_add_compare_bf16_stability_cmd = b.addSystemCommand(&.{
        "python3",
        "tools/bench_matmul_add_compare.py",
        "--smoke",
        "--m=512",
        "--n=512",
        "--k=512",
        "--warmup=5",
        "--iters=50",
        "--dtype=bf16",
        "--op=matmul_then_add_exp",
        "--baseline=torch_best",
        "--skip-torch-compile",
        "--repeat=2",
        "--max-ratio=1.10",
        "--max-first-error=0.01",
        "--max-checksum-error=64.0",
    });
    const matmul_add_compare_bf16_stability_step = b.step("bench-matmul-add-compare-bf16-stability", "Run repeated BF16 CUDA exp-chain stability and ratio gate");
    matmul_add_compare_bf16_stability_step.dependOn(&matmul_add_compare_bf16_stability_cmd.step);

    const matmul_add_compare_f64_exp_large_cmd = b.addSystemCommand(&.{
        "python3",
        "tools/bench_matmul_add_compare.py",
        "--smoke",
        "--m=2048",
        "--n=2048",
        "--k=2048",
        "--warmup=2",
        "--iters=3",
        "--dtype=f64",
        "--op=matmul_then_add_exp",
        "--skip-torch-compile",
        "--repeat=2",
        "--max-ratio=1.20",
        "--max-first-error=0.001",
        "--max-checksum-error=0.01",
    });
    const matmul_add_compare_f64_exp_large_step = b.step("bench-matmul-add-compare-f64-exp-large", "Run repeated f64 CUDA matmul+add+exp vs PyTorch ratio gate");
    matmul_add_compare_f64_exp_large_step.dependOn(&matmul_add_compare_f64_exp_large_cmd.step);

    const matmul_add_compare_cmd = b.addSystemCommand(&.{
        "python3",
        "tools/bench_matmul_add_compare.py",
    });
    if (b.args) |args| {
        matmul_add_compare_cmd.addArgs(args);
    }
    const matmul_add_compare_step = b.step("bench-matmul-add-compare", "Run Vectra/Axiom vs PyTorch/torch.compile CUDA matmul+add comparison; pass args after --");
    matmul_add_compare_step.dependOn(&matmul_add_compare_cmd.step);

    const axiom_cuda_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-cuda-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_cuda_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_cuda_smoke_cmd = b.addRunArtifact(axiom_cuda_smoke_exe);
    axiom_cuda_smoke_cmd.addArg("--json");
    if (axiom_cuda_expect) |expect| axiom_cuda_smoke_cmd.addArgs(&.{ "--expect", expect });
    const axiom_cuda_smoke_step = b.step("axiom-cuda-smoke", "Run Axiom CUDA f32 elementwise/SAXPY smoke bridge");
    axiom_cuda_smoke_step.dependOn(&axiom_cuda_smoke_cmd.step);

    const axiom_cuda_dispatch_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-cuda-dispatch-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_cuda_dispatch_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_cuda_dispatch_smoke_cmd = b.addRunArtifact(axiom_cuda_dispatch_smoke_exe);
    const axiom_cuda_dispatch_smoke_step = b.step("axiom-cuda-dispatch-smoke", "Run ordinary Array(f32) methods through Axiom CUDA dispatch");
    axiom_cuda_dispatch_smoke_step.dependOn(&axiom_cuda_dispatch_smoke_cmd.step);

    const axiom_cuda_device_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-cuda-device-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_cuda_device_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_cuda_device_smoke_cmd = b.addRunArtifact(axiom_cuda_device_smoke_exe);
    const axiom_cuda_device_smoke_step = b.step("axiom-cuda-device-smoke", "Run explicit Axiom CUDA device-buffer handle smoke");
    axiom_cuda_device_smoke_step.dependOn(&axiom_cuda_device_smoke_cmd.step);

    const axiom_cpu_dispatch_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-cpu-dispatch-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_cpu_dispatch_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_cpu_dispatch_smoke_cmd = b.addRunArtifact(axiom_cpu_dispatch_smoke_exe);
    const axiom_cpu_dispatch_smoke_step = b.step("axiom-cpu-dispatch-smoke", "Run ordinary Array(f32/f64).matmul through Axiom CPU-to-Veyra dispatch");
    axiom_cpu_dispatch_smoke_step.dependOn(&axiom_cpu_dispatch_smoke_cmd.step);

    const axiom_mps_storage_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-mps-storage-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_mps_storage_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_mps_storage_smoke_cmd = b.addRunArtifact(axiom_mps_storage_smoke_exe);
    const axiom_mps_storage_smoke_step = b.step("axiom-mps-storage-smoke", "Run MPS device storage creation/copy/download smoke");
    axiom_mps_storage_smoke_step.dependOn(&axiom_mps_storage_smoke_cmd.step);

    const axiom_mps_gelu_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-mps-gelu-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_mps_gelu_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_mps_gelu_smoke_cmd = b.addRunArtifact(axiom_mps_gelu_smoke_exe);
    const axiom_mps_gelu_smoke_step = b.step("axiom-mps-gelu-smoke", "Run focused MPS GELU composition smoke");
    axiom_mps_gelu_smoke_step.dependOn(&axiom_mps_gelu_smoke_cmd.step);
    axiom_mps_storage_smoke_step.dependOn(&axiom_mps_gelu_smoke_cmd.step);

    const axiom_mps_rank3_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-mps-rank3-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_mps_rank3_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_mps_rank3_smoke_cmd = b.addRunArtifact(axiom_mps_rank3_smoke_exe);
    const axiom_mps_rank3_smoke_step = b.step("axiom-mps-rank3-smoke", "Run focused MPS rank-3 reduction/stat smoke");
    axiom_mps_rank3_smoke_step.dependOn(&axiom_mps_rank3_smoke_cmd.step);
    axiom_mps_storage_smoke_step.dependOn(&axiom_mps_rank3_smoke_cmd.step);

    const axiom_mps_rank3_broadcast_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-mps-rank3-broadcast-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_mps_rank3_broadcast_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_mps_rank3_broadcast_smoke_cmd = b.addRunArtifact(axiom_mps_rank3_broadcast_smoke_exe);
    const axiom_mps_rank3_broadcast_smoke_step = b.step("axiom-mps-rank3-broadcast-smoke", "Run focused MPS rank-3 last-dim broadcast smoke");
    axiom_mps_rank3_broadcast_smoke_step.dependOn(&axiom_mps_rank3_broadcast_smoke_cmd.step);
    axiom_mps_storage_smoke_step.dependOn(&axiom_mps_rank3_broadcast_smoke_cmd.step);

    const fusion_smoke_step = b.step("fusion-smoke", "Run CPU/CUDA fusion correctness, status, and quick performance smoke gates");
    fusion_smoke_step.dependOn(&axiom_cpu_dispatch_smoke_cmd.step);
    fusion_smoke_step.dependOn(&axiom_cuda_dispatch_smoke_cmd.step);
    fusion_smoke_step.dependOn(&axiom_cuda_device_smoke_cmd.step);
    fusion_smoke_step.dependOn(&large_matmul_add_smoke_cmd.step);
    fusion_smoke_step.dependOn(&matmul_add_compare_smoke_cmd.step);

    const fusion_production_gate_step = b.step("fusion-production-gate", "Run production matmul+add PyTorch and torch.compile performance gates");
    fusion_production_gate_step.dependOn(&matmul_add_compare_production_cmd.step);
    fusion_production_gate_step.dependOn(&matmul_add_compare_compile_cmd.step);

    const axiom_backend_policy_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-backend-policy-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_backend_policy_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_backend_policy_smoke_cmd = b.addRunArtifact(axiom_backend_policy_smoke_exe);
    const axiom_backend_policy_smoke_step = b.step("axiom-backend-policy-smoke", "Run unified Axiom CPU/CUDA backend policy smoke");
    axiom_backend_policy_smoke_step.dependOn(&axiom_backend_policy_smoke_cmd.step);

    const axiom_dialect_lowering_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-dialect-lowering-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_dialect_lowering_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_dialect_lowering_smoke_cmd = b.addRunArtifact(axiom_dialect_lowering_smoke_exe);
    const axiom_dialect_lowering_smoke_step = b.step("axiom-dialect-lowering-smoke", "Run Axiom linalg/memref/gpu dialect lowering smoke");
    axiom_dialect_lowering_smoke_step.dependOn(&axiom_dialect_lowering_smoke_cmd.step);

    const axiom_descriptor_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-descriptor-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_descriptor_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_descriptor_smoke_cmd = b.addRunArtifact(axiom_descriptor_smoke_exe);
    const axiom_descriptor_smoke_step = b.step("axiom-descriptor-smoke", "Run Vectra Array/ArrayView to Axiom descriptor smoke");
    axiom_descriptor_smoke_step.dependOn(&axiom_descriptor_smoke_cmd.step);

    const axiom_gemm_layout_smoke_exe = b.addExecutable(.{
        .name = "vectra-axiom-gemm-layout-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/axiom_gemm_layout_smoke.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vectra", .module = mod },
            },
        }),
    });
    const axiom_gemm_layout_smoke_cmd = b.addRunArtifact(axiom_gemm_layout_smoke_exe);
    const axiom_gemm_layout_smoke_step = b.step("axiom-gemm-layout-smoke", "Run Vectra view to Axiom GEMM memref layout planning smoke");
    axiom_gemm_layout_smoke_step.dependOn(&axiom_gemm_layout_smoke_cmd.step);

    // Creates an executable that will run `test` blocks from the provided module.
    // Here `mod` needs to define a target, which is why earlier we made sure to
    // set the releative field.
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Creates an executable that will run `test` blocks from the executable's
    // root module. Note that test executables only test one module at a time,
    // hence why we have to create two separate ones.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // A top level step for running all tests. dependOn can be called multiple
    // times and since the two run steps do not depend on one another, this will
    // make the two of them run in parallel.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}
