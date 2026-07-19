//! CLI smoke gate for the Vectra -> Axiom CUDA bridge.

const std = @import("std");
const vx = @import("vectra");

const OutputFormat = enum { text, json };

const Args = struct {
    format: OutputFormat = .text,
    expect: ?vx.axiom_backend.cuda.Status = null,
};

pub fn main(init: std.process.Init) !void {
    const args = parseArgs(init) catch |err| {
        var stderr_buffer: [1024]u8 = undefined;
        var stderr = std.Io.File.stderr().writerStreaming(init.io, &stderr_buffer);
        try stderr.interface.print("invalid axiom-cuda-smoke args: {s}\n", .{@errorName(err)});
        try stderr.interface.flush();
        std.process.exit(2);
    };

    const report = vx.axiom_backend.cuda.runSmoke(std.heap.smp_allocator);
    const expectation_ok = if (args.expect) |expected| report.status == expected else true;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    switch (args.format) {
        .text => {
            try report.writeText(&stdout.interface);
            if (args.expect) |expected| {
                try stdout.interface.print("expect={s} expectation_ok={}\n", .{ expected.label(), expectation_ok });
            }
        },
        .json => {
            try report.writeJson(&stdout.interface);
            if (args.expect) |expected| {
                try stdout.interface.print(
                    "{{\"kind\":\"vectra_axiom_cuda_expectation\",\"expect\":\"{s}\",\"actual\":\"{s}\",\"ok\":{}}}\n",
                    .{ expected.label(), report.status.label(), expectation_ok },
                );
            }
        },
    }
    try stdout.interface.flush();

    if (!report.ok() or !expectation_ok) std.process.exit(1);
}

fn parseArgs(init: std.process.Init) !Args {
    var parsed: Args = .{};
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--json")) {
            parsed.format = .json;
        } else if (std.mem.eql(u8, arg, "--text")) {
            parsed.format = .text;
        } else if (std.mem.eql(u8, arg, "--expect")) {
            const value = args.next() orelse return error.MissingExpectation;
            parsed.expect = parseStatus(value) orelse return error.InvalidExpectation;
        } else {
            return error.UnknownArgument;
        }
    }
    return parsed;
}

fn parseStatus(value: []const u8) ?vx.axiom_backend.cuda.Status {
    const statuses = [_]vx.axiom_backend.cuda.Status{ .disabled, .skipped, .ran, .failed };
    for (statuses) |status| {
        if (std.mem.eql(u8, value, status.label())) return status;
    }
    return null;
}
