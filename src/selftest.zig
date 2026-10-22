//! `--selftest`: check the SIMD kernels against f64.

const std = @import("std");
const Allocator = std.mem.Allocator;

const math = @import("math.zig");
const simd = @import("simd.zig");

const gelu = math.gelu;

const expv = simd.expv;
const geluv = simd.geluv;

const matmul = @import("matmul.zig").matmul;

/// `--selftest`: check the SIMD kernels against f64 for the shapes the model uses.
pub fn selfTest(gpa: Allocator, w: *std.Io.Writer) !void {
    const Shape = struct { rows: usize, n: usize, k: usize, what: []const u8 };
    const shapes = [_]Shape{
        .{ .rows = 1, .n = 768, .k = 768, .what = "scorer / act head" },
        .{ .rows = 63, .n = 2304, .k = 768, .what = "qkv + mlp in" },
        .{ .rows = 63, .n = 768, .k = 1152, .what = "mlp out" },
        .{ .rows = 63, .n = 3072, .k = 768, .what = "head ff" },
        .{ .rows = 63, .n = 768, .k = 768, .what = "attn / head proj" },
    };
    var rng = std.Random.DefaultPrng.init(0x5eed);
    var worst: f64 = 0;
    for (shapes) |s| {
        const x = try gpa.alloc(f32, s.rows * s.k);
        defer gpa.free(x);
        const wm = try gpa.alloc(f32, s.n * s.k);
        defer gpa.free(wm);
        const out = try gpa.alloc(f32, s.rows * s.n);
        defer gpa.free(out);
        for (x) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
        for (wm) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

        matmul(gpa, out, x, wm, s.rows, s.n, s.k);

        var local_worst: f64 = 0;
        for (0..s.rows) |i| {
            var j: usize = 0;
            while (j < s.n) : (j += 29) {
                var ref: f64 = 0;
                for (0..s.k) |t| ref += @as(f64, x[i * s.k + t]) * @as(f64, wm[j * s.k + t]);
                const got: f64 = out[i * s.n + j];
                const rel = @abs(got - ref) / @max(1.0, @abs(ref));
                if (rel > local_worst) local_worst = rel;
            }
        }
        worst = @max(worst, local_worst);
        try w.print("  {d:>3}x{d:<5} x {d:>4}x{d:<5} {s:<18} max rel err {e:.2}\n", .{ s.rows, s.k, s.k, s.n, s.what, local_worst });
    }
    try w.print("\nworst relative error vs f64: {e:.3}\n", .{worst});

    // The elementwise transcendentals run once per activation (millions of times a
    // forward). expv is checked against the f64 library exp; geluv against the
    // scalar gelu the decision head still uses -- same coefficients, different code
    // path, so a vectorisation bug shows up here. Absolute truth for gelu comes from
    // tools/refcheck.py, which evaluates math.erf in f64.
    var worst_exp: f64 = 0;
    var worst_gelu: f64 = 0;
    {
        var lane: [8]f32 = undefined;
        var t: i32 = -8000;
        while (t <= 8000) : (t += 1) {
            const base: f32 = @as(f32, @floatFromInt(t)) * 0.001;
            for (0..8) |l| lane[l] = base + @as(f32, @floatFromInt(l)) * 0.00017;
            const ea: [8]f32 = @bitCast(expv(@bitCast(lane)));
            const ga: [8]f32 = @bitCast(geluv(@bitCast(lane)));
            for (0..8) |l| {
                const x = lane[l];
                const we = @exp(@as(f64, x));
                worst_exp = @max(worst_exp, @abs(@as(f64, ea[l]) - we) / we);
                const wg: f64 = gelu(x);
                // |x|/sqrt2 > 5 drives erf to saturation, and there both this and the
                // scalar path resolve a ~1e-7 cancellation against a true value of ~0,
                // so the comparison floors at an absolute error of 5e-7.
                const rel = @abs(@as(f64, ga[l]) - wg) / @max(0.05, @abs(wg));
                worst_gelu = @max(worst_gelu, rel);
            }
        }
    }
    try w.print("expv  max rel err vs f64 exp {e:.2}\ngeluv max rel err vs scalar gelu {e:.2}\n", .{ worst_exp, worst_gelu });

    if (worst > 1e-4) {
        try w.print("FAIL: the GEMM kernel is wrong (expected < 1e-4)\n", .{});
    } else if (worst_exp > 1e-6 or worst_gelu > 1e-5) {
        try w.print("FAIL: the vector transcendentals are wrong\n", .{});
    } else {
        try w.print("PASS\n", .{});
    }
    try w.flush();
}
