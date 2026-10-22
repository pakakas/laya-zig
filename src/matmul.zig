//! GEMM: f32 and f16-weight kernels, bias.

const std = @import("std");
const Allocator = std.mem.Allocator;

const config = @import("config.zig");
const simd = @import("simd.zig");
const pool = @import("pool.zig");

const D = config.D;
const INTER = config.INTER;
const HEAD_FF = config.HEAD_FF;

const V = simd.V;
const v8 = simd.v8;
const hsum = simd.hsum;
const dotvw = simd.dotvw;

const parallel = pool.parallel;

/// The parts of a matmul the kernel sees as f32. The weight slice lives in the job
/// structs below, because its element type is the only difference between them.
const MM = struct {
    out: []f32,
    x: []const f32,
    n: usize,
    k: usize,
    rows: usize,
};

const MM32 = struct { m: MM, w: []align(1) const f32 };

/// A row-major [n][k] weight matrix as it sits in the file, so the kernel converts
/// on load and no F32 copy of it is ever made.
const MM16 = struct { m: MM, w: []align(1) const f16 };

/// Widening four rows at a time needs room for the largest k a caller passes.
const tile_k_max: usize = @max(@max(D, INTER), HEAD_FF);

/// out[*][j..j+4] against four weight rows at once, two tokens at a time.
///
/// The shape matters: with one weight row at a time (the old form, `dotv` per
/// output) there are only two accumulator chains in flight, so the kernel waits
/// on FMA latency instead of issuing. Four rows x two tokens = eight independent
/// chains, which is what it takes to keep both FMA units busy, and each row's
/// k-sweep stays a straight contiguous stream.
///
/// The rows arrive already f32: converting half precision inside this loop would
/// repeat the same conversion for every token pair, and `vcvtph2ps` issues on the
/// two ports the FMAs live on, so it costs throughput rather than being free.
fn block4(m: *const MM, j: usize, w: [4][]align(1) const f32) void {
    const k = m.k;
    const n = m.n;
    const rows = m.rows;
    const w0 = w[0];
    const w1 = w[1];
    const w2 = w[2];
    const w3 = w[3];

    var i: usize = 0;
    while (i < rows) : (i += 2) {
        const step: usize = if (i + 1 < rows) 1 else 0;
        const x0 = m.x[i * k ..];
        const x1 = m.x[(i + step) * k ..];
        const o0 = m.out[i * n ..];
        const o1 = m.out[(i + step) * n ..];
        const two = step == 1;

        var a00: V = @splat(0);
        var a01: V = @splat(0);
        var a02: V = @splat(0);
        var a03: V = @splat(0);
        var a10: V = @splat(0);
        var a11: V = @splat(0);
        var a12: V = @splat(0);
        var a13: V = @splat(0);

        var t: usize = 0;
        while (t + 8 <= k) : (t += 8) {
            const p0 = v8(x0, t);
            const p1 = v8(x1, t);
            const y0 = v8(w0, t);
            const y1 = v8(w1, t);
            const y2 = v8(w2, t);
            const y3 = v8(w3, t);
            a00 = @mulAdd(V, p0, y0, a00);
            a01 = @mulAdd(V, p1, y0, a01);
            a02 = @mulAdd(V, p0, y1, a02);
            a03 = @mulAdd(V, p1, y1, a03);
            a10 = @mulAdd(V, p0, y2, a10);
            a11 = @mulAdd(V, p1, y2, a11);
            a12 = @mulAdd(V, p0, y3, a12);
            a13 = @mulAdd(V, p1, y3, a13);
        }
        while (t < k) : (t += 1) {
            const c0 = w0[t];
            const c1 = w1[t];
            const c2 = w2[t];
            const c3 = w3[t];
            a00[0] = @mulAdd(f32, x0[t], c0, a00[0]);
            a01[0] = @mulAdd(f32, x1[t], c0, a01[0]);
            a02[0] = @mulAdd(f32, x0[t], c1, a02[0]);
            a03[0] = @mulAdd(f32, x1[t], c1, a03[0]);
            a10[0] = @mulAdd(f32, x0[t], c2, a10[0]);
            a11[0] = @mulAdd(f32, x1[t], c2, a11[0]);
            a12[0] = @mulAdd(f32, x0[t], c3, a12[0]);
            a13[0] = @mulAdd(f32, x1[t], c3, a13[0]);
        }

        o0[j + 0] = hsum(a00);
        o0[j + 1] = hsum(a02);
        o0[j + 2] = hsum(a10);
        o0[j + 3] = hsum(a12);
        if (two) {
            o1[j + 0] = hsum(a01);
            o1[j + 1] = hsum(a03);
            o1[j + 2] = hsum(a11);
            o1[j + 3] = hsum(a13);
        }
    }
}

/// The four weight rows a block works on.
fn four(w: []align(1) const f32, k: usize) [4][]align(1) const f32 {
    return .{ w[0..k], w[k..][0..k], w[2 * k ..][0..k], w[3 * k ..][0..k] };
}

/// Widen four rows, in the order `block4` reads them. F16 -> F32 has nothing to
/// round, so this is the same arithmetic done in the same order.
fn widen4(src: []align(1) const f16, tile: []align(64) f32) void {
    const n = src.len;
    var t: usize = 0;
    while (t + 8 <= n) : (t += 8) {
        const h: @Vector(8, f16) = @bitCast(src[t..][0..8].*);
        tile[t..][0..8].* = @bitCast(@as(@Vector(8, f32), @floatCast(h)));
    }
    while (t < n) : (t += 1) tile[t] = @floatCast(src[t]);
}

/// out[i][j] = dot(x[i], w[j]) -- parallelised over 4-column blocks so that each
/// weight row is streamed exactly once (this is the difference between ~2 s and
/// ~200 ms per question: the other order re-reads the whole matrix per token).
///
/// The unit of work is a block, not a column: `parallel` splits whatever range it
/// is given into chunks of an arbitrary size, and a chunk that starts or ends
/// mid-block falls back to the per-column path below, which reads each weight row
/// once per token instead of once per four columns.
fn mmJob32(ctx: *anyopaque, blo: usize, bhi: usize) void {
    const m: *const MM32 = @ptrCast(@alignCast(ctx));
    const k = m.m.k;
    var b = blo;
    while (b < bhi) : (b += 1) {
        const j = b * 4;
        if (j + 4 <= m.m.n) {
            block4(&m.m, j, four(m.w[j * k ..], k));
            continue;
        }
        var jj = j;
        while (jj < m.m.n) : (jj += 1) {
            const wrow = m.w[jj * k ..][0..k];
            for (0..m.m.rows) |i| m.m.out[i * m.m.n + jj] = dotvw(f32, m.m.x[i * k ..][0..k], wrow);
        }
    }
}

/// The same sweep over half-precision weights. Each block is widened once into a
/// tile on this thread's stack and then reused by every token pair, which is what
/// keeps the conversions at one per weight element instead of one per pair.
fn mmJob16(ctx: *anyopaque, blo: usize, bhi: usize) void {
    const m: *const MM16 = @ptrCast(@alignCast(ctx));
    const k = m.m.k;
    var buf: [4 * tile_k_max]f32 align(64) = undefined;
    const tile: []align(64) f32 = (&buf)[0..];
    var b = blo;
    while (b < bhi) : (b += 1) {
        const j = b * 4;
        if (j + 4 <= m.m.n) {
            widen4(m.w[j * k ..][0 .. 4 * k], tile);
            block4(&m.m, j, four(tile[0 .. 4 * k], k));
            continue;
        }
        var jj = j;
        while (jj < m.m.n) : (jj += 1) {
            const wrow = m.w[jj * k ..][0..k];
            for (0..m.m.rows) |i| m.m.out[i * m.m.n + jj] = dotvw(f16, m.m.x[i * k ..][0..k], wrow);
        }
    }
}

const Bias = struct { out: []f32, b: []const f32, n: usize };

fn biasJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const m: *Bias = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const row = m.out[i * m.n ..][0..m.n];
        for (row, 0..) |*v, j| v.* += m.b[j];
    }
}

pub fn matmul(alloc: Allocator, out: []f32, x: []const f32, w: []const f32, rows: usize, n: usize, k: usize) void {
    var m = MM32{ .m = .{ .out = out, .x = x, .n = n, .k = k, .rows = rows }, .w = w };
    parallel(alloc, (n + 3) / 4, &m, mmJob32);
}

/// F16 weights straight out of the file, converted inside the kernel: no 478 MB
/// F32 copy, and half the bytes streamed per forward.
pub fn linear16(alloc: Allocator, out: []f32, x: []const f32, w: []align(1) const f16, bias: ?[]const f32, rows: usize, n: usize, k: usize) void {
    var m = MM16{ .m = .{ .out = out, .x = x, .n = n, .k = k, .rows = rows }, .w = w };
    parallel(alloc, (n + 3) / 4, &m, mmJob16);
    if (bias) |b| {
        var bias_job = Bias{ .out = out, .b = b, .n = n };
        parallel(alloc, rows, &bias_job, biasJob);
    }
}
