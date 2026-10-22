//! Multi-head attention (full and sliding-window) and rotary position embeddings.

const std = @import("std");
const Allocator = std.mem.Allocator;

const config = @import("config.zig");
const simd = @import("simd.zig");
const pool = @import("pool.zig");

const D = config.D;
const H = config.H;
const HD = config.HD;

const V = simd.V;
const v8 = simd.v8;
const hsum = simd.hsum;
const splatv = simd.splatv;
const expv = simd.expv;

const parallel = pool.parallel;

pub const AttJob = struct {
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []f32, // [L][D]
    att: []f32, // [L][stride]
    L: usize,
    stride: usize,
    win: ?usize,
};

/// one head's score against one key: 64 MACs in 8 vector FMAs
inline fn dot64(a: []const f32, ao: usize, b: []const f32, bo: usize) f32 {
    var a0 = v8(a, ao) * v8(b, bo);
    var a1 = v8(a, ao + 8) * v8(b, bo + 8);
    var a2 = v8(a, ao + 16) * v8(b, bo + 16);
    var a3 = v8(a, ao + 24) * v8(b, bo + 24);
    a0 = @mulAdd(V, v8(a, ao + 32), v8(b, bo + 32), a0);
    a1 = @mulAdd(V, v8(a, ao + 40), v8(b, bo + 40), a1);
    a2 = @mulAdd(V, v8(a, ao + 48), v8(b, bo + 48), a2);
    a3 = @mulAdd(V, v8(a, ao + 56), v8(b, bo + 56), a3);
    return hsum((a0 + a1) + (a2 + a3));
}

pub fn attWorker(ctx: *anyopaque, lo: usize, hi: usize) void {
    const a: *AttJob = @ptrCast(@alignCast(ctx));
    const L = a.L;
    const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
    for (lo..hi) |i| {
        const row = a.out[i * D ..][0..D];
        var klo: usize = 0;
        var khi: usize = L;
        var off: usize = 0;
        if (a.win) |w| {
            klo = if (i >= w) i - w else 0;
            khi = @min(L, i + w + 1);
            off = if (i >= w) 0 else w - i;
        }
        const cnt = khi - klo;
        const rowp = a.att[i * a.stride + off ..];
        for (0..H) |h| {
            const qo = i * D + h * HD;
            var mx: f32 = -std.math.inf(f32);
            for (klo..khi) |j| {
                const s = dot64(a.q, qo, a.k, j * D + h * HD) * scale;
                rowp[j - klo] = s;
                mx = @max(mx, s);
            }
            var sv_acc: V = splatv(0);
            const mxv = @as(V, @splat(mx));
            var tv: usize = 0;
            while (tv + 8 <= cnt) : (tv += 8) {
                const e = expv(v8(rowp, tv) - mxv);
                rowp[tv..][0..8].* = @as([8]f32, @bitCast(e));
                sv_acc += e;
            }
            var sum = hsum(sv_acc);
            while (tv < cnt) : (tv += 1) {
                const e = @exp(rowp[tv] - mx);
                rowp[tv] = e;
                sum += e;
            }
            const inv = 1.0 / sum;
            // the head's 64 outputs stay in registers across the whole key range;
            // accumulating through `row` instead turned every key into 8 loads
            // and 8 stores. Normalising afterwards rather than per key takes the
            // multiply out of the inner loop, and `sum >= 1` so dropping a weight
            // below 1e-20 cannot change a single output bit.
            var r: [HD / 8]V = undefined;
            inline for (0..HD / 8) |c| r[c] = splatv(0);
            for (0..cnt) |t| {
                const p = rowp[t];
                if (p < 1e-20) continue;
                const vo = (klo + t) * D + h * HD;
                const pv: V = @splat(p);
                inline for (0..HD / 8) |c| r[c] = @mulAdd(V, pv, v8(a.v, vo + c * 8), r[c]);
            }
            const iv: V = @splat(inv);
            const base = h * HD;
            inline for (0..HD / 8) |c| row[base + c * 8 ..][0..8].* = @as([8]f32, @bitCast(r[c] * iv));
        }
    }
}

const RopeCtx = struct {
    x: []f32,
    inv: *const [32]f32,
};

fn ropeJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const m: *RopeCtx = @ptrCast(@alignCast(ctx));
    const inv = m.inv;
    for (lo..hi) |p| {
        const pos: f32 = @floatFromInt(p);
        var cs: [64]f32 = undefined;
        var sn: [64]f32 = undefined;
        for (0..32) |j| {
            const f = inv[j] * pos;
            cs[j] = @cos(f);
            sn[j] = @sin(f);
            cs[j + 32] = cs[j];
            sn[j + 32] = sn[j];
        }
        const o = p * D;
        for (0..H) |h| {
            const b = o + h * HD;
            for (0..32) |d| {
                const x0 = m.x[b + d];
                const x1 = m.x[b + d + 32];
                m.x[b + d] = x0 * cs[d] - x1 * sn[d];
                m.x[b + d + 32] = x1 * cs[d + 32] + x0 * sn[d + 32];
            }
        }
    }
}

pub fn ropeApply(alloc: Allocator, x: []f32, L: usize, inv: *const [32]f32) void {
    var m = RopeCtx{ .x = x, .inv = inv };
    parallel(alloc, L, &m, ropeJob);
}
