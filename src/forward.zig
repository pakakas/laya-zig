//! The forward pass: embeddings, 22 encoder layers, the optional decision head, the
//! option-marker scorer and the act/escalate head.

const std = @import("std");
const Allocator = std.mem.Allocator;

const config = @import("config.zig");
const math = @import("math.zig");
const simd = @import("simd.zig");
const platform = @import("platform.zig");
const diag = @import("diag.zig");
const pool = @import("pool.zig");
const safetensors = @import("safetensors.zig");
const model = @import("model.zig");
const attention = @import("attention.zig");

const D = config.D;
const NL = config.NL;
const INTER = config.INTER;
const WINDOW = config.WINDOW;
const HEAD_LAYERS = config.HEAD_LAYERS;
const HEAD_FF = config.HEAD_FF;

const gelu = math.gelu;
const softmax = math.softmax;
const layerNorm = math.layerNorm;

const v8 = simd.v8;
const geluv = simd.geluv;
const rowDot = simd.rowDot;
const rowDot1 = simd.rowDot1;

const nowNs = platform.nowNs;

const dumpStats = diag.dumpStats;
const dumpIds = diag.dumpIds;
const ST_MATMUL = diag.ST_MATMUL;
const ST_ATTN = diag.ST_ATTN;
const ST_NORM = diag.ST_NORM;
const ST_ROPE = diag.ST_ROPE;
const ST_GELU = diag.ST_GELU;
const ST_RES = diag.ST_RES;
const ST_EMB = diag.ST_EMB;
const ST_SPLIT = diag.ST_SPLIT;
const ST_SERIAL = diag.ST_SERIAL;
const profAdd = diag.profAdd;

const parallel = pool.parallel;

const linear16 = @import("matmul.zig").linear16;

const f16at = safetensors.f16at;

const Model = model.Model;
const Scratch = model.Scratch;

const AttJob = attention.AttJob;
const attWorker = attention.attWorker;
const ropeApply = attention.ropeApply;

pub const FwdCtx = struct {
    m: *const Model,
    s: *Scratch,
    ids: []const u32,
    L: usize,
    qtype: u32,
    markers: []const u32,
    logits: []f32,
    act: [2]f32,
    pooled: []f32,
    feat: []f32,
    act_in: []f32,
};

pub const EnableLayers = enum { layers_full, no_head };

fn embedJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const off: usize = @as(usize, f.ids[i]) * D;
        const dst = f.s.x[i * D ..][0..D];
        for (0..D) |d| dst[d] = f16at(f.m.emb_raw, off + d);
    }
}

fn normJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| layerNorm(f.s.x[i * D ..][0..D], f.m.emb_norm, null);
}

fn mlpJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const m = f.s.mlp[i * 2 * INTER ..];
        const o = f.s.mlp2[i * INTER ..];
        var j: usize = 0;
        while (j + 8 <= INTER) : (j += 8) {
            const g = geluv(v8(m, j)) * v8(m, j + INTER);
            o[j..][0..8].* = @as([8]f32, @bitCast(g));
        }
        while (j < INTER) : (j += 1) o[j] = gelu(m[j]) * m[j + INTER];
    }
}

fn addResJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const dst = f.s.x[i * D ..][0..D];
        const src = f.s.xn[i * D ..][0..D];
        for (0..D) |d| dst[d] += src[d];
    }
}

fn reluJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const row = f.s.ff[i * HEAD_FF ..][0..HEAD_FF];
        for (row) |*v| v.* = if (v.* > 0) v.* else 0;
    }
}

const NormCtx = struct {
    x: []const f32,
    xn: []f32,
    w: []const f32,
    b: ?[]const f32,
};

/// xn[i] = LayerNorm(x[i])  -- x is left untouched (pre-norm, residual must survive)
fn normIntoJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const c: *NormCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const dst = c.xn[i * D ..][0..D];
        @memcpy(dst, c.x[i * D ..][0..D]);
        layerNorm(dst, c.w, c.b);
    }
}

fn normInto(alloc: Allocator, x: []const f32, xn: []f32, L: usize, w: []const f32, b: ?[]const f32) void {
    var c = NormCtx{ .x = x, .xn = xn, .w = w, .b = b };
    parallel(alloc, L, &c, normIntoJob);
}

fn typeEmbJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    const te = f.m.type_emb[@as(usize, f.qtype) * D ..][0..D];
    for (lo..hi) |i| {
        const row = f.s.x[i * D ..][0..D];
        for (0..D) |d| row[d] += te[d];
    }
}

const SplitJob = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    qkv: []const f32,
};

fn splitJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const s: *SplitJob = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const src = s.qkv[i * 3 * D ..];
        @memcpy(s.q[i * D ..][0..D], src[0..D]);
        @memcpy(s.k[i * D ..][0..D], src[D .. 2 * D]);
        @memcpy(s.v[i * D ..][0..D], src[2 * D .. 3 * D]);
    }
}

fn splitQkv(alloc: Allocator, q: []f32, k: []f32, v: []f32, qkv: []const f32, L: usize) void {
    var job = SplitJob{ .q = q, .k = k, .v = v, .qkv = qkv };
    parallel(alloc, L, &job, splitJob);
}

pub fn forward(alloc: Allocator, f: *FwdCtx, layers: EnableLayers) void {
    const m = f.m;
    const s = f.s;
    const L = f.L;

    // ---- embeddings + norm (layer 0 has no attn_norm; embeddings.norm is folded in)
    diag.g_stage = ST_EMB;
    parallel(alloc, L, f, embedJob);
    diag.g_stage = ST_NORM;
    parallel(alloc, L, f, normJob);
    const dumping = diag.g_dump_stats and diag.g_fwd_count == 0;
    if (dumping) {
        std.debug.print("[dump] meta L={d} qtype={d}\n", .{ L, f.qtype });
        dumpIds(f.ids);
        dumpStats("emb", s.x, L);
    }
    for (0..NL) |l| {
        const lay = &m.layers[l];
        var attn_in: []const f32 = s.x;
        diag.g_stage = ST_NORM;
        if (lay.attn_norm) |an| {
            normInto(alloc, s.x, s.xn, L, an, null);
            attn_in = s.xn;
        }
        diag.g_stage = ST_MATMUL;
        linear16(alloc, s.qkv, attn_in, lay.wqkv, null, L, 3 * D, D);
        diag.g_stage = ST_SPLIT;
        splitQkv(alloc, s.q, s.k, s.v, s.qkv, L);
        diag.g_stage = ST_ROPE;
        ropeApply(alloc, s.q, L, &m.inv_freq);
        ropeApply(alloc, s.k, L, &m.inv_freq);

        diag.g_stage = ST_ATTN;
        var aj = AttJob{
            .q = s.q,
            .k = s.k,
            .v = s.v,
            .out = s.ctx,
            .att = s.att,
            .L = L,
            .stride = if (lay.sliding) 2 * WINDOW + 1 else L,
            .win = if (lay.sliding) WINDOW else null,
        };
        parallel(alloc, L, &aj, attWorker);
        diag.g_stage = ST_MATMUL;
        linear16(alloc, s.xn, s.ctx, lay.wo, null, L, D, D);
        diag.g_stage = ST_RES;
        parallel(alloc, L, f, addResJob);

        diag.g_stage = ST_NORM;
        normInto(alloc, s.x, s.xn, L, lay.mlp_norm, null);
        diag.g_stage = ST_MATMUL;
        linear16(alloc, s.mlp, s.xn, lay.wi, null, L, 2 * INTER, D);
        diag.g_stage = ST_GELU;
        parallel(alloc, L, f, mlpJob);
        diag.g_stage = ST_MATMUL;
        linear16(alloc, s.xn, s.mlp2, lay.wo2, null, L, D, INTER);
        diag.g_stage = ST_RES;
        parallel(alloc, L, f, addResJob);
        if (dumping) {
            var tag: [8]u8 = undefined;
            const t = std.fmt.bufPrint(&tag, "l{d}", .{l}) catch "l?";
            dumpStats(t, s.x, L);
        }
    }

    diag.g_stage = ST_NORM;
    for (0..L) |i| layerNorm(s.x[i * D ..][0..D], m.final_norm, null);
    if (dumping) dumpStats("final", s.x, L);
    parallel(alloc, L, f, typeEmbJob);
    if (dumping) dumpStats("temb", s.x, L);

    if (layers == .layers_full) {
        // ---- decision head: 2 pre-norm transformer layers, bidirectional
        for (0..HEAD_LAYERS) |l| {
            const hl = &m.head[l];
            diag.g_stage = ST_NORM;
            normInto(alloc, s.x, s.xn, L, hl.norm1_w, hl.norm1_b);
            diag.g_stage = ST_MATMUL;
            linear16(alloc, s.qkv, s.xn, hl.in_proj_w, hl.in_proj_b, L, 3 * D, D);
            diag.g_stage = ST_SPLIT;
            splitQkv(alloc, s.q, s.k, s.v, s.qkv, L);
            diag.g_stage = ST_ATTN;
            var aj = AttJob{
                .q = s.q,
                .k = s.k,
                .v = s.v,
                .out = s.ctx,
                .att = s.att,
                .L = L,
                .stride = L,
                .win = null,
            };
            parallel(alloc, L, &aj, attWorker);
            diag.g_stage = ST_MATMUL;
            linear16(alloc, s.xn, s.ctx, hl.out_proj_w, hl.out_proj_b, L, D, D);
            diag.g_stage = ST_RES;
            parallel(alloc, L, f, addResJob);

            diag.g_stage = ST_NORM;
            normInto(alloc, s.x, s.xn, L, hl.norm2_w, hl.norm2_b);
            diag.g_stage = ST_MATMUL;
            linear16(alloc, s.ff, s.xn, hl.lin1_w, hl.lin1_b, L, HEAD_FF, D);
            diag.g_stage = ST_GELU;
            parallel(alloc, L, f, reluJob);
            diag.g_stage = ST_MATMUL;
            linear16(alloc, s.xn, s.ff, hl.lin2_w, hl.lin2_b, L, D, HEAD_FF);
            diag.g_stage = ST_RES;
            parallel(alloc, L, f, addResJob);
            if (dumping) {
                var tag: [8]u8 = undefined;
                const t = std.fmt.bufPrint(&tag, "h{d}", .{l}) catch "h?";
                dumpStats(t, s.x, L);
            }
        }

        @memcpy(f.pooled, s.x[0..D]);
    } else {
        @memcpy(f.pooled, s.x[(f.L - 1) * D ..][0..D]);
    }

    // ---- option markers -> scorer
    const t_ser = if (diag.g_prof) nowNs() else 0;
    const K = f.markers.len;
    for (0..K) |r| {
        var mvec: [D]f32 = undefined;
        @memcpy(&mvec, s.x[@as(usize, f.markers[r]) * D ..][0..D]);
        layerNorm(&mvec, m.scorer0_w, m.scorer0_b);
        var h1: [D]f32 = undefined;
        rowDot(&h1, &mvec, m.scorer1_w);
        for (0..D) |d| h1[d] = gelu(h1[d] + m.scorer1_b[d]);
        f.logits[r] = rowDot1(&h1, m.scorer3_w) + m.scorer3_b[0];
    }

    if (dumping) {
        std.debug.print("[dump] mark", .{});
        for (f.markers) |v| std.debug.print(" {d}", .{v});
        std.debug.print("\n[dump] logit", .{});
        for (f.logits[0..K]) |v| std.debug.print(" {d:.5}", .{v});
        std.debug.print("\n", .{});
    }

    if (diag.g_debug) {
        std.debug.print("[dbg] markers=", .{});
        for (f.markers) |mm| std.debug.print("{d} ", .{mm});
        std.debug.print("\n[dbg] logits=", .{});
        for (f.logits[0..K]) |v| std.debug.print("{d:.5} ", .{v});
        std.debug.print("\n", .{});
    }

    // ---- act / escalate head
    const p = f.feat[0..K];
    @memcpy(p, f.logits[0..K]);
    softmax(p);
    const kk: f32 = @floatFromInt(@max(K, @as(usize, 2)));
    var ent: f32 = 0;
    for (p) |v| {
        if (v > 1e-12) ent -= v * @log(v);
    }
    ent /= @log(kk);
    var t0: f32 = 0;
    var t1: f32 = 0;
    for (p) |v| {
        if (v > t0) {
            t1 = t0;
            t0 = v;
        } else if (v > t1) t1 = v;
    }
    const feats = [4]f32{ t0, t0 - t1, ent, kk / 255.0 };
    @memcpy(f.act_in[0..D], f.pooled);
    @memcpy(f.act_in[D .. D + 4], &feats);
    var h256: [256]f32 = undefined;
    rowDot(&h256, f.act_in, m.act0_w);
    for (0..256) |i| h256[i] = gelu(h256[i] + m.act0_b[i]);
    var a2: [2]f32 = undefined;
    for (0..2) |i| a2[i] = rowDot1(&h256, m.act2_w[i * 256 ..][0..256]) + m.act2_b[i];
    softmax(&a2);
    f.act = a2;
    if (diag.g_prof) profAdd(ST_SERIAL, nowNs() - t_ser);
    diag.g_fwd_count += 1;
}
