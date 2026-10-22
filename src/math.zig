//! Scalar math shared by the encoder, the decision head and the option scorer.

const std = @import("std");

const config = @import("config.zig");

const EPS = config.EPS;

fn erf(x: f32) f32 {
    const s: f32 = if (x < 0) -1.0 else 1.0;
    const a = @abs(x);
    const t: f32 = 1.0 / (1.0 + 0.3275911 * a);
    const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - poly * @exp(-a * a));
}

pub fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + erf(x * 0.70710678118654752440));
}

pub fn softmax(x: []f32) void {
    var mx: f32 = -std.math.inf(f32);
    for (x) |v| mx = @max(mx, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - mx);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

pub fn layerNorm(x: []f32, w: []const f32, b: ?[]const f32) void {
    const n: f32 = @floatFromInt(x.len);
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= n;
    var varr: f32 = 0;
    for (x) |v| {
        const d = v - mean;
        varr += d * d;
    }
    varr = 1.0 / @sqrt(varr / n + EPS);
    for (x, 0..) |*v, i| v.* = (v.* - mean) * varr * w[i] + if (b) |bb| bb[i] else 0.0;
}

pub fn confidenceFromProbs(p: []const f32, k: usize) f32 {
    if (k < 2) return 1.0;
    var ent: f32 = 0;
    for (p[0..k]) |v| {
        if (v > 1e-12) ent -= v * @log(v);
    }
    return @min(1.0, @max(0.0, 1.0 - ent / @log(@as(f32, @floatFromInt(k)))));
}
