//! 8-wide f32 SIMD primitives: dot products, exp and gelu.
//!
//! Read the NOTEs below before touching any of this: a zig build has miscompiled
//! these kernels before. Re-validate with `laya --selftest` after every change.

pub const V = @Vector(8, f32);

pub inline fn v8(a: []align(1) const f32, o: usize) V {
    return @bitCast(a[o..][0..8].*);
}

/// The 8 lanes hold partial sums for 8 k-elements, so the block kernel reduces
/// once per output instead of once per dot product.
pub inline fn hsum(v: V) f32 {
    return @reduce(.Add, v);
}

pub inline fn splatv(comptime k: f32) V {
    return @splat(k);
}

/// Eight weight elements as f32 lanes. F16 -> F32 is a widening with nothing to
/// round, so half-precision storage changes neither the products nor the
/// summation order: both instantiations of the kernel agree bit for bit, which is
/// what `--selftest` and `tools/refcheck.py` check.
///
/// The slice has to stay typed. Reading the same bytes out of a `[]const u8` with
/// a `@bitCast` load costs 8-12x here -- this form compiles to one `vcvtph2ps`
/// per eight weights.
inline fn wvec(comptime T: type, w: []align(1) const T, t: usize) V {
    const h: @Vector(8, T) = @bitCast(w[t..][0..8].*);
    return @as(V, @floatCast(h));
}

inline fn wat(comptime T: type, w: []align(1) const T, t: usize) f32 {
    return @floatCast(w[t]);
}

/// out[j] = dot(x, w[j*k .. (j+1)*k])   (w is [n][k] row-major)
/// NOTE: kept as a single 8-wide accumulator on purpose. A 4-accumulator / 32-wide
/// unrolled version is miscompiled by this zig build (0.17.0-dev.1737) and silently
/// returns wrong results, so do not "optimise" it back without re-validating.
/// dot product of two equal-length slices.
///
/// NOTE: an earlier 4-accumulator / 32-wide-unrolled variant of this function was
/// silently miscompiled by zig 0.17.0-dev.1737 (results ~40% off). Any change here
/// must be re-validated with `laya.exe --selftest`, which checks this kernel against
/// an f64 reference for every shape the model uses.
fn dotv(a: []const f32, b: []const f32) f32 {
    const n = @min(a.len, b.len);
    var a0: @Vector(8, f32) = @splat(0.0);
    var a1: @Vector(8, f32) = @splat(0.0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const x0: @Vector(8, f32) = @bitCast(a[i..][0..8].*);
        const x1: @Vector(8, f32) = @bitCast(a[i + 8 ..][0..8].*);
        const y0: @Vector(8, f32) = @bitCast(b[i..][0..8].*);
        const y1: @Vector(8, f32) = @bitCast(b[i + 8 ..][0..8].*);
        a0 = @mulAdd(V, x0, y0, a0);
        a1 = @mulAdd(V, x1, y1, a1);
    }
    while (i + 8 <= n) : (i += 8) {
        const x0: @Vector(8, f32) = @bitCast(a[i..][0..8].*);
        const y0: @Vector(8, f32) = @bitCast(b[i..][0..8].*);
        a0 = @mulAdd(V, x0, y0, a0);
    }
    var s: f32 = 0;
    while (i < n) : (i += 1) s += a[i] * b[i];
    return s + @reduce(.Add, a0 + a1);
}

/// `dotv` over a weight row; same accumulation order, so f32 and f16 weights give
/// identical results.
pub fn dotvw(comptime T: type, a: []const f32, b: []align(1) const T) f32 {
    const n = a.len;
    var a0: V = @splat(0);
    var a1: V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        a0 = @mulAdd(V, v8(a, i), wvec(T, b, i), a0);
        a1 = @mulAdd(V, v8(a, i + 8), wvec(T, b, i + 8), a1);
    }
    while (i + 8 <= n) : (i += 8) a0 = @mulAdd(V, v8(a, i), wvec(T, b, i), a0);
    var s: f32 = 0;
    while (i < n) : (i += 1) s += a[i] * wat(T, b, i);
    return s + @reduce(.Add, a0 + a1);
}

/// Elementwise exp for 8 lanes. Range reduction is on the binary exponent, so the
/// polynomial only has to cover |r| <= ln2/2 and 7th order is good to ~5e-9
/// relative. `--selftest` checks both this and `geluv` against f64.
pub inline fn expv(x: V) V {
    const clamped = @max(@min(x, splatv(88.0)), splatv(-87.0));
    const n = @round(clamped * splatv(1.4426950408889634));
    const ni: @Vector(8, i32) = @intFromFloat(n);
    const scale: V = @bitCast((ni + @as(@Vector(8, i32), @splat(127))) *%
        @as(@Vector(8, i32), @splat(0x800000)));
    const r = clamped - n * splatv(0.6931471805599453);
    var p: V = splatv(1.0 / 5040.0);
    p = @mulAdd(V, p, r, splatv(1.0 / 720.0));
    p = @mulAdd(V, p, r, splatv(1.0 / 120.0));
    p = @mulAdd(V, p, r, splatv(1.0 / 24.0));
    p = @mulAdd(V, p, r, splatv(1.0 / 6.0));
    p = @mulAdd(V, p, r, splatv(0.5));
    p = @mulAdd(V, p, r, splatv(1.0));
    p = @mulAdd(V, p, r, splatv(1.0));
    return p * scale;
}

/// gelu over 8 lanes. Written as 0.5*(x + |x|*erf(|x|/sqrt2)) so the sign of x never
/// has to be selected — that form is what the scalar `gelu` (math.zig) evaluates to,
/// and it collapses to the same value once erf saturates at 1.
pub inline fn geluv(x: V) V {
    const ax = @abs(x);
    const a = ax * splatv(0.7071067811865476);
    const t: V = splatv(1.0) / (splatv(1.0) + a * splatv(0.3275911));
    var p: V = splatv(1.061405429);
    p = @mulAdd(V, p, t, splatv(-1.453152027));
    p = @mulAdd(V, p, t, splatv(1.421413741));
    p = @mulAdd(V, p, t, splatv(-0.284496736));
    p = @mulAdd(V, p, t, splatv(0.254829592));
    const e = splatv(1.0) - (p * t) * expv(splatv(0.0) - a * a);
    return splatv(0.5) * (x + ax * e);
}

/// single input row against n weight rows
pub fn rowDot(out: []f32, x: []const f32, w: []const f32) void {
    for (0..out.len) |j| out[j] = dotv(x, w[j * x.len ..][0..x.len]);
}

pub fn rowDot1(a: []const f32, b: []const f32) f32 {
    var o: [1]f32 = undefined;
    rowDot(&o, a, b);
    return o[0];
}
