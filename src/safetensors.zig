//! Reading tensors out of `model.safetensors`: F16/F32 conversion and typed views.

const std = @import("std");
const Allocator = std.mem.Allocator;

const simd = @import("simd.zig");
const pool = @import("pool.zig");

const V = simd.V;

const parallel = pool.parallel;

pub const TensorInfo = struct {
    dtype: []const u8,
    offset: usize,
    len: usize,
};

pub fn f16at(raw: []const u8, i: usize) f32 {
    const u = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
    return @floatCast(@as(f16, @bitCast(u)));
}

const ConvJob = struct {
    raw: []const u8,
    out: []f32,
};

/// lo..hi are counted in groups of 8 elements. f16 -> f32 is exact, so this is a
/// pure bandwidth operation; doing it a value at a time made the weight load
/// (668 ms of the ~950 ms startup) cost more than a whole forward pass.
fn convF16Job(ctx: *anyopaque, lo: usize, hi: usize) void {
    const j: *ConvJob = @ptrCast(@alignCast(ctx));
    var u = lo;
    while (u < hi) : (u += 1) {
        const o = u * 8;
        const h: @Vector(8, f16) = @bitCast(j.raw[o * 2 ..][0..16].*);
        j.out[o..][0..8].* = @as([8]f32, @bitCast(@as(V, @floatCast(h))));
    }
}

fn toF32(alloc: Allocator, raw: []const u8, dtype: []const u8) ![]f32 {
    if (std.mem.eql(u8, dtype, "F32")) {
        const n = raw.len / 4;
        const out = try alloc.alloc(f32, n);
        for (0..n) |i| out[i] = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        return out;
    }
    if (!std.mem.eql(u8, dtype, "F16")) return error.UnsupportedDtype;
    const n = raw.len / 2;
    const out = try alloc.alloc(f32, n);
    var job = ConvJob{ .raw = raw, .out = out };
    parallel(alloc, n / 8, &job, convF16Job);
    var i = (n / 8) * 8;
    while (i < n) : (i += 1) out[i] = f16at(raw, i);
    return out;
}

pub fn tensorRaw(infos: *const std.StringHashMap(TensorInfo), data: []const u8, name: []const u8) ![]const u8 {
    const t = infos.get(name) orelse {
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    };
    return data[t.offset .. t.offset + t.len];
}

pub fn tensorF32(alloc: Allocator, infos: *const std.StringHashMap(TensorInfo), data: []const u8, name: []const u8) ![]const f32 {
    const t = infos.get(name) orelse {
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    };
    return try toF32(alloc, data[t.offset .. t.offset + t.len], t.dtype);
}

/// A matrix the kernels consume as half precision: a view of the file image, so
/// it costs no memory beyond the mapping itself.
pub fn tensorF16(infos: *const std.StringHashMap(TensorInfo), data: []const u8, name: []const u8) ![]align(1) const f16 {
    const t = infos.get(name) orelse {
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    };
    if (!std.mem.eql(u8, t.dtype, "F16")) {
        std.debug.print("tensor {s} is {s}, expected F16\n", .{ name, t.dtype });
        return error.UnsupportedDtype;
    }
    const raw = data[t.offset .. t.offset + t.len];
    if (raw.len % 2 != 0) {
        std.debug.print("tensor {s} has an odd byte length\n", .{name});
        return error.BadSafetensors;
    }
    return @ptrCast(raw);
}
