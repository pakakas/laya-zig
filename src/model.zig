//! The model in memory: weights (a view of `model.safetensors`), the scratch buffers
//! a forward pass works in, and the runtime knobs from `rl_agent_config.json`.

const std = @import("std");
const Allocator = std.mem.Allocator;

const config = @import("config.zig");
const platform = @import("platform.zig");
const safetensors = @import("safetensors.zig");

const D = config.D;
const NL = config.NL;
const INTER = config.INTER;
const ROPE_THETA = config.ROPE_THETA;
const WINDOW = config.WINDOW;
const HEAD_LAYERS = config.HEAD_LAYERS;
const HEAD_FF = config.HEAD_FF;
const MAX_LEN = config.MAX_LEN;
const HEAD_MAX_LEN = config.HEAD_MAX_LEN;

const readFile = platform.readFile;
const mapFile = platform.mapFile;

const TensorInfo = safetensors.TensorInfo;
const tensorRaw = safetensors.tensorRaw;
const tensorF32 = safetensors.tensorF32;
const tensorF16 = safetensors.tensorF16;

pub const EncLayer = struct {
    wqkv: []align(1) const f16,
    wo: []align(1) const f16,
    wi: []align(1) const f16,
    wo2: []align(1) const f16,
    attn_norm: ?[]const f32,
    mlp_norm: []const f32,
    sliding: bool,
};

pub const HeadLayer = struct {
    norm1_w: []const f32,
    norm1_b: []const f32,
    norm2_w: []const f32,
    norm2_b: []const f32,
    in_proj_w: []align(1) const f16,
    in_proj_b: []const f32,
    out_proj_w: []align(1) const f16,
    out_proj_b: []const f32,
    lin1_w: []align(1) const f16,
    lin1_b: []const f32,
    lin2_w: []align(1) const f16,
    lin2_b: []const f32,
};

pub const Model = struct {
    emb_raw: []const u8, // F16 [256000, 768]
    emb_norm: []const f32,
    layers: [NL]EncLayer,
    final_norm: []const f32,
    type_emb: []const f32, // [3, 768]
    head: [HEAD_LAYERS]HeadLayer,
    scorer0_w: []const f32,
    scorer0_b: []const f32,
    scorer1_w: []const f32,
    scorer1_b: []const f32,
    scorer3_w: []const f32,
    scorer3_b: []const f32,
    act0_w: []const f32,
    act0_b: []const f32,
    act2_w: []const f32,
    act2_b: []const f32,
    inv_freq: [32]f32,
    temperature: [3]f32,
};

pub const Scratch = struct {
    x: []f32,
    xn: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    qkv: []f32,
    ctx: []f32,
    mlp: []f32,
    mlp2: []f32,
    att: []f32,
    ff: []f32,
    feat: []f32,
    act_in: []f32,

    /// Buffers for sequences of up to `L` tokens with up to `kmax` option markers.
    pub fn init(gpa: Allocator, L: usize, kmax: usize) !Scratch {
        return .{
            .x = try gpa.alloc(f32, L * D),
            .xn = try gpa.alloc(f32, L * D),
            .q = try gpa.alloc(f32, L * D),
            .k = try gpa.alloc(f32, L * D),
            .v = try gpa.alloc(f32, L * D),
            .qkv = try gpa.alloc(f32, L * 3 * D),
            .ctx = try gpa.alloc(f32, L * D),
            .mlp = try gpa.alloc(f32, L * 2 * INTER),
            .mlp2 = try gpa.alloc(f32, L * INTER),
            // sliding layers use rows of 2*WINDOW+1; full layers use rows of L
            .att = try gpa.alloc(f32, L * @max(L, 2 * WINDOW + 1)),
            .ff = try gpa.alloc(f32, L * HEAD_FF),
            .feat = try gpa.alloc(f32, kmax),
            .act_in = try gpa.alloc(f32, D + 4),
        };
    }
};

pub const Cfg = struct {
    temperature: [3]f32 = .{ 1.0, 1.0, 1.0 },
    max_len: usize = MAX_LEN,
    head_max_len: usize = HEAD_MAX_LEN,
};

/// The weights are a view of `model.safetensors`, not a copy: 614 MB of private
/// commit becomes demand-paged file memory the OS can drop and re-read at will.
/// `m.emb_raw` points into it, so it lives as long as the process.
var g_weights: ?std.Io.File.MemoryMap = null;

pub fn loadModel(alloc: Allocator, io: std.Io, dir: []const u8, cfg: *const Cfg) !struct { Model, []u8 } {
    const path = try std.fs.path.join(alloc, &.{ dir, "model.safetensors" });
    defer alloc.free(path);
    var data: []u8 = undefined;
    if (mapFile(io, path)) |mm| {
        g_weights = mm;
        data = mm.memory;
    } else |_| data = try readFile(alloc, io, path);
    // Only the fallback copy is ours to free; unmapping a view with
    // alloc.free would hand the allocator a pointer it never handed out.
    errdefer if (g_weights == null) alloc.free(data);

    if (data.len < 8) return error.BadSafetensors;
    const hdr_len = std.mem.readInt(u64, data[0..8], .little);
    if (hdr_len + 8 > data.len) return error.BadSafetensors;
    const base: usize = 8 + hdr_len;

    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, data[8 .. 8 + hdr_len], .{});
    defer parsed.deinit();

    var infos = std.StringHashMap(TensorInfo).init(alloc);
    defer infos.deinit();
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        const o = e.value_ptr.object;
        const offs = o.get("data_offsets").?.array.items;
        const s: usize = @intCast(offs[0].integer);
        const en: usize = @intCast(offs[1].integer);
        try infos.put(e.key_ptr.*, .{ .dtype = o.get("dtype").?.string, .offset = base + s, .len = en - s });
    }

    var m: Model = undefined;
    m.emb_raw = try tensorRaw(&infos, data, "encoder.embeddings.tok_embeddings.weight");
    m.emb_norm = try tensorF32(alloc, &infos, data, "encoder.embeddings.norm.weight");
    m.final_norm = try tensorF32(alloc, &infos, data, "encoder.final_norm.weight");
    m.type_emb = try tensorF32(alloc, &infos, data, "type_emb.weight");

    // Two buffers, not one: `pre` must not point into the buffer being written, or
    // bufPrint copies a slice onto itself -- harmless under ReleaseFast, but @memcpy
    // asserts on overlap in Debug/Safe and panics at startup.
    var pre: [24]u8 = undefined;
    var nm: [160]u8 = undefined;
    for (0..NL) |l| {
        const pfx = try std.fmt.bufPrint(&pre, "encoder.layers.{d}.", .{l});
        var an: ?[]const f32 = null;
        if (l > 0) an = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}attn_norm.weight", .{pfx}));
        m.layers[l] = .{
            .wqkv = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}attn.Wqkv.weight", .{pfx})),
            .wo = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}attn.Wo.weight", .{pfx})),
            .wi = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}mlp.Wi.weight", .{pfx})),
            .wo2 = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}mlp.Wo.weight", .{pfx})),
            .attn_norm = an,
            .mlp_norm = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}mlp_norm.weight", .{pfx})),
            // config: layer_types[i] == "full_attention" iff i % global_attn_every_n_layers == 0 (3)
            .sliding = (l % 3) != 0,
        };
    }

    for (0..HEAD_LAYERS) |l| {
        const pfx = try std.fmt.bufPrint(&pre, "head.layers.{d}.", .{l});
        m.head[l] = .{
            .norm1_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm1.weight", .{pfx})),
            .norm1_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm1.bias", .{pfx})),
            .norm2_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm2.weight", .{pfx})),
            .norm2_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm2.bias", .{pfx})),
            .in_proj_w = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.in_proj_weight", .{pfx})),
            .in_proj_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.in_proj_bias", .{pfx})),
            .out_proj_w = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.out_proj.weight", .{pfx})),
            .out_proj_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.out_proj.bias", .{pfx})),
            .lin1_w = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}linear1.weight", .{pfx})),
            .lin1_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}linear1.bias", .{pfx})),
            .lin2_w = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}linear2.weight", .{pfx})),
            .lin2_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}linear2.bias", .{pfx})),
        };
    }

    m.scorer0_w = try tensorF32(alloc, &infos, data, "scorer.0.weight");
    m.scorer0_b = try tensorF32(alloc, &infos, data, "scorer.0.bias");
    m.scorer1_w = try tensorF32(alloc, &infos, data, "scorer.1.weight");
    m.scorer1_b = try tensorF32(alloc, &infos, data, "scorer.1.bias");
    m.scorer3_w = try tensorF32(alloc, &infos, data, "scorer.3.weight");
    m.scorer3_b = try tensorF32(alloc, &infos, data, "scorer.3.bias");
    m.act0_w = try tensorF32(alloc, &infos, data, "act_head.0.weight");
    m.act0_b = try tensorF32(alloc, &infos, data, "act_head.0.bias");
    m.act2_w = try tensorF32(alloc, &infos, data, "act_head.2.weight");
    m.act2_b = try tensorF32(alloc, &infos, data, "act_head.2.bias");

    for (0..32) |j| {
        m.inv_freq[j] = 1.0 / std.math.pow(f32, ROPE_THETA, @as(f32, @floatFromInt(j)) / 32.0);
    }
    m.temperature = cfg.temperature;
    return .{ m, data };
}
