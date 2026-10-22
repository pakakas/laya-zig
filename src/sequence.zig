//! Builds the token sequence (and the option-marker positions) the model is fed for
//! one question.

const std = @import("std");
const Allocator = std.mem.Allocator;
const AList = std.array_list.Managed;

const config = @import("config.zig");
const tokenizer = @import("tokenizer.zig");

const CLS_ID = config.CLS_ID;
const SEP_ID = config.SEP_ID;
const MASK_ID = config.MASK_ID;
const MASK_TOKEN = config.MASK_TOKEN;

const Tokenizer = tokenizer.Tokenizer;
const encodeAll = tokenizer.encodeAll;

const EnableLayers = @import("forward.zig").EnableLayers;

fn replaceMask(alloc: Allocator, s: []const u8) ![]const u8 {
    if (std.mem.indexOf(u8, s, MASK_TOKEN) == null) return s;
    var out = AList(u8).init(alloc);
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], MASK_TOKEN)) {
            try out.append(' ');
            i += MASK_TOKEN.len;
        } else {
            try out.append(s[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice();
}

pub const Seq = struct {
    ids: []u32,
    markers: []u32,
    qtype: u32,
};

pub fn buildSequence(alloc: Allocator, tok: *const Tokenizer, state: []const u8, qtype_name: []const u8, ins: []const u8, opts: []const []const u8, max_len: usize, head_max_len: usize, layers: EnableLayers) !Seq {
    var head_ids: []u32 = undefined;

    if (layers == .layers_full) {
        const ins2 = try replaceMask(alloc, ins);
        const head_text = try std.fmt.allocPrint(alloc, "{s} question: {s}", .{ qtype_name, ins2 });
        head_ids = try encodeAll(tok, alloc, head_text);
    }

    var opt_ids = AList([]u32).init(alloc);
    for (opts) |o| {
        const o2 = try replaceMask(alloc, o);
        const t = try std.fmt.allocPrint(alloc, " {s}", .{o2});
        var e = try encodeAll(tok, alloc, t);
        if (e.len > 48) e = e[0..48];
        const full = try alloc.alloc(u32, e.len + 1);
        full[0] = MASK_ID;
        @memcpy(full[1..], e);
        try opt_ids.append(full);
    }

    var total: usize = 0;
    for (opt_ids.items) |o| total += o.len;

    if (layers == .layers_full) {
        var opt_budget: i64 = @as(i64, @intCast(head_max_len)) - @as(i64, @intCast(total));
        if (opt_budget < 16) {
            const per: usize = @max(4, (head_max_len - 16) / @max(1, opt_ids.items.len));
            for (opt_ids.items) |*o| {
                if (o.*.len > per) o.* = o.*[0..per];
            }
            total = 0;
            for (opt_ids.items) |o| total += o.len;
            opt_budget = @as(i64, @intCast(head_max_len)) - @as(i64, @intCast(total));
        }
        const head_keep: usize = @max(8, @as(usize, @intCast(@max(opt_budget, 0))));
        if (head_ids.len > head_keep) head_ids = head_ids[0..head_keep];
    }

    var ids = AList(u32).init(alloc);
    try ids.append(CLS_ID);

    if (layers == .layers_full) {
        try ids.appendSlice(head_ids);
    }

    try ids.append(SEP_ID);
    var markers = AList(u32).init(alloc);
    for (opt_ids.items) |o| {
        try markers.append(@intCast(ids.items.len));
        try ids.appendSlice(o);
    }
    try ids.append(SEP_ID);

    const room: i64 = @as(i64, @intCast(max_len)) - @as(i64, @intCast(ids.items.len)) - 1;
    if (room > 0) {
        const st = try replaceMask(alloc, state);
        var st_ids = try encodeAll(tok, alloc, st);
        if (st_ids.len > room) st_ids = st_ids[0..@intCast(room)];
        try ids.appendSlice(st_ids);
    }
    try ids.append(SEP_ID);

    var final_ids = ids.items;
    if (final_ids.len > max_len) final_ids = final_ids[0..max_len];

    var cnt: usize = 0;
    for (markers.items) |v| {
        if (v < max_len) cnt += 1;
    }
    const mk2 = try alloc.alloc(u32, cnt);
    var c: usize = 0;
    for (markers.items) |v| {
        if (v < max_len) {
            mk2[c] = v;
            c += 1;
        }
    }
    return .{ .ids = final_ids, .markers = mk2, .qtype = 0 };
}
