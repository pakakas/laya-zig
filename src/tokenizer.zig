//! Byte-pair-encoding tokenizer loaded from tokenizer.json (vocab, merges, added
//! tokens), plus the `--tokcheck` validator.

const std = @import("std");
const Allocator = std.mem.Allocator;
const AList = std.array_list.Managed;

const config = @import("config.zig");
const platform = @import("platform.zig");
const jscan = @import("jscan.zig");

const SEP_CHAR = config.SEP_CHAR;

const readFile = platform.readFile;

const JScan = jscan.JScan;

pub const Tokenizer = struct {
    vocab: std.StringHashMap(u32),
    merges: std.StringHashMap(u32),
    added: std.StringHashMap(u32),
    added_lens: []usize, // distinct lengths, descending
    start_byte: [256]bool,
    byte_ids: [256]u32,

    fn findAdded(self: *const Tokenizer, s: []const u8) ?struct { usize, u32 } {
        for (self.added_lens) |len| {
            if (len <= s.len) {
                if (self.added.get(s[0..len])) |id| return .{ len, id };
            }
        }
        return null;
    }
};

const TokSink = struct {
    tok: *Tokenizer,
    alloc: Allocator,
};

fn walkValue(sc: *JScan, sink: *TokSink) !void {
    sc.ws();
    switch (try sc.peek()) {
        '{' => {
            _ = try sc.byte();
            while (true) {
                sc.ws();
                if (try sc.peek() == '}') {
                    _ = try sc.byte();
                    return;
                }
                if (try sc.peek() == ',') {
                    _ = try sc.byte();
                    continue;
                }
                var kbuf: [128]u8 = undefined;
                const rawlen = try sc.rawStringLen();
                if (rawlen > kbuf.len) {
                    try sc.skipString();
                    try sc.expect(':');
                    try walkValue(sc, sink);
                    continue;
                }
                const key = try sc.stringInto(&kbuf);
                try sc.expect(':');
                if (std.mem.eql(u8, key, "vocab")) {
                    try parseVocab(sc, sink);
                } else if (std.mem.eql(u8, key, "merges")) {
                    try parseMerges(sc, sink);
                } else if (std.mem.eql(u8, key, "added_tokens")) {
                    try parseAdded(sc, sink);
                } else {
                    try walkValue(sc, sink);
                }
            }
        },
        '[' => {
            _ = try sc.byte();
            while (true) {
                sc.ws();
                if (try sc.peek() == ']') {
                    _ = try sc.byte();
                    return;
                }
                if (try sc.peek() == ',') {
                    _ = try sc.byte();
                    continue;
                }
                try walkValue(sc, sink);
            }
        },
        '"' => try sc.skipString(),
        else => try sc.skipLit(),
    }
}

fn parseVocab(sc: *JScan, sink: *TokSink) !void {
    try sc.expect('{');
    const buf = try sink.alloc.alloc(u8, 1 << 20);
    defer sink.alloc.free(buf);
    while (true) {
        sc.ws();
        if (try sc.peek() == '}') {
            _ = try sc.byte();
            return;
        }
        if (try sc.peek() == ',') {
            _ = try sc.byte();
            continue;
        }
        const s = try sc.stringInto(buf);
        try sc.expect(':');
        const id = try sc.uint();
        try sink.tok.vocab.put(try sink.alloc.dupe(u8, s), id);
    }
}

fn parseMerges(sc: *JScan, sink: *TokSink) !void {
    try sc.expect('[');
    var buf = try sink.alloc.alloc(u8, 1 << 18);
    defer sink.alloc.free(buf);
    var rank: u32 = 0;
    while (true) {
        sc.ws();
        if (try sc.peek() == ']') {
            _ = try sc.byte();
            return;
        }
        if (try sc.peek() == ',') {
            _ = try sc.byte();
            continue;
        }
        try sc.expect('[');
        const a = try sc.stringInto(buf);
        sc.ws();
        _ = try sc.byte(); // ','
        const b = try sc.stringInto(buf[a.len..]);
        try sc.expect(']');
        const key = try std.fmt.allocPrint(sink.alloc, "{s}\x00{s}", .{ a, b });
        try sink.tok.merges.put(key, rank);
        rank += 1;
    }
}

fn parseAdded(sc: *JScan, sink: *TokSink) !void {
    try sc.expect('[');
    const buf = try sink.alloc.alloc(u8, 1 << 16);
    defer sink.alloc.free(buf);
    while (true) {
        sc.ws();
        if (try sc.peek() == ']') {
            _ = try sc.byte();
            return;
        }
        if (try sc.peek() == ',') {
            _ = try sc.byte();
            continue;
        }
        try sc.expect('{');
        var id: u32 = 0;
        var content: ?[]const u8 = null;
        while (true) {
            sc.ws();
            if (try sc.peek() == '}') {
                _ = try sc.byte();
                break;
            }
            if (try sc.peek() == ',') {
                _ = try sc.byte();
                continue;
            }
            var kbuf: [64]u8 = undefined;
            const key = try sc.stringInto(&kbuf);
            try sc.expect(':');
            if (std.mem.eql(u8, key, "id")) {
                id = try sc.uint();
            } else if (std.mem.eql(u8, key, "content")) {
                content = try sc.stringInto(buf);
            } else if (try sc.peek() == '"') {
                try sc.skipString();
            } else {
                try sc.skipLit();
            }
        }
        if (content) |c| try sink.tok.added.put(try sink.alloc.dupe(u8, c), id);
    }
}

pub fn loadTokenizer(alloc: Allocator, io: std.Io, path: []const u8) !Tokenizer {
    const data = try readFile(alloc, io, path);
    defer alloc.free(data);

    var tok = Tokenizer{
        .vocab = std.StringHashMap(u32).init(alloc),
        .merges = std.StringHashMap(u32).init(alloc),
        .added = std.StringHashMap(u32).init(alloc),
        .added_lens = &.{},
        .start_byte = undefined,
        .byte_ids = undefined,
    };
    @memset(tok.start_byte[0..], false);
    @memset(tok.byte_ids[0..], 0);
    try tok.vocab.ensureTotalCapacity(300000);
    try tok.merges.ensureTotalCapacity(700000);

    var sc = JScan{ .s = data, .i = 0 };
    var sink = TokSink{ .tok = &tok, .alloc = alloc };
    walkValue(&sc, &sink) catch |e| {
        const lo = if (sc.i > 120) sc.i - 120 else 0;
        const hi = @min(data.len, sc.i + 120);
        std.debug.print("tokenizer.json parse error {} at offset {d}\n...{s}...\n", .{ e, sc.i, data[lo..hi] });
        return e;
    };

    var lens = AList(usize).init(alloc);
    var it = tok.added.keyIterator();
    while (it.next()) |k| {
        const l = k.*.len;
        var seen = false;
        for (lens.items) |x| {
            if (x == l) seen = true;
        }
        if (!seen) try lens.append(l);
        tok.start_byte[k.*[0]] = true;
    }
    std.mem.sort(usize, lens.items, {}, struct {
        fn lt(_: void, a: usize, b: usize) bool {
            return a > b;
        }
    }.lt);
    tok.added_lens = try lens.toOwnedSlice();

    const hexdig = "0123456789ABCDEF";
    var hb: [6]u8 = undefined;
    for (0..256) |b| {
        hb[0] = '<';
        hb[1] = '0';
        hb[2] = 'x';
        hb[3] = hexdig[b >> 4];
        hb[4] = hexdig[b & 15];
        hb[5] = '>';
        tok.byte_ids[b] = tok.vocab.get(&hb) orelse 3;
    }
    return tok;
}

fn bpePiece(tok: *const Tokenizer, alloc: Allocator, piece: []const u8, out: *AList(u32)) !void {
    if (piece.len == 0) return;
    var syms = AList([2]usize).init(alloc);
    defer syms.deinit();
    var i: usize = 0;
    while (i < piece.len) {
        const n = std.unicode.utf8ByteSequenceLength(piece[i]) catch 1;
        const len = @min(n, piece.len - i);
        try syms.append(.{ i, i + len });
        i += len;
    }
    var keybuf: [1024]u8 = undefined;
    while (syms.items.len > 1) {
        var best: u32 = std.math.maxInt(u32);
        var bi: usize = 0;
        var found = false;
        for (0..syms.items.len - 1) |s| {
            const a = syms.items[s];
            const b = syms.items[s + 1];
            const la = a[1] - a[0];
            const lb = b[1] - b[0];
            if (la + lb + 1 > keybuf.len) continue;
            @memcpy(keybuf[0..la], piece[a[0]..a[1]]);
            keybuf[la] = 0;
            @memcpy(keybuf[la + 1 ..][0..lb], piece[b[0]..b[1]]);
            if (tok.merges.get(keybuf[0 .. la + lb + 1])) |r| {
                if (!found or r < best) {
                    best = r;
                    bi = s;
                    found = true;
                }
            }
        }
        if (!found) break;
        syms.items[bi][1] = syms.items[bi + 1][1];
        _ = syms.orderedRemove(bi + 1);
    }
    for (syms.items) |s| {
        const str = piece[s[0]..s[1]];
        if (tok.vocab.get(str)) |id| {
            try out.append(id);
        } else {
            for (str) |b| try out.append(tok.byte_ids[b]);
        }
    }
}

fn isSepAt(t: []const u8, i: usize) bool {
    return i + 3 <= t.len and std.mem.startsWith(u8, t[i..], SEP_CHAR);
}

fn encodeSegment(tok: *const Tokenizer, alloc: Allocator, s: []const u8, out: *AList(u32)) !void {
    var norm = AList(u8).init(alloc);
    defer norm.deinit();
    try norm.ensureTotalCapacity(s.len + 3);
    for (s) |c| {
        if (c == ' ') try norm.appendSlice(SEP_CHAR) else try norm.append(c);
    }
    const t = norm.items;
    if (t.len == 0) return;

    var scratch: ?[]u8 = null;
    defer if (scratch) |b| alloc.free(b);

    var j: usize = 0;
    if (!isSepAt(t, 0)) {
        while (j < t.len and !isSepAt(t, j)) j += 1;
        scratch = try alloc.alloc(u8, j + 3);
        @memcpy(scratch.?[0..3], SEP_CHAR);
        @memcpy(scratch.?[3..], t[0..j]);
        try bpePiece(tok, alloc, scratch.?, out);
    }
    while (j < t.len) {
        var e = j + 3;
        while (e < t.len and !isSepAt(t, e)) e += 1;
        try bpePiece(tok, alloc, t[j..e], out);
        j = e;
    }
}

const EncCtx = struct {
    tok: *const Tokenizer,
    alloc: Allocator,
    out: *AList(u32),
};

pub fn encodeAll(tok: *const Tokenizer, alloc: Allocator, text: []const u8) ![]u32 {
    var out = AList(u32).init(alloc);
    var ctx = EncCtx{ .tok = tok, .alloc = alloc, .out = &out };
    var i: usize = 0;
    var plain_start: usize = 0;
    while (i < text.len) {
        if (ctx.tok.start_byte[text[i]]) {
            if (ctx.tok.findAdded(text[i..])) |m| {
                const len: usize = m[0];
                if (i > plain_start) try encodeSegment(ctx.tok, ctx.alloc, text[plain_start..i], ctx.out);
                try ctx.out.append(m[1]);
                i += len;
                plain_start = i;
                continue;
            }
        }
        i += 1;
    }
    if (text.len > plain_start) try encodeSegment(ctx.tok, ctx.alloc, text[plain_start..], ctx.out);
    return out.toOwnedSlice();
}

pub fn tokCheck(gpa: Allocator, io: std.Io, tok: *const Tokenizer, path: []const u8) !void {
    const data = try readFile(gpa, io, path);
    defer gpa.free(data);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();
    var pass: usize = 0;
    var fail: usize = 0;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |ln| {
        if (ln.len < 3) continue;
        var p = try std.json.parseFromSlice(std.json.Value, aa, ln, .{});
        defer p.deinit();
        const text = p.value.object.get("text").?.string;
        const exp = p.value.object.get("ids").?.array.items;
        const got = try encodeAll(tok, aa, text);
        var ok = got.len == exp.len;
        if (ok) {
            for (got, 0..) |g, i| {
                if (g != @as(u32, @intCast(exp[i].integer))) ok = false;
            }
        }
        if (ok) {
            pass += 1;
        } else {
            fail += 1;
            std.debug.print("MISMATCH {s}\n  got  {any}\n  want ", .{ text, got });
            for (exp) |e| std.debug.print("{d} ", .{e.integer});
            std.debug.print("\n", .{});
        }
    }
    std.debug.print("tokcheck: {d} pass, {d} fail\n", .{ pass, fail });
}
