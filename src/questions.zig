//! The default CLI mode: `{"state": ..., "questions": {...}}` in, one forward pass
//! per question, probabilities out.
//!
//! q.json: {"state": "...", "questions": {"id": {"type": "choice"|"score"|"noul",
//!         "instructions": "...", "criteria": {...}|[...]}}}

const std = @import("std");
const Allocator = std.mem.Allocator;
const AList = std.array_list.Managed;

const config = @import("config.zig");
const math = @import("math.zig");
const platform = @import("platform.zig");
const diag = @import("diag.zig");
const pool = @import("pool.zig");
const tokenizer = @import("tokenizer.zig");
const model = @import("model.zig");
const sequence = @import("sequence.zig");

const D = config.D;

const confidenceFromProbs = math.confidenceFromProbs;

const nowMs = platform.nowMs;
const readFile = platform.readFile;

const profReset = diag.profReset;
const profReport = diag.profReport;

const Tokenizer = tokenizer.Tokenizer;

const Model = model.Model;
const Scratch = model.Scratch;
const Cfg = model.Cfg;

const FwdCtx = @import("forward.zig").FwdCtx;
const forward = @import("forward.zig").forward;

const Seq = sequence.Seq;
const buildSequence = sequence.buildSequence;

fn qtypeId(name: []const u8) !u32 {
    if (std.mem.eql(u8, name, "choice")) return 0;
    if (std.mem.eql(u8, name, "score")) return 1;
    if (std.mem.eql(u8, name, "noul")) return 2;
    return error.BadQuestionType;
}

const Question = struct {
    name: []const u8,
    qtype: u32,
    type_name: []const u8,
    ins: []const u8,
    opts: [][]const u8,
    keys: [][]const u8,
};

fn renderCriterion(alloc: Allocator, v: std.json.Value) ![]const u8 {
    return switch (v) {
        .string => |s| s,
        .null => "",
        else => try std.json.Stringify.valueAlloc(alloc, v, .{}),
    };
}

fn loadQuestions(alloc: Allocator, root: std.json.Value) ![]Question {
    const qs = root.object.get("questions").?.object;
    var list = AList(Question).init(alloc);
    var it = qs.iterator();
    while (it.next()) |e| {
        const q = e.value_ptr.object;
        const tn = q.get("type").?.string;
        const qt = try qtypeId(tn);
        const ins_v = q.get("instructions").?;
        const ins: []const u8 = if (ins_v == .string) ins_v.string else try std.json.Stringify.valueAlloc(alloc, ins_v, .{});

        var opts = AList([]const u8).init(alloc);
        var keys = AList([]const u8).init(alloc);
        const crit = q.get("criteria");
        if (qt == 0) {
            if (crit) |c| {
                if (c == .object) {
                    var ci = c.object.iterator();
                    while (ci.next()) |ce| {
                        const k = ce.key_ptr.*;
                        const v = ce.value_ptr.*;
                        const plain = (v == .null) or (v == .string and v.string.len == 0);
                        try keys.append(k);
                        try opts.append(if (plain) k else try std.fmt.allocPrint(alloc, "{s}: {s}", .{ k, try renderCriterion(alloc, v) }));
                    }
                } else if (c == .array) {
                    for (c.array.items) |v| {
                        try keys.append(v.string);
                        try opts.append(v.string);
                    }
                }
            }
        } else if (qt == 1) {
            if (crit) |c| {
                for (c.array.items, 0..) |v, idx| {
                    try keys.append(v.string);
                    try opts.append(try std.fmt.allocPrint(alloc, "level {d}: {s}", .{ idx, try renderCriterion(alloc, v) }));
                }
            }
        } else {
            var fc: []const u8 = "no, the statement does not hold";
            var tc: []const u8 = "yes, the statement holds";
            if (crit) |c| {
                if (c == .object) {
                    if (c.object.get("false")) |v| {
                        if (!(v == .null) and !(v == .string and v.string.len == 0)) fc = try renderCriterion(alloc, v);
                    }
                    if (c.object.get("true")) |v| {
                        if (!(v == .null) and !(v == .string and v.string.len == 0)) tc = try renderCriterion(alloc, v);
                    }
                }
            }
            try keys.append("false");
            try keys.append("true");
            try opts.append(try std.fmt.allocPrint(alloc, "false: {s}", .{fc}));
            try opts.append(try std.fmt.allocPrint(alloc, "true: {s}", .{tc}));
        }
        try list.append(.{
            .name = e.key_ptr.*,
            .qtype = qt,
            .type_name = tn,
            .ins = ins,
            .opts = try opts.toOwnedSlice(),
            .keys = try keys.toOwnedSlice(),
        });
    }
    return list.toOwnedSlice();
}

const DEMO_JSON =
    \\{"state":"मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।",
    \\ "questions":{
    \\  "department":{"type":"choice","instructions":"Which team should handle `body`?",
    \\                "criteria":{"billing":"invoices, payments, refunds","technical":"bugs and outages","sales":"pricing"}},
    \\  "refund_requested":{"type":"noul","instructions":"Does the sender ask for money back?"}}}
;

/// Reads the state + questions from `json_path` (or the built-in demo), runs one
/// forward pass per question and prints the answers. `t0` / `t2` are the clock
/// readings main took at startup and after the weights were loaded.
pub fn run(gpa: Allocator, io: std.Io, w: *std.Io.Writer, m: *const Model, tok: *const Tokenizer, cfg: *const Cfg, json_path: ?[]const u8, t0: i64, t2: i64) !void {
    // ---- questions
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();
    const json_text: []const u8 = if (json_path) |p| try readFile(aa, io, p) else DEMO_JSON;
    var parsed_q = try std.json.parseFromSlice(std.json.Value, aa, json_text, .{});
    const questions = try loadQuestions(aa, parsed_q.value);
    // A non-string state is JSON-serialized, matching upstream's serialize_state().
    // It used to fall back to "" here, so the upstream quickstart -- which passes
    // {"body": "..."} -- silently handed the model an empty state.
    const state: []const u8 = if (parsed_q.value.object.get("state")) |sv|
        (if (sv == .string) sv.string else try std.json.Stringify.valueAlloc(aa, sv, .{}))
    else
        "";

    try w.print("state: {s}\n", .{state});
    try w.flush();

    // ---- sequences
    const seqs = try aa.alloc(Seq, questions.len);
    var Lmax: usize = 1;
    var Kmax: usize = 1;
    for (questions, 0..) |q, qi| {
        seqs[qi] = try buildSequence(aa, tok, state, q.type_name, q.ins, q.opts, cfg.max_len, cfg.head_max_len, .no_head);
        seqs[qi].qtype = q.qtype;
        Lmax = @max(Lmax, seqs[qi].ids.len);
        Kmax = @max(Kmax, seqs[qi].markers.len);
    }

    if (diag.g_debug) {
        try w.print("[dbg] seq0 ids=", .{});
        for (seqs[0].ids) |t| try w.print("{d} ", .{t});
        try w.print("\n[dbg] seq0 markers=", .{});
        for (seqs[0].markers) |mm| try w.print("{d} ", .{mm});
        try w.print("\n", .{});
        try w.flush();
    }

    var s = try Scratch.init(gpa, Lmax, Kmax);
    const pooled = try gpa.alloc(f32, D);
    const logits = try gpa.alloc(f32, Kmax);

    // ---- forward, one question per pass
    profReset();
    for (questions, 0..) |q, qi| {
        const sq = seqs[qi];
        var f = FwdCtx{
            .m = m,
            .s = &s,
            .ids = sq.ids,
            .L = sq.ids.len,
            .qtype = sq.qtype,
            .markers = sq.markers,
            .logits = logits,
            .act = .{ 0, 0 },
            .pooled = pooled,
            .feat = s.feat,
            .act_in = s.act_in,
        };
        forward(gpa, &f, .no_head);

        const k = sq.markers.len;
        const temp = @max(1e-3, m.temperature[q.qtype]);
        const p = try gpa.alloc(f32, k);
        defer gpa.free(p);
        var mx: f32 = -std.math.inf(f32);
        for (0..k) |r| {
            p[r] = f.logits[r] / temp;
            mx = @max(mx, p[r]);
        }
        var sum: f32 = 0;
        for (p) |*v| {
            v.* = @exp(v.* - mx);
            sum += v.*;
        }
        for (p) |*v| v.* /= sum;

        try w.print("\n[{s}] ({s})\n", .{ q.name, q.type_name });
        if (q.qtype == 2) {
            try w.print("  noul        {d:.4}\n", .{p[1]});
            try w.print("  confidence  {d:.4}\n", .{@max(p[1], 1.0 - p[1])});
        } else {
            var best: usize = 0;
            for (0..k) |r| {
                if (p[r] > p[best]) best = r;
            }
            if (q.qtype == 0) {
                try w.print("  choice      {s}\n", .{q.keys[best]});
            } else {
                var exp_score: f32 = 0;
                for (0..k) |r| exp_score += @as(f32, @floatFromInt(r)) * p[r];
                try w.print("  score       {d:.4}\n", .{exp_score});
            }
            for (0..k) |r| try w.print("    {s:<24} {d:.4}\n", .{ q.keys[r], p[r] });
            try w.print("  confidence  {d:.4}\n", .{confidenceFromProbs(p, k)});
        }
        try w.print("  act_prob    {d:.4}\n", .{f.act[0]});
        try w.print("  tokens      {d}\n", .{sq.ids.len});
        try w.flush();
    }

    const t3 = nowMs(io);
    try profReport(w, pool.g_threads);
    try w.print("\nforward {d} ms | total {d} ms\n", .{ t3 - t2, t3 - t0 });
    try w.flush();
}
