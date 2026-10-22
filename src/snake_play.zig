//! Wires the model into src/snake.zig: the game asks for a direction, we hand it the
//! board text as the `state` of a 4-way `choice` question and take the argmax.
//! `--policy greedy|random` uses the baselines built into snake.zig instead.

const std = @import("std");
const Allocator = std.mem.Allocator;

const snake = @import("snake.zig");

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
const enableAnsi = platform.enableAnsi;

const profReset = diag.profReset;
const profReport = diag.profReport;

const Tokenizer = tokenizer.Tokenizer;

const Model = model.Model;
const Scratch = model.Scratch;
const Cfg = model.Cfg;

const FwdCtx = @import("forward.zig").FwdCtx;
const forward = @import("forward.zig").forward;

const buildSequence = sequence.buildSequence;

pub const Policy = enum { model, greedy, random, tiers, cycle };

const SNAKE_INS = "Which move keeps the snake alive and reaches the food?";

/// What the model last decided: shown on the status line and by the web UI.
pub const LastMove = struct {
    dir: snake.Dir,
    probs: [4]f32,
    conf: f32,
    act: f32,
    tokens: usize,
    ms: i64,
};

/// Model plus scratch buffers, kept alive across moves so a game never reloads 614 MB.
pub const ModelChooser = struct {
    gpa: Allocator,
    m: *const Model,
    tok: *const Tokenizer,
    cfg: *const Cfg,
    s: Scratch,
    logits: []f32,
    pooled: []f32,
    ms: i64 = 0,
    last: LastMove = .{ .dir = .up, .probs = .{ 0.25, 0.25, 0.25, 0.25 }, .conf = 0, .act = 0, .tokens = 0, .ms = 0 },

    pub fn init(gpa: Allocator, m: *const Model, tok: *const Tokenizer, cfg: *const Cfg) !ModelChooser {
        const L = cfg.max_len;
        return .{
            .gpa = gpa,
            .m = m,
            .tok = tok,
            .cfg = cfg,
            .s = try Scratch.init(gpa, L, 8),
            .logits = try gpa.alloc(f32, 8),
            .pooled = try gpa.alloc(f32, D),
        };
    }

    pub fn moveFn(ctx: *anyopaque, io: std.Io, g: *snake.Snake, status: *std.Io.Writer) anyerror!snake.Dir {
        const self: *ModelChooser = @ptrCast(@alignCast(ctx));

        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();

        const text = try g.stateText(a);
        var opts: [4][]const u8 = undefined;
        try g.optionTexts(a, &opts);

        const sq = try buildSequence(a, self.tok, text, "choice", SNAKE_INS, &opts, self.cfg.max_len, self.cfg.head_max_len, .no_head);
        if (diag.g_prof) profReset();
        const t0 = nowMs(io);
        var f = FwdCtx{
            .m = self.m,
            .s = &self.s,
            .ids = sq.ids,
            .L = sq.ids.len,
            .qtype = 0,
            .markers = sq.markers,
            .logits = self.logits,
            .act = .{ 0, 0 },
            .pooled = self.pooled,
            .feat = self.s.feat,
            .act_in = self.s.act_in,
        };
        forward(self.gpa, &f, .no_head);
        self.ms = nowMs(io) - t0;
        if (diag.g_prof) try profReport(status, pool.g_threads);

        const k = @min(sq.markers.len, 4);
        const temp = @max(1e-3, self.m.temperature[0]);
        var p: [4]f32 = .{ 0, 0, 0, 0 };
        var mx: f32 = -std.math.inf(f32);
        for (0..k) |r| {
            p[r] = self.logits[r] / temp;
            mx = @max(mx, p[r]);
        }
        var sum: f32 = 0;
        for (0..k) |r| {
            p[r] = @exp(p[r] - mx);
            sum += p[r];
        }
        for (0..k) |r| p[r] /= sum;

        // `--promptv 6` treats death as a constraint of the environment rather
        // than a preference to be learned: the model then ranks only the moves
        // that are still legal. `--promptv 7` also drops the moves that step
        // away from the food, leaving the model to break the tie between the
        // survivors that close the distance. `--promptv 8` replaces that single
        // distance rule with the tiered one in snake.zig, where a move only
        // counts if the snake can still reach the food afterwards. Without a
        // mask the reported probabilities are the model's own, fatal ones
        // included.
        if (snake.prompt_variant >= 6) {
            var allowed: [4]bool = .{ true, true, true, true };
            if (snake.prompt_variant == 8) allowed = g.rankedMoves();
            if (snake.prompt_variant == 9) allowed = g.cycleMoves();
            var left: f32 = 0;
            for (0..k) |r| {
                const keep = switch (snake.prompt_variant) {
                    8, 9 => allowed[r],
                    7 => g.moveIsBest(snake.DIRS[r]),
                    else => !g.moveIsFatal(snake.DIRS[r]),
                };
                if (keep) {
                    left += p[r];
                } else {
                    p[r] = 0;
                }
            }
            if (left > 0) {
                for (0..k) |r| p[r] /= left;
            }
        }

        var best: usize = 0;
        for (0..k) |r| {
            if (p[r] > p[best]) best = r;
        }
        const conf = confidenceFromProbs(p[0..k], k);
        self.last = .{ .dir = snake.DIRS[best], .probs = p, .conf = conf, .act = f.act[0], .tokens = sq.ids.len, .ms = self.ms };
        try status.print("model: chose {s}  (up {d:.3} / down {d:.3} / left {d:.3} / right {d:.3})  conf {d:.3}  act {d:.3}  {d} tok  {d} ms\n", .{
            snake.DIRS[best].label(), p[0], p[1], p[2], p[3], conf, f.act[0], sq.ids.len, self.ms,
        });
        return snake.DIRS[best];
    }
};

/// `--snake`: let the model (or one of the baseline policies) play, either animated
/// or as a summary over several games.
pub fn play(gpa: Allocator, io: std.Io, w: *std.Io.Writer, m: *const Model, tok: *const Tokenizer, cfg: *const Cfg, s_policy: Policy, s_opts: snake.Config) !void {
    var mc = try ModelChooser.init(gpa, m, tok, cfg);
    const chooser: snake.Chooser = switch (s_policy) {
        .model => .{ .ctx = &mc, .moveFn = ModelChooser.moveFn },
        .greedy => snake.greedyChooser(),
        .random => snake.randomChooser(),
        .tiers => snake.tierChooser(),
        .cycle => snake.cycleChooser(),
    };
    const name = switch (s_policy) {
        .model => "model",
        .greedy => "greedy (baseline)",
        .random => "random (baseline)",
        .tiers => "tiered rules, no model (baseline)",
        .cycle => "cycle + shortcuts, no model (baseline)",
    };
    try snake.run(gpa, io, w, chooser, name, s_opts, s_policy == .model and enableAnsi());
}
