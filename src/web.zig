//! `zig-out/bin/laya.exe --serve` starts a tiny HTTP server; the page in
//! src/web/index.html draws the board and calls /api/step once per move. The game
//! still lives here in Zig, so the browser shows exactly what the CLI shows.

const std = @import("std");
const Allocator = std.mem.Allocator;
const AList = std.array_list.Managed;

const snake = @import("snake.zig");
const server = @import("server.zig");

const tokenizer = @import("tokenizer.zig");
const model = @import("model.zig");
const snake_play = @import("snake_play.zig");

const Tokenizer = tokenizer.Tokenizer;

const Model = model.Model;
const Cfg = model.Cfg;

const Policy = snake_play.Policy;
const LastMove = snake_play.LastMove;
const ModelChooser = snake_play.ModelChooser;

const INDEX_HTML = @embedFile("web/index.html");

const WebSession = struct {
    gpa: Allocator,
    m: *const Model,
    tok: *const Tokenizer,
    cfg: *const Cfg,
    mc: *ModelChooser,
    game: snake.Snake = undefined,
    has_game: bool = false,
    policy: Policy = .model,
    base: snake.Chooser = undefined,
    last: ?LastMove = null,
    fatal: [4]bool = .{ false, false, false, false },
    prompt: AList(u8),
    /// which rung of the prompt ladder the page is playing (`?v=` on /api/new)
    variant: u8 = 0,

    fn newGame(self: *WebSession, size: usize, seed: u64) !void {
        if (self.has_game) self.game.deinit();
        self.game = try snake.Snake.init(self.gpa, size, seed);
        self.has_game = true;
        self.last = null;
        try self.capture(&self.game);
    }

    fn chooser(self: *WebSession) snake.Chooser {
        return .{ .ctx = self, .moveFn = moveFn };
    }

    /// Record what the board looked like before the move (for the UI): the exact
    /// prompt text and which directions would have ended the game.
    fn capture(self: *WebSession, g: *snake.Snake) !void {
        self.prompt.clearRetainingCapacity();
        const txt = try g.stateText(self.gpa);
        defer self.gpa.free(txt);
        try self.prompt.appendSlice(txt);
        const h = g.head();
        const tail = g.tailCell();
        for (snake.DIRS, 0..) |d, i| {
            self.fatal[i] = if (g.neighbour(h, d)) |nb|
                // The tail vacates on this very move, so stepping onto it is how
                // a snake that fills the board finishes; only leaving the board
                // or entering the rest of the body is fatal.
                g.occupied[nb[0] * g.size + nb[1]] and !(nb[0] == tail[0] and nb[1] == tail[1])
            else
                true;
        }
    }

    fn moveFn(ctx: *anyopaque, io: std.Io, g: *snake.Snake, status: *std.Io.Writer) anyerror!snake.Dir {
        const self: *WebSession = @ptrCast(@alignCast(ctx));
        try self.capture(g);
        switch (self.policy) {
            .model => {
                const d = try ModelChooser.moveFn(self.mc, io, g, status);
                self.last = self.mc.last;
                return d;
            },
            .greedy, .random, .tiers, .cycle => {
                const d = try self.base.moveFn(self.base.ctx, io, g, status);
                // baselines are deterministic about the direction they pick
                var p: [4]f32 = .{ 0, 0, 0, 0 };
                p[@backingInt(d)] = 1;
                self.last = .{ .dir = d, .probs = p, .conf = 1, .act = 1, .tokens = 0, .ms = 0 };
                return d;
            },
        }
    }

    fn writeState(self: *WebSession, out: *std.Io.Writer, with_prompt: bool) !void {
        const g = &self.game;
        try out.print("{{\"size\":{d},\"score\":{d},\"step\":{d},\"alive\":{s},\"illegal\":{d},\"death\":\"{s}\",\"dir\":\"{s}\",\"food\":[{d},{d}],\"snake\":[", .{
            g.size,
            g.score,
            g.steps,
            if (g.alive) "true" else "false",
            g.illegal,
            if (g.alive) "" else if (g.won) "won" else if (g.hit_wall) "wall" else "self",
            g.dir.label(),
            g.food[0],
            g.food[1],
        });
        for (g.body.items, 0..) |c, i| {
            if (i > 0) try out.writeByte(',');
            try out.print("[{d},{d}]", .{ c[0], c[1] });
        }
        try out.writeAll("],\"fatal\":[");
        for (self.fatal, 0..) |f, i| {
            if (i > 0) try out.writeByte(',');
            try out.writeAll(if (f) "true" else "false");
        }
        try out.writeAll("],\"last\":");
        if (self.last) |l| {
            try out.print("{{\"dir\":\"{s}\",\"probs\":[{d:.5},{d:.5},{d:.5},{d:.5}],\"conf\":{d:.5},\"act\":{d:.5},\"tokens\":{d},\"ms\":{d}}}", .{
                l.dir.label(), l.probs[0], l.probs[1], l.probs[2], l.probs[3], l.conf, l.act, l.tokens, l.ms,
            });
        } else {
            try out.writeAll("null");
        }
        if (with_prompt) {
            try out.writeAll(",\"prompt\":");
            try server.jsonString(out, self.prompt.items);
        }
        try out.writeAll("}");
    }
};

fn webHandle(ctx: *anyopaque, io: std.Io, req: server.Request, out: *std.Io.Writer) anyerror!server.Response {
    const self: *WebSession = @ptrCast(@alignCast(ctx));

    if (std.mem.eql(u8, req.path, "/") or std.mem.eql(u8, req.path, "/index.html")) {
        return .{ .content_type = "text/html; charset=utf-8", .body = INDEX_HTML };
    }

    if (std.mem.eql(u8, req.path, "/api/new")) {
        const size: usize = @intCast(@max(6, @min(req.intParam("size", 10), 24)));
        const seed: u64 = @intCast(@max(0, req.intParam("seed", 12345)));
        const pol = req.param("policy") orelse "model";
        // The prompt ladder is a global the chooser reads, so it belongs to the
        // session: the page asks for one rung per game and every later request
        // puts it back, whatever the last request left behind.
        self.variant = @intCast(@max(0, @min(req.intParam("v", @as(i64, @intCast(snake.prompt_variant))), 9)));
        snake.prompt_variant = self.variant;
        self.policy = if (std.mem.eql(u8, pol, "greedy"))
            .greedy
        else if (std.mem.eql(u8, pol, "random"))
            .random
        else if (std.mem.eql(u8, pol, "tiers"))
            .tiers
        else if (std.mem.eql(u8, pol, "cycle"))
            .cycle
        else
            .model;
        self.base = switch (self.policy) {
            .random => snake.randomChooser(),
            .tiers => snake.tierChooser(),
            .cycle => snake.cycleChooser(),
            else => snake.greedyChooser(),
        };
        try self.newGame(size, seed);
        try self.writeState(out, true);
        return .{};
    }

    if (std.mem.eql(u8, req.path, "/api/step")) {
        snake.prompt_variant = self.variant;
        if (self.game.alive) {
            var scratch: std.Io.Writer.Allocating = .init(self.gpa);
            defer scratch.deinit();
            const d = try self.chooser().moveFn(self, io, &self.game, &scratch.writer);
            self.game.step(d);
        }
        try self.writeState(out, req.intParam("prompt", 1) != 0);
        return .{};
    }

    return .{ .status = 404, .body = "{\"error\":\"no route\"}" };
}

/// `--serve`: start the local HTTP server and block until it stops.
pub fn serve(gpa: Allocator, io: std.Io, w: *std.Io.Writer, m: *const Model, tok: *const Tokenizer, cfg: *const Cfg, port: u16) !void {
    const mc = try gpa.create(ModelChooser);
    mc.* = try ModelChooser.init(gpa, m, tok, cfg);
    var session = WebSession{
        .gpa = gpa,
        .m = m,
        .tok = tok,
        .cfg = cfg,
        .mc = mc,
        .prompt = AList(u8).init(gpa),
    };
    try session.newGame(10, 12345);
    const base = snake.greedyChooser();
    session.base = base;
    server.serve(gpa, io, .{ .port = port, .log = w }, .{ .ctx = &session, .handleFn = webHandle }) catch |e| {
        try w.print("server stopped: {s}\n", .{@errorName(e)});
        try w.flush();
    };
}
