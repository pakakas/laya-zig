//! Inference runtime for `convaiinnovations/laya-multilingual`.
//!
//! mmBERT-base encoder (22 layers, hidden 768, 12 heads, 256k vocab, RoPE, sliding window)
//! + the Laya decision head (2 transformer layers, option-marker scorer, act/escalate head).
//! The whole model runs locally: no torch, no Python, weights straight from model.safetensors.
//!
//! build:  zig build                 (fast, native CPU; -Doptimize=Debug to override)
//! run:    zig-out\bin\laya.exe                    # built-in demo (Hindi billing email)
//!         zig-out\bin\laya.exe --json q.json      # your own state + questions (UTF-8)
//!         zig-out\bin\laya.exe --threads 8 --dir .
//!         zig-out\bin\laya.exe --tokcheck golden.jsonl   # validate the tokenizer
//!
//! q.json: {"state": "...", "questions": {"id": {"type": "choice"|"score"|"noul",
//!         "instructions": "...", "criteria": {...}|[...]}}}
//!
//! browser: --serve [--port 8080]  start the local UI (see src/web/index.html,
//!                                 smoke-tested by tools/web-smoke.js)
//! misc:   --selftest               check the SIMD kernels against f64 (run after
//!                                 touching dotv/matmul — they have been miscompiled
//!                                 by this zig build before)
//!         --profile                print where the forward pass spends its time
//!
//! snake:  --snake                let the model play Snake (one game, animated board)
//!         --snake --games 10     summary over N games, no animation
//!         --snake --policy greedy|random|tiers|cycle  baselines to compare against
//!         --snake --prompt       print the board text the model is given
//!         --snake --promptv 0..9  how the position is phrased (ladder in snake.zig)
//!         --size N --max-steps N --delay MS --seed N
//!
//! Validated against an independent pure-Python reference implementation
//! (tools/refcheck.py): layer-by-layer activations, logits and probabilities agree
//! to ~1e-5. README.md documents that check and the two others (--selftest,
//! tools/web-smoke.js).
//!
//! Source layout (everything lives in src/):
//!   main.zig         command line parsing and mode dispatch (this file)
//!   config.zig       fixed architecture constants and special token ids
//!   math.zig         scalar gelu / softmax / layer norm
//!   simd.zig         8-wide f32 vector primitives (dot, exp, gelu)
//!   matmul.zig       GEMM kernels over f32 / f16 weights
//!   pool.zig         persistent worker-thread pool and `parallel()`
//!   diag.zig         --debug, --dumpstats and --profile
//!   platform.zig     clocks, file IO, memory-mapping, Windows console setup
//!   safetensors.zig  tensor access for model.safetensors
//!   jscan.zig        streaming JSON scanner (for tokenizer.json)
//!   tokenizer.zig    BPE tokenizer and --tokcheck
//!   model.zig        weights, scratch buffers, runtime config, loader
//!   attention.zig    attention and rotary embeddings
//!   forward.zig      the forward pass
//!   sequence.zig     token sequence / option-marker construction
//!   questions.zig    default mode: questions JSON in, probabilities out
//!   snake_play.zig   --snake: the model (or a baseline) plays Snake
//!   web.zig          --serve: HTTP handler for the browser UI
//!   selftest.zig     --selftest
//!   snake.zig, server.zig, web/index.html   (unchanged)

const std = @import("std");
const AList = std.array_list.Managed;

const snake = @import("snake.zig");
const platform = @import("platform.zig");
const diag = @import("diag.zig");
const pool = @import("pool.zig");
const tokenizer = @import("tokenizer.zig");
const model = @import("model.zig");
const questions = @import("questions.zig");
const snake_play = @import("snake_play.zig");
const web = @import("web.zig");

const nowMs = platform.nowMs;
const readFile = platform.readFile;

const selfTest = @import("selftest.zig").selfTest;

const loadTokenizer = tokenizer.loadTokenizer;
const tokCheck = tokenizer.tokCheck;

const Cfg = model.Cfg;
const loadModel = model.loadModel;

const Policy = snake_play.Policy;

pub fn main(init: std.process.Init) !void {
    platform.setup();

    const gpa = std.heap.page_allocator;
    const io = init.io;
    var argit = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer argit.deinit();
    var argv = AList([]const u8).init(gpa);
    defer argv.deinit();
    while (argit.next()) |a| try argv.append(a);
    const args = argv.items;

    var dir: []const u8 = ".";
    var json_path: ?[]const u8 = null;
    var tokcheck: ?[]const u8 = null;
    var play = false;
    var serve = false;
    var selftest = false;
    var port: u16 = 8080;
    var s_policy: Policy = .model;
    var s_opts = snake.Config{};
    var snake_prompt_v: u8 = 0;
    var s_steps_set = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--snake")) {
            play = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--serve")) {
            serve = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--prompt")) {
            s_opts.show_prompt = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--promptv") and i + 1 < args.len) {
            snake_prompt_v = std.fmt.parseInt(u8, args[i + 1], 10) catch 0;
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, a, "--debug")) {
            diag.g_debug = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--selftest")) {
            selftest = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--dumpstats")) {
            diag.g_dump_stats = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--profile")) {
            diag.g_prof = true;
            continue;
        }
        const takes_value = std.mem.eql(u8, a, "--dir") or std.mem.eql(u8, a, "--json") or
            std.mem.eql(u8, a, "--threads") or std.mem.eql(u8, a, "--tokcheck") or
            std.mem.eql(u8, a, "--size") or std.mem.eql(u8, a, "--games") or
            std.mem.eql(u8, a, "--policy") or std.mem.eql(u8, a, "--delay") or
            std.mem.eql(u8, a, "--seed") or std.mem.eql(u8, a, "--max-steps") or
            std.mem.eql(u8, a, "--port");
        if (!takes_value or i + 1 >= args.len) continue;
        const key = a;
        i += 1;
        const v = args[i];
        if (std.mem.eql(u8, key, "--dir")) dir = v;
        if (std.mem.eql(u8, key, "--json")) json_path = v;
        if (std.mem.eql(u8, key, "--tokcheck")) tokcheck = v;
        if (std.mem.eql(u8, key, "--threads")) pool.g_threads = std.fmt.parseInt(usize, v, 10) catch 0;
        if (std.mem.eql(u8, key, "--size")) s_opts.size = @max(6, @min(std.fmt.parseInt(usize, v, 10) catch 10, 24));
        if (std.mem.eql(u8, key, "--games")) s_opts.games = @max(1, std.fmt.parseInt(usize, v, 10) catch 1);
        if (std.mem.eql(u8, key, "--delay")) s_opts.delay_ms = std.fmt.parseInt(i64, v, 10) catch 0;
        if (std.mem.eql(u8, key, "--seed")) s_opts.seed = std.fmt.parseInt(u64, v, 10) catch 12345;
        if (std.mem.eql(u8, key, "--max-steps")) {
            s_opts.max_steps = std.fmt.parseInt(usize, v, 10) catch 0;
            s_steps_set = s_opts.max_steps > 0;
        }
        if (std.mem.eql(u8, key, "--port")) port = std.fmt.parseInt(u16, v, 10) catch 8080;
        if (std.mem.eql(u8, key, "--policy")) {
            s_policy = if (std.mem.eql(u8, v, "greedy"))
                .greedy
            else if (std.mem.eql(u8, v, "random"))
                .random
            else if (std.mem.eql(u8, v, "tiers"))
                .tiers
            else if (std.mem.eql(u8, v, "cycle"))
                .cycle
            else
                .model;
        }
    }
    if (pool.g_threads == 0) {
        // Measured here (4 P-cores + 8 E-cores, 16 logical): on the 60-180 token
        // prompts this model actually sees, 8 threads is as quick as 12 and costs
        // a third of the CPU-seconds -- the last four only add a straggler for the
        // barrier to wait on. Past ~300 tokens 12 wins again (~15% on a 634).
        if (std.Thread.getCpuCount()) |n| pool.g_threads = @max(1, @min(n, 8)) else |_| {
            pool.g_threads = 4;
        }
    }
    pool.g_threads = @max(1, @min(pool.g_threads, 64));
    snake.prompt_variant = @min(snake_prompt_v, 9);
    if (!s_steps_set) s_opts.max_steps = snake.defaultMaxSteps(s_opts.size);

    var out_buf: [1 << 16]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &out_buf);
    const w = &fw.interface;

    const t0 = nowMs(io);

    if (selftest) {
        try w.print("laya selftest (SIMD kernels vs f64)\n", .{});
        try selfTest(gpa, w);
        return;
    }

    pool.g_pool.start(gpa, pool.g_threads);
    defer pool.g_pool.stopAll();

    // ---- config
    var cfg = Cfg{};
    {
        const cfg_path = try std.fs.path.join(gpa, &.{ dir, "rl_agent_config.json" });
        defer gpa.free(cfg_path);
        if (readFile(gpa, io, cfg_path)) |cd| {
            defer gpa.free(cd);
            var p = try std.json.parseFromSlice(std.json.Value, gpa, cd, .{});
            defer p.deinit();
            const o = p.value.object;
            if (o.get("temperature")) |tv| {
                for (tv.array.items, 0..) |v, j| {
                    if (j < 3) cfg.temperature[j] = switch (v) {
                        .integer => |x| @floatFromInt(x),
                        .float => |x| @floatCast(x),
                        else => 1.0,
                    };
                }
            }
            if (o.get("max_len")) |v| cfg.max_len = @intCast(v.integer);
            if (o.get("head_max_len")) |v| cfg.head_max_len = @intCast(v.integer);
        } else |_| {}
    }

    // ---- tokenizer (kept alive for the whole run)
    var tok_arena = std.heap.ArenaAllocator.init(gpa);
    defer tok_arena.deinit();
    const ta = tok_arena.allocator();
    const tok_path = try std.fs.path.join(gpa, &.{ dir, "tokenizer", "tokenizer.json" });
    defer gpa.free(tok_path);
    const tok = try loadTokenizer(ta, io, tok_path);
    const t1 = nowMs(io);
    try w.print("tokenizer: {d} vocab / {d} merges / {d} added ({d} ms)\n", .{ tok.vocab.count(), tok.merges.count(), tok.added.count(), t1 - t0 });
    try w.flush();

    if (tokcheck) |tp| {
        try tokCheck(gpa, io, &tok, tp);
        return;
    }

    // ---- weights
    const mr = try loadModel(gpa, io, dir, &cfg);
    const m = mr[0];
    const t2 = nowMs(io);
    try w.print("weights: {d} MB, {d} threads ({d} ms)\n", .{ mr[1].len / (1024 * 1024), pool.g_threads, t2 - t1 });
    try w.flush();

    // ---- mode
    if (serve) return web.serve(gpa, io, w, &m, &tok, &cfg, port);
    if (play) return snake_play.play(gpa, io, w, &m, &tok, &cfg, s_policy, s_opts);
    return questions.run(gpa, io, w, &m, &tok, &cfg, json_path, t0, t2);
}
