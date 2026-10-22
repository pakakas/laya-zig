//! Global flags, --dumpstats, --profile.

const std = @import("std");

const config = @import("config.zig");

const D = config.D;

pub var g_debug: bool = false;

/// `--dumpstats`: after every encoder layer, print a few numbers describing the
/// hidden state, so a reference implementation can be diffed against this one
/// layer by layer (see tools/refcheck.py). Only the first forward pass dumps.
pub var g_dump_stats: bool = false;
pub var g_fwd_count: usize = 0;

/// Kept to short lines on purpose: tools/refcheck.py reads this after PowerShell
/// has captured stderr, and PowerShell hard-wraps long lines at the console width.
pub fn dumpStats(tag: []const u8, x: []const f32, L: usize) void {
    var sum: f64 = 0;
    var asum: f64 = 0;
    for (x[0 .. L * D]) |v| {
        sum += v;
        asum += @abs(v);
    }
    std.debug.print("[dump] {s:<5} {d:.5} {d:.5} {d:.5} {d:.3} {d:.3}\n", .{ tag, x[0], x[1], x[2], sum, asum });
}

pub fn dumpIds(ids: []const u32) void {
    var i: usize = 0;
    while (i < ids.len) : (i += 8) {
        std.debug.print("[dump] ids", .{});
        for (ids[i..@min(i + 8, ids.len)]) |v| std.debug.print(" {d}", .{v});
        std.debug.print("\n", .{});
    }
}

/// `--profile`: per-stage timing of the forward pass. Worker threads have no
/// std.Io to read a clock with, so this goes straight to QueryPerformanceCounter.
pub var g_prof: bool = false;

const STAGE_NAMES = [_][]const u8{
    "matmul",   "attention", "norm",     "rope",   "glu+gelu",
    "residual", "embed",     "splitQkv", "serial",
};
pub const ST_MATMUL: u8 = 0;
pub const ST_ATTN: u8 = 1;
pub const ST_NORM: u8 = 2;
pub const ST_ROPE: u8 = 3;
pub const ST_GELU: u8 = 4;
pub const ST_RES: u8 = 5;
pub const ST_EMB: u8 = 6;
pub const ST_SPLIT: u8 = 7;
pub const ST_SERIAL: u8 = 8;

fn zeroI64s() [STAGE_NAMES.len]i64 {
    var a: [STAGE_NAMES.len]i64 = undefined;
    @memset(&a, 0);
    return a;
}

pub var g_stage: u8 = ST_SERIAL;
var g_stage_ns = zeroI64s();
var g_stage_calls = zeroI64s();

pub fn profAdd(stage: u8, ns: i64) void {
    g_stage_ns[stage] += ns;
    g_stage_calls[stage] += 1;
}

pub fn profReset() void {
    @memset(&g_stage_ns, 0);
    @memset(&g_stage_calls, 0);
}

pub fn profReport(w: *std.Io.Writer, threads: usize) !void {
    var tot: i64 = 0;
    for (g_stage_ns) |v| tot += v;
    if (tot == 0) return;
    try w.print("\n[profile] {d} threads, {d} ms of measured stage time\n", .{ threads, @as(f64, @floatFromInt(tot)) / 1e6 });
    for (STAGE_NAMES, 0..) |name, i| {
        if (g_stage_ns[i] == 0) continue;
        const ms = @as(f64, @floatFromInt(g_stage_ns[i])) / 1e6;
        const pct = @as(f64, @floatFromInt(g_stage_ns[i])) * 100.0 / @as(f64, @floatFromInt(tot));
        try w.print("  {s:<9} {d:>8.1} ms {d:>6.1}% {d:>7} calls\n", .{ name, ms, pct, g_stage_calls[i] });
    }
    try w.flush();
}
