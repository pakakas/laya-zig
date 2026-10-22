//! OS: file, clock, console.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

var g_qpc_freq: i64 = 1_000_000_000;

extern "kernel32" fn QueryPerformanceCounter(lpLargeInteger: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;
extern "kernel32" fn SetConsoleOutputCP(wCodePageID: u32) callconv(.winapi) i32;
extern "kernel32" fn GetStdHandle(nStdHandle: u32) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetConsoleMode(h: ?*anyopaque, mode: *u32) callconv(.winapi) i32;
extern "kernel32" fn SetConsoleMode(h: ?*anyopaque, mode: u32) callconv(.winapi) i32;

/// dulu inline di awal main
pub fn setup() void {
    if (builtin.os.tag == .windows) _ = SetConsoleOutputCP(65001);
    if (builtin.os.tag == .windows) {
        var fr: i64 = 0;
        if (QueryPerformanceFrequency(&fr) != 0 and fr > 0) g_qpc_freq = fr;
    }
}

pub fn nowNs() i64 {
    if (comptime builtin.os.tag == .windows) {
        var c: i64 = undefined;
        _ = QueryPerformanceCounter(&c);
        return @divTrunc(c * std.time.ns_per_s, g_qpc_freq);
    }

    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, ts.sec) * std.time.ns_per_s + ts.nsec;
}

pub fn nowMs(io: std.Io) i64 {
    return @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).nanoseconds, std.time.ns_per_ms));
}

pub fn readFile(alloc: Allocator, io: std.Io, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const st = try f.stat(io);
    const buf = try alloc.alloc(u8, @max(@as(usize, @intCast(st.size)), 1));
    const got = try f.readPositionalAll(io, buf, 0);
    return buf[0..got];
}

/// The weight file as a read-only view instead of a copy. On Windows this is a
/// section plus a mapping of it, so the handle has to stay open for as long as
/// the view is used -- `MemoryMap` keeps it for exactly that reason.
pub fn mapFile(io: std.Io, path: []const u8) !std.Io.File.MemoryMap {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    const st = try f.stat(io);
    return f.createMemoryMap(io, .{
        .len = @intCast(st.size),
        .protection = .{ .read = true },
    });
}

/// Turn on ANSI escape handling so the snake board can repaint in place.
/// Returns false when the console cannot do it (redirected output, old conhost),
/// in which case frames are simply appended instead.
pub fn enableAnsi() bool {
    if (builtin.os.tag != .windows) return true;
    const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5; // (DWORD)-11
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    const h = GetStdHandle(STD_OUTPUT_HANDLE) orelse return false;
    var mode: u32 = 0;
    if (GetConsoleMode(h, &mode) == 0) return false;
    return SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
}
