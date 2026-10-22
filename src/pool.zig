//! Global flags, --dumpstats, --profile.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const AList = std.array_list.Managed;

const platform = @import("platform.zig");
const diag = @import("diag.zig");

const nowNs = platform.nowNs;

const profAdd = diag.profAdd;

pub var g_threads: usize = 0; // 0 = pick from the CPU count in main

fn genLow(gen: *std.atomic.Value(u64)) *const u32 {
    comptime std.debug.assert(builtin.cpu.arch.endian() == .little);
    return @ptrCast(&gen.raw);
}

fn wakeAll(gen: *std.atomic.Value(u64)) void {
    switch (comptime builtin.os.tag) {
        .windows => std.os.windows.ntdll.RtlWakeAddressAll(&gen.raw),
        .linux => {
            _ = std.os.linux.futex_3arg(genLow(gen), .{ .cmd = .WAKE, .private = true }, std.math.maxInt(i32));
        },
        else => {},
    }
}

fn waitWhileEqual(gen: *std.atomic.Value(u64), seen: u64) void {
    switch (comptime builtin.os.tag) {
        .windows => {
            var expected = seen;
            _ = std.os.windows.ntdll.RtlWaitOnAddress(&gen.raw, &expected, @sizeOf(u64), null);
        },
        .linux => {
            _ = std.os.linux.futex_4arg(genLow(gen), .{ .cmd = .WAIT, .private = true }, @truncate(seen), null);
        },
        else => std.Thread.yield() catch {},
    }
}

pub const JobFn = *const fn (ctx: *anyopaque, lo: usize, hi: usize) void;

/// Worker threads created once and parked between jobs. The previous code spawned
/// and joined `--threads` OS threads for every parallel() call, and with ~250 calls
/// per forward pass that cost more than every matmul in it (see --profile).
const Pool = struct {
    /// a job smaller than this per thread is not worth a barrier
    const min_per_thread: usize = 16;
    /// how long a worker spins before parking: longer than the gap between two
    /// back-to-back jobs, so a forward pass never pays a futex round-trip
    const spin_budget: u32 = 30_000;

    job_ctx: *anyopaque = undefined,
    job_f: JobFn = undefined,
    job_n: usize = 0,
    job_chunk: usize = 1,
    job_chunks: usize = 0,

    gen: std.atomic.Value(u64) = .init(0),
    next: std.atomic.Value(usize) = .init(0),
    live: std.atomic.Value(usize) = .init(0),
    parked: std.atomic.Value(usize) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),

    threads: AList(std.Thread) = undefined,
    nworkers: usize = 0,

    /// hand out index ranges until the job is exhausted
    fn drain(self: *Pool) void {
        const f = self.job_f;
        const ctx = self.job_ctx;
        const n = self.job_n;
        const chunk = self.job_chunk;
        const chunks = self.job_chunks;
        while (true) {
            const i = self.next.fetchAdd(1, .monotonic);
            if (i >= chunks) return;
            const lo = i * chunk;
            const hi = @min(lo + chunk, n);
            f(ctx, lo, hi);
        }
    }

    fn worker(self: *Pool) void {
        var seen: u64 = 0;
        while (true) {
            var spins: u32 = 0;
            while (true) {
                const g = self.gen.load(.acquire);
                if (g != seen) {
                    seen = g;
                    break;
                }
                if (self.stop.load(.acquire)) return;
                spins += 1;
                if (spins < spin_budget) std.atomic.spinLoopHint() else self.park(seen);
            }
            if (self.stop.load(.acquire)) return;
            self.drain();
            _ = self.live.fetchSub(1, .release);
        }
    }

    fn park(self: *Pool, seen: u64) void {
        _ = self.parked.fetchAdd(1, .monotonic);
        if (self.gen.load(.acquire) == seen and !self.stop.load(.acquire)) {
            waitWhileEqual(&self.gen, seen);
        }
        _ = self.parked.fetchSub(1, .monotonic);
    }

    fn run(self: *Pool, n: usize, ctx: *anyopaque, f: JobFn) void {
        const nw = self.nworkers;
        if (nw == 0 or n / min_per_thread == 0) {
            f(ctx, 0, n);
            return;
        }
        const nt = @min(nw + 1, n / min_per_thread + 1);
        const chunk = @max(1, (n + nt * 4 - 1) / (nt * 4));
        self.job_ctx = ctx;
        self.job_f = f;
        self.job_n = n;
        self.job_chunk = chunk;
        self.job_chunks = (n + chunk - 1) / chunk;
        self.next.store(0, .monotonic);
        self.live.store(nw, .monotonic);
        _ = self.gen.fetchAdd(1, .release);
        if (self.parked.load(.acquire) != 0) wakeAll(&self.gen);
        self.drain();
        var spins: u32 = 0;
        while (self.live.load(.acquire) != 0) {
            spins += 1;
            if (spins < 2000) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
        }
    }

    pub fn start(self: *Pool, alloc: Allocator, nt: usize) void {
        self.threads = AList(std.Thread).init(alloc);
        // A worker's largest frame is the 48 KB weight tile in `mmJob16`. The 16 MB
        // default reserved ~176 MB of commit charge across 11 workers for stacks
        // that never grow past that tile.
        const opts = std.Thread.SpawnConfig{ .stack_size = 256 * 1024 };
        for (1..nt) |_| {
            const t = std.Thread.spawn(opts, worker, .{self}) catch break;
            self.threads.append(t) catch break;
            self.nworkers += 1;
        }
    }

    pub fn stopAll(self: *Pool) void {
        self.stop.store(true, .release);
        wakeAll(&self.gen);
        for (self.threads.items) |t| t.join();
        self.threads.deinit();
        self.nworkers = 0;
    }
};

pub var g_pool: Pool = .{};

pub fn parallel(alloc: Allocator, n: usize, ctx: *anyopaque, f: JobFn) void {
    _ = alloc;
    if (n == 0) return;
    if (!diag.g_prof) {
        g_pool.run(n, ctx, f);
        return;
    }
    const t0 = nowNs();
    g_pool.run(n, ctx, f);
    profAdd(diag.g_stage, nowNs() - t0);
}
