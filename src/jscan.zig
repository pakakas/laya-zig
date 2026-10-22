//! A minimal streaming JSON scanner. tokenizer.json is tens of MB, so it is walked
//! in place instead of being parsed into a DOM.

pub const JScan = struct {
    s: []const u8,
    i: usize,

    pub fn ws(self: *JScan) void {
        while (self.i < self.s.len) : (self.i += 1) {
            switch (self.s[self.i]) {
                ' ', '\n', '\r', '\t' => {},
                else => return,
            }
        }
    }
    pub fn peek(self: *JScan) !u8 {
        if (self.i >= self.s.len) return error.TruncatedJson;
        return self.s[self.i];
    }
    pub fn byte(self: *JScan) !u8 {
        if (self.i >= self.s.len) return error.TruncatedJson;
        const c = self.s[self.i];
        self.i += 1;
        return c;
    }
    pub fn expect(self: *JScan, c: u8) !void {
        self.ws();
        if (try self.byte() != c) return error.BadJson;
    }
    pub fn skipLit(self: *JScan) !void {
        self.ws();
        while (self.i < self.s.len) : (self.i += 1) {
            switch (self.s[self.i]) {
                ',', '}', ']', ' ', '\n', '\r', '\t' => return,
                else => {},
            }
        }
    }
    pub fn skipString(self: *JScan) !void {
        try self.expect('"');
        while (true) {
            const c = try self.byte();
            if (c == '"') return;
            if (c == '\\') _ = try self.byte();
        }
    }
    /// raw (still escaped) length of the string starting at self.i (which must be '"')
    pub fn rawStringLen(self: *const JScan) !usize {
        var i = self.i + 1;
        var n: usize = 0;
        while (i < self.s.len) {
            const c = self.s[i];
            if (c == '"') return n;
            if (c == '\\') {
                i += 2;
                n += 2;
                continue;
            }
            i += 1;
            n += 1;
        }
        return error.TruncatedJson;
    }
    fn hex4(self: *JScan) !u32 {
        var v: u32 = 0;
        for (0..4) |_| {
            const c = try self.byte();
            const d: u32 = switch (c) {
                '0'...'9' => c - '0',
                'a'...'f' => c - 'a' + 10,
                'A'...'F' => c - 'A' + 10,
                else => return error.BadJson,
            };
            v = (v << 4) | d;
        }
        return v;
    }
    pub fn stringInto(self: *JScan, buf: []u8) ![]const u8 {
        try self.expect('"');
        var o: usize = 0;
        while (true) {
            const c = try self.byte();
            if (c == '"') break;
            if (c != '\\') {
                if (o >= buf.len) return error.BufferTooSmall;
                buf[o] = c;
                o += 1;
                continue;
            }
            const e = try self.byte();
            switch (e) {
                'n', 't', 'r', 'b', 'f' => {
                    if (o + 1 > buf.len) return error.BufferTooSmall;
                    buf[o] = switch (e) {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        'b' => 0x08,
                        else => 0x0c,
                    };
                    o += 1;
                },
                '/', '"', '\\' => {
                    if (o + 1 > buf.len) return error.BufferTooSmall;
                    buf[o] = e;
                    o += 1;
                },
                'u' => {
                    var cp: u32 = try self.hex4();
                    if (cp >= 0xD800 and cp < 0xDC00) {
                        try self.expect('\\');
                        if (try self.byte() != 'u') return error.BadJson;
                        const lo = try self.hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    o = try encodeUtf8(cp, buf, o);
                },
                else => return error.BadJson,
            }
        }
        return buf[0..o];
    }
    pub fn uint(self: *JScan) !u32 {
        self.ws();
        var v: u32 = 0;
        var any = false;
        while (self.i < self.s.len) {
            const c = self.s[self.i];
            if (c >= '0' and c <= '9') {
                v = v *% 10 + @as(u32, c - '0');
                any = true;
                self.i += 1;
            } else break;
        }
        if (!any) return error.BadJson;
        return v;
    }
};

fn encodeUtf8(cp: u32, buf: []u8, o: usize) !usize {
    var out: [4]u8 = undefined;
    const n: usize = if (cp < 0x80) 1 else if (cp < 0x800) 2 else if (cp < 0x10000) 3 else 4;
    switch (n) {
        1 => out[0] = @intCast(cp),
        2 => {
            out[0] = @intCast(0xC0 | (cp >> 6));
            out[1] = @intCast(0x80 | (cp & 0x3F));
        },
        3 => {
            out[0] = @intCast(0xE0 | (cp >> 12));
            out[1] = @intCast(0x80 | ((cp >> 6) & 0x3F));
            out[2] = @intCast(0x80 | (cp & 0x3F));
        },
        4 => {
            out[0] = @intCast(0xF0 | (cp >> 18));
            out[1] = @intCast(0x80 | ((cp >> 12) & 0x3F));
            out[2] = @intCast(0x80 | ((cp >> 6) & 0x3F));
            out[3] = @intCast(0x80 | (cp & 0x3F));
        },
        else => unreachable,
    }
    if (o + n > buf.len) return error.BufferTooSmall;
    @memcpy(buf[o .. o + n], out[0..n]);
    return o + n;
}
