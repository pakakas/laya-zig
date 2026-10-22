//! Fixed architecture of `convaiinnovations/laya-multilingual` (mmBERT-base encoder
//! + Laya decision head) and the special token ids.

pub const D: usize = 768;
pub const H: usize = 12;
pub const HD: usize = 64;
pub const NL: usize = 22;
pub const INTER: usize = 1152;
pub const EPS: f32 = 1e-5;
pub const ROPE_THETA: f32 = 160000.0;
pub const WINDOW: usize = 128 / 2; // sliding layers: |i-j| <= 64
pub const HEAD_LAYERS: usize = 2;
pub const HEAD_FF: usize = 4 * D;
pub const MAX_LEN: usize = 1024;
pub const HEAD_MAX_LEN: usize = 256;

pub const CLS_ID: u32 = 2; // <bos>
pub const SEP_ID: u32 = 1; // <eos>
pub const MASK_ID: u32 = 4; // <mask>
pub const MASK_TOKEN: []const u8 = "<mask>";
pub const SEP_CHAR: []const u8 = "▁"; // U+2581
