//! BFME-specific inner-loop HLE handlers. These live in the engine-handler
//! band (128..=255) and are armed only by byte-exact, title-specific hooks.

use crate::cpu::cpu::{readable_or_pagefault, read_reg32, safe_read32s, safe_write32, safe_write8, writable_or_pagefault, write_reg32, EAX, EBP, EBX, ECX, EDX, ESI, ESP};
use crate::cpu::hypercall::hc_safe_read8;

pub(crate) unsafe fn dispatch_inner_loop(handler_id: u8) -> bool {
    match handler_id {
        135 => handle_fold33_hash(),
        136 => handle_stringbase_lower(),
        137 => handle_stringbase_release(),
        138 => handle_stringbase_copy(),
        139 => handle_stringbase_assign(),
        140 => handle_stringbase_find(),
        141 => handle_matrix_push(),
        142 => handle_matrix_pop(),
        143 => handle_matrix_multiply(),
        144 => handle_transform_push(),
        145 => handle_transform_pop(),
        146 => handle_matrix_adjust(),
        147 => handle_tree_successor(),
        148 => handle_vertex_blend(),
        149 => handle_jpeg_idct_islow(),
        150 => handle_pixel_alpha_blend(),
        151 => handle_msvcr71_sscanf_scalar(),
        152 => handle_msvcr71_stricmp(),
        153 => handle_memory_stream_read1(),
        154 => handle_bc1_color_block(),
        155 => handle_dxt_encode_cache(),
        156 => handle_rgb24_expand(),
        157 => handle_sparse_float4(),
        _ => false,
    }
}

static mut RGB24_EXPAND_CALLS: u32 = 0;
static mut RGB24_EXPAND_PIXELS: u32 = 0;
static mut RGB24_EXPAND_ENABLED: bool = false;
static mut RGB24_EXPAND_ATTEMPTS: u32 = 0;
static mut RGB24_EXPAND_LAST_SOURCE: u32 = 0;
static mut RGB24_EXPAND_LAST_DESTINATION: u32 = 0;
static mut RGB24_EXPAND_LAST_END: u32 = 0;
static mut RGB24_EXPAND_LAST_COUNT: u32 = 0;
static mut RGB24_EXPAND_LAST_FAILURE: u32 = 0;

#[inline(always)]
unsafe fn rgb24_decline(code: u32) -> bool {
    RGB24_EXPAND_LAST_FAILURE = code;
    // The guest wrapper tests EBX and enters the relocated original loop when
    // zero. Returning true keeps this byte-exact hook entirely in WASM; a Rust
    // decline must never fall through to the JS stub and skip the loop.
    write_reg32(EBX, 0);
    true
}

#[inline(always)]
fn ranges_overlap(left: u32, left_len: u32, right: u32, right_len: u32) -> bool {
    let left_end = left as u64 + left_len as u64;
    let right_end = right as u64 + right_len as u64;
    (left as u64) < right_end && (right as u64) < left_end
}

/// Validate a complete non-wrapping guest range without touching its contents.
/// The paging helper deliberately accepts less than one page at a time.
#[inline]
unsafe fn validate_guest_range(address: u32, length: u32, writable: bool) -> bool {
    if length == 0 || (address as u64 + length as u64) > 0x1_0000_0000 { return false; }
    let mut cursor = address;
    let mut remaining = length;
    while remaining != 0 {
        let in_page = cursor & 0xfff;
        let chunk = remaining.min(0xfff - in_page).max(1);
        let valid = if writable {
            writable_or_pagefault(cursor as i32, chunk as i32).is_ok()
        }
        else {
            readable_or_pagefault(cursor as i32, chunk as i32).is_ok()
        };
        if !valid { return false; }
        cursor = cursor.wrapping_add(chunk);
        remaining -= chunk;
    }
    true
}

/// lotrbfme.exe 1.03 FR @ 0x00e29092. The byte-exact guest hook enters with
/// EAX=packed RGB source, ESI=XRGB destination and ECX=destination end. Consume
/// the complete loop in WASM and preserve every register live at 0x00e290ae.
unsafe fn handle_rgb24_expand() -> bool {
    let esp = read_reg32(ESP);
    let source = match safe_read32s(esp.wrapping_add(4)) { Ok(v) => v as u32, Err(_) => return rgb24_decline(9) };
    let destination = match safe_read32s(esp.wrapping_add(8)) { Ok(v) => v as u32, Err(_) => return rgb24_decline(9) };
    let end = match safe_read32s(esp.wrapping_add(12)) { Ok(v) => v as u32, Err(_) => return rgb24_decline(9) };
    let destination_bytes = end.wrapping_sub(destination);
    RGB24_EXPAND_ATTEMPTS = RGB24_EXPAND_ATTEMPTS.wrapping_add(1);
    RGB24_EXPAND_LAST_SOURCE = source;
    RGB24_EXPAND_LAST_DESTINATION = destination;
    RGB24_EXPAND_LAST_END = end;
    RGB24_EXPAND_LAST_COUNT = destination_bytes >> 2;
    RGB24_EXPAND_LAST_FAILURE = 0;
    if !RGB24_EXPAND_ENABLED { return rgb24_decline(8); }
    if destination == 0 || source == 0 || end <= destination
        || destination_bytes & 3 != 0 {
        return rgb24_decline(1);
    }
    let count = destination_bytes >> 2;
    if count == 0 || count > 0x0100_0000 { return rgb24_decline(2); }
    let source_bytes = match count.checked_mul(3) { Some(v) => v, None => return rgb24_decline(2) };
    if source.checked_add(source_bytes).is_none() || destination.checked_add(destination_bytes).is_none() {
        return rgb24_decline(2);
    }
    // The original loop is streaming. Cached reads would change its semantics
    // for overlapping input/output, so leave that rare case to the original.
    if ranges_overlap(source, source_bytes, destination, destination_bytes) {
        return rgb24_decline(6);
    }
    // Make the accelerated operation atomic with respect to fallback: once the
    // first output word is written, no later paging failure may route through
    // the original loop and apply a partially completed conversion twice.
    if !validate_guest_range(source, source_bytes, false)
        || !validate_guest_range(destination, destination_bytes, true) {
        return rgb24_decline(7);
    }

    let mut last = 0u32;
    for i in 0..count {
        let src = source.wrapping_add(i.wrapping_mul(3)) as i32;
        let value = if i + 1 < count {
            match safe_read32s(src) {
                Ok(v) => v as u32,
                Err(_) => return rgb24_decline(3),
            }
        }
        else {
            // Do not read a fourth byte beyond the source span on the final
            // pixel, even when it would usually remain inside a mapped page.
            let r = match hc_safe_read8(src) { Ok(v) => v as u32, Err(_) => return rgb24_decline(4) };
            let g = match hc_safe_read8(src.wrapping_add(1)) { Ok(v) => v as u32, Err(_) => return rgb24_decline(4) };
            let b = match hc_safe_read8(src.wrapping_add(2)) { Ok(v) => v as u32, Err(_) => return rgb24_decline(4) };
            r | g << 8 | b << 16
        };
        last = ((value & 0xff) << 16) | (value & 0xff00) | ((value >> 16) & 0xff);
        let dst = destination.wrapping_add(i.wrapping_mul(4)) as i32;
        if safe_write32(dst, last as i32).is_err() { return rgb24_decline(5); }
    }

    write_reg32(EAX, source.wrapping_add(count.wrapping_mul(3)) as i32);
    write_reg32(ESI, end as i32);
    write_reg32(EDX, last as i32);
    write_reg32(EBX, 1);
    RGB24_EXPAND_CALLS = RGB24_EXPAND_CALLS.wrapping_add(1);
    RGB24_EXPAND_PIXELS = RGB24_EXPAND_PIXELS.wrapping_add(count);
    true
}

#[no_mangle]
pub unsafe fn bfme_rgb24_stat(index: u32) -> u32 {
    match index {
        0 => RGB24_EXPAND_CALLS,
        1 => RGB24_EXPAND_PIXELS,
        2 => RGB24_EXPAND_ATTEMPTS,
        3 => RGB24_EXPAND_LAST_SOURCE,
        4 => RGB24_EXPAND_LAST_DESTINATION,
        5 => RGB24_EXPAND_LAST_END,
        6 => RGB24_EXPAND_LAST_COUNT,
        7 => RGB24_EXPAND_LAST_FAILURE,
        _ => 0,
    }
}

#[no_mangle]
pub unsafe fn bfme_rgb24_stat_reset() {
    RGB24_EXPAND_CALLS = 0;
    RGB24_EXPAND_PIXELS = 0;
    RGB24_EXPAND_ATTEMPTS = 0;
    RGB24_EXPAND_LAST_SOURCE = 0;
    RGB24_EXPAND_LAST_DESTINATION = 0;
    RGB24_EXPAND_LAST_END = 0;
    RGB24_EXPAND_LAST_COUNT = 0;
    RGB24_EXPAND_LAST_FAILURE = 0;
}

#[no_mangle]
pub unsafe fn bfme_rgb24_set_enabled(enabled: u32) {
    RGB24_EXPAND_ENABLED = enabled != 0;
}

#[no_mangle]
pub unsafe fn bfme_rgb24_get_enabled() -> u32 { RGB24_EXPAND_ENABLED as u32 }

static mut SPARSE_FLOAT4_ENABLED: bool = false;
static mut SPARSE_FLOAT4_ATTEMPTS: u32 = 0;
static mut SPARSE_FLOAT4_CALLS: u32 = 0;
static mut SPARSE_FLOAT4_ITEMS: u32 = 0;
static mut SPARSE_FLOAT4_LAST_FAILURE: u32 = 0;

#[inline(always)]
unsafe fn sparse_float4_decline(code: u32) -> bool {
    SPARSE_FLOAT4_LAST_FAILURE = code;
    write_reg32(ESI, 0);
    true
}

/// lotrbfme.exe 1.03 FR @ 0x00e2f4f0. Arguments materialized by the guest
/// wrapper are [entries, entries_end, destination_base, source_float4, frame].
/// Binary64 holds each exact f32 product plus f32 accumulator before the same
/// final binary32 rounding performed by the original x87 FSTP.
unsafe fn handle_sparse_float4() -> bool {
    let esp = read_reg32(ESP);
    let entries = match safe_read32s(esp.wrapping_add(4)) { Ok(v) => v as u32, Err(_) => return sparse_float4_decline(1) };
    let entries_end = match safe_read32s(esp.wrapping_add(8)) { Ok(v) => v as u32, Err(_) => return sparse_float4_decline(1) };
    let destination_base = match safe_read32s(esp.wrapping_add(12)) { Ok(v) => v as u32, Err(_) => return sparse_float4_decline(1) };
    let source = match safe_read32s(esp.wrapping_add(16)) { Ok(v) => v as u32, Err(_) => return sparse_float4_decline(1) };
    let frame = match safe_read32s(esp.wrapping_add(20)) { Ok(v) => v, Err(_) => return sparse_float4_decline(1) };
    SPARSE_FLOAT4_ATTEMPTS = SPARSE_FLOAT4_ATTEMPTS.wrapping_add(1);
    SPARSE_FLOAT4_LAST_FAILURE = 0;
    if !SPARSE_FLOAT4_ENABLED { return sparse_float4_decline(8); }
    let bytes = entries_end.wrapping_sub(entries);
    if entries == 0 || entries_end <= entries || bytes & 7 != 0 { return sparse_float4_decline(2); }
    let count = bytes >> 3;
    if count == 0 || count > 0x0100_0000 { return sparse_float4_decline(3); }
    let outer = match safe_read32s(frame.wrapping_sub(4)) { Ok(v) if v != 0 => v, _ => return sparse_float4_decline(4) };
    let multiplier = match read_f32(outer.wrapping_add(4)) { Some(v) => v as f64, None => return sparse_float4_decline(4) };
    let mut source_lanes = [0f64; 4];
    for lane in 0..4i32 {
        source_lanes[lane as usize] = match read_f32((source as i32).wrapping_sub(8).wrapping_add(lane * 4)) {
            Some(v) => v as f64,
            None => return sparse_float4_decline(5),
        };
    }


    let source_start = match source.checked_sub(8) { Some(v) => v, None => return sparse_float4_decline(5) };
    let multiplier_address = match (outer as u32).checked_add(4) { Some(v) => v, None => return sparse_float4_decline(4) };
    if entries.checked_add(bytes).is_none() { return sparse_float4_decline(2); }

    // Validate every scattered destination before mutating any of them. This
    // preserves the relocated original as a genuine all-or-nothing fallback.
    // Repeated destination indices remain valid and retain sequential sums.
    for i in 0..count {
        let entry = entries.wrapping_add(i.wrapping_mul(8)) as i32;
        let index = match safe_read32s(entry) { Ok(v) => v as u32, Err(_) => return sparse_float4_decline(6) };
        if read_f32(entry.wrapping_add(4)).is_none() { return sparse_float4_decline(6); }
        let destination_offset = match index.checked_mul(16) { Some(v) => v, None => return sparse_float4_decline(7) };
        let destination = match destination_base.checked_add(destination_offset) { Some(v) => v, None => return sparse_float4_decline(7) };
        if destination.checked_add(16).is_none()
            || ranges_overlap(destination, 16, entries, bytes)
            || ranges_overlap(destination, 16, source_start, 16)
            || ranges_overlap(destination, 16, multiplier_address, 4)
            || !validate_guest_range(destination, 16, true) {
            return sparse_float4_decline(7);
        }
        for lane in 0..4i32 {
            if read_f32(destination.wrapping_add((lane * 4) as u32) as i32).is_none() {
                return sparse_float4_decline(7);
            }
        }
    }

    let mut last_destination = 0u32;
    for i in 0..count {
        let entry = entries.wrapping_add(i.wrapping_mul(8)) as i32;
        let index = match safe_read32s(entry) { Ok(v) => v as u32, Err(_) => return sparse_float4_decline(6) };
        let weight = match read_f32(entry.wrapping_add(4)) { Some(v) => v as f64 * multiplier, None => return sparse_float4_decline(6) };
        let destination = destination_base.wrapping_add(index.wrapping_mul(16));
        for lane in 0..4i32 {
            let address = destination.wrapping_add((lane * 4) as u32) as i32;
            let current = match read_f32(address) { Some(v) => v as f64, None => return sparse_float4_decline(7) };
            if !write_f32(address, current + weight * source_lanes[lane as usize]) {
                return sparse_float4_decline(7);
            }
        }
        last_destination = destination.wrapping_add(12);
    }
    write_reg32(EAX, entries_end as i32);
    write_reg32(ESI, last_destination as i32);
    SPARSE_FLOAT4_CALLS = SPARSE_FLOAT4_CALLS.wrapping_add(1);
    SPARSE_FLOAT4_ITEMS = SPARSE_FLOAT4_ITEMS.wrapping_add(count);
    true
}

#[no_mangle]
pub unsafe fn bfme_sparse_float4_set_enabled(enabled: u32) { SPARSE_FLOAT4_ENABLED = enabled != 0; }
#[no_mangle]
pub unsafe fn bfme_sparse_float4_get_enabled() -> u32 { SPARSE_FLOAT4_ENABLED as u32 }
#[no_mangle]
pub unsafe fn bfme_sparse_float4_stat(index: u32) -> u32 {
    match index {
        0 => SPARSE_FLOAT4_ATTEMPTS,
        1 => SPARSE_FLOAT4_CALLS,
        2 => SPARSE_FLOAT4_ITEMS,
        3 => SPARSE_FLOAT4_LAST_FAILURE,
        _ => 0,
    }
}
#[no_mangle]
pub unsafe fn bfme_sparse_float4_stat_reset() {
    SPARSE_FLOAT4_ATTEMPTS = 0;
    SPARSE_FLOAT4_CALLS = 0;
    SPARSE_FLOAT4_ITEMS = 0;
    SPARSE_FLOAT4_LAST_FAILURE = 0;
}

// Keep the complete 256-byte keys compact enough for CPU cache locality.  The
// cold high-quality pass is benchmarked at several capacities before retention.
const DXT_CACHE_SLOTS: usize = 2048;
const DXT_CACHE_WAYS: usize = 4;
const DXT_CACHE_SETS: usize = DXT_CACHE_SLOTS / DXT_CACHE_WAYS;
const DXT_SOURCE_WORDS: usize = 64;
static mut DXT_CACHE_ENABLED: bool = true;
static mut DXT_CACHE_VALID: [u8; DXT_CACHE_SLOTS] = [0; DXT_CACHE_SLOTS];
static mut DXT_CACHE_HASH: [u32; DXT_CACHE_SLOTS] = [0; DXT_CACHE_SLOTS];
static mut DXT_CACHE_MODE: [i32; DXT_CACHE_SLOTS] = [0; DXT_CACHE_SLOTS];
static mut DXT_CACHE_OPTION: [i32; DXT_CACHE_SLOTS] = [0; DXT_CACHE_SLOTS];
static mut DXT_CACHE_SOURCE: [[u32; DXT_SOURCE_WORDS]; DXT_CACHE_SLOTS] =
    [[0; DXT_SOURCE_WORDS]; DXT_CACHE_SLOTS];
static mut DXT_CACHE_OUTPUT: [[u32; 2]; DXT_CACHE_SLOTS] = [[0; 2]; DXT_CACHE_SLOTS];
static mut DXT_CACHE_NEXT_WAY: [u8; DXT_CACHE_SETS] = [0; DXT_CACHE_SETS];
static mut DXT_CACHE_LOOKUPS: u32 = 0;
static mut DXT_CACHE_HITS: u32 = 0;
static mut DXT_CACHE_INSERTS: u32 = 0;
static mut DXT_CACHE_REPLACEMENTS: u32 = 0;
static mut DXT_CACHE_BYPASSES: u32 = 0;
static mut DXT_FAST_ENABLED: bool = false;
static mut DXT_FAST_ENCODES: u32 = 0;

#[inline(always)]
fn dxt_clamp01(value: f32) -> f32 {
    if !value.is_finite() || value <= 0.0 { 0.0 }
    else if value >= 1.0 { 1.0 }
    else { value }
}

#[inline(always)]
fn dxt_pack_565(rgb: [f32; 3]) -> u16 {
    let r = (dxt_clamp01(rgb[0]) * 31.0 + 0.5) as u16;
    let g = (dxt_clamp01(rgb[1]) * 63.0 + 0.5) as u16;
    let b = (dxt_clamp01(rgb[2]) * 31.0 + 0.5) as u16;
    (r << 11) | (g << 5) | b
}

#[inline(always)]
fn dxt_unpack_565(value: u16) -> [f32; 3] {
    [
        ((value >> 11) & 31) as f32 * (1.0 / 31.0),
        ((value >> 5) & 63) as f32 * (1.0 / 63.0),
        (value & 31) as f32 * (1.0 / 31.0),
    ]
}

#[inline(always)]
fn dxt_palette(c0: u16, c1: u16, three_colour: bool) -> [[f32; 3]; 4] {
    let a = dxt_unpack_565(c0);
    let b = dxt_unpack_565(c1);
    let mut result = [[0.0; 3]; 4];
    result[0] = a;
    result[1] = b;
    for lane in 0..3 {
        if three_colour {
            result[2][lane] = (a[lane] + b[lane]) * 0.5;
        } else {
            result[2][lane] = (a[lane] * 2.0 + b[lane]) * (1.0 / 3.0);
            result[3][lane] = (a[lane] + b[lane] * 2.0) * (1.0 / 3.0);
        }
    }
    result
}

#[inline(always)]
fn dxt_colour_error(pixel: [f32; 3], candidate: [f32; 3]) -> f32 {
    // Match BFME's strongly perceptual fit closely enough for a fast loading
    // path: green dominates, then red, then blue. Only relative weights matter.
    let dr = pixel[0] - candidate[0];
    let dg = pixel[1] - candidate[1];
    let db = pixel[2] - candidate[2];
    dr * dr * 0.29703665 + dg * dg + db * db * 0.10078278
}

/// Fast, deterministic BC1 range fit for BFME's cold texture load. The title's
/// high-quality x87 cluster fit is disproportionately expensive under a JIT.
/// Keep this path deliberately tiny: one min/max pass and one selector pass.
/// It affects texture pixels only, never simulation state.
fn dxt_fast_encode(words: &[u32; DXT_SOURCE_WORDS], mode: i32) -> [u32; 2] {
    let mut opaque_count = 0usize;
    let mut lo = [1.0f32; 3];
    let mut hi = [0.0f32; 3];

    for i in 0..16 {
        let alpha = f32::from_bits(words[i * 4 + 3]);
        let visible = mode == 0 || !alpha.is_finite() || alpha >= 0.5;
        if visible {
            for lane in 0..3 {
                let value = dxt_clamp01(f32::from_bits(words[i * 4 + lane]));
                if value < lo[lane] { lo[lane] = value; }
                if value > hi[lane] { hi[lane] = value; }
            }
            opaque_count += 1;
        }
    }

    if opaque_count == 0 {
        return [0xffff_0000, 0xffff_ffff];
    }

    let three_colour = mode != 0 && opaque_count != 16;
    let mut c0 = dxt_pack_565(hi);
    let mut c1 = dxt_pack_565(lo);

    if three_colour {
        if c0 > c1 { core::mem::swap(&mut c0, &mut c1); }
    } else if c0 < c1 {
        core::mem::swap(&mut c0, &mut c1);
    }
    let palette = dxt_palette(c0, c1, three_colour);
    let palette_len = if three_colour { 3 } else { 4 };
    let mut selectors = 0u32;
    for i in 0..16 {
        let alpha = f32::from_bits(words[i * 4 + 3]);
        let visible = mode == 0 || !alpha.is_finite() || alpha >= 0.5;
        let best = if !visible {
            3usize
        } else {
            let pixel = [
                dxt_clamp01(f32::from_bits(words[i * 4])),
                dxt_clamp01(f32::from_bits(words[i * 4 + 1])),
                dxt_clamp01(f32::from_bits(words[i * 4 + 2])),
            ];
            let mut selected = 0usize;
            let mut best_error = f32::INFINITY;
            for selector in 0..palette_len {
                let error = dxt_colour_error(pixel, palette[selector]);
                if error < best_error { best_error = error; selected = selector; }
            }
            selected
        };
        selectors |= (best as u32) << (i * 2);
    }
    [c0 as u32 | ((c1 as u32) << 16), selectors]
}

#[inline(always)]
fn dxt_hash_word(mut hash: u32, word: u32) -> u32 {
    // Four-byte FNV-1a. The full 256-byte key is compared on a hit, so this
    // hash selects a slot but can never make a colliding block authoritative.
    hash ^= word & 0xff;
    hash = hash.wrapping_mul(0x0100_0193);
    hash ^= (word >> 8) & 0xff;
    hash = hash.wrapping_mul(0x0100_0193);
    hash ^= (word >> 16) & 0xff;
    hash = hash.wrapping_mul(0x0100_0193);
    hash ^= word >> 24;
    hash.wrapping_mul(0x0100_0193)
}

#[inline(always)]
fn dxt_finish_hash(mut hash: u32) -> u32 {
    // FNV-1a's low bits retain visible regularity for arrays of IEEE-754
    // floats.  Using them directly for a power-of-two table caused a few hot
    // blocks to evict each other thousands of times even though the complete
    // working set fit in the table.  Avalanche all bits before selecting the
    // slot.  The complete key is still compared below, so this affects only
    // cache placement and can never turn a collision into a false hit.
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2_ae35);
    hash ^ (hash >> 16)
}

#[inline]
unsafe fn read_dxt_source(source: i32) -> Option<([u32; DXT_SOURCE_WORDS], u32)> {
    let mut words = [0u32; DXT_SOURCE_WORDS];
    let mut hash = 0x811c_9dc5u32;
    for i in 0..DXT_SOURCE_WORDS {
        let word = match safe_read32s(source.wrapping_add((i * 4) as i32)) {
            Ok(v) => v as u32,
            Err(_) => return None,
        };
        words[i] = word;
        hash = dxt_hash_word(hash, word);
    }
    Some((words, hash))
}

/// Exact memoization wrapper for lotrbfme.exe 1.03 FR @ 0x00e67124.
/// The generated x86 wrapper supplies [source, output, mode, option, phase].
/// A lookup copies cached bytes only after comparing all 64 source words plus
/// both scalar options. A miss runs the original encoder and calls phase 1 to
/// record its authoritative eight-byte output.
unsafe fn handle_dxt_encode_cache() -> bool {
    let esp = read_reg32(ESP);
    let source = match safe_read32s(esp.wrapping_add(4)) { Ok(v) if v != 0 => v, _ => return false };
    let output = match safe_read32s(esp.wrapping_add(8)) { Ok(v) if v != 0 => v, _ => return false };
    let mode = match safe_read32s(esp.wrapping_add(12)) { Ok(v) => v, Err(_) => return false };
    let option = match safe_read32s(esp.wrapping_add(16)) { Ok(v) => v, Err(_) => return false };
    let phase = match safe_read32s(esp.wrapping_add(20)) { Ok(v) => v, Err(_) => return false };

    if !DXT_CACHE_ENABLED {
        if phase == 0 { DXT_CACHE_BYPASSES = DXT_CACHE_BYPASSES.wrapping_add(1); }
        write_reg32(EAX, 0);
        return true;
    }
    let (words, mut hash) = match read_dxt_source(source) {
        Some(v) => v,
        None => return false,
    };
    hash = dxt_finish_hash(dxt_hash_word(dxt_hash_word(hash, mode as u32), option as u32));
    let set = (hash as usize) & (DXT_CACHE_SETS - 1);
    let set_base = set * DXT_CACHE_WAYS;

    if phase == 0 {
        DXT_CACHE_LOOKUPS = DXT_CACHE_LOOKUPS.wrapping_add(1);
        for way in 0..DXT_CACHE_WAYS {
            let slot = set_base + way;
            if DXT_CACHE_VALID[slot] != 0
                && DXT_CACHE_HASH[slot] == hash
                && DXT_CACHE_MODE[slot] == mode
                && DXT_CACHE_OPTION[slot] == option
                && DXT_CACHE_SOURCE[slot] == words
            {
                let encoded = DXT_CACHE_OUTPUT[slot];
                if safe_write32(output, encoded[0] as i32).is_err()
                    || safe_write32(output.wrapping_add(4), encoded[1] as i32).is_err()
                {
                    return false;
                }
                DXT_CACHE_HITS = DXT_CACHE_HITS.wrapping_add(1);
                write_reg32(EAX, 1);
                return true;
            }
        }
        if DXT_FAST_ENABLED {
            let encoded = dxt_fast_encode(&words, mode);
            if safe_write32(output, encoded[0] as i32).is_err()
                || safe_write32(output.wrapping_add(4), encoded[1] as i32).is_err()
            {
                return false;
            }
            DXT_FAST_ENCODES = DXT_FAST_ENCODES.wrapping_add(1);
            write_reg32(EAX, 1);
            return true;
        }
        write_reg32(EAX, 0);
        return true;
    }

    if phase == 1 {
        let out0 = match safe_read32s(output) { Ok(v) => v as u32, Err(_) => return false };
        let out1 = match safe_read32s(output.wrapping_add(4)) { Ok(v) => v as u32, Err(_) => return false };
        let mut slot = set_base;
        let mut found_empty = false;
        for way in 0..DXT_CACHE_WAYS {
            let candidate = set_base + way;
            if DXT_CACHE_VALID[candidate] == 0 {
                slot = candidate;
                found_empty = true;
                break;
            }
        }
        if !found_empty {
            slot = set_base + (DXT_CACHE_NEXT_WAY[set] as usize);
            DXT_CACHE_NEXT_WAY[set] = ((DXT_CACHE_NEXT_WAY[set] as usize + 1) % DXT_CACHE_WAYS) as u8;
            DXT_CACHE_REPLACEMENTS = DXT_CACHE_REPLACEMENTS.wrapping_add(1);
        }
        DXT_CACHE_HASH[slot] = hash;
        DXT_CACHE_MODE[slot] = mode;
        DXT_CACHE_OPTION[slot] = option;
        DXT_CACHE_SOURCE[slot] = words;
        DXT_CACHE_OUTPUT[slot] = [out0, out1];
        DXT_CACHE_VALID[slot] = 1;
        DXT_CACHE_INSERTS = DXT_CACHE_INSERTS.wrapping_add(1);
        write_reg32(EAX, 0);
        return true;
    }
    false
}

#[no_mangle]
pub unsafe fn bfme_dxt_cache_set_enabled(enabled: u32) {
    DXT_CACHE_ENABLED = enabled != 0;
}

#[no_mangle]
pub unsafe fn bfme_dxt_cache_get_enabled() -> u32 { DXT_CACHE_ENABLED as u32 }

#[no_mangle]
pub unsafe fn bfme_dxt_fast_set_enabled(enabled: u32) {
    DXT_FAST_ENABLED = enabled != 0;
}

#[no_mangle]
pub unsafe fn bfme_dxt_fast_get_enabled() -> u32 { DXT_FAST_ENABLED as u32 }

#[no_mangle]
pub unsafe fn bfme_dxt_cache_get_stat(index: u32) -> u32 {
    match index {
        0 => DXT_CACHE_LOOKUPS,
        1 => DXT_CACHE_HITS,
        2 => DXT_CACHE_INSERTS,
        3 => DXT_CACHE_REPLACEMENTS,
        4 => DXT_CACHE_BYPASSES,
        5 => DXT_FAST_ENCODES,
        _ => 0,
    }
}

#[no_mangle]
pub unsafe fn bfme_dxt_cache_reset_stats() {
    DXT_CACHE_LOOKUPS = 0;
    DXT_CACHE_HITS = 0;
    DXT_CACHE_INSERTS = 0;
    DXT_CACHE_REPLACEMENTS = 0;
    DXT_CACHE_BYPASSES = 0;
    DXT_FAST_ENCODES = 0;
}


#[inline(always)]
fn bc1_interpolate(left: f32, right: f32, scale: f32) -> f32 {
    (((right as f64 - left as f64) * scale as f64) + left as f64) as f32
}

#[inline(always)]
fn bc1_endpoint(value: u32) -> [f32; 4] {
    let scale5 = f32::from_bits(0x3d04_2108); // lotrbfme.exe 0x01149e2c
    let scale6 = f32::from_bits(0x3c82_0821); // lotrbfme.exe 0x01149e24
    [
        (((value >> 11) & 0x1f) as f64 * scale5 as f64) as f32,
        (((value >> 5) & 0x3f) as f64 * scale6 as f64) as f32,
        ((value & 0x1f) as f64 * scale5 as f64) as f32,
        1.0,
    ]
}

/// lotrbfme.exe 1.03 FR @ 0x00e679a5. Inputs are captured before the first
/// output write, preserving the original's alias behaviour and fault surface.
unsafe fn handle_bc1_color_block() -> bool {
    let esp = read_reg32(ESP);
    let output = match safe_read32s(esp.wrapping_add(4)) { Ok(v) if v != 0 => v, _ => return false };
    let input = match safe_read32s(esp.wrapping_add(8)) { Ok(v) if v != 0 => v, _ => return false };
    if !decode_bc1_color_block(output, input) { return false; }
    write_reg32(EAX, 0);
    true
}

unsafe fn decode_bc1_color_block(output: i32, input: i32) -> bool {
    let color0 = match read_u16(input) { Some(v) => v, None => return false };
    let color1 = match read_u16(input.wrapping_add(2)) { Some(v) => v, None => return false };
    let selectors = match safe_read32s(input.wrapping_add(4)) { Ok(v) => v as u32, Err(_) => return false };
    if safe_read32s(output).is_err() || safe_read32s(output.wrapping_add(252)).is_err() { return false; }

    let mut palette = [[0.0f32; 4]; 4];
    palette[0] = bc1_endpoint(color0);
    palette[1] = bc1_endpoint(color1);
    if color0 <= color1 {
        let scale = f32::from_bits(0x3f00_0000);
        for lane in 0..4 { palette[2][lane] = bc1_interpolate(palette[0][lane], palette[1][lane], scale); }
    } else {
        let third = f32::from_bits(0x3eaa_aaab);
        let two_thirds = f32::from_bits(0x3f2a_aaab);
        for lane in 0..4 {
            palette[2][lane] = bc1_interpolate(palette[0][lane], palette[1][lane], third);
            palette[3][lane] = bc1_interpolate(palette[0][lane], palette[1][lane], two_thirds);
        }
    }
    for pixel in 0..16i32 {
        let color = palette[((selectors >> (pixel * 2)) & 3) as usize];
        let destination = output.wrapping_add(pixel * 16);
        for lane in 0..4i32 {
            if safe_write32(destination.wrapping_add(lane * 4), color[lane as usize].to_bits() as i32).is_err() {
                return false;
            }
        }
    }
    true
}

/// lotrbfme.exe 1.03 FR @ 0x00dd1a70. The byte-exact entry filter admits
/// only the parser's overwhelmingly common one-byte request. Invalid state
/// declines to the relocated original before any guest mutation.
unsafe fn handle_memory_stream_read1() -> bool {
    let esp = read_reg32(ESP);
    let object = match safe_read32s(esp.wrapping_add(4)) {
        Ok(v) if v != 0 => v,
        _ => return false,
    };
    let destination = match safe_read32s(esp.wrapping_add(8)) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let requested = match safe_read32s(esp.wrapping_add(12)) {
        Ok(1) => 1,
        _ => return false,
    };
    let base = match safe_read32s(object.wrapping_add(0x14)) {
        Ok(0) => {
            write_reg32(EAX, -1);
            return true;
        }
        Ok(v) => v,
        Err(_) => return false,
    };
    let position = match safe_read32s(object.wrapping_add(0x18)) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let end = match safe_read32s(object.wrapping_add(0x1c)) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let available = end.wrapping_sub(position);
    let count = if requested <= available { requested } else { available };
    if count > 0 && destination != 0 {
        let byte = match hc_safe_read8(base.wrapping_add(position)) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if safe_write8(destination, byte).is_err() { return false; }
    }
    if safe_write32(object.wrapping_add(0x18), position.wrapping_add(count)).is_err() {
        return false;
    }
    write_reg32(EAX, count);
    true
}



#[inline(always)]
fn scanf_space(byte: i32) -> bool {
    byte == 0x20 || (byte >= 0x09 && byte <= 0x0d)
}

/// MSVCR71.dll 7.10.3052.4 `sscanf`: the entry filter admits only the exact
/// one-output formats `%d`, `%u` and `%f`. Complex/variadic formats never enter
/// this handler and remain authoritative in the original CRT.
unsafe fn handle_msvcr71_sscanf_scalar() -> bool {
    let esp = read_reg32(ESP);
    let mut at = match safe_read32s(esp.wrapping_add(4)) { Ok(v) if v != 0 => v, _ => return false };
    let fmt = match safe_read32s(esp.wrapping_add(8)) { Ok(v) if v != 0 => v, _ => return false };
    let output = match safe_read32s(esp.wrapping_add(12)) { Ok(v) if v != 0 => v, _ => return false };
    if hc_safe_read8(fmt).ok() != Some(b'%' as i32) || hc_safe_read8(fmt.wrapping_add(2)).ok() != Some(0) {
        return false;
    }
    let kind = match hc_safe_read8(fmt.wrapping_add(1)) { Ok(v) => v, Err(_) => return false };
    if kind != b'd' as i32 && kind != b'u' as i32 && kind != b'f' as i32 { return false; }
    if hc_safe_read8(output).is_err() { return false; }

    while scanf_space(match hc_safe_read8(at) { Ok(v) => v, Err(_) => return false }) {
        at = at.wrapping_add(1);
    }
    let first = match hc_safe_read8(at) { Ok(v) => v, Err(_) => return false };
    if first == 0 { write_reg32(EAX, -1); return true; }
    let mut negative = false;
    if first == b'+' as i32 || first == b'-' as i32 {
        negative = first == b'-' as i32;
        at = at.wrapping_add(1);
    }

    let parsed_bits = if kind == b'd' as i32 || kind == b'u' as i32 {
            let mut value: u64 = 0;
            let mut digits = 0u32;
            loop {
                let byte = match hc_safe_read8(at) { Ok(v) => v, Err(_) => return false };
                if byte < b'0' as i32 || byte > b'9' as i32 { break; }
                value = value * 10 + (byte - b'0' as i32) as u64;
                digits += 1;
                at = at.wrapping_add(1);
                if value > 0xffff_ffff { return false; }
            }
            if digits == 0 {
                write_reg32(EAX, 0);
                return true;
            }
            if kind == b'd' as i32 {
                let limit = if negative { 0x8000_0000u64 } else { 0x7fff_ffffu64 };
                if value > limit { return false; }
            }
            let mut bits = value as u32;
            if negative { bits = bits.wrapping_neg(); }
            bits
    } else {
            // Let the original CRT decide legacy spellings such as 1.#INF/NAN.
            let lead = match hc_safe_read8(at) { Ok(v) => v, Err(_) => return false };
            if lead == b'i' as i32 || lead == b'I' as i32 || lead == b'n' as i32 || lead == b'N' as i32 {
                return false;
            }
            let mut value = 0.0f64;
            let mut digits = 0u32;
            loop {
                let byte = match hc_safe_read8(at) { Ok(v) => v, Err(_) => return false };
                if byte < b'0' as i32 || byte > b'9' as i32 { break; }
                value = value * 10.0 + (byte - b'0' as i32) as f64;
                digits += 1;
                at = at.wrapping_add(1);
                if digits > 128 { return false; }
            }
            if hc_safe_read8(at).ok() == Some(b'.' as i32) {
                at = at.wrapping_add(1);
                let mut scale = 0.1f64;
                loop {
                    let byte = match hc_safe_read8(at) { Ok(v) => v, Err(_) => return false };
                    if byte < b'0' as i32 || byte > b'9' as i32 { break; }
                    value += (byte - b'0' as i32) as f64 * scale;
                    scale *= 0.1;
                    digits += 1;
                    at = at.wrapping_add(1);
                    if digits > 128 { return false; }
                }
            }
            if digits == 0 {
                write_reg32(EAX, 0);
                return true;
            }
            let marker = match hc_safe_read8(at) { Ok(v) => v, Err(_) => return false };
            if marker == b'e' as i32 || marker == b'E' as i32 {
                at = at.wrapping_add(1);
                let mut exp_negative = false;
                let exp_sign = match hc_safe_read8(at) { Ok(v) => v, Err(_) => return false };
                if exp_sign == b'+' as i32 || exp_sign == b'-' as i32 {
                    exp_negative = exp_sign == b'-' as i32;
                    at = at.wrapping_add(1);
                }
                let mut exponent = 0u32;
                let mut exp_digits = 0u32;
                loop {
                    let byte = match hc_safe_read8(at) { Ok(v) => v, Err(_) => return false };
                    if byte < b'0' as i32 || byte > b'9' as i32 { break; }
                    exponent = exponent.saturating_mul(10).saturating_add((byte - b'0' as i32) as u32);
                    exp_digits += 1;
                    at = at.wrapping_add(1);
                    if exponent > 64 { return false; }
                }
                if exp_digits == 0 { return false; }
                for _ in 0..exponent {
                    value = if exp_negative { value * 0.1 } else { value * 10.0 };
                }
            }
            if negative { value = -value; }
            (value as f32).to_bits()
    };

    if safe_write32(output, parsed_bits as i32).is_err() { return false; }
    write_reg32(EAX, 1);
    true
}

/// MSVCR71.dll 7.10.3052.4 @ RVA 0x32ec: ASCII-only `_stricmp`.
unsafe fn handle_msvcr71_stricmp() -> bool {
    let esp = read_reg32(ESP);
    let mut left = match safe_read32s(esp.wrapping_add(4)) { Ok(v) if v != 0 => v, _ => return false };
    let mut right = match safe_read32s(esp.wrapping_add(8)) { Ok(v) if v != 0 => v, _ => return false };
    for _ in 0..16_384 {
        let a = match hc_safe_read8(left) { Ok(v) => v, Err(_) => return false };
        let b = match hc_safe_read8(right) { Ok(v) => v, Err(_) => return false };
        if a == b {
            if a == 0 { write_reg32(EAX, 0); return true; }
        }
        else {
            let folded_a = if a >= 0x41 && a <= 0x5a { a + 0x20 } else { a };
            let folded_b = if b >= 0x41 && b <= 0x5a { b + 0x20 } else { b };
            if folded_a != folded_b {
                write_reg32(EAX, if folded_a < folded_b { -1 } else { 1 });
                return true;
            }
        }
        left = left.wrapping_add(1);
        right = right.wrapping_add(1);
    }
    false
}

/// lotrbfme.exe 1.03 FR @ 0x00b47940: blend three color bytes per BGRA pixel
/// using an 8-bit opacity stream while preserving destination alpha.
unsafe fn handle_pixel_alpha_blend() -> bool {
    let esp = read_reg32(ESP);
    let source = match safe_read32s(esp.wrapping_add(4)) { Ok(v) if v != 0 => v, _ => return false };
    let destination = match safe_read32s(esp.wrapping_add(8)) { Ok(v) if v != 0 => v, _ => return false };
    let alpha = match safe_read32s(esp.wrapping_add(12)) { Ok(v) if v != 0 => v, _ => return false };
    let count = match safe_read32s(esp.wrapping_add(16)) {
        Ok(v) if v > 0 && v <= 0x0010_0000 => v,
        _ => return false,
    };

    // Validate the complete input/output spans before the first mutation. A
    // fault must fall back to the original with destination still untouched.
    for i in 0..count {
        let offset = i.wrapping_mul(4);
        if safe_read32s(source.wrapping_add(offset)).is_err()
            || safe_read32s(destination.wrapping_add(offset)).is_err()
            || safe_read32s(alpha.wrapping_add(offset)).is_err() {
            return false;
        }
    }

    for i in 0..count {
        let offset = i.wrapping_mul(4);
        let src = safe_read32s(source.wrapping_add(offset)).unwrap() as u32;
        let old = safe_read32s(destination.wrapping_add(offset)).unwrap() as u32;
        let opacity = (safe_read32s(alpha.wrapping_add(offset)).unwrap() as u32) & 0xff;
        let inverse = 255u32 - opacity;
        let mut result = old & 0xff00_0000;
        for shift in [0u32, 8, 16] {
            let src_byte = (src >> shift) & 0xff;
            let dst_byte = (old >> shift) & 0xff;
            let blended = ((src_byte * opacity) >> 8) + ((dst_byte * inverse) >> 8);
            result |= (blended & 0xff) << shift;
        }
        if safe_write32(destination.wrapping_add(offset), result as i32).is_err() { return false; }
    }
    write_reg32(EAX, 0);
    true
}

const IDCT_FIX_0_298631336: i32 = 2446;
const IDCT_FIX_0_390180644: i32 = 3196;
const IDCT_FIX_0_541196100: i32 = 4433;
const IDCT_FIX_0_765366865: i32 = 6270;
const IDCT_FIX_0_899976223: i32 = 7373;
const IDCT_FIX_1_175875602: i32 = 9633;
const IDCT_FIX_1_501321110: i32 = 12299;
const IDCT_FIX_1_847759065: i32 = 15137;
const IDCT_FIX_1_961570560: i32 = 16069;
const IDCT_FIX_2_053119869: i32 = 16819;
const IDCT_FIX_2_562915447: i32 = 20995;
const IDCT_FIX_3_072711026: i32 = 25172;

#[inline(always)]
fn idct_descale(value: i32, bits: u32) -> i32 {
    value.wrapping_add(1i32 << (bits - 1)) >> bits
}

#[inline(always)]
fn idct_1d(v: [i32; 8], shift: u32) -> [i32; 8] {
    let mut z2 = v[2];
    let mut z3 = v[6];
    let mut z1 = z2.wrapping_add(z3).wrapping_mul(IDCT_FIX_0_541196100);
    let mut tmp2 = z1.wrapping_sub(z3.wrapping_mul(IDCT_FIX_1_847759065));
    let mut tmp3 = z1.wrapping_add(z2.wrapping_mul(IDCT_FIX_0_765366865));
    let mut tmp0 = v[0].wrapping_add(v[4]).wrapping_shl(13);
    let mut tmp1 = v[0].wrapping_sub(v[4]).wrapping_shl(13);
    let tmp10 = tmp0.wrapping_add(tmp3);
    let tmp13 = tmp0.wrapping_sub(tmp3);
    let tmp11 = tmp1.wrapping_add(tmp2);
    let tmp12 = tmp1.wrapping_sub(tmp2);

    tmp0 = v[7];
    tmp1 = v[5];
    tmp2 = v[3];
    tmp3 = v[1];
    z1 = tmp0.wrapping_add(tmp3);
    z2 = tmp1.wrapping_add(tmp2);
    z3 = tmp0.wrapping_add(tmp2);
    let mut z4 = tmp1.wrapping_add(tmp3);
    let z5 = z3.wrapping_add(z4).wrapping_mul(IDCT_FIX_1_175875602);
    tmp0 = tmp0.wrapping_mul(IDCT_FIX_0_298631336);
    tmp1 = tmp1.wrapping_mul(IDCT_FIX_2_053119869);
    tmp2 = tmp2.wrapping_mul(IDCT_FIX_3_072711026);
    tmp3 = tmp3.wrapping_mul(IDCT_FIX_1_501321110);
    z1 = z1.wrapping_mul(-IDCT_FIX_0_899976223);
    z2 = z2.wrapping_mul(-IDCT_FIX_2_562915447);
    z3 = z3.wrapping_mul(-IDCT_FIX_1_961570560).wrapping_add(z5);
    z4 = z4.wrapping_mul(-IDCT_FIX_0_390180644).wrapping_add(z5);
    tmp0 = tmp0.wrapping_add(z1.wrapping_add(z3));
    tmp1 = tmp1.wrapping_add(z2.wrapping_add(z4));
    tmp2 = tmp2.wrapping_add(z2.wrapping_add(z3));
    tmp3 = tmp3.wrapping_add(z1.wrapping_add(z4));
    [
        idct_descale(tmp10.wrapping_add(tmp3), shift),
        idct_descale(tmp11.wrapping_add(tmp2), shift),
        idct_descale(tmp12.wrapping_add(tmp1), shift),
        idct_descale(tmp13.wrapping_add(tmp0), shift),
        idct_descale(tmp13.wrapping_sub(tmp0), shift),
        idct_descale(tmp12.wrapping_sub(tmp1), shift),
        idct_descale(tmp11.wrapping_sub(tmp2), shift),
        idct_descale(tmp10.wrapping_sub(tmp3), shift),
    ]
}

#[inline(always)]
unsafe fn bfme_read_i16(address: i32) -> Option<i32> {
    let lo = hc_safe_read8(address).ok()? as u16;
    let hi = hc_safe_read8(address.wrapping_add(1)).ok()? as u16;
    Some(((lo | hi << 8) as i16) as i32)
}

/// lotrbfme.exe 1.03 FR @ 0x00ed1aa0: statically linked IJG
/// jpeg_idct_islow. It dequantizes one 8x8 block, applies the two integer IDCT
/// passes and writes eight bytes to each supplied output row.
unsafe fn handle_jpeg_idct_islow() -> bool {
    let esp = read_reg32(ESP);
    let cinfo = match safe_read32s(esp.wrapping_add(4)) { Ok(v) if v != 0 => v, _ => return false };
    let component = match safe_read32s(esp.wrapping_add(8)) { Ok(v) if v != 0 => v, _ => return false };
    let coefficients = match safe_read32s(esp.wrapping_add(12)) { Ok(v) if v != 0 => v, _ => return false };
    let output_rows = match safe_read32s(esp.wrapping_add(16)) { Ok(v) if v != 0 => v, _ => return false };
    let output_column = match safe_read32s(esp.wrapping_add(20)) { Ok(v) if v >= 0 && v <= 0x0010_0000 => v, _ => return false };
    let range_limit = match safe_read32s(cinfo.wrapping_add(0x148)) { Ok(v) if v != 0 => v.wrapping_add(0x80), _ => return false };
    let quantization = match safe_read32s(component.wrapping_add(0x50)) { Ok(v) if v != 0 => v, _ => return false };

    let mut coef = [0i32; 64];
    let mut quant = [0i32; 64];
    let mut rows = [0i32; 8];
    for i in 0..64i32 {
        coef[i as usize] = match bfme_read_i16(coefficients.wrapping_add(i * 2)) { Some(v) => v, None => return false };
        quant[i as usize] = match bfme_read_i16(quantization.wrapping_add(i * 2)) { Some(v) => v, None => return false };
    }
    for row in 0..8i32 {
        rows[row as usize] = match safe_read32s(output_rows.wrapping_add(row * 4)) {
            Ok(v) if v != 0 => v.wrapping_add(output_column),
            _ => return false,
        };
    }

    let mut workspace = [0i32; 64];
    for column in 0..8usize {
        if (1..8usize).all(|row| coef[row * 8 + column] == 0) {
            let dc = coef[column].wrapping_mul(quant[column]).wrapping_shl(2);
            for row in 0..8usize { workspace[row * 8 + column] = dc; }
            continue;
        }
        let mut input = [0i32; 8];
        for row in 0..8usize {
            let index = row * 8 + column;
            input[row] = coef[index].wrapping_mul(quant[index]);
        }
        let output = idct_1d(input, 11);
        for row in 0..8usize { workspace[row * 8 + column] = output[row]; }
    }

    for row in 0..8usize {
        let base = row * 8;
        let mut output = [0i32; 8];
        if (1..8usize).all(|x| workspace[base + x] == 0) {
            let value = idct_descale(workspace[base], 5);
            output.fill(value);
        } else {
            let mut input = [0i32; 8];
            input.copy_from_slice(&workspace[base..base + 8]);
            output = idct_1d(input, 18);
        }
        for x in 0..8usize {
            let sample = match hc_safe_read8(range_limit.wrapping_add(output[x] & 1023)) {
                Ok(v) => v,
                Err(_) => return false,
            };
            if safe_write8(rows[row].wrapping_add(x as i32), sample).is_err() { return false; }
        }
    }
    write_reg32(EAX, 0);
    true
}

/// lotrbfme.exe 1.03 FR @ 0x00e2dc30: consume the complete inner loop which
/// blends four float4 streams and applies a scalar. The exact byte-signature
/// hook enters after the surrounding function has acquired all buffers.
unsafe fn handle_vertex_blend() -> bool {
    let frame = read_reg32(EBP);
    let owner = read_reg32(EBX);
    if frame == 0 || owner == 0 { return false; }
    let stream = match safe_read32s(owner.wrapping_add(4)) { Ok(v) if v != 0 => v, _ => return false };
    let count = match safe_read32s(stream.wrapping_add(0x68)) { Ok(v) if v > 0 && v <= 0x0100_0000 => v, _ => return false };
    let output = match safe_read32s(frame.wrapping_sub(0x10)) { Ok(v) if v != 0 => v, _ => return false };
    let source_a = match safe_read32s(frame.wrapping_sub(0x18)) { Ok(v) if v != 0 => v, _ => return false };
    let source_b = match safe_read32s(frame.wrapping_sub(0x04)) { Ok(v) if v != 0 => v, _ => return false };
    let source_c = match safe_read32s(frame.wrapping_sub(0x08)) { Ok(v) if v != 0 => v, _ => return false };
    let source_d = match safe_read32s(frame.wrapping_sub(0x1c)) { Ok(v) if v != 0 => v, _ => return false };
    let scale = match read_f32(0x0108_3b6c) { Some(v) => v as f64, None => return false };

    for i in 0..count {
        let offset = i.wrapping_mul(16);
        for lane in 0..4i32 {
            let lane_offset = offset.wrapping_add(lane * 4);
            let a = match read_f32(source_a.wrapping_add(lane_offset)) { Some(v) => v as f64, None => return false };
            let b = match read_f32(source_b.wrapping_add(lane_offset)) { Some(v) => v as f64, None => return false };
            let c = match read_f32(source_c.wrapping_add(lane_offset)) { Some(v) => v as f64, None => return false };
            let d = match read_f32(source_d.wrapping_add(lane_offset)) { Some(v) => v as f64, None => return false };
            if !write_f32(output.wrapping_add(lane_offset), (((a + b) + c) + d) * scale) { return false; }
        }
    }
    write_reg32(EDX, count);
    write_reg32(ESI, source_b);
    true
}

/// lotrbfme.exe 1.03 FR @ 0x00c2b870: in-order successor for the
/// parent/left/right tree-node layout used by BFME's STL containers.
unsafe fn handle_tree_successor() -> bool {
    let esp = read_reg32(ESP);
    let mut node = match safe_read32s(esp.wrapping_add(4)) {
        Ok(v) if v != 0 => v,
        _ => return false,
    };
    let right = match safe_read32s(node.wrapping_add(0x0c)) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if right != 0 {
        node = right;
        for _ in 0..65536u32 {
            let left = match safe_read32s(node.wrapping_add(0x08)) {
                Ok(v) => v,
                Err(_) => return false,
            };
            if left == 0 {
                write_reg32(EAX, node);
                return true;
            }
            node = left;
        }
        return false;
    }

    let mut parent = match safe_read32s(node.wrapping_add(0x04)) {
        Ok(v) => v,
        Err(_) => return false,
    };
    for _ in 0..65536u32 {
        let parent_right = match safe_read32s(parent.wrapping_add(0x0c)) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if node != parent_right {
            let node_right = match safe_read32s(node.wrapping_add(0x0c)) {
                Ok(v) => v,
                Err(_) => return false,
            };
            write_reg32(EAX, if node_right == parent { node } else { parent });
            return true;
        }
        node = parent;
        parent = match safe_read32s(parent.wrapping_add(0x04)) {
            Ok(v) => v,
            Err(_) => return false,
        };
    }
    false
}

#[inline(always)]
unsafe fn read_f32(address: i32) -> Option<f32> {
    Some(f32::from_bits(safe_read32s(address).ok()? as u32))
}

#[inline(always)]
unsafe fn write_f32(address: i32, value: f64) -> bool {
    safe_write32(address, (value as f32).to_bits() as i32).is_ok()
}

/// 0x00cd2d10: compose two six-float 2D affine matrices. Every binary32
/// product and the longest three-term sum fits exactly in a binary64
/// significand, matching the original x87 extended intermediates before the
/// final binary32 store. Inputs are loaded first because the output may alias.
unsafe fn handle_matrix_multiply() -> bool {
    let esp = read_reg32(ESP);
    let left_address = match safe_read32s(esp.wrapping_add(4)) { Ok(v) if v != 0 => v, _ => return false };
    let right_address = match safe_read32s(esp.wrapping_add(8)) { Ok(v) if v != 0 => v, _ => return false };
    let output = match safe_read32s(esp.wrapping_add(12)) { Ok(v) if v != 0 => v, _ => return false };
    let mut left = [0f64; 6];
    let mut right = [0f64; 6];
    for i in 0..6i32 {
        left[i as usize] = match read_f32(left_address.wrapping_add(i * 4)) { Some(v) => v as f64, None => return false };
        right[i as usize] = match read_f32(right_address.wrapping_add(i * 4)) { Some(v) => v as f64, None => return false };
    }
    let result = [
        left[2] * right[1] + left[0] * right[0],
        left[3] * right[1] + left[1] * right[0],
        right[3] * left[2] + right[2] * left[0],
        right[3] * left[3] + right[2] * left[1],
        right[5] * left[2] + right[4] * left[0] + left[4],
        right[5] * left[3] + right[4] * left[1] + left[5],
    ];
    for i in 0..6i32 {
        if !write_f32(output.wrapping_add(i * 4), result[i as usize]) { return false; }
    }
    write_reg32(EAX, output);
    true
}

const MATRIX_DEPTH: i32 = 0x3b8;
const MATRIX_STACK: i32 = 0x38;
const TRANSFORM_DEPTH: i32 = 0x3bc;
const TRANSFORM_SOURCE: i32 = 0x20;
const TRANSFORM_STACK: i32 = 0x238;

#[inline(always)]
unsafe fn copy_matrix32(source: i32, destination: i32) -> bool {
    for offset in (0..32i32).step_by(4) {
        let value = match safe_read32s(source.wrapping_add(offset)) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if safe_write32(destination.wrapping_add(offset), value).is_err() { return false; }
    }
    true
}

/// 0x00cd2b50: save the current 32-byte matrix state into the inline stack.
unsafe fn handle_matrix_push() -> bool {
    let object = read_reg32(ECX);
    if object == 0 { return false; }
    let depth = match safe_read32s(object.wrapping_add(MATRIX_DEPTH)) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let destination = object.wrapping_add(MATRIX_STACK).wrapping_add(depth.wrapping_mul(32));
    if !copy_matrix32(object, destination) { return false; }
    if safe_write32(object.wrapping_add(MATRIX_DEPTH), depth.wrapping_add(1)).is_err() { return false; }
    write_reg32(EAX, object);
    true
}

/// 0x00cd2b80: restore the most recently saved 32-byte matrix state.
unsafe fn handle_matrix_pop() -> bool {
    let object = read_reg32(ECX);
    if object == 0 { return false; }
    let depth = match safe_read32s(object.wrapping_add(MATRIX_DEPTH)) {
        Ok(v) => v.wrapping_sub(1),
        Err(_) => return false,
    };
    if safe_write32(object.wrapping_add(MATRIX_DEPTH), depth).is_err() { return false; }
    let offset = depth.wrapping_mul(32);
    let source = object.wrapping_add(MATRIX_STACK).wrapping_add(offset);
    if !copy_matrix32(source, object) { return false; }
    write_reg32(EAX, offset);
    true
}

/// 0x00cd2c80: save the current six-float transform in the object's second
/// inline stack. The corresponding pop uses a guest wrapper so its original
/// transform-update callback still runs after the WASM copy.
unsafe fn handle_transform_push() -> bool {
    let object = read_reg32(ECX);
    if object == 0 { return false; }
    let depth = match safe_read32s(object.wrapping_add(TRANSFORM_DEPTH)) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let destination = object.wrapping_add(TRANSFORM_STACK).wrapping_add(depth.wrapping_mul(24));
    for offset in (0..24i32).step_by(4) {
        let value = match safe_read32s(object.wrapping_add(TRANSFORM_SOURCE).wrapping_add(offset)) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if safe_write32(destination.wrapping_add(offset), value).is_err() { return false; }
    }
    if safe_write32(object.wrapping_add(TRANSFORM_DEPTH), depth.wrapping_add(1)).is_err() { return false; }
    write_reg32(EAX, destination);
    true
}

unsafe fn handle_transform_pop() -> bool {
    let object = read_reg32(ECX);
    if object == 0 { return false; }
    let depth = match safe_read32s(object.wrapping_add(TRANSFORM_DEPTH)) {
        Ok(v) => v.wrapping_sub(1),
        Err(_) => return false,
    };
    if safe_write32(object.wrapping_add(TRANSFORM_DEPTH), depth).is_err() { return false; }
    let source = object.wrapping_add(TRANSFORM_STACK).wrapping_add(depth.wrapping_mul(24));
    let destination = object.wrapping_add(TRANSFORM_SOURCE);
    for offset in (0..24i32).step_by(4) {
        let value = match safe_read32s(source.wrapping_add(offset)) { Ok(v) => v, Err(_) => return false };
        if safe_write32(destination.wrapping_add(offset), value).is_err() { return false; }
    }
    write_reg32(EAX, destination);
    true
}

/// 0x00cd2bb0: component-wise scale of the first four floats and translation
/// of the next four. The guest wrapper retains the original update callback.
unsafe fn handle_matrix_adjust() -> bool {
    let object = read_reg32(ECX);
    let esp = read_reg32(ESP);
    let adjustment = match safe_read32s(esp.wrapping_add(4)) { Ok(v) if v != 0 => v, _ => return false };
    if object == 0 { return false; }
    let mut current = [0f64; 8];
    let mut delta = [0f64; 8];
    for i in 0..8i32 {
        current[i as usize] = match read_f32(object.wrapping_add(i * 4)) { Some(v) => v as f64, None => return false };
        delta[i as usize] = match read_f32(adjustment.wrapping_add(i * 4)) { Some(v) => v as f64, None => return false };
    }
    for i in 0..8i32 {
        let value = if i < 4 {
            delta[i as usize] * current[i as usize]
        } else {
            delta[i as usize] + current[i as usize]
        };
        if !write_f32(object.wrapping_add(i * 4), value) { return false; }
    }
    write_reg32(EAX, object);
    true
}

const STRINGBASE_LOCK_GUARD: i32 = 0x01336e2c;

#[inline(always)]
unsafe fn read_u16(address: i32) -> Option<u32> {
    let lo = hc_safe_read8(address).ok()? as u32;
    let hi = hc_safe_read8(address.wrapping_add(1)).ok()? as u32;
    Some(lo | hi << 8)
}

#[inline(always)]
unsafe fn read_stringbase(object: i32) -> Option<(i32, u32)> {
    let storage = safe_read32s(object).ok()?;
    if storage == 0 { return Some((0, 0)); }
    let length = read_u16(storage.wrapping_add(4))?;
    Some((storage.wrapping_add(8), length))
}

/// lotrbfme.exe 1.03 FR @ 0x008a0270. The original walks a single-linked
/// container chain (root at this+0x2c, next at node+0x60) and compares the
/// stringbase<char> key at node+0x0c with its sole stack argument.
unsafe fn handle_stringbase_find() -> bool {
    let container = read_reg32(ECX);
    let esp = read_reg32(ESP);
    let key_object = match safe_read32s(esp.wrapping_add(4)) {
        Ok(v) if v != 0 => v,
        _ => return false,
    };
    if container == 0 { return false; }
    let (key_chars, key_length) = match read_stringbase(key_object) {
        Some(v) => v,
        None => return false,
    };
    let mut node = match safe_read32s(container.wrapping_add(0x2c)) {
        Ok(v) => v,
        Err(_) => return false,
    };

    // Valid BFME chains are short. The cap is only a corrupt-cycle guard; a
    // cap hit declines to the JS safety tier without having modified memory.
    for _ in 0..65536u32 {
        if node == 0 {
            write_reg32(EAX, 0);
            return true;
        }
        let node_key_object = node.wrapping_add(0x0c);
        let (node_chars, node_length) = match read_stringbase(node_key_object) {
            Some(v) => v,
            None => return false,
        };
        if node_length == key_length {
            let mut equal = true;
            for i in 0..key_length {
                let lhs = match hc_safe_read8(node_chars.wrapping_add(i as i32)) {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                let rhs = match hc_safe_read8(key_chars.wrapping_add(i as i32)) {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                if lhs != rhs {
                    equal = false;
                    break;
                }
            }
            if equal {
                write_reg32(EAX, node);
                return true;
            }
        }
        node = match safe_read32s(node.wrapping_add(0x60)) {
            Ok(v) => v,
            Err(_) => return false,
        };
    }
    false
}

unsafe fn stringbase_lock_ready() -> bool {
    matches!(hc_safe_read8(STRINGBASE_LOCK_GUARD), Ok(v) if v & 1 != 0)
}

/// `stringbase<char>` release/reset @ 0x00c87940. Null and shared backing
/// stores need no allocator call, so they can be completed without entering
/// BFME's global reference-count lock. A unique store declines to guest code,
/// which performs the real free.
unsafe fn handle_stringbase_release() -> bool {
    if !stringbase_lock_ready() { return false; }
    let object = read_reg32(ECX);
    if object == 0 { return false; }
    let storage = match safe_read32s(object) { Ok(v) => v, Err(_) => return false };
    if storage == 0 {
        if safe_write32(object, 0).is_err() { return false; }
        write_reg32(EAX, 0);
        return true;
    }
    let refs = match safe_read32s(storage) { Ok(v) => v as u32, Err(_) => return false };
    if refs <= 1 { return false; }
    if safe_write32(storage, refs.wrapping_sub(1) as i32).is_err() { return false; }
    if safe_write32(object, 0).is_err() { return false; }
    write_reg32(EAX, 0);
    true
}

/// Copy construction @ 0x00c87b60. Guest threads are cooperatively serialized
/// by v86, so the source refcount update is indivisible with respect to another
/// guest thread even without the emulated process-wide lock.
unsafe fn handle_stringbase_copy() -> bool {
    if !stringbase_lock_ready() { return false; }
    let destination = read_reg32(ECX);
    let esp = read_reg32(ESP);
    let source = match safe_read32s(esp.wrapping_add(4)) { Ok(v) => v, Err(_) => return false };
    if destination == 0 || source == 0 { return false; }
    let storage = match safe_read32s(source) { Ok(v) => v, Err(_) => return false };
    if storage != 0 {
        let refs = match safe_read32s(storage) { Ok(v) => v as u32, Err(_) => return false };
        if safe_write32(storage, refs.wrapping_add(1) as i32).is_err() { return false; }
    }
    if safe_write32(destination, storage).is_err() { return false; }
    write_reg32(EAX, destination);
    true
}

/// Assignment @ 0x00c87c90. The fast path accepts self-assignment, an empty
/// destination, or a shared old value. A uniquely-owned old value declines so
/// the original function can invoke BFME's allocator/free routine.
unsafe fn handle_stringbase_assign() -> bool {
    if !stringbase_lock_ready() { return false; }
    let destination = read_reg32(ECX);
    let esp = read_reg32(ESP);
    let source = match safe_read32s(esp.wrapping_add(4)) { Ok(v) => v, Err(_) => return false };
    if destination == 0 || source == 0 { return false; }
    if destination == source {
        write_reg32(EAX, destination);
        return true;
    }
    let old_storage = match safe_read32s(destination) { Ok(v) => v, Err(_) => return false };
    let new_storage = match safe_read32s(source) { Ok(v) => v, Err(_) => return false };
    if old_storage == new_storage {
        write_reg32(EAX, destination);
        return true;
    }
    if old_storage != 0 {
        let refs = match safe_read32s(old_storage) { Ok(v) => v as u32, Err(_) => return false };
        if refs <= 1 { return false; }
        if safe_write32(old_storage, refs.wrapping_sub(1) as i32).is_err() { return false; }
    }
    if new_storage != 0 {
        let refs = match safe_read32s(new_storage) { Ok(v) => v as u32, Err(_) => return false };
        if safe_write32(new_storage, refs.wrapping_add(1) as i32).is_err() { return false; }
    }
    if safe_write32(destination, new_storage).is_err() { return false; }
    write_reg32(EAX, destination);
    true
}

/// lotrbfme.exe 1.03 FR @ 0x00c87da0 — allocation-free branch of
/// `stringbase<char>::tolower`. A guest-side filter admits only refcount=1 and
/// length<capacity; repeating those checks here keeps the handler independently
/// safe if its dispatch entry is ever reused accidentally.
unsafe fn handle_stringbase_lower() -> bool {
    let object = read_reg32(ECX);
    if object == 0 {
        return false;
    }
    let storage = match safe_read32s(object) {
        Ok(v) if v != 0 => v,
        _ => return false,
    };
    let refs = match safe_read32s(storage) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let len_lo = match hc_safe_read8(storage.wrapping_add(4)) { Ok(v) => v as u32, Err(_) => return false };
    let len_hi = match hc_safe_read8(storage.wrapping_add(5)) { Ok(v) => v as u32, Err(_) => return false };
    let cap_lo = match hc_safe_read8(storage.wrapping_add(6)) { Ok(v) => v as u32, Err(_) => return false };
    let cap_hi = match hc_safe_read8(storage.wrapping_add(7)) { Ok(v) => v as u32, Err(_) => return false };
    let length = len_lo | len_hi << 8;
    let capacity = cap_lo | cap_hi << 8;
    if refs != 1 || length >= capacity {
        return false;
    }

    let chars = storage.wrapping_add(8);
    let mut eax = storage;
    for i in 0..length {
        let at = chars.wrapping_add(i as i32);
        let mut byte = match hc_safe_read8(at) {
            Ok(v) => v as u8,
            Err(_) => return false,
        };
        if byte >= b'A' && byte <= b'Z' {
            byte += b'a' - b'A';
        }
        if safe_write8(at, byte as i32).is_err() {
            return false;
        }
        eax = byte as i32;
    }
    if safe_write8(chars.wrapping_add(length as i32), 0).is_err() {
        return false;
    }
    write_reg32(EAX, eax);
    true
}

/// lotrbfme.exe 1.03 FR @ 0x0048f3c0:
/// `hash = hash * 33 + tolower((signed char)*p)` until NUL.
///
/// This replaces a guest loop whose indirect call enters native MSVCR71 for
/// every byte. The exact 1.03 FR function signature gates registration;
/// malformed/unbounded strings decline to the JS safety tier.
unsafe fn handle_fold33_hash() -> bool {
    let esp = read_reg32(ESP);
    let base = match crate::cpu::cpu::safe_read32s(esp + 4) {
        Ok(v) if v != 0 => v,
        _ => return false,
    };
    let mut hash = 0i32;
    for i in 0..4096i32 {
        let byte = match hc_safe_read8(base.wrapping_add(i)) {
            Ok(v) => v as u8,
            Err(_) => return false,
        };
        if byte == 0 {
            write_reg32(EAX, hash);
            return true;
        }
        let mut folded = (byte as i8) as i32;
        if folded >= 0x41 && folded <= 0x5a {
            folded += 0x20;
        }
        hash = hash.wrapping_mul(33).wrapping_add(folded);
    }
    false
}
