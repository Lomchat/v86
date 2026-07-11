//! Inner-loop HLE handlers for EAGL (EA Graphics Layer) — the 128..=132 band
//! of the hypercall dispatch table (see hypercall.rs try_dispatch and
//! plan/inner-loop-hle.md / plan/eagl-state-commit-hle-rfc.md):
//!
//!   128       shader-constant converter (guest FUN_005cbd17)
//!   129..=131 shader-parameter APPLY converter family (FUN_005c85c1/8303/ad01)
//!   132       state-token dispatcher, hot classes 1/2/8 (FUN_005c97cb)
//!
//! JS twins: src/worker/core/hle-lib/libs/eagl/ (kernels are the validated
//! fallbacks; this module is the production tier). Guest-side plumbing
//! (detection, patching, filter trampoline, config assembly) lives there too.

use std::ptr::{addr_of, addr_of_mut};

use crate::cpu::cpu::{
    read_reg32, safe_read32s, safe_write32, write_reg32, EAX, ECX, ESP,
};
use crate::cpu::hypercall::{hc_safe_read8, hp_ptr, OFF_HC_EAGL_TOKEN_CFG_PTR};

/// Inner-loop band router (handler ids 128..=255, called from
/// hypercall.rs::try_dispatch). All EAGL today; when a second engine lands,
/// promote this into its own band-router module and keep one file per engine.
/// False = guard miss → the JS tier (shadow-validated kernels) completes.
pub(crate) unsafe fn dispatch_inner_loop(handler_id: u8) -> bool {
    match handler_id {
        // 128 = shader-constant converter (FUN_005cbd17): kilo-calls/frame;
        // the JS tier's per-call OUT round-trip was a net regression there.
        128 => handle_eagl_shader_const_convert(),
        // 129-131 = shader-parameter APPLY converter family (the FUN_005cdca7
        // apply walk's pure leaves; semantics in hle-lib/libs/eagl/).
        129 => handle_eagl_apply_reg_int(),
        130 => handle_eagl_apply_reg_float(),
        131 => handle_eagl_apply_packed(),
        // 132 = state-token dispatcher (FUN_005c97cb), hot classes 1/2/8 only —
        // the guest filter routes other classes to the original.
        132 => handle_eagl_token_dispatch(),
        _ => false,
    }
}

/// CRT `_ftol` truncation, EAX (low 32 bits of the i64) only — the guest's
/// mode-1/2 loops consume only EAX. NaN / |x| >= 2^63 → FPU integer-indefinite
/// (0x8000000000000000), whose low 32 bits are 0. Rust `as i64` saturates
/// (wrong for the guest), so the out-of-range cases are handled explicitly.
#[inline(always)]
fn ftol_low32(x: f64) -> u32 {
    if !x.is_finite() || x >= 9223372036854775808.0 || x < -9223372036854775808.0 {
        return 0;
    }
    (x as i64) as u32
}

/// handler_id 128 — EAGL shader-constant converter (guest FUN_005cbd17, stdcall
/// ret 0x10). Semantically identical to the JS kernel in
/// `hle-lib/libs/eagl/descriptor.ts` (RE-verified, unit-tested), executed
/// entirely in WASM so the ~thousands-of-calls/frame path pays no OUT→JS
/// round-trip. Any structural doubt (bad dims, unmapped memory) returns false →
/// the guest OUT falls through to the JS kernel (shadow-validated fallback).
///
///   u32 convert(desc*, dst*, src*, count)   [esp+4, +8, +12, +16]
///   mode=u32[desc], rows=u32[desc+0x14], cols=u32[desc+0x18]
///   r=min(rows,4) c=min(cols,4)
///   src cell f32/u32 @ src + i*0x40 + rr*0x10 + cc*4  (fixed 4x4 staging)
///   dst cell u32     @ dst + i*rows*cols*4 + rr*cols*4 + cc*4  (packed)
///   mode 1: f32→bool(ftol!=0)  2: f32→int(ftol)  3: u32 copy
///   unknown mode → EAX=0x8876086C (D3DERR_INVALIDCALL), no writes
unsafe fn handle_eagl_shader_const_convert() -> bool {
    let esp = read_reg32(ESP);
    let desc = match safe_read32s(esp + 4) { Ok(v) => v, Err(_) => return false };
    let dst = match safe_read32s(esp + 8) { Ok(v) => v, Err(_) => return false };
    let src = match safe_read32s(esp + 12) { Ok(v) => v, Err(_) => return false };
    let count = match safe_read32s(esp + 16) { Ok(v) => v as u32, Err(_) => return false };

    let mode = match safe_read32s(desc) { Ok(v) => v as u32, Err(_) => return false };
    let rows = match safe_read32s(desc + 0x14) { Ok(v) => v as u32, Err(_) => return false };
    let cols = match safe_read32s(desc + 0x18) { Ok(v) => v as u32, Err(_) => return false };

    if mode < 1 || mode > 3 {
        write_reg32(EAX, 0x8876086Cu32 as i32);
        return true;
    }
    // Same sane envelope as the JS guard — beyond it, defer to the guest.
    if rows > 16 || cols > 16 || count > 4096 {
        return false;
    }

    let r = rows.min(4);
    let c = cols.min(4);
    let dst_item_stride = (rows * cols * 4) as i32;
    let dst_row_stride = (cols * 4) as i32;

    for i in 0..count as i32 {
        let src_item = src + i * 0x40;
        let dst_item = dst + i * dst_item_stride;
        for rr in 0..r as i32 {
            let s = src_item + rr * 0x10;
            let d = dst_item + rr * dst_row_stride;
            for cc in 0..c as i32 {
                let sv = match safe_read32s(s + cc * 4) { Ok(v) => v, Err(_) => return false };
                let out = match mode {
                    3 => sv,
                    2 => ftol_low32(f32::from_bits(sv as u32) as f64) as i32,
                    _ => if ftol_low32(f32::from_bits(sv as u32) as f64) != 0 { 1 } else { 0 },
                };
                if safe_write32(d + cc * 4, out).is_err() { return false; }
            }
        }
    }
    write_reg32(EAX, 0);
    true
}

// --- EAGL shader-parameter APPLY converter family (handlers 129-131) ------

#[derive(Clone, Copy, PartialEq)]
enum ApplyFamily {
    /// FUN_005c85c1: 1 = i32→f32 (FILD/FSTP), 2 = u32→f32 (+2^32 correction),
    /// 3 = f32 copy (FLD/FSTP float).
    Int,
    /// FUN_005c8303 / FUN_005cad01: 1|2 = raw u32 copy (MOV), 3 = f32→i32
    /// via CRT _ftol (truncate; NaN/overflow → low32 = 0).
    Float,
}

#[derive(Clone, Copy, PartialEq)]
enum ApplyLayout {
    /// Budget counts 4-float registers; dst advances ceil(rows/4)*16 per column.
    Register,
    /// Budget counts elements; dst advances n*4 per column (FUN_005cad01).
    Packed,
}

const APPLY_E_FAIL: i32 = 0x80004005u32 as i32; // -0x7fffbffb
const APPLY_MAX_DEPTH: u32 = 8;

/// handler_id 129/130/131 — semantically identical to the JS kernels in
/// `hle-lib/libs/eagl/apply-kernels.ts` (RE-verified, unit-tested against a
/// decompilation-transcribed reference). ABI: stdcall ret 0x10, four BY-REF
/// args at [esp+4..]: desc**, src**, dst**, budget*. Any structural doubt
/// (unmapped memory, insane dims, over-deep nesting) restores the four cursor
/// cells to their entry values and returns false — the JS kernel re-runs the
/// whole call from clean cursors (its writes are a superset of any partial
/// WASM-side writes, so the retry converges).
unsafe fn handle_eagl_apply_reg_int() -> bool {
    handle_eagl_apply(ApplyFamily::Int, ApplyLayout::Register)
}
unsafe fn handle_eagl_apply_reg_float() -> bool {
    handle_eagl_apply(ApplyFamily::Float, ApplyLayout::Register)
}
unsafe fn handle_eagl_apply_packed() -> bool {
    handle_eagl_apply(ApplyFamily::Float, ApplyLayout::Packed)
}

unsafe fn handle_eagl_apply(family: ApplyFamily, layout: ApplyLayout) -> bool {
    let esp = read_reg32(ESP);
    let desc_cur = match safe_read32s(esp + 4) { Ok(v) => v, Err(_) => return false };
    let src_cur = match safe_read32s(esp + 8) { Ok(v) => v, Err(_) => return false };
    let dst_cur = match safe_read32s(esp + 12) { Ok(v) => v, Err(_) => return false };
    let budget = match safe_read32s(esp + 16) { Ok(v) => v, Err(_) => return false };

    // Entry snapshot of the by-ref cells for clean fall-through on abort.
    let d0 = match safe_read32s(desc_cur) { Ok(v) => v, Err(_) => return false };
    let s0 = match safe_read32s(src_cur) { Ok(v) => v, Err(_) => return false };
    let t0 = match safe_read32s(dst_cur) { Ok(v) => v, Err(_) => return false };
    let b0 = match safe_read32s(budget) { Ok(v) => v, Err(_) => return false };
    // Same sane envelope as the JS guard — beyond it, defer to the guest.
    if b0 as u32 > 4096 {
        return false;
    }

    match eagl_apply_walk(family, layout, desc_cur, src_cur, dst_cur, budget, 0) {
        Ok(eax) => {
            write_reg32(EAX, eax);
            true
        },
        Err(()) => {
            // Best-effort cursor restore; cells were readable at entry.
            let _ = safe_write32(desc_cur, d0);
            let _ = safe_write32(src_cur, s0);
            let _ = safe_write32(dst_cur, t0);
            let _ = safe_write32(budget, b0);
            false
        },
    }
}

/// The recursive walk. Err(()) = abort to JS (memory fault / insane shape);
/// Ok(eax) = completed with the guest-visible result (0 or E_FAIL).
unsafe fn eagl_apply_walk(
    family: ApplyFamily,
    layout: ApplyLayout,
    desc_cur: i32,
    src_cur: i32,
    dst_cur: i32,
    budget: i32,
    depth: u32,
) -> Result<i32, ()> {
    if depth > APPLY_MAX_DEPTH {
        return Err(());
    }
    let d = safe_read32s(desc_cur).map_err(|_| ())?;
    let cls = safe_read32s(d + 4).map_err(|_| ())?;
    let mut items = safe_read32s(d + 0x10).map_err(|_| ())? as u32;
    if items == 0 {
        items = 1;
    }
    if items > 1024 {
        return Err(());
    }

    if cls >= 0 && cls <= 3 {
        let mode = safe_read32s(d).map_err(|_| ())? as u32;
        let rows = safe_read32s(d + 0x14).map_err(|_| ())? as u32;
        let cols = safe_read32s(d + 0x18).map_err(|_| ())? as u32;
        if mode < 1 || mode > 3 {
            return Ok(APPLY_E_FAIL);
        }
        if rows > 64 || cols > 64 {
            return Err(());
        }

        // Sticky clamp state (the guest's spilled locals, set once per call).
        let mut regs = (rows >> 2) + ((rows & 3 != 0) as u32);
        let mut elems = rows;
        let mut n = rows;

        for _ in 0..items {
            if safe_read32s(budget).map_err(|_| ())? == 0 {
                break;
            }
            let src_base = safe_read32s(src_cur).map_err(|_| ())?;
            for j in 0..cols as i32 {
                let rem = safe_read32s(budget).map_err(|_| ())? as u32;
                if rem == 0 {
                    break;
                }
                let (count, dst_step, budget_step) = match layout {
                    ApplyLayout::Register => {
                        if rem < regs {
                            elems = rem * 4;
                            regs = rem;
                        }
                        (elems, regs * 16, regs)
                    },
                    ApplyLayout::Packed => {
                        if rem < n {
                            n = rem;
                        }
                        (n, n * 4, n)
                    },
                };
                let dst_base = safe_read32s(dst_cur).map_err(|_| ())?;
                for e in 0..count as i32 {
                    let s = src_base + (j + e * cols as i32) * 4;
                    let dst = dst_base + e * 4;
                    let sv = safe_read32s(s).map_err(|_| ())?;
                    let out = match family {
                        ApplyFamily::Int => match mode {
                            1 => ((sv as f64) as f32).to_bits() as i32,        // FILD i32 → FSTP f32
                            2 => ((sv as u32 as f64) as f32).to_bits() as i32, // u32 → f32 (FADD 2^32)
                            _ => ((f32::from_bits(sv as u32) as f64) as f32).to_bits() as i32, // FLD/FSTP
                        },
                        ApplyFamily::Float => match mode {
                            3 => ftol_low32(f32::from_bits(sv as u32) as f64) as i32, // _ftol
                            _ => sv,                                                  // MOV copy
                        },
                    };
                    safe_write32(dst, out).map_err(|_| ())?;
                }
                safe_write32(dst_cur, dst_base.wrapping_add(dst_step as i32)).map_err(|_| ())?;
                safe_write32(budget, (rem - budget_step) as i32).map_err(|_| ())?;
            }
            safe_write32(
                src_cur,
                src_base.wrapping_add((cols * rows * 4) as i32),
            ).map_err(|_| ())?;
        }
        safe_write32(desc_cur, d.wrapping_add(0x1c)).map_err(|_| ())?;
        return Ok(0);
    }

    if cls == 5 {
        let children = safe_read32s(d + 0x14).map_err(|_| ())? as u32;
        if children > 64 {
            return Err(());
        }
        safe_write32(desc_cur, d.wrapping_add(0x18)).map_err(|_| ())?;
        let mut ret = 0i32;
        for _ in 0..items {
            if safe_read32s(budget).map_err(|_| ())? == 0 {
                return Ok(ret);
            }
            safe_write32(desc_cur, d.wrapping_add(0x18)).map_err(|_| ())?;
            for _ in 0..children {
                if safe_read32s(budget).map_err(|_| ())? == 0 {
                    break;
                }
                // FUN_005cad01's container recurses into the REGISTER-layout
                // float walk (FUN_005c8303); the other two into themselves.
                ret = match layout {
                    ApplyLayout::Packed => eagl_apply_walk(
                        ApplyFamily::Float, ApplyLayout::Register,
                        desc_cur, src_cur, dst_cur, budget, depth + 1)?,
                    ApplyLayout::Register => eagl_apply_walk(
                        family, layout, desc_cur, src_cur, dst_cur, budget, depth + 1)?,
                };
                if ret < 0 {
                    return Ok(ret);
                }
            }
        }
        return Ok(ret);
    }

    Ok(APPLY_E_FAIL)
}

/// handler_id 132 — EAGL→D3D9 state-token dispatcher, hot classes only
/// (plan/eagl-state-commit-hle-rfc.md; guest FUN_005c97cb, __thiscall RET 8:
/// ECX = EAGL device ctx, [esp+4] = token node, [esp+8] = stage-or-index).
///
/// A guest-side filter trampoline (hle-lib libs/eagl/token-dispatch.ts)
/// classifies the token BEFORE the OUT and routes anything that is not
/// class 1 (SetRenderState), 2 (SetTextureStageState) or 8 (SetSamplerState)
/// to the original function — so this handler only replicates those three
/// case bodies: resolve node/stage exactly like the original, then perform
/// the same virtual call the guest would make, short-circuiting the KNOWN
/// callee shape (our own WBUF setter stub `B8 funcId …` + value-shadow /
/// ring-append trampoline). Anything off-script — vtable not pointing at the
/// expected stub, ring near-full (the trampoline's .ovf OUT path), unmapped
/// reads — returns false and the JS tier completes the call.
///
/// Config block (guest RAM, written by libs/eagl once the d3d9 WBUF ring and
/// shadow tables exist; pointer parked at OFF_HC_EAGL_TOKEN_CFG_PTR):
///   +0x00 u32 version (must be 1)
///   +0x04 u32 tokenTableBase   (EAGL token descriptor table, stride 0x1c)
///   +0x08 u32 ringCtrlAddr     (WBUF head u32; +4 overflow)
///   +0x0C u32 ringDataBase
///   +0x10 u32 ringCapacity
///   +0x14 u32 ownerGlobalAddr  (setter-shadow owner gate; 0 = no gate)
///   +0x18 u32 srsFuncId        (SetRenderState stub functionId)
///   +0x1C u32 srsShadowBase    (0 = plain ring, no shadow)
///   +0x20 u32 srsSkipCtrAddr
///   +0x24 u32 sampFuncId       (SetSamplerState)
///   +0x28 u32 sampShadowBase
///   +0x2C u32 sampSkipCtrAddr
///   +0x30 u32 tssFuncId        (SetTextureStageState — plain ring)
///   +0x38 u32 generation       (bumped by JS on every re-arm — cache key)
///
/// The block is IMMUTABLE while armed (JS writes all fields, then generation,
/// then version=1, then publishes the page pointer), so the ~10 config reads
/// are cached in statics keyed on (ptr, generation) — measured at ~1M calls/s
/// the uncached reads were half the handler's 200 ns self-time.
struct EaglTokenCfg {
    ptr: i32,
    generation: i32,
    token_table: i32,
    ring_ctrl: i32,
    ring_base: i32,
    capacity: i32,
    owner_global: i32,
    srs_fid: i32,
    srs_shadow: i32,
    srs_skip: i32,
    samp_fid: i32,
    samp_shadow: i32,
    samp_skip: i32,
    tss_fid: i32,
}
static mut EAGL_TOKEN_CFG: EaglTokenCfg = EaglTokenCfg {
    ptr: 0, generation: 0, token_table: 0, ring_ctrl: 0, ring_base: 0, capacity: 0,
    owner_global: 0, srs_fid: 0, srs_shadow: 0, srs_skip: 0,
    samp_fid: 0, samp_shadow: 0, samp_skip: 0, tss_fid: 0,
};

unsafe fn eagl_token_cfg_refresh(cfg: i32) -> Result<(), ()> {
    let r = |off: i32| safe_read32s(cfg + off).map_err(|_| ());
    let c = &mut *addr_of_mut!(EAGL_TOKEN_CFG);
    c.token_table = r(0x04)?;
    c.ring_ctrl = r(0x08)?;
    c.ring_base = r(0x0c)?;
    c.capacity = r(0x10)?;
    c.owner_global = r(0x14)?;
    c.srs_fid = r(0x18)?;
    c.srs_shadow = r(0x1c)?;
    c.srs_skip = r(0x20)?;
    c.samp_fid = r(0x24)?;
    c.samp_shadow = r(0x28)?;
    c.samp_skip = r(0x2c)?;
    c.tss_fid = r(0x30)?;
    c.generation = r(0x38)?;
    c.ptr = cfg;
    Ok(())
}

unsafe fn handle_eagl_token_dispatch() -> bool {
    let cfg = *(hp_ptr().add(OFF_HC_EAGL_TOKEN_CFG_PTR) as *const u32) as i32;
    if cfg == 0 {
        return false;
    }
    let ver = match safe_read32s(cfg) { Ok(v) => v, Err(_) => return false };
    if ver != 1 {
        return false;
    }
    // (ptr, generation) cache key — generation is one read instead of ten.
    let generation = match safe_read32s(cfg + 0x38) { Ok(v) => v, Err(_) => return false };
    {
        let c = &*addr_of!(EAGL_TOKEN_CFG);
        if c.ptr != cfg || c.generation != generation {
            if eagl_token_cfg_refresh(cfg).is_err() {
                return false;
            }
        }
    }

    let esp = read_reg32(ESP);
    let this_ctx = read_reg32(ECX);
    let node = match safe_read32s(esp + 4) { Ok(v) => v, Err(_) => return false };
    let mut stage = match safe_read32s(esp + 8) { Ok(v) => v, Err(_) => return false };
    if node == 0 {
        return false;
    }
    // Original entry semantics: param_3 == -1 → param_2[1] (the RAW node,
    // before alias resolution).
    if stage == -1 {
        stage = match safe_read32s(node + 4) { Ok(v) => v, Err(_) => return false };
    }
    // *node == -1 → the aliased/compiled node at node[0x19].
    let mut n = node;
    let mut tok = match safe_read32s(n) { Ok(v) => v, Err(_) => return false };
    if tok == -1 {
        n = match safe_read32s(node + 0x64) { Ok(v) => v, Err(_) => return false };
        tok = match safe_read32s(n) { Ok(v) => v, Err(_) => return false };
    }

    let c = &*addr_of!(EAGL_TOKEN_CFG);
    let desc = match safe_read32s(c.token_table.wrapping_add(tok.wrapping_mul(0x1c))) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let class = desc >> 24;
    let d3d_enum = (desc & 0xff_ffff) as i32;
    // Value = node[0x1a] for all three classes.
    let value = match safe_read32s(n + 0x68) { Ok(v) => v, Err(_) => return false };
    // dev = *(this + 8); vtable = *dev.
    let dev = match safe_read32s(this_ctx + 8) { Ok(v) => v, Err(_) => return false };
    let vt = match safe_read32s(dev) { Ok(v) => v, Err(_) => return false };

    // (vtable offset, expected funcId, shadow table/skip addr, shadow slot key, argc)
    let (vt_off, expect_fid, shadow_base, skip_addr, slot, argc): (i32, i32, i32, i32, i32, i32) =
        match class {
            1 => (0xe4, c.srs_fid, c.srs_shadow, c.srs_skip,
                  if (d3d_enum as u32) < 256 { d3d_enum } else { -1 }, 3),
            2 => (0x10c, c.tss_fid, 0, 0, -1, 4),
            8 => (0x114, c.samp_fid, c.samp_shadow, c.samp_skip,
                  if (stage as u32) < 16 && (d3d_enum as u32) < 16 { (stage << 4) | d3d_enum } else { -1 },
                  4),
            _ => return false,
        };

    // Perform the virtual call — but only for the KNOWN callee shape: our WBUF
    // setter stub starts `B8 <funcId:u32>`. Anything else (proxied device,
    // unpatched setter) → the JS tier / original.
    let target = match safe_read32s(vt + vt_off) { Ok(v) => v, Err(_) => return false };
    if match hc_safe_read8(target) { Ok(v) => v, Err(_) => return false } != 0xB8 {
        return false;
    }
    let fid = match safe_read32s(target + 1) { Ok(v) => v, Err(_) => return false };
    if fid != expect_fid || fid == 0 {
        return false;
    }

    // Ring capacity gate FIRST (before any shadow mutation): the trampoline's
    // .ovf path OUT-traps to the real setter thunk (drain-first) — replicated
    // by returning false so the JS tier (which runs after the standard
    // pre-dispatch drain) completes the call.
    let ring_ctrl = c.ring_ctrl;
    let ring_base = c.ring_base;
    let head = match safe_read32s(ring_ctrl) { Ok(v) => v, Err(_) => return false };
    if head < 0 || head >= c.capacity - 36 {
        return false;
    }

    // Value shadow (same fold + owner gate as writeShadowTrampoline). Decide
    // the skip HERE, but defer the slot update until the ring-entry bytes are
    // written: a false-return between shadow-update and head-bump would lose
    // the set (JS retry would see value==shadow and skip a state change the
    // device never received). Entry bytes below the un-bumped head are
    // invisible, so this order makes every abort point safe.
    let mut shadow_slot_addr = 0i32;
    if shadow_base != 0 && slot >= 0 && c.owner_global != 0 {
        let owner = match safe_read32s(c.owner_global) { Ok(v) => v, Err(_) => return false };
        if owner == dev {
            let slot_addr = shadow_base + slot * 4;
            let cur = match safe_read32s(slot_addr) { Ok(v) => v, Err(_) => return false };
            if cur == value {
                // Redundant set: bump the skip counter, EAX = D3D_OK.
                if skip_addr != 0 {
                    let cnt = match safe_read32s(skip_addr) { Ok(v) => v, Err(_) => return false };
                    if safe_write32(skip_addr, cnt.wrapping_add(1)).is_err() { return false; }
                }
                write_reg32(EAX, 0);
                return true;
            }
            shadow_slot_addr = slot_addr;
        }
    }

    // Ring append: [funcId][dev][(stage)][enum][value], head += stride.
    let entry = ring_base + head;
    if safe_write32(entry, fid).is_err() { return false; }
    if safe_write32(entry + 4, dev).is_err() { return false; }
    let ok = if argc == 3 {
        safe_write32(entry + 8, d3d_enum).is_ok() && safe_write32(entry + 12, value).is_ok()
    } else {
        safe_write32(entry + 8, stage).is_ok()
            && safe_write32(entry + 12, d3d_enum).is_ok()
            && safe_write32(entry + 16, value).is_ok()
    };
    if !ok {
        return false;
    }
    if shadow_slot_addr != 0 {
        if safe_write32(shadow_slot_addr, value).is_err() { return false; }
    }
    if safe_write32(ring_ctrl, head + (argc + 1) * 4).is_err() {
        return false;
    }
    write_reg32(EAX, 0);
    true
}
