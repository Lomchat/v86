//! Inner-loop HLE handlers for EAGL (EA Graphics Layer) — the 128..=132 band
//! of the hypercall dispatch table (see hypercall.rs try_dispatch):
//!
//!   128       shader-constant converter (guest FUN_005cbd17)
//!   129..=131 shader-parameter APPLY converter family (FUN_005c85c1/8303/ad01)
//!   132       state-token dispatcher (FUN_005c97cb): classes 1/2/8 single-pass
//!             + class-6 shader batches (bind/constants/sub-pass recursion)
//!             via a scan-then-commit walk
//!
//! JS twins: src/worker/core/hle-lib/libs/eagl/ (kernels are the validated
//! fallbacks; this module is the production tier). Guest-side plumbing
//! (detection, patching, filter trampoline, config assembly) lives there too.

use std::ptr::{addr_of, addr_of_mut};

use crate::cpu::cpu::{
    read_reg32, safe_read16, safe_read32s, safe_write32, write_reg32, EAX, ECX, ESP,
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
        // 132 = state-token dispatcher (FUN_005c97cb): classes 1/2/8 plus
        // class-6 shader batches — the guest filter routes everything else
        // (and class-6 record mode) to the original.
        132 => handle_eagl_token_dispatch(),
        // 133 = commit cluster (FUN_005d02d7 → FUN_005cf304 dirty-list walk),
        // 134 = pass-commit driver (FUN_005d01ec) — phase 3 batch boundaries;
        // both reuse the token engine + apply kernels as inner cores.
        133 => handle_eagl_commit_cluster(),
        134 => handle_eagl_pass_driver(),
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

    match eagl_apply_walk(family, layout, desc_cur, src_cur, dst_cur, budget, 0, true) {
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
///
/// `commit == false` is the phase-3 scan-pass DRY RUN: every read, bound
/// check and cursor-cell advance happens (the four cursor cells are cfg
/// scratch RAM on that path — our own structures), but the guest-visible DST
/// writes are suppressed, so a later decline leaves guest state pristine.
unsafe fn eagl_apply_walk(
    family: ApplyFamily,
    layout: ApplyLayout,
    desc_cur: i32,
    src_cur: i32,
    dst_cur: i32,
    budget: i32,
    depth: u32,
    commit: bool,
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
                    if commit {
                        safe_write32(dst, out).map_err(|_| ())?;
                    } else {
                        let _ = out;
                    }
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
                        desc_cur, src_cur, dst_cur, budget, depth + 1, commit)?,
                    ApplyLayout::Register => eagl_apply_walk(
                        family, layout, desc_cur, src_cur, dst_cur, budget, depth + 1, commit)?,
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

/// handler_id 132 — EAGL→D3D9 state-token dispatcher (guest FUN_005c97cb,
/// __thiscall RET 8: ECX = EAGL device ctx, [esp+4] = token node, [esp+8] =
/// stage-or-index).
///
/// A guest-side filter trampoline (hle-lib libs/eagl/token-dispatch.ts)
/// classifies the token BEFORE the OUT and routes only class 1
/// (SetRenderState), 2 (SetTextureStageState), 8 (SetSamplerState) and
/// class 6 outside record mode (shader programs, mode != 2) here; everything
/// else runs the original at native speed. Two coverage tiers:
///
///  - classes 1/2/8: single-pass hot path (~1M calls/s) — resolve node/stage
///    exactly like the original, then perform the same virtual call the guest
///    would make, short-circuiting the KNOWN callee shape (our own WBUF
///    setter stub `B8 funcId …` + value-shadow / ring-append trampoline).
///  - class 6 — the BATCH boundary: one crossing handles SetFVF / direct
///    shader-constant uploads / the vs/ps bind path with its default-constant
///    walk AND the type-3 sub-pass recursion (0xac stride, ~4.4 sub-tokens
///    avg), dispatching sub-tokens of classes {1,2,8,6} natively. Runs as
///    SCAN-then-COMMIT: the scan pass performs every read, bound check and
///    stub-shape check with zero side effects (any doubt → decline with the
///    ring untouched); the commit pass re-walks and writes. No guest code
///    runs between the passes, so the commit cannot fault where the scan did
///    not, and every write lands in our own RW structures (ring, shadows).
///
/// Declines (false → JS tier, which completes via the sync original): vtable
/// not pointing at the expected stub, ring near-full, unmapped reads, mode 2
/// (state-block record — must go through the real BeginStateBlock path),
/// integer/bool shader constants (no WBUF registration), classes
/// 3/4/5/7/9/10 anywhere in a batch, nested type-3 recursion, bound
/// violations.
///
/// Config block (guest RAM, written by libs/eagl once the d3d9 WBUF ring and
/// shadow tables exist; pointer parked at OFF_HC_EAGL_TOKEN_CFG_PTR):
///   +0x00 u32 version (must be 2)
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
///   +0x34 u32 enabledFlag      (guest-filter gate byte — not read here)
///   +0x38 u32 generation       (bumped by JS on every re-arm — cache key)
///   +0x3C u32 fvfFuncId        (SetFVF — plain ring, 2 args)
///   +0x40 u32 svsFuncId        (SetVertexShader — plain ring, 2 args)
///   +0x44 u32 spsFuncId        (SetPixelShader — plain ring, 2 args)
///   +0x48 u32 vscfFuncId       (SetVertexShaderConstantF — payload entry)
///   +0x4C u32 pscfFuncId       (SetPixelShaderConstantF — payload entry)
///   +0x50 u32 texFuncId        (SetTexture — class-5 sub-tokens, 3 args)
///
/// The block is IMMUTABLE while armed (JS writes all fields, then generation,
/// then version, then publishes the page pointer), so the config reads are
/// cached in statics keyed on (ptr, generation) — measured at ~1M calls/s
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
    fvf_fid: i32,
    svs_fid: i32,
    sps_fid: i32,
    vscf_fid: i32,
    pscf_fid: i32,
    tex_fid: i32,
}
static mut EAGL_TOKEN_CFG: EaglTokenCfg = EaglTokenCfg {
    ptr: 0, generation: 0, token_table: 0, ring_ctrl: 0, ring_base: 0, capacity: 0,
    owner_global: 0, srs_fid: 0, srs_shadow: 0, srs_skip: 0,
    samp_fid: 0, samp_shadow: 0, samp_skip: 0, tss_fid: 0,
    fvf_fid: 0, svs_fid: 0, sps_fid: 0, vscf_fid: 0, pscf_fid: 0,
    tex_fid: 0,
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
    c.fvf_fid = r(0x3c)?;
    c.svs_fid = r(0x40)?;
    c.sps_fid = r(0x44)?;
    c.vscf_fid = r(0x48)?;
    c.pscf_fid = r(0x4c)?;
    c.tex_fid = r(0x50)?;
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
    // v2 fields are a prefix of v3 (phase 3 adds validate/scratch cells only).
    if ver < 2 || ver > 3 {
        return false;
    }
    // (ptr, generation) cache key — generation is one read instead of ~16.
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
    // dev = *(this + 8); vtable = *dev.
    let dev = match safe_read32s(this_ctx + 8) { Ok(v) => v, Err(_) => return false };
    let vt = match safe_read32s(dev) { Ok(v) => v, Err(_) => return false };

    match class {
        1 | 2 | 8 => eagl_dispatch_simple(c, dev, vt, class, desc, stage, n),
        6 => eagl_dispatch_class6_batch(c, this_ctx, dev, vt, node, stage),
        _ => false,
    }
}

/// Single-pass hot path for classes 1/2/8 (~1M calls/s — every read counts).
unsafe fn eagl_dispatch_simple(
    c: &EaglTokenCfg, dev: i32, vt: i32, class: u32, desc: u32, stage: i32, n: i32,
) -> bool {
    let d3d_enum = (desc & 0xff_ffff) as i32;
    // Value = node[0x1a] for all three classes.
    let value = match safe_read32s(n + 0x68) { Ok(v) => v, Err(_) => return false };

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
    let fid = match eagl_stub_fid(vt, vt_off, expect_fid) { Ok(v) => v, Err(_) => return false };

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

// --- class-6 batch engine (scan-then-commit) -------------------------------

/// D3D9 device vtable offsets used by the batch paths (standard
/// IDirect3DDevice9 layout, RE-verified against the guest jump table).
const VT_SET_TEXTURE: i32 = 0x104;
const VT_SET_FVF: i32 = 0x164;
const VT_SET_VERTEX_SHADER: i32 = 0x170;
const VT_SET_VS_CONST_F: i32 = 0x178;
const VT_SET_PIXEL_SHADER: i32 = 0x1ac;
const VT_SET_PS_CONST_F: i32 = 0x1b4;

/// EAGL's abort HRESULT (-0x7fffbffb = E_FAIL) — returned by the guest when a
/// shader record is unbound or a pixel default-constant entry has an unknown
/// type. Guest-visible; must be replicated bit-exact.
const EAGL_E_FAIL: i32 = 0x80004005u32 as i32;

/// Ring cursor for the two-pass class-6 walk. The scan pass bumps `head`
/// without writing (every potential append counted — shadow skips are decided
/// only at commit, so the scan total is a safe upper bound for the room
/// check); the commit pass writes entry bytes at `head` and then bumps.
/// The guest-visible ring head (cfg ringCtrl) is written ONCE, after the
/// commit walk finishes — entry bytes above the un-bumped head are invisible,
/// so every abort point leaves a consistent ring.
struct EaglPass {
    commit: bool,
    head: i32,
    /// Scan-side model of guest-visible cell writes made earlier in the SAME
    /// crossing (shadow slots, dirty flags, entry statuses, header bits):
    /// the scan performs NO guest writes, so any read of a cell the crossing
    /// already "wrote" must come from this model or the scan diverges from
    /// the commit replay (e.g. the walk sets `node[10] |= 1` and the same
    /// node's cdca7 tail tests that bit). 64 slots covers observed crossings;
    /// overflow sets model_ovf (callers decline or degrade to upper bounds).
    model_addr: [i32; 64],
    model_val: [i32; 64],
    model_n: usize,
    model_ovf: bool,
}

impl EaglPass {
    fn new(commit: bool, head: i32) -> EaglPass {
        EaglPass {
            commit,
            head,
            model_addr: [0; 64],
            model_val: [0; 64],
            model_n: 0,
            model_ovf: false,
        }
    }
    fn model_get(&self, addr: i32) -> Option<i32> {
        for i in 0..self.model_n {
            if self.model_addr[i] == addr {
                return Some(self.model_val[i]);
            }
        }
        None
    }
    fn model_put(&mut self, addr: i32, val: i32) {
        for i in 0..self.model_n {
            if self.model_addr[i] == addr {
                self.model_val[i] = val;
                return;
            }
        }
        if self.model_n < 64 {
            self.model_addr[self.model_n] = addr;
            self.model_val[self.model_n] = val;
            self.model_n += 1;
        } else {
            self.model_ovf = true;
        }
    }
    /// Guest-visible cell write: real on commit, modeled on scan.
    unsafe fn write_cell(&mut self, addr: i32, val: i32) -> Result<(), ()> {
        if self.commit {
            safe_write32(addr, val).map_err(|_| ())
        } else {
            self.model_put(addr, val);
            if self.model_ovf { Err(()) } else { Ok(()) }
        }
    }
    /// Guest-visible cell read that must observe same-crossing writes.
    unsafe fn read_cell(&self, addr: i32) -> Result<i32, ()> {
        if !self.commit {
            if let Some(v) = self.model_get(addr) {
                return Ok(v);
            }
        }
        safe_read32s(addr).map_err(|_| ())
    }
}

/// Resolve a device vtable slot and require the KNOWN callee shape: our WBUF
/// setter stub `B8 <funcId:u32>` with the expected id. Err = decline.
unsafe fn eagl_stub_fid(vt: i32, vt_off: i32, expect_fid: i32) -> Result<i32, ()> {
    let target = safe_read32s(vt + vt_off).map_err(|_| ())?;
    if hc_safe_read8(target).map_err(|_| ())? != 0xB8 {
        return Err(());
    }
    let fid = safe_read32s(target + 1).map_err(|_| ())?;
    if fid != expect_fid || fid == 0 {
        return Err(());
    }
    Ok(fid)
}

/// Class 1/2/8 state set inside a class-6 batch (sub-pass recursion), in
/// two-pass form: scan validates and reserves ring room; commit replicates
/// the shadow compare/skip/update + ring append of the single-pass path.
unsafe fn eagl_emit_state(
    c: &EaglTokenCfg, dev: i32, vt: i32, class: u32,
    d3d_enum: i32, stage: i32, value: i32, p: &mut EaglPass,
) -> Result<(), ()> {
    let (vt_off, expect_fid, shadow_base, skip_addr, slot, argc): (i32, i32, i32, i32, i32, i32) =
        match class {
            1 => (0xe4, c.srs_fid, c.srs_shadow, c.srs_skip,
                  if (d3d_enum as u32) < 256 { d3d_enum } else { -1 }, 3),
            2 => (0x10c, c.tss_fid, 0, 0, -1, 4),
            8 => (0x114, c.samp_fid, c.samp_shadow, c.samp_skip,
                  if (stage as u32) < 16 && (d3d_enum as u32) < 16 { (stage << 4) | d3d_enum } else { -1 },
                  4),
            _ => return Err(()),
        };
    let fid = eagl_stub_fid(vt, vt_off, expect_fid)?;

    let mut shadow_slot_addr = 0i32;
    if shadow_base != 0 && slot >= 0 && c.owner_global != 0 {
        let owner = safe_read32s(c.owner_global).map_err(|_| ())?;
        if owner == dev {
            let slot_addr = shadow_base + slot * 4;
            // Scan consults its local model first so earlier same-crossing
            // writes are seen (exact skip decisions → exact head prediction);
            // commit reads the real slot (its own writes ARE the model).
            let cur = match if p.commit { None } else { p.model_get(slot_addr) } {
                Some(v) => v,
                None => safe_read32s(slot_addr).map_err(|_| ())?,
            };
            // Post-overflow the scan's view may miss an untracked earlier
            // write — stop skipping (over-reserve is safe, under-reserve is
            // not); commit always decides on real state.
            if cur == value && (p.commit || !p.model_ovf) {
                if p.commit {
                    if skip_addr != 0 {
                        let cnt = safe_read32s(skip_addr).map_err(|_| ())?;
                        safe_write32(skip_addr, cnt.wrapping_add(1)).map_err(|_| ())?;
                    }
                }
                return Ok(());
            }
            shadow_slot_addr = slot_addr;
        }
    }
    if !p.commit && shadow_slot_addr != 0 {
        p.model_put(shadow_slot_addr, value);
    }

    if p.commit {
        let entry = c.ring_base + p.head;
        safe_write32(entry, fid).map_err(|_| ())?;
        safe_write32(entry + 4, dev).map_err(|_| ())?;
        if argc == 3 {
            safe_write32(entry + 8, d3d_enum).map_err(|_| ())?;
            safe_write32(entry + 12, value).map_err(|_| ())?;
        } else {
            safe_write32(entry + 8, stage).map_err(|_| ())?;
            safe_write32(entry + 12, d3d_enum).map_err(|_| ())?;
            safe_write32(entry + 16, value).map_err(|_| ())?;
        }
        if shadow_slot_addr != 0 {
            safe_write32(shadow_slot_addr, value).map_err(|_| ())?;
        }
    }
    p.head += (argc + 1) * 4;
    Ok(())
}

/// Plain 2-arg ring entry (SetFVF / SetVertexShader / SetPixelShader):
/// [funcId][dev][value]. SetVertexShader/SetPixelShader carry the raw guest
/// COM pointer — the drain handler resolves it, same as the trampoline path.
unsafe fn eagl_emit_2arg(
    c: &EaglTokenCfg, vt: i32, vt_off: i32, expect_fid: i32, dev: i32, value: i32,
    p: &mut EaglPass,
) -> Result<(), ()> {
    let fid = eagl_stub_fid(vt, vt_off, expect_fid)?;
    if p.commit {
        let entry = c.ring_base + p.head;
        safe_write32(entry, fid).map_err(|_| ())?;
        safe_write32(entry + 4, dev).map_err(|_| ())?;
        safe_write32(entry + 8, value).map_err(|_| ())?;
    }
    p.head += 12;
    Ok(())
}

/// Shader-constant-F ring entry with inline payload capture:
/// [funcId][dev][startReg][vec4Count][vec4Count×4 dwords] — same layout the
/// shader-constant trampoline emits and getWbufEntryStride expects (count
/// 1..=256; a zero-count call is a device no-op the guest also makes, so it
/// is skipped rather than appended).
unsafe fn eagl_emit_const_f(
    c: &EaglTokenCfg, vt: i32, pix: bool, dev: i32, start_reg: i32, src: i32, cnt: i32,
    p: &mut EaglPass,
) -> Result<(), ()> {
    if cnt < 0 || cnt > 256 {
        return Err(());
    }
    if cnt == 0 {
        return Ok(());
    }
    let (vt_off, expect_fid) = if pix {
        (VT_SET_PS_CONST_F, c.pscf_fid)
    } else {
        (VT_SET_VS_CONST_F, c.vscf_fid)
    };
    let fid = eagl_stub_fid(vt, vt_off, expect_fid)?;
    let bytes = cnt * 16;
    if !p.commit {
        // Payload readability: ≤4KB spans at most two pages — first and last
        // dword touch both. The commit pass then reads every dword safely.
        safe_read32s(src).map_err(|_| ())?;
        safe_read32s(src.wrapping_add(bytes - 4)).map_err(|_| ())?;
    } else {
        let entry = c.ring_base + p.head;
        safe_write32(entry, fid).map_err(|_| ())?;
        safe_write32(entry + 4, dev).map_err(|_| ())?;
        safe_write32(entry + 8, start_reg).map_err(|_| ())?;
        safe_write32(entry + 12, cnt).map_err(|_| ())?;
        for i in 0..cnt * 4 {
            let v = safe_read32s(src.wrapping_add(i * 4)).map_err(|_| ())?;
            safe_write32(entry + 16 + i * 4, v).map_err(|_| ())?;
        }
    }
    p.head += 16 + bytes;
    Ok(())
}

/// Full FUN_005c97cb token dispatch inside a batch (top-level class-6 call or
/// a type-3 sub-pass recursion element). Replicates the entry semantics
/// (stage -1 → raw node[1], *node == -1 → alias at node[0x19]), then the
/// class switch. Ok(hr) = guest-visible result; Err = decline the WHOLE
/// top-level call (scan pass only — the ring is untouched).
unsafe fn eagl_dispatch_token(
    c: &EaglTokenCfg, ctx: i32, mode: i32, dev: i32, vt: i32,
    node: i32, stage_in: i32, depth: u32, p: &mut EaglPass,
) -> Result<i32, ()> {
    if node == 0 {
        return Err(());
    }
    let mut stage = stage_in;
    if stage == -1 {
        stage = safe_read32s(node + 4).map_err(|_| ())?;
    }
    let mut n = node;
    let mut tok = safe_read32s(n).map_err(|_| ())?;
    if tok == -1 {
        n = safe_read32s(node + 0x64).map_err(|_| ())?;
        tok = safe_read32s(n).map_err(|_| ())?;
    }
    let desc = safe_read32s(c.token_table.wrapping_add(tok.wrapping_mul(0x1c)))
        .map_err(|_| ())? as u32;
    match desc >> 24 {
        cls @ (1 | 2 | 8) => {
            let value = safe_read32s(n + 0x68).map_err(|_| ())?;
            eagl_emit_state(c, dev, vt, cls, (desc & 0xff_ffff) as i32, stage, value, p)?;
            Ok(0)
        },
        5 => eagl_class5_texture(c, ctx, mode, dev, vt, n, stage, p),
        6 => eagl_class6(c, ctx, mode, dev, vt, n, desc, stage, depth, p),
        // Lights/material/misc: real staging-struct writes we do not
        // replicate — decline the whole batch.
        3 | 4 | 7 | 9 | 10 => Err(()),
        // Original's switch default: no side effects, returns 0.
        _ => Ok(0),
    }
}

/// The class-6 case body (guest 0x5c9b15..): SetFVF, direct shader-constant
/// sets, vs/ps bind + default-constant walk + type-3 recursion. Token
/// sub-ids and their exact guest routing are transcribed from the decompile
/// (the 0x60001xx-0x60004xx chain collapses to the two constant-F sets).
unsafe fn eagl_class6(
    c: &EaglTokenCfg, ctx: i32, mode: i32, dev: i32, vt: i32,
    n: i32, desc: u32, stage: i32, depth: u32, p: &mut EaglPass,
) -> Result<i32, ()> {
    // Mode 2 = state-block record: every device call must go through the real
    // BeginStateBlock path, not the ring. The guest filter already routes
    // mode-2 calls to the original; this is the backstop.
    if mode == 2 {
        return Err(());
    }
    if desc == 0x6000008 {
        // SetFVF — the one class-6 token that also runs in mode 1 (degraded
        // mode forces FVF = 2 = D3DFVF_XYZ).
        let mut value = safe_read32s(n + 0x68).map_err(|_| ())?;
        if mode == 1 {
            value = 2;
        }
        eagl_emit_2arg(c, vt, VT_SET_FVF, c.fvf_fid, dev, value, p)?;
        return Ok(0);
    }
    if mode == 1 {
        // Degraded/FFP mode defers all other shader tokens — no side effects.
        return Ok(0);
    }
    match desc {
        // Direct constant-F uploads: (dev, stage, node[0x13] = data ptr,
        // node[0x2a] = vec4 count).
        0x6000002 | 0x6000102 | 0x6000202 | 0x6000302 | 0x6000402 => {
            let src = safe_read32s(n + 0x4c).map_err(|_| ())?;
            let cnt = safe_read32s(n + 0xa8).map_err(|_| ())?;
            eagl_emit_const_f(c, vt, false, dev, stage, src, cnt, p)?;
            Ok(0)
        },
        0x6000005 | 0x6000105 | 0x6000205 | 0x6000305 | 0x6000405 => {
            let src = safe_read32s(n + 0x4c).map_err(|_| ())?;
            let cnt = safe_read32s(n + 0xa8).map_err(|_| ())?;
            eagl_emit_const_f(c, vt, true, dev, stage, src, cnt, p)?;
            Ok(0)
        },
        // Integer/bool constant sets: no WBUF registration for the I/B
        // setters — the original handles these (rare) tokens.
        0x6000003 | 0x6000004 | 0x6000006 | 0x6000007 => Err(()),
        0x6000000 => eagl_class6_bind(c, ctx, mode, dev, vt, n, false, depth, p),
        0x6000001 => eagl_class6_bind(c, ctx, mode, dev, vt, n, true, depth, p),
        // Original's inner default: no side effects, returns 0.
        _ => Ok(0),
    }
}

/// The vs (0x6000000) / ps (0x6000001) bind path: resolve the shader resource
/// record through the ctx handle tables, bind the program, upload its
/// default-constant table, and (pixel only) recurse into type-3 sub-pass
/// arrays. Transcription notes (decompile, `re decompile 0x5c97cb`):
///   record   = *(ctx+0x24) + idx*0x1c, idx via *(ctx+0x8c)[node[3]] tables
///   rec+0x04 = shader COM pointer (SetVertexShader/SetPixelShader arg)
///   rec+0x0c = bound flag — 0 → return E_FAIL (before any device call)
///   rec+0x14 = default-constant table: count @+0xc, entries @ +(*(+0x10))+6,
///              stride 20 bytes: u16 type @-2, u16 reg @0, u16 vec4Count @+2
///   rec+0x18 → header ptr; *(hdr+0x44) = payload block: F data at +8+reg*16
///              (type 2); type 1 = int4, type 0 = bool — no WBUF path, decline;
///              *(hdrPtr+0x30)[k] = type-3 sub-pass descriptor
///   vertex walk: unknown type silently skipped; pixel walk: unknown type
///   aborts with E_FAIL (guest LAB_005c9ab3) — both replicated exactly.
/// mode==2 (CreateVertexDeclaration path) never reaches here — declined at
/// eagl_class6 entry.
/// The shared ctx handle-table resolve (classes 5 and 6 bind use the exact
/// same chain): *(ctx+0x8c)[node[3]] → per-resource record → index (direct
/// or via the double-indirect table) → the 0x1c-stride record at ctx+0x24.
unsafe fn eagl_resolve_record(ctx: i32, n: i32) -> Result<i32, ()> {
    let page_tbl = safe_read32s(ctx + 0x8c).map_err(|_| ())?;
    let nid = safe_read32s(n + 0xc).map_err(|_| ())?;
    let r = safe_read32s(page_tbl.wrapping_add(nid.wrapping_mul(4))).map_err(|_| ())?;
    let t = safe_read32s(r + 0x38).map_err(|_| ())?;
    let a = safe_read32s(r + 0x28).map_err(|_| ())?
        .wrapping_add(safe_read32s(n + 0x14).map_err(|_| ())?);
    let idx = if t == 0 {
        let off = safe_read32s(ctx + 0x2c).map_err(|_| ())?;
        safe_read32s(a.wrapping_add(off)).map_err(|_| ())?
    } else {
        let inner_off = safe_read32s(safe_read32s(ctx + 0xc).map_err(|_| ())? + 8).map_err(|_| ())?;
        let inner = safe_read32s(a.wrapping_add(inner_off)).map_err(|_| ())?;
        safe_read32s(safe_read32s(t + 8).map_err(|_| ())?.wrapping_add(inner.wrapping_mul(4)))
            .map_err(|_| ())?
    };
    Ok(idx.wrapping_mul(0x1c).wrapping_add(safe_read32s(ctx + 0x24).map_err(|_| ())?))
}

/// Class 5 — SetTexture (guest case 5, non-record mode): resolve the texture
/// record, ring-append SetTexture(dev, stage, *(rec+4)). The mode-2 branch
/// (GetDeviceCaps format validation) never reaches here — the whole batch is
/// gated on mode != 2.
unsafe fn eagl_class5_texture(
    c: &EaglTokenCfg, ctx: i32, mode: i32, dev: i32, vt: i32,
    n: i32, stage: i32, p: &mut EaglPass,
) -> Result<i32, ()> {
    if mode == 2 || c.tex_fid == 0 {
        return Err(());
    }
    let rec = eagl_resolve_record(ctx, n)?;
    let value = safe_read32s(rec + 4).map_err(|_| ())?;
    let fid = eagl_stub_fid(vt, VT_SET_TEXTURE, c.tex_fid)?;
    if p.commit {
        let entry = c.ring_base + p.head;
        safe_write32(entry, fid).map_err(|_| ())?;
        safe_write32(entry + 4, dev).map_err(|_| ())?;
        safe_write32(entry + 8, stage).map_err(|_| ())?;
        safe_write32(entry + 12, value).map_err(|_| ())?;
    }
    p.head += 16;
    Ok(0)
}

unsafe fn eagl_class6_bind(
    c: &EaglTokenCfg, ctx: i32, mode: i32, dev: i32, vt: i32,
    n: i32, pix: bool, depth: u32, p: &mut EaglPass,
) -> Result<i32, ()> {
    let rec = eagl_resolve_record(ctx, n)?;
    if safe_read32s(rec + 0xc).map_err(|_| ())? == 0 {
        // Unbound shader record: guest returns E_FAIL before any device call.
        return Ok(EAGL_E_FAIL);
    }
    let handle = safe_read32s(rec + 4).map_err(|_| ())?;
    if pix {
        eagl_emit_2arg(c, vt, VT_SET_PIXEL_SHADER, c.sps_fid, dev, handle, p)?;
    } else {
        eagl_emit_2arg(c, vt, VT_SET_VERTEX_SHADER, c.svs_fid, dev, handle, p)?;
    }

    let ct = safe_read32s(rec + 0x14).map_err(|_| ())?;
    if ct == 0 {
        return Ok(0);
    }
    let hdr_ptr = safe_read32s(rec + 0x18).map_err(|_| ())?;
    let hdr = safe_read32s(hdr_ptr + 0x44).map_err(|_| ())?;
    let f_base = hdr.wrapping_add(8);
    let total = safe_read32s(ct + 0xc).map_err(|_| ())?;
    if total as u32 > 1024 {
        return Err(());
    }
    let mut ep = ct
        .wrapping_add(safe_read32s(ct + 0x10).map_err(|_| ())?)
        .wrapping_add(6);
    for k in 0..total {
        let typ = safe_read16(ep - 2).map_err(|_| ())?;
        let reg = safe_read16(ep).map_err(|_| ())?;
        let cn = safe_read16(ep + 2).map_err(|_| ())?;
        match typ {
            2 => {
                let src = f_base.wrapping_add(reg.wrapping_mul(16));
                eagl_emit_const_f(c, vt, pix, dev, reg, src, cn, p)?;
            },
            // Types 0/1 = bool/int default constants — no WBUF registration.
            0 | 1 => return Err(()),
            3 if pix => {
                // Sub-pass recursion (the 0xac-stride array): one nesting
                // level is all the content uses — decline anything deeper.
                if depth != 0 {
                    return Err(());
                }
                let sub_arr = safe_read32s(hdr_ptr + 0x30).map_err(|_| ())?;
                let sub = safe_read32s(sub_arr.wrapping_add(k.wrapping_mul(4))).map_err(|_| ())?;
                if sub != 0 {
                    let pass_id = safe_read32s(safe_read32s(sub + 4).map_err(|_| ())? + 4)
                        .map_err(|_| ())?;
                    let page_tbl = safe_read32s(ctx + 0x8c).map_err(|_| ())?;
                    let pr = safe_read32s(page_tbl.wrapping_add(pass_id.wrapping_mul(4)))
                        .map_err(|_| ())?;
                    let m = safe_read32s(pr + 0x3c).map_err(|_| ())?;
                    if m as u32 > 256 {
                        return Err(());
                    }
                    let base = safe_read32s(pr + 0x40).map_err(|_| ())?;
                    for i in 0..m {
                        let sub_node = base.wrapping_add(i.wrapping_mul(0xac));
                        let hr = eagl_dispatch_token(c, ctx, mode, dev, vt, sub_node, reg, depth + 1, p)?;
                        if hr < 0 {
                            // Guest aborts the whole commit on a negative
                            // sub-result — partial device calls stand.
                            return Ok(hr);
                        }
                    }
                }
            },
            _ => {
                if pix {
                    // Pixel walk: unknown entry type aborts with E_FAIL after
                    // the calls made so far (guest LAB_005c9ab3).
                    return Ok(EAGL_E_FAIL);
                }
                // Vertex walk: unknown entry type is silently skipped.
            },
        }
        ep = ep.wrapping_add(20);
    }
    Ok(0)
}

/// Top-level class-6 entry: scan (no side effects, any doubt → decline to the
/// JS tier / original), room check, then commit. The guest-visible ring head
/// is published once, after the commit walk completes or aborts — every
/// intermediate state is invisible to the drain.
unsafe fn eagl_dispatch_class6_batch(
    c: &EaglTokenCfg, ctx: i32, dev: i32, vt: i32, node: i32, stage: i32,
) -> bool {
    let mode = match safe_read32s(ctx + 0x84) { Ok(v) => v, Err(_) => return false };
    if mode == 2 {
        return false;
    }
    let head0 = match safe_read32s(c.ring_ctrl) { Ok(v) => v, Err(_) => return false };
    if head0 < 0 || head0 > c.capacity {
        return false;
    }
    let mut scan = EaglPass::new(false, head0);
    let hr = match eagl_dispatch_token(c, ctx, mode, dev, vt, node, stage, 0, &mut scan) {
        Ok(v) => v,
        Err(_) => return false,
    };
    // Room for the whole batch plus slack; otherwise let the JS tier complete
    // via the original (its trampolines drain through the .ovf path).
    if scan.head > c.capacity - 64 {
        return false;
    }
    let mut com = EaglPass::new(true, head0);
    // The commit pass repeats exactly the scan's reads (no guest code ran in
    // between) and writes only to our own RW structures — an Err here is
    // effectively unreachable; fall back to the scan's hr with whatever
    // entries fully committed (head tracks complete entries only).
    let hr = eagl_dispatch_token(c, ctx, mode, dev, vt, node, stage, 0, &mut com).unwrap_or(hr);
    if safe_write32(c.ring_ctrl, com.head).is_err() {
        return false;
    }
    write_reg32(EAX, hr);
    true
}

// === Commit-cluster (133) & pass-driver (134) batch engines =======
//
// Two patch points:
//   133 FUN_005d02d7 (thiscall RET 4): dep-list loop over FUN_005cf304, the
//       dirty-list commit walk — replicated in full below (eagl_walk_native).
//   134 FUN_005d01ec (stdcall RET 8): pass-commit driver — stamp write +
//       element loop of {apply walk FUN_005cdca7, token dispatch(-1)}.
// Both run SCAN-then-COMMIT through EaglPass: the scan performs every read /
// bound / stub check with ZERO guest-visible writes (flag/status cells go
// through the pass model), any doubt → decline with pristine guest state; the
// commit replays the identical decision sequence with real writes. Inner
// cores reused: eagl_dispatch_token (the phase-2 token engine) and
// eagl_apply_walk (handlers 129-131 kernels, dry-run capable).
//
// DECLINE surface (→ JS tier → sync original, counted there): evaluator
// nodes (node[0x14] != 0 with dirty bit), compiled nodes (*node == -1),
// constant-limit growth (FUN_005c80ae would grow → state-block rebuild),
// int/bool constant uploads (no WBUF fids), cdca7 default-leg descs (the
// inline x87 converter incl. D3DCOLOR pack), pending __controlfp restore
// (ctx+0xb4), iteration/depth bounds, model overflow.

/// Per-crossing cached EAGL device-ctx fields (handler-body discipline: read
/// each ctx field once). `stamp30` is overridable for the pass driver, whose
/// guest original writes ctx+0x30 = passIdx BEFORE the element loop.
struct EaglCommitCtx {
    ctx: i32,
    mode: i32,
    dev: i32,
    vt: i32,
    stamp30: i32,
    stamp34: i32,
    handle_tbl: i32,
    off2c: i32,
    inner_off: i32,
    tech_base: i32,
}

unsafe fn eagl_load_commit_ctx(ctx: i32, stamp30_override: Option<i32>) -> Result<EaglCommitCtx, ()> {
    let r = |a: i32| safe_read32s(a).map_err(|_| ());
    // Pending __controlfp precision restore (left by a guest-run evaluator
    // path) — the epilogues would have to replicate CRT __controlfp; decline.
    if r(ctx + 0xb4)? != 0 {
        eagl_set_decline_reason(6);
        return Err(());
    }
    let dev = r(ctx + 8)?;
    Ok(EaglCommitCtx {
        ctx,
        mode: r(ctx + 0x84)?,
        dev,
        vt: r(dev)?,
        stamp30: match stamp30_override { Some(v) => v, None => r(ctx + 0x30)? },
        stamp34: r(ctx + 0x34)?,
        handle_tbl: r(ctx + 0x8c)?,
        off2c: r(ctx + 0x2c)?,
        inner_off: r(r(ctx + 0xc)? + 8)?,
        tech_base: r(ctx + 0x1c)?,
    })
}

/// Resolve wrapper (guest FUN_005cafde/caf72/cce58 bodies): derive the two
/// source addresses from the param record, then run the SHARED apply kernel
/// over four cursor cells in cfg scratch RAM (+0x60..0x6f) — our own memory,
/// so the scan's dry run can still advance cursors while suppressing dst
/// writes. rec = {+4 fileIdx, +8 descChainOff, +0xc valueOff}.
unsafe fn eagl_wrapper_apply(
    cc: &EaglCommitCtx, cfg: i32, family: ApplyFamily, layout: ApplyLayout,
    rec: i32, dst: i32, cnt: i32, commit: bool,
) -> Result<i32, ()> {
    if cnt as u32 > 4096 {
        return Err(());
    }
    let r = |a: i32| safe_read32s(a).map_err(|_| ());
    let hrec = r(cc.handle_tbl.wrapping_add(r(rec + 4)?.wrapping_mul(4)))?;
    let off = if r(hrec + 0x38)? == 0 { cc.off2c } else { cc.inner_off };
    let desc_chain = r(hrec + 0x24)?.wrapping_add(r(rec + 8)?).wrapping_add(off);
    let src = r(hrec + 0x28)?.wrapping_add(r(rec + 0xc)?).wrapping_add(off);
    let s = cfg + 0x60;
    safe_write32(s, desc_chain).map_err(|_| ())?;
    safe_write32(s + 4, src).map_err(|_| ())?;
    safe_write32(s + 8, dst).map_err(|_| ())?;
    safe_write32(s + 12, cnt).map_err(|_| ())?;
    eagl_apply_walk(family, layout, s, s + 4, s + 8, s + 12, 0, commit)
}

/// Sane token id → descriptor dword. Table index is bounded — a wild token
/// id means a structural problem (or a compiled node) → decline.
unsafe fn eagl_token_desc(c: &EaglTokenCfg, tok: i32) -> Result<u32, ()> {
    if tok as u32 > 0xffff {
        return Err(());
    }
    Ok(safe_read32s(c.token_table.wrapping_add(tok.wrapping_mul(0x1c))).map_err(|_| ())? as u32)
}

/// Native FUN_005c80ae in scan form: walk the default-constant table
/// computing register extents; any extent that would GROW a tech cursor cell
/// means the guest would take the state-block rebuild path → Err (decline).
/// Steady state writes nothing. Unknown entry type → guest E_FAIL.
unsafe fn eagl_scan_80ae(ct: i32, tech: i32, ps: bool, p: &EaglPass) -> Result<i32, ()> {
    if ct == 0 {
        return Ok(0);
    }
    let r = |a: i32| safe_read32s(a).map_err(|_| ());
    let total = r(ct + 0xc)?;
    if total == 0 {
        return Ok(0);
    }
    if total as u32 > 1024 {
        return Err(());
    }
    let (p_f, p_b, p_i) = if ps {
        (tech + 0x9c, tech + 0xa0, tech + 0xa4)
    } else {
        (tech + 0x90, tech + 0x94, tech + 0x98)
    };
    let p_s = tech + 0xa8;
    let mut ep = ct.wrapping_add(r(ct + 0x10)?).wrapping_add(6);
    for _ in 0..total {
        let typ = safe_read16(ep - 2).map_err(|_| ())?;
        let ext = safe_read16(ep).map_err(|_| ())? + safe_read16(ep + 2).map_err(|_| ())?;
        let cell = match typ {
            2 => p_f,
            0 => p_b,
            1 => p_i,
            3 => p_s,
            _ => return Ok(EAGL_E_FAIL),
        };
        let ext = if typ == 3 { safe_read16(ep).map_err(|_| ())? + 1 } else { ext };
        if (p.read_cell(cell)? as u32) < ext as u32 {
            eagl_set_decline_reason(3); // would grow → rebuild path
            return Err(());
        }
        ep = ep.wrapping_add(0x14);
    }
    Ok(0)
}

/// Native FUN_005cdca7 (the apply walk). Guest-visible writes (dirty-pair
/// clears, header-flag clears, node[10] clear, conversion staging) are
/// commit-gated; the scan models flag cells and dry-runs the apply kernels.
/// Declines: evaluator nodes, compiled nodes, default-leg descs (inline x87
/// converter), depth/bounds.
unsafe fn eagl_cdca7_native(
    c: &EaglTokenCfg, cc: &EaglCommitCtx, cfg: i32, node: i32, depth: u32, p: &mut EaglPass,
) -> Result<i32, ()> {
    if depth > APPLY_MAX_DEPTH {
        return Err(());
    }
    let r = |a: i32| safe_read32s(a).map_err(|_| ());
    let mut hr = 0i32;
    let tok0 = r(node)?;
    if tok0 != -1 {
        let desc0 = eagl_token_desc(c, tok0)?;
        if desc0 == 0x6000000 || desc0 == 0x6000001 {
            let prec = eagl_resolve_record(cc.ctx, node)?;
            if r(prec + 0xc)? != 0 && r(prec + 0x14)? != 0 {
                let hdrp = r(prec + 0x18)?;
                let flags = p.read_cell(hdrp + 0x28)?;
                if flags & 1 != 0 {
                    p.write_cell(hdrp + 0x28, flags & !1)?;
                    let ct = r(prec + 0x14)?;
                    let blk = r(hdrp + 0x44)?;
                    let f_base = blk.wrapping_add(8);
                    let i_base = blk.wrapping_add(r(blk)?);
                    let b_base = blk.wrapping_add(r(blk + 4)?);
                    let pairs = r(hdrp + 0x30)?;
                    let total = r(ct + 0xc)?;
                    if total as u32 > 1024 {
                        return Err(());
                    }
                    let mut ep = ct.wrapping_add(r(ct + 0x10)?).wrapping_add(8);
                    for k in 0..total {
                        let pair = r(pairs.wrapping_add(k.wrapping_mul(4)))?;
                        let typ = safe_read16(ep - 4).map_err(|_| ())?;
                        let reg = safe_read16(ep - 2).map_err(|_| ())?;
                        let cn = safe_read16(ep).map_err(|_| ())?;
                        match typ {
                            0 => {
                                if pair != 0 && p.read_cell(pair)? != 0 {
                                    hr = eagl_wrapper_apply(cc, cfg, ApplyFamily::Float, ApplyLayout::Packed,
                                        r(pair + 4)?, b_base.wrapping_add(reg.wrapping_mul(4)), cn, p.commit)?;
                                }
                            },
                            1 => {
                                if pair != 0 && p.read_cell(pair)? != 0 {
                                    hr = eagl_wrapper_apply(cc, cfg, ApplyFamily::Float, ApplyLayout::Register,
                                        r(pair + 4)?, i_base.wrapping_add(reg.wrapping_mul(0x10)), cn, p.commit)?;
                                }
                            },
                            2 => {
                                if pair != 0 && p.read_cell(pair)? != 0 {
                                    hr = eagl_wrapper_apply(cc, cfg, ApplyFamily::Int, ApplyLayout::Register,
                                        r(pair + 4)?, f_base.wrapping_add(reg.wrapping_mul(0x10)), cn, p.commit)?;
                                }
                            },
                            3 => {
                                if pair != 0 {
                                    let sub = r(cc.handle_tbl.wrapping_add(r(r(pair + 4)? + 4)?.wrapping_mul(4)))?;
                                    let m = r(sub + 0x3c)?;
                                    if m as u32 > 256 {
                                        return Err(());
                                    }
                                    let base = r(sub + 0x40)?;
                                    for i in 0..m {
                                        hr = eagl_cdca7_native(c, cc, cfg, base.wrapping_add(i.wrapping_mul(0xac)), depth + 1, p)?;
                                        if hr < 0 {
                                            return Ok(hr);
                                        }
                                    }
                                }
                            },
                            _ => return Ok(EAGL_E_FAIL),
                        }
                        if hr < 0 {
                            return Ok(hr);
                        }
                        if pair != 0 {
                            p.write_cell(pair, 0)?;
                        }
                        ep = ep.wrapping_add(0x14);
                    }
                    let f2 = p.read_cell(hdrp + 0x28)?;
                    p.write_cell(hdrp + 0x28, f2 & !1)?;
                }
            }
        }
    }
    // node[10] (+0x28) dirty bit — may have been set by the walk THIS crossing.
    if p.read_cell(node + 0x28)? & 1 == 0 {
        return Ok(hr);
    }
    if r(node + 0x50)? != 0 {
        eagl_set_decline_reason(1); // evaluator object (guest vtable call)
        return Err(());
    }
    let tok = r(node)?;
    if tok == -1 {
        eagl_set_decline_reason(2); // compiled node tail (__ftol cursor rewrite)
        return Err(());
    }
    let desc = eagl_token_desc(c, tok)?;
    let rec2 = node + 8;
    let dst = r(node + 0x4c)?;
    let cnt = r(node + 0xa8)?;
    match desc {
        0x6000002 | 0x6000102 | 0x6000202 | 0x6000302 | 0x6000402
        | 0x6000005 | 0x6000105 | 0x6000205 | 0x6000305 | 0x6000405 => {
            hr = eagl_wrapper_apply(cc, cfg, ApplyFamily::Int, ApplyLayout::Register, rec2, dst, cnt, p.commit)?;
            if hr < 0 {
                return Ok(hr);
            }
        },
        0x6000003 | 0x6000006 => {
            hr = eagl_wrapper_apply(cc, cfg, ApplyFamily::Float, ApplyLayout::Register, rec2, dst, cnt, p.commit)?;
            if hr < 0 {
                return Ok(hr);
            }
        },
        0x6000004 | 0x6000007 => {
            hr = eagl_wrapper_apply(cc, cfg, ApplyFamily::Float, ApplyLayout::Packed, rec2, dst, cnt, p.commit)?;
            if hr < 0 {
                return Ok(hr);
            }
        },
        0x6000000 | 0x6000001 => {
            let nx = r(node + 0x60)?;
            if nx != 0 {
                hr = eagl_cdca7_native(c, cc, cfg, nx, depth + 1, p)?;
                if hr < 0 {
                    return Ok(hr);
                }
            }
        },
        0x5000000 => {},
        0x9000000 => {
            let sub = r(cc.handle_tbl.wrapping_add(r(node + 0xc)?.wrapping_mul(4)))?;
            let m = r(sub + 0x3c)?;
            if m as u32 > 256 {
                return Err(());
            }
            let base = r(sub + 0x40)?;
            for i in 0..m {
                hr = eagl_cdca7_native(c, cc, cfg, base.wrapping_add(i.wrapping_mul(0xac)), depth + 1, p)?;
                if hr < 0 {
                    return Ok(hr);
                }
            }
        },
        // Default leg = the inline x87 converter (matrix trims, D3DCOLOR
        // pack) — not replicated; decline the whole crossing (counted).
        _ => {
            eagl_set_decline_reason(5);
            return Err(());
        },
    }
    let d2 = p.read_cell(node + 0x28)?;
    p.write_cell(node + 0x28, d2 & !1)?;
    Ok(hr)
}

/// One ==0-branch constant re-upload (shared by the outer walk and the
/// inner-list 0x6000000/1 leg): wrapper-convert into the shader block, then
/// the vtable constant-F upload as a ring entry. e = the DIRTY-list entry
/// whose [5] indexes the constant table; prec = the shader param record.
/// Returns the guest hr; Err = decline (I/B upload, structural).
unsafe fn eagl_walk_const_upload(
    c: &EaglTokenCfg, cc: &EaglCommitCtx, cfg: i32, e: i32, prec: i32, p: &mut EaglPass,
) -> Result<i32, ()> {
    let r = |a: i32| safe_read32s(a).map_err(|_| ());
    let ct = r(prec + 0x14)?;
    let ebase = ct.wrapping_add(r(ct + 0x10)?);
    let ce = ebase.wrapping_add(r(e + 0x14)?.wrapping_mul(0x14));
    let typ = safe_read16(ce + 4).map_err(|_| ())?;
    let reg = safe_read16(ce + 6).map_err(|_| ())?;
    let cn = safe_read16(ce + 8).map_err(|_| ())?;
    let mut hr;
    match typ {
        2 => {
            let hdrp = r(prec + 0x18)?;
            let f_base = r(hdrp + 0x44)?.wrapping_add(8);
            hr = eagl_wrapper_apply(cc, cfg, ApplyFamily::Int, ApplyLayout::Register,
                r(e + 4)?, f_base.wrapping_add(reg.wrapping_mul(0x10)), cn, p.commit)?;
            if hr < 0 {
                return Ok(hr);
            }
            let kind = r(prec)?;
            if kind == 0x10 {
                eagl_emit_const_f(c, cc.vt, false, cc.dev, reg, f_base.wrapping_add(reg.wrapping_mul(16)), cn, p)?;
                hr = 0;
            } else if kind == 0xf {
                eagl_emit_const_f(c, cc.vt, true, cc.dev, reg, f_base.wrapping_add(reg.wrapping_mul(16)), cn, p)?;
                hr = 0;
            } else {
                hr = EAGL_E_FAIL;
            }
        },
        // Int/bool constant uploads: no WBUF registration for the I/B
        // setters — decline (counted; register fids if content shows these).
        1 | 0 => {
            eagl_set_decline_reason(4);
            return Err(());
        },
        _ => hr = 0, // guest: unhandled type → no upload, falls to *e=0
    }
    Ok(hr)
}

/// Native FUN_005cf304 — the dirty-list commit walk, byte-faithful:
/// circular list (advance-first, head processed last, head slot re-read each
/// iteration), `node[10] |= 1` + `*entry = 1/0` status protocol, the three
/// per-entry branches, the inner dependent-list walk with its unlink writes.
unsafe fn eagl_walk_native(
    c: &EaglTokenCfg, cc: &EaglCommitCtx, cfg: i32, commit_rec: i32, p: &mut EaglPass,
) -> Result<i32, ()> {
    let r = |a: i32| safe_read32s(a).map_err(|_| ());
    let list_rec = r(cc.handle_tbl.wrapping_add(r(commit_rec + 4)?.wrapping_mul(4)))?;
    let head_slot = list_rec + 0x2c;
    let mut hr = 0i32;
    let mut e = r(head_slot)?;
    if e == 0 {
        return Ok(0);
    }
    let mut iters = 0;
    loop {
        iters += 1;
        if iters > 4096 {
            return Err(());
        }
        e = r(e + 0x10)?;
        let node = r(e + 8)?;
        let nflags = p.read_cell(node + 0x28)?;
        p.write_cell(node + 0x28, nflags | 1)?;
        p.write_cell(e, 1)?;
        let lk = r(node + 0x5c)?;
        if lk == 0 {
            if cc.mode == 3 && cc.stamp30 == r(node + 0x58)? && cc.stamp34 == r(node + 0x54)? {
                let prec = eagl_resolve_record(cc.ctx, node)?;
                if r(prec + 0xc)? != 0 && r(prec + 0x14)? != 0 {
                    let hdrp = r(prec + 0x18)?;
                    let hflags = p.read_cell(hdrp + 0x28)?;
                    if hflags & 1 != 0 {
                        hr = eagl_walk_const_upload(c, cc, cfg, e, prec, p)?;
                        if hr < 0 {
                            break;
                        }
                        p.write_cell(e, 0)?;
                        let h2 = p.read_cell(hdrp + 0x28)?;
                        p.write_cell(hdrp + 0x28, h2 & !1)?;
                    }
                }
            }
        } else if lk == -1 {
            let tech = r(node + 0x54)?.wrapping_mul(0xac).wrapping_add(cc.tech_base);
            let tok = r(node)?;
            if tok == -1 {
                eagl_set_decline_reason(2); // compiled shader node
                return Err(());
            }
            let desc = eagl_token_desc(c, tok)?;
            if desc == 0x6000000 || desc == 0x6000001 {
                let prec = eagl_resolve_record(cc.ctx, node)?;
                let ct = r(prec + 0x14)?;
                hr = eagl_scan_80ae(ct, tech, desc == 0x6000001, p)?;
                if hr < 0 {
                    break;
                }
            }
            if cc.mode == 3 && cc.stamp34 == r(node + 0x54)? {
                // Constant-limit growth (the state-block rebuild path) was
                // declined inside eagl_scan_80ae — reaching here means the
                // guest would skip the rebuild and go straight to the tail.
                if cc.stamp30 == r(node + 0x58)? {
                    hr = eagl_cdca7_native(c, cc, cfg, node, 0, p)?;
                    if hr < 0 {
                        break;
                    }
                    hr = eagl_dispatch_token(c, cc.ctx, cc.mode, cc.dev, cc.vt, node, -1, 0, p)?;
                    if hr < 0 {
                        break;
                    }
                }
            }
        } else {
            // Inner dependent-list walk (mode 3 only in the guest).
            if cc.mode == 3 {
                let inner_rec = r(cc.handle_tbl.wrapping_add(lk.wrapping_mul(4)))?;
                let mut ie = r(inner_rec + 0x2c)?;
                if ie != 0 {
                    let mut iters2 = 0;
                    loop {
                        iters2 += 1;
                        if iters2 > 4096 {
                            return Err(());
                        }
                        ie = r(ie + 0x10)?;
                        let n2 = r(ie + 8)?;
                        let lk2 = r(n2 + 0x5c)?;
                        if lk2 == 0 {
                            if cc.stamp30 == r(n2 + 0x58)? && cc.stamp34 == r(n2 + 0x54)? {
                                let prec2 = eagl_resolve_record(cc.ctx, n2)?;
                                if r(prec2)? == 0xf && r(prec2 + 0xc)? != 0 {
                                    let ct2 = r(prec2 + 0x14)?;
                                    if ct2 != 0 {
                                        let reg = safe_read16(
                                            ct2.wrapping_add(r(ct2 + 0x10)?).wrapping_add(6)
                                                .wrapping_add(r(ie + 0x14)?.wrapping_mul(0x14)),
                                        ).map_err(|_| ())?;
                                        hr = eagl_cdca7_native(c, cc, cfg, node, 0, p)?;
                                        if hr < 0 {
                                            return Ok(hr);
                                        }
                                        hr = eagl_dispatch_token(c, cc.ctx, cc.mode, cc.dev, cc.vt, node, reg, 0, p)?;
                                        if hr < 0 {
                                            return Ok(hr);
                                        }
                                    }
                                }
                            }
                        } else if lk2 == -1 {
                            if cc.stamp30 == r(n2 + 0x58)? && cc.stamp34 == r(n2 + 0x54)? {
                                let tok2 = r(n2)?;
                                if tok2 == -1 {
                                    return Err(());
                                }
                                let d2 = eagl_token_desc(c, tok2)?;
                                if d2 == 0x9000000 {
                                    hr = eagl_cdca7_native(c, cc, cfg, node, 0, p)?;
                                    if hr < 0 {
                                        return Ok(hr);
                                    }
                                    hr = eagl_dispatch_token(c, cc.ctx, cc.mode, cc.dev, cc.vt, node, r(n2 + 4)?, 0, p)?;
                                    if hr < 0 {
                                        return Ok(hr);
                                    }
                                } else if d2 == 0x6000000 || d2 == 0x6000001 {
                                    let prec2 = eagl_resolve_record(cc.ctx, n2)?;
                                    if r(prec2 + 0xc)? != 0 && r(prec2 + 0x14)? != 0 {
                                        let hdr2 = r(prec2 + 0x18)?;
                                        let hf = p.read_cell(hdr2 + 0x28)?;
                                        if hf & 1 != 0 {
                                            // Re-upload the OUTER entry's constant
                                            // against this dependent shader.
                                            hr = eagl_walk_const_upload(c, cc, cfg, e, prec2, p)?;
                                            if hr < 0 {
                                                return Ok(hr);
                                            }
                                            p.write_cell(e, 0)?;
                                            let h3 = p.read_cell(hdr2 + 0x28)?;
                                            p.write_cell(hdr2 + 0x28, h3 & !1)?;
                                            break;
                                        }
                                    }
                                }
                            }
                        } else {
                            // Deepest sub-walk: find the dependent ps-shader
                            // entry, re-dispatch the outer node at its reg,
                            // then bump that list's head to the entry's [3].
                            let dl_rec = r(cc.handle_tbl.wrapping_add(lk2.wrapping_mul(4)))?;
                            let dl_head = r(dl_rec + 0x2c)?;
                            if dl_head != 0 {
                                let mut de = dl_head;
                                let mut last_de;
                                let mut iters3 = 0;
                                loop {
                                    iters3 += 1;
                                    if iters3 > 4096 {
                                        return Err(());
                                    }
                                    de = r(de + 0x10)?;
                                    last_de = de;
                                    let n3 = r(de + 8)?;
                                    if cc.stamp30 == r(n3 + 0x58)? && cc.stamp34 == r(n3 + 0x54)? {
                                        let tok3 = r(n3)?;
                                        if tok3 == -1 {
                                            return Err(());
                                        }
                                        if eagl_token_desc(c, tok3)? == 0x6000001 {
                                            let prec3 = eagl_resolve_record(cc.ctx, n3)?;
                                            if r(prec3 + 0xc)? != 0 && r(prec3 + 0x14)? != 0 {
                                                let ct3 = r(prec3 + 0x14)?;
                                                let reg3 = safe_read16(
                                                    ct3.wrapping_add(r(ct3 + 0x10)?).wrapping_add(6)
                                                        .wrapping_add(r(ie + 0x14)?.wrapping_mul(0x14)),
                                                ).map_err(|_| ())?;
                                                hr = eagl_cdca7_native(c, cc, cfg, node, 0, p)?;
                                                if hr < 0 {
                                                    return Ok(hr);
                                                }
                                                hr = eagl_dispatch_token(c, cc.ctx, cc.mode, cc.dev, cc.vt, node, reg3, 0, p)?;
                                                if hr < 0 {
                                                    return Ok(hr);
                                                }
                                                break;
                                            }
                                        }
                                    }
                                    if de == dl_head {
                                        break;
                                    }
                                }
                                p.write_cell(
                                    r(cc.handle_tbl.wrapping_add(r(n2 + 0x5c)?.wrapping_mul(4)))? + 0x2c,
                                    r(last_de + 0xc)?,
                                )?;
                            }
                        }
                        // Termination re-reads the OUTER node's list head.
                        let cur_head = r(r(cc.handle_tbl.wrapping_add(lk.wrapping_mul(4)))? + 0x2c)?;
                        if ie == cur_head {
                            break;
                        }
                    }
                    p.write_cell(
                        r(cc.handle_tbl.wrapping_add(lk.wrapping_mul(4)))? + 0x2c,
                        r(ie + 0xc)?,
                    )?;
                }
            }
        }
        if e == r(head_slot)? {
            break;
        }
    }
    Ok(hr)
}

/// Validate-mode cells in the cfg block.
const CFG_OFF_VALIDATE_REMAINING: i32 = 0x54;
const CFG_OFF_PREDICTED_DELTA: i32 = 0x58;
const CFG_OFF_PREDICTED_HR: i32 = 0x5c;
/// predictedDelta sentinels for the JS fall-through.
const PREDICT_STRUCTURAL: i32 = -1i32; // 0xFFFFFFFF — structural decline
const PREDICT_UNRELIABLE: i32 = -2i32; // 0xFFFFFFFE — model overflow, skip compare

/// Structural-decline attribution: the origin site stamps a
/// reason code; the shell parks it in the predictedHr cell (only meaningful
/// alongside PREDICT_STRUCTURAL) for the JS fall-through's counters.
///   1 evaluator node        5 default-desc (inline converter)
///   2 compiled node         6 pending __controlfp restore
///   3 constant-limit growth 7 iteration/depth/shape bound
///   4 int/bool upload       0 everything else (read fault, stub shape, ...)
static mut EAGL_WALK_DECLINE_REASON: i32 = 0;

#[inline(always)]
fn eagl_set_decline_reason(code: i32) {
    unsafe { EAGL_WALK_DECLINE_REASON = code; }
}

/// Rotating negative-cache slots (cfg +0x70..0x7f): commit records whose walk
/// declined on an evaluator/compiled node. The guest FILTER compares its rec
/// argument against these and routes matches to the original at native speed
/// — without this, each such walk pays OUT + JS + sync-original (~30µs); at
/// the measured 3.2K declines/s that alone was a −12% FPS regression. The JS
/// side ages the slots periodically so a rec that stops carrying evaluator
/// work regains the native path.
const CFG_OFF_DECLINE_CACHE: i32 = 0x70;
static mut EAGL_DECLINE_CACHE_ROTOR: i32 = 0;

/// Shared 133/134 shell: cfg gate, scan, validate-mode protocol, room check,
/// commit replay, single ring-head publish. `decline_cache_key` (0 = off) is
/// parked in a negative-cache slot when the scan declines on an evaluator or
/// compiled node — content-stable decline causes worth bypassing in-guest.
unsafe fn eagl_batch_shell(
    cfg: i32,
    decline_cache_key: i32,
    run: &mut dyn FnMut(&EaglTokenCfg, i32, &mut EaglPass) -> Result<i32, ()>,
) -> bool {
    let ver = match safe_read32s(cfg) { Ok(v) => v, Err(_) => return false };
    if ver != 3 {
        return false;
    }
    let generation = match safe_read32s(cfg + 0x38) { Ok(v) => v, Err(_) => return false };
    {
        let c = &*addr_of!(EAGL_TOKEN_CFG);
        if c.ptr != cfg || c.generation != generation {
            if eagl_token_cfg_refresh(cfg).is_err() {
                return false;
            }
        }
    }
    let c = &*addr_of!(EAGL_TOKEN_CFG);
    // Structural sentinel FIRST — every early decline leaves it in place.
    if safe_write32(cfg + CFG_OFF_PREDICTED_DELTA, PREDICT_STRUCTURAL).is_err() {
        return false;
    }
    eagl_set_decline_reason(0);
    let head0 = match safe_read32s(c.ring_ctrl) { Ok(v) => v, Err(_) => return false };
    if head0 < 0 || head0 > c.capacity {
        return false;
    }
    let mut scan = EaglPass::new(false, head0);
    let hr = match run(c, cfg, &mut scan) {
        Ok(v) => v,
        Err(_) => {
            // Park the origin reason for the JS decline counters (the cell is
            // only read alongside the PREDICT_STRUCTURAL sentinel).
            let reason = EAGL_WALK_DECLINE_REASON;
            let _ = safe_write32(cfg + CFG_OFF_PREDICTED_HR, reason);
            if decline_cache_key != 0 && decline_cache_key != -1 && (reason == 1 || reason == 2) {
                let rotor = EAGL_DECLINE_CACHE_ROTOR & 3;
                let _ = safe_write32(cfg + CFG_OFF_DECLINE_CACHE + rotor * 4, decline_cache_key);
                EAGL_DECLINE_CACHE_ROTOR = rotor + 1;
            }
            return false;
        },
    };
    if scan.head > c.capacity - 64 {
        return false; // no room — JS completes via the original (drain path)
    }
    let validate = safe_read32s(cfg + CFG_OFF_VALIDATE_REMAINING).unwrap_or(0);
    if validate > 0 {
        if scan.model_ovf {
            let _ = safe_write32(cfg + CFG_OFF_PREDICTED_DELTA, PREDICT_UNRELIABLE);
        } else {
            let _ = safe_write32(cfg + CFG_OFF_PREDICTED_DELTA, scan.head - head0);
            let _ = safe_write32(cfg + CFG_OFF_PREDICTED_HR, hr);
            let _ = safe_write32(cfg + CFG_OFF_VALIDATE_REMAINING, validate - 1);
        }
        return false; // JS runs the sync original and compares
    }
    let mut com = EaglPass::new(true, head0);
    let hr = match run(c, cfg, &mut com) {
        Ok(v) => v,
        // The commit replays the scan's exact reads with no guest code in
        // between — an Err here means a write to our own structures failed;
        // publish nothing (head un-bumped ⇒ partial entries invisible).
        Err(_) => return false,
    };
    if safe_write32(c.ring_ctrl, com.head).is_err() {
        return false;
    }
    write_reg32(EAX, hr);
    true
}

/// handler 133 — FUN_005d02d7 (thiscall RET 4): the commit-walk cluster.
/// ECX = ctx, [esp+4] = commit record. Dep-list entries carry their OWN ctx
/// (disasm: ECX reloaded from [entry] per call — Ghidra drops this).
unsafe fn handle_eagl_commit_cluster() -> bool {
    let cfg = *(hp_ptr().add(OFF_HC_EAGL_TOKEN_CFG_PTR) as *const u32) as i32;
    if cfg == 0 {
        return false;
    }
    let esp = read_reg32(ESP);
    let ctx = read_reg32(ECX);
    let rec = match safe_read32s(esp + 4) { Ok(v) => v, Err(_) => return false };
    if rec == 0 || ctx == 0 {
        return false;
    }
    eagl_batch_shell(cfg, rec, &mut |c, cfgp, p| {
        let r = |a: i32| safe_read32s(a).map_err(|_| ());
        let cc = eagl_load_commit_ctx(ctx, None)?;
        let mut last_hr = eagl_walk_native(c, &cc, cfgp, rec, p)?;
        if last_hr >= 0 {
            let dep_rec = r(cc.handle_tbl.wrapping_add(r(rec + 4)?.wrapping_mul(4)))?;
            let dep = r(dep_rec + 0x38)?;
            if dep != 0 {
                let mut de = r(dep + 0x10)?;
                let mut iters = 0;
                while de != dep {
                    iters += 1;
                    if iters > 1024 {
                        return Err(());
                    }
                    let dctx = r(de)?;
                    if dctx != 0 {
                        let dcc = eagl_load_commit_ctx(dctx, None)?;
                        last_hr = eagl_walk_native(c, &dcc, cfgp, r(de + 4)?, p)?;
                        if last_hr < 0 {
                            break;
                        }
                    }
                    de = r(de + 0x10)?;
                }
            }
        }
        Ok(last_hr)
    })
}

/// handler 134 — FUN_005d01ec (stdcall RET 8): the pass-commit driver.
/// [esp+4] = ctx, [esp+8] = pass index. mode==0 / passIdx OOB →
/// D3DERR_INVALIDCALL exactly like the guest; otherwise ctx+0x30 = passIdx
/// (commit-gated; the scan sees it via the stamp30 override) and the element
/// loop runs {apply walk, token dispatch(-1)} per 0xac node.
unsafe fn handle_eagl_pass_driver() -> bool {
    let cfg = *(hp_ptr().add(OFF_HC_EAGL_TOKEN_CFG_PTR) as *const u32) as i32;
    if cfg == 0 {
        return false;
    }
    let esp = read_reg32(ESP);
    let ctx = match safe_read32s(esp + 4) { Ok(v) => v, Err(_) => return false };
    let pass_idx = match safe_read32s(esp + 8) { Ok(v) => v, Err(_) => return false };
    if ctx == 0 {
        return false;
    }
    // Negative-cache key mirrors the guest filter: (techniqueIdx ^ passIdx)+1
    // — stable per material-pass (the evaluator carriers) and never 0, since
    // tech == pass (key 0 pre-increment) is the dominant in-race case.
    let cache_key = match safe_read32s(ctx + 0x34) {
        Ok(v) => (v ^ pass_idx).wrapping_add(1),
        Err(_) => return false,
    };
    eagl_batch_shell(cfg, cache_key, &mut |c, cfgp, p| {
        let r = |a: i32| safe_read32s(a).map_err(|_| ());
        let mode = r(ctx + 0x84)?;
        if mode == 0 {
            return Ok(0x8876086Cu32 as i32);
        }
        let tech = r(ctx + 0x34)?.wrapping_mul(0xac).wrapping_add(r(ctx + 0x1c)?);
        if r(tech + 0x2c)? as u32 <= pass_idx as u32 {
            return Ok(0x8876086Cu32 as i32);
        }
        let cc = eagl_load_commit_ctx(ctx, Some(pass_idx))?;
        p.write_cell(ctx + 0x30, pass_idx)?;
        let pass = pass_idx.wrapping_mul(0x3c).wrapping_add(r(tech + 0x30)?);
        let count = r(pass + 0x24)?;
        if count as u32 > 4096 {
            return Err(());
        }
        let base = r(pass + 0x28)?;
        let mut hr = 0i32;
        for i in 0..count {
            let node = base.wrapping_add(i.wrapping_mul(0xac));
            hr = eagl_cdca7_native(c, &cc, cfgp, node, 0, p)?;
            if hr < 0 {
                break;
            }
            hr = eagl_dispatch_token(c, cc.ctx, cc.mode, cc.dev, cc.vt, node, -1, 0, p)?;
            if hr < 0 {
                break;
            }
        }
        Ok(hr)
    })
}
