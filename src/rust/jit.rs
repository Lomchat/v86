use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::iter::FromIterator;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard};

use crate::analysis;
use crate::analysis::AnalysisType;
use crate::codegen;
use crate::control_flow;
use crate::control_flow::WasmStructure;
use crate::cpu::cpu;
use crate::cpu::global_pointers;
use crate::cpu::hypercall;
use crate::cpu::memory;
use crate::cpu_context::CpuContext;
use crate::jit_instructions;
use crate::opstats;
use crate::page::Page;
use crate::profiler;
use crate::profiler::stat;
use crate::state_flags::CachedStateFlags;
use crate::trace_profiler;
use crate::wasmgen::wasm_builder::{Label, WasmBuilder, WasmLocal, WasmLocalI64};

#[derive(Copy, Clone, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct WasmTableIndex(u16);
impl WasmTableIndex {
    pub fn to_u16(self) -> u16 { self.0 }
}

mod unsafe_jit {
    use super::{CachedStateFlags, WasmTableIndex};

    extern "C" {
        pub fn codegen_finalize(
            wasm_table_index: WasmTableIndex,
            phys_addr: u32,
            state_flags: CachedStateFlags,
            ptr: u32,
            len: u32,
        );
        pub fn jit_clear_func(wasm_table_index: WasmTableIndex);
    }
}

fn codegen_finalize(
    wasm_table_index: WasmTableIndex,
    phys_addr: u32,
    state_flags: CachedStateFlags,
    ptr: u32,
    len: u32,
) {
    unsafe { unsafe_jit::codegen_finalize(wasm_table_index, phys_addr, state_flags, ptr, len) }
}

pub fn jit_clear_func(wasm_table_index: WasmTableIndex) {
    unsafe { unsafe_jit::jit_clear_func(wasm_table_index) }
}

static mut JIT_DISABLED: bool = false;

// Maximum number of pages per wasm module. Necessary for the following reasons:
// - There is an upper limit on the size of a single function in wasm (currently ~7MB in all browsers)
//   See https://github.com/WebAssembly/design/issues/1138
// - v8 poorly handles large br_table elements and OOMs on modules much smaller than the above limit
//   See https://bugs.chromium.org/p/v8/issues/detail?id=9697 and https://bugs.chromium.org/p/v8/issues/detail?id=9141
//   Will hopefully be fixed in the near future by generating direct control flow
static mut MAX_PAGES: u32 = 3;

static mut JIT_USE_LOOP_SAFETY: bool = true;
// Direct cross-module block chaining (idx 4). A direct JMP/Jcc whose successor
// was excluded from the current generated module can tail-call the successor's
// already-published module instead of returning through cycle_internal. Unlike
// the retired first prototype, the target and preemption guards are emitted
// directly in generated wasm and guest registers are spilled only after every
// guard has succeeded. Kept opt-in until a real-game A/B justifies the extra
// code at direct module exits.
static mut JIT_BLOCK_CHAINING: bool = false;
static mut BLOCK_CHAIN_SITES_COMPILED: u32 = 0;
// RET/AbsoluteEip dynamic chaining: when the in-module AbsoluteEip
// re-dispatch misses, attempt a cross-module tail-call at the runtime eip instead of
// exiting to main_loop. Gated at COMPILE time — toggle via
// set_jit_config(12) and clear the JIT cache.
static mut JIT_RET_CHAINING: bool = false;
// Per-AbsoluteEip primary cache in front of the dynamic RET chaining helper
// (idx 30). Unlike RET_CACHE, which is keyed only by runtime EIP and can collide
// across thousands of call sites, this remembers successful targets per
// generated site. A primary hit stays entirely in the generated module; misses
// may probe three bounded positive alternatives before using the exact
// historical helper. Epoch and scheduler-budget guards preserve
// invalidation/preemption semantics.
static mut JIT_DYNAMIC_CHAIN_SITE_PIC: bool = true;
static mut DYNAMIC_CHAIN_SITE_PIC_COMPILED: u32 = 0;
// Diagnostic-only miss classifier for the per-site PIC (idx 31). All work is
// confined to the existing cold helper: generated-cache hits remain completely
// untouched. The shadow second way predicts how many target misses a two-way
// cache would have absorbed without changing dispatch behaviour.
static mut JIT_DYNAMIC_CHAIN_SITE_PIC_DIAG: bool = false;
// Optional second target per site (idx 32). It is consulted only after the
// generated primary path misses, so stable sites retain identical code.
static mut JIT_DYNAMIC_CHAIN_SITE_PIC_SECOND_WAY: bool = true;
// Third and fourth targets are nested behind misses of the first two ways
// (idx 33). Like way two, they never add work to a primary generated hit.
static mut JIT_DYNAMIC_CHAIN_SITE_PIC_FOUR_WAY: bool = true;
// Compile every instruction that is wholly contained in the current physical
// page instead of conservatively abandoning the final MAX_INSTRUCTION_LENGTH
// bytes. Instructions that actually cross the page still stay interpreted: the
// analyser discards them before they enter the block. Runtime-tunable for an A/B
// with set_jit_config(34) + JIT cache clear.
static mut JIT_EXACT_PAGE_TAIL: bool = false;
static mut JIT_EXACT_PAGE_TAIL_INSTRUCTIONS_COMPILED: u32 = 0;
// Compile an instruction whose bytes straddle two guest pages when (and only
// when) the second virtual page currently maps to the physically adjacent page.
// Both physical pages become invalidation dependencies (config 38).
static mut JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS: bool = true;
static mut JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS_COMPILED: u32 = 0;
// Config 50: entry points in the last MAX_INSTRUCTION_LENGTH bytes of a page
// are recorded and compiled like any other. The decoder already proves the
// mapping before keeping an instruction that crosses into the next page
// (config 38), so a block starting in the page tail is safe; without that
// guard such blocks stay interpreted, which on BFME 1 was nine tenths of all
// interpreted blocks in play (the same loop heads, tens of thousands of times
// per second).
static mut JIT_PAGE_TAIL_ENTRIES: bool = true;

#[inline]
pub fn page_tail_entries_enabled() -> bool {
    unsafe { JIT_PAGE_TAIL_ENTRIES && JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS }
}
// Keep relaxed-x87 results in block-scoped wasm locals and materialise them at
// architectural boundaries instead of writing fpu_st after every arithmetic
// instruction (config 39). Experimental until differential and game A/B pass.
static mut JIT_X87_WRITEBACK: bool = false;
// Classify ordinary ordered x87 comparisons before checking the rare unordered
// (NaN) case. Config 40 is compile-time so A/Bs rebuild only the JIT cache.
static mut JIT_FPU_ORDERED_COMPARE_FIRST: bool = true;
// A generated dynamic-chain site already computes the live scheduler guard.
// When that guard fails, returning -1 locally is exactly equivalent to calling
// the shared resolver, which immediately repeats the same guard and returns -1.
// Config 41 keeps the optimization reversible for differential A/Bs.
static mut JIT_DYNAMIC_CHAIN_BUDGET_FAST_EXIT: bool = true;
// REP MOVS bridge with reduced register spilling and completed-copy direct
// continuation (idx 35). The generated call
// passes ESI/EDI/ECX directly to a JIT-aware wasm helper and reloads only those
// three registers. Unsupported memory shapes fall back to the historical full
// spill helper before any guest-visible copy takes place.
static mut JIT_REP_MOVS_REDUCED_SPILL: bool = true;
// Continue a synchronous block-boundary fallthrough (notably OUT hypercall
// stubs) inside the current module instead of returning through cycle_internal.
// The generated edge is guarded by the authoritative runtime EIP, the LIVE
// scheduler budget (the boundary may have changed it), and in_hlt. Any async
// park, interrupt, fault, or preemption request therefore keeps the historical
// module-exit path. Compile-time switch idx 36; enabled after a guarded redirect
// test, +52–55% synthetic throughput, 728,912 directly counted continuations in
// a BFME II setup window, and no regression in a long 3D A/B.
static mut JIT_SYNC_BOUNDARY_CONTINUATION: bool = true;
static mut JIT_SYNC_BOUNDARY_CONTINUATION_SITES_COMPILED: u32 = 0;
// Queue one hot page once when the asynchronous WebAssembly compile window is
// full, then admit it as soon as a slot completes. This avoids re-running the
// cap/compiling scans on every interpreted slice. Config idx 37; enabled after
// the bounded lifecycle benchmark, with a runtime kill-switch retained.
static mut JIT_DEFERRED_COMPILE_QUEUE: bool = true;
static mut DYNAMIC_CHAIN_SITE_PIC_DIAG_CALLS: u64 = 0;
static mut DYNAMIC_CHAIN_SITE_PIC_DIAG_TARGET_MISSES: u64 = 0;
static mut DYNAMIC_CHAIN_SITE_PIC_DIAG_SECOND_WAY_HITS: u64 = 0;
static mut DYNAMIC_CHAIN_SITE_PIC_DIAG_THIRD_WAY_HITS: u64 = 0;
static mut DYNAMIC_CHAIN_SITE_PIC_DIAG_FOURTH_WAY_HITS: u64 = 0;
static mut DYNAMIC_CHAIN_SITE_PIC_DIAG_EPOCH_MISSES: u64 = 0;
static mut DYNAMIC_CHAIN_SITE_PIC_DIAG_GUARD_MISSES: u64 = 0;
static mut DYNAMIC_CHAIN_SITE_PIC_DIAG_RESOLVER_HITS: u64 = 0;
// RET-target speculation (superblock lite): annotate the RET of a
// small module-local leaf with its call sites' return addresses and emit inline
// eip-compare + direct dispatcher re-entry, skipping the jit_find_cache_entry_in_page
// helper on the hot return path. Same-page CALL discovery already splices the callee's
// blocks into the caller's module (follow_jump), so no new SMC surface: the callee's
// page is already in the module's page set. set_jit_config(13); budget idx 14 caps the
// callee's total instruction count (leaf qualification).
static mut JIT_RET_SPECULATION: bool = false;
static mut JIT_RET_SPEC_MAX_INSTR: u32 = 24;
const RET_SPEC_MAX_CANDIDATES: usize = 4;
// Tier-2 direct leaf-call fusion (idx 27): duplicate a tiny, single-basic-block
// direct-call callee at its call site. The architectural CALL and RET still run
// unchanged (including their stack accesses); only the otherwise dynamic RET
// dispatch is replaced by a guarded direct re-entry at the known continuation.
// The runtime EIP guard preserves unusual callees that rewrite their return
// address: a mismatch falls through to the ordinary in-page resolver/exit path.
// Restricting this to promoted modules and a tiny instruction budget bounds wasm
// growth while making the optimization workload-agnostic.
static mut JIT_TIER2_LEAF_CALL_FUSION: bool = true;
// Keep the architectural return EIP in a wasm local while emitting a fused
// leaf. The guarded mismatch path materializes it in instruction_pointer before
// entering the legacy resolver, so this only removes the hot store/load pair.
// Kept behind a separate compile-time switch (idx 28) for controlled A/B.
static mut JIT_TIER2_LEAF_RETURN_LOCAL: bool = true;
// Runtime-tunable only for controlled cross-workload A/B (idx 29). Production
// starts at the validated four-instruction bound.
static mut LEAF_CALL_FUSION_MAX_INSTR: u32 = 4;
static mut LEAF_CALL_FUSION_SITES_COMPILED: u32 = 0;

// B1b: direct-mapped memo in front of the dynamic-chaining tlb_code walk
// (measured 1.5% self + part of the 7% indirect-jump bucket, NFSU in-race).
// Entries are (virt eip, state_flags, packed target, epoch); packed < 0 = empty.
// The cache holds MODULE-LIFETIME data (a packed wasm-table slot + dispatcher state),
// so every event that could invalidate a dispatch target the stock per-dispatch
// re-validation would have caught MUST bump RET_CACHE_EPOCH (O(1) invalidate-all;
// entries stamped with an older epoch miss on probe). Bump sites:
//   - free_wasm_table_index — the ONLY place a wasm-table slot is nulled
//     (jit_clear_func). NOT free_wasm_module: codegen_finalize_finished's
//     module-overwrite path (INVALIDATE_MODULE_UNUSED_AFTER_OVERWRITE) frees the
//     replaced module's index WITHOUT going through free_wasm_module — missing this
//     bump causes a null-function crash (Mechanism 0).
//   - clear_tlb_code, when it actually drops a Code entry — the stock resolver
//     re-derived liveness from tlb_code on every dispatch; after an eviction/remap
//     a memo hit would dispatch a stale-but-live module the resolver would have
//     rejected (Mechanism 1: wrong-code execution, not a trap).
// Fastmem-tracked units (generation != 0) are never cached — their per-dispatch
// generation check cannot be memoized. No per-thread state: entries are eip-keyed,
// and the budget/in_hlt guard still runs before every probe.
const RET_CACHE_SIZE: usize = 512;
static mut RET_CACHE: [(u32, u32, i32, u64); RET_CACHE_SIZE] = [(0, 0, -1, 0); RET_CACHE_SIZE];
// Starts at 1 so zero-initialized entries can never match before their first fill.
static mut RET_CACHE_EPOCH: u64 = 1;
static mut CHAIN_TARGET_EPOCH: u32 = 1;

// One compact primary inline-cache entry per generated AbsoluteEip site, plus
// three bounded positive alternatives consulted only from nested miss arms.
// Slots are monotonic for the complete wasm lifetime. In particular, do not
// reuse them after jit_clear_cache: a parallel compilation invalidated by that
// clear may still finish asynchronously, and reusing its slot would let two
// generated modules alias one memo. Exhaustion is safe and simply falls back to
// the historical resolver. Three SoA arrays keep generated loads at fixed,
// naturally aligned addresses and avoid relying on Rust tuple layout in codegen.
const DYNAMIC_CHAIN_SITE_PIC_COUNT: usize = 1 << 18;
static mut DYNAMIC_CHAIN_SITE_TARGETS: [u32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_PACKED: [i32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_EPOCHS: [u32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_DIAG_SECOND_TARGETS: [u32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_DIAG_SECOND_PACKED: [i32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_DIAG_SECOND_EPOCHS: [u32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_DIAG_THIRD_TARGETS: [u32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_DIAG_THIRD_PACKED: [i32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_DIAG_THIRD_EPOCHS: [u32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_DIAG_FOURTH_TARGETS: [u32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_DIAG_FOURTH_PACKED: [i32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_DIAG_FOURTH_EPOCHS: [u32; DYNAMIC_CHAIN_SITE_PIC_COUNT] =
    [0; DYNAMIC_CHAIN_SITE_PIC_COUNT];
static mut DYNAMIC_CHAIN_SITE_PIC_NEXT: usize = 0;
static mut DYNAMIC_CHAIN_SITE_PIC_HIGH_WATER: u32 = 0;
static mut DYNAMIC_CHAIN_SITE_PIC_OVERFLOWS: u32 = 0;

// Opt-in dynamic-chain miss diagnosis. These counters are updated only while
// dispatch statistics are enabled, so the production resolver keeps its normal
// zero-instrumentation path. A generic RET_CHAIN_MISS alone cannot distinguish a
// scheduler boundary from code that was never compiled, an x86-state mismatch,
// or a compiled module that does not expose the requested entry point.
static mut DYNAMIC_CHAIN_RESOLVE_BUDGET_MISSES: u64 = 0;
static mut DYNAMIC_CHAIN_RESOLVE_NO_META_MISSES: u64 = 0;
static mut DYNAMIC_CHAIN_RESOLVE_STATE_MISSES: u64 = 0;
static mut DYNAMIC_CHAIN_RESOLVE_NO_ENTRY_MISSES: u64 = 0;
static mut DYNAMIC_CHAIN_RESOLVE_MEMO_HITS: u64 = 0;
static mut DYNAMIC_CHAIN_RESOLVE_META_HITS: u64 = 0;

#[no_mangle]
pub unsafe fn jit_dynamic_chain_resolver_diag_reset() {
    DYNAMIC_CHAIN_RESOLVE_BUDGET_MISSES = 0;
    DYNAMIC_CHAIN_RESOLVE_NO_META_MISSES = 0;
    DYNAMIC_CHAIN_RESOLVE_STATE_MISSES = 0;
    DYNAMIC_CHAIN_RESOLVE_NO_ENTRY_MISSES = 0;
    DYNAMIC_CHAIN_RESOLVE_MEMO_HITS = 0;
    DYNAMIC_CHAIN_RESOLVE_META_HITS = 0;
}

#[no_mangle]
pub fn jit_dynamic_chain_resolver_diag_budget_misses() -> u64 {
    unsafe { DYNAMIC_CHAIN_RESOLVE_BUDGET_MISSES }
}
#[no_mangle]
pub fn jit_dynamic_chain_resolver_diag_no_meta_misses() -> u64 {
    unsafe { DYNAMIC_CHAIN_RESOLVE_NO_META_MISSES }
}
#[no_mangle]
pub fn jit_dynamic_chain_resolver_diag_state_misses() -> u64 {
    unsafe { DYNAMIC_CHAIN_RESOLVE_STATE_MISSES }
}
#[no_mangle]
pub fn jit_dynamic_chain_resolver_diag_no_entry_misses() -> u64 {
    unsafe { DYNAMIC_CHAIN_RESOLVE_NO_ENTRY_MISSES }
}
#[no_mangle]
pub fn jit_dynamic_chain_resolver_diag_memo_hits() -> u64 {
    unsafe { DYNAMIC_CHAIN_RESOLVE_MEMO_HITS }
}
#[no_mangle]
pub fn jit_dynamic_chain_resolver_diag_meta_hits() -> u64 {
    unsafe { DYNAMIC_CHAIN_RESOLVE_META_HITS }
}

/// Which hazard forced a global invalidation. The epoch covers two unrelated
/// ones — a recycled table slot and a changed code mapping — and only the first
/// could be narrowed to the affected slot, so knowing the split decides whether
/// narrowing is worth anything.
static mut RET_CACHE_INVALIDATED_BY_SLOT: u32 = 0;
static mut RET_CACHE_INVALIDATED_BY_TLB: u32 = 0;

#[no_mangle]
pub fn jit_ret_cache_invalidations_by_slot() -> u32 { unsafe { RET_CACHE_INVALIDATED_BY_SLOT } }

#[no_mangle]
pub fn jit_ret_cache_invalidations_by_tlb() -> u32 { unsafe { RET_CACHE_INVALIDATED_BY_TLB } }

#[no_mangle]
pub fn jit_ret_cache_invalidations_reset() {
    unsafe {
        RET_CACHE_INVALIDATED_BY_SLOT = 0;
        RET_CACHE_INVALIDATED_BY_TLB = 0;
    }
}

pub fn ret_cache_invalidate_all() {
    unsafe {
        RET_CACHE_EPOCH += 1;
        let next = CHAIN_TARGET_EPOCH.wrapping_add(1);
        CHAIN_TARGET_EPOCH = if next == 0 { 1 } else { next };
    }
}

/// Where module frees come from. In a settled skirmish the slow windows average
/// 260 frees per 10s against 54 in the fast ones, and each free invalidates every
/// return prediction AND the inline caches in generated code, so the pages fall
/// back to interpretation. Tier-2 promotion and the recompile divisor were both
/// ruled out by toggling them without moving the count, so the site has to be
/// attributed rather than guessed.
static mut FREE_SITE_WRITTEN_WHILE_COMPILING: u32 = 0;
static mut FREE_SITE_OVERWRITE: u32 = 0;
static mut FREE_SITE_PAGE_INVALIDATED: u32 = 0;

#[no_mangle]
pub fn jit_free_site_written() -> u32 { unsafe { FREE_SITE_WRITTEN_WHILE_COMPILING } }
#[no_mangle]
pub fn jit_free_site_overwrite() -> u32 { unsafe { FREE_SITE_OVERWRITE } }
#[no_mangle]
pub fn jit_free_site_page_invalidated() -> u32 { unsafe { FREE_SITE_PAGE_INVALIDATED } }
#[no_mangle]
pub fn jit_free_sites_reset() {
    unsafe {
        FREE_SITE_WRITTEN_WHILE_COMPILING = 0;
        FREE_SITE_OVERWRITE = 0;
        FREE_SITE_PAGE_INVALIDATED = 0;
    }
}

pub fn ret_cache_invalidate_all_slot_free() {
    unsafe { RET_CACHE_INVALIDATED_BY_SLOT = RET_CACHE_INVALIDATED_BY_SLOT.wrapping_add(1) };
    ret_cache_invalidate_all();
}

pub fn ret_cache_invalidate_all_tlb() {
    unsafe { RET_CACHE_INVALIDATED_BY_TLB = RET_CACHE_INVALIDATED_BY_TLB.wrapping_add(1) };
    ret_cache_invalidate_all();
}

/// Drop only the return predictions that live on the evicted page (config 47,
/// ON; `dbg.jitConfig(47, 0)` restores the global bump).
///
/// A code-TLB eviction invalidates predictions for that page, but the epoch bump
/// discarded all 512 of them. During a map load the guest remaps constantly —
/// 83-256 evictions per 10s window against none in a settled scene — so the
/// table was repeatedly emptied exactly when the code footprint is largest. The
/// entry key IS the return address, so the affected entries are identifiable,
/// and scanning 512 slots is far cheaper than refilling them: boot measures
/// 140.6 MIPS against 138.6 over three runs per arm, with the eviction count
/// itself unchanged, so only the scope moved.
static mut JIT_NARROW_RET_INVALIDATION: bool = true;

pub fn ret_cache_invalidate_page_tlb(page: u32) {
    unsafe {
        RET_CACHE_INVALIDATED_BY_TLB = RET_CACHE_INVALIDATED_BY_TLB.wrapping_add(1);
        if !JIT_NARROW_RET_INVALIDATION {
            ret_cache_invalidate_all();
            return;
        }
        for entry in RET_CACHE.iter_mut() {
            if entry.0 >> 12 == page {
                *entry = (0, 0, -1, 0);
            }
        }
        // Chain-site memos are keyed by site, not by target page, so they cannot
        // be narrowed the same way and keep the conservative global bump.
        let next = CHAIN_TARGET_EPOCH.wrapping_add(1);
        CHAIN_TARGET_EPOCH = if next == 0 { 1 } else { next };
    }
}

// Count of double-frees the release-safe guard in free_wasm_table_index absorbed.
// Nonzero means a free-discipline bug survives somewhere — investigate, don't shrug.
static mut WASM_TABLE_INDEX_DOUBLE_FREE_SKIPPED: u32 = 0;

#[no_mangle]
pub fn jit_get_double_free_skipped() -> u32 { unsafe { WASM_TABLE_INDEX_DOUBLE_FREE_SKIPPED } }

// B3 hotness tiering: a module whose RE-ENTRY count (bumped per cycle_internal entry —
// the cheapest per-module execution proxy that needs no codegen) crosses the threshold
// gets its pages marked tier-2 and is freed; the ordinary hotness path recompiles it,
// and jit_find_basic_blocks sees the tier-2 marking and compiles with expanded budgets
// (more pages per module + a deeper RET-speculation window). Cold code never pays for
// the expensive compilation. Threshold 0 disables (set_jit_config idx 15); the page-set
// cap bounds runaway promotion (compile-storm guard — once full, no new promotions).
static mut JIT_TIER2_THRESHOLD: u32 = 300_000;
static mut JIT_TIER2_RET_SPEC_MAX_INSTR: u32 = 96;
// Runtime-tunable (set_jit_config idx 17) so the tier-2 module-size budget can be
// A/B'd in-race without a rebuild; 8 was never tuned. Raising it grows only PROMOTED
// modules (cold code keeps the global MAX_PAGES), so the V8 large-function OOM risk
// that forbids raising the global cap doesn't apply at moderate values.
static mut TIER2_MAX_PAGES: u32 = 8;
// Split-range fastmem read shape: two early-exit range tests instead of the
// 4-compare and/or chain — hot below-guard reads drop ~25 → ~10 wasm ops. Same
// acceptance set; A/B via set_jit_config idx 18 + JIT cache clear (shape is baked in
// at module compile time).
static mut JIT_FASTMEM_READ_SPLIT: bool = true;
// Total guest-code pages allowed to retain their tier-2 marking. BFME exhausts
// the former hard-coded 256-page ceiling before entering sustained gameplay,
// permanently starving later hot modules because tier2_pages intentionally
// survives cache clears. Keep the cap bounded but runtime-visible (idx 20) so
// browser A/Bs can distinguish useful coverage from compilation/code-memory cost.
static mut TIER2_PAGE_SET_CAP: u32 = 256;
static mut MODULE_EXEC_COUNTS: [u32; 0x10000] = [0; 0x10000];
// Counting every module re-entry is disproportionately expensive for workloads
// made of tiny cross-module blocks. Sample dispatcher entries and add the
// reciprocal weight. The sample is derived from the architectural instruction
// counter: rejected entries perform no profiler-owned write. Multiplicative
// mixing plus the slot id avoids a plain modulo alias with periodic module cycles.
const TIER2_SAMPLE_SHIFT: u32 = 8;
const TIER2_SAMPLE_WEIGHT: u32 = 1 << TIER2_SAMPLE_SHIFT;
// Mirrored from JitState::tier2_pages so cycle_internal can skip the exported
// note function entirely once the retained page set is full. Calling even the
// threshold==0 fast path for every compiled-module entry is measurable in BFME
// (~2% of the worker on a saturated menu). The comparison remains dynamic: if
// diagnostics raise TIER2_PAGE_SET_CAP later, tracking resumes automatically.
static mut TIER2_PAGE_COUNT: u32 = 0;

// Once the retained-page set is full, do not freeze it forever. Startup and
// loading code can otherwise occupy every slot before a game's steady-state
// simulation begins. The normal per-entry gate stays closed, but every roughly
// four million guest instructions it admits one module as a sparse maintenance
// sample. That is enough to discover a phase change without restoring the
// measurable always-on note_execution cost of the old saturated path.
static mut JIT_TIER2_ADAPTIVE: bool = true;
const TIER2_MAINTENANCE_INTERVAL: u32 = 4_000_003;
const TIER2_MAINTENANCE_WEIGHT: u32 = 16384;
static mut TIER2_MAINTENANCE_NEXT: u32 = TIER2_MAINTENANCE_INTERVAL;
static mut TIER2_MAINTENANCE_DUE: bool = false;
static mut TIER2_MAINTENANCE_TICK: u32 = 1;
static mut TIER2_MAINTENANCE_SAMPLES: u32 = 0;
static mut TIER2_PAGE_EVICTIONS: u32 = 0;

// Tier-2 observability (read via dbg.tier2Stats()): without these there is no way to
// tell "promotions landed" apart from "promotions starved by the page-set cap" — the
// exact ambiguity that made the in-race B3 A/B unreadable (threshold changes showed
// zero FPS delta because the cap, not the threshold, was the limiter candidate).
static mut TIER2_PROMOTIONS: u32 = 0;
static mut TIER2_BLOCKED_BY_CAP: u32 = 0;

// Profile-guided Tier-2 region formation. The profiler observes only the sampled
// module entries after a module has crossed 75% of the promotion threshold. The
// sampling decision is folded into the existing Tier-2 hotness update: ordinary
// exits execute no extra profile check. Each live wasm-table slot keeps a bounded
// Misra-Gries table of runtime successor EIPs. Slots are cleared on allocation/free,
// so table-index reuse can never blend unrelated guest code. The first stage is
// deliberately observation-only; region formation consumes this data below when
// JIT_TIER2_REGIONS is on.
static mut JIT_TIER2_REGIONS: bool = true;
const TIER2_PROFILE_TARGETS: usize = 8;
const TIER2_PROFILE_SAMPLE_CAP: u32 = 4096;
static mut TIER2_EXIT_TARGETS: [[u32; TIER2_PROFILE_TARGETS]; WASM_TABLE_SIZE as usize] =
    [[0; TIER2_PROFILE_TARGETS]; WASM_TABLE_SIZE as usize];
static mut TIER2_EXIT_COUNTS: [[u32; TIER2_PROFILE_TARGETS]; WASM_TABLE_SIZE as usize] =
    [[0; TIER2_PROFILE_TARGETS]; WASM_TABLE_SIZE as usize];
static mut TIER2_PROFILE_SAMPLES: [u32; WASM_TABLE_SIZE as usize] =
    [0; WASM_TABLE_SIZE as usize];
static mut TIER2_PROFILED_EXITS: u32 = 0;
static mut TIER2_REGION_PROMOTIONS: u32 = 0;
static mut TIER2_REGION_SEEDS: u32 = 0;
static mut TIER2_REGION_CANDIDATES: u32 = 0;
static mut TIER2_REGION_REJECTED_TARGET: u32 = 0;
static mut TIER2_REGION_REJECTED_BUDGET: u32 = 0;

#[inline(always)]
pub fn jit_tier2_note_sampled_exit(wasm_table_index: u16, target_eip: u32) {
    unsafe {
        let slot = wasm_table_index as usize;
        debug_assert!(slot < WASM_TABLE_SIZE as usize);
        TIER2_PROFILE_SAMPLES[slot] += 1;
        TIER2_PROFILED_EXITS = TIER2_PROFILED_EXITS.wrapping_add(1);
        let targets = &mut TIER2_EXIT_TARGETS[slot];
        let counts = &mut TIER2_EXIT_COUNTS[slot];
        for i in 0..TIER2_PROFILE_TARGETS {
            if counts[i] != 0 && targets[i] == target_eip {
                counts[i] = counts[i].saturating_add(1);
                return;
            }
        }
        for i in 0..TIER2_PROFILE_TARGETS {
            if counts[i] == 0 {
                targets[i] = target_eip;
                counts[i] = 1;
                return;
            }
        }
        // Misra-Gries eviction: bounded memory and no hash-table work in the hot
        // dispatcher. A genuinely hot successor survives arbitrary cold noise.
        for count in counts.iter_mut() {
            *count -= 1;
        }
    }
}

#[no_mangle]
pub fn jit_get_tier2_profiled_exits() -> u32 { unsafe { TIER2_PROFILED_EXITS } }
#[no_mangle]
pub fn jit_get_tier2_region_promotions() -> u32 { unsafe { TIER2_REGION_PROMOTIONS } }
#[no_mangle]
pub fn jit_get_tier2_region_seeds() -> u32 { unsafe { TIER2_REGION_SEEDS } }
#[no_mangle]
pub fn jit_get_tier2_region_candidates() -> u32 { unsafe { TIER2_REGION_CANDIDATES } }
#[no_mangle]
pub fn jit_get_tier2_region_rejected_target() -> u32 {
    unsafe { TIER2_REGION_REJECTED_TARGET }
}
#[no_mangle]
pub fn jit_get_tier2_region_rejected_budget() -> u32 {
    unsafe { TIER2_REGION_REJECTED_BUDGET }
}

#[no_mangle]
pub fn jit_reset_tier2_state() {
    let mut ctx = get_jit_state();
    ctx.tier2_pages.clear();
    ctx.tier2_page_last_seen.clear();
    ctx.tier2_regions.clear();
    unsafe {
        TIER2_PAGE_COUNT = 0;
        TIER2_PROMOTIONS = 0;
        TIER2_BLOCKED_BY_CAP = 0;
        TIER2_PROFILED_EXITS = 0;
        TIER2_REGION_PROMOTIONS = 0;
        TIER2_REGION_SEEDS = 0;
        TIER2_REGION_CANDIDATES = 0;
        TIER2_REGION_REJECTED_TARGET = 0;
        TIER2_REGION_REJECTED_BUDGET = 0;
        TIER2_MAINTENANCE_NEXT =
            (*global_pointers::instruction_counter).wrapping_add(TIER2_MAINTENANCE_INTERVAL);
        TIER2_MAINTENANCE_DUE = false;
        TIER2_MAINTENANCE_TICK = 1;
        TIER2_MAINTENANCE_SAMPLES = 0;
        TIER2_PAGE_EVICTIONS = 0;
        MODULE_EXEC_COUNTS = [0; 0x10000];
        TIER2_EXIT_TARGETS = [[0; TIER2_PROFILE_TARGETS]; WASM_TABLE_SIZE as usize];
        TIER2_EXIT_COUNTS = [[0; TIER2_PROFILE_TARGETS]; WASM_TABLE_SIZE as usize];
        TIER2_PROFILE_SAMPLES = [0; WASM_TABLE_SIZE as usize];
    }
}

#[no_mangle]
pub fn jit_get_tier2_page_count() -> u32 {
    get_jit_state().tier2_pages.len() as u32
}
#[no_mangle]
pub fn jit_get_tier2_promotions() -> u32 {
    unsafe { TIER2_PROMOTIONS }
}
#[no_mangle]
pub fn jit_get_tier2_blocked_by_cap() -> u32 {
    unsafe { TIER2_BLOCKED_BY_CAP }
}
#[no_mangle]
pub fn jit_get_tier2_maintenance_samples() -> u32 {
    unsafe { TIER2_MAINTENANCE_SAMPLES }
}
#[no_mangle]
pub fn jit_get_tier2_page_evictions() -> u32 {
    unsafe { TIER2_PAGE_EVICTIONS }
}

#[inline(always)]
pub fn jit_tier2_tracking_active() -> bool {
    unsafe {
        if JIT_TIER2_THRESHOLD == 0 {
            return false;
        }
        if TIER2_PAGE_COUNT < TIER2_PAGE_SET_CAP {
            return true;
        }
        JIT_TIER2_ADAPTIVE && TIER2_MAINTENANCE_DUE
    }
}

/// Called once per outer execution slice, not once per JIT module entry. This
/// keeps the saturated steady-state gate as cheap as the historical boolean
/// check while still arming sparse hot-set maintenance after a phase change.
#[inline(always)]
pub fn jit_tier2_maintenance_poll() {
    unsafe {
        if JIT_TIER2_THRESHOLD == 0
            || !JIT_TIER2_ADAPTIVE
            || TIER2_PAGE_COUNT < TIER2_PAGE_SET_CAP
            || TIER2_MAINTENANCE_DUE
        {
            return;
        }
        let now = *global_pointers::instruction_counter;
        if now.wrapping_sub(TIER2_MAINTENANCE_NEXT) < 0x8000_0000 {
            TIER2_MAINTENANCE_DUE = true;
        }
    }
}
/// i-th tier-2 page address (page<<12), 0 when i >= count. Iteration order is the
/// HashSet's (arbitrary but stable between mutations) — callers use this to feed
/// trace2_watch_page with known-hot pages (tier-2 membership == crossed the re-entry
/// threshold), closing the "which pages should Tier-2R watch" loop without an EIP
/// sampler (which only sees idle/yield EIPs — JS timers can't fire mid-cycle-slice).
#[no_mangle]
pub fn jit_get_tier2_page_at(i: u32) -> u32 {
    let ctx = get_jit_state();
    match ctx.tier2_pages.iter().nth(i as usize) {
        Some(p) => p.to_address(),
        None => 0,
    }
}

/// Called from cycle_internal on every compiled-module entry. Bit 0 means that the
/// module was just promoted and freed (the caller must not dispatch into it); bit 1
/// asks the caller to record the EIP reached by this sampled execution. Folding the
/// latter into the existing 1/256 hotness sample avoids a new per-exit profiler tax.
#[inline(always)]
pub fn jit_tier2_note_execution(wasm_table_index: u16) -> u32 {
    let threshold = unsafe { JIT_TIER2_THRESHOLD };
    if threshold == 0 {
        return 0;
    }
    let adaptive_sample = unsafe {
        JIT_TIER2_ADAPTIVE
            && TIER2_PAGE_COUNT >= TIER2_PAGE_SET_CAP
            && TIER2_MAINTENANCE_DUE
    };
    let weight = if adaptive_sample {
        unsafe {
            let now = *global_pointers::instruction_counter;
            TIER2_MAINTENANCE_NEXT = now.wrapping_add(TIER2_MAINTENANCE_INTERVAL);
            TIER2_MAINTENANCE_DUE = false;
            TIER2_MAINTENANCE_TICK = TIER2_MAINTENANCE_TICK.wrapping_add(1).max(1);
            TIER2_MAINTENANCE_SAMPLES = TIER2_MAINTENANCE_SAMPLES.wrapping_add(1);
        }
        TIER2_MAINTENANCE_WEIGHT
    }
    else if threshold >= TIER2_SAMPLE_WEIGHT {
        let sample = unsafe {
            (*global_pointers::instruction_counter as u32)
                .wrapping_mul(0x9E37_79B9)
                ^ (wasm_table_index as u32).wrapping_mul(0x85EB_CA6B)
        };
        if sample >> (32 - TIER2_SAMPLE_SHIFT) != 0 {
            return 0;
        }
        TIER2_SAMPLE_WEIGHT
    }
    else {
        1
    };
    if adaptive_sample {
        let tick = unsafe { TIER2_MAINTENANCE_TICK };
        let mut ctx = get_jit_state();
        let index = WasmTableIndex(wasm_table_index);
        let live_pages: Vec<Page> = ctx
            .pages
            .iter()
            .filter(|(_, info)| info.wasm_table_index == index)
            .map(|(page, _)| *page)
            .collect();
        for page in live_pages {
            if ctx.tier2_pages.contains(&page) {
                ctx.tier2_page_last_seen.insert(page, tick);
            }
        }
    }

    let count = unsafe {
        let c = &mut (*std::ptr::addr_of_mut!(MODULE_EXEC_COUNTS))[wasm_table_index as usize];
        *c = c.saturating_add(weight);
        *c
    };
    let sample_exit = unsafe {
        JIT_TIER2_REGIONS
            && TIER2_PROFILE_SAMPLES[wasm_table_index as usize] < TIER2_PROFILE_SAMPLE_CAP
            && if TIER2_PAGE_COUNT < TIER2_PAGE_SET_CAP {
                count >= threshold / 4 * 3
            }
            else {
                // Once the bounded set is full, ordinary entries still pay
                // nothing. The already-sparse adaptive maintenance admission
                // must nevertheless retain its successor EIP: otherwise a
                // later gameplay phase can replace startup pages but can never
                // form a fused region for its newly-hot cross-module path.
                adaptive_sample
            }
    };
    if count < threshold {
        return if sample_exit { 2 } else { 0 };
    }
    unsafe { MODULE_EXEC_COUNTS[wasm_table_index as usize] = 0 };

    let mut ctx = get_jit_state();
    let index = WasmTableIndex(wasm_table_index);
    let pages: Vec<Page> = ctx
        .pages
        .iter()
        .filter(|(_, info)| info.wasm_table_index == index)
        .map(|(p, _)| *p)
        .collect();
    if pages.is_empty() {
        return 0;
    }
    // Already fully tier-2? Nothing to gain from another free/recompile churn.
    if pages.iter().all(|p| ctx.tier2_pages.contains(p)) {
        return 0;
    }
    let mut promoted_pages: HashSet<Page> = pages.iter().copied().collect();
    let mut region_seeds = Vec::new();

    if unsafe { JIT_TIER2_REGIONS } {
        let source_state = ctx
            .pages
            .values()
            .find(|info| info.wasm_table_index == index)
            .map(|info| info.state_flags);
        let mut candidates: Vec<(u32, u32)> = unsafe {
            TIER2_EXIT_TARGETS[wasm_table_index as usize]
                .iter()
                .copied()
                .zip(TIER2_EXIT_COUNTS[wasm_table_index as usize].iter().copied())
                .filter(|&(_, count)| count != 0)
                .collect()
        };
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        let total: u64 = candidates.iter().map(|&(_, count)| count as u64).sum();

        for (target, hits) in candidates {
            if total == 0
                || hits as u64 * 100 < total * 5
                || region_target_excluded(target)
            {
                continue;
            }
            unsafe { TIER2_REGION_CANDIDATES = TIER2_REGION_CANDIDATES.wrapping_add(1) };
            let phys = match cpu::translate_address_read_no_side_effects(target as i32) {
                Ok(phys) => phys,
                Err(()) => {
                    unsafe {
                        TIER2_REGION_REJECTED_TARGET =
                            TIER2_REGION_REJECTED_TARGET.wrapping_add(1)
                    };
                    continue;
                },
            };
            let target_page = Page::page_of(phys);
            let target_info = match ctx.pages.get(&target_page) {
                Some(info)
                    if Some(info.state_flags) == source_state
                        && info.wasm_table_index != index
                        && info
                            .entry_points
                            .iter()
                            .any(|&(offset, _)| offset == phys as u16 & 0xFFF) => info,
                _ => {
                    unsafe {
                        TIER2_REGION_REJECTED_TARGET =
                            TIER2_REGION_REJECTED_TARGET.wrapping_add(1)
                    };
                    continue;
                },
            };
            let target_index = target_info.wasm_table_index;
            let target_pages: Vec<Page> = ctx
                .pages
                .iter()
                .filter(|(_, info)| info.wasm_table_index == target_index)
                .map(|(page, _)| *page)
                .collect();
            let added = target_pages
                .iter()
                .filter(|page| !promoted_pages.contains(page))
                .count();
            if promoted_pages.len() + added > unsafe { TIER2_MAX_PAGES as usize } {
                unsafe {
                    TIER2_REGION_REJECTED_BUDGET = TIER2_REGION_REJECTED_BUDGET.wrapping_add(1)
                };
                continue;
            }
            promoted_pages.extend(target_pages);
            region_seeds.push(target as i32);
        }
    }

    let newly_promoted = promoted_pages
        .iter()
        .filter(|page| !ctx.tier2_pages.contains(page))
        .count();
    let cap = unsafe { TIER2_PAGE_SET_CAP as usize };
    if promoted_pages.len() > cap {
        unsafe { TIER2_BLOCKED_BY_CAP += 1 };
        return 0;
    }
    let required = ctx.tier2_pages.len() + newly_promoted;
    if required > cap {
        if !unsafe { JIT_TIER2_ADAPTIVE } {
            unsafe { TIER2_BLOCKED_BY_CAP += 1 };
            return 0;
        }
        let need = required - cap;
        let now = unsafe { TIER2_MAINTENANCE_TICK };
        let mut candidates: Vec<(Page, u32)> = ctx
            .tier2_pages
            .iter()
            .filter(|page| !promoted_pages.contains(page))
            .map(|page| {
                let seen = ctx.tier2_page_last_seen.get(page).copied().unwrap_or(0);
                (*page, now.wrapping_sub(seen))
            })
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        let evicted: HashSet<Page> = candidates
            .into_iter()
            .take(need)
            .map(|(page, _)| page)
            .collect();
        if evicted.len() != need {
            unsafe { TIER2_BLOCKED_BY_CAP += 1 };
            return 0;
        }
        for page in &evicted {
            ctx.tier2_pages.remove(page);
            ctx.tier2_page_last_seen.remove(page);
        }
        ctx.tier2_regions
            .retain(|_, region| region.pages.is_disjoint(&evicted));
        unsafe { TIER2_PAGE_EVICTIONS = TIER2_PAGE_EVICTIONS.wrapping_add(need as u32) };
    }
    let tick = unsafe { TIER2_MAINTENANCE_TICK };
    for p in &promoted_pages {
        ctx.tier2_pages.insert(*p);
        ctx.tier2_page_last_seen.insert(*p, tick);
    }
    if unsafe { JIT_TIER2_REGIONS } && !region_seeds.is_empty() {
        let region = Tier2Region {
            pages: promoted_pages.clone(),
            seeds: region_seeds.clone(),
        };
        for page in &pages {
            ctx.tier2_regions.insert(*page, region.clone());
        }
        unsafe {
            TIER2_REGION_PROMOTIONS += 1;
            TIER2_REGION_SEEDS += region_seeds.len() as u32;
        }
    }
    unsafe {
        TIER2_PAGE_COUNT = ctx.tier2_pages.len() as u32;
        TIER2_PROMOTIONS += 1;
    }
    free_wasm_module_tree(&mut ctx, index);
    1
}
static mut JIT_DEAD_FLAG_ELISION: bool = false;

/// Prove flags dead across instructions that can fault.
///
/// The walk stops at a faulting overwriter because a #PF before the overwrite
/// would leave the fault frame needing the architectural flags. That is right
/// for a system emulator, and it is why only 11% of candidates are proven dead
/// on a real BFME 1 frame (8,358 of 76,649): `mov` is 32% of this binary, so
/// almost every walk meets a memory access first.
///
/// The flags a fault frame would carry are only observable if a guest exception
/// handler reads EFlags out of its CONTEXT — which SEH allows and essentially no
/// game does. QEMU's TCG makes the same trade. Off by default because the risk
/// is real rather than theoretical; the payoff is the other 89%.
static mut JIT_DEAD_FLAG_ELISION_ACROSS_FAULTS: bool = false;
static mut JIT_FASTMEM_READS: bool = false;
static mut JIT_X87_LOCALS: bool = false;
static mut JIT_PUSH_RUN_COALESCING: bool = false;
// Fastmem WRITES behind a per-page writability map (idx 19).
// Off by default until the in-game gate passes.
static mut JIT_FASTMEM_WRITES: bool = false;
// Lazy-flag tuple in wasm
// LOCALS instead of linear-memory globals 96-120 (idx 21, default OFF). Removes
// per-ALU-op flag stores AND their TurboFan aliasing barriers. Correctness
// contract: locals are authoritative between spills; the builder-level call_fn
// funnel spills/reloads around every non-whitelisted helper call (covers arith
// flag-protocol helpers AND OUT/hypercall context saves), and the
// module epilogues spill at every exit.
static mut JIT_FLAG_LOCALS: bool = false;

// Inline the current-module AbsoluteEip resolver in generated wasm (idx 22).
// Every RET / indirect jump first asks whether its runtime target is another
// dispatcher entry in the same compiled module. DISPATCH_META / DISPATCH_SLABS
// already live in the shared linear memory, so calling back into the base Rust
// module just to perform two loads and comparisons adds an avoidable wasm-module
// boundary on one of the hottest x86 control-flow paths. The generated lookup is
// deliberately a byte-for-byte semantic mirror of jit_find_cache_entry_in_page:
// state flags and table slot must both match, and a u16::MAX state still means
// miss. Compile-time gated so a live workload can A/B it after a JIT-cache clear.
static mut JIT_INLINE_INTRA_MODULE_DISPATCH: bool = true;
static mut INLINE_INTRA_MODULE_DISPATCH_SITES_COMPILED: u32 = 0;

// Tier-2R region recompiler: grow page groups across
// indirect edges using trace_profiler target histograms, and make hot indirect
// targets dispatcher entries so AbsoluteEip re-dispatches stay intra-module.
// Off by default; requires collected trace2 data to have any effect.
static mut JIT_INDIRECT_REGIONS: bool = false;

// Region-growth safety: virtual-address range that indirect-region growth must NEVER pull
// targets from — the thunk/callback/spin bucket. Guest code indirect-calls thunk
// stubs constantly (GetProcAddress'd exports), so profiled indirect targets point
// into stub pages; compiling those into a guest superblock traps `unreachable`
// at CALLBACK_STUB+0x477f0. Set by JS via
// jit_set_region_exclusion (scheduler arms it with [THUNK_CODE_BASE, ROM_BASE)).
// hi == 0 → no exclusion (feature off, e.g. older TS).
static mut REGION_EXCLUDE_LO: u32 = 0;
static mut REGION_EXCLUDE_HI: u32 = 0;

#[no_mangle]
pub fn jit_set_region_exclusion(lo: u32, hi: u32) {
    unsafe {
        REGION_EXCLUDE_LO = lo;
        REGION_EXCLUDE_HI = hi;
    }
}

fn region_target_excluded(target: u32) -> bool {
    unsafe { REGION_EXCLUDE_HI != 0 && target >= REGION_EXCLUDE_LO && target < REGION_EXCLUDE_HI }
}
static mut JIT_INDIRECT_REGION_MIN_SHARE: u32 = 5; // percent of per-site hits
const JIT_INDIRECT_REGION_MAX_TARGETS: usize = 16;
// Page budget for region growth ACROSS INDIRECT EDGES only — kept separate from
// the global MAX_PAGES (which caps normal direct-jump BFS at 3). Raising the
// global cap instead bloats EVERY module via long direct-call chains and OOMs
// V8 on NFSU (large generated functions / br_tables — v86's own warning). Here
// only a dispatcher block that hits recorded hot targets grows, and only up to
// this many pages, prioritising the hottest targets first.
static mut JIT_INDIRECT_REGION_MAX_PAGES: u32 = 8;

pub static mut MAX_EXTRA_BASIC_BLOCKS: u32 = 250;

// Block-chaining dispatch characterisation toggle.
// When enabled, the JIT emits the always-on dispatch-characterisation counters (BLOCK_EXECUTION
// and MODULE_EXIT_*) and the runtime increments MODULE_REENTRY / MODULE_EXIT_INDIRECT. The
// codegen-emitted counters are gated at COMPILE time, so enable this BEFORE the workload compiles
// its hot modules (set_dispatch_stats(1) at boot, then clear the JIT cache) and read the result
// via profiler_dispatch_stat_get. OFF by default — zero cost on the production path.
pub static mut DISPATCH_STATS: bool = false;
pub fn dispatch_stats_enabled() -> bool { unsafe { DISPATCH_STATS } }
fn block_chaining_enabled() -> bool { unsafe { JIT_BLOCK_CHAINING } }
fn ret_chaining_enabled() -> bool { unsafe { JIT_RET_CHAINING } }
fn ret_speculation_enabled() -> bool { unsafe { JIT_RET_SPECULATION } }
fn dead_flag_elision_enabled() -> bool { unsafe { JIT_DEAD_FLAG_ELISION } }
fn dead_flag_elision_across_faults() -> bool { unsafe { JIT_DEAD_FLAG_ELISION_ACROSS_FAULTS } }

// Indices 0/1 remain as stable stat-array slots + TS label names ('tlbFullClear',
// 'tlbClear'); the TLB-clear sites no longer bump (see cpu.rs), so nothing emits
// them from Rust now — kept for the stats layout and possible future use.
#[allow(dead_code)]
pub const FASTMEM_BUMP_TLB_FULL_CLEAR: u32 = 0;
#[allow(dead_code)]
pub const FASTMEM_BUMP_TLB_CLEAR: u32 = 1;
pub const FASTMEM_BUMP_INVLPG: u32 = 2;
#[allow(dead_code)]
pub const FASTMEM_BUMP_ADDRESS_SPACE_PROTECT: u32 = 3;
#[allow(dead_code)]
pub const FASTMEM_BUMP_ADDRESS_SPACE_RELEASE: u32 = 4;
#[allow(dead_code)]
pub const FASTMEM_BUMP_PAGE_TABLE_DECOMMIT: u32 = 5;
#[allow(dead_code)]
pub const FASTMEM_BUMP_PAGE_TABLE_COMMIT: u32 = 6;
#[allow(dead_code)]
pub const FASTMEM_BUMP_PAGE_TABLE_PROTECT: u32 = 7;
pub const FASTMEM_BUMP_WRITE_WATCH: u32 = 8;
#[allow(dead_code)]
pub const FASTMEM_BUMP_MANUAL: u32 = 9;
const FASTMEM_BUMP_SOURCE_COUNT: usize = 10;

static mut FASTMEM_BUMPS_BY_SOURCE: [u32; FASTMEM_BUMP_SOURCE_COUNT] =
    [0; FASTMEM_BUMP_SOURCE_COUNT];
static mut FASTMEM_SPECULATED_LOADS_COMPILED: u32 = 0;
static mut FASTMEM_DEOPT_RECOMPILES: u32 = 0;
static mut FASTMEM_THRASH_LATCHED: bool = false;

// ── Fastmem WRITE map ─────────────────────────────────────────────────────────────
// One byte per 4 KB VA page across the full 4 GB space (1 MB static). The JIT store
// fast path (codegen::gen_fastmem_write_map) accepts a page IFF its byte == 1, so the
// byte is a bitfield where every restriction independently vetoes the fast path:
//   bit0  BASE_WRITABLE  committed, RW, plain identity-mapped RAM   — owner: TS choke points
//   bit1  HAS_CODE       page holds compiled code (SMC net)         — owner: rust tlb_set_has_code
//   bit2  WRITE_WATCH    debug write-watch armed on this page       — owner: rust dbg_set_write_watch
// Unlike read speculation this carries NO stale window and NO generation guard: the map
// is DATA, read per store, updated synchronously at the same choke points that keep the
// TLB honest, so compiled code never goes stale. A stale-writable byte on an
// RO/CoW/code page would be silent memory corruption, so the ONLY safe
// failure direction is leaving a byte != 1 (slow path, byte-precise). Init all zeros =
// conservative = correct; TS marks RW ranges as regions register/commit during boot.
pub const FASTMEM_WRITE_MAP_LEN: usize = 1 << 20; // 4 GB / 4 KB pages, one byte each
static mut FASTMEM_WRITE_MAP: [u8; FASTMEM_WRITE_MAP_LEN] = [0; FASTMEM_WRITE_MAP_LEN];
const FASTMEM_WRITE_BASE_WRITABLE: u8 = 1 << 0;
const FASTMEM_WRITE_HAS_CODE: u8 = 1 << 1;
const FASTMEM_WRITE_WATCH: u8 = 1 << 2;
static mut FASTMEM_SPECULATED_STORES_COMPILED: u32 = 0;
// Highest VA page for which TS has ever set bit0 — bounds the audit/count scans so they
// don't walk the whole 1 MB map every call (the populated prefix is tiny in practice).
static mut FASTMEM_WRITE_MAP_MAX_PAGE: u32 = 0;
// Hard "never fast-writable" page band [lo, hi) — TS points this at THUNK_CODE (which
// holds immutable RX stubs but has RW PTEs under the identity map, so a kind-blind PTE
// path could otherwise set bit0 on it). bit0 SET is refused inside this band regardless
// of caller. hi == 0 ⇒ no exclusion (feature off / older TS).
static mut FASTMEM_WRITE_EXCLUDE_LO_PAGE: u32 = 0;
static mut FASTMEM_WRITE_EXCLUDE_HI_PAGE: u32 = 0;

// ── DOD dispatch metadata ─────────────────────────────────────────────────────────
// Replaces the Box<Code> layout behind the old `cpu::tlb_code` pointer array. The
// hot resolvers (jit_find_cache_entry* — called on EVERY guest ret/indirect jump,
// hit or miss) previously walked THREE dependent loads, two through heap pointers:
//   tlb_code[page] → *Box<Code> → .state_table[offset]
// a cache-miss-bound pointer chase. The SoA replacement derives every address
// from `page` alone — the loads are INDEPENDENT and issue in parallel:
//   DISPATCH_META[page]                        ; packed word, dense 8 MB array
//   DISPATCH_SLABS[slab*0x1000 + (addr&0xFFF)] ; dense u16 pool, no chase
//
// meta packing: state_flags(u32) << 32 | wasm_table_index(u16) << 16 | slab(u16).
// meta == 0 ⇒ page has no compiled code. Slab index 0 is RESERVED-invalid so the
// zero word stays an unambiguous sentinel; usable slabs are 1..DISPATCH_SLAB_COUNT.
//
// The per-unit fastmem generation was deliberately DROPPED from the lookup path:
// a stale-generation unit that gets dispatched self-deopts via its own prologue
// guard (jit_generate_module's fastmem_generation check) on entry — one extra
// bounce right after a generation bump, identical observable behavior.
//
// Maintenance funnels are exactly the old tlb_code writers: set_tlb_code (compile/
// TLB-fill) and cpu::clear_tlb_code (eviction/invlpg/dirty) — no new choke points.
pub const DISPATCH_SLAB_COUNT: usize = 4096; // 4096 × 8 KB = 32 MB pool
static mut DISPATCH_META: [u64; 1 << 20] = [0; 1 << 20];
// Second dispatch table for external (ahead-of-time) modules, consulted only
// when the JIT's table has no entry for the address. Same packing, same slab
// pool, so a page can carry a JIT module and an external module at once and
// neither evicts the other.
static mut DISPATCH_META_EXT: [u64; 1 << 20] = [0; 1 << 20];
static mut DISPATCH_SLABS: [u16; DISPATCH_SLAB_COUNT * 0x1000] =
    [0; DISPATCH_SLAB_COUNT * 0x1000];
// Free stack of slab indices; filled 1..DISPATCH_SLAB_COUNT by rust_init.
static mut DISPATCH_SLAB_FREE: [u16; DISPATCH_SLAB_COUNT] = [0; DISPATCH_SLAB_COUNT];
static mut DISPATCH_SLAB_FREE_TOP: usize = 0;
static mut DISPATCH_SLAB_HIGH_WATER: u32 = 0;
static mut DISPATCH_SLAB_OVERFLOWS: u32 = 0;

// Exact cross-module dispatch index. The page-local DOD table above deliberately
// publishes one wasm module per virtual page, which is ideal for in-module RET
// dispatch but hides older modules that are still alive through another page.
// Direct JMP/Jcc chaining used to probe only that latest owner and therefore
// missed roughly 92% of BFME's otherwise-chainable exits.
//
// This open-addressed side index retains every exact (virtual EIP, architectural
// state) entry. Values carry the wasm-table generation, so recycling a table slot
// invalidates all of its old entries in O(1); stale buckets are reclaimed by later
// inserts. A matching older module is safe to use: exact EIP and state are equal,
// and its normal fastmem-generation prologue still self-deopts stale translations.
const EXACT_DISPATCH_BITS: usize = 20;
const EXACT_DISPATCH_SIZE: usize = 1 << EXACT_DISPATCH_BITS;
const EXACT_DISPATCH_MASK: usize = EXACT_DISPATCH_SIZE - 1;
const EXACT_DISPATCH_MAX_PROBES: usize = 12;
static mut EXACT_DISPATCH_KEYS: [u64; EXACT_DISPATCH_SIZE] = [0; EXACT_DISPATCH_SIZE];
// generation:u32 | wasm_table_index:u16 | unit_state:u16. Zero is empty because
// generated modules never use table index zero.
static mut EXACT_DISPATCH_VALUES: [u64; EXACT_DISPATCH_SIZE] = [0; EXACT_DISPATCH_SIZE];
static mut EXACT_DISPATCH_GENERATIONS: [u32; 1 << 16] = [0; 1 << 16];
static mut EXACT_DISPATCH_INSERTS: u32 = 0;
static mut EXACT_DISPATCH_HITS: u32 = 0;
static mut EXACT_DISPATCH_MISSES: u32 = 0;
static mut EXACT_DISPATCH_OVERFLOWS: u32 = 0;
static mut EXACT_DISPATCH_PUBLISH_EPOCH: u32 = 1;

// One generation-checked target memo per generated direct-exit site. A hot site
// loads this single u64 and bypasses the exact hash entirely; only an empty or
// stale memo calls the exact resolver. Slots are reset on full cache clears and
// otherwise allocated monotonically, so live generated code never aliases them.
const CHAIN_SITE_MEMO_COUNT: usize = 1 << 20;
static mut CHAIN_SITE_MEMOS: [u64; CHAIN_SITE_MEMO_COUNT] = [0; CHAIN_SITE_MEMO_COUNT];
static mut CHAIN_SITE_MEMO_NEXT: usize = 0;
static mut CHAIN_SITE_MEMO_HIGH_WATER: u32 = 0;
static mut CHAIN_SITE_MEMO_OVERFLOWS: u32 = 0;

#[inline]
fn exact_dispatch_hash(key: u64) -> usize {
    let mut x = key as u32 ^ (key >> 32) as u32;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x as usize & EXACT_DISPATCH_MASK
}

#[inline]
unsafe fn exact_dispatch_value_is_live(value: u64) -> bool {
    if value == 0 {
        return false;
    }
    let table = ((value >> 16) & 0xFFFF) as usize;
    let generation = (value >> 32) as u32;
    table != 0 && EXACT_DISPATCH_GENERATIONS[table] == generation
}

unsafe fn exact_dispatch_insert(
    virt_address: u32,
    state_flags: CachedStateFlags,
    wasm_table_index: WasmTableIndex,
    unit_state: u16,
) {
    let table = wasm_table_index.to_u16() as usize;
    dbg_assert!(table != 0 && unit_state != u16::MAX);
    let key = (state_flags.to_u32() as u64) << 32 | virt_address as u64;
    let generation = EXACT_DISPATCH_GENERATIONS[table];
    let value = (generation as u64) << 32 | (table as u64) << 16 | unit_state as u64;
    let start = exact_dispatch_hash(key);

    for probe in 0..EXACT_DISPATCH_MAX_PROBES {
        let slot = (start + probe) & EXACT_DISPATCH_MASK;
        let old_value = EXACT_DISPATCH_VALUES[slot];
        if old_value == value && EXACT_DISPATCH_KEYS[slot] == key {
            return;
        }
        if old_value == 0 || !exact_dispatch_value_is_live(old_value) {
            EXACT_DISPATCH_KEYS[slot] = key;
            EXACT_DISPATCH_VALUES[slot] = value;
            EXACT_DISPATCH_INSERTS = EXACT_DISPATCH_INSERTS.saturating_add(1);
            let next = EXACT_DISPATCH_PUBLISH_EPOCH.wrapping_add(1);
            EXACT_DISPATCH_PUBLISH_EPOCH = if next == 0 { 1 } else { next };
            return;
        }
        // Keep an already-live exact translation, even if it belongs to an older
        // module. It is semantically equivalent and avoids churn on page overwrite.
        if EXACT_DISPATCH_KEYS[slot] == key {
            return;
        }
    }

    // A full probe window is only a performance miss; main_loop remains the exact
    // correctness fallback. Keep the event visible for capacity tuning.
    EXACT_DISPATCH_OVERFLOWS = EXACT_DISPATCH_OVERFLOWS.saturating_add(1);
}

/// Resolve any still-live module at an exact virtual EIP/state pair. The packed
/// return is (actual wasm table slot << 16) | unit_state, or -1 on miss.
#[no_mangle]
pub unsafe fn jit_find_cache_entry_exact_chain(virt_address: u32, raw_state_flags: u32) -> i32 {
    let key = (raw_state_flags as u64) << 32 | virt_address as u64;
    let start = exact_dispatch_hash(key);

    for probe in 0..EXACT_DISPATCH_MAX_PROBES {
        let slot = (start + probe) & EXACT_DISPATCH_MASK;
        let value = EXACT_DISPATCH_VALUES[slot];
        if value == 0 {
            EXACT_DISPATCH_MISSES = EXACT_DISPATCH_MISSES.saturating_add(1);
            return -1;
        }
        if EXACT_DISPATCH_KEYS[slot] == key && exact_dispatch_value_is_live(value) {
            EXACT_DISPATCH_HITS = EXACT_DISPATCH_HITS.saturating_add(1);
            let table = ((value >> 16) & 0xFFFF) as i32 + cpu::WASM_TABLE_OFFSET as i32;
            let unit_state = (value & 0xFFFF) as i32;
            return table << 16 | unit_state;
        }
    }

    EXACT_DISPATCH_MISSES = EXACT_DISPATCH_MISSES.saturating_add(1);
    -1
}

/// Cold path for one generated direct-exit site: resolve the exact target and
/// memoize its current table generation. The hot generated path validates that
/// generation before using the packed target, so table-slot recycling is safe.
#[no_mangle]
pub unsafe fn jit_find_cache_entry_exact_chain_memo(
    virt_address: u32,
    raw_state_flags: u32,
    memo_slot: u32,
) -> i32 {
    let packed = jit_find_cache_entry_exact_chain(virt_address, raw_state_flags);
    if (memo_slot as usize) < CHAIN_SITE_MEMO_COUNT {
        if packed >= 0 {
            CHAIN_SITE_MEMOS[memo_slot as usize] =
                (CHAIN_TARGET_EPOCH as u64) << 32 | packed as u32 as u64;
        }
        else {
            // Negative memo. Any later exact-target publication bumps the epoch,
            // making generated code retry this site automatically.
            CHAIN_SITE_MEMOS[memo_slot as usize] =
                (EXACT_DISPATCH_PUBLISH_EPOCH as u64) << 32 | u32::MAX as u64;
        }
    }
    packed
}

fn allocate_chain_site_memo() -> Option<(u32, u32)> {
    unsafe {
        if CHAIN_SITE_MEMO_NEXT >= CHAIN_SITE_MEMO_COUNT {
            CHAIN_SITE_MEMO_OVERFLOWS = CHAIN_SITE_MEMO_OVERFLOWS.saturating_add(1);
            return None;
        }
        let slot = CHAIN_SITE_MEMO_NEXT;
        CHAIN_SITE_MEMO_NEXT += 1;
        CHAIN_SITE_MEMOS[slot] = 0;
        CHAIN_SITE_MEMO_HIGH_WATER =
            CHAIN_SITE_MEMO_HIGH_WATER.max(CHAIN_SITE_MEMO_NEXT as u32);
        let address = std::ptr::addr_of!(CHAIN_SITE_MEMOS) as u32 + (slot as u32 * 8);
        Some((slot as u32, address))
    }
}

pub fn dispatch_meta_init() {
    unsafe {
        // Stack of free slabs, slab 0 excluded (reserved sentinel).
        for i in 1..DISPATCH_SLAB_COUNT {
            DISPATCH_SLAB_FREE[i - 1] = i as u16;
        }
        DISPATCH_SLAB_FREE_TOP = DISPATCH_SLAB_COUNT - 1;
        for i in 0..(1 << 16) {
            EXACT_DISPATCH_GENERATIONS[i] = 1;
        }
    }
}

#[inline]
pub fn dispatch_meta_get(page: u32) -> u64 { unsafe { DISPATCH_META[page as usize & 0xFFFFF] } }

#[inline]
pub fn dispatch_meta_state_flags(meta: u64) -> u32 { (meta >> 32) as u32 }

#[inline]
pub fn dispatch_meta_table_index(meta: u64) -> u16 { (meta >> 16) as u16 }

#[inline]
pub fn dispatch_state_lookup(meta: u64, virt_address: u32) -> u16 {
    unsafe {
        let slab = (meta as u16) as usize;
        dbg_assert!(slab != 0 && slab < DISPATCH_SLAB_COUNT);
        DISPATCH_SLABS[slab * 0x1000 + (virt_address as usize & 0xFFF)]
    }
}

/// Publish (or refresh) a page's dispatch entries. Reuses the page's existing slab
/// when present. On pool exhaustion the page simply stays unpublished (meta 0) —
/// resolvers miss, the interpreter runs the code, correctness is unaffected; the
/// loud counter makes the condition visible in stats long before it can matter
/// (pool = 4095 pages-with-code, typical live set is a few hundred).
pub fn dispatch_meta_set(
    virt_page: Page,
    wasm_table_index: WasmTableIndex,
    entries: &Vec<(u16, u16)>,
    state_flags: CachedStateFlags,
) {
    unsafe {
        let page = virt_page.to_u32() as usize & 0xFFFFF;
        let existing = DISPATCH_META[page];
        let slab = if existing != 0 {
            (existing as u16) as usize
        }
        else {
            if DISPATCH_SLAB_FREE_TOP == 0 {
                DISPATCH_SLAB_OVERFLOWS = DISPATCH_SLAB_OVERFLOWS.saturating_add(1);
                dbg_log!("dispatch: slab pool exhausted, page {:x} unpublished", page);
                return;
            }
            DISPATCH_SLAB_FREE_TOP -= 1;
            let s = DISPATCH_SLAB_FREE[DISPATCH_SLAB_FREE_TOP] as usize;
            let in_use = (DISPATCH_SLAB_COUNT - 1 - DISPATCH_SLAB_FREE_TOP) as u32;
            if in_use > DISPATCH_SLAB_HIGH_WATER {
                DISPATCH_SLAB_HIGH_WATER = in_use;
            }
            s
        };
        dbg_assert!(slab != 0 && slab < DISPATCH_SLAB_COUNT);

        let table = &mut DISPATCH_SLABS[slab * 0x1000..slab * 0x1000 + 0x1000];
        table.fill(u16::MAX);
        for &(addr, state) in entries {
            dbg_assert!(state != u16::MAX);
            table[addr as usize] = state;
        }

        DISPATCH_META[page] = (state_flags.to_u32() as u64) << 32
            | (wasm_table_index.to_u16() as u64) << 16
            | slab as u64;
    }
}

#[inline]
pub fn dispatch_ext_get(page: u32) -> u64 { unsafe { DISPATCH_META_EXT[page as usize & 0xFFFFF] } }

// An external module that returned without retiring an instruction at this
// address: the next dispatch of exactly this address bypasses the external
// table once, so the interpreter can execute (or fault on) the instruction.
static mut EXT_STALL_EIP: u32 = 0;
static mut EXT_STALL_ARMED: bool = false;
static mut EXT_STALLS: u32 = 0;

pub fn ext_stall_note(eip: u32) {
    unsafe {
        EXT_STALL_EIP = eip;
        EXT_STALL_ARMED = true;
        EXT_STALLS = EXT_STALLS.wrapping_add(1);
    }
}

#[inline]
pub fn ext_stall_take(eip: u32) -> bool {
    unsafe {
        if EXT_STALL_ARMED && EXT_STALL_EIP == eip {
            EXT_STALL_ARMED = false;
            true
        }
        else {
            EXT_STALL_ARMED = false;
            false
        }
    }
}

#[no_mangle]
pub fn jit_external_stalls() -> u32 { unsafe { EXT_STALLS } }

// Addresses interpreted because the page's module has no entry for them:
// a direct-mapped histogram that ages out cold addresses, read sorted.
const MISS_ENTRY_SLOTS: usize = 4096;
static mut MISS_ENTRY_EIP: [u32; MISS_ENTRY_SLOTS] = [0; MISS_ENTRY_SLOTS];
static mut MISS_ENTRY_COUNT: [u32; MISS_ENTRY_SLOTS] = [0; MISS_ENTRY_SLOTS];

#[inline]
pub fn miss_entry_note(eip: u32) {
    unsafe {
        let i = ((eip >> 1) ^ (eip >> 13)) as usize & (MISS_ENTRY_SLOTS - 1);
        if MISS_ENTRY_EIP[i] == eip {
            MISS_ENTRY_COUNT[i] = MISS_ENTRY_COUNT[i].saturating_add(1);
        }
        else if MISS_ENTRY_COUNT[i] < 4 {
            MISS_ENTRY_EIP[i] = eip;
            MISS_ENTRY_COUNT[i] = 1;
        }
        else {
            MISS_ENTRY_COUNT[i] -= 1;
        }
    }
}

#[no_mangle]
pub fn jit_miss_entry_reset() {
    unsafe {
        MISS_ENTRY_EIP = [0; MISS_ENTRY_SLOTS];
        MISS_ENTRY_COUNT = [0; MISS_ENTRY_SLOTS];
    }
}

/// `rank`-th hottest missing entry (0 = hottest); field 0 = address, 1 = count.
#[no_mangle]
pub fn jit_miss_entry_top(rank: u32, field: u32) -> u32 {
    unsafe {
        let mut v: Vec<(u32, u32)> = (0..MISS_ENTRY_SLOTS)
            .filter(|&i| MISS_ENTRY_COUNT[i] != 0)
            .map(|i| (MISS_ENTRY_COUNT[i], MISS_ENTRY_EIP[i]))
            .collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        match v.get(rank as usize) {
            Some(&(count, eip)) => if field == 0 { eip } else { count },
            None => 0,
        }
    }
}

// External modules take precedence over the JIT's for an address both own.
// Measured on BFME 1 (2 September 2026): 12.6 FPS against 30 with the JIT
// first, the translation exiting at every call the batch does not cover.
// Off by default; the JIT serves what it has compiled and the external
// modules the rest.
static mut EXTERNAL_FIRST: bool = false;
#[inline]
pub fn external_first_enabled() -> bool { unsafe { EXTERNAL_FIRST } }
#[no_mangle]
pub fn jit_set_external_first(on: u32) { unsafe { EXTERNAL_FIRST = on != 0; } }
#[no_mangle]
pub fn jit_get_external_first() -> u32 { unsafe { EXTERNAL_FIRST as u32 } }

/// Called by an external module about to exit at an instruction it wants the
/// interpreter to run: the next dispatch of that address bypasses the
/// external table once.
#[no_mangle]
pub fn jit_ext_interpret_once(eip: u32) {
    unsafe {
        EXT_STALL_EIP = eip;
        EXT_STALL_ARMED = true;
    }
}

/// Publish an external module's entries for a virtual page (see DISPATCH_META_EXT).
pub fn dispatch_ext_set(
    virt_page: Page,
    wasm_table_index: WasmTableIndex,
    entries: &Vec<(u16, u16)>,
    state_flags: CachedStateFlags,
) {
    unsafe {
        let page = virt_page.to_u32() as usize & 0xFFFFF;
        let existing = DISPATCH_META_EXT[page];
        let slab = if existing != 0 {
            (existing as u16) as usize
        }
        else {
            if DISPATCH_SLAB_FREE_TOP == 0 {
                DISPATCH_SLAB_OVERFLOWS = DISPATCH_SLAB_OVERFLOWS.saturating_add(1);
                return;
            }
            DISPATCH_SLAB_FREE_TOP -= 1;
            let s = DISPATCH_SLAB_FREE[DISPATCH_SLAB_FREE_TOP] as usize;
            let in_use = (DISPATCH_SLAB_COUNT - 1 - DISPATCH_SLAB_FREE_TOP) as u32;
            if in_use > DISPATCH_SLAB_HIGH_WATER {
                DISPATCH_SLAB_HIGH_WATER = in_use;
            }
            s
        };
        dbg_assert!(slab != 0 && slab < DISPATCH_SLAB_COUNT);
        let table = &mut DISPATCH_SLABS[slab * 0x1000..slab * 0x1000 + 0x1000];
        table.fill(u16::MAX);
        for &(addr, state) in entries {
            table[addr as usize] = state;
        }
        DISPATCH_META_EXT[page] = (state_flags.to_u32() as u64) << 32
            | (wasm_table_index.to_u16() as u64) << 16
            | slab as u64;
    }
}

pub fn dispatch_ext_clear(page: u32) -> bool {
    unsafe {
        let page = page as usize & 0xFFFFF;
        let meta = DISPATCH_META_EXT[page];
        if meta == 0 {
            return false;
        }
        let slab = (meta as u16) as usize;
        dbg_assert!(slab != 0 && slab < DISPATCH_SLAB_COUNT);
        DISPATCH_SLAB_FREE[DISPATCH_SLAB_FREE_TOP] = slab as u16;
        DISPATCH_SLAB_FREE_TOP += 1;
        DISPATCH_META_EXT[page] = 0;
        true
    }
}

/// Unpublish a page. Returns true if the page actually had an entry (callers use
/// this to bump the B1b ret-memo epoch only on real evictions, as before).
pub fn dispatch_meta_clear(page: u32) -> bool {
    unsafe {
        let page = page as usize & 0xFFFFF;
        let meta = DISPATCH_META[page];
        if meta == 0 {
            return false;
        }
        let slab = (meta as u16) as usize;
        dbg_assert!(slab != 0 && slab < DISPATCH_SLAB_COUNT);
        dbg_assert!(DISPATCH_SLAB_FREE_TOP < DISPATCH_SLAB_COUNT);
        DISPATCH_SLAB_FREE[DISPATCH_SLAB_FREE_TOP] = slab as u16;
        DISPATCH_SLAB_FREE_TOP += 1;
        DISPATCH_META[page] = 0;
        true
    }
}

// Dispatch-slab occupancy counters exported for the TS stats verb.
#[no_mangle]
pub fn dispatch_slab_high_water() -> u32 { unsafe { DISPATCH_SLAB_HIGH_WATER } }
#[no_mangle]
pub fn dispatch_slab_overflows() -> u32 { unsafe { DISPATCH_SLAB_OVERFLOWS } }

#[no_mangle]
pub fn jit_exact_dispatch_inserts() -> u32 { unsafe { EXACT_DISPATCH_INSERTS } }
#[no_mangle]
pub fn jit_exact_dispatch_hits() -> u32 { unsafe { EXACT_DISPATCH_HITS } }
#[no_mangle]
pub fn jit_exact_dispatch_misses() -> u32 { unsafe { EXACT_DISPATCH_MISSES } }
#[no_mangle]
pub fn jit_exact_dispatch_overflows() -> u32 { unsafe { EXACT_DISPATCH_OVERFLOWS } }
#[no_mangle]
pub fn jit_chain_memo_high_water() -> u32 { unsafe { CHAIN_SITE_MEMO_HIGH_WATER } }
#[no_mangle]
pub fn jit_chain_memo_overflows() -> u32 { unsafe { CHAIN_SITE_MEMO_OVERFLOWS } }

#[no_mangle]
pub fn jit_inline_dispatch_sites_compiled() -> u32 {
    unsafe { INLINE_INTRA_MODULE_DISPATCH_SITES_COMPILED }
}

#[no_mangle]
pub fn jit_block_chain_sites_compiled() -> u32 { unsafe { BLOCK_CHAIN_SITES_COMPILED } }

// Coarse remap-thrash latch, using guest icount as the clock.
static mut FASTMEM_THRASH_WINDOW_START: u32 = 0;
static mut FASTMEM_THRASH_WINDOW_BUMPS: u32 = 0;
// ~50M guest instructions ≈ a fraction of a second at in-game rates.
const FASTMEM_THRASH_WINDOW_ICOUNT: u32 = 50_000_000;
// > ~4 remaps/frame sustained across the window.
const FASTMEM_THRASH_BUMP_LIMIT: u32 = 240;

// Compile-site counters; runtime hit/fill split is not instrumented.
static mut X87_LOCAL_CACHE_LOAD_SITES_COMPILED: u32 = 0;
static mut X87_LOCAL_CACHE_STORES_COMPILED: u32 = 0;
static mut X87_LOCAL_CACHE_INVALIDATES_COMPILED: u32 = 0;

static mut PUSH_RUN_SITES_COMPILED: u32 = 0;
static mut PUSH_RUN_REUSE_BRANCHES_COMPILED: u32 = 0;

// Mirrors emulator-config.ts; dbg.fastmemReads asserts equality.
pub const FASTMEM_LOW_MEM_END: u32 = 0x0010_0000;
pub const FASTMEM_GUARD_BASE: u32 = 0x2300_0000;
pub const FASTMEM_GUARD_SIZE: u32 = 0x0100_0000;

#[no_mangle]
pub fn fastmem_get_low_mem_end() -> u32 { FASTMEM_LOW_MEM_END }
#[no_mangle]
pub fn fastmem_get_guard_base() -> u32 { FASTMEM_GUARD_BASE }
#[no_mangle]
pub fn fastmem_get_guard_size() -> u32 { FASTMEM_GUARD_SIZE }

#[inline]
pub fn fastmem_current_generation() -> u64 {
    unsafe { *global_pointers::fastmem_generation }
}

#[inline]
pub fn fastmem_compile_generation(state_flags: CachedStateFlags) -> Option<u64> {
    unsafe {
        if !JIT_FASTMEM_READS
            || !state_flags.is_32()
            || !*global_pointers::protected_mode
            || (*global_pointers::cr & cpu::CR0_PG) == 0
            || FASTMEM_THRASH_LATCHED
        {
            None
        }
        else {
            Some(*global_pointers::fastmem_generation)
        }
    }
}

#[inline]
pub fn x87_locals_enabled() -> bool { unsafe { JIT_X87_LOCALS } }

#[inline]
pub fn push_run_coalescing_enabled() -> bool { unsafe { JIT_PUSH_RUN_COALESCING } }

pub fn fastmem_read_split_enabled() -> bool { unsafe { JIT_FASTMEM_READ_SPLIT } }

#[inline]
pub fn flag_locals_enabled() -> bool { unsafe { JIT_FLAG_LOCALS } }

// Compile-time gate for the store fast path. Same regime as fastmem reads (32-bit
// protected mode + paging): the map's identity-map store `mem8 + addr` is only valid
// where VA == PA, which BottleShip guarantees under paging. Any page NOT identity-RW
// simply never has bit0 set by TS, so even if the shape is emitted the fast path is
// never taken there — this gate only avoids emitting dead shape in other regimes.
pub fn fastmem_writes_compile_enabled(state_flags: CachedStateFlags) -> bool {
    unsafe {
        JIT_FASTMEM_WRITES
            && state_flags.is_32()
            && *global_pointers::protected_mode
            && (*global_pointers::cr & cpu::CR0_PG) != 0
    }
}

// Wasm-memory address of the write map, baked as a load base in the store fast path.
#[inline]
pub fn fastmem_write_map_base() -> u32 {
    unsafe { &FASTMEM_WRITE_MAP[0] as *const u8 as u32 }
}

#[inline]
pub fn fastmem_note_speculated_store_compiled() {
    unsafe {
        FASTMEM_SPECULATED_STORES_COMPILED = FASTMEM_SPECULATED_STORES_COMPILED.saturating_add(1);
    }
}

// ── Write-map maintenance ─────────────────────────────────────────────────────────
// TS owns bit0 only; rust owns bit1/bit2. The worker is single-threaded, so the split
// ownership is a plain (non-atomic) read-modify-write with no race.

/// bit0 (BASE_WRITABLE) over [start_page, start_page+page_count). Owner: TS choke
/// points (region register / commit / decommit / protect). Only bit0 is touched.
///
/// Setting bit0 is authoritatively clamped HERE to the SAME identity-RAM envelope the
/// read fast path trusts — [LOW_MEM_END, min(GUARD_BASE, ram)) ∪ [GUARD_END, ram), in
/// pages — so no TS caller can ever fast-enable a low-mem/MMIO page, the guard red zone,
/// or an unbacked (> ram) page, whatever range it passes. Clearing is unconditional
/// (slow path is always the safe direction).
#[no_mangle]
pub fn fastmem_write_map_set_base(start_page: u32, page_count: u32, writable: u32) {
    unsafe {
        let start = (start_page as usize).min(FASTMEM_WRITE_MAP_LEN);
        let end = (start_page as usize)
            .saturating_add(page_count as usize)
            .min(FASTMEM_WRITE_MAP_LEN);
        if writable == 0 {
            for p in start..end {
                FASTMEM_WRITE_MAP[p] &= !FASTMEM_WRITE_BASE_WRITABLE;
            }
            return;
        }
        let ram = *global_pointers::memory_size;
        let lo1 = (FASTMEM_LOW_MEM_END >> 12) as usize;
        let hi1 = (FASTMEM_GUARD_BASE.min(ram) >> 12) as usize;
        let lo2 = (FASTMEM_GUARD_BASE.wrapping_add(FASTMEM_GUARD_SIZE) >> 12) as usize;
        let hi2 = (ram >> 12) as usize;
        let excl_lo = FASTMEM_WRITE_EXCLUDE_LO_PAGE as usize;
        let excl_hi = FASTMEM_WRITE_EXCLUDE_HI_PAGE as usize;
        for p in start..end {
            let in_envelope = (p >= lo1 && p < hi1) || (p >= lo2 && p < hi2);
            let excluded = excl_hi != 0 && p >= excl_lo && p < excl_hi;
            if in_envelope && !excluded {
                FASTMEM_WRITE_MAP[p] |= FASTMEM_WRITE_BASE_WRITABLE;
                if (p as u32) > FASTMEM_WRITE_MAP_MAX_PAGE {
                    FASTMEM_WRITE_MAP_MAX_PAGE = p as u32;
                }
            }
        }
    }
}

/// TS points this at the THUNK_CODE band at boot so no PTE-level (kind-blind) SET can
/// ever fast-enable the immutable RX stubs. Also clears bit0 across the band defensively.
#[no_mangle]
pub fn fastmem_write_map_set_exclude(lo_page: u32, hi_page: u32) {
    unsafe {
        FASTMEM_WRITE_EXCLUDE_LO_PAGE = lo_page;
        FASTMEM_WRITE_EXCLUDE_HI_PAGE = hi_page;
        if hi_page > lo_page {
            let lo = (lo_page as usize).min(FASTMEM_WRITE_MAP_LEN);
            let hi = (hi_page as usize).min(FASTMEM_WRITE_MAP_LEN);
            for p in lo..hi {
                FASTMEM_WRITE_MAP[p] &= !FASTMEM_WRITE_BASE_WRITABLE;
            }
        }
    }
}

/// Wipe the whole map to zero (conservative). Called by TS at v86 (re)init before it
/// re-marks the RW regions, in case the wasm instance (and thus this static) persisted.
#[no_mangle]
pub fn fastmem_write_map_reset() {
    unsafe {
        core::ptr::write_bytes(&raw mut FASTMEM_WRITE_MAP as *mut u8, 0, FASTMEM_WRITE_MAP_LEN);
        FASTMEM_WRITE_MAP_MAX_PAGE = 0;
        FASTMEM_SPECULATED_STORES_COMPILED = 0;
    }
}

#[inline]
pub fn fastmem_write_map_set_code(page: u32) {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] |= FASTMEM_WRITE_HAS_CODE;
        }
    }
}

#[inline]
pub fn fastmem_write_map_clear_code(page: u32) {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] &= !FASTMEM_WRITE_HAS_CODE;
        }
    }
}

#[inline]
pub fn fastmem_write_map_set_watch(page: u32) {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] |= FASTMEM_WRITE_WATCH;
        }
    }
}

#[inline]
pub fn fastmem_write_map_clear_watch(page: u32) {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] &= !FASTMEM_WRITE_WATCH;
        }
    }
}

/// Raw map byte for one page (audit verb).
#[no_mangle]
pub fn fastmem_write_map_get(page: u32) -> u32 {
    unsafe {
        if (page as usize) < FASTMEM_WRITE_MAP_LEN {
            FASTMEM_WRITE_MAP[page as usize] as u32
        }
        else {
            0
        }
    }
}

/// Count pages within the populated prefix. mask == 0 → count acceptance (byte == 1);
/// otherwise count pages where (byte & mask) != 0. Bounded by the max marked page.
#[no_mangle]
pub fn fastmem_write_map_count(mask: u32) -> u32 {
    unsafe {
        let hi = (FASTMEM_WRITE_MAP_MAX_PAGE as usize + 1).min(FASTMEM_WRITE_MAP_LEN);
        let mut n = 0u32;
        for p in 0..hi {
            let b = FASTMEM_WRITE_MAP[p] as u32;
            let hit = if mask == 0 { b == 1 } else { (b & mask) != 0 };
            if hit {
                n = n.saturating_add(1);
            }
        }
        n
    }
}

#[no_mangle]
pub fn fastmem_get_speculated_stores_compiled() -> u32 {
    unsafe { FASTMEM_SPECULATED_STORES_COMPILED }
}

/// Highest VA page ever marked base-writable — upper bound for the audit scan.
#[no_mangle]
pub fn fastmem_write_map_max_page() -> u32 {
    unsafe { FASTMEM_WRITE_MAP_MAX_PAGE }
}

#[inline]
pub fn x87_locals_note_cache_load_site_compiled() {
    unsafe {
        X87_LOCAL_CACHE_LOAD_SITES_COMPILED =
            X87_LOCAL_CACHE_LOAD_SITES_COMPILED.saturating_add(1);
    }
}

#[inline]
pub fn x87_locals_note_cache_store_compiled() {
    unsafe {
        X87_LOCAL_CACHE_STORES_COMPILED =
            X87_LOCAL_CACHE_STORES_COMPILED.saturating_add(1);
    }
}

#[inline]
pub fn x87_locals_note_cache_invalidate_compiled() {
    unsafe {
        X87_LOCAL_CACHE_INVALIDATES_COMPILED =
            X87_LOCAL_CACHE_INVALIDATES_COMPILED.saturating_add(1);
    }
}

#[inline]
pub fn push_run_note_site_compiled() {
    unsafe {
        PUSH_RUN_SITES_COMPILED = PUSH_RUN_SITES_COMPILED.saturating_add(1);
    }
}

#[inline]
pub fn push_run_note_reuse_branch_compiled() {
    unsafe {
        PUSH_RUN_REUSE_BRANCHES_COMPILED =
            PUSH_RUN_REUSE_BRANCHES_COMPILED.saturating_add(1);
    }
}

#[inline]
pub fn fastmem_note_speculated_load_compiled() {
    unsafe {
        FASTMEM_SPECULATED_LOADS_COMPILED =
            FASTMEM_SPECULATED_LOADS_COMPILED.saturating_add(1);
    }
}

#[no_mangle]
pub fn fastmem_bump_generation(source: u32) {
    unsafe {
        // 0 is the non-fastmem Code sentinel.
        *global_pointers::fastmem_generation =
            (*global_pointers::fastmem_generation).wrapping_add(1);
        let idx = (source as usize).min(FASTMEM_BUMP_SOURCE_COUNT - 1);
        FASTMEM_BUMPS_BY_SOURCE[idx] = FASTMEM_BUMPS_BY_SOURCE[idx].saturating_add(1);

        // Thrash auto-latch: only relevant while speculation is live.
        if JIT_FASTMEM_READS && !FASTMEM_THRASH_LATCHED {
            FASTMEM_THRASH_WINDOW_BUMPS = FASTMEM_THRASH_WINDOW_BUMPS.saturating_add(1);
            let icount = *global_pointers::instruction_counter;
            let elapsed = icount.wrapping_sub(FASTMEM_THRASH_WINDOW_START);
            if elapsed >= FASTMEM_THRASH_WINDOW_ICOUNT {
                if FASTMEM_THRASH_WINDOW_BUMPS >= FASTMEM_THRASH_BUMP_LIMIT {
                    FASTMEM_THRASH_LATCHED = true;
                    let bumps = FASTMEM_THRASH_WINDOW_BUMPS;
                    dbg_log!("fastmem: thrash-latched off, {} bumps/window", bumps);
                }
                FASTMEM_THRASH_WINDOW_START = icount;
                FASTMEM_THRASH_WINDOW_BUMPS = 0;
            }
        }
    }
}

#[no_mangle]
pub fn fastmem_get_generation() -> u32 { fastmem_current_generation() as u32 }

#[no_mangle]
pub fn fastmem_get_bump_count(source: u32) -> u32 {
    unsafe {
        let idx = (source as usize).min(FASTMEM_BUMP_SOURCE_COUNT - 1);
        FASTMEM_BUMPS_BY_SOURCE[idx]
    }
}

#[no_mangle]
pub fn fastmem_get_speculated_loads_compiled() -> u32 {
    unsafe { FASTMEM_SPECULATED_LOADS_COMPILED }
}

#[no_mangle]
pub fn fastmem_get_deopt_recompiles() -> u32 { unsafe { FASTMEM_DEOPT_RECOMPILES } }

#[no_mangle]
pub fn fastmem_get_thrash_latched() -> u32 { unsafe { FASTMEM_THRASH_LATCHED as u32 } }

#[no_mangle]
pub fn x87_locals_get_cache_load_sites_compiled() -> u32 {
    unsafe { X87_LOCAL_CACHE_LOAD_SITES_COMPILED }
}

#[no_mangle]
pub fn x87_locals_get_cache_stores_compiled() -> u32 { unsafe { X87_LOCAL_CACHE_STORES_COMPILED } }

#[no_mangle]
pub fn x87_locals_get_cache_invalidates_compiled() -> u32 {
    unsafe { X87_LOCAL_CACHE_INVALIDATES_COMPILED }
}

#[no_mangle]
pub fn push_run_get_sites_compiled() -> u32 { unsafe { PUSH_RUN_SITES_COMPILED } }

#[no_mangle]
pub fn push_run_get_reuse_branches_compiled() -> u32 {
    unsafe { PUSH_RUN_REUSE_BRANCHES_COMPILED }
}

#[no_mangle]
pub fn set_dispatch_stats(enabled: u32) { unsafe { DISPATCH_STATS = enabled != 0; } }

#[no_mangle]
pub fn get_dispatch_stats() -> u32 { unsafe { DISPATCH_STATS as u32 } }

// Tier-1 hotness threshold. Runtime-tunable for cold-start A/Bs: a lower value
// trades more WebAssembly compilation/code memory for less interpreter time.
// Keep the stock 200k default until a multi-workload benchmark justifies moving it.
static mut JIT_THRESHOLD: u32 = 200 * 1000;

/// Divisor applied to JIT_THRESHOLD for a page that ALREADY owns a compiled
/// module. Such interpretation is a coverage gap, not a cold page: the module
/// exists and lacks this entry point. 1 disables (identical to the historical
/// single threshold).
///
/// Deliberately not the same knob as JIT_THRESHOLD: lowering that globally was
/// measured 17-19% slower on a cold boot because it also compiles cold pages,
/// nearly doubling the module count for identical guest work.
static mut JIT_RECOMPILE_DIVISOR: u32 = 8;

/// Whether the native cycle loop re-checks the urgent-exit signal each iteration.
///
/// requestImmediateExit() zeroes the shared budget and the JIT's cached copy, but
/// do_many_cycles_native tests a local snapshot taken at slice entry — so the loop
/// keeps running. The zero only stops generated edges from chaining. A caller that
/// asked to end the slice therefore gets the opposite of what it wanted: the guest
/// runs on to the full 500,003-instruction budget with every budget-guarded
/// optimisation disabled.
static mut JIT_HONOR_URGENT_EXIT_IN_SLICE: u32 = 0;

/// Guard dynamic chaining on the park address rather than on a zeroed budget.
///
/// The budget check exists to preserve one invariant: never chain past an async
/// park, because the parked thread's return address was overwritten with the spin
/// loop and chaining into it would loop inside the module without ever reaching
/// the outer loop's park check. Using "the budget is zero" as a proxy is far
/// wider than the invariant — it disables chaining for the whole remainder of any
/// slice in which any thunk asked to exit. Testing the park address directly is
/// the exact condition, and it changes no scheduling: the slice already runs to
/// its local budget either way.
static mut JIT_CHAIN_PARK_GUARD: u32 = 0;

#[no_mangle]
pub fn jit_chain_park_guard() -> u32 { unsafe { JIT_CHAIN_PARK_GUARD } }

/// Address the generated chaining guards read for the slice budget. With the
/// park guard on this is the slice's own budget, which an urgent exit does not
/// zero, so a chaining edge is refused only when the budget is genuinely spent.
fn chain_budget_address() -> u32 {
    if unsafe { JIT_CHAIN_PARK_GUARD } != 0 {
        std::ptr::addr_of!(cpu::jit_slice_limit) as u32
    }
    else {
        std::ptr::addr_of!(cpu::jit_cycle_limit_cached) as u32
    }
}

#[no_mangle]
pub fn jit_honor_urgent_exit_in_slice() -> u32 { unsafe { JIT_HONOR_URGENT_EXIT_IN_SLICE } }

// less branches will generate if-else, more will generate brtable
pub const BRTABLE_CUTOFF: usize = 10;

// needs to be synced to const.js
//
// 900 does not hold a game's working set. Measured in a BFME 1 session on real
// hardware: 8,906 compilations against 900 slots, i.e. the same pages compiled,
// evicted and recompiled about ten times, at 4.1 compilations per rendered
// frame. Every eviction also costs the return-prediction cache (freeing a slot
// invalidates it globally), so the churn compounds. Sized to hold the working
// set instead; the static tables below grow linearly and stay under 300 KB.
pub const WASM_TABLE_SIZE: u32 = 8192;

/// Count of full JIT cache flushes caused by wasm-table exhaustion.
static mut JIT_CACHE_FLUSHES: u32 = 0;

/// CLOCK reference bit per table slot, set when a module is dispatched into.
/// Exhaustion otherwise discards every module AND its page's hotness, so the
/// whole working set has to re-cross JIT_THRESHOLD interpreted instructions
/// before any of it runs compiled again — for a working set the size of the
/// table that is two orders of magnitude more interpretation than the flush
/// itself costs.
static mut MODULE_RECENTLY_USED: [bool; WASM_TABLE_SIZE as usize] =
    [false; WASM_TABLE_SIZE as usize];

/// 0 = upstream behaviour (full flush on exhaustion), 1 = evict only modules
/// unused since the previous sweep.
static mut JIT_PARTIAL_EVICTION: u32 = 0;
static mut JIT_PARTIAL_EVICTIONS: u32 = 0;
static mut JIT_EVICTED_MODULES: u32 = 0;
static mut JIT_EVICTION_FALLBACKS: u32 = 0;

/// Share of the cache dropped when every module was referenced since the last
/// sweep: enough to make progress, small enough that the hot set survives.
const EVICTION_FALLBACK_DIVISOR: usize = 4;

#[inline]
pub fn jit_note_module_used(wasm_table_index: u16) {
    unsafe {
        if JIT_PARTIAL_EVICTION != 0 {
            let slot = wasm_table_index as usize;
            if slot < WASM_TABLE_SIZE as usize {
                MODULE_RECENTLY_USED[slot] = true;
            }
        }
    }
}

#[no_mangle]
pub fn jit_get_partial_evictions() -> u32 { unsafe { JIT_PARTIAL_EVICTIONS } }

#[no_mangle]
pub fn jit_get_evicted_modules() -> u32 { unsafe { JIT_EVICTED_MODULES } }

#[no_mangle]
pub fn jit_get_eviction_fallbacks() -> u32 { unsafe { JIT_EVICTION_FALLBACKS } }

/// Reclaim table slots on exhaustion, preferring modules that have not been
/// dispatched into since the previous sweep. Returns how many pages were
/// dropped. A workload that streams through code faster than it revisits it can
/// legitimately have every slot referenced; rather than give up and let the
/// caller discard everything, a bounded fraction is dropped so the hot majority
/// still survives.
fn jit_evict_unused(ctx: &mut JitState) -> usize {
    let mut victims = Vec::new();
    for (&page, info) in ctx.pages.iter() {
        let slot = info.wasm_table_index.to_u16() as usize;
        if slot < WASM_TABLE_SIZE as usize && unsafe { MODULE_RECENTLY_USED[slot] } {
            // Second chance: referenced since the last sweep, so clear the bit
            // and let the next sweep judge it on fresh evidence.
            unsafe { MODULE_RECENTLY_USED[slot] = false };
        }
        else {
            victims.push(page);
        }
    }

    if victims.is_empty() {
        // Everything was referenced. Every bit is now clear, so the next sweep
        // will discriminate properly; this one still has to free something.
        let quota = (ctx.pages.len() / EVICTION_FALLBACK_DIVISOR).max(1);
        victims.extend(ctx.pages.keys().take(quota).copied());
        unsafe { JIT_EVICTION_FALLBACKS = JIT_EVICTION_FALLBACKS.wrapping_add(1) };
    }

    for &page in &victims {
        jit_dirty_page_ctx(ctx, page);
    }
    unsafe {
        JIT_PARTIAL_EVICTIONS = JIT_PARTIAL_EVICTIONS.wrapping_add(1);
        JIT_EVICTED_MODULES = JIT_EVICTED_MODULES.wrapping_add(victims.len() as u32);
    }
    victims.len()
}

/// Pages whose compiled module was discarded because the page was written to.
/// A module that keeps being thrown away is recompiled from scratch, and every
/// instruction executed between the two is interpreted — so this separates
/// "never got hot" from "got hot repeatedly and kept losing its code" as the
/// reason a hot page has no module.
static mut JIT_PAGE_INVALIDATIONS_WITH_CODE: u32 = 0;
static mut JIT_PAGE_INVALIDATIONS_NO_CODE: u32 = 0;

#[no_mangle]
pub fn jit_get_page_invalidations_with_code() -> u32 { unsafe { JIT_PAGE_INVALIDATIONS_WITH_CODE } }

#[no_mangle]
pub fn jit_get_page_invalidations_no_code() -> u32 { unsafe { JIT_PAGE_INVALIDATIONS_NO_CODE } }

#[no_mangle]
pub fn jit_reset_page_invalidations() {
    unsafe {
        JIT_PAGE_INVALIDATIONS_WITH_CODE = 0;
        JIT_PAGE_INVALIDATIONS_NO_CODE = 0;
    }
}

#[no_mangle]
pub fn jit_get_cache_flushes() -> u32 { unsafe { JIT_CACHE_FLUSHES } }

#[no_mangle]
pub fn jit_reset_cache_flushes() { unsafe { JIT_CACHE_FLUSHES = 0 } }

#[no_mangle]
pub fn jit_get_wasm_table_size() -> u32 { WASM_TABLE_SIZE }

// Light the invariant checks up in debug builds — the corruption class
// (silent ExitProcess via #PF on garbage state) needs the free/publish
// discipline asserted loudly, not assumed.
pub const CHECK_JIT_STATE_INVARIANTS: bool = cfg!(debug_assertions);

const MAX_INSTRUCTION_LENGTH: u32 = 16;

static JIT_STATE: Mutex<MaybeUninit<JitState>> = Mutex::new(MaybeUninit::uninit());
fn get_jit_state() -> JitStateRef { JitStateRef(JIT_STATE.try_lock().unwrap()) }

struct JitStateRef(MutexGuard<'static, MaybeUninit<JitState>>);

impl Deref for JitStateRef {
    type Target = JitState;
    fn deref(&self) -> &Self::Target { unsafe { self.0.assume_init_ref() } }
}
impl DerefMut for JitStateRef {
    fn deref_mut(&mut self) -> &mut Self::Target { unsafe { self.0.assume_init_mut() } }
}

#[no_mangle]
pub fn rust_init() {
    dispatch_meta_init();

    unsafe {
        TIER2_PAGE_COUNT = 0;
        TIER2_PROMOTIONS = 0;
        TIER2_BLOCKED_BY_CAP = 0;
        TIER2_PROFILED_EXITS = 0;
        TIER2_REGION_PROMOTIONS = 0;
        TIER2_REGION_SEEDS = 0;
        TIER2_REGION_CANDIDATES = 0;
        TIER2_REGION_REJECTED_TARGET = 0;
        TIER2_REGION_REJECTED_BUDGET = 0;
        TIER2_EXIT_TARGETS = [[0; TIER2_PROFILE_TARGETS]; WASM_TABLE_SIZE as usize];
        TIER2_EXIT_COUNTS = [[0; TIER2_PROFILE_TARGETS]; WASM_TABLE_SIZE as usize];
        TIER2_PROFILE_SAMPLES = [0; WASM_TABLE_SIZE as usize];
        MODULE_EXEC_COUNTS = [0; 0x10000];
        TIER2_MAINTENANCE_NEXT = TIER2_MAINTENANCE_INTERVAL;
        TIER2_MAINTENANCE_DUE = false;
        TIER2_MAINTENANCE_TICK = 1;
        TIER2_MAINTENANCE_SAMPLES = 0;
        TIER2_PAGE_EVICTIONS = 0;
        JIT_COMPILE_STARTED = 0;
        JIT_COMPILE_COMPLETED = 0;
        JIT_COMPILE_CAP_SKIPS = 0;
        JIT_COMPILE_PENDING_HIGH_WATER = 0;
        JIT_COMPILE_TOTAL_US = 0;
        JIT_COMPILE_MAX_US = 0;
        JIT_COMPILE_DEFERRED_QUEUED = 0;
        JIT_COMPILE_DEFERRED_STARTED = 0;
        JIT_COMPILE_DEFERRED_DROPPED = 0;
    }

    let _ = JIT_STATE
        .try_lock()
        .unwrap()
        .write(JitState::create_and_initialise());

    crate::d3d9_glue::init();

    use std::panic;

    panic::set_hook(Box::new(|panic_info| {
        console_log!("{}", panic_info.to_string());
    }));
}

struct PageInfo {
    wasm_table_index: WasmTableIndex,
    hidden_wasm_table_indices: Vec<WasmTableIndex>,
    entry_points: Vec<(u16, u16)>,
    state_flags: CachedStateFlags,
}

#[derive(Clone)]
struct Tier2Region {
    pages: HashSet<Page>,
    seeds: Vec<i32>,
}

enum CompilingPageState {
    Compiling { pages: HashMap<Page, PageInfo> },
    CompilingWritten,
}

#[derive(Copy, Clone)]
struct DeferredCompile {
    page: Page,
    virt_address: i32,
    phys_address: u32,
    cs_offset: u32,
    state_flags: CachedStateFlags,
}

const JIT_DEFERRED_COMPILE_QUEUE_CAP: usize = 1024;

static mut JIT_MAX_PENDING_COMPILES: u32 = 1;
static mut JIT_COMPILE_STARTED: u32 = 0;
static mut JIT_COMPILE_COMPLETED: u32 = 0;
static mut JIT_COMPILE_CAP_SKIPS: u32 = 0;
static mut JIT_COMPILE_PENDING_HIGH_WATER: u32 = 0;
static mut JIT_COMPILE_TOTAL_US: u64 = 0;
static mut JIT_COMPILE_MAX_US: u32 = 0;
static mut JIT_COMPILE_DEFERRED_QUEUED: u32 = 0;
static mut JIT_COMPILE_DEFERRED_STARTED: u32 = 0;
static mut JIT_COMPILE_DEFERRED_DROPPED: u32 = 0;
// Worker-thread CPU spent in analysis + wasm emission, measured around
// jit_analyze_and_generate. JIT_COMPILE_TOTAL_US is the browser's asynchronous
// compile latency; this is the synchronous cost the guest actually waits for.
static mut JIT_CODEGEN_TOTAL_US: f64 = 0.0;
static mut JIT_CODEGEN_MAX_US: f64 = 0.0;
static mut JIT_CODEGEN_COUNT: u32 = 0;
static mut JIT_CODEGEN_BYTES_TOTAL: u64 = 0;

// ── Hot-page profile ─────────────────────────────────────────────────────
// Pages a previous session compiled, with the entry points their modules had.
// Such a page is compiled at its first touch instead of after JIT_THRESHOLD
// interpreted instructions: for a cold path made of code that runs once per
// session, that interpretation is the dominant cost. A hash of the page's
// bytes guards against a different binary or a rewritten page.
#[derive(Clone)]
struct HotProfilePage {
    hash: u32,
    entries: Vec<u16>,
}
static HOT_PROFILE: Mutex<Option<HashMap<Page, HotProfilePage>>> = Mutex::new(None);
static HOT_PROFILE_IO: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static mut JIT_HOT_PROFILE_FORCED: u32 = 0;
static mut JIT_HOT_PROFILE_MISMATCH: u32 = 0;
// An external (ahead-of-time) page module replaced by a JIT compile of the
// same page: the batch left a dispatched entry uncovered on that page.
static mut JIT_EXTERNAL_PAGES_REPLACED: u32 = 0;
// Config 48. 0: force whenever the page is touched, queueing behind the compile
// cap. 1: force only while a compile slot is free, so a burst of known pages
// (a boot touches a thousand of them) cannot pile up a deferred queue whose
// latency exceeds the interpreted ramp it was meant to skip; the rest keep the
// ordinary ramp and compile normally if they stay hot.
static mut JIT_HOT_PROFILE_MODE: u32 = 1;
const HOT_PROFILE_MAGIC: u32 = 0x5054_4F48; // "HOTP"
const HOT_PROFILE_VERSION: u32 = 1;

/// Interpreted-execution state of one physical page that has no module yet.
struct PageHotness {
    hotness: u32,
    entry_points: HashSet<u16>,
}

struct JitState {
    wasm_builder: WasmBuilder,

    // as an alternative to HashSet, we could use a bitmap of 4096 bits here
    // (faster, but uses much more memory)
    // or a compressed bitmap (likely faster)
    // or HashSet<u32> rather than nested
    entry_points: HashMap<Page, PageHotness>,
    pages: HashMap<Page, PageInfo>,
    wasm_table_index_free_list: Vec<WasmTableIndex>,
    // WebAssembly compilation is asynchronous. Keep a small bounded set in
    // flight so one cold module does not force every other hot page to remain
    // interpreted until its Promise settles.
    compiling: HashMap<WasmTableIndex, CompilingPageState>,
    deferred_compiles: VecDeque<DeferredCompile>,
    deferred_compile_pages: HashSet<Page>,
    // B3 hotness tiering: pages promoted to tier-2 (jit_tier2_note_execution) — modules
    // whose entries land on these pages compile with the expanded tier-2 budgets.
    // Survives jit_clear_cache (the pages are still the hot ones); dies with the wasm
    // instance (per game load).
    tier2_pages: HashSet<Page>,
    // Sparse recency metadata used only after the bounded Tier-2 set fills.
    // Existing compiled modules remain valid when a page loses its marking;
    // the mark controls the budget of its next compilation, not code lifetime.
    tier2_page_last_seen: HashMap<Page, u32>,
    // Profile-selected unions of already-hot Tier-1 modules. Keyed by every
    // source page so whichever source entry triggers recompilation sees the plan.
    tier2_regions: HashMap<Page, Tier2Region>,
    // External (ahead-of-time) modules by physical page, alongside — not
    // instead of — whatever the JIT compiles for the same page.
    external_pages: HashMap<Page, PageInfo>,
}

fn check_jit_state_invariants(ctx: &mut JitState) {
    if !CHECK_JIT_STATE_INVARIANTS {
        return;
    }

    for state in ctx.compiling.values() {
        if let CompilingPageState::Compiling { pages } = state {
            dbg_assert!(pages.keys().all(|page| ctx.entry_points.contains_key(page)));
        }
    }

    let free: HashSet<WasmTableIndex> =
        HashSet::from_iter(ctx.wasm_table_index_free_list.iter().cloned());
    let used = HashSet::from_iter(ctx.pages.values().map(|info| info.wasm_table_index));
    let compiling = HashSet::from_iter(ctx.compiling.keys().copied());
    dbg_assert!(free.intersection(&used).next().is_none());
    dbg_assert!(used.intersection(&compiling).next().is_none());
    dbg_assert!(free.len() + used.len() + compiling.len() == (WASM_TABLE_SIZE - 1) as usize);

    for state in ctx.compiling.values() {
        if let CompilingPageState::Compiling { pages } = state {
            dbg_assert!(pages.keys().all(|page| ctx.entry_points.contains_key(page)));
        }
    }

    for i in 0..unsafe { cpu::valid_tlb_entries_count } {
        let page = unsafe { cpu::valid_tlb_entries[i as usize] };
        let entry = unsafe { cpu::tlb_data[page as usize] };
        if 0 != entry {
            let tlb_physical_page = Page::of_u32(
                (entry as u32 >> 12 ^ page as u32) - (unsafe { memory::mem8 } as u32 >> 12),
            );
            let meta = dispatch_meta_get(page as u32);
            let w = if meta != 0 {
                Some(WasmTableIndex(dispatch_meta_table_index(meta)))
            }
            else {
                None
            };
            let tlb_has_code = entry & cpu::TLB_HAS_CODE == cpu::TLB_HAS_CODE;
            let infos = ctx.pages.get(&tlb_physical_page);
            let entry_points = ctx.entry_points.get(&tlb_physical_page);
            dbg_assert!(tlb_has_code || !w.is_some());
            dbg_assert!(tlb_has_code || !infos.is_some());
            dbg_assert!(tlb_has_code || !entry_points.is_some());
            //dbg_assert!((w.is_some() || page.is_some() || entry_points.is_some()) == tlb_has_code); // XXX: check this
        }
    }
}

impl JitState {
    pub fn create_and_initialise() -> JitState {
        // don't assign 0 (XXX: Check)
        // The top EXTERNAL_MODULE_SLOTS indices are never handed to the JIT: they
        // hold modules produced outside it (ahead-of-time translations) that
        // JS places in the table and registers with jit_register_external_module.
        let wasm_table_indices =
            (1..=(WASM_TABLE_SIZE - 1 - EXTERNAL_MODULE_SLOTS) as u16).map(|x| WasmTableIndex(x));

        JitState {
            wasm_builder: WasmBuilder::new(),

            entry_points: HashMap::new(),
            pages: HashMap::new(),

            wasm_table_index_free_list: Vec::from_iter(wasm_table_indices),
            compiling: HashMap::new(),
            deferred_compiles: VecDeque::new(),
            deferred_compile_pages: HashSet::new(),
            tier2_pages: HashSet::new(),
            tier2_page_last_seen: HashMap::new(),
            tier2_regions: HashMap::new(),
            external_pages: HashMap::new(),
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum BasicBlockType {
    Normal {
        next_block_addr: Option<u32>,
        jump_offset: i32,
        jump_offset_is_32: bool,
    },
    ConditionalJump {
        next_block_addr: Option<u32>,
        next_block_branch_taken_addr: Option<u32>,
        condition: u8,
        jump_offset: i32,
        jump_offset_is_32: bool,
    },
    // Set eip to an absolute value (ret, jmp r/m, call r/m)
    AbsoluteEip,
    Exit,
}

pub struct BasicBlock {
    pub addr: u32,
    pub virt_addr: i32,
    pub last_instruction_addr: u32,
    pub end_addr: u32,
    pub is_entry_block: bool,
    pub ty: BasicBlockType,
    pub has_sti: bool,
    pub number_of_instructions: u32,
    /// Physical fallthrough after a non-control-flow block boundary. The
    /// instruction helper may change EIP or request preemption at runtime, so
    /// this is only a candidate for a guarded continuation, never an assumed
    /// CFG edge.
    pub sync_boundary_fallthrough: Option<u32>,
    /// RET-target speculation (superblock lite): for an AbsoluteEip
    /// block that is a genuine RET of a small leaf function called from within this
    /// module, the (virt, phys) return addresses of its module-local call sites. The
    /// emitter turns each into `if eip == virt { target_block = <dispatcher idx>;
    /// br main_loop }` ahead of the jit_find_cache_entry_in_page helper call, so a
    /// leaf's return stays intra-module without the dispatch helper. Filled by the
    /// post-pass in jit_find_basic_blocks when JIT_RET_SPECULATION is on; empty
    /// otherwise. The compare guards correctness — a stale/wrong candidate simply
    /// falls through to the existing dispatch.
    pub ret_speculation: Vec<(i32, u32)>,
    /// Tiny direct-call callee duplicated after this CALL in Tier-2. The caller's
    /// CFG remains unchanged; emission intercepts its terminal edge, runs this
    /// leaf, then guards the popped EIP before re-entering at the continuation.
    pub inline_leaf: Option<u32>,
}

#[derive(Copy, Clone, PartialEq)]
pub struct CachedCode {
    pub wasm_table_index: WasmTableIndex,
    pub initial_state: u16,
}

impl CachedCode {
    pub const NONE: CachedCode = CachedCode {
        wasm_table_index: WasmTableIndex(0),
        initial_state: 0,
    };
}

#[derive(PartialEq)]
pub enum InstructionOperandDest {
    WasmLocal(WasmLocal),
    Other,
}
#[derive(PartialEq)]
pub enum InstructionOperand {
    WasmLocal(WasmLocal),
    Immediate(i32),
    Other,
}
impl InstructionOperand {
    pub fn is_zero(&self) -> bool {
        match self {
            InstructionOperand::Immediate(0) => true,
            _ => false,
        }
    }
}
impl Into<InstructionOperand> for InstructionOperandDest {
    fn into(self: InstructionOperandDest) -> InstructionOperand {
        match self {
            InstructionOperandDest::WasmLocal(l) => InstructionOperand::WasmLocal(l),
            InstructionOperandDest::Other => InstructionOperand::Other,
        }
    }
}
pub enum Instruction {
    Cmp {
        dest: InstructionOperandDest,
        source: InstructionOperand,
        opsize: i32,
    },
    Sub {
        dest: InstructionOperandDest,
        source: InstructionOperand,
        opsize: i32,
        is_dec: bool,
    },
    Add {
        dest: InstructionOperandDest,
        source: InstructionOperand,
        opsize: i32,
        is_inc: bool,
    },
    AdcSbb {
        dest: InstructionOperandDest,
        #[allow(dead_code)]
        source: InstructionOperand,
        opsize: i32,
    },
    NonZeroShift {
        dest: InstructionOperandDest,
        opsize: i32,
    },
    Bitwise {
        dest: InstructionOperandDest,
        opsize: i32,
    },
    Other,
}

pub struct JitContext<'a> {
    pub cpu: &'a mut CpuContext,
    pub builder: &'a mut WasmBuilder,
    pub register_locals: &'a mut Vec<WasmLocal>,
    pub start_of_current_instruction: u32,
    pub exit_with_fault_label: Label,
    pub exit_label: Label,
    pub current_instruction: Instruction,
    pub previous_instruction: Instruction,
    pub fpu_simd_dirty_marked: bool,
    pub elide_current_flags: bool,
    pub instruction_counter: WasmLocal,
    pub fastmem_generation: Option<u64>,
    /// Emit the per-page-map store fast path for this unit.
    pub fastmem_writes: bool,
    pub x87_local_cache: [Option<X87LocalCacheSlot>; 8],
    pub push32_write_cache: Option<Push32WriteCache>,
    /// Compile-time handshake used only while duplicating a fused C3 leaf. The
    /// RET emitter records the popped architectural EIP here instead of storing
    /// it globally; the caller then guards it and restores the global on miss.
    pub capture_inline_leaf_return_eip: bool,
    pub inline_leaf_return_eip: Option<WasmLocal>,
    /// Set true by any x87 relaxed wrapper that leaves the block-scoped st-local
    /// cache coherent (it either updated the touched slot or invalidated all
    /// slots). Reset to false before each instruction; if an x87 opcode
    /// (D8–DF) is compiled through a raw-helper path that did NOT set this, the
    /// emission loop invalidates the cache so a later relaxed op re-reads memory
    /// and re-checks the tag. Without this, helper-dispatched FPU ops (FISTP m64
    /// aka _ftol, FSQRT/FSIN/FCOS, FINCSTP/FDECSTP, FLD m80, FIADD-family, …)
    /// silently mutate the FPU stack / shift TOP behind stale cached values.
    pub x87_cache_kept: bool,
}

pub struct X87LocalCacheSlot {
    pub bits: WasmLocalI64,
    pub valid: WasmLocal,
    /// Allocated only for blocks compiled with deferred writeback enabled.
    /// Keeping it optional removes all dirty-writeback bookkeeping from blocks
    /// compiled while config 39 is off.
    pub dirty: Option<WasmLocal>,
}

pub struct Push32WriteCache {
    pub page: WasmLocal,
    pub entry: WasmLocal,
    pub valid: WasmLocal,
}

pub fn rep_movs_reduced_spill_enabled() -> bool {
    unsafe { JIT_REP_MOVS_REDUCED_SPILL }
}

impl<'a> JitContext<'a> {
    pub fn reg(&self, i: u32) -> WasmLocal {
        match self.register_locals.get(i as usize) {
            Some(x) => x.unsafe_clone(),
            None => {
                dbg_assert!(false);
                unsafe { std::hint::unreachable_unchecked() }
            },
        }
    }
}

pub const JIT_INSTR_BLOCK_BOUNDARY_FLAG: u32 = 1 << 0;

pub fn is_near_end_of_page(address: u32) -> bool {
    address & 0xFFF >= 0x1000 - MAX_INSTRUCTION_LENGTH
}

// Classification of one x86 instruction for the dead-flag liveness walk. Safe by construction:
// the walk only ELIDES when it can prove flags dead, so anything not provably an `Overwrite` or a
// provably-clean `NeutralNoFault` is `Stop` — we never need to enumerate flag *readers* precisely.
#[derive(Copy, Clone, PartialEq)]
enum FlagClass {
    // Fully overwrites every lazy-tracked flag (CF/PF/AF/ZF/SF/OF) before reading any.
    // non_faulting = the instruction cannot fault before the overwrite (register-only form).
    Overwrite { non_faulting: bool },
    // Provably touches NO flags AND cannot fault (register-only mov/lea/movzx/movsx/nop) — safe to
    // skip over while walking forward to the next flag-overwriter.
    NeutralNoFault,
    // Touches no flags but has a memory operand, so it can fault. Distinct from
    // Stop because the only thing it costs is the fault frame's flags, not the
    // liveness answer — and `mov` is 32% of this binary, so folding it into Stop
    // is what holds the elision rate at 11%.
    NeutralMayFault,
    // Everything else: reads a flag (Jcc/ADC/SBB/SETcc/CMOVcc/PUSHF/LAHF/…), modifies flags
    // partially (INC/DEC/shift/rotate/SAHF/POPF), or is control-flow/unrecognized.
    // Conservatively stops the walk WITHOUT eliding.
    Stop,
}

fn read_jit_u8(addr: u32) -> u8 { memory::read8(addr) as u8 }

fn skip_instruction_prefixes(mut addr: u32) -> u32 {
    loop {
        match read_jit_u8(addr) {
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {
                addr += 1;
            },
            _ => return addr,
        }
    }
}

fn decode_jit_opcode(addr: u32) -> (u32, u32) {
    let opcode_addr = skip_instruction_prefixes(addr);
    let opcode = read_jit_u8(opcode_addr);
    if opcode == 0x0F {
        (0x100 | read_jit_u8(opcode_addr + 1) as u32, opcode_addr + 2)
    }
    else {
        (opcode as u32, opcode_addr + 1)
    }
}

fn group_alu_is_full_overwrite(group: u8) -> bool {
    matches!(group, 0 | 1 | 4 | 5 | 6 | 7)
}

fn classify_flag_class(addr: u32) -> FlagClass {
    let (opcode, operand_addr) = decode_jit_opcode(addr);
    // Read the byte after the opcode; only meaningful for opcodes that actually have a ModRM.
    // For no-ModRM opcodes (imm/reg-encoded/NOP) the branches below don't consult it.
    let modrm = read_jit_u8(operand_addr);
    let reg_only = modrm & 0xC0 == 0xC0;

    match opcode {
        // --- Full flag overwriters: ADD/OR/AND/SUB/XOR/CMP r/m<->reg, TEST r/m,reg ---
        // Register-only forms can't fault before overwriting flags; memory forms can.
        0x00..=0x03 | 0x08..=0x0B | 0x20..=0x23 | 0x28..=0x2B | 0x30..=0x33
        | 0x38..=0x3B | 0x84 | 0x85 => FlagClass::Overwrite { non_faulting: reg_only },

        // Accumulator-immediate ALU/TEST (no ModRM, no memory) — always non-faulting.
        0x04 | 0x05 | 0x0C | 0x0D | 0x24 | 0x25 | 0x2C | 0x2D | 0x34 | 0x35
        | 0x3C | 0x3D | 0xA8 | 0xA9 => FlagClass::Overwrite { non_faulting: true },

        // Group 1 ALU immediates: /0 ADD /1 OR /4 AND /5 SUB /6 XOR /7 CMP overwrite;
        // /2 ADC /3 SBB read CF → Stop.
        0x80 | 0x81 | 0x82 | 0x83 => {
            if group_alu_is_full_overwrite((modrm >> 3) & 7) {
                FlagClass::Overwrite { non_faulting: reg_only }
            }
            else {
                FlagClass::Stop
            }
        },

        // Group 3 /0 TEST imm overwrites; /2 NOT doesn't touch flags but other /n
        // (NEG/MUL/IMUL/DIV/IDIV) set them in special ways → Stop for everything but /0.
        0xF6 | 0xF7 => {
            if (modrm >> 3) & 7 == 0 {
                FlagClass::Overwrite { non_faulting: reg_only }
            }
            else {
                FlagClass::Stop
            }
        },

        // --- Flag-neutral, non-faulting (register-only forms only; memory forms can #PF) ---
        0x88 | 0x89 | 0x8A | 0x8B => if reg_only { FlagClass::NeutralNoFault } else { FlagClass::NeutralMayFault }, // MOV r/m<->reg
        0x8D => FlagClass::NeutralNoFault,                                                               // LEA (no deref, no flags)
        0xB0..=0xBF => FlagClass::NeutralNoFault,                                                        // MOV reg, imm
        0xC6 | 0xC7 => if reg_only { FlagClass::NeutralNoFault } else { FlagClass::NeutralMayFault },               // MOV r/m, imm
        0x90 => FlagClass::NeutralNoFault,                                                               // NOP
        0x1B6 | 0x1B7 | 0x1BE | 0x1BF => if reg_only { FlagClass::NeutralNoFault } else { FlagClass::NeutralMayFault }, // MOVZX/MOVSX

        _ => FlagClass::Stop,
    }
}

fn instruction_end(cpu: &CpuContext, addr: u32) -> u32 {
    let mut step_cpu = cpu.clone();
    step_cpu.eip = addr;
    analysis::analyze_step(&mut step_cpu);
    step_cpu.eip
}

// How far to look ahead for the next flag-overwriter before giving up (bounded so compile time
// stays predictable; pointer-chase prologues rarely have long flag-neutral runs).
const FLAG_LIVENESS_WALK_LIMIT: u32 = 8;

// Continues the flag-liveness walk from `addr_in` (inside `block`, which ends at
// `block.end_addr`) onward. `origin_addr` is the address of the original flag-overwriting
// instruction we're trying to elide — fixed for the whole (possibly cross-block) walk, used only
// as a wraparound sanity guard, same as in the original single-block version.
//
// When the walk runs off the end of `block` still undecided, it does NOT stop there: the
// compiled module already knows this block's successor edge(s) exactly (BasicBlockType's
// next_block_addr / next_block_branch_taken_addr are resolved at compile time from the real CFG,
// not profiled/speculative), so we keep walking into the sole Normal successor. Only Normal
// reaches this point in practice: the block-discovery loop's only way to end a block WITHOUT a
// real control-flow instruction is the artificial merge-split for Normal (jit.rs, discovery loop
// — `basic_blocks.contains_key(&current_address)` cuts the block at a plain non-branching
// instruction because another path already made that address a block entry). ConditionalJump has
// no equivalent: it is only ever created from an actually-decoded Jcc (AnalysisType::Jump
// {condition: Some(_)}), so `block.last_instruction_addr` is always that Jcc — and since Jcc
// necessarily reads flags, classify_flag_class's Stop for it fires during the walk above, before
// `addr` can reach `block.end_addr`. So the ConditionalJump arm below is unreachable by
// construction, not just untested — see its comment.
fn flags_dead_from_addr(
    cpu: &CpuContext,
    origin_addr: u32,
    addr_in: u32,
    block: &BasicBlock,
    basic_blocks: &HashMap<u32, BasicBlock>,
    steps: &mut u32,
) -> bool {
    let mut addr = addr_in;
    while addr > origin_addr && addr < block.end_addr && *steps < FLAG_LIVENESS_WALK_LIMIT {
        match classify_flag_class(addr) {
            // A faulting overwriter could #PF before overwriting, and the fault frame would need the
            // (now elided) architectural flags — so only a non-faulting overwriter proves dead.
            FlagClass::Overwrite { non_faulting: true } => {
                profiler::stat_increment_always(stat::DEAD_FLAG_ELIDED);
                return true;
            },
            FlagClass::Overwrite { non_faulting: false } => {
                if !dead_flag_elision_across_faults() {
                    return false;
                }
                profiler::stat_increment_always(stat::DEAD_FLAG_ELIDED);
                return true;
            },
            FlagClass::NeutralNoFault => {
                addr = instruction_end(cpu, addr);
                *steps += 1;
            },
            FlagClass::NeutralMayFault => {
                // Walking past it only risks the flags a fault frame would carry.
                if !dead_flag_elision_across_faults() {
                    return false;
                }
                addr = instruction_end(cpu, addr);
                *steps += 1;
            },
            FlagClass::Stop => return false,
        }
    }

    if addr != block.end_addr || *steps >= FLAG_LIVENESS_WALK_LIMIT {
        // Either resolved to false already (Stop / faulting overwriter) or ran out of budget.
        return false;
    }

    match &block.ty {
        BasicBlockType::Normal { next_block_addr: Some(next), .. } => match basic_blocks.get(next) {
            Some(next_block) => {
                flags_dead_from_addr(cpu, origin_addr, *next, next_block, basic_blocks, steps)
            },
            // Successor isn't part of this compiled module (e.g. not yet discovered) — can't
            // prove anything about it, so don't elide.
            None => false,
        },
        BasicBlockType::ConditionalJump { .. } => {
            // Unreachable by construction (see the walk's doc comment above): a
            // ConditionalJump block always ends on a real Jcc, and Jcc always reads flags, so
            // the while-loop above always resolves via Stop before addr can reach
            // block.end_addr. Canary, not a load-bearing check — debug_assert! is compiled out
            // in release, zero cost. If block-discovery ever changes so a ConditionalJump block
            // can end elsewhere, this fires instead of silently mis-proving flags dead across a
            // branch we never actually evaluated.
            dbg_assert!(
                false,
                "flags_dead_from_addr: ConditionalJump reached block.end_addr — should be \
                 unreachable, its terminating Jcc always resolves via Stop first"
            );
            false
        },
        // AbsoluteEip / Exit (no statically known successor) or a partially-unresolved
        // Normal edge (e.g. target crosses into an unmapped page): stop here, same as the
        // original single-block behavior.
        _ => false,
    }
}

fn should_elide_current_flags(
    cpu: &CpuContext,
    current_addr: u32,
    block: &BasicBlock,
    basic_blocks: &HashMap<u32, BasicBlock>,
) -> bool {
    if !dead_flag_elision_enabled() {
        return false;
    }
    // The current instruction must itself fully overwrite the flags — only then is there a flag
    // computation to skip (the elision-aware emitters fire on these opcodes).
    if !matches!(classify_flag_class(current_addr), FlagClass::Overwrite { .. }) {
        return false;
    }
    profiler::stat_increment_always(stat::DEAD_FLAG_ELISION_CANDIDATE);

    // Walk forward, skipping flag-neutral non-faulting instructions, until the flags are proven
    // dead (a non-faulting full overwriter reached first) or possibly-live (a reader / partial /
    // faulting / control-flow instruction, a dead end at module scope, or the step limit).
    let addr = instruction_end(cpu, current_addr);
    let mut steps = 0;
    flags_dead_from_addr(cpu, current_addr, addr, block, basic_blocks, &mut steps)
}

pub fn jit_find_cache_entry(phys_address: u32, state_flags: CachedStateFlags) -> CachedCode {
    // TODO: dedup with jit_find_cache_entry_in_page?
    // NOTE: This is currently only used for invariant/missed-entry-point checking
    let ctx = get_jit_state();

    match ctx.pages.get(&Page::page_of(phys_address)) {
        Some(PageInfo {
            wasm_table_index,
            state_flags: s,
            entry_points,
            hidden_wasm_table_indices: _,
        }) => {
            if *s == state_flags {
                let page_offset = phys_address as u16 & 0xFFF;
                if let Some(&(_, initial_state)) =
                    entry_points.iter().find(|(p, _)| p == &page_offset)
                {
                    return CachedCode {
                        wasm_table_index: *wasm_table_index,
                        initial_state,
                    };
                }
            }
        },
        None => {},
    }

    return CachedCode::NONE;
}

#[no_mangle]
pub fn jit_find_cache_entry_in_page(
    virt_address: u32,
    wasm_table_index: WasmTableIndex,
    state_flags: u32,
) -> i32 {
    // TODO: generate code for this
    profiler::stat_increment(stat::INDIRECT_JUMP);
    if dispatch_stats_enabled() {
        profiler::stat_increment_always(stat::ABSEIP_DISPATCH);
    }

    let state_flags = CachedStateFlags::of_u32(state_flags);

    // DOD SoA lookup (no pointer chase; stale-generation units self-deopt on entry).
    let meta = dispatch_meta_get(virt_address >> 12);
    if meta != 0
        && dispatch_meta_state_flags(meta) == state_flags.to_u32()
        && dispatch_meta_table_index(meta) == wasm_table_index.to_u16()
    {
        let unit_state = dispatch_state_lookup(meta, virt_address);
        if unit_state != u16::MAX {
            return unit_state.into();
        }
    }

    profiler::stat_increment(stat::INDIRECT_JUMP_NO_ENTRY);

    // Block-chaining: an indirect jmp/call (AbsoluteEip) whose target is not in this
    // module → real exit to main_loop. eip was computed at runtime, so not statically chainable.
    if dispatch_stats_enabled() {
        profiler::stat_increment_always(stat::MODULE_EXIT_INDIRECT);
    }

    return -1;
}

/// RET/AbsoluteEip dynamic chaining: budget-guarded tlb_code lookup at the runtime eip,
/// returning the packed (table_slot << 16 | unit_state) target convention, with its own
/// RET_CHAIN_HIT/RET_CHAIN_MISS stats.
static mut DYNAMIC_CHAIN_BUDGET_ZERO: u64 = 0;
static mut DYNAMIC_CHAIN_BUDGET_SPENT: u64 = 0;
static mut DYNAMIC_CHAIN_BUDGET_HLT: u64 = 0;

#[no_mangle]
pub fn jit_dynamic_chain_budget_zero() -> u64 { unsafe { DYNAMIC_CHAIN_BUDGET_ZERO } }
#[no_mangle]
pub fn jit_dynamic_chain_budget_spent() -> u64 { unsafe { DYNAMIC_CHAIN_BUDGET_SPENT } }
#[no_mangle]
pub fn jit_dynamic_chain_budget_hlt() -> u64 { unsafe { DYNAMIC_CHAIN_BUDGET_HLT } }

#[no_mangle]
pub fn jit_dynamic_chain_budget_reset() {
    unsafe {
        DYNAMIC_CHAIN_BUDGET_ZERO = 0;
        DYNAMIC_CHAIN_BUDGET_SPENT = 0;
        DYNAMIC_CHAIN_BUDGET_HLT = 0;
    }
}

#[no_mangle]
pub unsafe fn jit_find_cache_entry_for_dynamic_chaining(state_flags: u32) -> i32 {
    // same quantum as do_many_cycles_native (limit==0 urgent exit and in_hlt still bail) —
    // this is what keeps the async-park/spin-loop invariant: an urgent
    // exit request zeroes the budget, so we never chain past it.
    let park_guard = JIT_CHAIN_PARK_GUARD != 0;
    // The park signal, not the urgent-exit signal: a parked thread's return
    // address is the spin loop, so refusing that one target is the whole
    // invariant. Everything else stays bounded by the slice's real budget.
    let limit = if park_guard {
        cpu::jit_slice_limit
    }
    else {
        hypercall::read_cycle_limit()
    };
    let elapsed = (*global_pointers::instruction_counter)
        .wrapping_sub(cpu::jit_cycle_start_instruction_counter);
    let parked = park_guard
        && hypercall::eip_at_park(*global_pointers::instruction_pointer as u32);

    if limit == 0 || elapsed >= limit || *global_pointers::in_hlt || parked {
        if dispatch_stats_enabled() {
            profiler::stat_increment_always(stat::RET_CHAIN_MISS);
            DYNAMIC_CHAIN_RESOLVE_BUDGET_MISSES =
                DYNAMIC_CHAIN_RESOLVE_BUDGET_MISSES.saturating_add(1);
            // The three conditions have nothing in common as causes: an urgent
            // exit requested by the host, a slice that genuinely ran its course,
            // and a halted CPU. Only the first two are worth acting on, and they
            // call for opposite fixes.
            if limit == 0 {
                DYNAMIC_CHAIN_BUDGET_ZERO = DYNAMIC_CHAIN_BUDGET_ZERO.saturating_add(1);
            }
            else if elapsed >= limit {
                DYNAMIC_CHAIN_BUDGET_SPENT = DYNAMIC_CHAIN_BUDGET_SPENT.saturating_add(1);
            }
            else {
                DYNAMIC_CHAIN_BUDGET_HLT = DYNAMIC_CHAIN_BUDGET_HLT.saturating_add(1);
            }
        }
        return -1;
    }

    let virt_address = *global_pointers::instruction_pointer as u32;

    // B1b: direct-mapped memo probe. An entry is valid only if its epoch is current —
    // any table-slot free or code-TLB eviction since the fill bumps the epoch and
    // invalidates everything (see RET_CACHE).
    let cache_idx = (virt_address >> 2) as usize & (RET_CACHE_SIZE - 1);
    let cached = RET_CACHE[cache_idx];
    if cached.0 == virt_address
        && cached.1 == state_flags
        && cached.2 >= 0
        && cached.3 == RET_CACHE_EPOCH
    {
        if dispatch_stats_enabled() {
            profiler::stat_increment_always(stat::RET_CHAIN_HIT);
            DYNAMIC_CHAIN_RESOLVE_MEMO_HITS =
                DYNAMIC_CHAIN_RESOLVE_MEMO_HITS.saturating_add(1);
        }
        return cached.2;
    }

    let raw_state_flags = state_flags;
    let state_flags = CachedStateFlags::of_u32(state_flags);

    // DOD SoA lookup (no pointer chase). Generation staleness is handled by the
    // unit's own prologue guard (self-deopt on entry), and a deopt/free bumps
    // RET_CACHE_EPOCH via free_wasm_table_index — so unlike the old Box walk, the
    // memo may now cache EVERY unit: there is no lookup-time generation check left
    // for it to skip.
    let meta = dispatch_meta_get(virt_address >> 12);
    if meta != 0 && dispatch_meta_state_flags(meta) == state_flags.to_u32() {
        let unit_state = dispatch_state_lookup(meta, virt_address);
        if unit_state != u16::MAX {
            if dispatch_stats_enabled() {
                profiler::stat_increment_always(stat::RET_CHAIN_HIT);
                DYNAMIC_CHAIN_RESOLVE_META_HITS =
                    DYNAMIC_CHAIN_RESOLVE_META_HITS.saturating_add(1);
            }

            let table_slot =
                dispatch_meta_table_index(meta) as i32 + cpu::WASM_TABLE_OFFSET as i32;
            let packed = table_slot << 16 | unit_state as i32;
            RET_CACHE[cache_idx] = (virt_address, raw_state_flags, packed, RET_CACHE_EPOCH);
            return packed;
        }
    }

    if dispatch_stats_enabled() {
        profiler::stat_increment_always(stat::RET_CHAIN_MISS);
        if meta == 0 {
            DYNAMIC_CHAIN_RESOLVE_NO_META_MISSES =
                DYNAMIC_CHAIN_RESOLVE_NO_META_MISSES.saturating_add(1);
        }
        else if dispatch_meta_state_flags(meta) != state_flags.to_u32() {
            DYNAMIC_CHAIN_RESOLVE_STATE_MISSES =
                DYNAMIC_CHAIN_RESOLVE_STATE_MISSES.saturating_add(1);
        }
        else {
            DYNAMIC_CHAIN_RESOLVE_NO_ENTRY_MISSES =
                DYNAMIC_CHAIN_RESOLVE_NO_ENTRY_MISSES.saturating_add(1);
        }
    }
    -1
}

/// Cold/miss path for one generated dynamic-chain site. The normal resolver is
/// still the sole authority; a successful result is merely copied into the
/// site's generated-code cache with the current invalidation epoch.
#[no_mangle]
pub unsafe fn jit_find_cache_entry_for_dynamic_chaining_site(
    state_flags: u32,
    memo_slot: u32,
) -> i32 {
    let slot = memo_slot as usize;
    let diag = JIT_DYNAMIC_CHAIN_SITE_PIC_DIAG && slot < DYNAMIC_CHAIN_SITE_PIC_COUNT;
    let track_second = (diag
        || JIT_DYNAMIC_CHAIN_SITE_PIC_SECOND_WAY
        || JIT_DYNAMIC_CHAIN_SITE_PIC_FOUR_WAY)
        && slot < DYNAMIC_CHAIN_SITE_PIC_COUNT;
    let virt_address = *global_pointers::instruction_pointer as u32;
    let epoch = RET_CACHE_EPOCH as u32;
    let mut previous_target = 0;
    let mut previous_epoch = 0;
    if track_second {
        previous_target = DYNAMIC_CHAIN_SITE_TARGETS[slot];
        previous_epoch = DYNAMIC_CHAIN_SITE_EPOCHS[slot];
    }
    if diag {
        DYNAMIC_CHAIN_SITE_PIC_DIAG_CALLS = DYNAMIC_CHAIN_SITE_PIC_DIAG_CALLS.saturating_add(1);
        if previous_epoch != epoch {
            DYNAMIC_CHAIN_SITE_PIC_DIAG_EPOCH_MISSES =
                DYNAMIC_CHAIN_SITE_PIC_DIAG_EPOCH_MISSES.saturating_add(1);
        }
        else if previous_target != virt_address {
            DYNAMIC_CHAIN_SITE_PIC_DIAG_TARGET_MISSES =
                DYNAMIC_CHAIN_SITE_PIC_DIAG_TARGET_MISSES.saturating_add(1);
            if DYNAMIC_CHAIN_SITE_DIAG_SECOND_EPOCHS[slot] == epoch
                && DYNAMIC_CHAIN_SITE_DIAG_SECOND_TARGETS[slot] == virt_address
            {
                DYNAMIC_CHAIN_SITE_PIC_DIAG_SECOND_WAY_HITS =
                    DYNAMIC_CHAIN_SITE_PIC_DIAG_SECOND_WAY_HITS.saturating_add(1);
            }
            else if DYNAMIC_CHAIN_SITE_DIAG_THIRD_EPOCHS[slot] == epoch
                && DYNAMIC_CHAIN_SITE_DIAG_THIRD_TARGETS[slot] == virt_address
            {
                DYNAMIC_CHAIN_SITE_PIC_DIAG_THIRD_WAY_HITS =
                    DYNAMIC_CHAIN_SITE_PIC_DIAG_THIRD_WAY_HITS.saturating_add(1);
            }
            else if DYNAMIC_CHAIN_SITE_DIAG_FOURTH_EPOCHS[slot] == epoch
                && DYNAMIC_CHAIN_SITE_DIAG_FOURTH_TARGETS[slot] == virt_address
            {
                DYNAMIC_CHAIN_SITE_PIC_DIAG_FOURTH_WAY_HITS =
                    DYNAMIC_CHAIN_SITE_PIC_DIAG_FOURTH_WAY_HITS.saturating_add(1);
            }
        }
        else {
            // The target and epoch matched, so generated code reached the helper
            // only because the scheduler budget or HLT guard rejected chaining.
            DYNAMIC_CHAIN_SITE_PIC_DIAG_GUARD_MISSES =
                DYNAMIC_CHAIN_SITE_PIC_DIAG_GUARD_MISSES.saturating_add(1);
        }
    }

    // Keep the generated primary hit path byte-for-byte unchanged. Only
    // after that path has already missed do we consult the optional second
    // target here. Recheck the scheduler boundary because a matching primary
    // can also enter this helper when its budget guard rejects chaining.
    if JIT_DYNAMIC_CHAIN_SITE_PIC_SECOND_WAY
        && slot < DYNAMIC_CHAIN_SITE_PIC_COUNT
        && previous_epoch == epoch
        && previous_target != virt_address
        && DYNAMIC_CHAIN_SITE_DIAG_SECOND_EPOCHS[slot] == epoch
        && DYNAMIC_CHAIN_SITE_DIAG_SECOND_TARGETS[slot] == virt_address
    {
        let limit = hypercall::read_cycle_limit();
        let elapsed = (*global_pointers::instruction_counter)
            .wrapping_sub(cpu::jit_cycle_start_instruction_counter);
        if limit != 0 && elapsed < limit && !*global_pointers::in_hlt {
            if dispatch_stats_enabled() {
                profiler::stat_increment_always(stat::RET_CHAIN_HIT);
            }
            return DYNAMIC_CHAIN_SITE_DIAG_SECOND_PACKED[slot];
        }
    }
    if JIT_DYNAMIC_CHAIN_SITE_PIC_FOUR_WAY
        && slot < DYNAMIC_CHAIN_SITE_PIC_COUNT
        && previous_epoch == epoch
        && previous_target != virt_address
    {
        let packed = if DYNAMIC_CHAIN_SITE_DIAG_THIRD_EPOCHS[slot] == epoch
            && DYNAMIC_CHAIN_SITE_DIAG_THIRD_TARGETS[slot] == virt_address
        {
            DYNAMIC_CHAIN_SITE_DIAG_THIRD_PACKED[slot]
        }
        else if DYNAMIC_CHAIN_SITE_DIAG_FOURTH_EPOCHS[slot] == epoch
            && DYNAMIC_CHAIN_SITE_DIAG_FOURTH_TARGETS[slot] == virt_address
        {
            DYNAMIC_CHAIN_SITE_DIAG_FOURTH_PACKED[slot]
        }
        else {
            -1
        };
        if packed >= 0 {
            let limit = hypercall::read_cycle_limit();
            let elapsed = (*global_pointers::instruction_counter)
                .wrapping_sub(cpu::jit_cycle_start_instruction_counter);
            if limit != 0 && elapsed < limit && !*global_pointers::in_hlt {
                if dispatch_stats_enabled() {
                    profiler::stat_increment_always(stat::RET_CHAIN_HIT);
                }
                return packed;
            }
        }
    }

    let packed = jit_find_cache_entry_for_dynamic_chaining(state_flags);
    if packed >= 0 && slot < DYNAMIC_CHAIN_SITE_PIC_COUNT {
        if diag {
            DYNAMIC_CHAIN_SITE_PIC_DIAG_RESOLVER_HITS =
                DYNAMIC_CHAIN_SITE_PIC_DIAG_RESOLVER_HITS.saturating_add(1);
        }
        if track_second && previous_epoch == epoch && previous_target != virt_address {
            DYNAMIC_CHAIN_SITE_DIAG_FOURTH_TARGETS[slot] =
                DYNAMIC_CHAIN_SITE_DIAG_THIRD_TARGETS[slot];
            DYNAMIC_CHAIN_SITE_DIAG_FOURTH_PACKED[slot] =
                DYNAMIC_CHAIN_SITE_DIAG_THIRD_PACKED[slot];
            DYNAMIC_CHAIN_SITE_DIAG_FOURTH_EPOCHS[slot] =
                DYNAMIC_CHAIN_SITE_DIAG_THIRD_EPOCHS[slot];
            DYNAMIC_CHAIN_SITE_DIAG_THIRD_TARGETS[slot] =
                DYNAMIC_CHAIN_SITE_DIAG_SECOND_TARGETS[slot];
            DYNAMIC_CHAIN_SITE_DIAG_THIRD_PACKED[slot] =
                DYNAMIC_CHAIN_SITE_DIAG_SECOND_PACKED[slot];
            DYNAMIC_CHAIN_SITE_DIAG_THIRD_EPOCHS[slot] =
                DYNAMIC_CHAIN_SITE_DIAG_SECOND_EPOCHS[slot];
            DYNAMIC_CHAIN_SITE_DIAG_SECOND_TARGETS[slot] = previous_target;
            DYNAMIC_CHAIN_SITE_DIAG_SECOND_PACKED[slot] = DYNAMIC_CHAIN_SITE_PACKED[slot];
            DYNAMIC_CHAIN_SITE_DIAG_SECOND_EPOCHS[slot] = epoch;
        }
        DYNAMIC_CHAIN_SITE_TARGETS[slot] = virt_address;
        DYNAMIC_CHAIN_SITE_PACKED[slot] = packed;
        DYNAMIC_CHAIN_SITE_EPOCHS[slot] = epoch;
    }
    packed
}

#[no_mangle]
pub unsafe fn jit_dynamic_chain_site_pic_diag_reset() {
    DYNAMIC_CHAIN_SITE_PIC_DIAG_CALLS = 0;
    DYNAMIC_CHAIN_SITE_PIC_DIAG_TARGET_MISSES = 0;
    DYNAMIC_CHAIN_SITE_PIC_DIAG_SECOND_WAY_HITS = 0;
    DYNAMIC_CHAIN_SITE_PIC_DIAG_THIRD_WAY_HITS = 0;
    DYNAMIC_CHAIN_SITE_PIC_DIAG_FOURTH_WAY_HITS = 0;
    DYNAMIC_CHAIN_SITE_PIC_DIAG_EPOCH_MISSES = 0;
    DYNAMIC_CHAIN_SITE_PIC_DIAG_GUARD_MISSES = 0;
    DYNAMIC_CHAIN_SITE_PIC_DIAG_RESOLVER_HITS = 0;
    std::ptr::write_bytes(
        std::ptr::addr_of_mut!(DYNAMIC_CHAIN_SITE_DIAG_SECOND_EPOCHS) as *mut u32,
        0,
        DYNAMIC_CHAIN_SITE_PIC_COUNT,
    );
    std::ptr::write_bytes(
        std::ptr::addr_of_mut!(DYNAMIC_CHAIN_SITE_DIAG_THIRD_EPOCHS) as *mut u32,
        0,
        DYNAMIC_CHAIN_SITE_PIC_COUNT,
    );
    std::ptr::write_bytes(
        std::ptr::addr_of_mut!(DYNAMIC_CHAIN_SITE_DIAG_FOURTH_EPOCHS) as *mut u32,
        0,
        DYNAMIC_CHAIN_SITE_PIC_COUNT,
    );
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_diag_calls() -> u64 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_DIAG_CALLS }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_diag_target_misses() -> u64 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_DIAG_TARGET_MISSES }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_diag_second_way_hits() -> u64 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_DIAG_SECOND_WAY_HITS }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_diag_third_way_hits() -> u64 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_DIAG_THIRD_WAY_HITS }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_diag_fourth_way_hits() -> u64 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_DIAG_FOURTH_WAY_HITS }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_diag_epoch_misses() -> u64 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_DIAG_EPOCH_MISSES }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_diag_guard_misses() -> u64 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_DIAG_GUARD_MISSES }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_diag_resolver_hits() -> u64 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_DIAG_RESOLVER_HITS }
}

fn allocate_dynamic_chain_site_pic() -> Option<u32> {
    unsafe {
        if DYNAMIC_CHAIN_SITE_PIC_NEXT >= DYNAMIC_CHAIN_SITE_PIC_COUNT {
            DYNAMIC_CHAIN_SITE_PIC_OVERFLOWS =
                DYNAMIC_CHAIN_SITE_PIC_OVERFLOWS.saturating_add(1);
            return None;
        }
        let slot = DYNAMIC_CHAIN_SITE_PIC_NEXT;
        DYNAMIC_CHAIN_SITE_PIC_NEXT += 1;
        DYNAMIC_CHAIN_SITE_TARGETS[slot] = 0;
        DYNAMIC_CHAIN_SITE_PACKED[slot] = 0;
        DYNAMIC_CHAIN_SITE_EPOCHS[slot] = 0;
        DYNAMIC_CHAIN_SITE_DIAG_SECOND_TARGETS[slot] = 0;
        DYNAMIC_CHAIN_SITE_DIAG_SECOND_PACKED[slot] = 0;
        DYNAMIC_CHAIN_SITE_DIAG_SECOND_EPOCHS[slot] = 0;
        DYNAMIC_CHAIN_SITE_DIAG_THIRD_TARGETS[slot] = 0;
        DYNAMIC_CHAIN_SITE_DIAG_THIRD_PACKED[slot] = 0;
        DYNAMIC_CHAIN_SITE_DIAG_THIRD_EPOCHS[slot] = 0;
        DYNAMIC_CHAIN_SITE_DIAG_FOURTH_TARGETS[slot] = 0;
        DYNAMIC_CHAIN_SITE_DIAG_FOURTH_PACKED[slot] = 0;
        DYNAMIC_CHAIN_SITE_DIAG_FOURTH_EPOCHS[slot] = 0;
        DYNAMIC_CHAIN_SITE_PIC_HIGH_WATER =
            DYNAMIC_CHAIN_SITE_PIC_HIGH_WATER.max(DYNAMIC_CHAIN_SITE_PIC_NEXT as u32);
        Some(slot as u32)
    }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_compiled() -> u32 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_COMPILED }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_high_water() -> u32 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_HIGH_WATER }
}

#[no_mangle]
pub fn jit_dynamic_chain_site_pic_overflows() -> u32 {
    unsafe { DYNAMIC_CHAIN_SITE_PIC_OVERFLOWS }
}

fn jit_find_basic_blocks(
    ctx: &mut JitState,
    entry_points: HashSet<i32>,
    cpu: CpuContext,
    tier2_region: Option<&Tier2Region>,
) -> Vec<BasicBlock> {
    /// A target in the page tail becomes a block only if its first
    /// instruction either fits in the page or crosses into a physically
    /// contiguous mapped page, the same proof the decode loop applies; the
    /// emitter has no fallback for an edge to a block that was never built.
    fn tail_target_decodable(
        virt_target: i32,
        phys_target: u32,
        template: &CpuContext,
        pages: &HashSet<Page>,
        max_pages: u32,
    ) -> bool {
        if !is_near_end_of_page(phys_target) {
            return true;
        }
        let mut probe = CpuContext { eip: phys_target, ..template.clone() };
        let analysis = analysis::analyze_step(&mut probe);
        let end = probe.eip;
        if Page::page_of(end) == Page::page_of(phys_target) {
            return true;
        }
        if !matches!(analysis.ty, AnalysisType::Normal) {
            return false;
        }
        let virt_after = (virt_target as u32).wrapping_add(end.wrapping_sub(phys_target));
        let next_page = Page::page_of(end);
        (pages.contains(&next_page) || (pages.len() as u32) < max_pages)
            && cpu::translate_address_read_no_side_effects(virt_after as i32) == Ok(end)
    }

    fn follow_jump(
        virt_target: i32,
        ctx: &mut JitState,
        pages: &mut HashSet<Page>,
        page_blacklist: &mut HashSet<Page>,
        max_pages: u32,
        tier2_region: Option<&Tier2Region>,
        marked_as_entry: &mut HashSet<i32>,
        to_visit_stack: &mut Vec<i32>,
        template: &CpuContext,
    ) -> Option<u32> {
        if is_near_end_of_page(virt_target as u32) && !page_tail_entries_enabled() {
            return None;
        }
        let phys_target = match cpu::translate_address_read_no_side_effects(virt_target) {
            Err(()) => {
                dbg_log!("Not analysing {:x} (page not mapped)", virt_target);
                return None;
            },
            Ok(t) => t,
        };
        if !tail_target_decodable(virt_target, phys_target, template, pages, max_pages) {
            profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
            return None;
        }

        let phys_page = Page::page_of(phys_target);

        // A profile-guided Tier-2 compile is a union of known-hot Tier-1 modules,
        // not an unconstrained wider BFS. Edges leaving that selected union stay
        // ordinary side exits, which bounds code size and avoids pulling cold call
        // trees into the generated wasm function.
        if tier2_region.map_or(false, |region| !region.pages.contains(&phys_page)) {
            return None;
        }

        // Never GROW a module INTO the thunk/callback/spin bucket (REGION_EXCLUDE_*):
        // stub pages are full of OUT traps and must stay standalone modules. This must
        // gate EVERY growth edge, not just profiled indirect targets — direct IAT-style
        // CALLs into stubs reached CALLBACK_STUB pages once the tier-2 page budget grew
        // past the default (fatal 0x3003).
        // `!pages.is_empty()` is load-bearing: the INITIAL entry points are seeded
        // through follow_jump with an empty page set — gating those blocks stub-page
        // modules from compiling at all (their dispatch then hits `unreachable` at the
        // stub EIP on the first execution, deterministic at boot).
        if !pages.is_empty()
            && region_target_excluded(virt_target as u32)
            && !pages.contains(&phys_page)
        {
            return None;
        }

        // `>=` (not `==`): a proper ceiling. Equivalent to the original for the
        // single-cap default path (growth is monotonic), but required so the cap
        // holds when region formation seeds the page set above the base cap.
        if !pages.contains(&phys_page) && pages.len() as u32 >= max_pages
            || page_blacklist.contains(&phys_page)
        {
            return None;
        }

        if !pages.contains(&phys_page) {
            // page seen for the first time, handle entry points
            if let Some(PageHotness { hotness, entry_points, .. }) =
                ctx.entry_points.get_mut(&phys_page)
            {
                let existing_entry_points = match ctx.pages.get(&phys_page) {
                    Some(PageInfo { entry_points, .. }) => {
                        HashSet::from_iter(entry_points.iter().map(|x| x.0))
                    },
                    None => HashSet::new(),
                };

                if entry_points
                    .iter()
                    .all(|entry_point| existing_entry_points.contains(entry_point))
                    && tier2_region.is_none()
                {
                    page_blacklist.insert(phys_page);
                    return None;
                }

                // XXX: Remove this paragraph
                //let old_length = entry_points.len();
                //entry_points.extend(existing_entry_points);
                //dbg_assert!(
                //    entry_points.union(&existing_entry_points).count() == entry_points.len()
                //);

                *hotness = 0;

                for &addr_low in entry_points.iter() {
                    let addr = virt_target & !0xFFF | addr_low as i32;
                    to_visit_stack.push(addr);
                    marked_as_entry.insert(addr);
                }
            }
            else {
                // no entry points: ignore this page?
                page_blacklist.insert(phys_page);
                return None;
            }

            pages.insert(phys_page);
            dbg_assert!(pages.len() as u32 <= max_pages);
        }

        to_visit_stack.push(virt_target);
        Some(phys_target)
    }

    let mut to_visit_stack: Vec<i32> = Vec::new();
    let mut marked_as_entry: HashSet<i32> = HashSet::new();
    let mut basic_blocks: BTreeMap<u32, BasicBlock> = BTreeMap::new();
    let mut pages: HashSet<Page> = HashSet::new();
    let mut page_blacklist = HashSet::new();

    // B3 hotness tiering: a compilation whose entry lands on a tier-2-promoted page
    // gets the expanded budgets (more pages per module + deeper RET-speculation).
    let tier2 = ctx.tier2_pages.len() > 0
        && entry_points.iter().any(|&virt| {
            match cpu::translate_address_read_no_side_effects(virt) {
                Ok(phys) => ctx.tier2_pages.contains(&Page::page_of(phys)),
                Err(()) => false,
            }
        });

    // 16-bit doesn't work correctly, most likely due to instruction pointer wrap-around
    // When indirect regions are on, use the (larger) region page budget as the
    // compilation-wide cap so dispatchers can absorb hot targets. Non-dispatcher
    // modules rarely reach it (hot direct-jump chains are short); it stays far
    // below the global-MAX_PAGES=48 setting that OOM'd V8.
    let max_pages = if let Some(region) = tier2_region {
        region.pages.len() as u32
    } else if cpu.state_flags.is_32() {
        let base = if unsafe { JIT_INDIRECT_REGIONS } {
            unsafe { MAX_PAGES.max(JIT_INDIRECT_REGION_MAX_PAGES) }
        } else {
            unsafe { MAX_PAGES }
        };
        if tier2 { base.max(unsafe { TIER2_MAX_PAGES }) } else { base }
    } else {
        1
    };

    let cpu_template = cpu.clone();
    for virt_addr in entry_points {
        let ok = follow_jump(
            virt_addr,
            ctx,
            &mut pages,
            &mut page_blacklist,
            max_pages,
            tier2_region,
            &mut marked_as_entry,
            &mut to_visit_stack,
            &cpu_template,
        );
        dbg_assert!(ok.is_some());
        dbg_assert!(marked_as_entry.contains(&virt_addr));
    }

    while let Some(to_visit) = to_visit_stack.pop() {
        let phys_addr = match cpu::translate_address_read_no_side_effects(to_visit) {
            Err(()) => {
                dbg_log!("Not analysing {:x} (page not mapped)", to_visit);
                continue;
            },
            Ok(phys_addr) => phys_addr,
        };

        if basic_blocks.contains_key(&phys_addr) {
            continue;
        }

        if is_near_end_of_page(phys_addr) && !page_tail_entries_enabled() {
            // Empty basic block, don't insert
            profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
            continue;
        }

        let mut current_address = phys_addr;
        let mut current_block = BasicBlock {
            addr: current_address,
            virt_addr: to_visit,
            last_instruction_addr: 0,
            end_addr: 0,
            ty: BasicBlockType::Exit,
            is_entry_block: false,
            has_sti: false,
            number_of_instructions: 0,
            sync_boundary_fallthrough: None,
            ret_speculation: Vec::new(),
            inline_leaf: None,
        };
        loop {
            let addr_before_instruction = current_address;
            let mut cpu = &mut CpuContext {
                eip: current_address,
                ..cpu
            };
            let analysis = analysis::analyze_step(&mut cpu);
            let has_next_instruction = !analysis.no_next_instruction;
            current_address = cpu.eip;

            // The decoder reads physically adjacent bytes. That is correct for
            // an instruction crossing a guest-page boundary only when the next
            // virtual page maps to that exact physical page. Prove the mapping
            // before retaining the instruction; otherwise preserve the legacy
            // interpreter fallback. Crossing control transfers stay excluded
            // until their segment and relative-EIP semantics are proven too.
            let crossed_page =
                Page::page_of(current_address) != Page::page_of(addr_before_instruction);
            let mut crossed_page_virt_target = None;
            if crossed_page {
                let virt_instruction = (to_visit as u32 & !0xFFF)
                    | (addr_before_instruction & 0xFFF);
                let virt_after = virt_instruction.wrapping_add(
                    current_address.wrapping_sub(addr_before_instruction),
                );
                let next_phys_page = Page::page_of(current_address);
                let mapping_is_contiguous = unsafe { JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS }
                    && matches!(analysis.ty, AnalysisType::Normal)
                    && (pages.contains(&next_phys_page) || pages.len() < max_pages as usize)
                    && cpu::translate_address_read_no_side_effects(virt_after as i32)
                        == Ok(current_address);
                if !mapping_is_contiguous {
                    profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
                    break;
                }
                crossed_page_virt_target = Some(virt_after as i32);
            }

            dbg_assert!(
                !crossed_page || unsafe { JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS }
            );
            let current_virt_addr = to_visit & !0xFFF | current_address as i32 & 0xFFF;

            if analysis.ty == AnalysisType::STI && is_near_end_of_page(current_address) {
                // cut off before the STI so that it is handled by interpreted mode
                profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
                break;
            }

            current_block.number_of_instructions += 1;
            current_block.last_instruction_addr = addr_before_instruction;
            current_block.end_addr = current_address;
            if unsafe { JIT_EXACT_PAGE_TAIL } && is_near_end_of_page(current_address) {
                unsafe {
                    JIT_EXACT_PAGE_TAIL_INSTRUCTIONS_COMPILED =
                        JIT_EXACT_PAGE_TAIL_INSTRUCTIONS_COMPILED.saturating_add(1);
                }
            }

            if let Some(virt_after) = crossed_page_virt_target {
                // End immediately after the crossing instruction. +0x1000 makes
                // the existing page-change EIP update produce the true sequential
                // virtual address from the old page's base and the new low bits.
                current_block.ty = BasicBlockType::Normal {
                    next_block_addr: follow_jump(
                        virt_after,
                        ctx,
                        &mut pages,
                        &mut page_blacklist,
                        max_pages,
                        tier2_region,
                        &mut marked_as_entry,
                        &mut to_visit_stack,
                        &cpu_template,
                    ),
                    jump_offset: 0x1000,
                    jump_offset_is_32: true,
                };
                unsafe {
                    JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS_COMPILED =
                        JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS_COMPILED.saturating_add(1);
                }
                break;
            }

            match analysis.ty {
                AnalysisType::Normal | AnalysisType::STI => {
                    dbg_assert!(has_next_instruction);
                    dbg_assert!(!analysis.absolute_jump);

                    if current_block.has_sti {
                        // Convert next instruction after STI (i.e., the current instruction) into block boundary

                        marked_as_entry.insert(current_virt_addr);
                        to_visit_stack.push(current_virt_addr);

                        break;
                    }

                    if analysis.ty == AnalysisType::STI {
                        dbg_assert!(
                            !is_near_end_of_page(current_address) || page_tail_entries_enabled(),
                            "should be handled above"
                        );

                        current_block.has_sti = true;
                    }
                    else {
                        // Only split non-STI blocks (one instruction needs to run after STI before
                        // handle_irqs may be called)

                        if basic_blocks.contains_key(&current_address) {
                            dbg_assert!(!is_near_end_of_page(current_address) || page_tail_entries_enabled());
                            current_block.ty = BasicBlockType::Normal {
                                next_block_addr: Some(current_address),
                                jump_offset: 0,
                                jump_offset_is_32: true,
                            };
                            break;
                        }
                    }
                },
                AnalysisType::Jump {
                    offset,
                    is_32,
                    condition: Some(condition),
                } => {
                    dbg_assert!(!analysis.absolute_jump);
                    // conditional jump: continue at next and continue at jump target

                    let jump_target = if is_32 {
                        current_virt_addr + offset
                    }
                    else {
                        cpu.cs_offset as i32
                            + (current_virt_addr - cpu.cs_offset as i32 + offset & 0xFFFF)
                    };

                    dbg_assert!(has_next_instruction);
                    to_visit_stack.push(current_virt_addr);

                    let next_block_addr = if is_near_end_of_page(current_address) {
                        None
                    }
                    else {
                        Some(current_address)
                    };

                    current_block.ty = BasicBlockType::ConditionalJump {
                        next_block_addr,
                        next_block_branch_taken_addr: follow_jump(
                            jump_target,
                            ctx,
                            &mut pages,
                            &mut page_blacklist,
                            max_pages,
                            tier2_region,
                            &mut marked_as_entry,
                            &mut to_visit_stack,
                            &cpu_template,
                        ),
                        condition,
                        jump_offset: offset,
                        jump_offset_is_32: is_32,
                    };

                    break;
                },
                AnalysisType::Jump {
                    offset,
                    is_32,
                    condition: None,
                } => {
                    dbg_assert!(!analysis.absolute_jump);
                    // non-conditional jump: continue at jump target

                    let jump_target = if is_32 {
                        current_virt_addr + offset
                    }
                    else {
                        cpu.cs_offset as i32
                            + (current_virt_addr - cpu.cs_offset as i32 + offset & 0xFFFF)
                    };

                    if has_next_instruction {
                        // Execution will eventually come back to the next instruction (CALL)
                        marked_as_entry.insert(current_virt_addr);
                        to_visit_stack.push(current_virt_addr);
                    }

                    current_block.ty = BasicBlockType::Normal {
                        next_block_addr: follow_jump(
                            jump_target,
                            ctx,
                            &mut pages,
                            &mut page_blacklist,
                            max_pages,
                            tier2_region,
                            &mut marked_as_entry,
                            &mut to_visit_stack,
                            &cpu_template,
                        ),
                        jump_offset: offset,
                        jump_offset_is_32: is_32,
                    };

                    break;
                },
                AnalysisType::BlockBoundary => {
                    // a block boundary but not a jump, get out

                    if has_next_instruction {
                        // block boundary, but execution will eventually come back
                        // to the next instruction. Create a new basic block
                        // starting at the next instruction and register it as an
                        // entry point
                        marked_as_entry.insert(current_virt_addr);
                        to_visit_stack.push(current_virt_addr);
                        if !analysis.absolute_jump {
                            current_block.sync_boundary_fallthrough = Some(current_address);
                        }
                    }

                    if analysis.absolute_jump {
                        current_block.ty = BasicBlockType::AbsoluteEip;

                        // Tier-2R: grow the region across this indirect
                        // edge using profiled targets. Targets that join the
                        // region are marked as entries so the runtime
                        // jit_find_cache_entry_in_page re-dispatch stays
                        // intra-module instead of exiting to main_loop.
                        if unsafe { JIT_INDIRECT_REGIONS } {
                            let targets = trace_profiler::hot_indirect_targets(
                                addr_before_instruction,
                                JIT_INDIRECT_REGION_MAX_TARGETS,
                                unsafe { JIT_INDIRECT_REGION_MIN_SHARE } as u64,
                            );
                            // max_pages already carries the region budget (set at
                            // the top of jit_find_basic_blocks when regions are on).
                            // Targets are hottest-first, so the cap keeps the
                            // most-executed cases when it binds.
                            for target in targets {
                                if region_target_excluded(target) {
                                    // Thunk/callback/spin bucket — stub pages full of
                                    // OUT traps must never join a guest superblock.
                                    continue;
                                }
                                // Profiled eips are RAW runtime values: they can be
                                // stale (recorded before the page was overwritten) or
                                // land mid-instruction relative to the CURRENT bytes.
                                // Seeding such an offset as a dispatcher entry makes
                                // the runtime dispatch ENTER a misdecoded block —
                                // wrong ModRM/base → stores through garbage/NULL
                                // pointers at random guest sites. Direct edges
                                // only ever mark interpreter-registered boundaries;
                                // hold profiled targets to the same standard: the
                                // exact offset must be a registered entry point of
                                // its page (hot indirect targets are, via the
                                // module-exit hotness path). Conservative: an
                                // unregistered target just doesn't grow the region.
                                let registered = cpu::translate_address_read_no_side_effects(
                                    target as i32,
                                )
                                .ok()
                                .map_or(false, |phys| {
                                    ctx.entry_points
                                        .get(&Page::page_of(phys))
                                        .map_or(false, |page_hotness| {
                                            page_hotness
                                                .entry_points
                                                .contains(&(phys as u16 & 0xFFF))
                                        })
                                });
                                if !registered {
                                    continue;
                                }
                                if follow_jump(
                                    target as i32,
                                    ctx,
                                    &mut pages,
                                    &mut page_blacklist,
                                    max_pages,
                                    tier2_region,
                                    &mut marked_as_entry,
                                    &mut to_visit_stack,
                                    &cpu_template,
                                )
                                .is_some()
                                {
                                    marked_as_entry.insert(target as i32);
                                }
                            }
                        }
                    }
                    else if unsafe { JIT_REP_MOVS_REDUCED_SPILL }
                        && opcode_is_rep_movs(addr_before_instruction)
                    {
                        // REP MOVS remains a separate block so a page-sized
                        // partial copy can yield for interrupts. The generated
                        // helper branches to the module exit while ECX is still
                        // non-zero; a completed copy can therefore take this
                        // ordinary intra-module fallthrough without a full
                        // dispatcher round-trip.
                        current_block.ty = BasicBlockType::Normal {
                            next_block_addr: if is_near_end_of_page(current_address) {
                                None
                            }
                            else {
                                Some(current_address)
                            },
                            jump_offset: 0,
                            jump_offset_is_32: true,
                        };
                    }

                    break;
                },
            }

            if !unsafe {
                JIT_EXACT_PAGE_TAIL || JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS
            } && is_near_end_of_page(current_address)
            {
                profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
                break;
            }
        }

        if current_block.number_of_instructions == 0 {
            // Empty basic block, don't insert (only happens when STI is found near end of page)
            continue;
        }

        let previous_block = basic_blocks
            .range(..current_block.addr)
            .next_back()
            .filter(|(_, previous_block)| !previous_block.has_sti)
            .map(|(_, previous_block)| previous_block);

        if let Some(previous_block) = previous_block {
            if current_block.addr < previous_block.end_addr {
                // If this block overlaps with the previous block, re-analyze the previous block
                to_visit_stack.push(previous_block.virt_addr);

                let addr = previous_block.addr;
                let old_block = basic_blocks.remove(&addr);
                dbg_assert!(old_block.is_some());

                // Note that this does not ensure the invariant that two consecutive blocks don't
                // overlay. For that, we also need to check the following block.
            }
        }

        if current_block.number_of_instructions == 0 {
            // The first instruction could not be kept (it crosses into a page
            // that is not contiguous, or the module's page budget is spent):
            // the block stays as an exit at its own address, so an edge that
            // reaches it hands the address to the dispatcher, and it is never
            // an entry, which would loop without progress.
            profiler::stat_increment(stat::COMPILE_CUT_OFF_AT_END_OF_PAGE);
            current_block.ty = BasicBlockType::Exit;
            current_block.end_addr = current_block.addr;
            current_block.last_instruction_addr = current_block.addr;
            marked_as_entry.remove(&current_block.virt_addr);
            basic_blocks.insert(current_block.addr, current_block);
            continue;
        }

        dbg_assert!(current_block.addr < current_block.end_addr);
        dbg_assert!(current_block.addr <= current_block.last_instruction_addr);
        dbg_assert!(current_block.last_instruction_addr < current_block.end_addr);

        basic_blocks.insert(current_block.addr, current_block);
    }

    dbg_assert!(pages.len() as u32 <= max_pages);

    for block in basic_blocks.values_mut() {
        if marked_as_entry.contains(&block.virt_addr) {
            block.is_entry_block = true;
        }
    }

    // Tier-2 tiny-leaf call fusion. CALL discovery represents a direct call as a
    // Normal edge whose target differs from end_addr and marks the fall-through
    // continuation as an entry. A single-block C3 leaf cannot contain another
    // control transfer; duplicating it therefore removes one AbsoluteEip dispatch
    // without changing CALL/RET stack semantics. Emission still checks the popped
    // runtime EIP, so self-modifying/hand-written code that changes the return
    // address takes the exact legacy resolver path.
    if tier2 && unsafe { JIT_TIER2_LEAF_CALL_FUSION } {
        let mut sites = Vec::new();
        for block in basic_blocks.values() {
            // The CFG shape below is shared by CALL and JMP. RET speculation can
            // tolerate a false candidate because it guards runtime EIP, but fusion
            // would actually execute the target and therefore requires a proven
            // near direct CALL instruction.
            if memory::read8(block.last_instruction_addr) as u8 != 0xE8 {
                continue;
            }
            let target = match block.ty {
                BasicBlockType::Normal { next_block_addr: Some(target), .. }
                    if target != block.end_addr => target,
                _ => continue,
            };
            if !basic_blocks.contains_key(&block.end_addr) {
                continue;
            }
            let Some(callee) = basic_blocks.get(&target) else { continue };
            if callee.number_of_instructions > unsafe { LEAF_CALL_FUSION_MAX_INSTR }
                || callee.has_sti
                || callee.ty != BasicBlockType::AbsoluteEip
                || memory::read8(callee.last_instruction_addr) as u8 != 0xC3
            {
                continue;
            }
            sites.push((block.addr, target));
        }
        for (call_addr, target) in sites {
            if let Some(block) = basic_blocks.get_mut(&call_addr) {
                block.inline_leaf = Some(target);
                unsafe {
                    LEAF_CALL_FUSION_SITES_COMPILED =
                        LEAF_CALL_FUSION_SITES_COMPILED.wrapping_add(1);
                }
            }
        }
    }

    // RET-target speculation post-pass. For every module-local
    // call site (a Normal block whose jump target isn't its fall-through AND whose
    // fall-through was registered as an entry — the CALL discovery shape at the
    // Jump{condition:None} arm above), walk the callee's blocks within a bounded
    // instruction budget. If every path ends in a genuine RET (opcode C3/C2 — an
    // AbsoluteEip block can also be jmp/call r/m, which must NOT be speculated as a
    // return) with no nested calls/STI/module-exits, annotate those RET blocks with
    // the call site's return address. Wrong or stale candidates are harmless: the
    // emitter guards each with an eip compare and falls through to the normal
    // dispatch. A page dirtied under this module frees the whole module (multi-page
    // sweep in free_wasm_module), annotations included — no new SMC surface.
    if ret_speculation_enabled() {
        let fall_through_virt =
            |b: &BasicBlock| b.virt_addr & !0xFFF | b.end_addr as i32 & 0xFFF;

        let mut call_sites: Vec<(u32, i32, u32)> = Vec::new();
        for block in basic_blocks.values() {
            if let BasicBlockType::Normal { next_block_addr: Some(target), .. } = block.ty {
                if target != block.end_addr
                    && marked_as_entry.contains(&fall_through_virt(block))
                    && basic_blocks.contains_key(&block.end_addr)
                {
                    call_sites.push((target, fall_through_virt(block), block.end_addr));
                }
            }
        }

        let mut annotations: Vec<(u32, (i32, u32))> = Vec::new();
        for &(callee, ret_virt, ret_phys) in &call_sites {
            let mut visited: HashSet<u32> = HashSet::new();
            let mut stack = vec![callee];
            let mut instr_budget = unsafe {
                if tier2 { JIT_TIER2_RET_SPEC_MAX_INSTR } else { JIT_RET_SPEC_MAX_INSTR }
            };
            let mut rets: Vec<u32> = Vec::new();
            let mut ok = true;
            while let Some(addr) = stack.pop() {
                if !visited.insert(addr) {
                    continue;
                }
                let b = match basic_blocks.get(&addr) {
                    Some(b) => b,
                    None => {
                        ok = false;
                        break;
                    },
                };
                if b.has_sti {
                    ok = false;
                    break;
                }
                match instr_budget.checked_sub(b.number_of_instructions) {
                    Some(rest) => instr_budget = rest,
                    None => {
                        ok = false;
                        break;
                    },
                }
                match &b.ty {
                    BasicBlockType::AbsoluteEip => {
                        let opcode = memory::read8(b.last_instruction_addr) as u8;
                        if opcode == 0xC3 || opcode == 0xC2 {
                            rets.push(b.addr);
                        }
                        else {
                            ok = false;
                            break;
                        }
                    },
                    BasicBlockType::Normal { next_block_addr: Some(t), .. } => {
                        // A nested call returns INTO the callee, not to our site —
                        // its RETs must not be annotated with our return address.
                        if *t != b.end_addr && marked_as_entry.contains(&fall_through_virt(b)) {
                            ok = false;
                            break;
                        }
                        stack.push(*t);
                    },
                    BasicBlockType::Normal { next_block_addr: None, .. }
                    | BasicBlockType::Exit => {
                        ok = false;
                        break;
                    },
                    BasicBlockType::ConditionalJump {
                        next_block_addr,
                        next_block_branch_taken_addr,
                        ..
                    } => {
                        match (next_block_addr, next_block_branch_taken_addr) {
                            (Some(n), Some(t)) => {
                                stack.push(*n);
                                stack.push(*t);
                            },
                            _ => {
                                ok = false;
                                break;
                            },
                        }
                    },
                }
            }
            if ok {
                for ret_addr in rets {
                    annotations.push((ret_addr, (ret_virt, ret_phys)));
                }
            }
        }

        for (ret_addr, cand) in annotations {
            if let Some(b) = basic_blocks.get_mut(&ret_addr) {
                if b.ret_speculation.len() < RET_SPEC_MAX_CANDIDATES
                    && !b.ret_speculation.contains(&cand)
                {
                    b.ret_speculation.push(cand);
                }
            }
        }
    }

    let basic_blocks: Vec<BasicBlock> = basic_blocks.into_iter().map(|(_, block)| block).collect();

    for i in 0..basic_blocks.len() - 1 {
        let next_block_addr = basic_blocks[i + 1].addr;
        let next_block_end_addr = basic_blocks[i + 1].end_addr;
        let next_block_is_entry = basic_blocks[i + 1].is_entry_block;
        let block = &basic_blocks[i];
        dbg_assert!(block.addr < next_block_addr);
        if next_block_addr < block.end_addr {
            dbg_log!(
                "Overlapping first=[from={:x} to={:x} is_entry={}] second=[from={:x} to={:x} is_entry={}]",
                block.addr,
                block.end_addr,
                block.is_entry_block as u8,
                next_block_addr,
                next_block_end_addr,
                next_block_is_entry as u8
            );
        }
    }

    basic_blocks
}

#[no_mangle]
#[cfg(debug_assertions)]
pub fn jit_force_generate_unsafe(virt_addr: i32) {
    dbg_assert!(
        !is_near_end_of_page(virt_addr as u32) || page_tail_entries_enabled(),
        "cannot force compile near end of page"
    );
    jit_increase_hotness_and_maybe_compile(
        virt_addr,
        cpu::translate_address_read(virt_addr).unwrap(),
        cpu::get_seg_cs() as u32,
        cpu::get_state_flags(),
        unsafe { JIT_THRESHOLD },
    );
    dbg_assert!(!get_jit_state().compiling.is_empty());
}

#[inline(never)]
fn jit_analyze_and_generate(
    ctx: &mut JitState,
    virt_entry_point: i32,
    phys_entry_point: u32,
    cs_offset: u32,
    state_flags: CachedStateFlags,
) {
    let t0 = unsafe { cpu::js::microtick() };
    jit_analyze_and_generate_untimed(ctx, virt_entry_point, phys_entry_point, cs_offset, state_flags);
    let us = (unsafe { cpu::js::microtick() } - t0) * 1000.0;
    unsafe {
        JIT_CODEGEN_TOTAL_US += us;
        if us > JIT_CODEGEN_MAX_US {
            JIT_CODEGEN_MAX_US = us;
        }
        JIT_CODEGEN_COUNT = JIT_CODEGEN_COUNT.wrapping_add(1);
    }
}

fn jit_analyze_and_generate_untimed(
    ctx: &mut JitState,
    virt_entry_point: i32,
    phys_entry_point: u32,
    cs_offset: u32,
    state_flags: CachedStateFlags,
) {
    let page = Page::page_of(phys_entry_point);

    dbg_assert!(ctx.compiling.len() < unsafe { JIT_MAX_PENDING_COMPILES.max(1) as usize });

    let entry_points = match ctx.entry_points.get(&page) {
        None => return,
        Some(page_hotness) => &page_hotness.entry_points,
    };

    let existing_entry_points = match ctx.pages.get(&page) {
        Some(PageInfo { entry_points, .. }) => HashSet::from_iter(entry_points.iter().map(|x| x.0)),
        None => HashSet::new(),
    };

    if entry_points
        .iter()
        .all(|entry_point| existing_entry_points.contains(entry_point))
    {
        profiler::stat_increment(stat::COMPILE_SKIPPED_NO_NEW_ENTRY_POINTS);
        return;
    }

    // XXX: check and remove
    //let old_length = entry_points.len();
    //entry_points.extend(existing_entry_points);
    //dbg_log!(
    //    "{} + {} = {}",
    //    entry_points.len(),
    //    existing_entry_points.len(),
    //    entry_points.union(&existing_entry_points).count()
    //);
    //dbg_assert!(entry_points.union(&existing_entry_points).count() == entry_points.len());

    profiler::stat_increment(stat::COMPILE);

    let cpu = CpuContext {
        eip: 0,
        prefixes: 0,
        cs_offset,
        state_flags,
    };

    dbg_assert!(
        cpu::translate_address_read_no_side_effects(virt_entry_point).unwrap() == phys_entry_point
    );
    let virt_page = Page::page_of(virt_entry_point as u32);
    let mut entry_points: HashSet<i32> = entry_points
        .iter()
        .map(|e| virt_page.to_address() as i32 | *e as i32)
        .collect();
    let tier2_region = ctx.tier2_regions.get(&page).cloned();
    if let Some(region) = &tier2_region {
        entry_points.extend(region.seeds.iter().copied());
    }
    let basic_blocks =
        jit_find_basic_blocks(ctx, entry_points, cpu.clone(), tier2_region.as_ref());

    let mut pages = HashSet::new();

    for b in basic_blocks.iter() {
        pages.insert(Page::page_of(b.addr));
        // A retained boundary-straddling instruction consumes bytes from the
        // page containing end_addr - 1. Register it even if region growth could
        // not retain the following block, so dirty-page invalidation covers
        // every byte used by this translation.
        pages.insert(Page::page_of(b.end_addr.wrapping_sub(1)));
    }

    let print = false;

    for b in basic_blocks.iter() {
        if !print {
            break;
        }
        let last_instruction_opcode = memory::read32s(b.last_instruction_addr);
        let op = opstats::decode(last_instruction_opcode as u32);
        dbg_log!(
            "BB: 0x{:x} {}{:02x} {} {}",
            b.addr,
            if op.is_0f { "0f" } else { "" },
            op.opcode,
            if b.is_entry_block { "entry" } else { "noentry" },
            match &b.ty {
                BasicBlockType::ConditionalJump {
                    next_block_addr: Some(next_block_addr),
                    next_block_branch_taken_addr: Some(next_block_branch_taken_addr),
                    ..
                } => format!(
                    "0x{:x} 0x{:x}",
                    next_block_addr, next_block_branch_taken_addr
                ),
                BasicBlockType::ConditionalJump {
                    next_block_addr: None,
                    next_block_branch_taken_addr: Some(next_block_branch_taken_addr),
                    ..
                } => format!("0x{:x}", next_block_branch_taken_addr),
                BasicBlockType::ConditionalJump {
                    next_block_addr: Some(next_block_addr),
                    next_block_branch_taken_addr: None,
                    ..
                } => format!("0x{:x}", next_block_addr),
                BasicBlockType::ConditionalJump {
                    next_block_addr: None,
                    next_block_branch_taken_addr: None,
                    ..
                } => format!(""),
                BasicBlockType::Normal {
                    next_block_addr: Some(next_block_addr),
                    ..
                } => format!("0x{:x}", next_block_addr),
                BasicBlockType::Normal {
                    next_block_addr: None,
                    ..
                } => format!(""),
                BasicBlockType::Exit => format!(""),
                BasicBlockType::AbsoluteEip => format!(""),
            }
        );
    }

    let graph = control_flow::make_graph(&basic_blocks);
    let mut structure = control_flow::loopify(&graph);

    if print {
        dbg_log!("before blockify:");
        for group in &structure {
            dbg_log!("=> Group");
            group.print(0);
        }
    }

    control_flow::blockify(&mut structure, &graph);

    if cfg!(debug_assertions) {
        control_flow::assert_invariants(&structure);
    }

    if print {
        dbg_log!("after blockify:");
        for group in &structure {
            dbg_log!("=> Group");
            group.print(0);
        }
    }

    if ctx.wasm_table_index_free_list.is_empty() && unsafe { JIT_PARTIAL_EVICTION } != 0 {
        // Reclaim only what is cold, so the hot working set keeps both its
        // modules and its hotness. A sweep that frees nothing means every module
        // is in use, and the full flush below is the only way to make progress.
        jit_evict_unused(ctx);
    }

    if ctx.wasm_table_index_free_list.is_empty() {
        // Always-on: a full flush discards every compiled module and forces the
        // whole working set back through the interpreter and the hotness ramp.
        // A cold boot compiles more modules than the table holds, so this fires
        // during normal play and its rate is not otherwise observable.
        unsafe { JIT_CACHE_FLUSHES = JIT_CACHE_FLUSHES.wrapping_add(1) };
        dbg_log!("wasm_table_index_free_list empty, clearing cache");

        // When no free slots are available, delete all cached modules. We could increase the
        // size of the table, but this way the initial size acts as an upper bound for the
        // number of wasm modules that we generate, which we want anyway to avoid getting our
        // tab killed by browsers due to memory constraints.
        jit_clear_cache(ctx);

        profiler::stat_increment(stat::INVALIDATE_ALL_MODULES_NO_FREE_WASM_INDICES);

        dbg_log!(
            "after jit_clear_cache: {} free",
            ctx.wasm_table_index_free_list.len(),
        );

        // This assertion can fail if all entries are pending (not possible unless
        // WASM_TABLE_SIZE is set very low)
        dbg_assert!(!ctx.wasm_table_index_free_list.is_empty());
    }

    // allocate an index in the wasm table
    let wasm_table_index = ctx
        .wasm_table_index_free_list
        .pop()
        .expect("allocate wasm table index");
    dbg_assert!(wasm_table_index != WasmTableIndex(0));

    dbg_assert!(!pages.is_empty());
    // The effective cap can exceed the global MAX_PAGES when indirect regions or
    // tier-2 budgets are active (see max_pages in jit_find_basic_blocks) — assert
    // against the widest configured budget, not the base knob.
    dbg_assert!(
        pages.len()
            <= unsafe {
                MAX_PAGES
                    .max(JIT_INDIRECT_REGION_MAX_PAGES)
                    .max(TIER2_MAX_PAGES)
            } as usize
    );

    let basic_block_by_addr: HashMap<u32, BasicBlock> =
        basic_blocks.into_iter().map(|b| (b.addr, b)).collect();

    let fastmem_generation = fastmem_compile_generation(state_flags);

    let entries = jit_generate_module(
        structure,
        &basic_block_by_addr,
        cpu,
        &mut ctx.wasm_builder,
        wasm_table_index,
        state_flags,
        fastmem_generation,
    );
    dbg_assert!(!entries.is_empty());

    let mut page_info = HashMap::new();
    for &(addr, state) in &entries {
        let code = page_info
            .entry(Page::page_of(addr))
            .or_insert_with(|| PageInfo {
                wasm_table_index,
                state_flags,
                entry_points: Vec::new(),
                hidden_wasm_table_indices: Vec::new(),
            });
        code.entry_points.push((addr as u16 & 0xFFF, state));
    }
    // Invalidation completeness: EVERY page the module compiled code from must be
    // findable by jit_dirty_page, or a write to it leaves a STALE module running old
    // code. page_info above only covers pages that materialized a dispatcher entry;
    // entry blocks can be dropped by overlap elimination or near-end-of-page cutoffs
    // while non-entry blocks from the page remain. Register the rest with an empty
    // entry list — set_tlb_code leaves their state_table all-miss (u16::MAX), so the
    // only effect is that free_wasm_module's page sweep covers them.
    for &p in &pages {
        page_info.entry(p).or_insert_with(|| PageInfo {
            wasm_table_index,
            state_flags,
            entry_points: Vec::new(),
            hidden_wasm_table_indices: Vec::new(),
        });
    }

    profiler::stat_increment_by(
        stat::COMPILE_WASM_TOTAL_BYTES,
        ctx.wasm_builder.get_output_len() as u64,
    );
    unsafe {
        JIT_CODEGEN_BYTES_TOTAL =
            JIT_CODEGEN_BYTES_TOTAL.wrapping_add(ctx.wasm_builder.get_output_len() as u64);
    }
    profiler::stat_increment_by(stat::COMPILE_PAGE, pages.len() as u64);

    for &p in &pages {
        ctx.deferred_compile_pages.remove(&p);
        ctx.entry_points
            .entry(p)
            .or_insert_with(|| PageHotness { hotness: 0, entry_points: HashSet::new() });
    }

    cpu::tlb_set_has_code_multiple(&pages, true);

    dbg_assert!(!ctx.compiling.contains_key(&wasm_table_index));
    ctx.compiling.insert(
        wasm_table_index,
        CompilingPageState::Compiling { pages: page_info },
    );
    unsafe {
        JIT_COMPILE_STARTED = JIT_COMPILE_STARTED.wrapping_add(1);
        JIT_COMPILE_PENDING_HIGH_WATER =
            JIT_COMPILE_PENDING_HIGH_WATER.max(ctx.compiling.len() as u32);
    }

    let phys_addr = page.to_address();

    // will call codegen_finalize_finished asynchronously when finished
    codegen_finalize(
        wasm_table_index,
        phys_addr,
        state_flags,
        ctx.wasm_builder.get_output_ptr() as u32,
        ctx.wasm_builder.get_output_len(),
    );

    check_jit_state_invariants(ctx);
}

fn page_is_compiling(ctx: &JitState, page: Page) -> bool {
    ctx.compiling.values().any(|state| match state {
        CompilingPageState::Compiling { pages } => pages.contains_key(&page),
        CompilingPageState::CompilingWritten => false,
    })
}

fn drain_deferred_compiles(ctx: &mut JitState) {
    if !unsafe { JIT_DEFERRED_COMPILE_QUEUE } {
        return;
    }

    let max_pending = unsafe { JIT_MAX_PENDING_COMPILES.max(1) as usize };
    while ctx.compiling.len() < max_pending {
        let Some(candidate) = ctx.deferred_compiles.pop_front() else { break };

        // Dirty-page invalidation and cache clears lazily cancel a candidate by
        // removing it from this set; its small FIFO record can then be skipped.
        if !ctx.deferred_compile_pages.remove(&candidate.page) {
            continue;
        }
        if !ctx.entry_points.contains_key(&candidate.page)
            || page_is_compiling(ctx, candidate.page)
            || cpu::translate_address_read_no_side_effects(candidate.virt_address)
                != Ok(candidate.phys_address)
        {
            unsafe {
                JIT_COMPILE_DEFERRED_DROPPED = JIT_COMPILE_DEFERRED_DROPPED.wrapping_add(1);
            }
            continue;
        }

        let pending_before = ctx.compiling.len();
        jit_analyze_and_generate(
            ctx,
            candidate.virt_address,
            candidate.phys_address,
            candidate.cs_offset,
            candidate.state_flags,
        );
        if ctx.compiling.len() > pending_before {
            unsafe {
                JIT_COMPILE_DEFERRED_STARTED = JIT_COMPILE_DEFERRED_STARTED.wrapping_add(1);
            }
        }
    }
}

#[no_mangle]
pub fn codegen_finalize_finished(
    wasm_table_index: WasmTableIndex,
    phys_addr: u32,
    state_flags: CachedStateFlags,
    compile_us: u32,
) {
    let mut ctx = get_jit_state();

    dbg_assert!(wasm_table_index != WasmTableIndex(0));

    dbg_log!(
        "Finished compiling for page at {:x}",
        Page::page_of(phys_addr).to_address()
    );

    unsafe {
        JIT_COMPILE_COMPLETED = JIT_COMPILE_COMPLETED.wrapping_add(1);
        JIT_COMPILE_TOTAL_US = JIT_COMPILE_TOTAL_US.wrapping_add(compile_us as u64);
        JIT_COMPILE_MAX_US = JIT_COMPILE_MAX_US.max(compile_us);
    }
    let pages = match ctx.compiling.remove(&wasm_table_index) {
        None => {
            dbg_assert!(false);
            return;
        },
        Some(CompilingPageState::CompilingWritten) => {
            profiler::stat_increment(stat::INVALIDATE_MODULE_WRITTEN_WHILE_COMPILED);
            unsafe {
                FREE_SITE_WRITTEN_WHILE_COMPILING =
                    FREE_SITE_WRITTEN_WHILE_COMPILING.wrapping_add(1)
            };
            free_wasm_table_index(&mut ctx, wasm_table_index);
            drain_deferred_compiles(&mut ctx);
            check_jit_state_invariants(&mut ctx);
            return;
        },
        Some(CompilingPageState::Compiling { pages }) => {
            dbg_assert!(!pages.is_empty());
            pages
        },
    };

    for i in 0..unsafe { cpu::valid_tlb_entries_count } {
        let page = unsafe { cpu::valid_tlb_entries[i as usize] };
        let entry = unsafe { cpu::tlb_data[page as usize] };
        if 0 != entry {
            let tlb_physical_page = Page::of_u32(
                (entry as u32 >> 12 ^ page as u32) - (unsafe { memory::mem8 } as u32 >> 12),
            );
            if let Some(info) = pages.get(&tlb_physical_page) {
                set_tlb_code(
                    Page::of_u32(page as u32),
                    wasm_table_index,
                    &info.entry_points,
                    state_flags,
                );
            }
        }
    }

    let mut check_for_unused_wasm_table_index = HashSet::new();

    for (page, mut info) in pages {
        if let Some(old_entry) = ctx.pages.remove(&page) {
            info.hidden_wasm_table_indices
                .extend(old_entry.hidden_wasm_table_indices);
            info.hidden_wasm_table_indices
                .push(old_entry.wasm_table_index);
            check_for_unused_wasm_table_index.insert(old_entry.wasm_table_index);
        }
        ctx.pages.insert(page, info);
    }

    let unused: Vec<&WasmTableIndex> = check_for_unused_wasm_table_index
        .iter()
        .filter(|&&i| ctx.pages.values().all(|page| page.wasm_table_index != i))
        .collect();

    for &index in unused {
        for p in ctx.pages.values_mut() {
            p.hidden_wasm_table_indices.retain(|&w| w != index);
        }

        dbg_log!("unused after overwrite {}", index.to_u16());
        profiler::stat_increment(stat::INVALIDATE_MODULE_UNUSED_AFTER_OVERWRITE);
        unsafe { FREE_SITE_OVERWRITE = FREE_SITE_OVERWRITE.wrapping_add(1) };
        free_wasm_table_index(&mut ctx, index);
    }

    drain_deferred_compiles(&mut ctx);
    check_jit_state_invariants(&mut ctx);
}

pub fn update_tlb_code(virt_page: Page, phys_page: Page) {
    let ctx = get_jit_state();

    match ctx.pages.get(&phys_page) {
        Some(PageInfo {
            wasm_table_index,
            entry_points,
            state_flags,
            hidden_wasm_table_indices: _,
        }) => set_tlb_code(virt_page, *wasm_table_index, entry_points, *state_flags),
        None => {
            if dispatch_meta_clear(virt_page.to_u32()) {
                ret_cache_invalidate_page_tlb(virt_page.to_u32());
            }
        },
    };
    match ctx.external_pages.get(&phys_page) {
        Some(info) => dispatch_ext_set(virt_page, info.wasm_table_index, &info.entry_points, info.state_flags),
        None => { dispatch_ext_clear(virt_page.to_u32()); },
    };
}

// Publish a page's dispatch entries into the DOD SoA (see DISPATCH_META above).
// The per-unit fastmem generation is intentionally NOT stored: the unit's own
// prologue guard self-deopts a stale unit on entry (see the SoA header comment).
pub fn set_tlb_code(
    virt_page: Page,
    wasm_table_index: WasmTableIndex,
    entries: &Vec<(u16, u16)>,
    state_flags: CachedStateFlags,
) {
    dispatch_meta_set(virt_page, wasm_table_index, entries, state_flags);
    if !block_chaining_enabled() {
        return;
    }
    let base = virt_page.to_address();
    for &(offset, unit_state) in entries {
        unsafe {
            exact_dispatch_insert(
                base | offset as u32,
                state_flags,
                wasm_table_index,
                unit_state,
            );
        }
    }
}

// Statically-chainable direct-jump exit. When enabled, resolve the already-written
// runtime EIP through the exact cross-module index, verify the scheduler budget in
// generated wasm, and tail-call the target module. The lookup deliberately happens
// before spilling guest registers: an unpublished target or exhausted budget pays
// no more writeback than the ordinary module exit.
fn gen_chain_or_exit_to_known_successor(
    ctx: &mut JitContext,
    state_flags: CachedStateFlags,
    last_instruction_addr: u32,
) {
    if !block_chaining_enabled() {
        codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_CHAINABLE);
        ctx.builder.br(ctx.exit_label);
        return;
    }

    if let Some((memo_slot, memo_address)) = allocate_chain_site_memo() {
        ctx.builder.load_fixed_i64(memo_address);
        let memo = ctx.builder.set_new_local_i64();
        ctx.builder.get_local_i64(&memo);
        ctx.builder.wrap_i64_to_i32();
        let memo_packed = ctx.builder.set_new_local();

        // A negative memo is valid until any exact target is newly published.
        ctx.builder.get_local(&memo_packed);
        ctx.builder.const_i32(-1);
        ctx.builder.eq_i32();
        ctx.builder.if_i32();
        ctx.builder.load_fixed_i32(std::ptr::addr_of!(EXACT_DISPATCH_PUBLISH_EPOCH) as u32);
        ctx.builder.get_local_i64(&memo);
        ctx.builder.const_i64(32);
        ctx.builder.shr_u_i64();
        ctx.builder.wrap_i64_to_i32();
        ctx.builder.eq_i32();
        ctx.builder.if_i32();
        ctx.builder.const_i32(-1);
        ctx.builder.else_();
        codegen::gen_get_eip(ctx.builder);
        ctx.builder.const_i32(state_flags.to_u32() as i32);
        ctx.builder.const_i32(memo_slot as i32);
        ctx.builder
            .call_fn3_ret("jit_find_cache_entry_exact_chain_memo");
        ctx.builder.block_end();
        ctx.builder.else_();

        // Positive memo: a single global target epoch is sufficient because the
        // table-slot free/dispatch-eviction funnel invalidates every chain memo.
        ctx.builder.get_local_i64(&memo);
        ctx.builder.const_i64(0);
        ctx.builder.ne_i64();
        ctx.builder.load_fixed_i32(std::ptr::addr_of!(CHAIN_TARGET_EPOCH) as u32);
        ctx.builder.get_local_i64(&memo);
        ctx.builder.const_i64(32);
        ctx.builder.shr_u_i64();
        ctx.builder.wrap_i64_to_i32();
        ctx.builder.eq_i32();
        ctx.builder.and_i32();
        ctx.builder.if_i32();
        ctx.builder.get_local(&memo_packed);
        ctx.builder.else_();
        codegen::gen_get_eip(ctx.builder);
        ctx.builder.const_i32(state_flags.to_u32() as i32);
        ctx.builder.const_i32(memo_slot as i32);
        ctx.builder
            .call_fn3_ret("jit_find_cache_entry_exact_chain_memo");
        ctx.builder.block_end();
        ctx.builder.block_end();

        ctx.builder.free_local(memo_packed);
        ctx.builder.free_local_i64(memo);
    }
    else {
        codegen::gen_get_eip(ctx.builder);
        ctx.builder.const_i32(state_flags.to_u32() as i32);
        ctx.builder.call_fn2_ret("jit_find_cache_entry_exact_chain");
    }
    let packed_target = ctx.builder.set_new_local();

    ctx.builder.get_local(&packed_target);
    ctx.builder.const_i32(0);
    ctx.builder.ge_i32();
    ctx.builder.if_void();

    // do_many_cycles_native already decoded the writable hypercall budget for
    // this slice. A direct JIT edge cannot cross the thunk/module exit that may
    // change it, so use the cached value instead of re-reading and branching on
    // the hypercall page at every tiny-block edge.
    ctx.builder.load_fixed_i32(chain_budget_address());
    let cycle_limit = ctx.builder.set_new_local();

    // limit != 0 && (global + pending - slice_start) < limit && !in_hlt
    ctx.builder.get_local(&cycle_limit);
    ctx.builder.const_i32(0);
    ctx.builder.ne_i32();
    ctx.builder.load_fixed_i32(global_pointers::instruction_counter as u32);
    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.add_i32();
    ctx.builder.load_fixed_i32(
        std::ptr::addr_of!(cpu::jit_cycle_start_instruction_counter) as u32,
    );
    ctx.builder.sub_i32();
    ctx.builder.get_local(&cycle_limit);
    ctx.builder.ltu_i32();
    ctx.builder.and_i32();
    ctx.builder.load_fixed_u8(global_pointers::in_hlt as u32);
    ctx.builder.eqz_i32();
    ctx.builder.and_i32();
    ctx.builder.if_void();

    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_CHAINED_EDGE);
    codegen::gen_move_registers_from_locals_to_memory(ctx);
    codegen::gen_update_instruction_counter(ctx);
    ctx.builder.const_i32(0);
    ctx.builder.set_local(&ctx.instruction_counter);

    ctx.builder.get_local(&packed_target);
    ctx.builder.const_i32(0xFFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&packed_target);
    ctx.builder.const_i32(16);
    ctx.builder.shr_u_i32();
    ctx.builder.return_call_indirect_fn1();
    ctx.builder.block_end();

    // A published target existed, but yielding now is architecturally required.
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_CHAINABLE);
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_CHAIN_BUDGET_EXIT);
    codegen::gen_debug_track_jit_exit(ctx.builder, last_instruction_addr);
    ctx.builder.free_local(cycle_limit);
    ctx.builder.br(ctx.exit_label);
    ctx.builder.block_end();

    // No live target was published for this exact EIP/state pair.
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_CHAINABLE);
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_CHAIN_MISS);
    ctx.builder.free_local(packed_target);
    codegen::gen_debug_track_jit_exit(ctx.builder, last_instruction_addr);
    ctx.builder.br(ctx.exit_label);

    unsafe {
        BLOCK_CHAIN_SITES_COMPILED = BLOCK_CHAIN_SITES_COMPILED.saturating_add(1);
    }
}

/// Emit the same lookup as `jit_find_cache_entry_in_page`, but directly into
/// the generated module. Leaves the dispatcher state (or -1) on the wasm stack.
///
/// Layout recap:
///   meta[virt >> 12] = state_flags:u32 | table_index:u16 | slab:u16
///   slabs[(slab << 12) | (virt & 0xfff)] = dispatcher_state:u16
///
/// `virt >> 9` is `(virt >> 12) * sizeof(u64)`, and every slab entry is u16.
/// Both derived addresses are naturally aligned. The arrays share the imported
/// linear memory with every generated JIT module, so no helper or JS transition
/// is required.
fn gen_find_cache_entry_in_page_inline(
    ctx: &mut JitContext,
    wasm_table_index: WasmTableIndex,
    state_flags: CachedStateFlags,
) {
    codegen::gen_profiler_stat_increment(ctx.builder, stat::INDIRECT_JUMP);
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::ABSEIP_DISPATCH);

    codegen::gen_get_eip(ctx.builder);
    let virt_address = ctx.builder.set_new_local();

    let meta_base = std::ptr::addr_of!(DISPATCH_META) as u32;
    ctx.builder.const_i32(meta_base as i32);
    ctx.builder.get_local(&virt_address);
    ctx.builder.const_i32(9);
    ctx.builder.shr_u_i32();
    ctx.builder.add_i32();
    ctx.builder.load_aligned_i64(0);
    let meta = ctx.builder.set_new_local_i64();

    // Comparing meta>>16 checks state_flags and table_index together while
    // intentionally ignoring the low slab index. A zero/unpublished meta cannot
    // match because wasm table slot zero is never assigned to generated code.
    let expected = ((state_flags.to_u32() as u64) << 16)
        | wasm_table_index.to_u16() as u64;
    ctx.builder.get_local_i64(&meta);
    ctx.builder.const_i64(16);
    ctx.builder.shr_u_i64();
    ctx.builder.const_i64(expected as i64);
    ctx.builder.eq_i64();
    ctx.builder.if_i32();

    let slabs_base = std::ptr::addr_of!(DISPATCH_SLABS) as u32;
    ctx.builder.const_i32(slabs_base as i32);
    ctx.builder.get_local_i64(&meta);
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.const_i32(0xFFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(13); // slab * 0x1000 entries * sizeof(u16)
    ctx.builder.shl_i32();
    ctx.builder.add_i32();
    ctx.builder.get_local(&virt_address);
    ctx.builder.const_i32(0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(1);
    ctx.builder.shl_i32();
    ctx.builder.add_i32();
    ctx.builder.load_aligned_u16(0);
    let unit_state = ctx.builder.set_new_local();

    ctx.builder.get_local(&unit_state);
    ctx.builder.const_i32(u16::MAX as i32);
    ctx.builder.ne_i32();
    ctx.builder.if_i32();
    ctx.builder.get_local(&unit_state);
    ctx.builder.else_();
    codegen::gen_profiler_stat_increment(ctx.builder, stat::INDIRECT_JUMP_NO_ENTRY);
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_INDIRECT);
    ctx.builder.const_i32(-1);
    ctx.builder.block_end();

    ctx.builder.else_();
    codegen::gen_profiler_stat_increment(ctx.builder, stat::INDIRECT_JUMP_NO_ENTRY);
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_INDIRECT);
    ctx.builder.const_i32(-1);
    ctx.builder.block_end();

    ctx.builder.free_local(unit_state);
    ctx.builder.free_local_i64(meta);
    ctx.builder.free_local(virt_address);
    unsafe {
        INLINE_INTRA_MODULE_DISPATCH_SITES_COMPILED =
            INLINE_INTRA_MODULE_DISPATCH_SITES_COMPILED.saturating_add(1);
    }
}

/// Return a packed cross-module target for the current AbsoluteEip. A stable
/// per-site target avoids the generated-module -> base-wasm resolver call while
/// a miss retains the complete legacy lookup and refreshes the cache. This runs
/// after guest registers/instruction count have already been flushed.
fn gen_dynamic_chain_site_pic_lookup(
    ctx: &mut JitContext,
    state_flags: CachedStateFlags,
) {
    let Some(slot) = allocate_dynamic_chain_site_pic()
    else {
        ctx.builder.const_i32(state_flags.to_u32() as i32);
        ctx.builder.call_fn1_ret("jit_find_cache_entry_for_dynamic_chaining");
        return;
    };
    let slot_offset = slot * 4;
    let target_address = std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_TARGETS) as u32 + slot_offset;
    let packed_address = std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_PACKED) as u32 + slot_offset;
    let epoch_address = std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_EPOCHS) as u32 + slot_offset;
    let second_target_address =
        std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_DIAG_SECOND_TARGETS) as u32 + slot_offset;
    let second_packed_address =
        std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_DIAG_SECOND_PACKED) as u32 + slot_offset;
    let second_epoch_address =
        std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_DIAG_SECOND_EPOCHS) as u32 + slot_offset;
    let third_target_address =
        std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_DIAG_THIRD_TARGETS) as u32 + slot_offset;
    let third_packed_address =
        std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_DIAG_THIRD_PACKED) as u32 + slot_offset;
    let third_epoch_address =
        std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_DIAG_THIRD_EPOCHS) as u32 + slot_offset;
    let fourth_target_address =
        std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_DIAG_FOURTH_TARGETS) as u32 + slot_offset;
    let fourth_packed_address =
        std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_DIAG_FOURTH_PACKED) as u32 + slot_offset;
    let fourth_epoch_address =
        std::ptr::addr_of!(DYNAMIC_CHAIN_SITE_DIAG_FOURTH_EPOCHS) as u32 + slot_offset;

    codegen::gen_get_eip(ctx.builder);
    let virt_address = ctx.builder.set_new_local();

    // A memo is usable only while its dispatch epoch is current and its last
    // target equals this site's runtime target.
    ctx.builder.load_fixed_i32(epoch_address);
    ctx.builder.load_fixed_i64(std::ptr::addr_of!(RET_CACHE_EPOCH) as u32);
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.eq_i32();
    ctx.builder.load_fixed_i32(target_address);
    ctx.builder.get_local(&virt_address);
    ctx.builder.eq_i32();
    ctx.builder.and_i32();

    // The historical helper checks this before every chain. Keep the same
    // scheduler boundary on the generated hit path; a failed guard enters the
    // helper, which returns -1 without refreshing the memo.
    ctx.builder.load_fixed_i32(chain_budget_address());
    let cycle_limit = ctx.builder.tee_new_local();
    ctx.builder.const_i32(0);
    ctx.builder.ne_i32();
    ctx.builder.and_i32();
    ctx.builder.load_fixed_i32(global_pointers::instruction_counter as u32);
    ctx.builder.load_fixed_i32(
        std::ptr::addr_of!(cpu::jit_cycle_start_instruction_counter) as u32,
    );
    ctx.builder.sub_i32();
    ctx.builder.get_local(&cycle_limit);
    ctx.builder.ltu_i32();
    ctx.builder.and_i32();
    ctx.builder.load_fixed_u8(global_pointers::in_hlt as u32);
    ctx.builder.eqz_i32();
    ctx.builder.and_i32();

    ctx.builder.if_i32();
    ctx.builder.load_fixed_i32(packed_address);
    ctx.builder.else_();

    // Compute the scheduler guard once for every polymorphic way. This whole
    // arm is skipped by the unchanged primary hit path above.
    ctx.builder.load_fixed_i32(chain_budget_address());
    let miss_cycle_limit = ctx.builder.tee_new_local();
    ctx.builder.const_i32(0);
    ctx.builder.ne_i32();
    ctx.builder.load_fixed_i32(global_pointers::instruction_counter as u32);
    ctx.builder.load_fixed_i32(
        std::ptr::addr_of!(cpu::jit_cycle_start_instruction_counter) as u32,
    );
    ctx.builder.sub_i32();
    ctx.builder.get_local(&miss_cycle_limit);
    ctx.builder.ltu_i32();
    ctx.builder.and_i32();
    ctx.builder.load_fixed_u8(global_pointers::in_hlt as u32);
    ctx.builder.eqz_i32();
    ctx.builder.and_i32();
    let miss_scheduler_ok = ctx.builder.set_new_local();

    // The shared resolver starts by reading this exact budget/HLT state and
    // returns -1 when it is false. Keep that overwhelmingly common quantum-exit
    // path inside generated wasm: no cross-instance Rust call, no repeated
    // instruction-counter loads. The true arm below remains the complete PIC +
    // authoritative resolver path.
    let budget_fast_exit = dynamic_chain_budget_fast_exit_enabled();
    if budget_fast_exit {
        ctx.builder.get_local(&miss_scheduler_ok);
        ctx.builder.if_i32();
    }

    // The primary-hit instruction stream above is identical whether this
    // option is enabled or not. Only its already-cold miss arm consults the
    // second target, so monomorphic sites pay no extra branch or memory load.
    ctx.builder.load_fixed_u8(
        std::ptr::addr_of!(JIT_DYNAMIC_CHAIN_SITE_PIC_SECOND_WAY) as u32,
    );
    ctx.builder.load_fixed_i32(second_epoch_address);
    ctx.builder.load_fixed_i64(std::ptr::addr_of!(RET_CACHE_EPOCH) as u32);
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.eq_i32();
    ctx.builder.and_i32();
    ctx.builder.load_fixed_i32(second_target_address);
    ctx.builder.get_local(&virt_address);
    ctx.builder.eq_i32();
    ctx.builder.and_i32();
    ctx.builder.get_local(&miss_scheduler_ok);
    ctx.builder.and_i32();
    ctx.builder.if_i32();
    ctx.builder.load_fixed_i32(second_packed_address);
    ctx.builder.else_();

    ctx.builder.load_fixed_u8(
        std::ptr::addr_of!(JIT_DYNAMIC_CHAIN_SITE_PIC_FOUR_WAY) as u32,
    );
    ctx.builder.load_fixed_i32(third_epoch_address);
    ctx.builder.load_fixed_i64(std::ptr::addr_of!(RET_CACHE_EPOCH) as u32);
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.eq_i32();
    ctx.builder.and_i32();
    ctx.builder.load_fixed_i32(third_target_address);
    ctx.builder.get_local(&virt_address);
    ctx.builder.eq_i32();
    ctx.builder.and_i32();
    ctx.builder.get_local(&miss_scheduler_ok);
    ctx.builder.and_i32();
    ctx.builder.if_i32();
    ctx.builder.load_fixed_i32(third_packed_address);
    ctx.builder.else_();

    ctx.builder.load_fixed_u8(
        std::ptr::addr_of!(JIT_DYNAMIC_CHAIN_SITE_PIC_FOUR_WAY) as u32,
    );
    ctx.builder.load_fixed_i32(fourth_epoch_address);
    ctx.builder.load_fixed_i64(std::ptr::addr_of!(RET_CACHE_EPOCH) as u32);
    ctx.builder.wrap_i64_to_i32();
    ctx.builder.eq_i32();
    ctx.builder.and_i32();
    ctx.builder.load_fixed_i32(fourth_target_address);
    ctx.builder.get_local(&virt_address);
    ctx.builder.eq_i32();
    ctx.builder.and_i32();
    ctx.builder.get_local(&miss_scheduler_ok);
    ctx.builder.and_i32();
    ctx.builder.if_i32();
    ctx.builder.load_fixed_i32(fourth_packed_address);
    ctx.builder.else_();
    ctx.builder.const_i32(state_flags.to_u32() as i32);
    ctx.builder.const_i32(slot as i32);
    ctx.builder
        .call_fn2_ret("jit_find_cache_entry_for_dynamic_chaining_site");
    ctx.builder.block_end();
    ctx.builder.block_end();
    ctx.builder.block_end();
    if budget_fast_exit {
        ctx.builder.else_();
        codegen::gen_dispatch_stat_increment(ctx.builder, stat::RET_CHAIN_MISS);
        ctx.builder.const_i32(-1);
        ctx.builder.block_end();
    }
    ctx.builder.block_end();

    ctx.builder.free_local(miss_scheduler_ok);
    ctx.builder.free_local(miss_cycle_limit);
    ctx.builder.free_local(cycle_limit);
    ctx.builder.free_local(virt_address);
    unsafe {
        DYNAMIC_CHAIN_SITE_PIC_COMPILED =
            DYNAMIC_CHAIN_SITE_PIC_COMPILED.saturating_add(1);
    }
}

fn jit_generate_module(
    structure: Vec<WasmStructure>,
    basic_blocks: &HashMap<u32, BasicBlock>,
    mut cpu: CpuContext,
    builder: &mut WasmBuilder,
    wasm_table_index: WasmTableIndex,
    state_flags: CachedStateFlags,
    fastmem_generation: Option<u64>,
) -> Vec<(u32, u16)> {
    builder.reset();

    let mut register_locals = (0..8)
        .map(|i| {
            builder.load_fixed_i32(global_pointers::get_reg32_offset(i));
            builder.set_new_local()
        })
        .collect();

    builder.const_i32(0);
    let instruction_counter = builder.set_new_local();

    // Flag-locals (idx 21): lazy-flag tuple lives in wasm locals for the whole
    // module — initialized from the memory globals here, spilled back at every
    // exit epilogue and around every non-whitelisted helper call (builder funnel).
    if flag_locals_enabled() {
        let addrs: [u32; 5] = [
            global_pointers::last_op1 as u32,
            global_pointers::last_result as u32,
            global_pointers::last_op_size as u32,
            global_pointers::flags_changed as u32,
            global_pointers::flags as u32,
        ];
        let mut locals = [(0u8, 0u32); 5];
        for (i, &addr) in addrs.iter().enumerate() {
            builder.load_fixed_i32(addr);
            let l = builder.set_new_local();
            locals[i] = (l.idx(), addr);
            std::mem::forget(l); // whole-module lifetime; freed via free_flag_locals
        }
        builder.flag_locals = Some(locals);
    }

    let exit_label = builder.block_void();
    let exit_with_fault_label = builder.block_void();
    let main_loop_label = builder.loop_void();
    if let Some(compiled_generation) = fastmem_generation {
        builder.load_fixed_i64(global_pointers::fastmem_generation as u32);
        builder.const_i64(compiled_generation as i64);
        builder.ne_i64();
        builder.if_void();
        builder.const_i32(wasm_table_index.to_u16() as i32);
        builder.call_fn1("fastmem_deopt_jit_unit");
        builder.br(exit_label);
        builder.block_end();
    }
    if unsafe { JIT_USE_LOOP_SAFETY } {
        builder.get_local(&instruction_counter);
        builder.const_i32(cpu::LOOP_COUNTER);
        builder.geu_i32();
        if cfg!(feature = "profiler") {
            builder.if_void();
            codegen::gen_debug_track_jit_exit(builder, 0);
            builder.br(exit_label);
            builder.block_end();
        }
        else {
            builder.br_if(exit_label);
        }
    }
    let brtable_default = builder.block_void();

    let ctx = &mut JitContext {
        cpu: &mut cpu,
        builder,
        register_locals: &mut register_locals,
        start_of_current_instruction: 0,
        exit_with_fault_label,
        exit_label,
        current_instruction: Instruction::Other,
        previous_instruction: Instruction::Other,
        fpu_simd_dirty_marked: false,
        elide_current_flags: false,
        instruction_counter,
        fastmem_generation,
        fastmem_writes: fastmem_writes_compile_enabled(state_flags),
        x87_local_cache: std::array::from_fn(|_| None),
        push32_write_cache: None,
        capture_inline_leaf_return_eip: false,
        inline_leaf_return_eip: None,
        x87_cache_kept: false,
    };

    let entry_blocks = {
        let mut nodes = &structure;
        let result;
        loop {
            match &nodes[0] {
                WasmStructure::Dispatcher(e) => {
                    result = e.clone();
                    break;
                },
                WasmStructure::Loop { .. } => {
                    dbg_assert!(false);
                },
                WasmStructure::BasicBlock(_) => {
                    dbg_assert!(false);
                },
                // Note: We could use these blocks as entry points, which will yield
                // more entries for free, but it requires adding those to the dispatcher
                // It's to be investigated if this yields a performance improvement
                // See also the comment at the bottom of this function when creating entry
                // points
                WasmStructure::Block(children) => {
                    nodes = children;
                },
            }
        }
        result
    };

    let mut index_for_addr = HashMap::new();
    for (i, &addr) in entry_blocks.iter().enumerate() {
        dbg_assert!(i < 0x10000);
        index_for_addr.insert(addr, i as u16);
    }
    for b in basic_blocks.values() {
        if !index_for_addr.contains_key(&b.addr) {
            let i = index_for_addr.len();
            dbg_assert!(i < 0x10000);
            index_for_addr.insert(b.addr, i as u16);
        }
    }

    let mut label_for_addr: HashMap<u32, (Label, Option<u16>)> = HashMap::new();

    enum Work {
        WasmStructure(WasmStructure),
        BlockEnd {
            label: Label,
            targets: Vec<u32>,
            olds: HashMap<u32, (Label, Option<u16>)>,
        },
        LoopEnd {
            label: Label,
            entries: Vec<u32>,
            olds: HashMap<u32, (Label, Option<u16>)>,
        },
    }
    let mut work: VecDeque<Work> = structure
        .into_iter()
        .map(|x| Work::WasmStructure(x))
        .collect();

    while let Some(block) = work.pop_front() {
        let next_addr: Option<Vec<u32>> = work.iter().find_map(|x| match x {
            Work::WasmStructure(l) => Some(l.head().collect()),
            _ => None,
        });
        let target_block = &ctx.builder.arg_local_initial_state.unsafe_clone();

        match block {
            Work::WasmStructure(WasmStructure::BasicBlock(addr)) => {
                let block = basic_blocks.get(&addr).unwrap();
                // An empty block is an exit at its own address: nothing to emit
                // before the exit arm below sets EIP.
                if block.number_of_instructions > 0 {
                    jit_generate_basic_block(ctx, block, basic_blocks);
                }

                if let Some(leaf_addr) = block.inline_leaf {
                    let leaf = basic_blocks.get(&leaf_addr).unwrap();
                    let use_return_local = unsafe { JIT_TIER2_LEAF_RETURN_LOCAL };
                    if use_return_local {
                        ctx.capture_inline_leaf_return_eip = true;
                        ctx.inline_leaf_return_eip = None;
                    }
                    jit_generate_basic_block(ctx, leaf, basic_blocks);
                    let return_eip_local = if use_return_local {
                        ctx.capture_inline_leaf_return_eip = false;
                        Some(
                            ctx.inline_leaf_return_eip
                                .take()
                                .expect("fused C3 leaf must capture its return EIP"),
                        )
                    }
                    else {
                        None
                    };

                    // Normal returns take the zero-dispatch direct continuation.
                    // A callee that deliberately changed [esp] still gets the
                    // ordinary AbsoluteEip lookup and module-exit semantics.
                    let return_virt = block.virt_addr & !0xFFF
                        | block.end_addr as i32 & 0xFFF;
                    if let Some(local) = return_eip_local.as_ref() {
                        ctx.builder.get_local(local);
                    }
                    else {
                        codegen::gen_get_eip(ctx.builder);
                    }
                    ctx.builder.const_i32(return_virt);
                    ctx.builder.eq_i32();
                    ctx.builder.if_void();
                    let return_index = *index_for_addr.get(&block.end_addr).unwrap();
                    ctx.builder.const_i32(return_index.into());
                    ctx.builder.set_local(target_block);
                    ctx.builder.br(main_loop_label);
                    ctx.builder.block_end();

                    // The legacy resolver reads instruction_pointer. Only the
                    // unusual mismatching return reaches this store.
                    if let Some(local) = return_eip_local.as_ref() {
                        ctx.builder.get_local(local);
                        ctx.builder.store_aligned_i32(
                            global_pointers::instruction_pointer as u32,
                        );
                    }

                    if unsafe { JIT_INLINE_INTRA_MODULE_DISPATCH } {
                        gen_find_cache_entry_in_page_inline(
                            ctx,
                            wasm_table_index,
                            state_flags,
                        );
                    }
                    else {
                        codegen::gen_get_eip(ctx.builder);
                        ctx.builder.const_i32(wasm_table_index.to_u16() as i32);
                        ctx.builder.const_i32(state_flags.to_u32() as i32);
                        ctx.builder.call_fn3_ret("jit_find_cache_entry_in_page");
                    }
                    ctx.builder.tee_local(target_block);
                    ctx.builder.const_i32(0);
                    ctx.builder.ge_i32();
                    ctx.builder.br_if(main_loop_label);
                    ctx.builder.br(ctx.exit_label);
                    if let Some(local) = return_eip_local {
                        ctx.builder.free_local(local);
                    }
                    continue;
                }

                if block.has_sti {
                    match block.ty {
                        BasicBlockType::ConditionalJump {
                            condition,
                            jump_offset,
                            jump_offset_is_32,
                            ..
                        } => {
                            codegen::gen_set_eip_low_bits(
                                ctx.builder,
                                block.end_addr as i32 & 0xFFF,
                            );
                            codegen::gen_condition_fn(ctx, condition);
                            ctx.builder.if_void();
                            if jump_offset_is_32 {
                                codegen::gen_relative_jump(ctx.builder, jump_offset);
                            }
                            else {
                                codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                            }
                            ctx.builder.block_end();
                        },
                        BasicBlockType::Normal {
                            jump_offset,
                            jump_offset_is_32,
                            ..
                        } => {
                            if jump_offset_is_32 {
                                codegen::gen_set_eip_low_bits_and_jump_rel32(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                    jump_offset,
                                );
                            }
                            else {
                                codegen::gen_set_eip_low_bits(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                );
                                codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                            }
                        },
                        BasicBlockType::Exit => {},
                        BasicBlockType::AbsoluteEip => {},
                    };
                    codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                    // STI forces a module exit (one instruction must run before handle_irqs).
                    // Not a chainable dispatch — count as dynamic.
                    codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_DYNAMIC);
                    codegen::gen_move_registers_from_locals_to_memory(ctx);
                    codegen::gen_fn0_const(ctx.builder, "handle_irqs");
                    codegen::gen_update_instruction_counter(ctx);
                    ctx.builder.return_();
                    continue;
                }

                match &block.ty {
                    BasicBlockType::Exit => {
                        if unsafe { JIT_SYNC_BOUNDARY_CONTINUATION } {
                            if let Some(next_block_addr) = block.sync_boundary_fallthrough {
                                if let Some(&next_index) = index_for_addr.get(&next_block_addr) {
                                    // The boundary helper may have parked/switched the
                                    // thread or delivered an interrupt. Continue only
                                    // when its architectural fallthrough is untouched.
                                    let next_virt = block.virt_addr & !0xFFF
                                        | next_block_addr as i32 & 0xFFF;
                                    codegen::gen_get_eip(ctx.builder);
                                    ctx.builder.const_i32(next_virt);
                                    ctx.builder.eq_i32();

                                    // Unlike ordinary direct edges, a JS/wasm boundary
                                    // CAN lower the live cycle limit. Read the shared
                                    // page rather than jit_cycle_limit_cached, then
                                    // account for this module's pending instructions.
                                    ctx.builder.load_fixed_i32(
                                        std::ptr::addr_of!(hypercall::HYPERCALL_PAGE) as u32,
                                    );
                                    let live_limit = ctx.builder.tee_new_local();
                                    ctx.builder.const_i32(0);
                                    ctx.builder.ne_i32();
                                    ctx.builder.and_i32();
                                    ctx.builder.load_fixed_i32(
                                        global_pointers::instruction_counter as u32,
                                    );
                                    ctx.builder.get_local(&ctx.instruction_counter);
                                    ctx.builder.add_i32();
                                    ctx.builder.load_fixed_i32(
                                        std::ptr::addr_of!(cpu::jit_cycle_start_instruction_counter)
                                            as u32,
                                    );
                                    ctx.builder.sub_i32();
                                    ctx.builder.get_local(&live_limit);
                                    ctx.builder.ltu_i32();
                                    ctx.builder.and_i32();
                                    ctx.builder.load_fixed_u8(global_pointers::in_hlt as u32);
                                    ctx.builder.eqz_i32();
                                    ctx.builder.and_i32();
                                    ctx.builder.if_void();
                                    codegen::gen_dispatch_stat_increment(
                                        ctx.builder,
                                        stat::SYNC_BOUNDARY_CONTINUE,
                                    );
                                    ctx.builder.const_i32(next_index.into());
                                    ctx.builder.set_local(target_block);
                                    ctx.builder.br(main_loop_label);
                                    ctx.builder.block_end();
                                    ctx.builder.free_local(live_limit);
                                    unsafe {
                                        JIT_SYNC_BOUNDARY_CONTINUATION_SITES_COMPILED =
                                            JIT_SYNC_BOUNDARY_CONTINUATION_SITES_COMPILED
                                                .saturating_add(1);
                                    }
                                }
                            }
                        }
                        // Exit this function
                        codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                        codegen::gen_profiler_stat_increment(ctx.builder, stat::DIRECT_EXIT);
                        // Terminating instruction set eip at runtime (ret/int/iret/far jmp)
                        codegen::gen_dispatch_stat_increment(ctx.builder, stat::MODULE_EXIT_DYNAMIC);
                        ctx.builder.br(ctx.exit_label);
                    },
                    BasicBlockType::AbsoluteEip => {
                        // Indirect-target histogram for watched pages.
                        // Records (terminal instruction addr, runtime eip) via an import
                        // into the generated module; emitted only when the page is watched.
                        if trace_profiler::is_page_watched(Page::page_of(block.addr)) {
                            ctx.builder.const_i32(block.last_instruction_addr as i32);
                            codegen::gen_get_eip(ctx.builder);
                            ctx.builder.call_fn2("trace2_record_indirect");
                        }
                        // RET-target speculation: a leaf's RET
                        // whose module-local call sites are known compares the runtime
                        // eip against each return address and re-enters the module
                        // dispatcher directly, skipping the helper call below. Return
                        // addresses were marked_as_entry at their CALLs, so they are
                        // top-dispatcher entries (index < entry_blocks.len()); anything
                        // else is skipped defensively.
                        for &(cand_virt, cand_phys) in &block.ret_speculation {
                            if let Some(&idx) = index_for_addr.get(&cand_phys) {
                                if (idx as usize) < entry_blocks.len() {
                                    codegen::gen_get_eip(ctx.builder);
                                    ctx.builder.const_i32(cand_virt);
                                    ctx.builder.eq_i32();
                                    ctx.builder.if_void();
                                    ctx.builder.const_i32(idx.into());
                                    ctx.builder.set_local(target_block);
                                    ctx.builder.br(main_loop_label);
                                    ctx.builder.block_end();
                                }
                            }
                        }

                        // Check if we can stay in this module, if not exit
                        if unsafe { JIT_INLINE_INTRA_MODULE_DISPATCH } {
                            gen_find_cache_entry_in_page_inline(
                                ctx,
                                wasm_table_index,
                                state_flags,
                            );
                        }
                        else {
                            codegen::gen_get_eip(ctx.builder);
                            ctx.builder.const_i32(wasm_table_index.to_u16() as i32);
                            ctx.builder.const_i32(state_flags.to_u32() as i32);
                            ctx.builder.call_fn3_ret("jit_find_cache_entry_in_page");
                        }
                        ctx.builder.tee_local(target_block);
                        ctx.builder.const_i32(0);
                        ctx.builder.ge_i32();
                        // The branch stays conditional by design: the miss path below is no
                        // longer a plain exit — it attempts a cross-module tail-call (RET
                        // dynamic chaining), which must run between the in-page miss and the
                        // module exit. Folding the miss into the dispatcher br_table (the old
                        // idea here) would lose that attempt for the price of one predictable
                        // branch on the hit path.
                        ctx.builder.br_if(main_loop_label);

                        // RET/indirect dynamic chaining: the in-module
                        // re-dispatch missed, but the runtime eip may hit ANOTHER compiled
                        // module — tail-call straight into it instead of round-tripping
                        // through main_loop. Same flush + packed-slot convention as
                        // gen_chain_or_exit_to_known_successor; on a chain miss we fall
                        // through to the plain module exit (the register flush is idempotent
                        // and the instruction counter was zeroed after flushing, so the
                        // epilogue's second flush/add is harmless).
                        if ret_chaining_enabled() {
                            codegen::gen_move_registers_from_locals_to_memory(ctx);
                            codegen::gen_update_instruction_counter(ctx);
                            ctx.builder.const_i32(0);
                            ctx.builder.set_local(&ctx.instruction_counter);

                            if unsafe { JIT_DYNAMIC_CHAIN_SITE_PIC } {
                                gen_dynamic_chain_site_pic_lookup(ctx, state_flags);
                            }
                            else {
                                ctx.builder.const_i32(state_flags.to_u32() as i32);
                                ctx.builder.call_fn1_ret(
                                    "jit_find_cache_entry_for_dynamic_chaining",
                                );
                            }
                            let packed_target = ctx.builder.tee_new_local();

                            ctx.builder.get_local(&packed_target);
                            ctx.builder.const_i32(0);
                            ctx.builder.ge_i32();
                            ctx.builder.if_void();
                            ctx.builder.get_local(&packed_target);
                            ctx.builder.const_i32(0xFFFF);
                            ctx.builder.and_i32();
                            ctx.builder.get_local(&packed_target);
                            ctx.builder.const_i32(16);
                            ctx.builder.shr_u_i32();
                            ctx.builder.return_call_indirect_fn1();
                            ctx.builder.block_end();
                            ctx.builder.free_local(packed_target);
                        }

                        codegen::gen_debug_track_jit_exit(ctx.builder, block.last_instruction_addr);
                        ctx.builder.br(ctx.exit_label);
                    },
                    &BasicBlockType::Normal {
                        next_block_addr: None,
                        jump_offset,
                        jump_offset_is_32,
                    } => {
                        if jump_offset_is_32 {
                            codegen::gen_set_eip_low_bits_and_jump_rel32(
                                ctx.builder,
                                block.end_addr as i32 & 0xFFF,
                                jump_offset,
                            );
                        }
                        else {
                            codegen::gen_set_eip_low_bits(
                                ctx.builder,
                                block.end_addr as i32 & 0xFFF,
                            );
                            codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                        }

                        codegen::gen_profiler_stat_increment(ctx.builder, stat::DIRECT_EXIT);
                        // Direct unconditional JMP whose target is outside this module —
                        // successor eip is a compile-time constant (jump_offset). Chainable.
                        gen_chain_or_exit_to_known_successor(
                            ctx,
                            state_flags,
                            block.last_instruction_addr,
                        );
                    },
                    &BasicBlockType::Normal {
                        next_block_addr: Some(next_block_addr),
                        jump_offset,
                        jump_offset_is_32,
                    } => {
                        // Unconditional jump to next basic block
                        // - All instructions that don't change eip
                        // - Unconditional jumps

                        if Page::page_of(next_block_addr) != Page::page_of(block.addr) {
                            if jump_offset_is_32 {
                                codegen::gen_set_eip_low_bits_and_jump_rel32(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                    jump_offset,
                                );
                            }
                            else {
                                codegen::gen_set_eip_low_bits(
                                    ctx.builder,
                                    block.end_addr as i32 & 0xFFF,
                                );
                                codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                            }

                            codegen::gen_profiler_stat_increment(
                                ctx.builder,
                                stat::NORMAL_PAGE_CHANGE,
                            );

                            codegen::gen_page_switch_check(
                                ctx,
                                next_block_addr,
                                block.last_instruction_addr,
                            );

                            #[cfg(debug_assertions)]
                            codegen::gen_fn2_const(
                                ctx.builder,
                                "check_page_switch",
                                block.addr,
                                next_block_addr,
                            );
                        }

                        if next_addr
                            .as_ref()
                            .map_or(false, |n| n.contains(&next_block_addr))
                        {
                            // Blocks are consecutive
                            if next_addr.unwrap().len() > 1 {
                                let target_index = *index_for_addr.get(&next_block_addr).unwrap();
                                if cfg!(feature = "profiler") {
                                    ctx.builder.const_i32(target_index.into());
                                    ctx.builder.call_fn1("debug_set_dispatcher_target");
                                }
                                ctx.builder.const_i32(target_index.into());
                                ctx.builder.set_local(target_block);
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_FALLTHRU_WITH_TARGET_BLOCK,
                                );
                            }
                            else {
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_FALLTHRU,
                                );
                            }
                        }
                        else {
                            let &(br, target_index) = label_for_addr.get(&next_block_addr).unwrap();
                            if let Some(target_index) = target_index {
                                if cfg!(feature = "profiler") {
                                    ctx.builder.const_i32(target_index.into());
                                    ctx.builder.call_fn1("debug_set_dispatcher_target");
                                }
                                ctx.builder.const_i32(target_index.into());
                                ctx.builder.set_local(target_block);
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_BRANCH_WITH_TARGET_BLOCK,
                                );
                            }
                            else {
                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::NORMAL_BRANCH,
                                );
                            }
                            ctx.builder.br(br);
                        }
                    },
                    &BasicBlockType::ConditionalJump {
                        next_block_addr,
                        next_block_branch_taken_addr,
                        condition,
                        jump_offset,
                        jump_offset_is_32,
                    } => {
                        // Conditional jump to next basic block
                        // - jnz, jc, loop, jcxz, etc.

                        // Generate:
                        // (1) condition()
                        // (2) br_if()
                        // (3) br()
                        // Except:
                        // If we need to update eip in case (2), it's replaced by if { update_eip(); br() }
                        // If case (3) can fall through to the next basic block, the branch is eliminated
                        // Dispatcher target writes can be generated in either case
                        // Condition may be inverted if it helps generate a fallthrough instead of the second branch

                        codegen::gen_profiler_stat_increment(ctx.builder, stat::CONDITIONAL_JUMP);

                        #[derive(PartialEq)]
                        enum Case {
                            BranchTaken,
                            BranchNotTaken,
                        }

                        let mut handle_case = |case: Case, is_first| {
                            // first case generates condition and *has* to branch away,
                            // second case branches unconditionally or falls through

                            if is_first {
                                if case == Case::BranchNotTaken {
                                    codegen::gen_condition_fn_negated(ctx, condition);
                                }
                                else {
                                    codegen::gen_condition_fn(ctx, condition);
                                }
                            }

                            let next_block_addr = if case == Case::BranchTaken {
                                next_block_branch_taken_addr
                            }
                            else {
                                next_block_addr
                            };

                            if let Some(next_block_addr) = next_block_addr {
                                if Page::page_of(next_block_addr) != Page::page_of(block.addr) {
                                    dbg_assert!(case == Case::BranchTaken); // currently not possible in other case
                                    if is_first {
                                        ctx.builder.if_i32();
                                    }
                                    if jump_offset_is_32 {
                                        codegen::gen_set_eip_low_bits_and_jump_rel32(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                            jump_offset,
                                        );
                                    }
                                    else {
                                        codegen::gen_set_eip_low_bits(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                        );
                                        codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                                    }

                                    codegen::gen_profiler_stat_increment(
                                        ctx.builder,
                                        stat::CONDITIONAL_JUMP_PAGE_CHANGE,
                                    );
                                    codegen::gen_page_switch_check(
                                        ctx,
                                        next_block_addr,
                                        block.last_instruction_addr,
                                    );

                                    #[cfg(debug_assertions)]
                                    codegen::gen_fn2_const(
                                        ctx.builder,
                                        "check_page_switch",
                                        block.addr,
                                        next_block_addr,
                                    );

                                    if is_first {
                                        ctx.builder.const_i32(1);
                                        ctx.builder.else_();
                                        ctx.builder.const_i32(0);
                                        ctx.builder.block_end();
                                    }
                                }

                                if next_addr
                                    .as_ref()
                                    .map_or(false, |n| n.contains(&next_block_addr))
                                {
                                    // blocks are consecutive

                                    // fallthrough, has to be second
                                    dbg_assert!(!is_first);

                                    if next_addr.as_ref().unwrap().len() > 1 {
                                        let target_index =
                                            *index_for_addr.get(&next_block_addr).unwrap();
                                        if cfg!(feature = "profiler") {
                                            ctx.builder.const_i32(target_index.into());
                                            ctx.builder.call_fn1("debug_set_dispatcher_target");
                                        }
                                        ctx.builder.const_i32(target_index.into());
                                        ctx.builder.set_local(target_block);
                                        codegen::gen_profiler_stat_increment(
                                            ctx.builder,
                                            stat::CONDITIONAL_JUMP_FALLTHRU_WITH_TARGET_BLOCK,
                                        );
                                    }
                                    else {
                                        codegen::gen_profiler_stat_increment(
                                            ctx.builder,
                                            stat::CONDITIONAL_JUMP_FALLTHRU,
                                        );
                                    }
                                }
                                else {
                                    let &(br, target_index) =
                                        label_for_addr.get(&next_block_addr).unwrap();
                                    if let Some(target_index) = target_index {
                                        if cfg!(feature = "profiler") {
                                            // Note: Currently called unconditionally, even if the
                                            // br_if below doesn't branch
                                            ctx.builder.const_i32(target_index.into());
                                            ctx.builder.call_fn1("debug_set_dispatcher_target");
                                        }
                                        ctx.builder.const_i32(target_index.into());
                                        ctx.builder.set_local(target_block);
                                    }

                                    if is_first {
                                        if cfg!(feature = "profiler") {
                                            ctx.builder.if_void();
                                            codegen::gen_profiler_stat_increment(
                                                ctx.builder,
                                                if target_index.is_some() {
                                                    stat::CONDITIONAL_JUMP_BRANCH_WITH_TARGET_BLOCK
                                                }
                                                else {
                                                    stat::CONDITIONAL_JUMP_BRANCH
                                                },
                                            );
                                            ctx.builder.br(br);
                                            ctx.builder.block_end();
                                        }
                                        else {
                                            ctx.builder.br_if(br);
                                        }
                                    }
                                    else {
                                        codegen::gen_profiler_stat_increment(
                                            ctx.builder,
                                            if target_index.is_some() {
                                                stat::CONDITIONAL_JUMP_BRANCH_WITH_TARGET_BLOCK
                                            }
                                            else {
                                                stat::CONDITIONAL_JUMP_BRANCH
                                            },
                                        );
                                        ctx.builder.br(br);
                                    }
                                }
                            }
                            else {
                                // target is outside of this module, update eip and exit
                                if is_first {
                                    ctx.builder.if_void();
                                }

                                if case == Case::BranchTaken {
                                    if jump_offset_is_32 {
                                        codegen::gen_set_eip_low_bits_and_jump_rel32(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                            jump_offset,
                                        );
                                    }
                                    else {
                                        codegen::gen_set_eip_low_bits(
                                            ctx.builder,
                                            block.end_addr as i32 & 0xFFF,
                                        );
                                        codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                                    }
                                }
                                else {
                                    codegen::gen_set_eip_low_bits(
                                        ctx.builder,
                                        block.end_addr as i32 & 0xFFF,
                                    );
                                }

                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::CONDITIONAL_JUMP_EXIT,
                                );
                                // Conditional JMP leaving the module — successor eip is a
                                // compile-time constant (taken=jump_offset, not-taken=end_addr). Chainable.
                                gen_chain_or_exit_to_known_successor(
                                    ctx,
                                    state_flags,
                                    block.last_instruction_addr,
                                );

                                if is_first {
                                    ctx.builder.block_end();
                                }
                            }
                        };

                        let branch_taken_is_fallthrough = next_block_branch_taken_addr
                            .map_or(false, |addr| {
                                next_addr.as_ref().map_or(false, |n| n.contains(&addr))
                            });
                        let branch_not_taken_is_fallthrough = next_block_addr
                            .map_or(false, |addr| {
                                next_addr.as_ref().map_or(false, |n| n.contains(&addr))
                            });

                        if branch_not_taken_is_fallthrough && branch_taken_is_fallthrough {
                            let next_block_addr = next_block_addr.unwrap();
                            let next_block_branch_taken_addr =
                                next_block_branch_taken_addr.unwrap();

                            dbg_log!(
                                "Conditional control flow: fallthrough in both cases, page_switch={} next_is_multi={}",
                                Page::page_of(next_block_branch_taken_addr)
                                    != Page::page_of(block.addr),
                                next_addr.as_ref().unwrap().len() > 1,
                            );

                            dbg_assert!(
                                Page::page_of(next_block_addr) == Page::page_of(block.addr)
                            ); // currently not possible

                            if Page::page_of(next_block_branch_taken_addr)
                                != Page::page_of(block.addr)
                            {
                                codegen::gen_condition_fn(ctx, condition);
                                ctx.builder.if_void();

                                if jump_offset_is_32 {
                                    codegen::gen_set_eip_low_bits_and_jump_rel32(
                                        ctx.builder,
                                        block.end_addr as i32 & 0xFFF,
                                        jump_offset,
                                    );
                                }
                                else {
                                    codegen::gen_set_eip_low_bits(
                                        ctx.builder,
                                        block.end_addr as i32 & 0xFFF,
                                    );
                                    codegen::gen_jmp_rel16(ctx.builder, jump_offset as u16);
                                }

                                codegen::gen_profiler_stat_increment(
                                    ctx.builder,
                                    stat::CONDITIONAL_JUMP_PAGE_CHANGE,
                                );
                                codegen::gen_page_switch_check(
                                    ctx,
                                    next_block_branch_taken_addr,
                                    block.last_instruction_addr,
                                );

                                #[cfg(debug_assertions)]
                                codegen::gen_fn2_const(
                                    ctx.builder,
                                    "check_page_switch",
                                    block.addr,
                                    next_block_branch_taken_addr,
                                );

                                dbg_assert!(next_addr.unwrap().len() > 1);

                                let target_index_taken =
                                    *index_for_addr.get(&next_block_branch_taken_addr).unwrap();
                                let target_index_not_taken =
                                    *index_for_addr.get(&next_block_addr).unwrap();

                                ctx.builder.const_i32(target_index_taken.into());
                                ctx.builder.set_local(target_block);

                                ctx.builder.else_();
                                ctx.builder.const_i32(target_index_not_taken.into());
                                ctx.builder.set_local(target_block);

                                ctx.builder.block_end();
                            }
                            else if next_addr.unwrap().len() > 1 {
                                let target_index_taken =
                                    *index_for_addr.get(&next_block_branch_taken_addr).unwrap();
                                let target_index_not_taken =
                                    *index_for_addr.get(&next_block_addr).unwrap();

                                codegen::gen_condition_fn(ctx, condition);
                                ctx.builder.if_i32();
                                ctx.builder.const_i32(target_index_taken.into());
                                ctx.builder.else_();
                                ctx.builder.const_i32(target_index_not_taken.into());
                                ctx.builder.block_end();
                                ctx.builder.set_local(target_block);
                            }
                        }
                        else if branch_taken_is_fallthrough {
                            handle_case(Case::BranchNotTaken, true);
                            handle_case(Case::BranchTaken, false);
                        }
                        else {
                            handle_case(Case::BranchTaken, true);
                            handle_case(Case::BranchNotTaken, false);
                        }
                    },
                }
            },
            Work::WasmStructure(WasmStructure::Dispatcher(entries)) => {
                profiler::stat_increment(stat::COMPILE_DISPATCHER);

                if cfg!(feature = "profiler") {
                    ctx.builder.get_local(target_block);
                    ctx.builder.const_i32(index_for_addr.len() as i32);
                    ctx.builder.call_fn2("check_dispatcher_target");
                }

                if entries.len() > BRTABLE_CUTOFF {
                    // generate a brtable
                    codegen::gen_profiler_stat_increment(ctx.builder, stat::DISPATCHER_LARGE);
                    let mut cases = Vec::new();
                    for &addr in &entries {
                        let &(label, target_index) = label_for_addr.get(&addr).unwrap();
                        let &index = index_for_addr.get(&addr).unwrap();
                        dbg_assert!(target_index.is_none() || target_index == Some(index));
                        while index as usize >= cases.len() {
                            cases.push(brtable_default);
                        }
                        cases[index as usize] = label;
                    }
                    ctx.builder.get_local(target_block);
                    ctx.builder.brtable(brtable_default, &mut cases.iter());
                }
                else {
                    // generate a if target == block.addr then br block.label ...
                    codegen::gen_profiler_stat_increment(ctx.builder, stat::DISPATCHER_SMALL);
                    let nexts: HashSet<u32> = next_addr
                        .as_ref()
                        .map_or(HashSet::new(), |nexts| nexts.iter().copied().collect());
                    for &addr in &entries {
                        if nexts.contains(&addr) {
                            continue;
                        }
                        let index = *index_for_addr.get(&addr).unwrap();
                        let &(label, _) = label_for_addr.get(&addr).unwrap();
                        ctx.builder.get_local(target_block);
                        ctx.builder.const_i32(index.into());
                        ctx.builder.eq_i32();
                        ctx.builder.br_if(label);
                    }
                }
            },
            Work::WasmStructure(WasmStructure::Loop(children)) => {
                profiler::stat_increment(stat::COMPILE_WASM_LOOP);

                let entries: Vec<u32> = children[0].head().collect();
                let label = ctx.builder.loop_void();
                codegen::gen_profiler_stat_increment(ctx.builder, stat::LOOP);

                if entries.len() == 1 {
                    let addr = entries[0];
                    codegen::gen_set_eip_low_bits(ctx.builder, addr as i32 & 0xFFF);
                    profiler::stat_increment(stat::COMPILE_WITH_LOOP_SAFETY);
                    codegen::gen_profiler_stat_increment(ctx.builder, stat::LOOP_SAFETY);
                    if unsafe { JIT_USE_LOOP_SAFETY } {
                        ctx.builder.get_local(&ctx.instruction_counter);
                        ctx.builder.const_i32(cpu::LOOP_COUNTER);
                        ctx.builder.geu_i32();
                        if cfg!(feature = "profiler") {
                            ctx.builder.if_void();
                            codegen::gen_debug_track_jit_exit(ctx.builder, addr);
                            ctx.builder.br(exit_label);
                            ctx.builder.block_end();
                        }
                        else {
                            ctx.builder.br_if(exit_label);
                        }
                    }
                }

                let mut olds = HashMap::new();
                for &target in entries.iter() {
                    let index = if entries.len() == 1 {
                        None
                    }
                    else {
                        Some(*index_for_addr.get(&target).unwrap())
                    };
                    let old = label_for_addr.insert(target, (label, index));
                    if let Some(old) = old {
                        olds.insert(target, old);
                    }
                }

                work.push_front(Work::LoopEnd {
                    label,
                    entries,
                    olds,
                });
                for c in children.into_iter().rev() {
                    work.push_front(Work::WasmStructure(c));
                }
            },
            Work::LoopEnd {
                label,
                entries,
                olds,
            } => {
                for target in entries {
                    let old = label_for_addr.remove(&target);
                    dbg_assert!(old.map(|(l, _)| l) == Some(label));
                }
                for (target, old) in olds {
                    let old = label_for_addr.insert(target, old);
                    dbg_assert!(old.is_none());
                }

                ctx.builder.block_end();
            },
            Work::WasmStructure(WasmStructure::Block(children)) => {
                profiler::stat_increment(stat::COMPILE_WASM_BLOCK);

                let targets = next_addr.clone().unwrap();
                let label = ctx.builder.block_void();
                let mut olds = HashMap::new();
                for &target in targets.iter() {
                    let index = if targets.len() == 1 {
                        None
                    }
                    else {
                        Some(*index_for_addr.get(&target).unwrap())
                    };
                    let old = label_for_addr.insert(target, (label, index));
                    if let Some(old) = old {
                        olds.insert(target, old);
                    }
                }

                work.push_front(Work::BlockEnd {
                    label,
                    targets,
                    olds,
                });
                for c in children.into_iter().rev() {
                    work.push_front(Work::WasmStructure(c));
                }
            },
            Work::BlockEnd {
                label,
                targets,
                olds,
            } => {
                for target in targets {
                    let old = label_for_addr.remove(&target);
                    dbg_assert!(old.map(|(l, _)| l) == Some(label));
                }
                for (target, old) in olds {
                    let old = label_for_addr.insert(target, old);
                    dbg_assert!(old.is_none());
                }

                ctx.builder.block_end();
            },
        }
    }

    dbg_assert!(label_for_addr.is_empty());

    {
        ctx.builder.block_end(); // default case for the brtable
        // A dispatch index with no matching case is a STALE dispatch — a recycled
        // wasm_table_index whose old tlb_code state_table wasn't swept yet, or a
        // racing invalidation mid-slice. Every path that can land here (module
        // entry, in-page re-dispatch, ret-speculation) has already materialized
        // the runtime instruction_pointer, so the recoverable move is a clean
        // module exit: the interpreter re-resolves the eip through the normal
        // cache path (worst case: recompile). `unreachable` turned this race
        // into a fatal wasm trap. Debug builds still flag it via check_dispatcher_target.
        ctx.builder.br(ctx.exit_label);
    }
    {
        ctx.builder.block_end(); // main loop
    }
    {
        // exit-with-fault case
        ctx.builder.block_end();
        codegen::gen_move_registers_from_locals_to_memory(ctx);
        codegen::gen_fn0_const(ctx.builder, "trigger_fault_end_jit");
        codegen::gen_update_instruction_counter(ctx);
        ctx.builder.return_();
    }
    {
        // exit
        ctx.builder.block_end();
        codegen::gen_move_registers_from_locals_to_memory(ctx);
        codegen::gen_update_instruction_counter(ctx);
    }

    for local in ctx.register_locals.drain(..) {
        ctx.builder.free_local(local);
    }
    ctx.builder
        .free_local(ctx.instruction_counter.unsafe_clone());
    ctx.builder.free_flag_locals();

    ctx.builder.finish();

    let entries = Vec::from_iter(entry_blocks.iter().map(|addr| {
        let block = basic_blocks.get(&addr).unwrap();
        let index = *index_for_addr.get(&addr).unwrap();

        profiler::stat_increment(stat::COMPILE_ENTRY_POINT);

        dbg_assert!(block.addr < block.end_addr);
        // Note: We also insert blocks that weren't originally marked as entries here
        //       This doesn't have any downside, besides making the hash table slightly larger

        (block.addr, index)
    }));

    for b in basic_blocks.values() {
        if b.is_entry_block {
            dbg_assert!(entries.iter().find(|(addr, _)| *addr == b.addr).is_some());
        }
    }

    return entries;
}

/// True if the instruction at `eip` is REP MOVSB/MOVSW/MOVSD, after prefixes.
/// The distinction matters only for the optional completed-copy fallthrough;
/// every other string instruction keeps its historical block-exit semantics.
fn opcode_is_rep_movs(eip: u32) -> bool {
    let mut addr = eip;
    let mut saw_rep = false;
    for _ in 0..8 {
        match read_jit_u8(addr) {
            0xF2 | 0xF3 => {
                saw_rep = true;
                addr = addr.wrapping_add(1);
            },
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 => {
                addr = addr.wrapping_add(1);
            },
            0xA4 | 0xA5 => return saw_rep,
            _ => return false,
        }
    }
    false
}

/// True if the instruction at `eip` is an x87 escape, after prefixes.
fn opcode_is_x87(eip: u32) -> bool {
    let mut addr = eip;
    for _ in 0..4 {
        match read_jit_u8(addr) {
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {
                addr = addr.wrapping_add(1);
            },
            op => return (0xD8..=0xDF).contains(&op),
        }
    }
    false
}

/// True only for x87 encodings whose JIT wrappers explicitly preserve the
/// relaxed ST-local cache.  Everything else is flushed before execution when
/// deferred writeback is enabled.  Keeping this list deliberately small makes
/// unknown/rare x87 helpers correct by construction while retaining the hot
/// compiler-generated arithmetic chains.
fn opcode_can_keep_x87_writeback(eip: u32) -> bool {
    let opcode = read_jit_u8(eip);
    // Prefixes are uncommon in the measured hot loops. Conservatively force a
    // flush until their exact operand/address-size variants are covered.
    if !(0xD8..=0xDF).contains(&opcode) {
        return false;
    }
    let modrm = read_jit_u8(eip.wrapping_add(1));
    let mod_bits = modrm >> 6;
    let reg = modrm >> 3 & 7;
    let rm = modrm & 7;
    match opcode {
        // All D8 arithmetic/compare forms use relaxed cache-aware wrappers.
        0xD8 => true,
        // FLD/FST/FSTP m32 and the cache-aware register forms.
        0xD9 => {
            (mod_bits != 3 && matches!(reg, 0 | 2 | 3))
                || (mod_bits == 3 && matches!(reg, 0 | 1 | 3))
                || (mod_bits == 3 && reg == 2 && rm == 0)
                || (mod_bits == 3 && reg == 4 && rm <= 1)
                || (mod_bits == 3 && reg == 5 && rm <= 6)
        },
        // Integer arithmetic/compare memory forms and FUCOMPP.
        0xDA => mod_bits != 3 || (reg == 5 && rm == 1),
        // FILD/FIST m32 and F(U)COMI register forms.
        0xDB => (mod_bits != 3 && matches!(reg, 0 | 1 | 2 | 3))
            || (mod_bits == 3 && matches!(reg, 5 | 6)),
        // All DC arithmetic/compare forms use the same wrappers as D8.
        0xDC => true,
        // FLD/FST/FSTP m64, FXCH/FST(P) and FUCOMP register forms.
        0xDD => (mod_bits != 3 && matches!(reg, 0 | 2 | 3))
            || (mod_bits == 3 && matches!(reg, 1 | 2 | 3 | 5)),
        // Integer memory arithmetic and register arithmetic/compare forms.
        0xDE => true,
        // FIST m16 plus the cache-aware register forms.
        0xDF => (mod_bits != 3 && matches!(reg, 1 | 2 | 3))
            || (mod_bits == 3 && matches!(reg, 1 | 2 | 3 | 5 | 6))
            || (mod_bits == 3 && reg == 4 && rm == 0),
        _ => false,
    }
}

/// True if the instruction at `eip` MAY be an MMX op (0F-escape into the MMX opcode
/// ranges, incl. EMMS 0F 77), after prefixes. MMX registers alias fpu_st storage
/// (get_reg_mmx_offset), so these mutate st memory behind the x87 local cache exactly
/// like raw x87 helpers do — the emission loop must invalidate live slots after them.
/// Deliberately conservative: prefixed SSE forms of the same opcodes (66/F2/F3) match
/// too, costing only a spurious runtime invalidate when x87 slots happen to be live.
fn opcode_is_mmx(eip: u32) -> bool {
    let mut addr = eip;
    for _ in 0..4 {
        match read_jit_u8(addr) {
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {
                addr = addr.wrapping_add(1);
            },
            0x0F => {
                let op = read_jit_u8(addr.wrapping_add(1));
                return (0x60..=0x77).contains(&op)
                    || op == 0x7E
                    || op == 0x7F
                    || (0xD1..=0xFE).contains(&op);
            },
            _ => return false,
        }
    }
    false
}

fn jit_generate_basic_block(
    ctx: &mut JitContext,
    block: &BasicBlock,
    basic_blocks: &HashMap<u32, BasicBlock>,
) {
    let needs_eip_updated = match block.ty {
        BasicBlockType::Exit => true,
        BasicBlockType::Normal { .. }
            if unsafe { JIT_REP_MOVS_REDUCED_SPILL }
                && opcode_is_rep_movs(block.last_instruction_addr) =>
        {
            true
        },
        _ => false,
    };

    profiler::stat_increment(stat::COMPILE_BASIC_BLOCK);

    let start_addr = block.addr;
    let last_instruction_addr = block.last_instruction_addr;
    let stop_addr = block.end_addr;

    // First iteration of do-while assumes the caller confirms this condition
    dbg_assert!(!is_near_end_of_page(start_addr) || page_tail_entries_enabled());

    if cfg!(feature = "profiler") {
        ctx.builder.const_i32(start_addr as i32);
        ctx.builder.call_fn1("enter_basic_block");
    }

    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.const_i32(block.number_of_instructions as i32);
    ctx.builder.add_i32();
    ctx.builder.set_local(&ctx.instruction_counter);

    // Block-chaining: count every basic-block execution. INTRA_MODULE_EDGE is derived as
    // BLOCK_EXECUTION - MODULE_REENTRY in the readout (each module run executes one entry block via
    // dispatch; every further block it runs was reached by an in-module edge).
    codegen::gen_dispatch_stat_increment(ctx.builder, stat::BLOCK_EXECUTION);

    // Tier-2 trace-compiler: per-block exec
    // counter + CFG registration for watched pages. Emits one fixed-address u64 increment;
    // nothing is emitted for unwatched pages (zero cost when profiling is off).
    if trace_profiler::is_enabled() && trace_profiler::is_page_watched(Page::page_of(block.addr)) {
        let (kind, condition, succ_fallthrough, succ_taken) = match block.ty {
            BasicBlockType::Normal { next_block_addr, .. } => {
                (trace_profiler::KIND_NORMAL, 0u8, next_block_addr.unwrap_or(0), 0)
            },
            BasicBlockType::ConditionalJump {
                next_block_addr,
                next_block_branch_taken_addr,
                condition,
                ..
            } => (
                trace_profiler::KIND_CONDITIONAL,
                condition,
                next_block_addr.unwrap_or(0),
                next_block_branch_taken_addr.unwrap_or(0),
            ),
            BasicBlockType::AbsoluteEip => (trace_profiler::KIND_ABSOLUTE_EIP, 0, 0, 0),
            BasicBlockType::Exit => (trace_profiler::KIND_EXIT, 0, 0, 0),
        };
        if let Some(counter_addr) = trace_profiler::register_block(trace_profiler::BlockRecord {
            addr: block.addr,
            last_instruction_addr: block.last_instruction_addr,
            end_addr: block.end_addr,
            kind,
            condition,
            succ_fallthrough,
            succ_taken,
            number_of_instructions: block.number_of_instructions,
            is_entry_block: block.is_entry_block,
            slot: u32::MAX,
        }) {
            ctx.builder.increment_fixed_i64(counter_addr, 1);
        }
    }

    ctx.cpu.eip = start_addr;
    ctx.current_instruction = Instruction::Other;
    ctx.previous_instruction = Instruction::Other;
    ctx.fpu_simd_dirty_marked = false;
    ctx.elide_current_flags = false;

    loop {
        let mut instruction = 0;
        if cfg!(feature = "profiler") {
            instruction = memory::read32s(ctx.cpu.eip) as u32;
            opstats::gen_opstats(ctx.builder, instruction);
            opstats::record_opstat_compiled(instruction);
        }

        if ctx.cpu.eip == last_instruction_addr {
            // Before the last instruction:
            // - Set eip to *after* the instruction
            // - Set previous_eip to *before* the instruction
            if needs_eip_updated {
                codegen::gen_set_previous_eip_offset_from_eip_with_low_bits(
                    ctx.builder,
                    last_instruction_addr as i32 & 0xFFF,
                );
                codegen::gen_set_eip_low_bits(ctx.builder, stop_addr as i32 & 0xFFF);
            }
        }

        let wasm_length_before = ctx.builder.instruction_body_length();

        ctx.start_of_current_instruction = ctx.cpu.eip;
        let start_eip = ctx.cpu.eip;
        ctx.elide_current_flags =
            should_elide_current_flags(&*ctx.cpu, start_eip, block, basic_blocks);
        // Relaxed x87 wrappers set this when they keep the st cache coherent.
        ctx.x87_cache_kept = false;
        if start_eip == block.last_instruction_addr
            && Page::page_of(start_eip) != Page::page_of(stop_addr.wrapping_sub(1))
        {
            codegen::gen_cross_page_instruction_mapping_guard(
                ctx,
                stop_addr as i32 & 0xFFF,
                stop_addr,
            );
        }
        // Unknown x87 helpers and all MMX instructions access the architectural
        // fpu_st array directly. Make deferred values visible before they run;
        // the existing post-instruction invalidation then discards stale locals.
        let x87_writeback_barrier = x87_writeback_enabled()
            && ctx.x87_local_cache.iter().any(|s| s.is_some())
            && ((opcode_is_x87(start_eip) && !opcode_can_keep_x87_writeback(start_eip))
                || opcode_is_mmx(start_eip));
        if x87_writeback_barrier {
            codegen::gen_x87_local_cache_flush_all_runtime(ctx);
        }
        let mut instruction_flags = 0;
        jit_instructions::jit_instruction(ctx, &mut instruction_flags);
        let end_eip = ctx.cpu.eip;

        // Raw x87 helpers mutate TOP/st memory behind the local cache; MMX ops
        // (incl. EMMS) alias the same fpu_st storage and must invalidate too.
        if ctx.x87_local_cache.iter().any(|s| s.is_some())
            && (x87_writeback_barrier
                || (!ctx.x87_cache_kept
                    && (opcode_is_x87(start_eip) || opcode_is_mmx(start_eip))))
        {
            codegen::gen_x87_local_cache_invalidate_all_runtime(ctx);
        }

        let instruction_length = end_eip - start_eip;
        let was_block_boundary = instruction_flags & JIT_INSTR_BLOCK_BOUNDARY_FLAG != 0;

        let wasm_length = ctx.builder.instruction_body_length() - wasm_length_before;
        opstats::record_opstat_size_wasm(instruction, wasm_length as u64);

        dbg_assert!((end_eip == stop_addr) == (start_eip == last_instruction_addr));
        dbg_assert!(instruction_length < MAX_INSTRUCTION_LENGTH);

        let end_addr = ctx.cpu.eip;

        if end_addr == stop_addr {
            dbg_assert!(
                Page::page_of(end_addr) == Page::page_of(start_addr)
                    || unsafe { JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS }
            );
            codegen::gen_x87_local_cache_free_all(ctx);
            codegen::gen_push32_write_cache_free(ctx);
            break;
        }

        if was_block_boundary
            || (!unsafe {
                JIT_EXACT_PAGE_TAIL || JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS
            } && is_near_end_of_page(end_addr))
            || end_addr > stop_addr
        {
            dbg_log!(
                "Overlapping basic blocks start={:x} expected_end={:x} end={:x} was_block_boundary={} near_end_of_page={}",
                start_addr,
                stop_addr,
                end_addr,
                was_block_boundary,
                is_near_end_of_page(end_addr)
            );
            dbg_assert!(false);
            codegen::gen_x87_local_cache_free_all(ctx);
            codegen::gen_push32_write_cache_free(ctx);
            break;
        }

        ctx.previous_instruction = mem::replace(&mut ctx.current_instruction, Instruction::Other);
    }
}

pub fn jit_increase_hotness_and_maybe_compile(
    virt_address: i32,
    phys_address: u32,
    cs_offset: u32,
    state_flags: CachedStateFlags,
    heat: u32,
) {
    if unsafe { JIT_DISABLED } {
        return;
    }

    let mut ctx = get_jit_state();
    let page = Page::page_of(phys_address);
    let already_deferred = unsafe { JIT_DEFERRED_COMPILE_QUEUE }
        && ctx.deferred_compile_pages.contains(&page);
    // Interpreting on a page that already owns a module means compiled code
    // exists here and simply does not cover this entry point. Such a page has
    // already proven itself hot, and recompiling it only widens an existing
    // module, so it does not need the full cold-page ramp. Read before the
    // entry_points borrow below.
    let page_has_module = unsafe { JIT_RECOMPILE_DIVISOR } > 1 && ctx.pages.contains_key(&page);
    let threshold_reached = {
        let threshold = unsafe { JIT_THRESHOLD };
        let page_hotness = ctx.entry_points.entry(page).or_insert_with(|| {
            cpu::tlb_set_has_code(page, true);
            profiler::stat_increment(stat::RUN_INTERPRETED_NEW_PAGE);
            PageHotness { hotness: 0, entry_points: HashSet::new() }
        });

        if !is_near_end_of_page(phys_address) || page_tail_entries_enabled() {
            page_hotness.entry_points.insert(phys_address as u16 & 0xFFF);
        }

        // A queued page keeps learning entry points, but does not repeatedly pay
        // the hotness/cap path while waiting for a compiler slot.
        if already_deferred {
            return;
        }
        page_hotness.hotness = page_hotness.hotness.wrapping_add(heat);
        let effective = if page_has_module {
            (threshold / unsafe { JIT_RECOMPILE_DIVISOR }).max(1)
        }
        else {
            threshold
        };
        page_hotness.hotness >= effective
    };

    let forced = !threshold_reached && hot_profile_force(&mut ctx, page, phys_address);
    if !threshold_reached && !forced {
        return;
    }

    if page_is_compiling(&ctx, page) {
        return;
    }

    let compile_cap_reached =
        ctx.compiling.len() >= unsafe { JIT_MAX_PENDING_COMPILES.max(1) as usize };
    if compile_cap_reached {
        unsafe {
            JIT_COMPILE_CAP_SKIPS = JIT_COMPILE_CAP_SKIPS.wrapping_add(1);
        }
        if unsafe { JIT_DEFERRED_COMPILE_QUEUE }
            && ctx.deferred_compiles.len() < JIT_DEFERRED_COMPILE_QUEUE_CAP
            && ctx.deferred_compile_pages.insert(page)
        {
            ctx.deferred_compiles.push_back(DeferredCompile {
                page,
                virt_address,
                phys_address,
                cs_offset,
                state_flags,
            });
            if let Some(page_hotness) = ctx.entry_points.get_mut(&page) {
                page_hotness.hotness = 0;
            }
            unsafe {
                JIT_COMPILE_DEFERRED_QUEUED = JIT_COMPILE_DEFERRED_QUEUED.wrapping_add(1);
            }
        }
        return;
    }

    // only try generating if we're in the correct address space
    if cpu::translate_address_read_no_side_effects(virt_address) == Ok(phys_address) {
        if let Some(page_hotness) = ctx.entry_points.get_mut(&page) {
            page_hotness.hotness = 0;
        }
        jit_analyze_and_generate(&mut ctx, virt_address, phys_address, cs_offset, state_flags)
    }
    else {
        profiler::stat_increment(stat::COMPILE_WRONG_ADDRESS_SPACE);
    }
}

fn free_wasm_table_index(ctx: &mut JitState, wasm_table_index: WasmTableIndex) {
    if CHECK_JIT_STATE_INVARIANTS {
        dbg_assert!(!ctx.wasm_table_index_free_list.contains(&wasm_table_index));

        dbg_assert!(
            !ctx.compiling.contains_key(&wasm_table_index),
            "Attempt to free wasm table index that is currently being compiled"
        );

        dbg_assert!(!ctx
            .pages
            .values()
            .any(|info| info.wasm_table_index == wasm_table_index));

        dbg_assert!(!ctx
            .pages
            .values()
            .any(|info| info.hidden_wasm_table_indices.contains(&wasm_table_index)));

        for i in 0..unsafe { cpu::valid_tlb_entries_count } {
            let page = unsafe { cpu::valid_tlb_entries[i as usize] };
            let meta = dispatch_meta_get(page as u32);
            dbg_assert!(
                meta == 0 || dispatch_meta_table_index(meta) != wasm_table_index.to_u16()
            );
        }
    }

    // Release-safe double-free guard (see
    // free_wasm_module_forest): a second push would hand the SAME table slot to
    // two future modules — silent cross-module dispatch corruption. The debug
    // assert above screams first; release skips loudly and keeps the state sane.
    if ctx.wasm_table_index_free_list.contains(&wasm_table_index) {
        unsafe { WASM_TABLE_INDEX_DOUBLE_FREE_SKIPPED += 1 };
        dbg_log!(
            "BUG: double-free of wasm table index {} skipped",
            wasm_table_index.to_u16()
        );
        return;
    }

    // Invalidate every exact-dispatch entry owned by this table slot before the
    // slot can be handed to another module. No hash-table scan is required.
    unsafe {
        let slot = wasm_table_index.to_u16() as usize;
        let next = EXACT_DISPATCH_GENERATIONS[slot].wrapping_add(1);
        EXACT_DISPATCH_GENERATIONS[slot] = if next == 0 { 1 } else { next };
    }

    ctx.wasm_table_index_free_list.push(wasm_table_index);

    // This is the ONLY place a table slot is nulled — invalidate the B1b ret-target
    // memo HERE, not in free_wasm_module: codegen_finalize_finished's module-overwrite
    // path frees replaced indices without going through free_wasm_module (that gap was
    // the null-function crash of the first landing — see the RET_CACHE comment). Also
    // reset the tier-2 execution counter for the recycled index (B3).
    ret_cache_invalidate_all_slot_free();
    unsafe {
        let slot = wasm_table_index.to_u16() as usize;
        MODULE_EXEC_COUNTS[slot] = 0;
        if slot < WASM_TABLE_SIZE as usize {
            TIER2_EXIT_TARGETS[slot] = [0; TIER2_PROFILE_TARGETS];
            TIER2_EXIT_COUNTS[slot] = [0; TIER2_PROFILE_TARGETS];
            TIER2_PROFILE_SAMPLES[slot] = 0;
        }
    }

    // It is not strictly necessary to clear the function, but it will fail more predictably if we
    // accidentally use the function and may garbage collect unused modules earlier
    jit_clear_func(wasm_table_index);
}

fn free_wasm_module(ctx: &mut JitState, wasm_table_index: WasmTableIndex) -> Vec<WasmTableIndex> {
    // B1b memo invalidation lives in free_wasm_table_index (reached below), the one
    // true funnel — this function is NOT on every free path (module-overwrite frees
    // bypass it).
    for i in 0..unsafe { cpu::valid_tlb_entries_count } {
        let page = unsafe { cpu::valid_tlb_entries[i as usize] };
        let entry = unsafe { cpu::tlb_data[page as usize] };
        if 0 != entry {
            let tlb_physical_page = Page::of_u32(
                (entry as u32 >> 12 ^ page as u32) - (unsafe { memory::mem8 } as u32 >> 12),
            );
            let meta = dispatch_meta_get(page as u32);
            if meta != 0 && dispatch_meta_table_index(meta) == wasm_table_index.to_u16() {
                dispatch_meta_clear(page as u32);
                if !ctx.entry_points.contains_key(&tlb_physical_page) {
                    // XXX
                    unsafe { cpu::tlb_data[page as usize] &= !cpu::TLB_HAS_CODE };
                }
            }
        }
    }

    let mut hidden_to_free = Vec::new();
    ctx.pages.retain(
        |_,
         PageInfo {
             wasm_table_index: w,
             hidden_wasm_table_indices,
             ..
         }| {
            if *w == wasm_table_index {
                hidden_to_free.extend(hidden_wasm_table_indices.iter().copied());
                false
            }
            else {
                true
            }
        },
    );

    for info in ctx.pages.values_mut() {
        info.hidden_wasm_table_indices
            .retain(|&w| w != wasm_table_index)
    }

    unsafe { FREE_SITE_PAGE_INVALIDATED = FREE_SITE_PAGE_INVALIDATED.wrapping_add(1) };
    free_wasm_table_index(ctx, wasm_table_index);
    hidden_to_free
}

fn free_wasm_module_tree(ctx: &mut JitState, root: WasmTableIndex) {
    free_wasm_module_forest(ctx, vec![root]);
}

/// Free a SET of module roots under ONE `seen` guard. The roots of a dirtied
/// page (primary + its captured hidden list) can reach each other: a sibling
/// page sharing the primary carries the same hidden index in its own list, so
/// the primary's tree walk already frees it. Walking each root with a fresh
/// `seen` set would double-free such shared indices —
/// the free list would then hold the index TWICE, two later modules would be handed the
/// SAME wasm table slot, and dispatch_meta of the first would point into the
/// second: cross-module dispatch corruption = the silent-ExitProcess class
/// (garbage-register #PF / wild EIP into data / stale brtable traps).
fn free_wasm_module_forest(ctx: &mut JitState, roots: Vec<WasmTableIndex>) {
    // Hidden entries cannot be promoted; free them with the removed primary.
    let mut seen = HashSet::new();
    let mut stack = roots;
    while let Some(index) = stack.pop() {
        if !seen.insert(index) {
            continue;
        }
        stack.extend(free_wasm_module(ctx, index));
    }
}

/// Register a write in this page: Delete all present code
fn jit_dirty_page_ctx(ctx: &mut JitState, page: Page) {
    // A deferred record is left in the FIFO as a cheap tombstone and skipped by
    // drain_deferred_compiles; the membership set is the cancellation authority.
    ctx.deferred_compile_pages.remove(&page);
    // A region is valid only for the exact code bytes/modules from which it was
    // learned. Any member page becoming dirty invalidates the complete plan;
    // ordinary hotness can learn a fresh one from the replacement code.
    ctx.tier2_regions
        .retain(|_, region| !region.pages.contains(&page));
    let mut did_have_code = false;

    if let Some(PageInfo {
        wasm_table_index,
        hidden_wasm_table_indices,
        state_flags: _,
        entry_points: _,
    }) = ctx.pages.remove(&page)
    {
        profiler::stat_increment(stat::INVALIDATE_PAGE_HAD_CODE);
        did_have_code = true;

        // ONE forest walk for primary + hidden: the captured hidden list and the
        // primary's tree overlap (see free_wasm_module_forest) — separate walks
        // double-free the shared indices.
        let mut roots = hidden_wasm_table_indices;
        roots.push(wasm_table_index);
        free_wasm_module_forest(ctx, roots);
    }

    if ctx.external_pages.remove(&page).is_some() {
        did_have_code = true;
        unsafe {
            JIT_EXTERNAL_PAGES_REPLACED = JIT_EXTERNAL_PAGES_REPLACED.wrapping_add(1);
        }
    }

    match ctx.entry_points.remove(&page) {
        None => {},
        Some(_) => {
            profiler::stat_increment(stat::INVALIDATE_PAGE_HAD_ENTRY_POINTS);
            did_have_code = true;

            for state in ctx.compiling.values_mut() {
                let touches_page = match state {
                    CompilingPageState::Compiling { pages } => pages.contains_key(&page),
                    CompilingPageState::CompilingWritten => false,
                };
                if touches_page {
                    *state = CompilingPageState::CompilingWritten;
                }
            }
        },
    }

    for state in ctx.compiling.values() {
        if let CompilingPageState::Compiling { pages } = state {
            dbg_assert!(!pages.contains_key(&page));
        }
    }

    check_jit_state_invariants(ctx);

    dbg_assert!(!jit_page_has_code_ctx(ctx, page));

    if did_have_code {
        cpu::tlb_set_has_code(page, false);
        unsafe {
            JIT_PAGE_INVALIDATIONS_WITH_CODE = JIT_PAGE_INVALIDATIONS_WITH_CODE.wrapping_add(1)
        };
    }

    if !did_have_code {
        unsafe {
            JIT_PAGE_INVALIDATIONS_NO_CODE = JIT_PAGE_INVALIDATIONS_NO_CODE.wrapping_add(1)
        };
        profiler::stat_increment(stat::DIRTY_PAGE_DID_NOT_HAVE_CODE);
    }
}

#[no_mangle]
pub fn jit_dirty_cache(start_addr: u32, end_addr: u32) {
    dbg_assert!(start_addr < end_addr);

    let start_page = Page::page_of(start_addr);
    let end_page = Page::page_of(end_addr - 1);

    for page in start_page.to_u32()..end_page.to_u32() + 1 {
        jit_dirty_page_ctx(&mut get_jit_state(), Page::page_of(page << 12));
    }
}

#[no_mangle]
pub fn jit_dirty_page(page: Page) { jit_dirty_page_ctx(&mut get_jit_state(), page) }

#[no_mangle]
pub fn fastmem_deopt_jit_unit(wasm_table_index: u32) {
    let target = WasmTableIndex(wasm_table_index as u16);
    let mut ctx = get_jit_state();

    let present = ctx.pages.values().any(|info| {
        info.wasm_table_index == target || info.hidden_wasm_table_indices.contains(&target)
    });
    if !present {
        return;
    }

    unsafe {
        FASTMEM_DEOPT_RECOMPILES = FASTMEM_DEOPT_RECOMPILES.saturating_add(1);
    }
    free_wasm_module_tree(&mut ctx, target);
}

/// dirty pages in the range of start_addr and end_addr, which must span at most two pages
pub fn jit_dirty_cache_small(start_addr: u32, end_addr: u32) {
    dbg_assert!(start_addr < end_addr);

    let start_page = Page::page_of(start_addr);
    let end_page = Page::page_of(end_addr - 1);

    let mut ctx = get_jit_state();
    jit_dirty_page_ctx(&mut ctx, start_page);

    // Note: This can't happen when paging is enabled, as writes across
    //       boundaries are split up on two pages
    if start_page != end_page {
        dbg_assert!(start_page.to_u32() + 1 == end_page.to_u32());
        jit_dirty_page_ctx(&mut ctx, end_page);
    }
}

#[no_mangle]
pub fn jit_clear_cache_js() { jit_clear_cache(&mut get_jit_state()) }

fn jit_clear_cache(ctx: &mut JitState) {
    ctx.deferred_compiles.clear();
    ctx.deferred_compile_pages.clear();
    let mut pages_with_code = HashSet::new();

    for &p in ctx.entry_points.keys() {
        pages_with_code.insert(p);
    }
    for &p in ctx.pages.keys() {
        pages_with_code.insert(p);
    }

    for page in pages_with_code {
        jit_dirty_page_ctx(ctx, page);
    }

    // Existing exact-chain memos retain their historical clear-time reuse.
    // Dynamic site-PIC slots deliberately remain monotonic: async compilation
    // can still finish after this clear (see its declaration above).
    unsafe { CHAIN_SITE_MEMO_NEXT = 0 };

    // A stale reference bit would protect whichever module is handed that slot
    // next, letting a cold module survive a sweep it never earned.
    unsafe { MODULE_RECENTLY_USED = [false; WASM_TABLE_SIZE as usize] };
}

pub fn jit_page_has_code(page: Page) -> bool { jit_page_has_code_ctx(&mut get_jit_state(), page) }

fn jit_page_has_code_ctx(ctx: &mut JitState, page: Page) -> bool {
    ctx.pages.contains_key(&page) || ctx.entry_points.contains_key(&page) || ctx.external_pages.contains_key(&page)
}

#[no_mangle]
pub fn jit_get_wasm_table_index_free_list_count() -> u32 {
    if cfg!(feature = "profiler") {
        get_jit_state().wasm_table_index_free_list.len() as u32
    }
    else {
        0
    }
}
#[no_mangle]
pub fn jit_get_cache_size() -> u32 {
    if cfg!(feature = "profiler") {
        get_jit_state()
            .pages
            .values()
            .map(|p| p.entry_points.len() as u32)
            .sum()
    }
    else {
        0
    }
}

// Ungated JIT-table diagnostics for slot pressure / hidden-index pile-up.
#[no_mangle]
pub fn jit_debug_free_slots() -> u32 {
    get_jit_state().wasm_table_index_free_list.len() as u32
}
#[no_mangle]
pub fn jit_debug_module_count() -> u32 {
    let ctx = get_jit_state();
    let mut set = HashSet::new();
    for info in ctx.pages.values() {
        set.insert(info.wasm_table_index);
    }
    set.len() as u32
}
#[no_mangle]
pub fn jit_debug_page_count() -> u32 {
    get_jit_state().pages.len() as u32
}
#[no_mangle]
pub fn jit_debug_hidden_count() -> u32 {
    get_jit_state()
        .pages
        .values()
        .map(|p| p.hidden_wasm_table_indices.len() as u32)
        .sum()
}
#[no_mangle]
pub fn jit_debug_max_region_pages() -> u32 {
    let ctx = get_jit_state();
    let mut per_index: HashMap<WasmTableIndex, u32> = HashMap::new();
    for info in ctx.pages.values() {
        *per_index.entry(info.wasm_table_index).or_insert(0) += 1;
    }
    per_index.values().copied().max().unwrap_or(0)
}

#[cfg(feature = "profiler")]
pub fn check_missed_entry_points(phys_address: u32, state_flags: CachedStateFlags) {
    let ctx = get_jit_state();

    if let Some(infos) = ctx.pages.get(&Page::page_of(phys_address)) {
        if infos.state_flags != state_flags {
            return;
        }

        #[allow(static_mut_refs)]
        let last_jump_type = unsafe { cpu::debug_last_jump.name() };
        #[allow(static_mut_refs)]
        let last_jump_addr = unsafe { cpu::debug_last_jump.phys_address() }.unwrap_or(0);
        let last_jump_opcode =
            if last_jump_addr != 0 { memory::read32s(last_jump_addr) } else { 0 };

        let opcode = memory::read32s(phys_address);
        dbg_log!(
            "Compiled exists, but no entry point, \
                 phys_addr={:x} opcode={:02x} {:02x} {:02x} {:02x}. \
                 Last jump at {:x} ({}) opcode={:02x} {:02x} {:02x} {:02x}",
            phys_address,
            opcode & 0xFF,
            opcode >> 8 & 0xFF,
            opcode >> 16 & 0xFF,
            opcode >> 16 & 0xFF,
            last_jump_addr,
            last_jump_type,
            last_jump_opcode & 0xFF,
            last_jump_opcode >> 8 & 0xFF,
            last_jump_opcode >> 16 & 0xFF,
            last_jump_opcode >> 16 & 0xFF,
        );
    }
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn debug_set_dispatcher_target(_target_index: i32) {
    //dbg_log!("About to call dispatcher target_index={}", target_index);
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn check_dispatcher_target(target_index: i32, max: i32) {
    //dbg_log!("Dispatcher called target={}", target_index);
    dbg_assert!(target_index >= 0);
    dbg_assert!(target_index < max);
}

#[no_mangle]
#[cfg(feature = "profiler")]
pub fn enter_basic_block(phys_eip: u32) {
    let eip =
        unsafe { cpu::translate_address_read(*global_pointers::instruction_pointer).unwrap() };
    if Page::page_of(eip) != Page::page_of(phys_eip) {
        dbg_log!(
            "enter basic block failed block=0x{:x} actual eip=0x{:x}",
            phys_eip,
            eip
        );
        panic!();
    }
}

#[no_mangle]
pub unsafe fn set_jit_config(index: u32, value: u32) {
    match index {
        0 => JIT_DISABLED = value != 0,
        1 => MAX_PAGES = value,
        2 => JIT_USE_LOOP_SAFETY = value != 0,
        3 => MAX_EXTRA_BASIC_BLOCKS = value,
        4 => JIT_BLOCK_CHAINING = value != 0,
        5 => JIT_DEAD_FLAG_ELISION = value != 0,
        46 => JIT_DEAD_FLAG_ELISION_ACROSS_FAULTS = value != 0,
        47 => JIT_NARROW_RET_INVALIDATION = value != 0,
        48 => JIT_HOT_PROFILE_MODE = value,
        6 => JIT_INDIRECT_REGIONS = value != 0,
        7 => JIT_INDIRECT_REGION_MIN_SHARE = value,
        8 => JIT_INDIRECT_REGION_MAX_PAGES = value,
        9 => JIT_FASTMEM_READS = value != 0,
        10 => JIT_X87_LOCALS = value != 0,
        11 => JIT_PUSH_RUN_COALESCING = value != 0,
        12 => JIT_RET_CHAINING = value != 0,
        13 => JIT_RET_SPECULATION = value != 0,
        14 => JIT_RET_SPEC_MAX_INSTR = value,
        15 => JIT_TIER2_THRESHOLD = value,
        16 => JIT_TIER2_RET_SPEC_MAX_INSTR = value,
        17 => TIER2_MAX_PAGES = value,
        18 => JIT_FASTMEM_READ_SPLIT = value != 0,
        19 => JIT_FASTMEM_WRITES = value != 0,
        20 => TIER2_PAGE_SET_CAP = value.clamp(1, 4096),
        21 => JIT_FLAG_LOCALS = value != 0,
        22 => JIT_INLINE_INTRA_MODULE_DISPATCH = value != 0,
        23 => JIT_TIER2_REGIONS = value != 0,
        24 => JIT_TIER2_ADAPTIVE = value != 0,
        25 => JIT_MAX_PENDING_COMPILES = value.clamp(1, 8),
        26 => JIT_THRESHOLD = value.clamp(10_000, 2_000_000),
        27 => JIT_TIER2_LEAF_CALL_FUSION = value != 0,
        28 => JIT_TIER2_LEAF_RETURN_LOCAL = value != 0,
        29 => LEAF_CALL_FUSION_MAX_INSTR = value.clamp(1, 64),
        30 => JIT_DYNAMIC_CHAIN_SITE_PIC = value != 0,
        31 => JIT_DYNAMIC_CHAIN_SITE_PIC_DIAG = value != 0,
        32 => JIT_DYNAMIC_CHAIN_SITE_PIC_SECOND_WAY = value != 0,
        33 => JIT_DYNAMIC_CHAIN_SITE_PIC_FOUR_WAY = value != 0,
        34 => JIT_EXACT_PAGE_TAIL = value != 0,
        35 => JIT_REP_MOVS_REDUCED_SPILL = value != 0,
        36 => JIT_SYNC_BOUNDARY_CONTINUATION = value != 0,
        37 => JIT_DEFERRED_COMPILE_QUEUE = value != 0,
        38 => JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS = value != 0,
        50 => JIT_PAGE_TAIL_ENTRIES = value != 0,
        39 => JIT_X87_WRITEBACK = value != 0,
        40 => JIT_FPU_ORDERED_COMPARE_FIRST = value != 0,
        41 => JIT_DYNAMIC_CHAIN_BUDGET_FAST_EXIT = value != 0,
        42 => JIT_RECOMPILE_DIVISOR = value.clamp(1, 64),
        43 => JIT_PARTIAL_EVICTION = (value != 0) as u32,
        44 => JIT_HONOR_URGENT_EXIT_IN_SLICE = (value != 0) as u32,
        45 => JIT_CHAIN_PARK_GUARD = (value != 0) as u32,
        _ => dbg_assert!(false),
    }
}

#[no_mangle]
pub unsafe fn get_jit_config(index: u32) -> u32 {
    match index {
        0 => JIT_DISABLED as u32,
        1 => MAX_PAGES as u32,
        2 => JIT_USE_LOOP_SAFETY as u32,
        3 => MAX_EXTRA_BASIC_BLOCKS as u32,
        4 => JIT_BLOCK_CHAINING as u32,
        5 => JIT_DEAD_FLAG_ELISION as u32,
        46 => JIT_DEAD_FLAG_ELISION_ACROSS_FAULTS as u32,
        47 => JIT_NARROW_RET_INVALIDATION as u32,
        48 => JIT_HOT_PROFILE_MODE,
        6 => JIT_INDIRECT_REGIONS as u32,
        7 => JIT_INDIRECT_REGION_MIN_SHARE,
        8 => JIT_INDIRECT_REGION_MAX_PAGES,
        9 => JIT_FASTMEM_READS as u32,
        10 => JIT_X87_LOCALS as u32,
        11 => JIT_PUSH_RUN_COALESCING as u32,
        12 => JIT_RET_CHAINING as u32,
        13 => JIT_RET_SPECULATION as u32,
        14 => JIT_RET_SPEC_MAX_INSTR,
        15 => JIT_TIER2_THRESHOLD,
        16 => JIT_TIER2_RET_SPEC_MAX_INSTR,
        17 => TIER2_MAX_PAGES,
        18 => JIT_FASTMEM_READ_SPLIT as u32,
        19 => JIT_FASTMEM_WRITES as u32,
        20 => TIER2_PAGE_SET_CAP,
        21 => JIT_FLAG_LOCALS as u32,
        22 => JIT_INLINE_INTRA_MODULE_DISPATCH as u32,
        23 => JIT_TIER2_REGIONS as u32,
        24 => JIT_TIER2_ADAPTIVE as u32,
        25 => JIT_MAX_PENDING_COMPILES,
        26 => JIT_THRESHOLD,
        27 => JIT_TIER2_LEAF_CALL_FUSION as u32,
        28 => JIT_TIER2_LEAF_RETURN_LOCAL as u32,
        29 => LEAF_CALL_FUSION_MAX_INSTR,
        30 => JIT_DYNAMIC_CHAIN_SITE_PIC as u32,
        31 => JIT_DYNAMIC_CHAIN_SITE_PIC_DIAG as u32,
        32 => JIT_DYNAMIC_CHAIN_SITE_PIC_SECOND_WAY as u32,
        33 => JIT_DYNAMIC_CHAIN_SITE_PIC_FOUR_WAY as u32,
        34 => JIT_EXACT_PAGE_TAIL as u32,
        35 => JIT_REP_MOVS_REDUCED_SPILL as u32,
        36 => JIT_SYNC_BOUNDARY_CONTINUATION as u32,
        37 => JIT_DEFERRED_COMPILE_QUEUE as u32,
        38 => JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS as u32,
        50 => JIT_PAGE_TAIL_ENTRIES as u32,
        39 => JIT_X87_WRITEBACK as u32,
        40 => JIT_FPU_ORDERED_COMPARE_FIRST as u32,
        41 => JIT_DYNAMIC_CHAIN_BUDGET_FAST_EXIT as u32,
        42 => JIT_RECOMPILE_DIVISOR,
        43 => JIT_PARTIAL_EVICTION,
        44 => JIT_HONOR_URGENT_EXIT_IN_SLICE,
        45 => JIT_CHAIN_PARK_GUARD,
        _ => 0,
    }
}

#[no_mangle]
pub fn jit_leaf_call_fusion_sites_compiled() -> u32 {
    unsafe { LEAF_CALL_FUSION_SITES_COMPILED }
}

#[no_mangle]
pub fn jit_exact_page_tail_instructions_compiled() -> u32 {
    unsafe { JIT_EXACT_PAGE_TAIL_INSTRUCTIONS_COMPILED }
}

#[no_mangle]
pub fn jit_contiguous_cross_page_instructions_compiled() -> u32 {
    unsafe { JIT_CONTIGUOUS_CROSS_PAGE_INSTRUCTIONS_COMPILED }
}

pub fn x87_writeback_enabled() -> bool {
    // Deferred architectural stores require the block-local x87 cache. Keep
    // config 39 inert when config 10 is disabled instead of emitting fault
    // guards and writeback bookkeeping that can never retain a value.
    unsafe { JIT_X87_WRITEBACK && JIT_X87_LOCALS }
}

pub fn fpu_ordered_compare_first_enabled() -> bool {
    unsafe { JIT_FPU_ORDERED_COMPARE_FIRST }
}

pub fn dynamic_chain_budget_fast_exit_enabled() -> bool {
    unsafe { JIT_DYNAMIC_CHAIN_BUDGET_FAST_EXIT }
}

#[no_mangle]
pub fn jit_sync_boundary_continuation_sites_compiled() -> u32 {
    unsafe { JIT_SYNC_BOUNDARY_CONTINUATION_SITES_COMPILED }
}

#[no_mangle]
pub fn jit_get_compile_started() -> u32 { unsafe { JIT_COMPILE_STARTED } }

#[no_mangle]
pub fn jit_get_compile_completed() -> u32 { unsafe { JIT_COMPILE_COMPLETED } }

#[no_mangle]
pub fn jit_get_compile_cap_skips() -> u32 { unsafe { JIT_COMPILE_CAP_SKIPS } }

#[no_mangle]
pub fn jit_get_compile_deferred_queued() -> u32 { unsafe { JIT_COMPILE_DEFERRED_QUEUED } }

#[no_mangle]
pub fn jit_get_compile_deferred_started() -> u32 { unsafe { JIT_COMPILE_DEFERRED_STARTED } }

#[no_mangle]
pub fn jit_get_compile_deferred_dropped() -> u32 { unsafe { JIT_COMPILE_DEFERRED_DROPPED } }

#[no_mangle]
pub fn jit_get_compile_deferred_pending() -> u32 {
    get_jit_state().deferred_compile_pages.len() as u32
}

#[no_mangle]
pub fn jit_get_compile_pending() -> u32 { get_jit_state().compiling.len() as u32 }

#[no_mangle]
pub fn jit_get_compile_pending_high_water() -> u32 {
    unsafe { JIT_COMPILE_PENDING_HIGH_WATER }
}

#[no_mangle]
pub fn jit_get_compile_total_us() -> f64 { unsafe { JIT_COMPILE_TOTAL_US as f64 } }

#[no_mangle]
pub fn jit_get_compile_max_us() -> u32 { unsafe { JIT_COMPILE_MAX_US } }

#[no_mangle]
pub fn jit_get_codegen_total_us() -> f64 { unsafe { JIT_CODEGEN_TOTAL_US } }

#[no_mangle]
pub fn jit_get_codegen_max_us() -> f64 { unsafe { JIT_CODEGEN_MAX_US } }

#[no_mangle]
pub fn jit_get_codegen_count() -> u32 { unsafe { JIT_CODEGEN_COUNT } }

#[no_mangle]
pub fn jit_get_codegen_bytes() -> f64 { unsafe { JIT_CODEGEN_BYTES_TOTAL as f64 } }

// ── Hot-page profile: import / export / force ────────────────────────────

/// FNV-1a over the page's 4 KiB of guest RAM; None for MMIO or out of range.
fn hot_profile_page_hash(page: Page) -> Option<u32> {
    let addr = page.to_address();
    let end = addr.checked_add(4096)?;
    if end > unsafe { *global_pointers::memory_size } || memory::in_mapped_range(addr) {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(memory::mem8.offset(addr as isize), 4096) };
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    Some(h)
}

/// Compile-at-first-touch for a page the profile knows. Returns true when the
/// page's hotness ramp is skipped; the caller then takes the ordinary
/// cap/deferred/compile path with the recorded entry points merged in.
fn hot_profile_force(ctx: &mut JitState, page: Page, phys_address: u32) -> bool {
    // In flight or queued: nothing to force, and the counter must not tick
    // once per block executed while a compile lands.
    if page_is_compiling(ctx, page) || ctx.deferred_compile_pages.contains(&page) {
        return false;
    }
    if unsafe { JIT_HOT_PROFILE_MODE } == 1
        && ctx.compiling.len() >= unsafe { JIT_MAX_PENDING_COMPILES.max(1) as usize }
    {
        return false;
    }
    let mut guard = HOT_PROFILE.lock().unwrap();
    let map = match guard.as_mut() {
        Some(m) => m,
        None => return false,
    };
    let (hash, entries) = match map.get(&page) {
        Some(p) => (p.hash, p.entries.clone()),
        None => return false,
    };
    // A page that already owns a module only earns a forced recompile when
    // the block being interpreted starts at a recorded entry the module does
    // not cover — the secondary page of a multi-page module has exactly that
    // shape, and would otherwise wait out threshold / JIT_RECOMPILE_DIVISOR.
    if let Some(info) = ctx.pages.get(&page) {
        let off = (phys_address & 0xFFF) as u16;
        if !entries.contains(&off) || info.entry_points.iter().any(|(e, _)| *e == off) {
            return false;
        }
    }
    match hot_profile_page_hash(page) {
        Some(h) if h == hash => {},
        _ => {
            map.remove(&page);
            unsafe {
                JIT_HOT_PROFILE_MISMATCH = JIT_HOT_PROFILE_MISMATCH.wrapping_add(1);
            }
            return false;
        },
    }
    let hot = ctx.entry_points.entry(page).or_insert_with(|| {
        cpu::tlb_set_has_code(page, true);
        PageHotness { hotness: 0, entry_points: HashSet::new() }
    });
    for e in entries {
        if !is_near_end_of_page(page.to_address() | e as u32) || page_tail_entries_enabled() {
            hot.entry_points.insert(e);
        }
    }
    hot.hotness = 0;
    unsafe {
        JIT_HOT_PROFILE_FORCED = JIT_HOT_PROFILE_FORCED.wrapping_add(1);
    }
    true
}

fn hot_profile_parse(data: &[u8]) -> Option<HashMap<Page, HotProfilePage>> {
    let rd = |i: usize| -> Option<u32> {
        data.get(i..i + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    if rd(0)? != HOT_PROFILE_MAGIC || rd(4)? != HOT_PROFILE_VERSION {
        return None;
    }
    let count = rd(8)? as usize;
    let mut map = HashMap::with_capacity(count);
    let mut i = 12;
    for _ in 0..count {
        let page = rd(i)?;
        let hash = rd(i + 4)?;
        let n = rd(i + 8)? as usize;
        i += 12;
        let mut entries = Vec::with_capacity(n);
        for k in 0..n {
            let b = data.get(i + 2 * k..i + 2 * k + 2)?;
            entries.push(u16::from_le_bytes([b[0], b[1]]) & 0xFFF);
        }
        i += (2 * n + 3) & !3;
        map.insert(Page::of_u32(page), HotProfilePage { hash, entries });
    }
    Some(map)
}

#[no_mangle]
pub fn jit_hot_profile_clear() {
    *HOT_PROFILE.lock().unwrap() = None;
    unsafe {
        JIT_HOT_PROFILE_FORCED = 0;
        JIT_HOT_PROFILE_MISMATCH = 0;
    }
}

/// Reserve `len` bytes for an import; JS copies the profile there, then commits.
#[no_mangle]
pub fn jit_hot_profile_io_alloc(len: u32) -> u32 {
    let mut io = HOT_PROFILE_IO.lock().unwrap();
    io.clear();
    io.resize(len as usize, 0);
    io.as_mut_ptr() as u32
}

#[no_mangle]
pub fn jit_hot_profile_io_ptr() -> u32 { HOT_PROFILE_IO.lock().unwrap().as_ptr() as u32 }

/// Merge the staged bytes into the live profile. Returns the page count of the
/// merged profile, or 0 when the bytes are not a HOTP v1 image.
#[no_mangle]
pub fn jit_hot_profile_import_commit(len: u32) -> u32 {
    let parsed = {
        let io = HOT_PROFILE_IO.lock().unwrap();
        match io.get(..len as usize).and_then(hot_profile_parse) {
            Some(p) => p,
            None => return 0,
        }
    };
    let mut guard = HOT_PROFILE.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    for (page, p) in parsed {
        let e = map.entry(page).or_insert_with(|| HotProfilePage { hash: p.hash, entries: Vec::new() });
        e.hash = p.hash;
        for off in p.entries {
            if !e.entries.contains(&off) {
                e.entries.push(off);
            }
        }
    }
    map.len() as u32
}

/// Serialize every page that currently owns a module (or is being compiled),
/// merged with the imported profile, into the IO buffer. Returns its length.
#[no_mangle]
pub fn jit_hot_profile_export_build() -> u32 {
    let ctx = get_jit_state();
    let mut merged: HashMap<Page, HotProfilePage> =
        HOT_PROFILE.lock().unwrap().clone().unwrap_or_default();
    let mut add = |page: Page, entry_points: &mut dyn Iterator<Item = u16>| {
        if let Some(h) = hot_profile_page_hash(page) {
            let e = merged.entry(page).or_insert_with(|| HotProfilePage { hash: h, entries: Vec::new() });
            e.hash = h;
            for off in entry_points {
                if !e.entries.contains(&off) {
                    e.entries.push(off);
                }
            }
        }
    };
    // A page's recorded entries are the union of what its module covers and
    // every block start the interpreter saw on it: the secondary page of a
    // multi-page module has an empty module list but real entries of its own,
    // and those are what the next session's forced recompile needs.
    let mut hot_entries = |page: Page| -> Vec<u16> {
        ctx.entry_points
            .get(&page)
            .map(|h| h.entry_points.iter().copied().collect())
            .unwrap_or_default()
    };
    for (page, info) in ctx.pages.iter() {
        let mut it = info.entry_points.iter().map(|(e, _)| *e).chain(hot_entries(*page));
        add(*page, &mut it);
    }
    for state in ctx.compiling.values() {
        if let CompilingPageState::Compiling { pages } = state {
            for (page, info) in pages.iter() {
                let mut it = info.entry_points.iter().map(|(e, _)| *e).chain(hot_entries(*page));
                add(*page, &mut it);
            }
        }
    }
    let mut out: Vec<u8> = Vec::with_capacity(12 + merged.len() * 64);
    out.extend_from_slice(&HOT_PROFILE_MAGIC.to_le_bytes());
    out.extend_from_slice(&HOT_PROFILE_VERSION.to_le_bytes());
    out.extend_from_slice(&(merged.len() as u32).to_le_bytes());
    let mut pages: Vec<(&Page, &HotProfilePage)> = merged.iter().collect();
    pages.sort_by_key(|(p, _)| p.to_u32());
    for (page, p) in pages {
        out.extend_from_slice(&page.to_u32().to_le_bytes());
        out.extend_from_slice(&p.hash.to_le_bytes());
        out.extend_from_slice(&(p.entries.len() as u32).to_le_bytes());
        for e in &p.entries {
            out.extend_from_slice(&e.to_le_bytes());
        }
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }
    let len = out.len() as u32;
    *HOT_PROFILE_IO.lock().unwrap() = out;
    len
}

// ── External (ahead-of-time) modules ─────────────────────────────────────
// A module compiled outside the JIT, in the same ABI as a JIT module
// (`f(initial_state)`, registers at the fixed global offsets, exit by
// materialising EIP and returning), is placed in the table by JS at one of
// the reserved top indices and registered here for one entry point. From
// then on the dispatcher, page-write invalidation and slot sweeping treat it
// exactly like a compiled page.
pub const EXTERNAL_MODULE_SLOTS: u32 = 4096;

#[no_mangle]
pub fn jit_external_module_first_index() -> u32 { WASM_TABLE_SIZE - EXTERNAL_MODULE_SLOTS }

#[no_mangle]
pub fn jit_external_module_slots() -> u32 { EXTERNAL_MODULE_SLOTS }

/// The dispatcher's current state flags, so an external module can be
/// registered under exactly the key the dispatcher will look up.
#[no_mangle]
pub fn jit_get_current_state_flags() -> u32 { cpu::get_state_flags().to_u32() }

#[no_mangle]
pub fn jit_external_pages_replaced() -> u32 { unsafe { JIT_EXTERNAL_PAGES_REPLACED } }

/// Register `wasm_table_index` (a reserved external slot) as the module for the
/// physical entry `phys_address` under `state_flags`. `initial_state` is the
/// br_table case the module expects for that entry. Returns 1 on success, 0
/// when the index is outside the reserved range or the page is being compiled.
#[no_mangle]
pub fn jit_register_external_module(
    wasm_table_index: u32,
    phys_address: u32,
    raw_state_flags: u32,
    initial_state: u32,
) -> u32 {
    if wasm_table_index < WASM_TABLE_SIZE - EXTERNAL_MODULE_SLOTS || wasm_table_index >= WASM_TABLE_SIZE {
        return 0;
    }
    let index = WasmTableIndex(wasm_table_index as u16);
    let state_flags = CachedStateFlags::of_u32(raw_state_flags);
    let page = Page::page_of(phys_address);
    let mut ctx = get_jit_state();
    let offset = (phys_address & 0xFFF) as u16;
    let mut entry_points = vec![(offset, initial_state as u16)];
    if let Some(old) = ctx.external_pages.remove(&page) {
        if old.wasm_table_index == index {
            for e in old.entry_points {
                if e.0 != offset {
                    entry_points.push(e);
                }
            }
        }
    }
    cpu::tlb_set_has_code(page, true);
    let info = PageInfo {
        wasm_table_index: index,
        state_flags,
        entry_points,
        hidden_wasm_table_indices: Vec::new(),
    };
    publish_external(&info, page);
    ctx.external_pages.insert(page, info);
    1
}

/// Publish an external page module to every virtual page currently mapping it.
fn publish_external(info: &PageInfo, phys_page: Page) {
    for i in 0..unsafe { cpu::valid_tlb_entries_count } {
        let virt_page = unsafe { cpu::valid_tlb_entries[i as usize] };
        let entry = unsafe { cpu::tlb_data[virt_page as usize] };
        if 0 != entry {
            let tlb_physical_page = Page::of_u32(
                (entry as u32 >> 12 ^ virt_page as u32) - (unsafe { memory::mem8 } as u32 >> 12),
            );
            if tlb_physical_page == phys_page {
                dispatch_ext_set(Page::of_u32(virt_page as u32), info.wasm_table_index, &info.entry_points, info.state_flags);
            }
        }
    }
}

#[no_mangle]
pub fn jit_external_pages() -> u32 { get_jit_state().external_pages.len() as u32 }

// Dispatches into external modules, and lookups that found the page's
// external table but no entry for the offset (or another CPU state).
static mut EXTERNAL_DISPATCHES: u32 = 0;
static mut EXTERNAL_MISSES: u32 = 0;
#[inline]
pub fn note_external_dispatch(hit: bool) {
    unsafe {
        if hit { EXTERNAL_DISPATCHES = EXTERNAL_DISPATCHES.wrapping_add(1); }
        else { EXTERNAL_MISSES = EXTERNAL_MISSES.wrapping_add(1); }
    }
}

// Flight recorder of the last external dispatches: entry address, the
// address the module exited at and how many instructions it retired. Cheap
// enough to stay on: external entries are rare compared with block entries.
const EXT_TRACE_LEN: usize = 32;
static mut EXT_TRACE: [[u32; 3]; EXT_TRACE_LEN] = [[0; 3]; EXT_TRACE_LEN];
static mut EXT_TRACE_NEXT: u32 = 0;

pub fn ext_trace_enter(eip: u32) {
    unsafe {
        let i = (EXT_TRACE_NEXT as usize) % EXT_TRACE_LEN;
        EXT_TRACE[i] = [eip, 0, 0];
    }
}

pub fn ext_trace_exit(eip: u32, retired: u32) {
    unsafe {
        let i = (EXT_TRACE_NEXT as usize) % EXT_TRACE_LEN;
        EXT_TRACE[i][1] = eip;
        EXT_TRACE[i][2] = retired;
        EXT_TRACE_NEXT = EXT_TRACE_NEXT.wrapping_add(1);
    }
}

/// Recorder read-out: `slot` counts back from the most recent dispatch
/// (0 = latest); `field` 0 = entry, 1 = exit, 2 = retired, 3 = total count.
#[no_mangle]
pub fn jit_ext_trace(slot: u32, field: u32) -> u32 {
    unsafe {
        if field == 3 { return EXT_TRACE_NEXT; }
        if slot as usize >= EXT_TRACE_LEN || slot >= EXT_TRACE_NEXT { return 0; }
        let i = (EXT_TRACE_NEXT.wrapping_sub(1).wrapping_sub(slot) as usize) % EXT_TRACE_LEN;
        EXT_TRACE[i][(field as usize).min(2)]
    }
}
#[no_mangle]
pub fn jit_external_dispatches() -> u32 { unsafe { EXTERNAL_DISPATCHES } }
#[no_mangle]
pub fn jit_external_misses() -> u32 { unsafe { EXTERNAL_MISSES } }

/// Diagnostic: what the dispatcher would find for `virt_address` right now.
/// Bits: 31 = JIT table has the page, 30 = external table has the page,
/// 29 = external state flags match the CPU's, 15..0 = the external state
/// for that offset (0xFFFF = none).
#[no_mangle]
pub fn jit_debug_dispatch(virt_address: u32) -> u32 {
    let page = virt_address >> 12;
    let meta = dispatch_meta_get(page);
    let meta2 = dispatch_ext_get(page);
    let mut out = 0u32;
    if meta != 0 { out |= 1 << 31; }
    if meta2 != 0 {
        out |= 1 << 30;
        if cpu::get_state_flags().to_u32() == dispatch_meta_state_flags(meta2) { out |= 1 << 29; }
        out |= dispatch_state_lookup(meta2, virt_address) as u32;
    } else {
        out |= 0xFFFF;
    }
    out
}

#[no_mangle]
pub fn jit_hot_profile_pages() -> u32 {
    HOT_PROFILE.lock().unwrap().as_ref().map(|m| m.len() as u32).unwrap_or(0)
}

#[no_mangle]
pub fn jit_hot_profile_forced() -> u32 { unsafe { JIT_HOT_PROFILE_FORCED } }

#[no_mangle]
pub fn jit_hot_profile_mismatches() -> u32 { unsafe { JIT_HOT_PROFILE_MISMATCH } }

#[no_mangle]
pub fn jit_reset_compile_stats() {
    unsafe {
        JIT_COMPILE_STARTED = 0;
        JIT_COMPILE_COMPLETED = 0;
        JIT_COMPILE_CAP_SKIPS = 0;
        JIT_COMPILE_PENDING_HIGH_WATER = get_jit_state().compiling.len() as u32;
        JIT_COMPILE_TOTAL_US = 0;
        JIT_COMPILE_MAX_US = 0;
        JIT_COMPILE_DEFERRED_QUEUED = 0;
        JIT_COMPILE_DEFERRED_STARTED = 0;
        JIT_COMPILE_DEFERRED_DROPPED = 0;
        JIT_CODEGEN_TOTAL_US = 0.0;
        JIT_CODEGEN_MAX_US = 0.0;
        JIT_CODEGEN_COUNT = 0;
        JIT_CODEGEN_BYTES_TOTAL = 0;
        JIT_HOT_PROFILE_FORCED = 0;
        JIT_HOT_PROFILE_MISMATCH = 0;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// JIT cache snapshot for diagnostics (BottleShip dumpHotJitBlocks).
//
// Pattern: JS calls `jit_snapshot_cache()` to take a point-in-time snapshot
// of the current JitState.pages map, sorted by physical page address for
// stable output. Entry fields are then read one at a time via the three
// accessor functions. Kept in a mutable static mirroring the existing
// JIT_DISABLED / MAX_PAGES pattern.
// ──────────────────────────────────────────────────────────────────────────

// (wasm_table_index, phys_page_addr, entry_points_count)
static mut JIT_CACHE_SNAPSHOT: Option<Vec<(u16, u32, u16)>> = None;

#[no_mangle]
pub unsafe fn jit_snapshot_cache() -> u32 {
    let ctx = get_jit_state();
    let mut snapshot: Vec<(u16, u32, u16)> = ctx
        .pages
        .iter()
        .map(|(page, info)| {
            (
                info.wasm_table_index.to_u16(),
                page.to_address(),
                info.entry_points.len() as u16,
            )
        })
        .collect();
    // Sort by physical page address so repeated snapshots give stable indexing.
    snapshot.sort_by_key(|&(_, addr, _)| addr);
    let len = snapshot.len() as u32;
    JIT_CACHE_SNAPSHOT = Some(snapshot);
    len
}

#[no_mangle]
pub unsafe fn jit_snapshot_get_wasm_idx(i: u32) -> u32 {
    if let Some(ref snap) = JIT_CACHE_SNAPSHOT {
        if let Some(&(idx, _, _)) = snap.get(i as usize) {
            return idx as u32;
        }
    }
    0
}

#[no_mangle]
pub unsafe fn jit_snapshot_get_phys_addr(i: u32) -> u32 {
    if let Some(ref snap) = JIT_CACHE_SNAPSHOT {
        if let Some(&(_, addr, _)) = snap.get(i as usize) {
            return addr;
        }
    }
    0
}

#[no_mangle]
pub unsafe fn jit_snapshot_get_entry_count(i: u32) -> u32 {
    if let Some(ref snap) = JIT_CACHE_SNAPSHOT {
        if let Some(&(_, _, count)) = snap.get(i as usize) {
            return count as u32;
        }
    }
    0
}
