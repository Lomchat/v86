#![allow(static_mut_refs)]

// Single-threaded wasm profiler state; all access stays on the CPU worker.
// Tier-2 trace compiler dynamic profiler.
//
// Observation only: NO new execution path. When a page is watched, its Tier-1
// compilations additionally emit
//   - one fixed-address u64 increment at every basic-block entry (exec count), and
//   - a `trace2_record_indirect(from, target)` call before every AbsoluteEip
//     terminal (indirect jmp/call/ret target histogram).
// The static CFG (block boundaries, types, successors) is registered at compile
// time; the TS side (dbg.trace2*) derives edge biases from block exec counts +
// CFG predecessor structure and builds trace candidate descriptors.
//
// Zero cost when disabled: instrumentation is emitted only for watched pages, and
// watching/unwatching dirties the page so Tier-1 recompiles with/without it.
// Slots are keyed by physical block address and survive recompiles/SMC, so counts
// accumulate monotonically until trace2_reset().

use std::collections::HashMap;

use crate::jit;
use crate::page::Page;

/// Discovery histogram: dispatch entries per guest page, with no prior knowledge.
///
/// The trace2 machinery above can only instrument pages named in advance, so it
/// cannot find a hot page nobody has guessed — and a JS-timer sampler cannot
/// either, because a guest that stalls for seconds inside one synchronous slice
/// never lets the event loop run. This counts inside the dispatch loop instead,
/// so a stall is exactly the thing it can see.
///
/// Direct-mapped and lossy on collision: a discovery pass wants the few dominant
/// pages, and the collision count says whether the answer can be trusted.
const HOTPAGE_SLOTS: usize = 16384;
static mut HOTPAGE_ARMED: bool = false;
static mut HOTPAGE_TAG: [u32; HOTPAGE_SLOTS] = [0; HOTPAGE_SLOTS];
static mut HOTPAGE_COUNT: [u64; HOTPAGE_SLOTS] = [0; HOTPAGE_SLOTS];
static mut HOTPAGE_COLLISIONS: u64 = 0;
static mut HOTPAGE_TOTAL: u64 = 0;
static mut HOTPAGE_SNAPSHOT: Vec<(u32, u64)> = Vec::new();

#[inline]
pub fn hotpage_note(eip: u32) {
    unsafe {
        if !HOTPAGE_ARMED {
            return;
        }
        // +1 so tag 0 can mean "slot empty" without losing page 0.
        let tag = (eip >> 12).wrapping_add(1);
        let idx = (tag.wrapping_mul(2654435761) >> 13) as usize & (HOTPAGE_SLOTS - 1);
        HOTPAGE_TOTAL += 1;
        if HOTPAGE_TAG[idx] == tag {
            HOTPAGE_COUNT[idx] += 1;
        }
        else if HOTPAGE_TAG[idx] == 0 {
            HOTPAGE_TAG[idx] = tag;
            HOTPAGE_COUNT[idx] = 1;
        }
        else {
            HOTPAGE_COLLISIONS += 1;
        }
    }
}

#[no_mangle]
pub unsafe fn hotpage_arm(on: u32) {
    HOTPAGE_ARMED = on != 0;
}

#[no_mangle]
pub unsafe fn hotpage_reset() {
    HOTPAGE_TAG = [0; HOTPAGE_SLOTS];
    HOTPAGE_COUNT = [0; HOTPAGE_SLOTS];
    HOTPAGE_COLLISIONS = 0;
    HOTPAGE_TOTAL = 0;
    HOTPAGE_SNAPSHOT.clear();
}

/// Freeze the live table into a descending snapshot; returns its length.
#[no_mangle]
pub unsafe fn hotpage_snapshot() -> u32 {
    let mut v: Vec<(u32, u64)> = Vec::new();
    for i in 0..HOTPAGE_SLOTS {
        if HOTPAGE_TAG[i] != 0 {
            v.push(((HOTPAGE_TAG[i] - 1) << 12, HOTPAGE_COUNT[i]));
        }
    }
    v.sort_by(|a, b| b.1.cmp(&a.1));
    HOTPAGE_SNAPSHOT = v;
    HOTPAGE_SNAPSHOT.len() as u32
}

#[no_mangle]
pub unsafe fn hotpage_addr(i: u32) -> u32 {
    HOTPAGE_SNAPSHOT.get(i as usize).map_or(0, |e| e.0)
}

#[no_mangle]
pub unsafe fn hotpage_count_at(i: u32) -> u32 {
    HOTPAGE_SNAPSHOT
        .get(i as usize)
        .map_or(0, |e| e.1.min(u32::MAX as u64) as u32)
}

#[no_mangle]
pub unsafe fn hotpage_collisions() -> u32 { HOTPAGE_COLLISIONS.min(u32::MAX as u64) as u32 }

#[no_mangle]
pub unsafe fn hotpage_total() -> u32 { HOTPAGE_TOTAL.min(u32::MAX as u64) as u32 }

pub const MAX_WATCHED_PAGES: usize = 64;
pub const MAX_BLOCK_SLOTS: usize = 8192;
pub const MAX_INDIRECT_TARGETS: usize = 8192;

// Block terminal kinds, mirrored by the TS reader (dbg-commands trace2*).
pub const KIND_NORMAL: u8 = 0;
pub const KIND_CONDITIONAL: u8 = 1;
pub const KIND_ABSOLUTE_EIP: u8 = 2;
pub const KIND_EXIT: u8 = 3;

static mut ENABLED: bool = false;
static mut WATCHED_PAGES: Vec<Page> = Vec::new();

// Execution counters incremented from generated code (fixed linear-memory address).
pub static mut BLOCK_EXEC_COUNTS: [u64; MAX_BLOCK_SLOTS] = [0; MAX_BLOCK_SLOTS];

#[derive(Clone, Copy)]
pub struct BlockRecord {
    pub addr: u32,                  // physical block start
    pub last_instruction_addr: u32, // physical address of the terminal instruction
    pub end_addr: u32,
    pub kind: u8,
    pub condition: u8,           // Jcc condition for KIND_CONDITIONAL, else 0
    pub succ_fallthrough: u32,   // 0 = none/unknown (target outside module)
    pub succ_taken: u32,         // 0 = none/unknown
    pub number_of_instructions: u32,
    pub is_entry_block: bool,
    pub slot: u32, // index into BLOCK_EXEC_COUNTS, u32::MAX = unassigned (slot cap hit)
}

struct State {
    slot_for_addr: HashMap<u32, u32>,
    blocks: HashMap<u32, BlockRecord>,
    indirects: HashMap<(u32, u32), u64>,
    indirect_overflow: u64,
    slot_overflow: u64,
    // Snapshots (built on demand, read back index-wise from JS)
    block_snapshot: Vec<(BlockRecord, u64)>,
    indirect_snapshot: Vec<(u32, u32, u64)>,
}

static mut STATE: Option<State> = None;

fn state() -> &'static mut State {
    unsafe {
        if STATE.is_none() {
            STATE = Some(State {
                slot_for_addr: HashMap::new(),
                blocks: HashMap::new(),
                indirects: HashMap::new(),
                indirect_overflow: 0,
                slot_overflow: 0,
                block_snapshot: Vec::new(),
                indirect_snapshot: Vec::new(),
            });
        }
        STATE.as_mut().unwrap()
    }
}

pub fn is_enabled() -> bool { unsafe { ENABLED } }

pub fn is_page_watched(page: Page) -> bool {
    unsafe { ENABLED && WATCHED_PAGES.contains(&page) }
}

/// Assign (or look up) the exec-counter slot for a block and register its CFG
/// record. Called from jit_generate_module for blocks on watched pages.
/// Returns the linear-memory byte offset of the block's u64 counter, or None if
/// the slot table is full.
pub fn register_block(record_in: BlockRecord) -> Option<u32> {
    let s = state();
    let slot = match s.slot_for_addr.get(&record_in.addr) {
        Some(&slot) => slot,
        None => {
            let next = s.slot_for_addr.len();
            if next >= MAX_BLOCK_SLOTS {
                s.slot_overflow += 1;
                return None;
            }
            s.slot_for_addr.insert(record_in.addr, next as u32);
            next as u32
        },
    };
    let mut record = record_in;
    record.slot = slot;
    s.blocks.insert(record.addr, record);
    let addr = unsafe { &raw mut BLOCK_EXEC_COUNTS[slot as usize] } as u32;
    Some(addr)
}

/// Runtime hook, called FROM GENERATED CODE (import into JIT modules) before an
/// AbsoluteEip terminal on a watched page. `target` is the runtime virtual eip.
#[no_mangle]
pub unsafe fn trace2_record_indirect(from: u32, target: u32) {
    let s = state();
    if let Some(hits) = s.indirects.get_mut(&(from, target)) {
        *hits += 1;
        return;
    }
    if s.indirects.len() >= MAX_INDIRECT_TARGETS {
        s.indirect_overflow += 1;
        return;
    }
    s.indirects.insert((from, target), 1);
}

/// Tier-2R region formation: hot indirect targets recorded for the
/// AbsoluteEip terminal at physical `from`. Returns up to `max_k` runtime
/// virtual targets whose per-site share is at least `min_share_percent`,
/// hottest first. Used by jit_find_basic_blocks when JIT_INDIRECT_REGIONS is
/// enabled.
pub fn hot_indirect_targets(from: u32, max_k: usize, min_share_percent: u64) -> Vec<u32> {
    let s = state();
    let mut targets: Vec<(u32, u64)> = s
        .indirects
        .iter()
        .filter(|((f, _), _)| *f == from)
        .map(|((_, t), &hits)| (*t, hits))
        .collect();
    if targets.is_empty() {
        return Vec::new();
    }
    let total: u64 = targets.iter().map(|(_, h)| h).sum();
    targets.sort_by(|a, b| b.1.cmp(&a.1));
    targets
        .into_iter()
        .take(max_k)
        .filter(|(_, hits)| hits * 100 >= total * min_share_percent)
        .map(|(t, _)| t)
        .collect()
}

// Control exports (called from TS via raw wasm exports).

/// Watch the 4K page containing `addr` (physical). Dirties the page so Tier-1
/// recompiles it with instrumentation. Returns 1 on success, 0 if the watch
/// table is full or the page is already watched.
#[no_mangle]
pub unsafe fn trace2_watch_page(addr: u32) -> u32 {
    let page = Page::page_of(addr);
    if WATCHED_PAGES.contains(&page) {
        return 0;
    }
    if WATCHED_PAGES.len() >= MAX_WATCHED_PAGES {
        return 0;
    }
    WATCHED_PAGES.push(page);
    ENABLED = true;
    jit::jit_dirty_page(page);
    1
}

/// Disable recording and clear all state. Dirties previously watched pages so
/// their next compilation drops the instrumentation.
#[no_mangle]
pub unsafe fn trace2_reset() {
    for &page in WATCHED_PAGES.iter() {
        jit::jit_dirty_page(page);
    }
    WATCHED_PAGES.clear();
    ENABLED = false;
    STATE = None;
    for c in BLOCK_EXEC_COUNTS.iter_mut() {
        *c = 0;
    }
}

/// Stop recording and drop instrumentation from watched pages, but KEEP the
/// collected histograms - so a subsequent (uninstrumented) recompile can still
/// consume them for region formation (Tier-2R staged flow:
/// watch -> collect -> unwatch_all -> recompile-with-regions).
#[no_mangle]
pub unsafe fn trace2_unwatch_all() {
    for &page in WATCHED_PAGES.iter() {
        jit::jit_dirty_page(page);
    }
    WATCHED_PAGES.clear();
    ENABLED = false;
}

#[no_mangle]
pub unsafe fn trace2_enabled() -> u32 { ENABLED as u32 }

#[no_mangle]
pub unsafe fn trace2_watched_page_count() -> u32 { WATCHED_PAGES.len() as u32 }

#[no_mangle]
pub unsafe fn trace2_slot_overflow() -> u32 { state().slot_overflow as u32 }

#[no_mangle]
pub unsafe fn trace2_indirect_overflow() -> u32 { state().indirect_overflow as u32 }

// Snapshot exports (index-wise readback, jit_snapshot_cache style).

/// Build the block snapshot (sorted by exec count, desc). Returns entry count.
#[no_mangle]
pub unsafe fn trace2_block_snapshot() -> u32 {
    let s = state();
    let mut v: Vec<(BlockRecord, u64)> = s
        .blocks
        .values()
        .map(|r| {
            let count = if r.slot != u32::MAX && (r.slot as usize) < MAX_BLOCK_SLOTS {
                BLOCK_EXEC_COUNTS[r.slot as usize]
            }
            else {
                0
            };
            (*r, count)
        })
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    s.block_snapshot = v;
    s.block_snapshot.len() as u32
}

#[no_mangle]
pub unsafe fn trace2_block_addr(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.0.addr)
}
#[no_mangle]
pub unsafe fn trace2_block_last_ins_addr(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.0.last_instruction_addr)
}
#[no_mangle]
pub unsafe fn trace2_block_end_addr(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.0.end_addr)
}
#[no_mangle]
pub unsafe fn trace2_block_kind(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.0.kind as u32)
}
#[no_mangle]
pub unsafe fn trace2_block_condition(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.0.condition as u32)
}
#[no_mangle]
pub unsafe fn trace2_block_succ_fallthrough(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.0.succ_fallthrough)
}
#[no_mangle]
pub unsafe fn trace2_block_succ_taken(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.0.succ_taken)
}
#[no_mangle]
pub unsafe fn trace2_block_instructions(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.0.number_of_instructions)
}
#[no_mangle]
pub unsafe fn trace2_block_is_entry(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.0.is_entry_block as u32)
}
/// Exec count, truncated to u32 (4 billion executions per snapshot window is
/// beyond any realistic profiling session).
#[no_mangle]
pub unsafe fn trace2_block_exec(i: u32) -> u32 {
    state().block_snapshot.get(i as usize).map_or(0, |e| e.1.min(u32::MAX as u64) as u32)
}

/// Build the indirect-target snapshot (sorted by hits, desc). Returns count.
#[no_mangle]
pub unsafe fn trace2_indirect_snapshot() -> u32 {
    let s = state();
    let mut v: Vec<(u32, u32, u64)> =
        s.indirects.iter().map(|(&(f, t), &h)| (f, t, h)).collect();
    v.sort_by(|a, b| b.2.cmp(&a.2));
    s.indirect_snapshot = v;
    s.indirect_snapshot.len() as u32
}

#[no_mangle]
pub unsafe fn trace2_indirect_from(i: u32) -> u32 {
    state().indirect_snapshot.get(i as usize).map_or(0, |e| e.0)
}
#[no_mangle]
pub unsafe fn trace2_indirect_target(i: u32) -> u32 {
    state().indirect_snapshot.get(i as usize).map_or(0, |e| e.1)
}
#[no_mangle]
pub unsafe fn trace2_indirect_hits(i: u32) -> u32 {
    state().indirect_snapshot.get(i as usize).map_or(0, |e| e.2.min(u32::MAX as u64) as u32)
}
