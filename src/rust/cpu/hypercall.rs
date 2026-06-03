//! Hypercall page: shared data between WASM and JS for fast thunk dispatch
//! and preemption control.
//!
//! Layout of HYPERCALL_PAGE (8192 bytes):
//!   0x000: cycle_limit          u32 — writable replacement for LOOP_COUNTER
//!   0x004: (reserved)           u32
//!   0x008: hc_enabled           u32 — hypercall master switch
//!   0x00C: hc_call_count        u32 — total WASM-handled calls
//!   0x010: hc_tick_count        u32 — GetTickCount value (written by JS)
//!   0x014: hc_perf_counter_lo   u32 — QPC low word
//!   0x018: hc_perf_counter_hi   u32 — QPC high word
//!   0x01C: hc_perf_freq_lo      u32 — QPF low word (constant)
//!   0x020: hc_perf_freq_hi      u32 — QPF high word (constant)
//!   0x024: hc_last_error        u32 — current thread's LastError
//!   0x028: hc_teb_base          u32 — current thread's TEB guest address
//!   0x02C: hc_insn_at_time_update u32 — instruction_counter snapshot for QPC interpolation
//!   0x030: hc_mips_estimate     u32 — instructions per microsecond
//!   0x034: hc_current_thread_id u32 — current thread ID (for CS ownership)
//!   0x038: hc_pending_wake_count u32 — number of pending wakeAddress calls
//!   0x03C: hc_pending_wake_addrs [u32; 16] — addresses for pending wakeAddress
//!   0x080: hc_cursor_x          i32 — mouse X position (written by JS)
//!   0x084: hc_cursor_y          i32 — mouse Y position (written by JS)
//!   0x088: hc_window_x          i32 — main window X offset (written by JS)
//!   0x08C: hc_window_y          i32 — main window Y offset (written by JS)
//!   0x090: hc_msg_queue_flag    u32 — 1 if message queue non-empty (written by JS)
//!   0x094: hc_peek_starvation_counter u32 — consecutive WASM-handled PeekMessage calls
//!   0x098: hc_peek_starvation_limit   u32 — max consecutive before JS fallthrough
//!   0x09C: hc_has_runnable_peers      u32 — 1 if other threads are READY/RUNNING (written by JS)
//!   0x0A0: hc_sleep_starvation_counter u32 — consecutive WASM-handled Sleep(0) calls with peers
//!   0x0A4: hc_sleep_starvation_limit   u32 — max consecutive before JS fallthrough
//!   0x100: hc_dispatch_table    [u8; 4096] — dispatch_table[functionId] = handler_id
//!   0x1100: hc_fls_allocated    [u8; 129] — FLS slot allocation bitmap (written by JS, slot 0 unused)
//!   0x1184: hc_fls_values       [u32; 129] — FLS slot values (written by JS)
//!   0x1400: hc_slab_base        u32 — heap arena slab start (0=disabled, written by JS)
//!   0x1404: hc_slab_end         u32 — heap arena slab end (exclusive)
//!   0x1408: hc_slab_bump        u32 — current bump pointer
//!   0x140C: hc_slab_generation  u32 — incremented on process reset
//!   0x1410: hc_slab_alloc_count u32 — stats: allocations from slab
//!   0x1414: hc_slab_free_count  u32 — stats: frees returned to slab
//!   0x1418: hc_slab_fallback_count u32 — stats: fallbacks to JS
//!   0x1420: hc_slab_freelist    [u32; 9] — per-bin free list heads (bins: 16..4096)

use std::ptr::{addr_of, addr_of_mut};

use crate::cpu::memory;
use crate::cpu::cpu::{
    read_reg32, safe_read32s, safe_write32, write_reg32, EAX, EDX, ESP,
};
use crate::cpu::fpu::{fpu_get_st0, fpu_get_sti, fpu_pop, fpu_push, fpu_write_st};
use crate::cpu::global_pointers::{fpu_stack_ptr, instruction_counter};
use crate::softfloat::F80;

/// Dedicated page for hypercall shared data + preemption control.
/// Lives in WASM data section, not in CPU state area.
#[no_mangle]
pub static mut HYPERCALL_PAGE: [u8; 8192] = [0u8; 8192];

// Offset constants
const OFF_CYCLE_LIMIT: usize = 0x000;
const OFF_HC_ENABLED: usize = 0x008;
const OFF_HC_CALL_COUNT: usize = 0x00C;
const OFF_HC_TICK_COUNT: usize = 0x010;
const OFF_HC_PERF_COUNTER_LO: usize = 0x014;
const OFF_HC_PERF_COUNTER_HI: usize = 0x018;
const OFF_HC_PERF_FREQ_LO: usize = 0x01C;
const OFF_HC_PERF_FREQ_HI: usize = 0x020;
const OFF_HC_LAST_ERROR: usize = 0x024;
const OFF_HC_TEB_BASE: usize = 0x028;
const OFF_HC_INSN_AT_TIME_UPDATE: usize = 0x02C;
const OFF_HC_MIPS_ESTIMATE: usize = 0x030;
const OFF_HC_CURRENT_THREAD_ID: usize = 0x034;
// (pending wake buffer removed — CS wake now uses LockSemaphore events)
const OFF_HC_CURSOR_X: usize = 0x080;
const OFF_HC_CURSOR_Y: usize = 0x084;
const OFF_HC_WINDOW_X: usize = 0x088;
const OFF_HC_WINDOW_Y: usize = 0x08C;
const OFF_HC_MSG_QUEUE_FLAG: usize = 0x090;
const OFF_HC_PEEK_STARVATION_COUNTER: usize = 0x094;
const OFF_HC_PEEK_STARVATION_LIMIT: usize = 0x098;
const OFF_HC_HAS_RUNNABLE_PEERS: usize = 0x09C;
const OFF_HC_SLEEP_STARVATION_COUNTER: usize = 0x0A0;
const OFF_HC_SLEEP_STARVATION_LIMIT: usize = 0x0A4;
const OFF_HC_DISPATCH_TABLE: usize = 0x100;
const OFF_HC_FLS_ALLOCATED: usize = 0x1100;
const OFF_HC_FLS_VALUES: usize = 0x1184;
const HC_FLS_SLOT_COUNT: usize = 129;

// Arena slab control block (HeapAlloc/HeapFree fast path)
const OFF_HC_SLAB_BASE: usize = 0x1400;
const OFF_HC_SLAB_END: usize = 0x1404;
const OFF_HC_SLAB_BUMP: usize = 0x1408;
#[allow(dead_code)]
const OFF_HC_SLAB_GENERATION: usize = 0x140C;
const OFF_HC_SLAB_ALLOC_COUNT: usize = 0x1410;
const OFF_HC_SLAB_FREE_COUNT: usize = 0x1414;
const OFF_HC_SLAB_FALLBACK_COUNT: usize = 0x1418;
const OFF_HC_SLAB_FREELIST: usize = 0x1420; // 9 × u32

const SLAB_MAGIC: u32 = 0x534C4100; // "SLA\0" with low nibble reserved for bin index
const HEAP_ZERO_MEMORY_FLAG: u32 = 0x08;
const BIN_SIZES: [u32; 9] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

/// Helper: raw pointer to HYPERCALL_PAGE (avoids static_mut_refs warnings).
#[inline(always)]
unsafe fn hp_ptr() -> *const u8 {
    addr_of!(HYPERCALL_PAGE).cast::<u8>()
}

/// Helper: mutable raw pointer to HYPERCALL_PAGE.
#[inline(always)]
unsafe fn hp_mut() -> *mut u8 {
    addr_of_mut!(HYPERCALL_PAGE).cast::<u8>()
}

/// Returns the WASM-linear pointer to HYPERCALL_PAGE for JS to create typed views.
#[no_mangle]
pub fn get_hypercall_page_ptr() -> u32 {
    unsafe { hp_ptr() as u32 }
}

/// Read the cycle limit from shared page. Used by do_many_cycles_native().
#[inline(always)]
pub unsafe fn read_cycle_limit() -> u32 {
    let val = *(hp_ptr().add(OFF_CYCLE_LIMIT) as *const u32);
    if val == 0 {
        // Default: match original LOOP_COUNTER when JS hasn't initialized yet
        100_003
    } else {
        val
    }
}

/// Try to dispatch a hypercall for the given function ID.
/// Returns true if handled in WASM, false to fall through to JS.
pub unsafe fn try_dispatch(function_id: i32) -> bool {
    let page = hp_ptr();

    // Check enabled flag
    if *(page.add(OFF_HC_ENABLED) as *const u32) == 0 {
        return false;
    }

    // Bounds check dispatch table (4096 entries at offset 0x100)
    if function_id <= 0 || function_id >= 4096 {
        return false;
    }

    let handler_id = *page.add(OFF_HC_DISPATCH_TABLE + function_id as usize);
    if handler_id == 0 {
        return false;
    }

    let handled = match handler_id {
        1 => handle_get_tick_count(),
        2 => handle_get_tick_count(), // GetTickCount64 → same 32-bit value
        3 => handle_qpc(),
        4 => handle_qpf(),
        5 => handle_get_last_error(),
        6 => handle_set_last_error(),
        7 => handle_interlocked_inc(),
        8 => handle_interlocked_dec(),
        9 => handle_interlocked_xchg(),
        10 => handle_interlocked_cmp_xchg(),
        11 => handle_enter_critical_section(),
        12 => handle_leave_critical_section(),
        13 => handle_is_iconic(),
        14 => handle_screen_to_client(),
        15 => handle_get_cursor_pos(),
        16 => handle_peek_message(),
        // Math/FPU hypercalls (Tier 2)
        17 => handle_ftol(),
        18 => handle_ci_sin(),
        19 => handle_ci_cos(),
        20 => handle_ci_tan(),
        21 => handle_ci_sqrt(),
        22 => handle_ci_log(),
        23 => handle_ci_exp(),
        24 => handle_ci_acos(),
        25 => handle_ci_asin(),
        26 => handle_ci_log10(),
        27 => handle_ci_atan2(),
        28 => handle_ci_fmod(),
        29 => handle_ci_pow(),
        30 => handle_cdecl_sin(),
        31 => handle_cdecl_cos(),
        32 => handle_cdecl_tan(),
        33 => handle_cdecl_sqrt(),
        34 => handle_cdecl_log(),
        35 => handle_cdecl_exp(),
        36 => handle_cdecl_acos(),
        37 => handle_cdecl_asin(),
        38 => handle_cdecl_log10(),
        39 => handle_cdecl_atan(),
        40 => handle_cdecl_fabs(),
        41 => handle_cdecl_atan2(),
        42 => handle_cdecl_fmod(),
        43 => handle_cdecl_pow(),
        44 => handle_cdecl_ceil(),
        45 => handle_cdecl_floor(),
        // String/memory hypercalls (Tier 3)
        51 => handle_wcslen(),
        52 => handle_wcscpy(),
        53 => handle_wcscat(),
        54 => handle_wcsicmp(),
        55 => handle_wcschr(),
        56 => handle_memcpy(),
        57 => handle_memset(),
        58 => handle_strlen(),
        59 => handle_strcmp(),
        60 => handle_strcpy(),
        61 => handle_stricmp(),
        62 => handle_memcmp(),
        // Scheduler hypercalls (Tier 4)
        63 => handle_sleep(),
        64 => handle_tls_get_value(),
        69 => handle_fls_get_value(),
        // Heap arena hypercalls (Tier 5)
        70 => handle_heap_alloc(),
        71 => handle_heap_free(),
        _ => false,
    };

    if handled {
        let count_ptr = hp_mut().add(OFF_HC_CALL_COUNT) as *mut u32;
        *count_ptr = (*count_ptr).wrapping_add(1);
    }
    handled
}

// ---------------------------------------------------------------------------
// Handler implementations
// ---------------------------------------------------------------------------

/// GetTickCount / GetTickCount64 — interpolated tick from shared page
///
/// Without interpolation, all GetTickCount calls within a single tick
/// (up to 16ms in single-thread mode) return the SAME value, causing
/// spin-waits to burn through entire quantums and timing code to see
/// 0ms deltas. We interpolate using the instruction counter delta and
/// MIPS estimate, same approach as QPC.
unsafe fn handle_get_tick_count() -> bool {
    let page = hp_ptr();
    let base_tick = *(page.add(OFF_HC_TICK_COUNT) as *const u32);
    let insn_at_update = *(page.add(OFF_HC_INSN_AT_TIME_UPDATE) as *const u32);
    let mips_est = *(page.add(OFF_HC_MIPS_ESTIMATE) as *const u32);

    // Interpolate: delta_ms = instructions_since_update / (insns_per_us * 1000)
    // mips_est = instructions/microsecond, so mips_est * 1000 = instructions/millisecond
    let delta_ms = if mips_est > 0 {
        let delta_insn = (*instruction_counter).wrapping_sub(insn_at_update);
        delta_insn / (mips_est * 1000)
    } else {
        0
    };

    write_reg32(EAX, base_tick.wrapping_add(delta_ms) as i32);
    true
}

/// QueryPerformanceCounter — interpolates between JS updates using instruction_counter
unsafe fn handle_qpc() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }

    let page = hp_ptr();
    let base_lo = *(page.add(OFF_HC_PERF_COUNTER_LO) as *const u32);
    let base_hi = *(page.add(OFF_HC_PERF_COUNTER_HI) as *const u32);
    let insn_at_update = *(page.add(OFF_HC_INSN_AT_TIME_UPDATE) as *const u32);
    let mips_est = *(page.add(OFF_HC_MIPS_ESTIMATE) as *const u32);

    // Interpolate: delta_micros ≈ (insn_now - insn_at_update) / mips_estimate
    let delta_insn = (*instruction_counter).wrapping_sub(insn_at_update);
    let delta_micros = if mips_est > 0 {
        delta_insn / mips_est
    } else {
        0
    };

    let qpc = (base_lo as u64 | (base_hi as u64) << 32) + delta_micros as u64;

    if safe_write32(ptr as i32, qpc as i32).is_err() {
        return false;
    }
    if safe_write32((ptr + 4) as i32, (qpc >> 32) as i32).is_err() {
        return false;
    }

    write_reg32(EAX, 1);
    true
}

/// QueryPerformanceFrequency — read constant from shared page
unsafe fn handle_qpf() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }

    let page = hp_ptr();
    let freq_lo = *(page.add(OFF_HC_PERF_FREQ_LO) as *const u32);
    let freq_hi = *(page.add(OFF_HC_PERF_FREQ_HI) as *const u32);

    if safe_write32(ptr as i32, freq_lo as i32).is_err() {
        return false;
    }
    if safe_write32((ptr + 4) as i32, freq_hi as i32).is_err() {
        return false;
    }

    write_reg32(EAX, 1);
    true
}

/// GetLastError — read from shared page
unsafe fn handle_get_last_error() -> bool {
    let err = *(hp_ptr().add(OFF_HC_LAST_ERROR) as *const u32);
    write_reg32(EAX, err as i32);
    true
}

/// SetLastError — write to shared page AND guest TEB
unsafe fn handle_set_last_error() -> bool {
    let esp = read_reg32(ESP);
    let error_code = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    // Update shared area
    *(hp_mut().add(OFF_HC_LAST_ERROR) as *mut u32) = error_code;

    // Also write to guest TEB at offset 0x34 if TEB address is known
    let teb_base = *(hp_ptr().add(OFF_HC_TEB_BASE) as *const u32);
    if teb_base != 0 {
        let _ = safe_write32((teb_base + 0x34) as i32, error_code as i32);
    }

    write_reg32(EAX, 0); // SetLastError returns void, EAX convention
    true
}

/// InterlockedIncrement — atomic in single-threaded WASM
unsafe fn handle_interlocked_inc() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let old = match safe_read32s(ptr) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let new_val = old.wrapping_add(1);
    if safe_write32(ptr, new_val).is_err() {
        return false;
    }
    write_reg32(EAX, new_val);
    true
}

/// InterlockedDecrement — atomic in single-threaded WASM
unsafe fn handle_interlocked_dec() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let old = match safe_read32s(ptr) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let new_val = old.wrapping_sub(1);
    if safe_write32(ptr, new_val).is_err() {
        return false;
    }
    write_reg32(EAX, new_val);
    true
}

/// InterlockedExchange(ptr, value) → old value
unsafe fn handle_interlocked_xchg() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let new_value = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let old = match safe_read32s(ptr) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if safe_write32(ptr, new_value).is_err() {
        return false;
    }
    write_reg32(EAX, old);
    true
}

/// InterlockedCompareExchange(ptr, exchange, comparand) → old value
unsafe fn handle_interlocked_cmp_xchg() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let exchange = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let comparand = match safe_read32s(esp + 12) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let current = match safe_read32s(ptr) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if current == comparand {
        if safe_write32(ptr, exchange).is_err() {
            return false;
        }
    }
    write_reg32(EAX, current);
    true
}

/// EnterCriticalSection — uncontended fast path only.
/// CRITICAL_SECTION layout (24 bytes):
///   +0:  DebugInfo        (ptr, ignored)
///   +4:  LockCount        (i32)
///   +8:  RecursionCount   (i32)
///   +12: OwningThread     (u32, thread ID)
///   +16: LockSemaphore    (handle, ignored)
///   +20: SpinCount        (u32, ignored)
///
/// Returns false (→ JS fallback) if contended.
unsafe fn handle_enter_critical_section() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if ptr == 0 { return false; }

    let current_thread = *(hp_ptr().add(OFF_HC_CURRENT_THREAD_ID) as *const u32);
    if current_thread == 0 { return false; } // no thread info → JS fallback

    // Read OwningThread at offset 12
    let owner = match safe_read32s(ptr + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    if owner == 0 {
        // FREE — acquire it
        if safe_write32(ptr + 4, 0).is_err() { return false; }         // LockCount = 0
        if safe_write32(ptr + 8, 1).is_err() { return false; }         // RecursionCount = 1
        if safe_write32(ptr + 12, current_thread as i32).is_err() { return false; } // OwningThread
        write_reg32(EAX, 0);
        return true;
    }

    if owner == current_thread {
        // RECURSIVE — increment RecursionCount
        let rec = match safe_read32s(ptr + 8) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if safe_write32(ptr + 8, rec + 1).is_err() { return false; }
        write_reg32(EAX, 0);
        return true;
    }

    // CONTENDED — fall through to JS (needs scheduler wait)
    false
}

/// LeaveCriticalSection — handles both recursive and full release.
/// If LockSemaphore (CS+16) != 0, returns false so JS handles release + SetEvent.
/// If LockSemaphore == 0 (no waiters ever), releases entirely in WASM (zero JS overhead).
///
/// CRITICAL: Must verify OwningThread matches current thread. Without this check,
/// a non-owner thread calling LeaveCS (game bug or race) silently releases the CS,
/// allowing another thread to acquire it — corrupting shared state and causing
/// downstream crashes (ESP corruption → infinite #PF).
unsafe fn handle_leave_critical_section() -> bool {
    let esp = read_reg32(ESP);
    let ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if ptr == 0 { return false; }

    // Verify ownership: only the owning thread may release.
    // Non-owner releases fall to JS for proper error reporting (FATAL_GUARD 0x3009).
    let current_thread = *(hp_ptr().add(OFF_HC_CURRENT_THREAD_ID) as *const u32);
    let owner = match safe_read32s(ptr + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if current_thread == 0 || (owner != 0 && owner != current_thread) {
        return false; // Non-owner or unknown thread → JS slow path handles it
    }

    // Read RecursionCount at offset 8
    let rec = match safe_read32s(ptr + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };

    if rec > 1 {
        // Still recursive — just decrement, no wake needed
        if safe_write32(ptr + 8, rec - 1).is_err() { return false; }
        write_reg32(EAX, 0);
        return true;
    }

    // Check LockSemaphore at offset 16 BEFORE releasing.
    // If event handle exists, waiters may be present — fall to JS for SetEvent.
    let lock_sem = match safe_read32s(ptr + 16) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if lock_sem != 0 {
        // JS handles: release CS fields + SetEvent(lockSem) + ownership transfer
        return false;
    }

    // No LockSemaphore — no waiter has ever contended. Release fully in WASM.
    if safe_write32(ptr + 4, -1).is_err() { return false; }  // LockCount = -1 (free)
    if safe_write32(ptr + 8, 0).is_err() { return false; }   // RecursionCount = 0
    if safe_write32(ptr + 12, 0).is_err() { return false; }  // OwningThread = 0

    write_reg32(EAX, 0);
    true
}

/// IsIconic — always returns FALSE (we never minimize windows)
unsafe fn handle_is_iconic() -> bool {
    // Argument: HWND at [ESP+4], ignored — always return FALSE
    write_reg32(EAX, 0);
    true
}

/// ScreenToClient — subtract window offset from POINT
unsafe fn handle_screen_to_client() -> bool {
    let esp = read_reg32(ESP);
    // Args: HWND at [ESP+4] (ignored), lpPoint at [ESP+8]
    let lp_point = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if lp_point == 0 { return false; }

    let page = hp_ptr();
    let win_x = *(page.add(OFF_HC_WINDOW_X) as *const i32);
    let win_y = *(page.add(OFF_HC_WINDOW_Y) as *const i32);

    let sx = match safe_read32s(lp_point) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let sy = match safe_read32s(lp_point + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };

    if safe_write32(lp_point, sx - win_x).is_err() { return false; }
    if safe_write32(lp_point + 4, sy - win_y).is_err() { return false; }

    write_reg32(EAX, 1);
    true
}

/// PeekMessage — empty-queue fast path.
/// Returns FALSE (0) immediately when the queue is empty and starvation
/// counter hasn't reached the limit. Falls through to JS otherwise so
/// that onThunkComplete() and the scheduler still get to run periodically.
unsafe fn handle_peek_message() -> bool {
    let page = hp_ptr();
    let counter_ptr = hp_mut().add(OFF_HC_PEEK_STARVATION_COUNTER) as *mut u32;
    let counter = *counter_ptr;
    let limit = *(page.add(OFF_HC_PEEK_STARVATION_LIMIT) as *const u32);

    // Uninitialized or starvation limit hit → fall through to JS
    if limit == 0 || counter >= limit {
        *counter_ptr = 0;
        return false;
    }

    // Queue has messages → fall through to JS for full processing
    if *(page.add(OFF_HC_MSG_QUEUE_FLAG) as *const u32) != 0 {
        *counter_ptr = 0;
        return false;
    }

    // Queue empty → return FALSE immediately, no JS transition
    *counter_ptr = counter + 1;
    write_reg32(EAX, 0);
    true
}

/// GetCursorPos — write cached cursor position to POINT
unsafe fn handle_get_cursor_pos() -> bool {
    let esp = read_reg32(ESP);
    let lp_point = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if lp_point == 0 { return false; }

    let page = hp_ptr();
    let cx = *(page.add(OFF_HC_CURSOR_X) as *const i32);
    let cy = *(page.add(OFF_HC_CURSOR_Y) as *const i32);

    if safe_write32(lp_point, cx).is_err() { return false; }
    if safe_write32(lp_point + 4, cy).is_err() { return false; }

    write_reg32(EAX, 1);
    true
}

// ---------------------------------------------------------------------------
// Tier 2: Math/FPU hypercall handlers
// ---------------------------------------------------------------------------

/// Helper: set ST(0) in-place without push/pop
#[inline(always)]
unsafe fn fpu_set_st0(value: F80) {
    fpu_write_st(*fpu_stack_ptr as i32, value);
}

/// Helper: read an f64 from the x86 stack at the given address
#[inline(always)]
unsafe fn read_stack_f64(addr: i32) -> f64 {
    let lo = match safe_read32s(addr) {
        Ok(v) => v as u32,
        Err(_) => return 0.0,
    };
    let hi = match safe_read32s(addr + 4) {
        Ok(v) => v as u32,
        Err(_) => return 0.0,
    };
    f64::from_bits((hi as u64) << 32 | lo as u64)
}

/// _ftol / __ftol: convert ST(0) to a signed __int64 (truncate toward zero), pop the stack,
/// return the FULL 64-bit result in EDX:EAX. The caller uses EAX alone for `(long)` or EDX:EAX
/// for `__int64`. The old code returned only a 32-bit value CLAMPED to INT32_MAX/MIN — that
/// broke every 64-bit user, notably the UE1 (Harry Potter) launcher's RDTSC timebase:
/// `now = _ftol(GSecondsPerCycle * TSC * 2^32)` exceeds INT32_MAX within ~0.5s of uptime, so the
/// clamp pinned `now` to 0x7fffffff every frame -> per-frame DeltaTime = (now-last)/2^32 = 0 ->
/// frozen splash countdown / racing storybook / audio underrun (all one defect). Must be int64.
unsafe fn handle_ftol() -> bool {
    let st0 = fpu_get_st0();
    let val = f64::from_bits(st0.to_f64());
    fpu_pop();
    let v = val as i64; // Rust float->int `as` truncates toward zero and saturates on overflow
    write_reg32(EAX, v as i32);          // low 32
    write_reg32(EDX, (v >> 32) as i32);  // high 32
    true
}

// --- _CI* intrinsics: read ST(0)/ST(1), operate, write result back ---

/// _CIsin: ST(0) = sin(ST(0))
unsafe fn handle_ci_sin() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.sin().to_bits()));
    true
}

/// _CIcos: ST(0) = cos(ST(0))
unsafe fn handle_ci_cos() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.cos().to_bits()));
    true
}

/// _CItan: ST(0) = tan(ST(0))
unsafe fn handle_ci_tan() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.tan().to_bits()));
    true
}

/// _CIsqrt: ST(0) = sqrt(ST(0))
unsafe fn handle_ci_sqrt() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.sqrt().to_bits()));
    true
}

/// _CIlog: ST(0) = ln(ST(0))
unsafe fn handle_ci_log() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.ln().to_bits()));
    true
}

/// _CIexp: ST(0) = exp(ST(0))
unsafe fn handle_ci_exp() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.exp().to_bits()));
    true
}

/// _CIacos: ST(0) = acos(ST(0))
unsafe fn handle_ci_acos() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.acos().to_bits()));
    true
}

/// _CIasin: ST(0) = asin(ST(0))
unsafe fn handle_ci_asin() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.asin().to_bits()));
    true
}

/// _CIlog10: ST(0) = log10(ST(0))
unsafe fn handle_ci_log10() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    fpu_set_st0(F80::of_f64(x.log10().to_bits()));
    true
}

/// _CIatan2: atan2(y=ST(1), x=ST(0)), pop ST(0), write result to new ST(0)
unsafe fn handle_ci_atan2() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    let y = f64::from_bits(fpu_get_sti(1).to_f64());
    let result = y.atan2(x);
    fpu_pop();
    fpu_set_st0(F80::of_f64(result.to_bits()));
    true
}

/// _CIfmod: fmod(y=ST(1), x=ST(0)), pop ST(0), write result to new ST(0)
unsafe fn handle_ci_fmod() -> bool {
    let x = f64::from_bits(fpu_get_st0().to_f64());
    let y = f64::from_bits(fpu_get_sti(1).to_f64());
    let result = if x != 0.0 { y % x } else { 0.0 };
    fpu_pop();
    fpu_set_st0(F80::of_f64(result.to_bits()));
    true
}

/// _CIpow: pow(base=ST(1), exp=ST(0)), pop ST(0), write result to new ST(0)
unsafe fn handle_ci_pow() -> bool {
    let exponent = f64::from_bits(fpu_get_st0().to_f64());
    let base = f64::from_bits(fpu_get_sti(1).to_f64());
    let result = base.powf(exponent);
    fpu_pop();
    fpu_set_st0(F80::of_f64(result.to_bits()));
    true
}

// --- cdecl math functions: read double arg from stack, push result to FPU ---

/// cdecl sin(double x): [ESP+4..+11] = x, result pushed to FPU ST(0)
unsafe fn handle_cdecl_sin() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.sin().to_bits()));
    true
}

unsafe fn handle_cdecl_cos() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.cos().to_bits()));
    true
}

unsafe fn handle_cdecl_tan() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.tan().to_bits()));
    true
}

unsafe fn handle_cdecl_sqrt() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.sqrt().to_bits()));
    true
}

unsafe fn handle_cdecl_log() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.ln().to_bits()));
    true
}

unsafe fn handle_cdecl_exp() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.exp().to_bits()));
    true
}

unsafe fn handle_cdecl_acos() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.acos().to_bits()));
    true
}

unsafe fn handle_cdecl_asin() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.asin().to_bits()));
    true
}

unsafe fn handle_cdecl_log10() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.log10().to_bits()));
    true
}

unsafe fn handle_cdecl_atan() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.atan().to_bits()));
    true
}

unsafe fn handle_cdecl_fabs() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    fpu_push(F80::of_f64(x.abs().to_bits()));
    true
}

/// cdecl atan2(double y, double x): y at [ESP+4], x at [ESP+12]
unsafe fn handle_cdecl_atan2() -> bool {
    let esp = read_reg32(ESP);
    let y = read_stack_f64(esp + 4);
    let x = read_stack_f64(esp + 12);
    fpu_push(F80::of_f64(y.atan2(x).to_bits()));
    true
}

/// cdecl fmod(double x, double y): x at [ESP+4], y at [ESP+12]
unsafe fn handle_cdecl_fmod() -> bool {
    let esp = read_reg32(ESP);
    let x = read_stack_f64(esp + 4);
    let y = read_stack_f64(esp + 12);
    let result = if y != 0.0 { x % y } else { 0.0 };
    fpu_push(F80::of_f64(result.to_bits()));
    true
}

/// cdecl pow(double x, double y): x at [ESP+4], y at [ESP+12]
unsafe fn handle_cdecl_pow() -> bool {
    let esp = read_reg32(ESP);
    let x = read_stack_f64(esp + 4);
    let y = read_stack_f64(esp + 12);
    fpu_push(F80::of_f64(x.powf(y).to_bits()));
    true
}

/// cdecl ceil(double x): result pushed to FPU ST(0), EAX = truncated int
unsafe fn handle_cdecl_ceil() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    let result = x.ceil();
    fpu_push(F80::of_f64(result.to_bits()));
    write_reg32(EAX, result as i32);
    true
}

/// cdecl floor(double x): result pushed to FPU ST(0), EAX = truncated int
unsafe fn handle_cdecl_floor() -> bool {
    let x = read_stack_f64(read_reg32(ESP) + 4);
    let result = x.floor();
    fpu_push(F80::of_f64(result.to_bits()));
    write_reg32(EAX, result as i32);
    true
}

// ---------------------------------------------------------------------------
// Tier 3: String/memory hypercall handlers
// ---------------------------------------------------------------------------

/// Helper: safe_read16 wrapper (local, to avoid import issues)
unsafe fn hc_safe_read16(addr: i32) -> Result<i32, ()> {
    crate::cpu::cpu::safe_read16(addr).map_err(|_| ())
}

/// Helper: safe_read8 wrapper
unsafe fn hc_safe_read8(addr: i32) -> Result<i32, ()> {
    crate::cpu::cpu::safe_read8(addr).map_err(|_| ())
}

/// Helper: safe_write16 wrapper
unsafe fn hc_safe_write16(addr: i32, value: i32) -> Result<(), ()> {
    crate::cpu::cpu::safe_write16(addr, value).map_err(|_| ())
}

/// Helper: safe_write8 wrapper
unsafe fn hc_safe_write8(addr: i32, value: i32) -> Result<(), ()> {
    crate::cpu::cpu::safe_write8(addr, value).map_err(|_| ())
}

/// wcslen(wchar_t* str): count of wchar_t until NUL
unsafe fn handle_wcslen() -> bool {
    let esp = read_reg32(ESP);
    let str_ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if str_ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    let mut len: u32 = 0;
    loop {
        let ch = match hc_safe_read16(str_ptr + len as i32 * 2) {
            Ok(v) => v,
            Err(_) => break,
        };
        if ch == 0 { break; }
        len += 1;
        if len > 0x100000 { break; }
    }
    write_reg32(EAX, len as i32);
    true
}

/// wcscpy(wchar_t* dst, wchar_t* src): copy wide string, return dst
unsafe fn handle_wcscpy() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if dst == 0 || src == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    let mut i: u32 = 0;
    loop {
        let ch = match hc_safe_read16(src + i as i32 * 2) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc_safe_write16(dst + i as i32 * 2, ch).is_err() { return false; }
        if ch == 0 { break; }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, dst);
    true
}

/// wcscat(wchar_t* dst, wchar_t* src): append src to dst, return dst
unsafe fn handle_wcscat() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if dst == 0 || src == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    // Find end of dst
    let mut dst_len: u32 = 0;
    loop {
        let ch = match hc_safe_read16(dst + dst_len as i32 * 2) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if ch == 0 { break; }
        dst_len += 1;
        if dst_len > 0x100000 { break; }
    }
    // Copy src
    let mut i: u32 = 0;
    loop {
        let ch = match hc_safe_read16(src + i as i32 * 2) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc_safe_write16(dst + (dst_len + i) as i32 * 2, ch).is_err() { return false; }
        if ch == 0 { break; }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, dst);
    true
}

/// _wcsicmp(wchar_t* s1, wchar_t* s2): case-insensitive wide string compare
unsafe fn handle_wcsicmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut i: u32 = 0;
    let result = loop {
        let c1 = match hc_safe_read16(s1 + i as i32 * 2) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read16(s2 + i as i32 * 2) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        // ASCII tolower
        let l1 = if c1 >= 0x41 && c1 <= 0x5A { c1 + 0x20 } else { c1 };
        let l2 = if c2 >= 0x41 && c2 <= 0x5A { c2 + 0x20 } else { c2 };
        if l1 != l2 { break (l1 as i32) - (l2 as i32); }
        if c1 == 0 { break 0i32; }
        i += 1;
        if i > 0x100000 { break 0i32; }
    };
    write_reg32(EAX, result);
    true
}

/// wcschr(wchar_t* str, wchar_t ch): find first occurrence
unsafe fn handle_wcschr() -> bool {
    let esp = read_reg32(ESP);
    let str_ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let target = match safe_read32s(esp + 8) {
        Ok(v) => v & 0xFFFF,
        Err(_) => return false,
    };
    if str_ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    let mut i: u32 = 0;
    loop {
        let ch = match hc_safe_read16(str_ptr + i as i32 * 2) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if ch == target {
            write_reg32(EAX, str_ptr + i as i32 * 2);
            return true;
        }
        if ch == 0 { break; }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, 0);
    true
}

/// memcpy(void* dst, void* src, size_t n): copy n bytes
unsafe fn handle_memcpy() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let len = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if dst == 0 || src == 0 || len == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    // Copy 4 bytes at a time, then remaining bytes
    let mut i: u32 = 0;
    while i + 4 <= len {
        let val = match safe_read32s(src + i as i32) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if safe_write32(dst + i as i32, val).is_err() { return false; }
        i += 4;
    }
    while i < len {
        let byte = match hc_safe_read8(src + i as i32) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc_safe_write8(dst + i as i32, byte).is_err() { return false; }
        i += 1;
    }
    write_reg32(EAX, dst);
    true
}

/// memset(void* dst, int c, size_t n): fill n bytes with c
unsafe fn handle_memset() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let c = match safe_read32s(esp + 8) {
        Ok(v) => v & 0xFF,
        Err(_) => return false,
    };
    let len = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    if dst == 0 || len == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    // Fill 4 bytes at a time
    let fill32 = c | (c << 8) | (c << 16) | (c << 24);
    let mut i: u32 = 0;
    while i + 4 <= len {
        if safe_write32(dst + i as i32, fill32).is_err() { return false; }
        i += 4;
    }
    while i < len {
        if hc_safe_write8(dst + i as i32, c).is_err() { return false; }
        i += 1;
    }
    write_reg32(EAX, dst);
    true
}

/// strlen(char* str): count bytes until NUL
unsafe fn handle_strlen() -> bool {
    let esp = read_reg32(ESP);
    let str_ptr = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if str_ptr == 0 {
        write_reg32(EAX, 0);
        return true;
    }
    let mut len: u32 = 0;
    loop {
        let ch = match hc_safe_read8(str_ptr + len as i32) {
            Ok(v) => v,
            Err(_) => break,
        };
        if ch == 0 { break; }
        len += 1;
        if len > 0x100000 { break; }
    }
    write_reg32(EAX, len as i32);
    true
}

/// strcmp(char* s1, char* s2): byte-by-byte comparison
unsafe fn handle_strcmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut i: u32 = 0;
    let result = loop {
        let c1 = match hc_safe_read8(s1 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read8(s2 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        if c1 != c2 { break (c1 as i32) - (c2 as i32); }
        if c1 == 0 { break 0i32; }
        i += 1;
        if i > 0x100000 { break 0i32; }
    };
    write_reg32(EAX, result);
    true
}

/// strcpy(char* dst, char* src): copy ANSI string, return dst
unsafe fn handle_strcpy() -> bool {
    let esp = read_reg32(ESP);
    let dst = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let src = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if dst == 0 || src == 0 {
        write_reg32(EAX, dst);
        return true;
    }
    let mut i: u32 = 0;
    loop {
        let ch = match hc_safe_read8(src + i as i32) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if hc_safe_write8(dst + i as i32, ch).is_err() { return false; }
        if ch == 0 { break; }
        i += 1;
        if i > 0x100000 { break; }
    }
    write_reg32(EAX, dst);
    true
}

/// _stricmp(char* s1, char* s2): case-insensitive ANSI string compare
unsafe fn handle_stricmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut i: u32 = 0;
    let result = loop {
        let c1 = match hc_safe_read8(s1 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read8(s2 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let l1 = if c1 >= 0x41 && c1 <= 0x5A { c1 + 0x20 } else { c1 };
        let l2 = if c2 >= 0x41 && c2 <= 0x5A { c2 + 0x20 } else { c2 };
        if l1 != l2 { break (l1 as i32) - (l2 as i32); }
        if c1 == 0 { break 0i32; }
        i += 1;
        if i > 0x100000 { break 0i32; }
    };
    write_reg32(EAX, result);
    true
}

/// memcmp(void* s1, void* s2, size_t n): compare n bytes
unsafe fn handle_memcmp() -> bool {
    let esp = read_reg32(ESP);
    let s1 = match safe_read32s(esp + 4) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let s2 = match safe_read32s(esp + 8) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let len = match safe_read32s(esp + 12) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };
    let mut i: u32 = 0;
    while i < len {
        let c1 = match hc_safe_read8(s1 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        let c2 = match hc_safe_read8(s2 + i as i32) {
            Ok(v) => v as u32,
            Err(_) => return false,
        };
        if c1 != c2 {
            write_reg32(EAX, (c1 as i32) - (c2 as i32));
            return true;
        }
        i += 1;
    }
    write_reg32(EAX, 0);
    true
}

// ---------------------------------------------------------------------------
// Tier 4: Scheduler hypercall handlers
// ---------------------------------------------------------------------------

/// Sleep(dwMilliseconds) — stdcall(1).
/// Fast path: Sleep(0) with no other runnable threads is a pure no-op.
/// When peers exist, uses a starvation counter: only every Nth call falls
/// through to JS for actual context switch. This matches real Windows behavior
/// where Sleep(0) yields the remainder of the time quantum — often a near no-op
/// if the thread was just scheduled.
unsafe fn handle_sleep() -> bool {
    let esp = read_reg32(ESP);
    let ms = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    if ms == 0 {
        let page = hp_ptr();
        let has_peers = *(page.add(OFF_HC_HAS_RUNNABLE_PEERS) as *const u32);
        if has_peers == 0 {
            // No-op: no peer thread to yield to. Return void (EAX=0 by convention).
            write_reg32(EAX, 0);
            return true;
        }

        // Peers exist — use starvation counter to avoid excessive JS transitions.
        let counter_ptr = hp_mut().add(OFF_HC_SLEEP_STARVATION_COUNTER) as *mut u32;
        let counter = *counter_ptr;
        let limit = *(page.add(OFF_HC_SLEEP_STARVATION_LIMIT) as *const u32);

        if limit > 0 && counter < limit {
            // Not yet starved — no-op in WASM, peers can wait a few more µs
            *counter_ptr = counter + 1;
            write_reg32(EAX, 0);
            return true;
        }

        // Starvation limit reached or unset → reset counter and fall through to JS
        *counter_ptr = 0;
    }

    // Non-zero sleep or starvation limit hit → fall through to JS for context switch
    false
}

/// TlsGetValue(dwTlsIndex) — stdcall(1).
/// Reads TLS slot from guest TEB memory: TEB+0x2C → TLS array ptr, then array[index*4].
/// Sets LastError to 0 on success (Windows behavior).
/// Falls through to JS for out-of-range indices or memory faults.
unsafe fn handle_tls_get_value() -> bool {
    let esp = read_reg32(ESP);
    let index = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    // Only handle standard TLS slots (0..63)
    if index >= 64 {
        return false;
    }

    // Read TEB base from hypercall page
    let teb_base = *(hp_ptr().add(OFF_HC_TEB_BASE) as *const u32);
    if teb_base == 0 {
        return false;
    }

    // TEB+0x2C = ThreadLocalStoragePointer → address of TLS array
    // Use direct memory read — TEB is in identity-mapped HEAP, TLB never has these pages
    // (all guest TEB access is HLE via FS:), so safe_read32s always faults → JS fallback.
    let tls_array_ptr = memory::read32_no_mmap_check(teb_base + 0x2C) as u32;
    if tls_array_ptr == 0 {
        return false;
    }

    // Read TLS slot value (direct memory — same identity-mapped HEAP region)
    let value = memory::read32_no_mmap_check(tls_array_ptr + index * 4);

    // SetLastError(0) — TlsGetValue clears last error on success
    *(hp_mut().add(OFF_HC_LAST_ERROR) as *mut u32) = 0;
    memory::write32_no_mmap_or_dirty_check(teb_base + 0x34, 0);

    write_reg32(EAX, value);
    true
}

/// FlsGetValue(dwFlsIndex) — stdcall(1).
/// Reads FLS slot value from HYPERCALL_PAGE shadow written by JS.
/// Falls through to JS for out-of-range slots or for slots that are not allocated.
unsafe fn handle_fls_get_value() -> bool {
    let esp = read_reg32(ESP);
    let index = match safe_read32s(esp + 4) {
        Ok(v) => v as u32,
        Err(_) => return false,
    };

    if index as usize >= HC_FLS_SLOT_COUNT {
        return false;
    }

    let allocated = *(hp_ptr().add(OFF_HC_FLS_ALLOCATED + index as usize) as *const u8);
    if allocated == 0 {
        return false;
    }

    let value = *(hp_ptr().add(OFF_HC_FLS_VALUES + index as usize * 4) as *const u32);
    write_reg32(EAX, value as i32);
    true
}

// ---------------------------------------------------------------------------
// Heap arena: WASM-resident slab allocator for HeapAlloc/HeapFree
// ---------------------------------------------------------------------------

/// Map requested byte count to a size-class bin index (0–8).
/// Returns None for 0 bytes or >4KB (JS fallback).
fn size_to_bin(bytes: u32) -> Option<u32> {
    if bytes == 0 { return None; }
    if bytes <= 16 { return Some(0); }
    if bytes <= 32 { return Some(1); }
    if bytes <= 64 { return Some(2); }
    if bytes <= 128 { return Some(3); }
    if bytes <= 256 { return Some(4); }
    if bytes <= 512 { return Some(5); }
    if bytes <= 1024 { return Some(6); }
    if bytes <= 2048 { return Some(7); }
    if bytes <= 4096 { return Some(8); }
    None
}

/// Zero a block of guest memory using 4-byte writes (identity-mapped).
unsafe fn zero_block(addr: u32, size: u32) {
    let mut off = 0u32;
    while off < size {
        memory::write32_no_mmap_or_dirty_check(addr + off, 0);
        off += 4;
    }
}

/// HeapAlloc(hHeap, dwFlags, dwBytes) — stdcall, 3 args.
/// Fast path: slab bump allocation or free-list pop for blocks ≤4KB.
/// Returns false to fall through to JS for large/zero-size/slab-exhausted cases.
unsafe fn handle_heap_alloc() -> bool {
    let page = hp_ptr();
    let slab_base = *(page.add(OFF_HC_SLAB_BASE) as *const u32);
    if slab_base == 0 { return false; } // slab not initialized

    let esp = read_reg32(ESP);
    // HeapAlloc(hHeap, dwFlags, dwBytes) at ESP+4, +8, +12
    let dw_flags = match safe_read32s(esp + 8) { Ok(v) => v as u32, Err(_) => return false };
    let dw_bytes = match safe_read32s(esp + 12) { Ok(v) => v as u32, Err(_) => return false };

    if dw_bytes == 0 { return false; } // let JS set ERROR_NOT_ENOUGH_MEMORY

    let bin = match size_to_bin(dw_bytes) {
        Some(b) => b,
        None => {
            // >4KB — count fallback, let JS handle
            let p = hp_mut().add(OFF_HC_SLAB_FALLBACK_COUNT) as *mut u32;
            *p = (*p).wrapping_add(1);
            return false;
        }
    };
    let size_class = BIN_SIZES[bin as usize];
    let zero = (dw_flags & HEAP_ZERO_MEMORY_FLAG) != 0;

    // Try free list first
    let fl_ptr = hp_mut().add(OFF_HC_SLAB_FREELIST + (bin as usize) * 4) as *mut u32;
    let head = *fl_ptr;
    if head != 0 {
        // Pop from free list: first 4 bytes of user data = next pointer
        let next = memory::read32_no_mmap_check(head) as u32;
        *fl_ptr = next;
        if zero { zero_block(head, size_class); }
        write_reg32(EAX, head as i32);
        let c = hp_mut().add(OFF_HC_SLAB_ALLOC_COUNT) as *mut u32;
        *c = (*c).wrapping_add(1);
        return true;
    }

    // Bump allocate: [bump..bump+16) = header zone, [bump+16..bump+16+size_class) = user data
    let bump_ptr = hp_mut().add(OFF_HC_SLAB_BUMP) as *mut u32;
    let bump = *bump_ptr;
    let slab_end = *(page.add(OFF_HC_SLAB_END) as *const u32);
    let user_ptr = bump + 16;
    let new_bump = user_ptr + size_class;
    if new_bump > slab_end {
        let p = hp_mut().add(OFF_HC_SLAB_FALLBACK_COUNT) as *mut u32;
        *p = (*p).wrapping_add(1);
        return false; // slab exhausted → JS allocates + can refill slab
    }

    // Write block header at user_ptr - 4
    memory::write32_no_mmap_or_dirty_check(user_ptr - 4, (SLAB_MAGIC | bin) as i32);
    *bump_ptr = new_bump;

    if zero { zero_block(user_ptr, size_class); }

    write_reg32(EAX, user_ptr as i32);
    let c = hp_mut().add(OFF_HC_SLAB_ALLOC_COUNT) as *mut u32;
    *c = (*c).wrapping_add(1);
    true
}

/// HeapFree(hHeap, dwFlags, lpMem) — stdcall, 3 args.
/// Fast path: push freed block onto per-bin free list if it's within slab range.
unsafe fn handle_heap_free() -> bool {
    let page = hp_ptr();
    let slab_base = *(page.add(OFF_HC_SLAB_BASE) as *const u32);
    if slab_base == 0 { return false; }

    let esp = read_reg32(ESP);
    let lp_mem = match safe_read32s(esp + 12) { Ok(v) => v as u32, Err(_) => return false };

    if lp_mem == 0 {
        write_reg32(EAX, 1); // HeapFree(NULL) = TRUE per Windows spec
        return true;
    }

    let slab_end = *(page.add(OFF_HC_SLAB_END) as *const u32);
    // Must be within slab with room for header
    if lp_mem < slab_base + 16 || lp_mem >= slab_end { return false; }

    // Validate slab header at [lp_mem - 4]
    let header = memory::read32_no_mmap_check(lp_mem - 4) as u32;
    if (header & 0xFFFFFF00) != SLAB_MAGIC { return false; }
    let bin = header & 0x0F;
    if bin > 8 { return false; }

    // Push to free list: store current head in freed block's first 4 bytes
    let fl_ptr = hp_mut().add(OFF_HC_SLAB_FREELIST + (bin as usize) * 4) as *mut u32;
    let old_head = *fl_ptr;
    memory::write32_no_mmap_or_dirty_check(lp_mem, old_head as i32);
    *fl_ptr = lp_mem;

    let c = hp_mut().add(OFF_HC_SLAB_FREE_COUNT) as *mut u32;
    *c = (*c).wrapping_add(1);
    write_reg32(EAX, 1); // TRUE
    true
}

