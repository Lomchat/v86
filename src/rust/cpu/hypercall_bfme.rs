//! BFME-specific inner-loop HLE handlers. These live in the engine-handler
//! band (128..=255) and are armed only by byte-exact, title-specific hooks.

use crate::cpu::cpu::{read_reg32, safe_read32s, safe_write32, safe_write8, write_reg32, EAX, ECX, ESP};
use crate::cpu::hypercall::hc_safe_read8;

pub(crate) unsafe fn dispatch_inner_loop(handler_id: u8) -> bool {
    match handler_id {
        135 => handle_fold33_hash(),
        136 => handle_stringbase_lower(),
        137 => handle_stringbase_release(),
        138 => handle_stringbase_copy(),
        139 => handle_stringbase_assign(),
        _ => false,
    }
}

const STRINGBASE_LOCK_GUARD: i32 = 0x01336e2c;

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
