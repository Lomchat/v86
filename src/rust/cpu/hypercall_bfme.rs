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
        140 => handle_stringbase_find(),
        141 => handle_matrix_push(),
        142 => handle_matrix_pop(),
        143 => handle_matrix_multiply(),
        144 => handle_transform_push(),
        145 => handle_transform_pop(),
        146 => handle_matrix_adjust(),
        147 => handle_tree_successor(),
        _ => false,
    }
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
