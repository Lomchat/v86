//! Host-side glue for the d3d9-webgpu crate: injects v86's paging-aware guest-memory
//! readers into the crate's GuestMem table. Called once from rust_init (jit.rs).
//!
//! The bulk read_block mirrors the word+tail loop the arena's copy_guest_bytes used
//! when it lived in-tree (and handle_memcpy's tail-byte handling, hypercall.rs).

use crate::cpu::cpu::{safe_read32s, safe_read8};
use d3d9_webgpu::guest_mem::{set_guest_mem, GuestMem};

fn read_u32(addr: i32) -> Result<i32, ()> {
    unsafe { safe_read32s(addr) }
}

fn read_u8(addr: i32) -> Result<i32, ()> {
    unsafe { safe_read8(addr) }
}

fn read_block(addr: i32, dst: *mut u8, len: u32) -> bool {
    unsafe {
        let mut i: u32 = 0;
        while i + 4 <= len {
            match safe_read32s(addr + i as i32) {
                Ok(val) => *(dst.add(i as usize) as *mut i32) = val,
                Err(()) => return false,
            }
            i += 4;
        }
        while i < len {
            match safe_read8(addr + i as i32) {
                Ok(byte) => *dst.add(i as usize) = byte as u8,
                Err(()) => return false,
            }
            i += 1;
        }
        true
    }
}

pub fn init() {
    set_guest_mem(GuestMem {
        read_u32,
        read_u8,
        read_block,
    });
}
