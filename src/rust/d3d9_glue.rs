//! Host-side glue for the d3d9-webgpu crate: injects v86's paging-aware guest-memory
//! reader into the crate's GuestMem table. Called once from rust_init (jit.rs).

use crate::cpu::cpu::{safe_read32s, safe_read8};
use d3d9_webgpu::guest_mem::{set_guest_mem, GuestMem};

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
        read_block,
    });
}
