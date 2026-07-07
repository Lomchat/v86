//! d3d9-webgpu — WASM-resident D3D9 state mirror + command arena.
//!
//! Host-independent by construction: the only outside-world dependency is guest-memory
//! reads, injected through [`guest_mem::set_guest_mem`] at boot. The embedding emulator
//! (today: the v86 fork's cdylib) links this rlib and provides the accessors; the
//! `#[no_mangle]` exports in [`arena`] surface directly as wasm exports of the final
//! cdylib (presence is asserted post-build by tools/check-wasm-exports.ts).

pub mod arena;
pub mod guest_mem;
