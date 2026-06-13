#!/usr/bin/env node
// Reproduction for "OUT-trap inside CALL inside a hot loop never terminates
// under the JIT" (Harry Potter UE1 / URender::DrawStats).
//
// JIT_THRESHOLD is 200_000, so we drive the inner 108-loop from an OUTER loop
// until the inner block compiles *naturally* (like the game's per-frame loop),
// then check whether it still terminates. NO jit_force_generate (its test hook
// does not fire in the release build).
//
// Oracle: interpreter is known-correct. If the JIT run hangs (esi never reaches
// OUTER) while the interpreter halts with esi==OUTER, the bug is reproduced.
//
//   node tests/jit-out-loop-repro.mjs

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000, ENTRY_OFF = 0x20, SUB_OFF = 0x1000;
const INNER = 0x6c;        // 108, same as the game
const OUTER = 5000;        // 5000*108 = 540k inner-block hits > JIT_THRESHOLD(200k)
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 10000;

function build_image()
{
    const buf = new Uint8Array(SUB_OFF + 16);
    const dv  = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);                   // header_addr
    dv.setUint32(0x10, BASE, true);                   // load_addr
    dv.setUint32(0x14, BASE + SUB_OFF + 16, true);    // load_end_addr
    dv.setUint32(0x18, BASE + 0x4000, true);          // bss_end_addr
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);       // entry_addr

    // tiny label/patch assembler
    let o = ENTRY_OFF;
    const labels = {}, patches = [];
    const here = () => o;
    const label = (n) => { labels[n] = o; };
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const rel8 = (n) => { patches.push({ at:o, sz:1, to:n, end:o+1 }); emit(0); };
    const rel32 = (n) => { patches.push({ at:o, sz:4, to:n, end:o+4 }); emit(0,0,0,0); };

    emit(0xBC, 0x00, 0x00, 0x20, 0x00);     // mov esp, 0x200000
    emit(0xBA, 0x80, 0x00, 0x00, 0x00);     // mov edx, 0x80
    emit(0x31, 0xF6);                        // xor esi, esi   (outer counter)
    label("outer");
    emit(0x81, 0xFE); dv.setUint32(o, OUTER >>> 0, true); o += 4;   // cmp esi, OUTER
    emit(0x0F, 0x8D); rel32("done");         // jge done
    emit(0x31, 0xFF);                        // xor edi, edi   (inner counter)
    label("inner");
    emit(0x83, 0xFF, INNER);                 // cmp edi, 0x6c
    emit(0x7D); rel8("inner_done");          // jge inner_done
    emit(0xE8); rel32("sub");                // call sub
    emit(0x47);                              // inc edi
    emit(0xE9); rel32("inner");              // jmp inner
    label("inner_done");
    emit(0x46);                              // inc esi
    emit(0xE9); rel32("outer");              // jmp outer
    label("done");
    emit(0xF4);                              // hlt
    emit(0xEB, 0xFE);                        // jmp $

    o = SUB_OFF; label("sub");
    emit(0xEF);                              // out dx, eax  (block boundary)
    emit(0xC3);                              // ret

    for(const p of patches)
    {
        const delta = labels[p.to] - p.end;
        if(p.sz === 1) buf[p.at] = delta & 0xff;
        else dv.setInt32(p.at, delta, true);
    }
    return { buf, loopPage: BASE };
}

function run({ jit })
{
    return new Promise((resolve) => {
        const img = build_image();
        const emulator = new V86({ autostart:false, memory_size:MEM_SIZE,
                                   disable_jit: jit ? 0 : 1, log_level:0 });
        let halted = false, timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({ status, esi: cpu.reg32[6] >>> 0, edi: cpu.reg32[7] >>> 0,
                      eip: cpu.instruction_pointer[0] >>> 0 });
        };
        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(img.buf.buffer);
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

console.log("=== JIT OUT-in-CALL-in-hot-loop repro (natural hotness, OUTER=%d) ===", OUTER);
const interp = await run({ jit:false });
console.log("interpreter:", JSON.stringify(interp),
            (interp.status==="halt" && interp.esi===OUTER) ? "  <- correct" : "  <- ?");
const jit = await run({ jit:true });
console.log("JIT        :", JSON.stringify(jit),
            jit.status==="HANG" ? "  <- REPRODUCED (hung)"
            : jit.esi===OUTER ? "  <- ok (no divergence)" : "  <- DIVERGED");

console.log("\nverdict:",
    (interp.status==="halt" && interp.esi===OUTER && jit.status==="HANG")
        ? "BUG REPRODUCED (interp halts, JIT hangs)"
        : (interp.esi===OUTER && jit.esi===OUTER)
            ? "NO repro — vanilla pattern compiles & runs fine under JIT"
            : "inconclusive");
process.exit(0);
