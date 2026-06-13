#!/usr/bin/env node
// Deterministic repro attempt for the DrawStats hang, mirroring the game's
// 3-page call chain:  loop(page0) -> mid(page1) -> leaf(page2, OUT).
// (DrawStats render.dll -> appFrand core.dll -> rand-stub THUNK_CODE)
//
// Natural hotness (JIT_THRESHOLD=200k) via an OUTER loop. Oracle: interpreter.
// Also probes MAX_PAGES (set_jit_config idx 1) to test the multi-page theory.
//
//   node tests/jit-multipage-repro.mjs

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000, ENTRY_OFF = 0x20;
const MID_OFF = 0x1000, LEAF_OFF = 0x2000;
const INNER = 0x6c, OUTER = 6000;          // 6000*108 = 648k > 200k threshold
const MEM_SIZE = 16 * 1024 * 1024, TIMEOUT_MS = 12000;

function build_image()
{
    const buf = new Uint8Array(LEAF_OFF + 16);
    const dv  = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + LEAF_OFF + 16, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const labels = {}, patches = [];
    const label = (n) => { labels[n] = o; };
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const rel8 = (n) => { patches.push({ at:o, sz:1, end:o+1, to:n }); emit(0); };
    const rel32 = (n) => { patches.push({ at:o, sz:4, end:o+4, to:n }); emit(0,0,0,0); };

    emit(0xBC, 0x00, 0x00, 0x30, 0x00);   // mov esp, 0x300000
    emit(0xBA, 0x80, 0x00, 0x00, 0x00);   // mov edx, 0x80
    emit(0x31, 0xF6);                      // xor esi,esi
    label("outer");
    emit(0x81, 0xFE); dv.setUint32(o, OUTER>>>0, true); o += 4;  // cmp esi, OUTER
    emit(0x0F, 0x8D); rel32("done");       // jge done
    emit(0x31, 0xFF);                      // xor edi,edi
    label("inner");
    emit(0x83, 0xFF, INNER);               // cmp edi, 0x6c
    emit(0x7D); rel8("inner_done");        // jge inner_done
    emit(0xE8); rel32("mid");              // call mid (page1)
    emit(0x47);                            // inc edi
    emit(0xE9); rel32("inner");            // jmp inner
    label("inner_done");
    emit(0x46);                            // inc esi
    emit(0xE9); rel32("outer");            // jmp outer
    label("done");
    emit(0xF4); emit(0xEB, 0xFE);          // hlt; jmp $

    o = MID_OFF; label("mid");             // mirrors appFrand
    emit(0x51);                            // push ecx
    emit(0xE8); rel32("leaf");             // call leaf (page2)
    emit(0x59);                            // pop ecx
    emit(0xC3);                            // ret

    o = LEAF_OFF; label("leaf");           // mirrors rand stub
    emit(0xEF);                            // out dx, eax  (block boundary)
    emit(0xC3);                            // ret

    for(const p of patches)
    {
        const d = labels[p.to] - p.end;
        if(p.sz === 1) buf[p.at] = d & 0xff; else dv.setInt32(p.at, d, true);
    }
    return buf;
}

function run({ jit, maxPages })
{
    return new Promise((resolve) => {
        const buf = build_image();
        const emulator = new V86({ autostart:false, memory_size:MEM_SIZE,
                                   disable_jit: jit ? 0 : 1, log_level:0 });
        let halted = false, timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({ status, esi: cpu.reg32[6] >>> 0, edi: cpu.reg32[7] >>> 0 });
        };
        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(buf.buffer);
            if(jit && maxPages !== undefined && cpu.set_jit_config) {
                cpu.set_jit_config(1, maxPages >>> 0);     // MAX_PAGES
                if(cpu.jit_clear_cache) cpu.jit_clear_cache();
            }
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

const show = (label, r) => console.log(label.padEnd(22) + JSON.stringify(r) +
    (r.status==="halt" && r.esi===OUTER ? "  <- ok" : r.status==="HANG" ? "  <- HANG" : "  <- ?"));

console.log("=== 3-page chain (loop->mid->leaf:OUT), OUTER=%d ===", OUTER);
show("interpreter",            await run({ jit:false }));
show("JIT (MAX_PAGES=3 dflt)", await run({ jit:true }));
show("JIT MAX_PAGES=1",        await run({ jit:true, maxPages:1 }));
show("JIT MAX_PAGES=2",        await run({ jit:true, maxPages:2 }));
process.exit(0);
