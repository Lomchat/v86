#!/usr/bin/env node
// Does enabling x86 PAGING make the 3-page DrawStats-style chain hang under JIT?
// The game runs with paging on (PageTableManager identity+CoW); the isolated
// tests so far ran paging-off and did NOT hang. v86's JIT tracks code by
// physical page + TLB, so paging is the prime-suspect runtime ingredient.
//
// Guest sets up a 4KB-page identity map of 0..4MB, enables CR0.PG, then runs:
//   loop(p0) -> mid(p1) -> leaf(p2: OUT)   driven hot by an OUTER loop.
// Oracle: interpreter (must halt with esi==OUTER — also validates the paging
// setup). If JIT hangs while interpreter halts -> reproduced with paging.
//
//   node tests/jit-paging-repro.mjs

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000, ENTRY_OFF = 0x20, MID_OFF = 0x1000, LEAF_OFF = 0x2000;
const PD_ADDR = 0x108000, PT0_ADDR = 0x109000;     // page tables (in 0..4MB)
const INNER = 0x6c, OUTER = 6000;
const MEM_SIZE = 16 * 1024 * 1024, TIMEOUT_MS = 12000;

function build_image({ paging })
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
    const u32at = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };
    const rel8  = (n) => { patches.push({ at:o, sz:1, end:o+1, to:n }); emit(0); };
    const rel32 = (n) => { patches.push({ at:o, sz:4, end:o+4, to:n }); emit(0,0,0,0); };

    if(paging)
    {
        // identity-map 0..4MB with one PT, then turn on CR0.PG
        emit(0xFC);                                   // cld
        emit(0xBF); u32at(PT0_ADDR);                  // mov edi, PT0_ADDR
        emit(0xB8); u32at(0x00000003);                // mov eax, 0|P|RW
        emit(0xB9); u32at(0x400);                     // mov ecx, 1024
        const fill = o;
        emit(0xAB);                                   // stosd
        emit(0x05); u32at(0x1000);                    // add eax, 0x1000
        emit(0xE2, (fill - (o + 2)) & 0xff);          // loop fill
        emit(0xBF); u32at(PD_ADDR);                   // mov edi, PD_ADDR
        emit(0x31, 0xC0);                             // xor eax, eax
        emit(0xB9); u32at(0x400);                     // mov ecx, 1024
        emit(0xF3, 0xAB);                             // rep stosd  (zero PD)
        emit(0xC7, 0x05); u32at(PD_ADDR); u32at((PT0_ADDR | 3) >>> 0); // mov [PD],PT0|3
        emit(0xB8); u32at(PD_ADDR);                   // mov eax, PD_ADDR
        emit(0x0F, 0x22, 0xD8);                       // mov cr3, eax
        emit(0x0F, 0x20, 0xC0);                       // mov eax, cr0
        emit(0x0D); u32at(0x80000000);                // or eax, 0x80000000
        emit(0x0F, 0x22, 0xC0);                       // mov cr0, eax  (PG on)
    }

    emit(0xBC); u32at(0x300000);           // mov esp, 0x300000
    emit(0xBA); u32at(0x80);               // mov edx, 0x80
    emit(0x31, 0xF6);                      // xor esi,esi
    label("outer");
    emit(0x81, 0xFE); u32at(OUTER);        // cmp esi, OUTER
    emit(0x0F, 0x8D); rel32("done");       // jge done
    emit(0x31, 0xFF);                      // xor edi,edi
    label("inner");
    emit(0x83, 0xFF, INNER);               // cmp edi, 0x6c
    emit(0x7D); rel8("inner_done");        // jge inner_done
    emit(0xE8); rel32("mid");              // call mid
    emit(0x47);                            // inc edi
    emit(0xE9); rel32("inner");            // jmp inner
    label("inner_done");
    emit(0x46);                            // inc esi
    emit(0xE9); rel32("outer");            // jmp outer
    label("done");
    emit(0xF4); emit(0xEB, 0xFE);          // hlt; jmp $

    o = MID_OFF; label("mid");
    emit(0x51);                            // push ecx
    emit(0xE8); rel32("leaf");             // call leaf
    emit(0x59);                            // pop ecx
    emit(0xC3);                            // ret
    o = LEAF_OFF; label("leaf");
    emit(0xEF);                            // out dx, eax
    emit(0xC3);                            // ret

    for(const p of patches)
    {
        const d = labels[p.to] - p.end;
        if(p.sz === 1) buf[p.at] = d & 0xff; else dv.setInt32(p.at, d, true);
    }
    return buf;
}

function run({ jit, paging, maxPages })
{
    return new Promise((resolve) => {
        const buf = build_image({ paging });
        const emulator = new V86({ autostart:false, memory_size:MEM_SIZE,
                                   disable_jit: jit ? 0 : 1, log_level:0 });
        let halted = false, timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({ status, esi: cpu.reg32[6] >>> 0, edi: cpu.reg32[7] >>> 0,
                      paging: (cpu.cr[0] >>> 0) & 0x80000000 ? 1 : 0 });
        };
        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(buf.buffer);
            if(jit && maxPages !== undefined && cpu.set_jit_config) {
                cpu.set_jit_config(1, maxPages >>> 0);
                if(cpu.jit_clear_cache) cpu.jit_clear_cache();
            }
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

const show = (l, r) => console.log(l.padEnd(26) + JSON.stringify(r) +
    (r.status==="halt" && r.esi===OUTER ? "  <- ok" : r.status==="HANG" ? "  <- HANG" : "  <- ?"));

console.log("=== paging-on 3-page chain (loop->mid->leaf:OUT), OUTER=%d ===", OUTER);
show("interp  paging-OFF", await run({ jit:false, paging:false }));
show("interp  paging-ON",  await run({ jit:false, paging:true  }));
show("JIT     paging-OFF", await run({ jit:true,  paging:false }));
show("JIT     paging-ON",  await run({ jit:true,  paging:true  }));
show("JIT pg-ON MAX_PAGES=1", await run({ jit:true, paging:true, maxPages:1 }));
process.exit(0);
