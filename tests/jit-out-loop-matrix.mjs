#!/usr/bin/env node
// Variant matrix to localize the "block-boundary instr inside a hot loop hangs
// under JIT" bug. Oracle: interpreter vs forced-JIT (interpreter is correct).
//
//   node tests/jit-out-loop-matrix.mjs

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000, ENTRY_OFF = 0x20, SUB_OFF = 0x1000;
const LOOP_COUNT = 0x6c, MEM_SIZE = 16 * 1024 * 1024, TIMEOUT_MS = 3000;

// boundary instruction encodings (all are block_boundary in v86)
const BOUNDARY = {
    out:   [0xEF],             // out dx, eax
    in:    [0xED],             // in  eax, dx
    cpuid: [0x0F, 0xA2],       // cpuid (block_boundary)
    nop:   [0x90],             // nop  (NOT a boundary — control)
    none:  [],                 // emit nothing (pure loop control)
};

function build_image(opts)
{
    // opts: { crossPage, useCall, boundary }
    const { crossPage, useCall, boundary } = opts;
    const bytes = BOUNDARY[boundary];

    const buf = new Uint8Array(SUB_OFF + 32);
    const dv  = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + SUB_OFF + 32, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };

    emit(0xBC, 0x00, 0x00, 0x20, 0x00);   // mov esp, 0x200000
    emit(0xBA, 0x80, 0x00, 0x00, 0x00);   // mov edx, 0x80
    emit(0x31, 0xFF);                      // xor edi, edi
    const loop_top = o;
    emit(0x83, 0xFF, LOOP_COUNT);          // cmp edi, 0x6c
    const jge_at = o; emit(0x7D, 0x00);    // jge done

    if(useCall)
    {
        const call_at = o; emit(0xE8, 0,0,0,0);   // call sub
        // sub target offset chosen below
        emit(0x47);                                // inc edi
        const jmp_at = o; emit(0xEB, 0x00);        // jmp loop_top
        const done = o;
        emit(0xF4); emit(0xEB, 0xFE);              // hlt; jmp $

        // sub location: separate page or right here (same page)
        const sub_off = crossPage ? SUB_OFF : o;
        dv.setInt32(call_at + 1, sub_off - (call_at + 5), true);
        buf[jge_at + 1] = (done - (jge_at + 2)) & 0xff;
        buf[jmp_at + 1] = (loop_top - (jmp_at + 2)) & 0xff;

        o = sub_off;
        emit(...bytes);    // boundary instr
        emit(0xC3);        // ret
    }
    else
    {
        // boundary inlined in the loop body (no CALL)
        emit(...bytes);                            // boundary instr
        emit(0x47);                                // inc edi
        const jmp_at = o; emit(0xEB, 0x00);        // jmp loop_top
        const done = o;
        emit(0xF4); emit(0xEB, 0xFE);              // hlt; jmp $
        buf[jge_at + 1] = (done - (jge_at + 2)) & 0xff;
        buf[jmp_at + 1] = (loop_top - (jmp_at + 2)) & 0xff;
    }

    return { buf, loopPage: BASE, subPage: BASE + SUB_OFF, crossPage, useCall };
}

function run(opts, { force_jit })
{
    return new Promise((resolve) => {
        const img = build_image(opts);
        const emulator = new V86({ autostart:false, memory_size:MEM_SIZE,
                                   disable_jit: force_jit ? 0 : 1, log_level:0 });
        let halted = false, timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({ status, edi: cpu.reg32[7] >>> 0, eip: cpu.instruction_pointer[0] >>> 0 });
        };
        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(img.buf.buffer);
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            if(force_jit)
            {
                const pages = img.useCall && img.crossPage ? [img.loopPage, img.subPage] : [img.loopPage];
                let idx = 0;
                cpu.test_hook_did_finalize_wasm = function() {
                    if(++idx < pages.length) cpu.jit_force_generate(pages[idx]);
                    else { cpu.test_hook_did_finalize_wasm = null; setTimeout(() => emulator.run(), 0); }
                };
                cpu.jit_force_generate(pages[0]);
            }
            else emulator.run();
        });
    });
}

const variants = [
    { name: "pureloop",       crossPage:false, useCall:false, boundary:"none"  },
    { name: "callcross-out",  crossPage:true,  useCall:true,  boundary:"out"   },
    { name: "callsame-out",   crossPage:false, useCall:true,  boundary:"out"   },
    { name: "callcross-in",   crossPage:true,  useCall:true,  boundary:"in"    },
    { name: "callcross-cpuid",crossPage:true,  useCall:true,  boundary:"cpuid" },
    { name: "callcross-nop",  crossPage:true,  useCall:true,  boundary:"nop"   },
    { name: "inline-out",     crossPage:false, useCall:false, boundary:"out"   },
    { name: "inline-cpuid",   crossPage:false, useCall:false, boundary:"cpuid" },
];

console.log("variant              | interp edi | JIT status (edi)");
console.log("---------------------+------------+------------------");
for(const v of variants)
{
    const i = await run(v, { force_jit:false });
    const j = await run(v, { force_jit:true });
    const flag = (i.edi === LOOP_COUNT && (j.status === "HANG" || j.edi !== LOOP_COUNT)) ? "  <== BUG" : "";
    console.log(
        v.name.padEnd(20) + " |    " +
        String(i.edi).padStart(3) + "     | " +
        (j.status === "HANG" ? "HANG eip=0x"+j.eip.toString(16)+" edi="+j.edi
                             : "halt edi=" + j.edi) + flag);
}
process.exit(0);
