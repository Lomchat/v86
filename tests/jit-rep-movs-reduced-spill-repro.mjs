#!/usr/bin/env node
// Generic semantic/performance probe for JIT config 35. It repeatedly executes
// one REP MOVS site so the normal hotness threshold compiles it, then compares
// reduced-spill ON/OFF for forward, cross-page, backward and overlapping copies.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = 0x20;
const MEM_SIZE = 16 * 1024 * 1024;
const ITER = 600_000;
const TIMEOUT_MS = 20_000;
const PD_ADDR = 0x308000, PT_ADDR = 0x309000;

function buildImage(test)
{
    const buf = new Uint8Array(0x1000);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY, true);

    let o = ENTRY;
    const emit = (...bytes) => { for(const byte of bytes) buf[o++] = byte & 0xff; };
    const u32 = value => { dv.setUint32(o, value >>> 0, true); o += 4; };
    const movReg = (opcode, value) => { emit(opcode); u32(value); };

    if(test.paging)
    {
        emit(0xFC);                               // cld
        movReg(0xBF, PT_ADDR);                    // mov edi, page table
        movReg(0xB8, 3);                          // mov eax, present|rw
        movReg(0xB9, 0x400);                      // mov ecx, 1024
        const fillPt = o;
        emit(0xAB);                               // stosd
        emit(0x05); u32(0x1000);                  // add eax, 0x1000
        emit(0xE2, (fillPt - (o + 2)) & 0xff);    // loop fillPt
        movReg(0xBF, PD_ADDR);                    // mov edi, page directory
        emit(0x31, 0xC0);                         // xor eax, eax
        movReg(0xB9, 0x400);                      // mov ecx, 1024
        emit(0xF3, 0xAB);                         // rep stosd
        emit(0xC7, 0x05); u32(PD_ADDR); u32(PT_ADDR | 3);
        movReg(0xB8, PD_ADDR);                    // mov eax, cr3 value
        emit(0x0F, 0x22, 0xD8);                  // mov cr3, eax
        emit(0x0F, 0x20, 0xC0);                  // mov eax, cr0
        emit(0x0D); u32(0x80000000);              // or eax, PG
        emit(0x0F, 0x22, 0xC0);                  // mov cr0, eax
    }

    movReg(0xBC, 0x300000);                    // mov esp, ...
    movReg(0xBD, ITER);                        // mov ebp, ITER
    const loop = o;
    if(test.resetOverlap)
    {
        for(let i = 0; i < 8; i++)
        {
            emit(0xC7, 0x05); u32(test.src + i * 4); u32(0x04030201 + i * 0x04040404);
        }
    }
    movReg(0xBE, test.startSrc);                // mov esi, source
    movReg(0xBF, test.startDst);                // mov edi, destination
    movReg(0xB9, test.count);                   // mov ecx, count
    if(test.backward) emit(0xFD);               // std
    emit(0xF3, test.dword ? 0xA5 : 0xA4);       // rep movsd / rep movsb
    if(test.backward) emit(0xFC);               // cld
    emit(0x4D);                                 // dec ebp
    emit(0x0F, 0x85);                           // jnz loop
    dv.setInt32(o, loop - (o + 4), true); o += 4;
    emit(0xF4, 0xEB, 0xFE);                     // hlt; jmp $
    return buf;
}

const tests = [
    { name:"movsb-forward", src:0x180100, dst:0x181200, startSrc:0x180100, startDst:0x181200, count:3, bytes:3 },
    { name:"movsd-forward", src:0x180100, dst:0x181200, startSrc:0x180100, startDst:0x181200, count:7, bytes:28, dword:true },
    { name:"movsb-cross-page", src:0x180ff0, dst:0x182ff8, startSrc:0x180ff0, startDst:0x182ff8, count:32, bytes:32 },
    { name:"movsd-backward", src:0x180100, dst:0x181200, startSrc:0x18011c, startDst:0x18121c, count:8, bytes:32, dword:true, backward:true },
    { name:"movsd-paging", src:0x180100, dst:0x181200, startSrc:0x180100, startDst:0x181200, count:7, bytes:28, dword:true, paging:true },
    { name:"movsb-overlap-fallback", src:0x180100, dst:0x180101, startSrc:0x180100, startDst:0x180101, count:32, bytes:32, resetOverlap:true, overlap:true },
];

function sourcePattern(length)
{
    return Uint8Array.from({ length }, (_, i) => (i * 37 + 11) & 0xff);
}

function run(test, enabled)
{
    return new Promise(resolve => {
        const emulator = new V86({ autostart:false, memory_size:MEM_SIZE, disable_jit:0, log_level:0 });
        let timer;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch {}
            const cpu = emulator.v86.cpu;
            resolve({
                status,
                enabled: cpu.get_jit_config?.(35) >>> 0,
                elapsedMs: performance.now() - startedAt,
                esi: cpu.reg32[6] >>> 0,
                edi: cpu.reg32[7] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                bytes: Array.from(cpu.mem8.slice(test.dst, test.dst + test.bytes)),
            });
        };
        let startedAt = 0;
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(35, enabled ? 1 : 0);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(buildImage(test).buffer);
            if(!test.resetOverlap) cpu.mem8.set(sourcePattern(test.bytes), test.src);
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

for(const test of tests)
{
    const off = await run(test, false);
    const on = await run(test, true);
    console.log("jit-rep-movs-reduced-spill " + JSON.stringify({ name:test.name, off, on }));
    const expected = test.overlap ? new Array(test.bytes).fill(1) : Array.from(sourcePattern(test.bytes));
    const forwardDelta = test.count * (test.dword ? 4 : 1);
    const expectedEsi = test.backward ? test.startSrc - forwardDelta : test.startSrc + forwardDelta;
    const expectedEdi = test.backward ? test.startDst - forwardDelta : test.startDst + forwardDelta;
    for(const result of [off, on])
    {
        if(result.status !== "halt" || result.ecx !== 0 || result.esi !== (expectedEsi >>> 0)
            || result.edi !== (expectedEdi >>> 0) || JSON.stringify(result.bytes) !== JSON.stringify(expected))
        {
            console.error("FAIL: semantic mismatch", test.name, result, { expectedEsi, expectedEdi, expected });
            process.exit(1);
        }
    }
}

process.exit(0);
