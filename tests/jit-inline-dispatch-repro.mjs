#!/usr/bin/env node
// Deterministic correctness + microbenchmark for config idx 22.
//
// A hot x86 CALL/RET loop makes every RET execute the current-module
// AbsoluteEip resolver. The inline form must preserve the exact result while
// avoiding the generated-module -> base-wasm helper call. A second cross-page
// case forces the lookup to miss (MAX_PAGES=1) and verifies that ordinary
// main-loop redispatch still completes correctly.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x20;
const LEAF_PAGE_OFF = 0x1000;
const ITERATIONS = 8_000_000;
const MISS_ITERATIONS = 800_000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30_000;
const COLD_OFF = 0xD0;
const COLD_MAGIC = 0x51A7B00B;

function buildImage(iterations, crossPage)
{
    const size = crossPage ? LEAF_PAGE_OFF + 16 : 0x100;
    const buf = new Uint8Array(size);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002;
    const FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + size, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const labels = {};
    const patches = [];
    const label = name => { labels[name] = o; };
    const emit = (...bytes) => { for(const byte of bytes) buf[o++] = byte & 0xff; };
    const u32 = value => { dv.setUint32(o, value >>> 0, true); o += 4; };
    const rel8 = name => { patches.push({ at:o, size:1, end:o + 1, name }); emit(0); };
    const rel32 = name => {
        patches.push({ at:o, size:4, end:o + 4, name });
        emit(0, 0, 0, 0);
    };

    emit(0xBC); u32(0x00300000);           // mov esp, 0x300000
    emit(0xB9); u32(iterations);           // mov ecx, iterations
    emit(0x31, 0xC0);                      // xor eax, eax
    label("loop");
    emit(0xE8); rel32("leaf");             // call leaf
    emit(0x49);                            // dec ecx
    emit(0x75); rel8("loop");              // jnz loop
    emit(0x89, 0xC2);                      // mov edx, eax (preserve arithmetic result)
    emit(0xB8); u32(BASE + COLD_OFF);       // mov eax, cold (same page, not in CFG)
    emit(0xFF, 0xE0);                      // jmp eax: same-module sentinel miss

    if(crossPage) o = LEAF_PAGE_OFF;
    label("leaf");
    emit(0x83, 0xC0, 0x03);                // add eax, 3
    emit(0xC3);                            // ret

    o = COLD_OFF;
    label("cold");
    emit(0xBB); u32(COLD_MAGIC);            // mov ebx, magic
    emit(0xF4, 0xEB, 0xFE);                // hlt; jmp $

    for(const patch of patches)
    {
        const delta = labels[patch.name] - patch.end;
        if(patch.size === 1) buf[patch.at] = delta & 0xff;
        else dv.setInt32(patch.at, delta, true);
    }
    return buf;
}

function run({ inline, crossPage = false })
{
    return new Promise(resolve => {
        const iterations = crossPage ? MISS_ITERATIONS : ITERATIONS;
        const emulator = new V86({
            autostart: false,
            memory_size: MEM_SIZE,
            disable_jit: 0,
            log_level: 0,
        });
        let timer;
        let started = 0;
        const finish = status => {
            clearTimeout(timer);
            const elapsedMs = performance.now() - started;
            try { emulator.stop(); } catch {}
            const cpu = emulator.v86.cpu;
            resolve({
                status,
                inline: cpu.get_jit_config?.(22) >>> 0,
                crossPage,
                elapsedMs: Math.round(elapsedMs * 100) / 100,
                eax: cpu.reg32[0] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                edx: cpu.reg32[2] >>> 0,
                ebx: cpu.reg32[3] >>> 0,
                compiledSites: cpu.wm.exports.jit_inline_dispatch_sites_compiled?.() >>> 0,
            });
        };

        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            if(inline !== undefined) cpu.set_jit_config(22, inline ? 1 : 0);
            if(crossPage) cpu.set_jit_config(1, 1);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(buildImage(iterations, crossPage).buffer);
            started = performance.now();
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            emulator.run();
        });
    });
}

function assertCorrect(result, iterations)
{
    const expected = Math.imul(iterations, 3) >>> 0;
    if(result.status !== "halt" || result.ebx !== COLD_MAGIC || result.ecx !== 0 || result.edx !== expected)
    {
        throw new Error(`incorrect execution: ${JSON.stringify(result)}, expected ebx=${COLD_MAGIC} edx=${expected}`);
    }
}

const defaultOn = await run({ inline:undefined });
assertCorrect(defaultOn, ITERATIONS);
if(defaultOn.inline !== 1)
{
    throw new Error(`inline dispatch must default ON in v86: ${JSON.stringify(defaultOn)}`);
}

const missOff = await run({ inline:false, crossPage:true });
const missOn = await run({ inline:true, crossPage:true });
assertCorrect(missOff, MISS_ITERATIONS);
assertCorrect(missOn, MISS_ITERATIONS);

const order = [false, true, true, false, false, true];
const samples = [];
for(const inline of order)
{
    const result = await run({ inline });
    assertCorrect(result, ITERATIONS);
    samples.push(result);
}

const median = values => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
};
const offMs = median(samples.filter(x => !x.inline).map(x => x.elapsedMs));
const onMs = median(samples.filter(x => x.inline).map(x => x.elapsedMs));

console.log("jit-inline-dispatch-default " + JSON.stringify(defaultOn));
console.log("jit-inline-dispatch-miss " + JSON.stringify({ off:missOff, on:missOn }));
console.log("jit-inline-dispatch-samples " + JSON.stringify(samples));
console.log("jit-inline-dispatch-summary " + JSON.stringify({
    offMs,
    onMs,
    speedupPercent: Math.round((offMs / onMs - 1) * 10000) / 100,
}));

if(!samples.some(x => x.inline && x.compiledSites > 0))
{
    throw new Error("inline resolver enabled but no compiled site was emitted");
}
