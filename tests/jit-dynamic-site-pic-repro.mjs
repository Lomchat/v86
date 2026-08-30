#!/usr/bin/env node
// Correctness + throughput benchmark for config idx 30.
//
// A cross-page CALL/RET loop (MAX_PAGES=1) makes one AbsoluteEip site return to
// the same caller millions of times. Dynamic RET chaining remains authoritative;
// the optional per-site PIC may only remove its repeated resolver lookup.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = 0x20;
const LEAF = 0x1000;
const ITERATIONS = 4_000_000;
const MAGIC = 0x51A7B00B;
const TIMEOUT_MS = 30_000;

function image()
{
    const bytes = new Uint8Array(LEAF + 16);
    const dv = new DataView(bytes.buffer);
    dv.setUint32(0x00, 0x1BADB002, true);
    dv.setUint32(0x04, 0x10000, true);
    dv.setUint32(0x08, (-(0x1BADB002 + 0x10000)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + bytes.length, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY, true);

    let o = ENTRY;
    const emit = (...xs) => { for(const x of xs) bytes[o++] = x; };
    const u32 = x => { dv.setUint32(o, x >>> 0, true); o += 4; };
    emit(0xBC); u32(0x300000);              // mov esp, 0x300000
    emit(0xB9); u32(ITERATIONS);            // mov ecx, iterations
    emit(0x31, 0xC0);                       // xor eax, eax
    const loop = o;
    emit(0xE8);                             // call leaf
    const rel = LEAF - (o + 4);
    dv.setInt32(o, rel, true); o += 4;
    emit(0x49);                             // dec ecx
    emit(0x75, (loop - (o + 2)) & 0xFF);    // jnz loop
    emit(0xBB); u32(MAGIC);                 // mov ebx, magic
    emit(0xF4, 0xEB, 0xFE);                 // hlt; jmp $

    o = LEAF;
    emit(0x83, 0xC0, 0x03);                 // add eax, 3
    emit(0xC3);                             // ret
    return bytes;
}

function run(pic, secondWay = false, budgetFastExit = true, cycleLimit = 100_003)
{
    return new Promise(resolve => {
        const emulator = new V86({
            autostart: false,
            memory_size: 16 * 1024 * 1024,
            disable_jit: 0,
            log_level: 0,
        });
        let timer;
        let started = 0;
        const finish = status => {
            clearTimeout(timer);
            const cpu = emulator.v86.cpu;
            const elapsedMs = performance.now() - started;
            try { emulator.stop(); } catch {}
            resolve({
                status,
                pic: cpu.get_jit_config?.(30) >>> 0,
                secondWay: cpu.get_jit_config?.(32) >>> 0,
                budgetFastExit: cpu.get_jit_config?.(41) >>> 0,
                cycleLimit,
                elapsedMs: Math.round(elapsedMs * 100) / 100,
                eax: cpu.reg32[0] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                ebx: cpu.reg32[3] >>> 0,
                compiledSites: cpu.wm.exports.jit_dynamic_chain_site_pic_compiled?.() >>> 0,
                highWater: cpu.wm.exports.jit_dynamic_chain_site_pic_high_water?.() >>> 0,
                overflows: cpu.wm.exports.jit_dynamic_chain_site_pic_overflows?.() >>> 0,
            });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, 1);   // one page/module
            cpu.set_jit_config(12, 1);  // dynamic RET chaining
            cpu.set_jit_config(30, pic ? 1 : 0);
            cpu.set_jit_config(32, secondWay ? 1 : 0);
            cpu.set_jit_config(41, budgetFastExit ? 1 : 0);
            const hp = cpu.wm.exports.get_hypercall_page_ptr() >>> 0;
            new DataView(cpu.wasm_memory.buffer).setUint32(hp, cycleLimit >>> 0, true);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(image().buffer);
            started = performance.now();
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            emulator.run();
        });
    });
}

function check(r)
{
    const expected = Math.imul(ITERATIONS, 3) >>> 0;
    if(r.status !== "halt" || r.eax !== expected || r.ecx !== 0 || r.ebx !== MAGIC)
        throw new Error(`incorrect execution: ${JSON.stringify(r)}, expected eax=${expected}`);
}

const order = [false, true, true, false, false, true];
const samples = [];
for(const pic of order)
{
    const result = await run(pic);
    check(result);
    samples.push(result);
}

const median = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const offMs = median(samples.filter(x => !x.pic).map(x => x.elapsedMs));
const onMs = median(samples.filter(x => x.pic).map(x => x.elapsedMs));
console.log("jit-dynamic-site-pic-samples " + JSON.stringify(samples));
console.log("jit-dynamic-site-pic-summary " + JSON.stringify({
    offMs,
    onMs,
    speedupPercent: Math.round((offMs / onMs - 1) * 10000) / 100,
}));

if(!samples.some(x => x.pic && x.compiledSites > 0 && x.highWater > 0 && x.overflows === 0))
    throw new Error("site PIC enabled but no memoized site was generated");

// A second way must not significantly penalize a genuinely monomorphic site.
const wayOrder = [false, true, true, false, false, true];
const waySamples = [];
for(const secondWay of wayOrder)
{
    const result = await run(true, secondWay);
    check(result);
    waySamples.push(result);
}
const oneWayMs = median(waySamples.filter(x => !x.secondWay).map(x => x.elapsedMs));
const twoWayMs = median(waySamples.filter(x => x.secondWay).map(x => x.elapsedMs));
console.log("jit-dynamic-site-pic-monomorphic-way-samples " + JSON.stringify(waySamples));
console.log("jit-dynamic-site-pic-monomorphic-way-summary " + JSON.stringify({
    oneWayMs,
    twoWayMs,
    overheadPercent: Math.round((twoWayMs / oneWayMs - 1) * 10000) / 100,
}));

// Once a site has observed an exhausted cycle budget, the shared resolver can
// only repeat that same synchronous check and return -1. Compare the generated
// local exit against the historical resolver call over many real budget ends.
const budgetOrder = [false, true, true, false, false, true];
const budgetSamples = [];
for(const budgetFastExit of budgetOrder)
{
    const result = await run(true, true, budgetFastExit, 64);
    check(result);
    budgetSamples.push(result);
}
const budgetOffMs = median(budgetSamples.filter(x => !x.budgetFastExit).map(x => x.elapsedMs));
const budgetOnMs = median(budgetSamples.filter(x => x.budgetFastExit).map(x => x.elapsedMs));
console.log("jit-dynamic-budget-fast-exit-samples " + JSON.stringify(budgetSamples));
console.log("jit-dynamic-budget-fast-exit-summary " + JSON.stringify({
    budgetOffMs,
    budgetOnMs,
    speedupPercent: Math.round((budgetOffMs / budgetOnMs - 1) * 10000) / 100,
}));
