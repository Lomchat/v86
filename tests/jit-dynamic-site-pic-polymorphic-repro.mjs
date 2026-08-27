#!/usr/bin/env node
// Diagnostic benchmark for the site-PIC miss classifier (config idx 31).
//
// One cross-page leaf RET alternates between two caller continuations. The
// production one-way PIC must remain architecturally exact, while the shadow
// second way should predict almost every post-warmup target miss as a hit.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = 0x20;
const LEAF = 0x1000;
const ITERATIONS = 1_000_000;
const MAGIC = 0x52A7B00C;
const TIMEOUT_MS = 30_000;

function image(callCount = 2)
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
    const callLeaf = () => {
        emit(0xE8);
        dv.setInt32(o, LEAF - (o + 4), true);
        o += 4;
    };
    emit(0xBC); u32(0x300000);              // mov esp, 0x300000
    emit(0xB9); u32(ITERATIONS);            // mov ecx, iterations
    emit(0x31, 0xC0);                       // xor eax, eax
    const loop = o;
    for(let i = 0; i < callCount; i++) {
        callLeaf();                         // distinct return target
        emit(0x90);                         // keep continuations distinct
    }
    emit(0x49);                             // dec ecx
    emit(0x75, (loop - (o + 2)) & 0xFF);    // jnz loop
    emit(0xBB); u32(MAGIC);                 // mov ebx, magic
    emit(0xF4, 0xEB, 0xFE);                 // hlt; jmp $

    o = LEAF;
    emit(0x83, 0xC0, 0x03);                 // add eax, 3
    emit(0xC3);                             // ret
    return bytes;
}

const result = await new Promise(resolve => {
    const emulator = new V86({
        autostart: false,
        memory_size: 16 * 1024 * 1024,
        disable_jit: 0,
        log_level: 0,
    });
    let timer;
    emulator.bus.register("cpu-event-halt", () => finish("halt"));
    const finish = status => {
        clearTimeout(timer);
        const cpu = emulator.v86.cpu;
        const ex = cpu.wm.exports;
        try { emulator.stop(); } catch {}
        resolve({
            status,
            eax: cpu.reg32[0] >>> 0,
            ecx: cpu.reg32[1] >>> 0,
            ebx: cpu.reg32[3] >>> 0,
            calls: Number(ex.jit_dynamic_chain_site_pic_diag_calls()),
            targetMisses: Number(ex.jit_dynamic_chain_site_pic_diag_target_misses()),
            secondWayHits: Number(ex.jit_dynamic_chain_site_pic_diag_second_way_hits()),
            epochMisses: Number(ex.jit_dynamic_chain_site_pic_diag_epoch_misses()),
            guardMisses: Number(ex.jit_dynamic_chain_site_pic_diag_guard_misses()),
            resolverHits: Number(ex.jit_dynamic_chain_site_pic_diag_resolver_hits()),
        });
    };
    emulator.add_listener("emulator-loaded", () => {
        const cpu = emulator.v86.cpu;
        cpu.reboot_internal();
        cpu.reset_memory();
        cpu.set_jit_config(1, 1);
        cpu.set_jit_config(12, 1);
        cpu.set_jit_config(30, 1);
        cpu.set_jit_config(31, 1);
        cpu.set_jit_config(32, 0);
        cpu.set_jit_config(33, 0);
        cpu.wm.exports.jit_dynamic_chain_site_pic_diag_reset();
        cpu.jit_clear_cache?.();
        cpu.load_multiboot(image().buffer);
        timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
        emulator.run();
    });
});

const expected = Math.imul(ITERATIONS, 6) >>> 0;
console.log("jit-dynamic-site-pic-polymorphic " + JSON.stringify(result));
if(result.status !== "halt" || result.eax !== expected || result.ecx !== 0 || result.ebx !== MAGIC)
    throw new Error(`incorrect execution, expected eax=${expected}`);
if(result.targetMisses < ITERATIONS || result.secondWayHits < ITERATIONS - 10)
    throw new Error("shadow second way did not absorb the alternating targets");

function runTimed(secondWay, fourWay = false, callCount = 2)
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
                secondWay: cpu.get_jit_config?.(32) >>> 0,
                fourWay: cpu.get_jit_config?.(33) >>> 0,
                elapsedMs: Math.round(elapsedMs * 100) / 100,
                eax: cpu.reg32[0] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                ebx: cpu.reg32[3] >>> 0,
            });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, 1);
            cpu.set_jit_config(12, 1);
            cpu.set_jit_config(30, 1);
            cpu.set_jit_config(31, 0);
            cpu.set_jit_config(32, secondWay ? 1 : 0);
            cpu.set_jit_config(33, fourWay ? 1 : 0);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(image(callCount).buffer);
            started = performance.now();
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            emulator.run();
        });
    });
}

const order = [false, true, true, false, false, true];
const samples = [];
for(const secondWay of order)
{
    const sample = await runTimed(secondWay);
    if(sample.status !== "halt" || sample.eax !== expected || sample.ecx !== 0 || sample.ebx !== MAGIC)
        throw new Error(`incorrect timed execution: ${JSON.stringify(sample)}`);
    samples.push(sample);
}
const median = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const oneWayMs = median(samples.filter(x => !x.secondWay).map(x => x.elapsedMs));
const twoWayMs = median(samples.filter(x => x.secondWay).map(x => x.elapsedMs));
console.log("jit-dynamic-site-pic-polymorphic-samples " + JSON.stringify(samples));
console.log("jit-dynamic-site-pic-polymorphic-summary " + JSON.stringify({
    oneWayMs,
    twoWayMs,
    speedupPercent: Math.round((oneWayMs / twoWayMs - 1) * 10000) / 100,
}));

const fourWayOrder = [1, 2, 4, 4, 2, 1, 1, 2, 4];
const fourWaySamples = [];
const fourWayExpected = Math.imul(ITERATIONS, 12) >>> 0;
for(const ways of fourWayOrder)
{
    const sample = await runTimed(ways >= 2, ways >= 4, 4);
    if(sample.status !== "halt" || sample.eax !== fourWayExpected || sample.ecx !== 0 || sample.ebx !== MAGIC)
        throw new Error(`incorrect four-way execution: ${JSON.stringify(sample)}`);
    fourWaySamples.push({ ...sample, ways });
}
const byWays = ways => median(fourWaySamples.filter(x => x.ways === ways).map(x => x.elapsedMs));
const oneTargetMs = byWays(1);
const twoTargetMs = byWays(2);
const fourTargetMs = byWays(4);
console.log("jit-dynamic-site-pic-four-way-samples " + JSON.stringify(fourWaySamples));
console.log("jit-dynamic-site-pic-four-way-summary " + JSON.stringify({
    oneTargetMs,
    twoTargetMs,
    fourTargetMs,
    speedupVsOnePercent: Math.round((oneTargetMs / fourTargetMs - 1) * 10000) / 100,
    speedupVsTwoPercent: Math.round((twoTargetMs / fourTargetMs - 1) * 10000) / 100,
}));
