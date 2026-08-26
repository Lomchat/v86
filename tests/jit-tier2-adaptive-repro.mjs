#!/usr/bin/env node
// Generic phase-change regression for the bounded adaptive Tier-2 hot set.
//
// Phase A alternates between pages 0/1 until they fill a two-page Tier-2 set.
// Phase B then moves permanently to pages 2/3. The legacy policy freezes the
// startup pages forever; the adaptive policy must retain the same hard cap while
// replacing them after sparse maintenance samples. No game address is involved.

const debugBuild = process.env.V86_DEBUG === "1";
const { V86 } = await import(debugBuild ? "../build/libv86-debug.mjs" : "../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = 0x20;
const PHASE_A_ITER = 2_000_000;
const PHASE_B_ITER = 24_000_000;
const THRESHOLD = 20_000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30_000;

function image()
{
    const buf = new Uint8Array(4 * 0x1000);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + buf.length + 0x1000, true);
    dv.setUint32(0x1c, BASE + ENTRY, true);

    let o = ENTRY;
    const labels = {}, patches = [];
    const label = n => { labels[n] = o; };
    const emit = (...xs) => { for(const x of xs) buf[o++] = x & 0xff; };
    const u32 = x => { dv.setUint32(o, x >>> 0, true); o += 4; };
    const rel32 = n => { patches.push({ at: o, end: o + 4, to: n }); emit(0, 0, 0, 0); };

    emit(0xB9); u32(PHASE_A_ITER);           // mov ecx, phase-A iterations
    label("aLoop");
    emit(0x49);                              // dec ecx
    emit(0x0F, 0x84); rel32("bStart");       // jz phase B
    emit(0xE9); rel32("aPeer");              // jmp page 1

    o = 0x1000;
    label("aPeer"); emit(0xE9); rel32("aLoop");

    o = 0x2000;
    label("bStart");
    emit(0xB9); u32(PHASE_B_ITER);           // mov ecx, phase-B iterations
    label("bLoop");
    emit(0x49);
    emit(0x0F, 0x84); rel32("done");
    emit(0xE9); rel32("bPeer");              // jmp page 3
    label("done"); emit(0x31, 0xC0, 0xF4, 0xEB, 0xFE); // eax=0; hlt; jmp $

    o = 0x3000;
    label("bPeer"); emit(0xE9); rel32("bLoop");

    for(const p of patches) dv.setInt32(p.at, labels[p.to] - p.end, true);
    return buf;
}

function steadyImage()
{
    const buf = new Uint8Array(2 * 0x1000);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + buf.length + 0x1000, true);
    dv.setUint32(0x1c, BASE + ENTRY, true);

    let o = ENTRY;
    const patches = [];
    const emit = (...xs) => { for(const x of xs) buf[o++] = x & 0xff; };
    const u32 = x => { dv.setUint32(o, x >>> 0, true); o += 4; };
    const rel32 = target => { const at = o, end = o + 4; emit(0, 0, 0, 0); patches.push({ at, end, target }); };
    emit(0xB9); u32(PHASE_A_ITER + PHASE_B_ITER);
    const loop = o;
    emit(0x49);
    emit(0x0F, 0x84); rel32(null);
    emit(0xE9); rel32(0x1000);
    const done = o;
    emit(0x31, 0xC0, 0xF4, 0xEB, 0xFE);
    o = 0x1000;
    emit(0xE9); rel32(loop);
    patches[0].target = done;
    for(const p of patches) dv.setInt32(p.at, p.target - p.end, true);
    return buf;
}

function run(adaptive, program = image(), regions = false)
{
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: 0, log_level: 0 });
        let timer, startedAt = 0, done = false;
        const finish = status => {
            if(done) return;
            done = true;
            clearTimeout(timer);
            const cpu = emulator.v86.cpu, w = cpu.wm.exports;
            const count = w.jit_get_tier2_page_count?.() ?? 0;
            const pages = [];
            for(let i = 0; i < count; i++) pages.push(w.jit_get_tier2_page_at(i) >>> 0);
            const result = {
                adaptive, regions, status,
                elapsedMs: +(performance.now() - startedAt).toFixed(2),
                eax: cpu.reg32[0] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                pages,
                promotions: w.jit_get_tier2_promotions?.() ?? -1,
                maintenanceSamples: w.jit_get_tier2_maintenance_samples?.() ?? -1,
                evictions: w.jit_get_tier2_page_evictions?.() ?? -1,
                blocked: w.jit_get_tier2_blocked_by_cap?.() ?? -1,
                regionPromotions: w.jit_get_tier2_region_promotions?.() ?? -1,
                regionSeeds: w.jit_get_tier2_region_seeds?.() ?? -1,
                profiledExits: w.jit_get_tier2_profiled_exits?.() ?? -1,
                regionCandidates: w.jit_get_tier2_region_candidates?.() ?? -1,
                regionRejectedTarget: w.jit_get_tier2_region_rejected_target?.() ?? -1,
                regionRejectedBudget: w.jit_get_tier2_region_rejected_budget?.() ?? -1,
            };
            try { emulator.stop(); } catch {}
            resolve(result);
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.set_jit_config(1, 1);             // one page per Tier-1 module
            cpu.set_jit_config(17, 1);            // isolate replacement/profile admission
            cpu.set_jit_config(20, 2);            // intentionally tiny retained set
            cpu.set_jit_config(23, regions ? 1 : 0);
            cpu.set_jit_config(24, adaptive ? 1 : 0);
            cpu.set_jit_config(15, THRESHOLD);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(program.buffer);
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

const legacy = await run(false);
const adaptive = await run(true);
const steadyLegacy = await run(false, steadyImage());
const steadyAdaptive = await run(true, steadyImage());
const adaptiveRegions = await run(true, image(), true);
console.log("jit-tier2-adaptive " + JSON.stringify({ legacy, adaptive, steadyLegacy, steadyAdaptive, adaptiveRegions }));

for(const result of [legacy, adaptive, steadyLegacy, steadyAdaptive, adaptiveRegions])
{
    if(result.status !== "halt" || result.eax !== 0 || result.ecx !== 0 || result.pages.length !== 2)
    {
        console.error("FAIL: incorrect architectural result or Tier-2 bound");
        process.exit(1);
    }
}
if(legacy.evictions !== 0 || legacy.maintenanceSamples !== 0)
{
    console.error("FAIL: legacy kill-switch performed adaptive maintenance");
    process.exit(1);
}
if(adaptive.evictions < 1 || adaptive.maintenanceSamples < 1 ||
    !adaptive.pages.some(page => page >= BASE + 2 * 0x1000))
{
    console.error("FAIL: adaptive Tier-2 did not replace the saturated startup set");
    process.exit(1);
}
if(steadyAdaptive.evictions !== 0 || steadyAdaptive.maintenanceSamples < 1 ||
    steadyLegacy.evictions !== 0)
{
    console.error("FAIL: steady-state maintenance changed the retained hot set");
    process.exit(1);
}
if(adaptiveRegions.profiledExits < 1 || adaptiveRegions.regionCandidates < 1 ||
    !adaptiveRegions.pages.some(page => page >= BASE + 2 * 0x1000))
{
    console.error("FAIL: saturated adaptive phase did not retain successor profiles for later region formation");
    process.exit(1);
}
