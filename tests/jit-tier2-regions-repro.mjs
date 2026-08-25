#!/usr/bin/env node
// Generic profile-guided Tier-2 region benchmark.
//
// The hot path alternates between page 0 and page 12. Page 0 also exposes ten
// statically reachable but never-taken cold branches. Blindly raising MAX_PAGES
// spends the wider Tier-2 budget on those cold pages before reaching page 12;
// the profile-guided mode must instead coalesce the two modules actually seen at
// runtime. No BFME address or game-specific signature is involved.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = 0x20;
const HOT_PAGE = 12;
const ITER = 4_000_000;
const THRESHOLD = 200_000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30_000;

function image(iterations)
{
    const buf = new Uint8Array((HOT_PAGE + 1) * 0x1000);
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

    emit(0x31, 0xC0);                       // xor eax,eax (all cold JNZs false)
    emit(0xB9); u32(iterations);            // mov ecx,ITER
    label("loop");
    for(let i = 1; i <= 10; i++)
    {
        emit(0x85, 0xC0);                   // test eax,eax
        emit(0x0F, 0x85); rel32(`cold${i}`);// jnz cold page
    }
    emit(0x49);                             // dec ecx
    emit(0x0F, 0x84); rel32("done");       // jz done
    emit(0xE9); rel32("hot");              // jmp hot page (real module exit)
    label("done"); emit(0xF4, 0xEB, 0xFE); // hlt; jmp $

    for(let i = 1; i <= 10; i++)
    {
        o = i * 0x1000;
        label(`cold${i}`);
        emit(0xB8); u32(0xBAD00000 + i);     // failure marker
        emit(0xF4, 0xEB, 0xFE);
    }
    o = HOT_PAGE * 0x1000;
    label("hot"); emit(0xE9); rel32("loop");

    for(const p of patches) dv.setInt32(p.at, labels[p.to] - p.end, true);
    return buf;
}

function run(mode)
{
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: 0, log_level: 0 });
        let timer, startedAt = 0, done = false;
        const finish = status => {
            if(done) return;
            done = true;
            clearTimeout(timer);
            const cpu = emulator.v86.cpu, w = cpu.wm.exports;
            const result = {
                mode, status,
                elapsedMs: +(performance.now() - startedAt).toFixed(2),
                eax: cpu.reg32[0] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                pages: w.jit_get_tier2_page_count?.() ?? -1,
                promotions: w.jit_get_tier2_promotions?.() ?? -1,
                profiledExits: w.jit_get_tier2_profiled_exits?.() ?? -1,
                regionPromotions: w.jit_get_tier2_region_promotions?.() ?? -1,
                regionSeeds: w.jit_get_tier2_region_seeds?.() ?? -1,
            };
            try { emulator.stop(); } catch {}
            resolve(result);
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.set_jit_config(1, 1);       // Tier-1 modules: one page
            cpu.set_jit_config(17, 8);      // equal Tier-2 code-size budget
            cpu.set_jit_config(20, 256);
            cpu.set_jit_config(23, (mode === "profiled" || mode === "profile-armed") ? 1 : 0);
            cpu.set_jit_config(15, mode === "off" ? 0 :
                ((mode === "profile-armed" || mode === "legacy-armed") ? 50_000_000 : THRESHOLD));
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(image(ITER).buffer);
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

const sequence = [
    "off", "legacy-armed", "profile-armed", "legacy", "profiled",
    "profiled", "legacy", "profile-armed", "legacy-armed", "off",
    "off", "legacy-armed", "profile-armed", "legacy", "profiled",
];
const results = [];
for(const mode of sequence)
{
    const result = await run(mode);
    results.push(result);
    console.log("jit-tier2-regions-run " + JSON.stringify(result));
    if(result.status !== "halt" || result.ecx !== 0 || result.eax !== 0)
    {
        console.error("FAIL: incorrect architectural result");
        process.exit(1);
    }
    if(mode === "profiled" && (result.regionPromotions < 1 || result.regionSeeds < 1))
    {
        console.error("FAIL: profile-guided region did not form");
        process.exit(1);
    }
}
const median = xs => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const med = Object.fromEntries(["off", "legacy-armed", "profile-armed", "legacy", "profiled"].map(mode => [mode,
    median(results.filter(x => x.mode === mode).map(x => x.elapsedMs))]));
console.log("jit-tier2-regions " + JSON.stringify({
    iterations: ITER,
    offMedianMs: med.off,
    legacyArmedMedianMs: med["legacy-armed"],
    profileArmedMedianMs: med["profile-armed"],
    legacyMedianMs: med.legacy,
    profiledMedianMs: med.profiled,
    legacyArmedCostPct: +((med["legacy-armed"] / med.off - 1) * 100).toFixed(2),
    profileIncrementalCostPct: +((med["profile-armed"] / med["legacy-armed"] - 1) * 100).toFixed(2),
    gainVsOffPct: +((med.off / med.profiled - 1) * 100).toFixed(2),
    gainVsLegacyPct: +((med.legacy / med.profiled - 1) * 100).toFixed(2),
}));
