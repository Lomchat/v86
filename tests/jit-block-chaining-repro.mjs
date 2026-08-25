#!/usr/bin/env node
// Cross-page direct-jump loop for block chaining.
//
// MAX_PAGES=1 forces page0 and page1 into separate JIT modules:
//   page0: dec ecx; jz done; jmp page1
//   page1: jmp page0
//
// The retained implementation memoizes each exact target behind a table-generation
// guard, performs the preemption guard directly in generated wasm, then spills
// registers only for a proven tail-call.
// This test A/Bs the exact same loop with config 4 OFF and ON, while checking
// cold misses, budget exits, arithmetic state and the default kill switch.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x20;
const PAGE1_OFF = 0x1000;
const ITER = 4_000_000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30000;

function build_image(iterations)
{
    const buf = new Uint8Array(PAGE1_OFF + 16);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + PAGE1_OFF + 16, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const labels = {}, patches = [];
    const label = n => { labels[n] = o; };
    const emit = (...bytes) => { for(const b of bytes) buf[o++] = b & 0xff; };
    const u32 = value => { dv.setUint32(o, value >>> 0, true); o += 4; };
    const rel8 = n => { patches.push({ at: o, sz: 1, end: o + 1, to: n }); emit(0); };
    const rel32 = n => { patches.push({ at: o, sz: 4, end: o + 4, to: n }); emit(0, 0, 0, 0); };

    emit(0xB9); u32(iterations);           // mov ecx, ITER
    label("page0");
    emit(0x49);                            // dec ecx
    emit(0x74); rel8("done");             // jz done
    emit(0xE9); rel32("page1");           // jmp page1
    label("done");
    emit(0xF4); emit(0xEB, 0xFE);          // hlt; jmp $

    o = PAGE1_OFF;
    label("page1");
    emit(0xE9); rel32("page0");           // jmp page0

    for(const p of patches)
    {
        const d = labels[p.to] - p.end;
        if(p.sz === 1) buf[p.at] = d & 0xff;
        else dv.setInt32(p.at, d, true);
    }
    return buf;
}

function run(chaining, iterations = ITER)
{
    return new Promise(resolve => {
        const emulator = new V86({
            autostart: false,
            memory_size: MEM_SIZE,
            disable_jit: 0,
            log_level: 0,
        });
        let halted = false, timer, startedAt = 0;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            const dget = cpu.wm.exports["profiler_dispatch_stat_get"];
            resolve({
                status,
                elapsedMs: performance.now() - startedAt,
                ecx: cpu.reg32[1] >>> 0,
                chaining: cpu.get_jit_config ? cpu.get_jit_config(4) >>> 0 : 0,
                reentry: dget ? dget(1) : 0,
                chainableFallback: dget ? dget(2) : 0,
                chainedEdge: dget ? dget(5) : 0,
                budgetExit: dget ? dget(6) : 0,
                miss: dget ? dget(7) : 0,
                sites: cpu.wm.exports["jit_block_chain_sites_compiled"]?.() ?? -1,
                exactHits: cpu.wm.exports["jit_exact_dispatch_hits"]?.() ?? -1,
                exactMisses: cpu.wm.exports["jit_exact_dispatch_misses"]?.() ?? -1,
                memoHighWater: cpu.wm.exports["jit_chain_memo_high_water"]?.() ?? -1,
                memoOverflows: cpu.wm.exports["jit_chain_memo_overflows"]?.() ?? -1,
            });
        };

        emulator.bus.register("cpu-event-halt", () => {
            halted = true;
            finish("halt");
        });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, 1); // MAX_PAGES=1, force cross-page module exits
            const defaultChaining = cpu.get_jit_config(4) >>> 0;
            if(defaultChaining !== 0)
            {
                finish(`BAD_DEFAULT_${defaultChaining}`);
                return;
            }
            cpu.set_jit_config(4, chaining ? 1 : 0);
            cpu.wm.exports["set_dispatch_stats"]?.(1);
            cpu.wm.exports["profiler_init"]?.();
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(build_image(iterations).buffer);
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

const sequence = [false, true, true, false, false, true];
const results = [];
for(const chaining of sequence)
{
    const result = await run(chaining);
    results.push(result);
    console.log("jit-block-chaining-run " + JSON.stringify(result));

    if(result.status !== "halt" || result.ecx !== 0)
    {
        console.error("FAIL: loop did not halt cleanly with ecx=0");
        process.exit(1);
    }
    if(chaining && (result.chaining !== 1 || result.chainedEdge <= 0 || result.sites <= 0))
    {
        console.error("FAIL: chaining was enabled but no direct edge/site was observed");
        process.exit(1);
    }
    if(chaining && (result.exactHits <= 0 || result.memoHighWater <= 0 || result.memoOverflows !== 0))
    {
        console.error("FAIL: exact target memo did not fill cleanly");
        process.exit(1);
    }
    if(!chaining && (result.chaining !== 0 || result.chainedEdge !== 0))
    {
        console.error("FAIL: disabled chaining executed a chained edge");
        process.exit(1);
    }
}

const median = values => {
    const s = values.slice().sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
};
const offMs = median(results.filter(x => !x.chaining).map(x => x.elapsedMs));
const onMs = median(results.filter(x => x.chaining).map(x => x.elapsedMs));
const summary = {
    iterations: ITER,
    offMedianMs: +offMs.toFixed(2),
    onMedianMs: +onMs.toFixed(2),
    throughputGainPct: +((offMs / onMs - 1) * 100).toFixed(2),
};
console.log("jit-block-chaining " + JSON.stringify(summary));

process.exit(0);
