#!/usr/bin/env node
// Generic cold-phase benchmark for bounded asynchronous JIT compilation.
//
// A round-robin loop visits 64 distinct guest pages. With MAX_PAGES=1 every
// page becomes an independent wasm module, exposing the historical global
// one-Promise bottleneck without relying on a game address or Win32 thunk.

const debugBuild = process.env.V86_DEBUG === "1";
const { V86 } = await import(debugBuild ? "../build/libv86-debug.mjs" : "../build/libv86.mjs");

const BASE = 0x100000;
const PAGE_COUNT = 64;
const ITERATIONS = 500_000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30_000;

function buildImage()
{
    const buf = new Uint8Array(PAGE_COUNT * 0x1000);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + buf.length + 0x1000, true);
    dv.setUint32(0x1c, BASE + 0x100, true);

    for(let page = 0; page < PAGE_COUNT; page++)
    {
        let o = page === 0 ? 0x100 : page * 0x1000;
        const emit = (...xs) => { for(const x of xs) buf[o++] = x & 0xff; };
        if(page === 0)
        {
            emit(0xB9); dv.setUint32(o, ITERATIONS, true); o += 4; // mov ecx, iterations
            emit(0x31, 0xC0);                                    // xor eax,eax
            emit(0x49);                                          // dec ecx
            emit(0x0F, 0x84);                                    // jz done
            const relAt = o; o += 4;
            emit(0x40);                                          // inc eax
            emit(0xE9);                                          // jmp page 1
            const nextEnd = o + 4;
            dv.setInt32(o, BASE + 0x1000 - (BASE + nextEnd), true); o += 4;
            const done = o;
            emit(0xF4, 0xEB, 0xFE);                              // hlt; jmp $
            dv.setInt32(relAt, done - (relAt + 4), true);
        }
        else
        {
            emit(0x40);                                          // inc eax
            emit(0xE9);
            const nextOffset = page === PAGE_COUNT - 1 ? 0x107 : (page + 1) * 0x1000;
            const end = o + 4;
            dv.setInt32(o, nextOffset - end, true);
        }
    }
    return buf;
}

function run(maxPending, clearDuringRun = false, deferredQueue = false)
{
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: 0, log_level: 0 });
        let timer, clearTimer, clearCount = 0, done = false, startedAt = 0, haltedAt = 0;
        const finish = status => {
            if(done) return;
            done = true;
            clearTimeout(timer);
            clearInterval(clearTimer);
            const cpu = emulator.v86.cpu, w = cpu.wm.exports;
            const result = {
                maxPending,
                clearDuringRun,
                deferredQueue,
                clearCount,
                status,
                elapsedMs: +((haltedAt || performance.now()) - startedAt).toFixed(2),
                eax: cpu.reg32[0] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                started: w.jit_get_compile_started?.() ?? -1,
                completed: w.jit_get_compile_completed?.() ?? -1,
                pending: w.jit_get_compile_pending?.() ?? -1,
                highWater: w.jit_get_compile_pending_high_water?.() ?? -1,
                capSkips: w.jit_get_compile_cap_skips?.() ?? -1,
                deferredQueued: w.jit_get_compile_deferred_queued?.() ?? -1,
                deferredStarted: w.jit_get_compile_deferred_started?.() ?? -1,
                deferredDropped: w.jit_get_compile_deferred_dropped?.() ?? -1,
                deferredPending: w.jit_get_compile_deferred_pending?.() ?? -1,
                compileTotalMs: +((w.jit_get_compile_total_us?.() ?? 0) / 1000).toFixed(2),
                compileMaxMs: +((w.jit_get_compile_max_us?.() ?? 0) / 1000).toFixed(2),
            };
            try { emulator.stop(); } catch {}
            resolve(result);
        };
        emulator.bus.register("cpu-event-halt", () => {
            haltedAt = performance.now();
            // Let the last compilation Promise publish before inspecting the
            // pending-set invariant; architectural time remains haltedAt.
            setTimeout(() => finish("halt"), 50);
        });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, 1);             // one guest page per wasm module
            cpu.set_jit_config(25, maxPending);   // bounded async compile window
            cpu.set_jit_config(37, deferredQueue ? 1 : 0);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(buildImage().buffer);
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
            if(clearDuringRun)
            {
                clearTimer = setInterval(() => {
                    cpu.jit_clear_cache?.();
                    clearCount++;
                    if(clearCount >= 50) clearInterval(clearTimer);
                }, 1);
            }
        });
    });
}

const results = [];
for(const maxPending of [1, 2, 4, 1, 2, 4, 1, 2, 4]) results.push(await run(maxPending));
// Regression guard for per-generated-site memo ownership: clearing while async
// compiles are completing must never let a new module reuse an old memo slot.
results.push(await run(4, true));
results.push(await run(4, true, true));
// Controlled admission A/B at the production compile width.
for(const deferredQueue of [false, true, true, false, false, true])
    results.push(await run(2, false, deferredQueue));
console.log("jit-parallel-compile " + JSON.stringify(results));

const expectedEax = (ITERATIONS - 1) * PAGE_COUNT;
for(const result of results)
{
    if(result.status !== "halt" || result.eax !== expectedEax || result.ecx !== 0 || result.pending !== 0 || result.deferredPending !== 0)
    {
        console.error("FAIL: incorrect architectural result or unfinished compilation", result);
        process.exit(1);
    }
    if(result.highWater < 1 || result.highWater > result.maxPending)
    {
        console.error("FAIL: compile concurrency escaped its configured bound", result);
        process.exit(1);
    }
}
if(!results.filter(x => x.maxPending === 4).some(x => x.highWater > 1))
{
    console.error("FAIL: parallel mode never had more than one compilation in flight");
    process.exit(1);
}
const queuedRuns = results.filter(x => x.deferredQueue);
if(!queuedRuns.every(x => x.deferredQueued > 0 && x.deferredStarted > 0))
{
    console.error("FAIL: deferred queue never admitted hot pages", queuedRuns);
    process.exit(1);
}
const clearRun = results.find(x => x.clearDuringRun);
if(!clearRun || clearRun.clearCount === 0)
{
    console.error("FAIL: cache-clear race scenario did not execute", clearRun);
    process.exit(1);
}
