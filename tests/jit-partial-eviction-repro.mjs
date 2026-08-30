#!/usr/bin/env node
// When the wasm table runs out of slots, v86 discards every compiled module —
// and jit_dirty_page_ctx drops each page's hotness with it, so the entire
// working set has to re-cross JIT_THRESHOLD interpreted instructions before any
// of it runs compiled again. For a working set the size of the table that is far
// more interpretation than the flush itself costs.
//
// The fixture is the shape a map load has: one page called on every iteration
// (the hot core) interleaved with a stream of pages touched once each (parsers,
// decompressors), more of them than the table can hold. Config 43 must keep the
// hot core across exhaustion while still reclaiming the stream.
//
// The correctness guard is EAX: both modes execute the same arithmetic.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x1000;
const HOT_OFF = 0x2000;
const COLD_OFF = 0x10000;
const COLD_PAGES = 1000;
const LOOPN = 5000;
const SWEEPS = 4;
const THRESHOLD = 10_000;
const MEM_SIZE = 64 * 1024 * 1024;
const TIMEOUT_MS = 180_000;

const EXPECTED_EAX = SWEEPS * 2 * COLD_PAGES * LOOPN;

// mov ecx,LOOPN / inc eax / dec ecx / jnz / ret — long enough to cross the
// lowered threshold on its first visit, so every page really does get compiled.
function emitLeaf(buf, dv, off)
{
    let o = off;
    buf[o++] = 0xB9; dv.setUint32(o, LOOPN, true); o += 4;
    const loop = o;
    buf[o++] = 0x40;
    buf[o++] = 0x49;
    buf[o++] = 0x75;
    // The displacement must be computed before the store: `buf[o++] = f(o)`
    // evaluates the index first, so f would see the already-incremented o.
    const rel = loop - (o + 1);
    buf[o++] = rel & 0xFF;
    buf[o++] = 0xC3;
}

function buildImage()
{
    const len = COLD_OFF + (COLD_PAGES + 1) * 0x1000;
    const buf = new Uint8Array(len);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + len, true);
    dv.setUint32(0x18, BASE + len + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    emitLeaf(buf, dv, HOT_OFF);
    for(let i = 0; i < COLD_PAGES; i++) emitLeaf(buf, dv, COLD_OFF + i * 0x1000);

    let o = ENTRY_OFF;
    buf[o++] = 0xBC; dv.setUint32(o, 0x600000, true); o += 4;  // mov esp
    buf[o++] = 0x31; buf[o++] = 0xC0;                          // xor eax, eax
    buf[o++] = 0xBB; dv.setUint32(o, SWEEPS, true); o += 4;    // mov ebx, SWEEPS
    const outer = o;
    buf[o++] = 0xBA; dv.setUint32(o, BASE + COLD_OFF, true); o += 4; // mov edx, cold
    const inner = o;
    buf[o++] = 0xE8; dv.setInt32(o, (BASE + HOT_OFF) - (BASE + o + 4), true); o += 4; // call hot
    buf[o++] = 0xFF; buf[o++] = 0xD2;                          // call edx
    buf[o++] = 0x81; buf[o++] = 0xC2; dv.setUint32(o, 0x1000, true); o += 4; // add edx, 0x1000
    buf[o++] = 0x81; buf[o++] = 0xFA;
    dv.setUint32(o, BASE + COLD_OFF + COLD_PAGES * 0x1000, true); o += 4;   // cmp edx, end
    buf[o++] = 0x0F; buf[o++] = 0x82; dv.setInt32(o, inner - (o + 4), true); o += 4; // jb inner
    buf[o++] = 0x4B;                                           // dec ebx
    buf[o++] = 0x0F; buf[o++] = 0x85; dv.setInt32(o, outer - (o + 4), true); o += 4; // jnz outer
    buf[o++] = 0xF4; buf[o++] = 0xEB; buf[o++] = 0xFE;         // hlt; jmp $

    if(o >= HOT_OFF) throw new Error("driver overran the hot page");
    return buf;
}

const image = buildImage();

function run(partialEviction)
{
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, log_level: 0 });
        let timer, startedAt = 0;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch {}
            const cpu = emulator.v86.cpu;
            const ex = cpu.wm.exports;
            resolve({
                status, partialEviction,
                elapsedMs: +(performance.now() - startedAt).toFixed(2),
                eax: cpu.reg32[0] >>> 0,
                flushes: ex["jit_get_cache_flushes"]?.() >>> 0,
                evictions: ex["jit_get_partial_evictions"]?.() >>> 0,
                evicted: ex["jit_get_evicted_modules"]?.() >>> 0,
                fallbacks: ex["jit_get_eviction_fallbacks"]?.() >>> 0,
                applied: cpu.get_jit_config?.(43) >>> 0,
                started: ex["jit_get_compile_started"]?.() >>> 0,
                completed: ex["jit_get_compile_completed"]?.() >>> 0,
                capSkips: ex["jit_get_compile_cap_skips"]?.() >>> 0,
            });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(26, THRESHOLD);
            // Without these the two-in-flight compile cap refuses almost every
            // request and only ~130 modules are ever built, so the table never
            // fills and the fixture silently tests nothing.
            cpu.set_jit_config(25, 8);
            cpu.set_jit_config(37, 1);
            cpu.set_jit_config(43, partialEviction ? 1 : 0);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(image.buffer.slice(0));
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

let failed = false;
const times = { off: [], on: [] };
const flushes = { off: [], on: [] };

for(const on of [false, true, true, false, false, true])
{
    const r = await run(on);
    console.log("jit-partial-eviction " + JSON.stringify(r));
    if(r.status !== "halt" || r.eax !== EXPECTED_EAX)
    {
        console.error(`FAIL: expected eax=${EXPECTED_EAX}, got ${r.eax} (${r.status})`);
        failed = true;
    }
    if(r.applied !== (on ? 1 : 0)) { console.error("FAIL: config 43 not applied"); failed = true; }
    if(!on && r.evictions !== 0) { console.error("FAIL: evicted while disabled"); failed = true; }
    if(!on && r.flushes === 0) { console.error("FAIL: fixture never exhausted the table"); failed = true; }
    if(on && r.flushes !== 0) { console.error(`FAIL: ${r.flushes} full flush(es) survived eviction`); failed = true; }
    times[on ? "on" : "off"].push(r.elapsedMs);
    flushes[on ? "on" : "off"].push(r.flushes);
}

const median = a => a.slice().sort((x, y) => x - y)[a.length >> 1];
console.log(`MEDIAN off=${median(times.off).toFixed(2)}ms on=${median(times.on).toFixed(2)}ms ` +
    `delta=${(((median(times.off) - median(times.on)) / median(times.off)) * 100).toFixed(2)}% ` +
    `flushes off=${median(flushes.off)} on=${median(flushes.on)}`);
console.log(failed ? "RESULT FAIL" : "RESULT PASS");
process.exit(failed ? 1 : 0);
