#!/usr/bin/env node
// A hot loop whose head sits in the last 16 bytes of a page and whose body
// continues into the next (physically contiguous) page. Without config 50 the
// JIT never records or compiles an entry there, so the block runs interpreted
// on every iteration; with it, the block compiles under the same contiguity
// proof config 38 applies to a crossing instruction. Both modes must produce
// the same registers; the enabled mode must interpret far fewer steps.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const HEAD_OFF = 0x0ff4;
const START_OFF = 0x1100;
const ITER = 3_000_000;
const INCREMENTS = 24;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30_000;

function buildImage()
{
    const buf = new Uint8Array(0x2000);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + START_OFF, true);
    const rel32 = (at, target) => dv.setInt32(at, target - (at + 4), true);

    // Loop head at page+0xff4: INCs run past the page end into the next page.
    let o = HEAD_OFF;
    for(let i = 0; i < INCREMENTS; i++) buf[o++] = 0x40;       // inc eax
    buf[o++] = 0x49;                                             // dec ecx
    buf[o++] = 0x0F; buf[o++] = 0x85; rel32(o, HEAD_OFF); o += 4; // jnz head
    buf[o++] = 0xF4; buf[o++] = 0xEB; buf[o++] = 0xFE;           // hlt; jmp $

    o = START_OFF;
    buf[o++] = 0xBC; dv.setUint32(o, 0x300000, true); o += 4;    // mov esp, 0x300000
    buf[o++] = 0xB9; dv.setUint32(o, ITER, true); o += 4;        // mov ecx, ITER
    buf[o++] = 0x31; buf[o++] = 0xC0;                            // xor eax, eax
    buf[o++] = 0xE9; rel32(o, HEAD_OFF); o += 4;                 // jmp head
    return buf;
}

function run(enabled)
{
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, disable_jit: 0, log_level: 0 });
        let timer;
        let startedAt = 0;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch {}
            const cpu = emulator.v86.cpu;
            resolve({
                status,
                elapsedMs: +(performance.now() - startedAt).toFixed(2),
                eax: cpu.reg32[0] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                enabled: cpu.get_jit_config?.(50) >>> 0,
                interpreted: Number(cpu.wm.exports["profiler_interpreted_steps_get"]?.() ?? -1),
            });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(38, 1);
            cpu.set_jit_config(50, enabled ? 1 : 0);
            cpu.jit_clear_cache?.();
            cpu.wm.exports["profiler_interpreted_steps_reset"]?.();
            cpu.load_multiboot(buildImage().buffer);
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

const sequence = [false, true, true, false, false, true];
const results = [];
for(const enabled of sequence)
{
    const r = await run(enabled);
    results.push(r);
    console.log("jit-page-tail-entries-run " + JSON.stringify(r));
    if(r.status !== "halt" || r.eax !== ITER * INCREMENTS || r.ecx !== 0)
    {
        console.error("FAIL: arithmetic/control-flow mismatch");
        process.exit(1);
    }
}
const median = v => v.slice().sort((a, b) => a - b)[Math.floor(v.length / 2)];
const off = results.filter(x => !x.enabled), on = results.filter(x => x.enabled);
const offInterp = median(off.map(x => x.interpreted)), onInterp = median(on.map(x => x.interpreted));
const offMs = median(off.map(x => x.elapsedMs)), onMs = median(on.map(x => x.elapsedMs));
console.log("jit-page-tail-entries " + JSON.stringify({
    iterations: ITER, offInterpreted: offInterp, onInterpreted: onInterp,
    offMedianMs: offMs, onMedianMs: onMs, throughputGainPct: +((offMs / onMs - 1) * 100).toFixed(2),
}));
if(!(onInterp < offInterp / 10))
{
    console.error("FAIL: page-tail entries still interpreted");
    process.exit(1);
}
console.log("RESULT PASS");
