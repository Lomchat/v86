#!/usr/bin/env node
// Exercises a hot basic block that enters at page+0xfe0. The historical JIT
// stops once execution reaches page+0xff0 even though four INCs and the final
// JMP still fit wholly in the page. Exact-tail mode (config 34) must compile
// those complete instructions while preserving the interpreter for a genuine
// cross-page instruction.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const TAIL_OFF = 0x0fe0;
const CONT_OFF = 0x1020;
const ITER = 4_000_000;
const INCREMENTS = 20;
const CROSS_ITER = 500_000;
const CROSS_INCREMENTS = 31;
const CROSS_VALUE = 0x76543210;
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
    dv.setUint32(0x1c, BASE + CONT_OFF, true);

    const rel32 = (at, target) => dv.setInt32(at, target - (at + 4), true);

    let o = TAIL_OFF;
    for(let i = 0; i < INCREMENTS; i++) buf[o++] = 0x40; // inc eax
    buf[o++] = 0xE9;                                     // jmp continuation
    rel32(o, CONT_OFF); o += 4;

    o = CONT_OFF;
    buf[o++] = 0xBC; dv.setUint32(o, 0x300000, true); o += 4; // mov esp, 0x300000
    buf[o++] = 0xB9; dv.setUint32(o, ITER, true); o += 4;     // mov ecx, ITER
    buf[o++] = 0x31; buf[o++] = 0xC0;                        // xor eax, eax
    const loop = o;
    buf[o++] = 0xE9; rel32(o, TAIL_OFF); o += 4;              // jmp tail
    const afterTail = o;
    buf[o++] = 0x49;                                         // dec ecx
    buf[o++] = 0x0F; buf[o++] = 0x85; rel32(o, loop); o += 4; // jnz loop
    buf[o++] = 0xF4; buf[o++] = 0xEB; buf[o++] = 0xFE;       // hlt; jmp $

    // Tail jumps to the instruction following the main loop's jump-to-tail.
    dv.setInt32(TAIL_OFF + INCREMENTS + 1,
        afterTail - (TAIL_OFF + INCREMENTS + 5), true);
    return buf;
}

function buildCrossingImage()
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
    dv.setUint32(0x1c, BASE + 0x1040, true);

    const rel32 = (at, target) => dv.setInt32(at, target - (at + 4), true);
    let o = TAIL_OFF;
    for(let i = 0; i < CROSS_INCREMENTS; i++) buf[o++] = 0x40; // through 0xffe
    buf[o++] = 0xBA;                                           // mov edx, imm32 at 0xfff
    dv.setUint32(o, CROSS_VALUE, true); o += 4;                // immediate crosses the page
    const afterCrossing = o;
    buf[o++] = 0x49;                                           // dec ecx
    buf[o++] = 0x0F; buf[o++] = 0x85; rel32(o, TAIL_OFF); o += 4;
    buf[o++] = 0xF4; buf[o++] = 0xEB; buf[o++] = 0xFE;

    o = 0x1040;
    buf[o++] = 0xBC; dv.setUint32(o, 0x300000, true); o += 4;
    buf[o++] = 0xB9; dv.setUint32(o, CROSS_ITER, true); o += 4;
    buf[o++] = 0x31; buf[o++] = 0xC0;                          // xor eax, eax
    buf[o++] = 0x31; buf[o++] = 0xD2;                          // xor edx, edx
    buf[o++] = 0xE9; rel32(o, TAIL_OFF); o += 4;

    if(afterCrossing !== 0x1004) throw new Error("cross-page fixture layout drifted");
    return buf;
}

function run(exactTail, image = buildImage, crossPage = false)
{
    return new Promise(resolve => {
        const emulator = new V86({
            autostart: false,
            memory_size: MEM_SIZE,
            disable_jit: 0,
            log_level: 0,
        });
        let timer;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch {}
            const cpu = emulator.v86.cpu;
            resolve({
                status,
                elapsedMs: performance.now() - startedAt,
                eax: cpu.reg32[0] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                edx: cpu.reg32[2] >>> 0,
                exactTail: cpu.get_jit_config?.(34) >>> 0,
                crossPage: cpu.get_jit_config?.(38) >>> 0,
                compiledTailInstructions:
                    cpu.wm.exports["jit_exact_page_tail_instructions_compiled"]?.() ?? -1,
                compiledCrossPageInstructions:
                    cpu.wm.exports["jit_contiguous_cross_page_instructions_compiled"]?.() ?? -1,
            });
        };
        let startedAt = 0;
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, crossPage ? 2 : 1);
            cpu.set_jit_config(34, exactTail ? 1 : 0);
            cpu.set_jit_config(38, crossPage ? 1 : 0);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(image().buffer);
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
    const result = await run(enabled);
    results.push(result);
    console.log("jit-exact-page-tail-run " + JSON.stringify(result));
    if(result.status !== "halt" || result.eax !== ITER * INCREMENTS || result.ecx !== 0)
    {
        console.error("FAIL: arithmetic/control-flow mismatch");
        process.exit(1);
    }
    if(enabled && result.compiledTailInstructions <= 0)
    {
        console.error("FAIL: exact-tail mode compiled no page-tail instruction");
        process.exit(1);
    }
    if(!enabled && result.compiledTailInstructions !== 0)
    {
        console.error("FAIL: disabled exact-tail mode changed compilation");
        process.exit(1);
    }
}

const median = values => {
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
};
const offMs = median(results.filter(x => !x.exactTail).map(x => x.elapsedMs));
const onMs = median(results.filter(x => x.exactTail).map(x => x.elapsedMs));
console.log("jit-exact-page-tail " + JSON.stringify({
    iterations: ITER,
    offMedianMs: +offMs.toFixed(2),
    onMedianMs: +onMs.toFixed(2),
    throughputGainPct: +((offMs / onMs - 1) * 100).toFixed(2),
}));

for(const [exactTail, crossPage] of [[false, false], [true, false], [false, true], [true, true]])
{
    const result = await run(exactTail, buildCrossingImage, crossPage);
    console.log("jit-exact-page-tail-crossing " + JSON.stringify(result));
    if(result.status !== "halt"
        || result.eax !== CROSS_ITER * CROSS_INCREMENTS
        || result.ecx !== 0
        || result.edx !== CROSS_VALUE)
    {
        console.error("FAIL: genuine cross-page instruction semantics changed");
        process.exit(1);
    }
    if(exactTail && result.compiledTailInstructions <= 0)
    {
        console.error("FAIL: exact-tail mode did not compile the safe prefix");
        process.exit(1);
    }
    if(!exactTail && result.compiledTailInstructions !== 0)
    {
        console.error("FAIL: disabled mode compiled the page tail");
        process.exit(1);
    }
    if(crossPage && result.compiledCrossPageInstructions <= 0)
    {
        console.error("FAIL: contiguous cross-page mode compiled no crossing instruction");
        process.exit(1);
    }
    if(!crossPage && result.compiledCrossPageInstructions !== 0)
    {
        console.error("FAIL: disabled cross-page mode compiled a crossing instruction");
        process.exit(1);
    }
}

process.exit(0);
