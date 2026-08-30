#!/usr/bin/env node
// Correctness, SMC-invalidation and throughput probe for JIT config 36.
//
// MAX_PAGES=1 first gives the cross-page leaf its own module. After the caller
// becomes Tier-2, its wider budget would normally still refuse that already
// covered target page. Config 36 may then copy only the registered tiny C3 leaf
// into the caller and must invalidate the copy when guest code rewrites it.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY = 0x20;
const LEAF = 0x1000;
const STACK = 0x300000;
const ITERATIONS = 2_000_000;
const MAGIC = 0x36EAF00D;
const TIMEOUT_MS = 30_000;

function image(mutate)
{
    const bytes = new Uint8Array(LEAF + 16);
    const dv = new DataView(bytes.buffer);
    const MAGIC_HEADER = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC_HEADER, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC_HEADER + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + bytes.length, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY, true);

    let o = ENTRY;
    const emit = (...xs) => { for(const x of xs) bytes[o++] = x & 0xff; };
    const u32 = x => { dv.setUint32(o, x >>> 0, true); o += 4; };
    const calls = count => {
        emit(0xB9); u32(count);                // mov ecx, count
        const loop = o;
        emit(0xE8);                            // call leaf
        dv.setInt32(o, LEAF - (o + 4), true); o += 4;
        emit(0x49);                            // dec ecx
        emit(0x75, (loop - (o + 2)) & 0xff);  // jnz loop
    };

    emit(0xBC); u32(STACK);                    // mov esp, stack
    emit(0x31, 0xC0);                          // xor eax, eax
    calls(ITERATIONS);
    if(mutate)
    {
        emit(0xC6, 0x05); u32(BASE + LEAF + 2); emit(0x05); // add eax,3 -> add eax,5
        calls(ITERATIONS);
    }
    emit(0xBB); u32(MAGIC);                    // mov ebx, magic
    emit(0xF4, 0xEB, 0xFE);                    // hlt; jmp $

    o = LEAF;
    emit(0x83, 0xC0, 0x03);                    // add eax, 3
    emit(0xC3);                                // ret
    return bytes;
}

function run(enabled, mutate = false)
{
    return new Promise(resolve => {
        const emulator = new V86({
            autostart: false,
            memory_size: 16 * 1024 * 1024,
            disable_jit: 0,
            log_level: 0,
        });
        let timer, startedAt = 0, done = false;
        const finish = status => {
            if(done) return;
            done = true;
            clearTimeout(timer);
            const cpu = emulator.v86.cpu, w = cpu.wm.exports;
            const result = {
                enabled: cpu.get_jit_config?.(36) >>> 0,
                mutate,
                status,
                elapsedMs: +(performance.now() - startedAt).toFixed(2),
                eax: cpu.reg32[0] >>> 0,
                ecx: cpu.reg32[1] >>> 0,
                ebx: cpu.reg32[3] >>> 0,
                tier2Promotions: w.jit_get_tier2_promotions?.() ?? -1,
                externalSites: w.jit_external_leaf_call_fusion_sites_compiled?.() ?? -1,
            };
            try { emulator.stop(); } catch {}
            resolve(result);
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(1, 1);       // Tier-1: one page per module
            cpu.set_jit_config(17, 1);      // copied dependency has a separate strict budget
            cpu.set_jit_config(20, 8);
            cpu.set_jit_config(23, 0);      // isolate this mechanism from regions
            cpu.set_jit_config(24, 0);
            cpu.set_jit_config(15, 20_000);
            cpu.set_jit_config(26, 10_000);
            cpu.set_jit_config(27, 1);
            cpu.set_jit_config(28, 1);
            cpu.set_jit_config(36, enabled ? 1 : 0);
            cpu.jit_clear_cache?.();
            cpu.load_multiboot(image(mutate).buffer);
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

function check(result)
{
    const expected = Math.imul(ITERATIONS, result.mutate ? 8 : 3) >>> 0;
    if(result.status !== "halt" || result.eax !== expected || result.ecx !== 0 || result.ebx !== MAGIC)
        throw new Error(`incorrect execution: ${JSON.stringify(result)}, expected eax=${expected}`);
    if(result.enabled && (result.tier2Promotions < 1 || result.externalSites < 1))
        throw new Error(`external fusion did not compile: ${JSON.stringify(result)}`);
}

const smcOff = await run(false, true);
const smcOn = await run(true, true);
check(smcOff); check(smcOn);
console.log("jit-external-leaf-call-fusion-smc " + JSON.stringify({ smcOff, smcOn }));

const order = [false, true, true, false, false, true];
const samples = [];
for(const enabled of order)
{
    const result = await run(enabled, false);
    check(result);
    samples.push(result);
}
const median = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const offMs = median(samples.filter(x => !x.enabled).map(x => x.elapsedMs));
const onMs = median(samples.filter(x => x.enabled).map(x => x.elapsedMs));
console.log("jit-external-leaf-call-fusion-samples " + JSON.stringify(samples));
console.log("jit-external-leaf-call-fusion-summary " + JSON.stringify({
    offMs,
    onMs,
    speedupPercent: +((offMs / onMs - 1) * 100).toFixed(2),
}));

process.exit(0);
