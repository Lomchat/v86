#!/usr/bin/env node
// Generic proof for guarded continuation after a synchronous JIT block
// boundary. The hot loop calls a canonical OUT 0xB077 + RET stub. Function id
// 1 is routed to the pure WASM GetLastError handler, so a successful candidate
// avoids the historical module exit/re-entry between OUT and RET without any
// game-specific address or API implementation.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x20;
const STUB_OFF = 0x100;
const ITER = 3_000_000;
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 20_000;
const REDIRECT_MAGIC = 0x5A17B00B;

function buildImage()
{
    const buf = new Uint8Array(0x200);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const emit = (...bytes) => { for(const b of bytes) buf[o++] = b & 0xff; };
    emit(0xBC, 0x00, 0x00, 0x20, 0x00);                 // mov esp,0x200000
    emit(0xB9); dv.setUint32(o, ITER, true); o += 4;     // mov ecx,ITER
    const loop = o;
    emit(0xE8);                                         // call stub
    dv.setInt32(o, (BASE + STUB_OFF) - (BASE + o + 4), true); o += 4;
    emit(0x49);                                         // dec ecx
    emit(0x0F, 0x85);                                   // jnz loop
    dv.setInt32(o, (BASE + loop) - (BASE + o + 4), true); o += 4;
    emit(0xF4, 0xEB, 0xFE);                             // hlt; jmp $

    o = STUB_OFF;
    emit(0xB8, 0x01, 0x00, 0x00, 0x00);                 // mov eax,1 (function id)
    emit(0xBA, 0x77, 0xB0, 0x00, 0x00);                 // mov edx,0xB077
    emit(0xEF);                                         // out dx,eax
    emit(0xC3);                                         // ret
    return buf;
}

function buildRedirectImage()
{
    const buf = new Uint8Array(0x240);
    const dv = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);
    dv.setUint32(0x10, BASE, true);
    dv.setUint32(0x14, BASE + buf.length, true);
    dv.setUint32(0x18, BASE + 0x4000, true);
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);

    let o = ENTRY_OFF;
    const emit = (...bytes) => { for(const b of bytes) buf[o++] = b & 0xff; };
    emit(0xBC, 0x00, 0x00, 0x20, 0x00);                 // mov esp,0x200000
    emit(0x31, 0xDB);                                   // xor ebx,ebx
    const loop = o;
    emit(0xE8);
    dv.setInt32(o, (BASE + STUB_OFF) - (BASE + o + 4), true); o += 4;
    emit(0x81, 0xFB); dv.setUint32(o, 1_000_000, true); o += 4; // cmp ebx,1m
    emit(0x0F, 0x82);                                   // jb loop
    dv.setInt32(o, (BASE + loop) - (BASE + o + 4), true); o += 4;
    emit(0xBE, 0xAD, 0xDE, 0xAD, 0xDE);                 // mov esi,0xDEADDEAD (bad)
    emit(0xF4, 0xEB, 0xFE);

    const redirectOff = 0x80;
    o = redirectOff;
    emit(0xBE); dv.setUint32(o, REDIRECT_MAGIC, true); o += 4;
    emit(0xF4, 0xEB, 0xFE);

    o = STUB_OFF;
    emit(0x43);                                         // inc ebx
    emit(0x89, 0xD8);                                   // mov eax,ebx
    emit(0xBA, 0x80, 0x00, 0x00, 0x00);                 // mov edx,0x80
    emit(0xEF);                                         // out dx,eax
    emit(0xC3);                                         // ret
    return { buf, redirectEip: BASE + redirectOff };
}

function run(enabled)
{
    return new Promise(resolve => {
        const emulator = new V86({
            autostart: false,
            memory_size: MEM_SIZE,
            disable_jit: 0,
            log_level: 0,
        });
        let timer, startedAt = 0;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch {}
            const cpu = emulator.v86.cpu;
            resolve({
                status,
                enabled: cpu.get_jit_config?.(36) >>> 0,
                elapsedMs: performance.now() - startedAt,
                ecx: cpu.reg32[1] >>> 0,
                esp: cpu.reg32[4] >>> 0,
                sites: cpu.wm.exports.jit_sync_boundary_continuation_sites_compiled?.() ?? -1,
            });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(36, enabled ? 1 : 0);
            cpu.jit_clear_cache?.();

            // Minimal authoritative hypercall page: nonzero live scheduler
            // budget, master enable, function id 1 -> handler 5 (GetLastError).
            const ex = cpu.wm.exports;
            const hp = ex.get_hypercall_page_ptr() >>> 0;
            const view = new DataView(ex.memory.buffer);
            view.setUint32(hp + 0x000, 100_003, true);
            view.setUint32(hp + 0x008, 1, true);
            view.setUint8(hp + 0x100 + 1, 5);

            cpu.load_multiboot(buildImage().buffer);
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

function runRedirectGuard(enabled)
{
    return new Promise(resolve => {
        const emulator = new V86({ autostart:false, memory_size:MEM_SIZE, disable_jit:0, log_level:0 });
        const image = buildRedirectImage();
        let timer;
        const finish = status => {
            clearTimeout(timer);
            try { emulator.stop(); } catch {}
            const cpu = emulator.v86.cpu;
            resolve({
                status,
                enabled: cpu.get_jit_config?.(36) >>> 0,
                ebx: cpu.reg32[3] >>> 0,
                esi: cpu.reg32[6] >>> 0,
                eip: cpu.instruction_pointer[0] >>> 0,
            });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(36, enabled ? 1 : 0);
            cpu.jit_clear_cache?.();
            const ex = cpu.wm.exports;
            const hp = ex.get_hypercall_page_ptr() >>> 0;
            new DataView(ex.memory.buffer).setUint32(hp + 0x000, 100_003, true);

            // At a hot, already-JIT-compiled boundary, redirect EIP exactly as
            // an async park/interrupt handler may do. The candidate must notice
            // and MUST NOT execute the following RET.
            cpu.io.register_write(0x80, null, undefined, undefined, value => {
                if((value >>> 0) === 600_000) cpu.instruction_pointer[0] = image.redirectEip;
            });
            cpu.load_multiboot(image.buf.buffer);
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
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
    console.log("jit-sync-boundary-continuation-run " + JSON.stringify(result));
    if(result.status !== "halt" || result.ecx !== 0 || result.esp !== 0x200000)
    {
        console.error("FAIL: arithmetic/stack/control-flow mismatch", result);
        process.exit(1);
    }
    if(enabled && result.sites <= 0)
    {
        console.error("FAIL: continuation enabled but no site was compiled", result);
        process.exit(1);
    }
}

const off = results.filter(x => !x.enabled).map(x => x.elapsedMs).sort((a,b) => a-b)[1];
const on = results.filter(x => x.enabled).map(x => x.elapsedMs).sort((a,b) => a-b)[1];
console.log("jit-sync-boundary-continuation-summary " + JSON.stringify({
    offMedianMs: off,
    onMedianMs: on,
    throughputGainPct: (off / on - 1) * 100,
}));

for(const enabled of [false, true])
{
    const redirected = await runRedirectGuard(enabled);
    console.log("jit-sync-boundary-continuation-redirect " + JSON.stringify(redirected));
    if(redirected.status !== "halt" || redirected.esi !== REDIRECT_MAGIC || redirected.ebx !== 600_000)
    {
        console.error("FAIL: runtime EIP redirect was not preserved", redirected);
        process.exit(1);
    }
}
process.exit(0);
