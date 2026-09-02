#!/usr/bin/env node
// Plumbing proof for ahead-of-time translated code: a WebAssembly function
// compiled outside the JIT (tests/aot/leaf.c, built by clang) is placed in
// v86's function table at a reserved index and registered as the module for
// one guest entry point. The dispatcher must then enter it in place of the
// guest's own bytes, and the run must end with the same registers and the
// same number of retired instructions as the plain run — the odometer parity
// is what shows the module accounted for its work like compiled code does.
//
// Requires build/v86.wasm with jit_register_external_module and
// tests/aot/leaf.wasm (see tests/aot/leaf.c for the build line).

import { readFileSync } from "node:fs";

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x1000;
const FUNC_OFF = 0x3000;
const ITER = Number(process.env.AOT_ITER ?? 2000);
const MEM_SIZE = 64 * 1024 * 1024;
const TIMEOUT_MS = 120_000;
const WASM_TABLE_OFFSET = 1024;

// eax = n(n+1)/2 per call; ebx sums the results over n = ITER..1.
const EXPECTED_EBX = (() => { let s = 0; for(let n = ITER; n >= 1; n--) s += n * (n + 1) / 2; return s >>> 0; })();

function buildImage()
{
    const len = 0x5000;
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

    // FUNC: mov ecx,[esp+4] / xor eax,eax / L: add eax,ecx / dec ecx / jnz L / ret
    let o = FUNC_OFF;
    buf[o++] = 0x8B; buf[o++] = 0x4C; buf[o++] = 0x24; buf[o++] = 0x04;
    buf[o++] = 0x31; buf[o++] = 0xC0;
    const L = o;
    buf[o++] = 0x01; buf[o++] = 0xC8;
    buf[o++] = 0x49;
    buf[o++] = 0x75;
    // Compute the displacement before the store: `buf[o++] = f(o)` evaluates the
    // index first, so f would see the already-incremented o.
    const rel = L - (o + 1);
    buf[o++] = rel & 0xFF;
    buf[o++] = 0xC3;

    // ENTRY: mov esp / xor ebx,ebx / mov esi,ITER / outer: push esi / call FUNC /
    // add esp,4 / add ebx,eax / dec esi / jnz outer / hlt / jmp $
    o = ENTRY_OFF;
    buf[o++] = 0xBC; dv.setUint32(o, 0x600000, true); o += 4;
    buf[o++] = 0x31; buf[o++] = 0xDB;
    buf[o++] = 0xBE; dv.setUint32(o, ITER, true); o += 4;
    const outer = o;
    buf[o++] = 0x56;
    buf[o++] = 0xE8; dv.setInt32(o, (BASE + FUNC_OFF) - (BASE + o + 4), true); o += 4;
    buf[o++] = 0x83; buf[o++] = 0xC4; buf[o++] = 0x04;
    buf[o++] = 0x01; buf[o++] = 0xC3;
    buf[o++] = 0x4E;
    buf[o++] = 0x0F; buf[o++] = 0x85; dv.setInt32(o, outer - (o + 4), true); o += 4;
    buf[o++] = 0xF4; buf[o++] = 0xEB; buf[o++] = 0xFE;
    return buf;
}

const leafWasm = readFileSync(new URL("./aot/leaf.wasm", import.meta.url));

function run(label, { external = false } = {})
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
                label, status,
                elapsedMs: +(performance.now() - startedAt).toFixed(2),
                eax: cpu.reg32[0] >>> 0,
                ebx: cpu.reg32[3] >>> 0,
                esp: cpu.reg32[4] >>> 0,
                retired: cpu.instruction_counter[0] >>> 0,
                interpreted: Number(ex["profiler_interpreted_steps_get"]?.() ?? -1),
                started: ex["jit_get_compile_started"]?.() >>> 0,
            });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", async () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(26, 10_000);
            cpu.jit_clear_cache?.();
            cpu.wm.exports["profiler_interpreted_steps_reset"]?.();
            cpu.load_multiboot(buildImage().buffer.slice(0));
            if(external)
            {
                const ex = cpu.wm.exports;
                // Guest RAM lives at an offset inside v86's linear memory; the module asks
                // for it once per entry through the imported mem_base().
                const memBase = cpu.mem8.byteOffset >>> 0;
                const { instance } = await WebAssembly.instantiate(leafWasm, { env: { memory: cpu.wasm_memory, mem_base: () => memBase } });
                const index = ex["jit_external_module_first_index"]() >>> 0;
                cpu.wm.wasm_table.set(index + WASM_TABLE_OFFSET, instance.exports.entry);
                const flags = ex["jit_get_current_state_flags"]() >>> 0;
                const ok = ex["jit_register_external_module"](index, BASE + FUNC_OFF, flags, 0) >>> 0;
                console.log(`${label}: external module at table index ${index}, state_flags=0x${flags.toString(16)}, registered=${ok}`);
                if(ok !== 1) { finish("REGISTER-FAILED"); return; }
            }
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

let failed = false;
const fail = msg => { console.error("FAIL: " + msg); failed = true; };

const plain = await run("plain");
console.log("aot-external " + JSON.stringify(plain));
if(plain.status !== "halt" || plain.ebx !== EXPECTED_EBX) fail(`plain ebx=${plain.ebx} expected ${EXPECTED_EBX} (${plain.status})`);

const ext = await run("external", { external: true });
console.log("aot-external " + JSON.stringify(ext));
if(ext.status !== "halt" || ext.ebx !== EXPECTED_EBX) fail(`external ebx=${ext.ebx} expected ${EXPECTED_EBX} (${ext.status})`);
if(ext.eax !== plain.eax || ext.esp !== plain.esp) fail(`external eax/esp ${ext.eax}/${ext.esp} vs plain ${plain.eax}/${plain.esp}`);
if(ext.retired !== plain.retired) fail(`retired instructions differ: external ${ext.retired} vs plain ${plain.retired}`);
// The leaf's page is never interpreted or compiled once the module owns it.
if(!(ext.interpreted < plain.interpreted)) fail(`external interpreted ${ext.interpreted} >= plain ${plain.interpreted}: module not entered`);
if(!(ext.started < plain.started)) fail(`external compiled ${ext.started} modules, plain ${plain.started}: the leaf page was still JIT-compiled`);

console.log(`SUMMARY plain=${plain.elapsedMs}ms external=${ext.elapsedMs}ms retired=${plain.retired}/${ext.retired} interpreted=${plain.interpreted}/${ext.interpreted} compiled=${plain.started}/${ext.started}`);
console.log(failed ? "RESULT FAIL" : "RESULT PASS");
process.exit(failed ? 1 : 0);
