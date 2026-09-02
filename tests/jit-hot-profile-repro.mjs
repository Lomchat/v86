#!/usr/bin/env node
// Hot-page profile: pages a previous session compiled are compiled at first
// touch in the next one, instead of after JIT_THRESHOLD interpreted
// instructions. For a cold path made of code that runs once per session, that
// interpreted ramp is the dominant cost, and the profile removes it without
// compiling anything the previous session did not already find worth compiling.
//
// The fixture is a stream of pages executed once each, every one long enough
// to cross the (lowered) threshold, so session 1 compiles them all after their
// ramp. Session 2 imports session 1's profile and must (a) compute the same
// EAX, (b) force-compile every page it touches, (c) interpret far fewer
// instructions. A third session imports the profile with one page's bytes
// changed: that page must be rejected by the hash, and the result must still
// be exact. The export must be deterministic, so a re-import re-exports
// byte-identically.

const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000;
const ENTRY_OFF = 0x1000;
const COLD_OFF = 0x10000;
const COLD_PAGES = 200;
// Short enough that one visit stays under the threshold: a page needs five
// visits to compile the ordinary way, and one forced visit with the profile.
const LOOPN = 500;
// Long enough for the asynchronous module instantiation to land: a run of a
// few event-loop turns would start compiles and finish before any completes.
const SWEEPS = 100;
const THRESHOLD = 10_000;
const MEM_SIZE = 64 * 1024 * 1024;
const TIMEOUT_MS = 120_000;

const EXPECTED_EAX = SWEEPS * COLD_PAGES * LOOPN;

// mov ecx,LOOPN / inc eax / dec ecx / jnz / ret
function emitLeaf(buf, dv, off)
{
    let o = off;
    buf[o++] = 0xB9; dv.setUint32(o, LOOPN, true); o += 4;
    const loop = o;
    buf[o++] = 0x40;
    buf[o++] = 0x49;
    buf[o++] = 0x75;
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

    for(let i = 0; i < COLD_PAGES; i++) emitLeaf(buf, dv, COLD_OFF + i * 0x1000);

    let o = ENTRY_OFF;
    buf[o++] = 0xBC; dv.setUint32(o, 0x600000, true); o += 4;  // mov esp
    buf[o++] = 0x31; buf[o++] = 0xC0;                          // xor eax, eax
    buf[o++] = 0xBB; dv.setUint32(o, SWEEPS, true); o += 4;    // mov ebx, SWEEPS
    const outer = o;
    buf[o++] = 0xBA; dv.setUint32(o, BASE + COLD_OFF, true); o += 4; // mov edx, cold
    const inner = o;
    buf[o++] = 0xFF; buf[o++] = 0xD2;                          // call edx
    buf[o++] = 0x81; buf[o++] = 0xC2; dv.setUint32(o, 0x1000, true); o += 4; // add edx, 0x1000
    buf[o++] = 0x81; buf[o++] = 0xFA;
    dv.setUint32(o, BASE + COLD_OFF + COLD_PAGES * 0x1000, true); o += 4;   // cmp edx, end
    buf[o++] = 0x0F; buf[o++] = 0x82; dv.setInt32(o, inner - (o + 4), true); o += 4; // jb inner
    buf[o++] = 0x4B;                                           // dec ebx
    buf[o++] = 0x0F; buf[o++] = 0x85; dv.setInt32(o, outer - (o + 4), true); o += 4; // jnz outer
    buf[o++] = 0xF4; buf[o++] = 0xEB; buf[o++] = 0xFE;         // hlt; jmp $
    return buf;
}

function importProfile(cpu, bytes)
{
    const ex = cpu.wm.exports;
    const ptr = ex["jit_hot_profile_io_alloc"](bytes.length) >>> 0;
    new Uint8Array(cpu.wasm_memory.buffer, ptr, bytes.length).set(bytes);
    return ex["jit_hot_profile_import_commit"](bytes.length) >>> 0;
}

function exportProfile(cpu)
{
    const ex = cpu.wm.exports;
    const len = ex["jit_hot_profile_export_build"]() >>> 0;
    const ptr = ex["jit_hot_profile_io_ptr"]() >>> 0;
    return new Uint8Array(cpu.wasm_memory.buffer, ptr, len).slice();
}

/** Import then export on a fresh instance without running anything. */
function roundTrip(bytes)
{
    return new Promise(resolve => {
        const emulator = new V86({ autostart: false, memory_size: MEM_SIZE, log_level: 0 });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.wm.exports["jit_hot_profile_clear"]();
            const pages = importProfile(cpu, bytes);
            const out = exportProfile(cpu);
            try { emulator.stop(); } catch {}
            resolve({ pages, out });
        });
    });
}

/** Runs the fixture to halt; `profile` (bytes) is installed before it starts,
 *  `patch(image)` may alter the image the guest executes. */
function run(label, { profile = null, patch = null, mode = 0 } = {})
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
                started: ex["jit_get_compile_started"]?.() >>> 0,
                completed: ex["jit_get_compile_completed"]?.() >>> 0,
                interpreted: Number(ex["profiler_interpreted_steps_get"]?.() ?? -1),
                codegenMs: +((ex["jit_get_codegen_total_us"]?.() ?? 0) / 1000).toFixed(2),
                codegenCount: ex["jit_get_codegen_count"]?.() >>> 0,
                codegenBytes: Number(ex["jit_get_codegen_bytes"]?.() ?? 0),
                profilePages: ex["jit_hot_profile_pages"]?.() >>> 0,
                forced: ex["jit_hot_profile_forced"]?.() >>> 0,
                mismatches: ex["jit_hot_profile_mismatches"]?.() >>> 0,
                deferredQueued: ex["jit_get_compile_deferred_queued"]?.() >>> 0,
                mode: cpu.get_jit_config?.(48) >>> 0,
                exported: exportProfile(cpu),
            });
        };
        emulator.bus.register("cpu-event-halt", () => finish("halt"));
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            cpu.reboot_internal();
            cpu.reset_memory();
            cpu.set_jit_config(26, THRESHOLD);
            cpu.set_jit_config(25, 8);
            cpu.set_jit_config(37, 1);
            // Mode 0 forces every known page on first touch (queueing behind
            // the compile cap); mode 1 only while a compile slot is free.
            cpu.set_jit_config(48, mode);
            cpu.jit_clear_cache?.();
            cpu.wm.exports["jit_hot_profile_clear"]();
            cpu.wm.exports["profiler_interpreted_steps_reset"]?.();
            const image = buildImage();
            if(patch) patch(image);
            cpu.load_multiboot(image.buffer.slice(0));
            if(profile)
            {
                const pages = importProfile(cpu, profile);
                console.log(`${label}: imported profile pages=${pages}`);
            }
            timer = setTimeout(() => finish("HANG"), TIMEOUT_MS);
            startedAt = performance.now();
            emulator.run();
        });
    });
}

const show = r => console.log("jit-hot-profile " + JSON.stringify({ ...r, exported: r.exported.length + " bytes" }));
let failed = false;
const fail = msg => { console.error("FAIL: " + msg); failed = true; };

/** Pages listed in a HOTP image, in file order. */
function profilePages(bytes)
{
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if(dv.getUint32(0, true) !== 0x50544F48 || dv.getUint32(4, true) !== 1) throw new Error("not a HOTP v1 image");
    const count = dv.getUint32(8, true);
    const pages = [];
    let i = 12;
    for(let k = 0; k < count; k++)
    {
        pages.push(dv.getUint32(i, true));
        const n = dv.getUint32(i + 8, true);
        i += 12 + ((2 * n + 3) & ~3);
    }
    return pages;
}
const coldPage = i => (BASE + COLD_OFF + i * 0x1000) >>> 12;
const isCold = p => p >= coldPage(0) && p < coldPage(COLD_PAGES);

// Session 1: no profile — every page pays its interpreted ramp.
const s1 = await run("session1");
show(s1);
if(s1.status !== "halt" || s1.eax !== EXPECTED_EAX) fail(`session1 eax=${s1.eax} (${s1.status})`);
if(s1.forced !== 0) fail("session1 forced a compile without a profile");
const cold1 = profilePages(s1.exported).filter(isCold);
console.log(`session1 profile: ${cold1.length} of ${COLD_PAGES} cold pages`);
if(cold1.length < COLD_PAGES * 0.9) fail(`session1 profile covers only ${cold1.length} cold pages`);

// Session 2: profile installed, mode 0 — every known page compiles at first touch.
const s2 = await run("session2", { profile: s1.exported });
show(s2);
if(s2.status !== "halt" || s2.eax !== EXPECTED_EAX) fail(`session2 eax=${s2.eax} (${s2.status})`);
if(s2.profilePages < cold1.length) fail(`session2 profile pages=${s2.profilePages}`);
if(s2.forced < cold1.length) fail(`session2 forced only ${s2.forced} of ${cold1.length} known pages`);
if(s2.mismatches !== 0) fail(`session2 rejected ${s2.mismatches} pages of an identical image`);
if(!(s2.interpreted * 2 < s1.interpreted)) fail(`session2 interpreted ${s2.interpreted} vs ${s1.interpreted}: no ramp skipped`);
if(s2.deferredQueued === 0) fail("session2 (mode 0) never queued a page behind the compile cap: fixture too small to test mode 1");

// Session 2b: same profile, mode 1 — a known page is only forced while a
// compile slot is free, so the burst cannot pile up a deferred queue, and the
// pages it skips take the ordinary ramp and still compile.
const s2b = await run("session2b", { profile: s1.exported, mode: 1 });
show(s2b);
if(s2b.status !== "halt" || s2b.eax !== EXPECTED_EAX) fail(`session2b eax=${s2b.eax} (${s2b.status})`);
if(s2b.mode !== 1) fail("session2b: config 48 not applied");
// The deferred queue still fills with ordinary threshold crossings; what the
// gate changes is how many pages were forced while the window was saturated.
if(s2b.forced === 0 || s2b.forced > s2.forced) fail(`session2b forced=${s2b.forced}, mode 0 forced ${s2.forced}: the slot gate held nothing back`);
if(!(s2b.interpreted * 2 < s1.interpreted)) fail(`session2b interpreted ${s2b.interpreted} vs ${s1.interpreted}: no ramp skipped`);

// Session 3: one known cold page has a byte changed outside its executed code —
// its hash no longer matches, so it must fall back to the ordinary ramp.
const patchedPage = cold1[cold1.length >> 1];
const s3 = await run("session3", {
    profile: s1.exported,
    patch: image => { image[(patchedPage << 12) - BASE + 0x800] ^= 0xFF; },
});
show(s3);
if(s3.status !== "halt" || s3.eax !== EXPECTED_EAX) fail(`session3 eax=${s3.eax} (${s3.status})`);
if(s3.mismatches !== 1) fail(`session3 mismatches=${s3.mismatches}, expected exactly the patched page`);
if(s3.forced < cold1.length - 1) fail(`session3 forced=${s3.forced}, expected at least ${cold1.length - 1}`);

// Determinism: an import followed by an export, with nothing compiled in
// between, must reproduce the image byte for byte.
const rt = await roundTrip(s2.exported);
console.log(`round-trip: pages=${rt.pages} bytes=${rt.out.length}`);
if(Buffer.compare(Buffer.from(s2.exported), Buffer.from(rt.out)) !== 0)
{
    fail(`profile export not deterministic: ${s2.exported.length} vs ${rt.out.length} bytes`);
}

console.log(`SUMMARY interpreted session1=${s1.interpreted} session2=${s2.interpreted} ` +
    `(${((1 - s2.interpreted / s1.interpreted) * 100).toFixed(1)}% fewer) ` +
    `codegen session1=${s1.codegenMs}ms/${s1.codegenCount} session2=${s2.codegenMs}ms/${s2.codegenCount} ` +
    `wall session1=${s1.elapsedMs}ms session2=${s2.elapsedMs}ms`);
console.log(failed ? "RESULT FAIL" : "RESULT PASS");
process.exit(failed ? 1 : 0);
