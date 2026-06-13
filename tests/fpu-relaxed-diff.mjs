#!/usr/bin/env node
// Differential test for the relaxed-FPU inline JIT fast path.
//
// Oracle:  interpreter + relaxed(1)  — helper-based semantics (known-good, shipped for weeks)
// Suspect: JIT         + relaxed(1)  — new inline codegen (gen_fpu_relaxed_*)
// The same multiboot image runs in both; final results land in eax/ebx/ecx/edx
// and must match BIT-EXACT (same f64 arithmetic on both paths).
//
// Variants isolate instruction families so a divergence pinpoints the buggy
// codegen path without rebuild-bisecting:
//   push   - FLD m32/m64/st(i) inline pushes (+helper FILD/FSTP)
//   d8mem  - D8 m32 binops (FADD/FMUL/FSUB/FSUBR/FDIV/FDIVR m32)
//   dcmem  - DC m64 binops
//   reg    - D8 (st0,sti) and DC (sti,st0) register binops
//   pfx    - DE P-forms (binop + inline pop)
//   full   - mixed chain incl. FXCH (helper) interplay
//
//   node tests/fpu-relaxed-diff.mjs
const { V86 } = await import("../build/libv86.mjs");

const BASE = 0x100000, ENTRY_OFF = 0x40;
const DATA = BASE + 0x3000;
const N = 400000;                 // > JIT_THRESHOLD(200k) so the loop compiles naturally
const MEM_SIZE = 16 * 1024 * 1024;
const TIMEOUT_MS = 30000;

// data layout (absolute guest addresses)
const COUNTER = DATA + 0;         // i32, increments per iteration
const C32     = DATA + 8;         // f32 1.5
const C64     = DATA + 16;        // f64 0.75
const C64B    = DATA + 24;        // f64 1.25
const ACC     = DATA + 32;        // f64 result slot
const LAST    = DATA + 40;        // f32 scratch slot
const C80     = DATA + 48;        // f80 1.5 (untagged when FLD'd)

function build_image(bodyName)
{
    const IMG_SIZE = 0x4000;
    const buf = new Uint8Array(IMG_SIZE);
    const dv  = new DataView(buf.buffer);
    const MAGIC = 0x1BADB002, FLAGS = 0x10000;
    dv.setUint32(0x00, MAGIC, true);
    dv.setUint32(0x04, FLAGS, true);
    dv.setUint32(0x08, (-(MAGIC + FLAGS)) >>> 0, true);
    dv.setUint32(0x0c, BASE, true);                 // header_addr
    dv.setUint32(0x10, BASE, true);                 // load_addr
    dv.setUint32(0x14, BASE + IMG_SIZE, true);      // load_end_addr
    dv.setUint32(0x18, BASE + IMG_SIZE, true);      // bss_end_addr
    dv.setUint32(0x1c, BASE + ENTRY_OFF, true);     // entry_addr

    // initialized data
    dv.setUint32(COUNTER - BASE, 1, true);          // counter starts at 1 (avoid div-by-0)
    dv.setFloat32(C32 - BASE, 1.5, true);
    dv.setFloat64(C64 - BASE, 0.75, true);
    dv.setFloat64(C64B - BASE, 1.25, true);
    dv.setFloat64(ACC - BASE, 0.0, true);
    dv.setFloat32(LAST - BASE, 0.0, true);
    // f80 1.5: mantissa 0xC000000000000000, sign_exponent 0x3FFF
    dv.setUint32(C80 - BASE, 0, true);
    dv.setUint32(C80 - BASE + 4, 0xC0000000, true);
    dv.setUint16(C80 - BASE + 8, 0x3FFF, true);

    let o = ENTRY_OFF;
    const labels = {}, patches = [];
    const label = (n) => { labels[n] = o; };
    const emit = (...b) => { for(const x of b) buf[o++] = x & 0xff; };
    const imm32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };
    const rel32 = (n) => { patches.push({ at:o, to:n, end:o+4 }); emit(0,0,0,0); };

    // mem-operand emitters (mod=00 r/m=101 disp32)
    const fild32  = (a) => { emit(0xDB, 0x05); imm32(a); };
    const fld32   = (a) => { emit(0xD9, 0x05); imm32(a); };
    const fld64   = (a) => { emit(0xDD, 0x05); imm32(a); };
    const fldsti  = (i) => emit(0xD9, 0xC0 + i);
    const fstp32  = (a) => { emit(0xD9, 0x1D); imm32(a); };
    const fstp64  = (a) => { emit(0xDD, 0x1D); imm32(a); };
    const d8mem   = (op, a) => { emit(0xD8, op); imm32(a); };   // op: 05 add, 0D mul, 25 sub, 2D subr, 35 div, 3D divr
    const dcmem   = (op, a) => { emit(0xDC, op); imm32(a); };
    const fxch1   = () => emit(0xD9, 0xC9);

    const bodies = {
        // inline pushes: FLD m32, FLD m64, FLD st(i); pops via helper FSTP
        push() {
            fild32(COUNTER);              // st0=n                      (helper push)
            fld32(C32);                   // st0=1.5  st1=n             (inline push m32)
            fld64(C64);                   // st0=.75  st1=1.5 st2=n     (inline push m64)
            fldsti(2);                    // st0=n    ...               (inline push sti)
            fstp64(ACC);                  // acc=n
            fstp32(LAST);                 // .75
            fstp32(LAST);                 // 1.5
            fstp32(LAST);                 // n
        },
        d8mem() {
            fild32(COUNTER);
            d8mem(0x05, C32);             // fadd  st0 = n+1.5
            d8mem(0x0D, C32);             // fmul  *1.5
            d8mem(0x2D, C32);             // fsubr 1.5-st0
            d8mem(0x35, C32);             // fdiv  /1.5
            d8mem(0x3D, C32);             // fdivr 1.5/st0
            d8mem(0x25, C32);             // fsub  st0-1.5
            fstp64(ACC);
        },
        dcmem() {
            fild32(COUNTER);
            dcmem(0x05, C64);             // fadd m64
            dcmem(0x0D, C64B);            // fmul m64
            dcmem(0x2D, C64);             // fsubr m64
            dcmem(0x35, C64B);            // fdiv m64
            dcmem(0x3D, C64);             // fdivr m64
            dcmem(0x25, C64B);            // fsub m64
            fstp64(ACC);
        },
        reg() {
            fild32(COUNTER);              // st0=n
            fld32(C32);                   // st0=1.5 st1=n
            emit(0xD8, 0xC1);             // fadd  st0,st1
            emit(0xD8, 0xC9);             // fmul  st0,st1
            emit(0xD8, 0xE1);             // fsub  st0,st1
            emit(0xD8, 0xE9);             // fsubr st0,st1
            emit(0xD8, 0xF1);             // fdiv  st0,st1
            emit(0xD8, 0xF9);             // fdivr st0,st1
            emit(0xDC, 0xC1);             // fadd  st1,st0
            emit(0xDC, 0xE9);             // fsub  st1,st0
            emit(0xDC, 0xF9);             // fdiv  st1,st0
            fstp32(LAST);
            fstp64(ACC);
        },
        pfx() {
            fild32(COUNTER);              // st0=n
            fld32(C32);  emit(0xDE, 0xC1);   // faddp  st1,st0
            fld32(C32);  emit(0xDE, 0xC9);   // fmulp  st1,st0
            fld32(C32);  emit(0xDE, 0xE9);   // fsubp  st1,st0
            fld32(C32);  emit(0xDE, 0xE1);   // fsubrp st1,st0
            fld32(C32);  emit(0xDE, 0xF9);   // fdivp  st1,st0
            fld32(C32);  emit(0xDE, 0xF1);   // fdivrp st1,st0
            fstp64(ACC);
        },
        // reg+disp8 / SIB addressing — exercises the other gen_modrm_resolve paths
        // (TLB fast/slow if-else nesting inside the new inline branches)
        addr() {
            // ebx = DATA set in prologue below via patch hook (see emitAddrSetup)
            fild32(COUNTER);
            emit(0xD8, 0x43, C32 - DATA);    // fadd dword [ebx+8]
            emit(0xD8, 0x4B, C32 - DATA);    // fmul dword [ebx+8]
            emit(0xDC, 0x63, C64 - DATA);    // fsub qword [ebx+16]
            emit(0xDC, 0x73, C64B - DATA);   // fdiv qword [ebx+24]
            emit(0xD9, 0x44, 0x23, C32 - DATA); // fld dword [ebx+ahem SIB: base=ebx]
            emit(0xDE, 0xC1);                // faddp
            fstp64(ACC);
        },
        // untagged F80 interplay: FLD m80 pushes a GENUINE 80-bit value (no RELAXED_TAG)
        // → inline binops must take their SLOW (helper) branch inside compiled code
        m80() {
            fild32(COUNTER);                 // tagged
            emit(0xDB, 0x2D); imm32(C80);    // fld tbyte [c80] — untagged 1.5
            emit(0xD8, 0xC1);                // fadd st0,st1   (one side untagged → slow)
            d8mem(0x0D, C32);                // fmul m32 on untagged-result chain
            emit(0xDE, 0xC1);                // faddp st1,st0  (slow branch + inline pop)
            dcmem(0x2D, C64);                // fsubr m64
            fstp64(ACC);
        },
        // minimal slow-branch repros
        m80_sti() {                          // sti binop slow branch only
            emit(0xDB, 0x2D); imm32(C80);    // fld tbyte — untagged
            emit(0xD8, 0xC0);                // fadd st0,st0 (slow: untagged)
            fstp64(ACC);
        },
        m80_mem() {                          // m32 binop slow branch only
            emit(0xDB, 0x2D); imm32(C80);    // fld tbyte — untagged
            d8mem(0x05, C32);                // fadd m32 (slow: untagged st0)
            fstp64(ACC);
        },
        m80_pop() {                          // slow binop + inline pop
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDE, 0xC1);                // faddp st1,st0 (slow + inline pop)
            fstp64(ACC);
        },
        m80_nopop() {                        // slow binop r=1, helper pops only
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDC, 0xC1);                // fadd st1,st0 (slow, no pop)
            fstp64(ACC);                     // helper pop
            fstp64(ACC);                     // helper pop (overwrites — fine)
        },
        tag_pop() {                          // tagged values through the SAME DE faddp
            fld32(C32);
            fld32(C32);
            emit(0xDE, 0xC1);                // faddp (fast + inline pop)
            fstp64(ACC);
        },
        // diagnostic: m80_pop body, but bail out at the FIRST NaN in ACC and
        // dump FPU env. Exit regs: eax=iteration, ebx=status word(TOP in 11-13),
        // ecx=tag word, edx=counter.
        diag() {
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDB, 0x2D); imm32(C80);
            emit(0xDE, 0xC1);                // faddp st1,st0 (slow + inline pop)
            fstp64(ACC);
            emit(0xA1); imm32(ACC + 4);      // mov eax, [acc hi]
            emit(0x3D); imm32(0x7ff80000);   // cmp eax, NaN-hi
            emit(0x0F, 0x84); rel32("nan_hit"); // je nan_hit
        },
        full() {
            fild32(COUNTER);              // st0=n
            fld32(C32);                   // push 1.5
            emit(0xD8, 0xC1);             // fadd st0,st1
            fld64(C64);                   // push .75
            emit(0xDE, 0xC9);             // fmulp st1,st0
            d8mem(0x2D, C32);             // fsubr m32
            dcmem(0x35, C64B);            // fdiv m64
            fxch1();                      // helper interplay
            emit(0xD8, 0xE1);             // fsub st0,st1
            dcmem(0x05, ACC);             // fadd qword [acc]  — accumulate across iterations
            fstp64(ACC);
            fstp32(LAST);
        },
    };

    emit(0xBC); imm32(0x200000);          // mov esp, 0x200000
    emit(0xBB); imm32(DATA);              // mov ebx, DATA (for reg-based addressing)
    emit(0xDB, 0xE3);                     // fninit
    emit(0x31, 0xF6);                     // xor esi, esi
    label("outer");
    emit(0x81, 0xFE); imm32(N);           // cmp esi, N
    emit(0x0F, 0x8D); rel32("done");      // jge done
    bodies[bodyName]();
    emit(0xFF, 0x05); imm32(COUNTER);     // inc dword [counter]
    emit(0x46);                           // inc esi
    emit(0xE9); rel32("outer");           // jmp outer
    label("done");
    emit(0xA1); imm32(ACC);               // mov eax, [acc lo]
    emit(0x8B, 0x1D); imm32(ACC + 4);     // mov ebx, [acc hi]
    emit(0x8B, 0x0D); imm32(LAST);        // mov ecx, [last]
    emit(0x8B, 0x15); imm32(COUNTER);     // mov edx, [counter]
    emit(0xF4);                           // hlt
    label("nan_hit");
    emit(0xD9, 0x35); imm32(DATA + 0x200); // fnstenv [env]
    emit(0x89, 0xF0);                      // mov eax, esi (iteration)
    emit(0x8B, 0x1D); imm32(DATA + 0x204); // mov ebx, [status word]
    emit(0x8B, 0x0D); imm32(DATA + 0x208); // mov ecx, [tag word]
    emit(0x8B, 0x15); imm32(COUNTER);      // mov edx, [counter]
    emit(0xF4);                            // hlt
    emit(0xEB, 0xFE);                      // jmp $

    for(const p of patches)
        dv.setInt32(p.at, labels[p.to] - p.end, true);
    return buf;
}

function run(bodyName, { jit, relaxed })
{
    return new Promise((resolve) => {
        const img = build_image(bodyName);
        const emulator = new V86({ autostart:false, memory_size:MEM_SIZE,
                                   disable_jit: jit ? 0 : 1, log_level:0 });
        let halted = false, timer;
        const finish = (status) => {
            clearTimeout(timer);
            try { emulator.stop(); } catch(e) {}
            const cpu = emulator.v86.cpu;
            resolve({ status,
                      eax: cpu.reg32[0] >>> 0, ecx: cpu.reg32[1] >>> 0,
                      edx: cpu.reg32[2] >>> 0, ebx: cpu.reg32[3] >>> 0,
                      eip: cpu.instruction_pointer[0] >>> 0 });
        };
        emulator.bus.register("cpu-event-halt", () => { halted = true; finish("halt"); });
        emulator.add_listener("emulator-loaded", () => {
            const cpu = emulator.v86.cpu;
            const setRelaxed = cpu.wm?.exports?.set_relaxed_fpu;
            if(!setRelaxed) { console.error("FATAL: set_relaxed_fpu export not found"); process.exit(2); }
            setRelaxed(relaxed ? 1 : 0);
            cpu.reboot_internal(); cpu.reset_memory();
            cpu.load_multiboot(img.buffer);
            setRelaxed(relaxed ? 1 : 0);   // re-assert in case reset touched it
            timer = setTimeout(() => { if(!halted) finish("HANG"); }, TIMEOUT_MS);
            emulator.run();
        });
    });
}

const fmt = (r) => `${r.status} acc=${r.ebx.toString(16).padStart(8,"0")}:${r.eax.toString(16).padStart(8,"0")} last=${r.ecx.toString(16).padStart(8,"0")} n=${r.edx}`;

const variants = ["push", "d8mem", "dcmem", "reg", "pfx", "addr", "m80", "m80_sti", "m80_mem", "m80_pop", "m80_nopop", "tag_pop", "full"];
let anyDiverged = false;
for(const v of variants) {
    const oracle  = await run(v, { jit:false, relaxed:true });
    const suspect = await run(v, { jit:true,  relaxed:true });
    const same = oracle.status === "halt" && suspect.status === "halt"
              && oracle.eax === suspect.eax && oracle.ebx === suspect.ebx
              && oracle.ecx === suspect.ecx && oracle.edx === suspect.edx;
    if(!same) anyDiverged = true;
    console.log(`${v.padEnd(6)} interp: ${fmt(oracle)}`);
    console.log(`${"".padEnd(6)} jit   : ${fmt(suspect)}  ${same ? "OK" : "<<< DIVERGED"}`);
}
console.log(anyDiverged ? "\nVERDICT: inline fast path DIVERGES from helpers" : "\nVERDICT: all variants match");
process.exit(anyDiverged ? 1 : 0);
