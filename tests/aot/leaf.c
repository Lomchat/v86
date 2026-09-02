// Hand-written stand-in for a translated guest function, in the ABI of a v86
// JIT module: `entry(initial_state)` runs with guest registers at the fixed
// offsets of cpu.rs's global pointers, in the same linear memory as v86.
//
// The guest function it replaces (see aot-external-module-repro.mjs):
//     mov ecx, [esp+4]        ; count
//     xor eax, eax
//   loop:
//     add eax, ecx
//     dec ecx
//     jnz loop
//     ret
// i.e. eax = n(n+1)/2, ecx = 0, and the return pops the return address.
#include <stdint.h>

#define REG32 ((volatile int32_t *)64)
#define EAX 0
#define ECX 1
#define ESP 4
#define INSTRUCTION_POINTER ((volatile int32_t *)556)
#define PREVIOUS_IP ((volatile int32_t *)560)
#define INSTRUCTION_COUNTER ((volatile uint32_t *)664)

// Guest RAM is a region inside v86's linear memory; its base is a runtime
// constant the host provides. Registers and CPU state are NOT in guest RAM —
// they sit at the fixed low offsets above.
__attribute__((import_module("env"), import_name("mem_base"))) uint32_t mem_base(void);

static inline uint32_t ld32(uint32_t base, uint32_t addr) { return *(volatile uint32_t *)(uintptr_t)(base + addr); }

__attribute__((export_name("entry")))
void entry(int32_t initial_state)
{
    (void)initial_state;
    const uint32_t base = mem_base();
    uint32_t esp = (uint32_t)REG32[ESP];
    uint32_t count = ld32(base, esp + 4);
    uint32_t eax = 0;
    uint32_t ecx = count;
    uint32_t insns = 2;
    do {
        eax += ecx;
        ecx -= 1;
        insns += 3;
    } while (ecx != 0);
    uint32_t ret = ld32(base, esp);
    REG32[EAX] = (int32_t)eax;
    REG32[ECX] = (int32_t)ecx;
    REG32[ESP] = (int32_t)(esp + 4);
    *PREVIOUS_IP = (int32_t)ret;
    *INSTRUCTION_POINTER = (int32_t)ret;
    *INSTRUCTION_COUNTER += insns + 1;
}
