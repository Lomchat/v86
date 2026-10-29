var global = {};
var process = { hrtime: function() {} };

/**
 * @param {string} name
 * @param {function()} processor
 */
var registerProcessor = function(name, processor) {};

const sampleRate = 0;

// JIT switches read from globalThis by cpu.js (must not be renamed)
globalThis.ENABLE_JIT_BLOCK_CHAINING;
globalThis.DISABLE_JIT_DEAD_FLAG_ELISION;

var WabtModule = {
    readWasm: function(buf, opt) {},
    generateNames: function() {},
    applyNames: function() {},
    toText: function() {},
};
var cs = {
    Capstone: function() {},
    ARCH_X86: 0,
    MODE_16: 0,
    MODE_32: 0,
    disasm: { bytes: "", mnemonic: "", op_str: "", },
};

const Buffer = {
    allocUnsafe : function(length) {},
};
