/* @ts-self-types="./vibeboy.d.ts" */

export class WasmEmulator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmEmulatorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmemulator_free(ptr, 0);
    }
    /**
     * Attach a Game Boy Printer to the serial port.
     */
    attach_printer() {
        wasm.wasmemulator_attach_printer(this.__wbg_ptr);
    }
    /**
     * Downsample the internal audio buffer by an integer factor using a
     * Blackman-windowed sinc low-pass filter.
     * @param {number} factor
     */
    audio_downsample(factor) {
        wasm.wasmemulator_audio_downsample(this.__wbg_ptr, factor);
    }
    /**
     * Drain audio samples into internal buffer; read via audio_ptr/audio_len.
     */
    audio_drain() {
        wasm.wasmemulator_audio_drain(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    audio_len() {
        const ret = wasm.wasmemulator_audio_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    audio_ptr() {
        const ret = wasm.wasmemulator_audio_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Reverse the internal audio buffer in-place (stereo pairs).
     */
    audio_reverse() {
        wasm.wasmemulator_audio_reverse(this.__wbg_ptr);
    }
    /**
     * Apply CPU scaling filter and write result to the RGBA buffer.
     * Returns [scaled_width, scaled_height]. The JS side should use these
     * dimensions for the Canvas2D ImageData.
     * @param {number} disp_w
     * @param {number} disp_h
     * @returns {Uint32Array}
     */
    cpu_scaled_frame_update(disp_w, disp_h) {
        const ret = wasm.wasmemulator_cpu_scaled_frame_update(this.__wbg_ptr, disp_w, disp_h);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get the CLI name of the current filter.
     * @returns {string}
     */
    current_filter() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmemulator_current_filter(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    frame_buffer_len() {
        const ret = wasm.wasmemulator_frame_buffer_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    frame_buffer_ptr() {
        const ret = wasm.wasmemulator_frame_buffer_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get a pointer to the RGBA frame buffer in wasm memory.
     * Call frame_buffer_update() first to refresh, then use
     * frame_buffer_ptr() + frame_buffer_len() to read directly.
     */
    frame_buffer_update() {
        wasm.wasmemulator_frame_buffer_update(this.__wbg_ptr);
    }
    /**
     * Returns true if the cartridge has an accelerometer (MBC7 / Kirby Tilt 'n' Tumble).
     * @returns {boolean}
     */
    has_accelerometer() {
        const ret = wasm.wasmemulator_has_accelerometer(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Returns true if the cartridge has battery-backed save RAM.
     * @returns {boolean}
     */
    has_battery() {
        const ret = wasm.wasmemulator_has_battery(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Returns true if the loaded ROM has a camera sensor (Pocket Camera).
     * @returns {boolean}
     */
    has_camera() {
        const ret = wasm.wasmemulator_has_camera(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Check if the printer has a completed print ready for download.
     * @returns {boolean}
     */
    has_print() {
        const ret = wasm.wasmemulator_has_print(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Returns true if the cartridge has a rumble motor (MBC5+Rumble).
     * @returns {boolean}
     */
    has_rumble() {
        const ret = wasm.wasmemulator_has_rumble(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    height() {
        const ret = wasm.wasmemulator_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Initialize WebGPU rendering on the given canvas element.
     * @param {HTMLCanvasElement} canvas
     * @returns {Promise<void>}
     */
    init_gpu(canvas) {
        const ret = wasm.wasmemulator_init_gpu(this.__wbg_ptr, canvas);
        return ret;
    }
    /**
     * Load save RAM from bytes (from localStorage).
     * @param {Uint8Array} data
     */
    load_save(data) {
        const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmemulator_load_save(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Load emulator state from serialized bytes into slot.
     * @param {number} slot
     * @param {Uint8Array} data
     * @returns {boolean}
     */
    load_state_from_bytes(slot, data) {
        const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmemulator_load_state_from_bytes(this.__wbg_ptr, slot, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Create a new emulator from ROM bytes. Call init_gpu() after to enable WebGPU rendering.
     * @param {Uint8Array} rom
     */
    constructor(rom) {
        const ptr0 = passArray8ToWasm0(rom, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmemulator_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmEmulatorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    print_height() {
        const ret = wasm.wasmemulator_print_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    print_width() {
        const ret = wasm.wasmemulator_print_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Render the current frame via WebGPU.
     * Returns true if GPU rendering was used, false if fallback needed.
     * @returns {boolean}
     */
    render_gpu() {
        const ret = wasm.wasmemulator_render_gpu(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Resize the GPU surface (call when canvas size changes).
     * @param {number} width
     * @param {number} height
     */
    resize_gpu(width, height) {
        wasm.wasmemulator_resize_gpu(this.__wbg_ptr, width, height);
    }
    /**
     * Pop one rewind frame, regenerating the display. Returns true if rewound.
     * @returns {boolean}
     */
    rewind_one_frame() {
        const ret = wasm.wasmemulator_rewind_one_frame(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Returns true if the rumble motor was active at any point since the last call.
     * @returns {boolean}
     */
    rumble_active() {
        const ret = wasm.wasmemulator_rumble_active(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get cartridge save RAM as bytes (for persisting to localStorage).
     * @returns {Uint8Array}
     */
    save_data() {
        const ret = wasm.wasmemulator_save_data(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Get a save key derived from the ROM title + checksum.
     * @returns {string}
     */
    save_key() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmemulator_save_key(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Save emulator state to slot and return serialized bytes for storage.
     * @param {number} slot
     * @returns {Uint8Array | undefined}
     */
    save_state_to_bytes(slot) {
        const ret = wasm.wasmemulator_save_state_to_bytes(this.__wbg_ptr, slot);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * Feed accelerometer data. gx/gy are in g-force units (±1.0 = ±1g).
     * @param {number} gx
     * @param {number} gy
     */
    set_accelerometer(gx, gy) {
        wasm.wasmemulator_set_accelerometer(this.__wbg_ptr, gx, gy);
    }
    /**
     * Enable or disable audio sample generation. Disabling saves CPU.
     * @param {boolean} enabled
     */
    set_audio_enabled(enabled) {
        wasm.wasmemulator_set_audio_enabled(this.__wbg_ptr, enabled);
    }
    /**
     * @param {number} button
     * @param {boolean} pressed
     */
    set_button(button, pressed) {
        wasm.wasmemulator_set_button(this.__wbg_ptr, button, pressed);
    }
    /**
     * Feed a 128x112 grayscale image from a webcam into the camera sensor.
     * @param {Uint8Array} grayscale
     */
    set_camera_image(grayscale) {
        const ptr0 = passArray8ToWasm0(grayscale, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmemulator_set_camera_image(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set the active scaling filter by name.
     * Valid names: "nearest", "vectorize", "epx", "eagle", "scale3x",
     * "bicubic", "nearest-aa", "omniscale".
     * Returns true if the filter was recognized.
     * @param {string} name
     * @returns {boolean}
     */
    set_filter(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmemulator_set_filter(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Set the hardware model and restart emulation.
     * Valid names: "auto", "dmg0", "dmg", "mgb", "sgb", "sgb2", "cgb0", "cgb", "agb".
     * Returns true if the model was recognized.
     * @param {string} name
     * @returns {boolean}
     */
    set_model(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmemulator_set_model(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Set rewind mode (call before rewind_one_frame).
     * @param {boolean} active
     */
    set_rewinding(active) {
        wasm.wasmemulator_set_rewinding(this.__wbg_ptr, active);
    }
    /**
     * Set whether to skip the boot ROM animation.
     * Takes effect on next set_model() or ROM load.
     * @param {boolean} skip
     */
    set_skip_boot(skip) {
        wasm.wasmemulator_set_skip_boot(this.__wbg_ptr, skip);
    }
    /**
     * Step one frame of emulation.
     */
    step_frame() {
        wasm.wasmemulator_step_frame(this.__wbg_ptr);
    }
    /**
     * Take the next print as RGBA pixel data. Also stores width/height
     * for retrieval via print_width()/print_height().
     * @returns {Uint8Array}
     */
    take_print_rgba() {
        const ret = wasm.wasmemulator_take_print_rgba(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * @returns {number}
     */
    width() {
        const ret = wasm.wasmemulator_width(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WasmEmulator.prototype[Symbol.dispose] = WasmEmulator.prototype.free;

/**
 * @returns {number}
 */
export function btn_a() {
    const ret = wasm.btn_a();
    return ret;
}

/**
 * @returns {number}
 */
export function btn_b() {
    const ret = wasm.btn_b();
    return ret;
}

/**
 * @returns {number}
 */
export function btn_down() {
    const ret = wasm.btn_down();
    return ret;
}

/**
 * @returns {number}
 */
export function btn_left() {
    const ret = wasm.btn_left();
    return ret;
}

/**
 * @returns {number}
 */
export function btn_right() {
    const ret = wasm.btn_right();
    return ret;
}

/**
 * @returns {number}
 */
export function btn_select() {
    const ret = wasm.btn_select();
    return ret;
}

/**
 * @returns {number}
 */
export function btn_start() {
    const ret = wasm.btn_start();
    return ret;
}

/**
 * @returns {number}
 */
export function btn_up() {
    const ret = wasm.btn_up();
    return ret;
}

/**
 * Get the filter registry as a JSON string for building UI dropdowns.
 * Returns an array of `{ "cli_name": "...", "display_name": "...", "group": "..." }`.
 * Group is "main", "hqx", "xbr", "xbrz", or "edge_detect".
 * @returns {string}
 */
export function filter_registry_json() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.filter_registry_json();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Get the wasm linear memory for zero-copy buffer access from JS.
 * @returns {any}
 */
export function wasm_memory() {
    const ret = wasm.wasm_memory();
    return ret;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Window_5bac5165340af82e: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_d0d150069210a6e8: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_debug_string_ab4b34d23d6778bd: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_3baa9db1a987f47d: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_52ff4ec04186736f: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_undefined_29a43b4d42920abd: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_memory_dfa12096f400c9bd: function() {
            const ret = wasm.memory;
            return ret;
        },
        __wbg___wbindgen_throw_6b64449b9b9ed33c: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_b46c9b5a9f08ec37: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_beginComputePass_8409c5720049090e: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_beginRenderPass_a19cc6156a7858b4: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.beginRenderPass(arg1);
            return ret;
        }, arguments); },
        __wbg_call_a24592a6f349a97e: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_configure_16541864db644c70: function() { return handleError(function (arg0, arg1) {
            arg0.configure(arg1);
        }, arguments); },
        __wbg_copyBufferToBuffer_ab55c9b7610f063b: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4, arg5);
        }, arguments); },
        __wbg_copyBufferToBuffer_b9173642596dac6a: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4);
        }, arguments); },
        __wbg_createBindGroupLayout_adb8337a6808ae24: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBindGroupLayout(arg1);
            return ret;
        }, arguments); },
        __wbg_createBindGroup_91159ca759115307: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_59de141e89014140: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        }, arguments); },
        __wbg_createCommandEncoder_dc2b2ca6f09bd4c3: function(arg0, arg1) {
            const ret = arg0.createCommandEncoder(arg1);
            return ret;
        },
        __wbg_createComputePipeline_d9b5cceb5f793856: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createPipelineLayout_a5290f84492f8b1e: function(arg0, arg1) {
            const ret = arg0.createPipelineLayout(arg1);
            return ret;
        },
        __wbg_createRenderPipeline_f7aca470ad8ce865: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createRenderPipeline(arg1);
            return ret;
        }, arguments); },
        __wbg_createSampler_6b972cd00bcc5dfb: function(arg0, arg1) {
            const ret = arg0.createSampler(arg1);
            return ret;
        },
        __wbg_createShaderModule_bbe0476992dd060e: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_createTexture_011d4b0badf853e3: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createTexture(arg1);
            return ret;
        }, arguments); },
        __wbg_createView_1ef8f1ddc16facb0: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createView(arg1);
            return ret;
        }, arguments); },
        __wbg_debug_c014a160490283dc: function(arg0) {
            console.debug(arg0);
        },
        __wbg_dispatchWorkgroups_863a24e86337dc58: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_document_7a41071f2f439323: function(arg0) {
            const ret = arg0.document;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_draw_9a35daa0096c6f2c: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.draw(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_end_176171edf82ded94: function(arg0) {
            arg0.end();
        },
        __wbg_end_1db12af2e0ff1235: function(arg0) {
            arg0.end();
        },
        __wbg_error_2001591ad2463697: function(arg0) {
            console.error(arg0);
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_finish_48a7b6da7b76999e: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_finish_68d7c5925d3fa394: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_getBindGroupLayout_3f7c6978558a8c4e: function(arg0, arg1) {
            const ret = arg0.getBindGroupLayout(arg1 >>> 0);
            return ret;
        },
        __wbg_getContext_69ddc504535a2e7b: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getContext_fc146f8ec021d074: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getCurrentTexture_9b00da7f6bc38606: function() { return handleError(function (arg0) {
            const ret = arg0.getCurrentTexture();
            return ret;
        }, arguments); },
        __wbg_getPreferredCanvasFormat_54381f1ef7aec03d: function(arg0) {
            const ret = arg0.getPreferredCanvasFormat();
            return (__wbindgen_enum_GpuTextureFormat.indexOf(ret) + 1 || 96) - 1;
        },
        __wbg_get_0cfbe604d86bac03: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_gpu_3f9d7df9a18237f8: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_height_528848d067cc2221: function(arg0) {
            const ret = arg0.height;
            return ret;
        },
        __wbg_info_7479429238bffbce: function(arg0) {
            console.info(arg0);
        },
        __wbg_instanceof_GpuAdapter_dc7e13c1676da9bd: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUAdapter;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuCanvasContext_c2609c698a76a6b6: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUCanvasContext;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Window_cc64c86c8ef9e02b: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_label_18cae34ff19933d7: function(arg0, arg1) {
            const ret = arg1.label;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_log_7e1aa9064a1dbdbd: function(arg0) {
            console.log(arg0);
        },
        __wbg_mapAsync_288e2fddbc3f7f7b: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.mapAsync(arg1 >>> 0, arg2, arg3);
            return ret;
        },
        __wbg_navigator_353318de944ca7f6: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_bc077756492232c5: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_682678e2f47e32bc: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_aa8d0fa9762c29bd: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_typed_323f37fd55ab048d: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_30888b4dce7fd75a___convert__closures_____invoke___js_sys_12065f71e6567827___Function_fn_wasm_bindgen_30888b4dce7fd75a___JsValue_____wasm_bindgen_30888b4dce7fd75a___sys__Undefined___js_sys_12065f71e6567827___Function_fn_wasm_bindgen_30888b4dce7fd75a___JsValue_____wasm_bindgen_30888b4dce7fd75a___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_now_a9b7df1cbee90986: function() {
            const ret = Date.now();
            return ret;
        },
        __wbg_onSubmittedWorkDone_81e152567230130a: function(arg0) {
            const ret = arg0.onSubmittedWorkDone();
            return ret;
        },
        __wbg_push_471a5b068a5295f6: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_querySelectorAll_e9e3fbd41310476e: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.querySelectorAll(getStringFromWasm0(arg1, arg2));
            return ret;
        }, arguments); },
        __wbg_queueMicrotask_5d15a957e6aa920e: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queueMicrotask_f8819e5ffc402f36: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queue_81f5d725809ccd54: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_requestAdapter_90f7496e67f82c21: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_5c307ce72228d3f7: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolve_e6c466bc1052f16c: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_setBindGroup_2b53fd8bfb43674d: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        }, arguments); },
        __wbg_setBindGroup_58960c4b1bcdd182: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setBindGroup_5d9cc0ccc87c9640: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setBindGroup_a62f9de1cb2449b2: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        }, arguments); },
        __wbg_setPipeline_1fc98ad28be0d04a: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_setPipeline_9f6b0a3c5901572d: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_set_022bee52d0b05b19: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_a_2f4495829c853bba: function(arg0, arg1) {
            arg0.a = arg1;
        },
        __wbg_set_access_802ef755476d4064: function(arg0, arg1) {
            arg0.access = __wbindgen_enum_GpuStorageTextureAccess[arg1];
        },
        __wbg_set_address_mode_u_c13cdf94d097b16d: function(arg0, arg1) {
            arg0.addressModeU = __wbindgen_enum_GpuAddressMode[arg1];
        },
        __wbg_set_address_mode_v_c09db9861cd052a6: function(arg0, arg1) {
            arg0.addressModeV = __wbindgen_enum_GpuAddressMode[arg1];
        },
        __wbg_set_address_mode_w_0b49c35f3d4322bf: function(arg0, arg1) {
            arg0.addressModeW = __wbindgen_enum_GpuAddressMode[arg1];
        },
        __wbg_set_alpha_29642d2219224544: function(arg0, arg1) {
            arg0.alpha = arg1;
        },
        __wbg_set_alpha_mode_65ba0adaef90e1f3: function(arg0, arg1) {
            arg0.alphaMode = __wbindgen_enum_GpuCanvasAlphaMode[arg1];
        },
        __wbg_set_alpha_to_coverage_enabled_ab6a22e18e338493: function(arg0, arg1) {
            arg0.alphaToCoverageEnabled = arg1 !== 0;
        },
        __wbg_set_array_layer_count_de83f575c3f6d15e: function(arg0, arg1) {
            arg0.arrayLayerCount = arg1 >>> 0;
        },
        __wbg_set_array_stride_2033aeb8a42130f9: function(arg0, arg1) {
            arg0.arrayStride = arg1;
        },
        __wbg_set_aspect_feb0fac859e82372: function(arg0, arg1) {
            arg0.aspect = __wbindgen_enum_GpuTextureAspect[arg1];
        },
        __wbg_set_attributes_39e5a71bf05309a6: function(arg0, arg1) {
            arg0.attributes = arg1;
        },
        __wbg_set_b_7081554879455e65: function(arg0, arg1) {
            arg0.b = arg1;
        },
        __wbg_set_base_array_layer_ab196aad24c8fac6: function(arg0, arg1) {
            arg0.baseArrayLayer = arg1 >>> 0;
        },
        __wbg_set_base_mip_level_15d29fc182e25a82: function(arg0, arg1) {
            arg0.baseMipLevel = arg1 >>> 0;
        },
        __wbg_set_beginning_of_pass_write_index_0cb71e33ea66ffc8: function(arg0, arg1) {
            arg0.beginningOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_beginning_of_pass_write_index_c2f97408798615ca: function(arg0, arg1) {
            arg0.beginningOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_bind_group_layouts_5c298441f47e30a1: function(arg0, arg1) {
            arg0.bindGroupLayouts = arg1;
        },
        __wbg_set_binding_234b4c508d19a0a8: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_binding_fd933455b600a07f: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_blend_1dbdd086fc4fdebf: function(arg0, arg1) {
            arg0.blend = arg1;
        },
        __wbg_set_buffer_8f0ef5be1b92d605: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_b04e4d70b1eb4630: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffers_3f9c487ea01dddcf: function(arg0, arg1) {
            arg0.buffers = arg1;
        },
        __wbg_set_clear_value_1663cbe7da00e7e4: function(arg0, arg1) {
            arg0.clearValue = arg1;
        },
        __wbg_set_code_3bb44fc02aa17153: function(arg0, arg1, arg2) {
            arg0.code = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_color_attachments_b740d060dacde5c0: function(arg0, arg1) {
            arg0.colorAttachments = arg1;
        },
        __wbg_set_color_d0208d092af4f2e6: function(arg0, arg1) {
            arg0.color = arg1;
        },
        __wbg_set_compare_00dc33383c873ad5: function(arg0, arg1) {
            arg0.compare = __wbindgen_enum_GpuCompareFunction[arg1];
        },
        __wbg_set_compare_11834994f7d75687: function(arg0, arg1) {
            arg0.compare = __wbindgen_enum_GpuCompareFunction[arg1];
        },
        __wbg_set_compute_6c1b9ba0e3041f2c: function(arg0, arg1) {
            arg0.compute = arg1;
        },
        __wbg_set_count_ab42cbc78635ed91: function(arg0, arg1) {
            arg0.count = arg1 >>> 0;
        },
        __wbg_set_cull_mode_c4f1ef740bd14c40: function(arg0, arg1) {
            arg0.cullMode = __wbindgen_enum_GpuCullMode[arg1];
        },
        __wbg_set_depth_bias_clamp_f573c2dda55692a6: function(arg0, arg1) {
            arg0.depthBiasClamp = arg1;
        },
        __wbg_set_depth_bias_ebe05aecbb98e11f: function(arg0, arg1) {
            arg0.depthBias = arg1;
        },
        __wbg_set_depth_bias_slope_scale_27c8208740c46086: function(arg0, arg1) {
            arg0.depthBiasSlopeScale = arg1;
        },
        __wbg_set_depth_clear_value_57c2283d39fbb181: function(arg0, arg1) {
            arg0.depthClearValue = arg1;
        },
        __wbg_set_depth_compare_a9c538cec0e01535: function(arg0, arg1) {
            arg0.depthCompare = __wbindgen_enum_GpuCompareFunction[arg1];
        },
        __wbg_set_depth_fail_op_42b9d46a7c67baae: function(arg0, arg1) {
            arg0.depthFailOp = __wbindgen_enum_GpuStencilOperation[arg1];
        },
        __wbg_set_depth_load_op_f95fdb158b819261: function(arg0, arg1) {
            arg0.depthLoadOp = __wbindgen_enum_GpuLoadOp[arg1];
        },
        __wbg_set_depth_or_array_layers_7335d3fc04cd5ade: function(arg0, arg1) {
            arg0.depthOrArrayLayers = arg1 >>> 0;
        },
        __wbg_set_depth_read_only_878b741b02a4dd71: function(arg0, arg1) {
            arg0.depthReadOnly = arg1 !== 0;
        },
        __wbg_set_depth_stencil_1c7bed669574dd1e: function(arg0, arg1) {
            arg0.depthStencil = arg1;
        },
        __wbg_set_depth_stencil_attachment_82ce8924f4e0e79b: function(arg0, arg1) {
            arg0.depthStencilAttachment = arg1;
        },
        __wbg_set_depth_store_op_4c56ab1d005c7bf6: function(arg0, arg1) {
            arg0.depthStoreOp = __wbindgen_enum_GpuStoreOp[arg1];
        },
        __wbg_set_depth_write_enabled_f726d4f27a24ff7e: function(arg0, arg1) {
            arg0.depthWriteEnabled = arg1 !== 0;
        },
        __wbg_set_device_f991f8a955db69f7: function(arg0, arg1) {
            arg0.device = arg1;
        },
        __wbg_set_dimension_7ca3d24380d365e4: function(arg0, arg1) {
            arg0.dimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_dimension_87dd70a08e54ea98: function(arg0, arg1) {
            arg0.dimension = __wbindgen_enum_GpuTextureDimension[arg1];
        },
        __wbg_set_dst_factor_1382684d97e2aec4: function(arg0, arg1) {
            arg0.dstFactor = __wbindgen_enum_GpuBlendFactor[arg1];
        },
        __wbg_set_end_of_pass_write_index_3476a9a4411846af: function(arg0, arg1) {
            arg0.endOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_end_of_pass_write_index_5e969b5aa2f94e75: function(arg0, arg1) {
            arg0.endOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_entries_44ee8dc60918063d: function(arg0, arg1) {
            arg0.entries = arg1;
        },
        __wbg_set_entries_803b89386febf57c: function(arg0, arg1) {
            arg0.entries = arg1;
        },
        __wbg_set_entry_point_418e5aecbf7f95b4: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_entry_point_a84dd78ae4a97c6d: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_entry_point_ac45ddee35909233: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_external_texture_73d5e5303574a1e8: function(arg0, arg1) {
            arg0.externalTexture = arg1;
        },
        __wbg_set_fail_op_6f4612035f584d02: function(arg0, arg1) {
            arg0.failOp = __wbindgen_enum_GpuStencilOperation[arg1];
        },
        __wbg_set_format_2bd90cb220cc6884: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_3cc5d6ead9a8cce0: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_40d793124494a9df: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_723d6bb38a9e71d3: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuVertexFormat[arg1];
        },
        __wbg_set_format_c23f7c142762c3a7: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_e0af83ab86ee58dc: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_fcbaa54d6b5c186a: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_fragment_9b5673b1b740fe0e: function(arg0, arg1) {
            arg0.fragment = arg1;
        },
        __wbg_set_front_face_bb590812353fd2e0: function(arg0, arg1) {
            arg0.frontFace = __wbindgen_enum_GpuFrontFace[arg1];
        },
        __wbg_set_g_aa23517844bd7f61: function(arg0, arg1) {
            arg0.g = arg1;
        },
        __wbg_set_has_dynamic_offset_ea1fb6bd94b0c904: function(arg0, arg1) {
            arg0.hasDynamicOffset = arg1 !== 0;
        },
        __wbg_set_height_24d07d982f176ac6: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_height_66583e77881d3a51: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_height_be9b2b920bd68401: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_label_08e9f27a97fdc9f7: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_0e9f90ea4e961823: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_1736939dde71ec96: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_280bd57b618e4cf6: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_34d2766c2203f76a: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_4bf9f5458cdc0a68: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_797345a8c9c86146: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_8fdd5f28eea3ca08: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_a4be4acc3510c62f: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_bb92451e0d92abf4: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_c3405868bd8f6ab5: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_d5519c3081c41e5a: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_d73358f96a62d3bc: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_f00eb249a34df7db: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_f571593aaa82f18b: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_layout_9590b02a1d72ac45: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_a065a939d1d05a2d: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_d008ec94bedc0844: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_load_op_07c59d4ab60a3a01: function(arg0, arg1) {
            arg0.loadOp = __wbindgen_enum_GpuLoadOp[arg1];
        },
        __wbg_set_lod_max_clamp_fd1548dc78538913: function(arg0, arg1) {
            arg0.lodMaxClamp = arg1;
        },
        __wbg_set_lod_min_clamp_b489016289e378d2: function(arg0, arg1) {
            arg0.lodMinClamp = arg1;
        },
        __wbg_set_mag_filter_b4e8d7f2fa665d2e: function(arg0, arg1) {
            arg0.magFilter = __wbindgen_enum_GpuFilterMode[arg1];
        },
        __wbg_set_mapped_at_creation_c78869832c67816c: function(arg0, arg1) {
            arg0.mappedAtCreation = arg1 !== 0;
        },
        __wbg_set_mask_cee9de29cbe61459: function(arg0, arg1) {
            arg0.mask = arg1 >>> 0;
        },
        __wbg_set_max_anisotropy_a019fd38d9ba634e: function(arg0, arg1) {
            arg0.maxAnisotropy = arg1;
        },
        __wbg_set_min_binding_size_26f877007450686c: function(arg0, arg1) {
            arg0.minBindingSize = arg1;
        },
        __wbg_set_min_filter_cd8cf3dcdeebaa5b: function(arg0, arg1) {
            arg0.minFilter = __wbindgen_enum_GpuFilterMode[arg1];
        },
        __wbg_set_mip_level_count_1993f039035d2469: function(arg0, arg1) {
            arg0.mipLevelCount = arg1 >>> 0;
        },
        __wbg_set_mip_level_count_9a86e098393fe360: function(arg0, arg1) {
            arg0.mipLevelCount = arg1 >>> 0;
        },
        __wbg_set_mipmap_filter_a436d61249cfa785: function(arg0, arg1) {
            arg0.mipmapFilter = __wbindgen_enum_GpuMipmapFilterMode[arg1];
        },
        __wbg_set_module_77c9a4994de5185d: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_module_951f2b6e5477a260: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_module_a7b3448454ca8879: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_multisample_bb6537e862d91237: function(arg0, arg1) {
            arg0.multisample = arg1;
        },
        __wbg_set_multisampled_9642e942e4d9d3ee: function(arg0, arg1) {
            arg0.multisampled = arg1 !== 0;
        },
        __wbg_set_offset_3e55dd16ffd7aac5: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_offset_debfe602a5fbf272: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_operation_74a529d361734388: function(arg0, arg1) {
            arg0.operation = __wbindgen_enum_GpuBlendOperation[arg1];
        },
        __wbg_set_pass_op_8abd39478c76666a: function(arg0, arg1) {
            arg0.passOp = __wbindgen_enum_GpuStencilOperation[arg1];
        },
        __wbg_set_power_preference_b8b4ea5da6674cf7: function(arg0, arg1) {
            arg0.powerPreference = __wbindgen_enum_GpuPowerPreference[arg1];
        },
        __wbg_set_primitive_f189fcdcb22d09e0: function(arg0, arg1) {
            arg0.primitive = arg1;
        },
        __wbg_set_query_set_2862e48f0ed8ffe8: function(arg0, arg1) {
            arg0.querySet = arg1;
        },
        __wbg_set_query_set_dcf406a51ece8f85: function(arg0, arg1) {
            arg0.querySet = arg1;
        },
        __wbg_set_r_8961014434a7656e: function(arg0, arg1) {
            arg0.r = arg1;
        },
        __wbg_set_required_features_ec67124fd26c4d29: function(arg0, arg1) {
            arg0.requiredFeatures = arg1;
        },
        __wbg_set_required_limits_c9ee7006f1d1f2ab: function(arg0, arg1) {
            arg0.requiredLimits = arg1;
        },
        __wbg_set_resolve_target_cc7a6f0d2973ea34: function(arg0, arg1) {
            arg0.resolveTarget = arg1;
        },
        __wbg_set_resource_86645e7515651c0e: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_sample_count_4d7160817d98838f: function(arg0, arg1) {
            arg0.sampleCount = arg1 >>> 0;
        },
        __wbg_set_sample_type_8d4d5b141ce0f724: function(arg0, arg1) {
            arg0.sampleType = __wbindgen_enum_GpuTextureSampleType[arg1];
        },
        __wbg_set_sampler_35bcbac78bd4356f: function(arg0, arg1) {
            arg0.sampler = arg1;
        },
        __wbg_set_shader_location_3ce5152f6d464a63: function(arg0, arg1) {
            arg0.shaderLocation = arg1 >>> 0;
        },
        __wbg_set_size_81a77f7f4f34fbed: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_85cb1c2c4c3ea73a: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_981550e5d7941340: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_src_factor_9a8e0943a05c9174: function(arg0, arg1) {
            arg0.srcFactor = __wbindgen_enum_GpuBlendFactor[arg1];
        },
        __wbg_set_stencil_back_596ea9628419413d: function(arg0, arg1) {
            arg0.stencilBack = arg1;
        },
        __wbg_set_stencil_clear_value_15afeb03c22cd51d: function(arg0, arg1) {
            arg0.stencilClearValue = arg1 >>> 0;
        },
        __wbg_set_stencil_front_31be994e05be5aaa: function(arg0, arg1) {
            arg0.stencilFront = arg1;
        },
        __wbg_set_stencil_load_op_1cd94e9e8c54f611: function(arg0, arg1) {
            arg0.stencilLoadOp = __wbindgen_enum_GpuLoadOp[arg1];
        },
        __wbg_set_stencil_read_mask_1635f30a0e6539e3: function(arg0, arg1) {
            arg0.stencilReadMask = arg1 >>> 0;
        },
        __wbg_set_stencil_read_only_f071431988182ad8: function(arg0, arg1) {
            arg0.stencilReadOnly = arg1 !== 0;
        },
        __wbg_set_stencil_store_op_a244d5347f386c8c: function(arg0, arg1) {
            arg0.stencilStoreOp = __wbindgen_enum_GpuStoreOp[arg1];
        },
        __wbg_set_stencil_write_mask_7809f82a1debe58f: function(arg0, arg1) {
            arg0.stencilWriteMask = arg1 >>> 0;
        },
        __wbg_set_step_mode_eb762c8c4264418f: function(arg0, arg1) {
            arg0.stepMode = __wbindgen_enum_GpuVertexStepMode[arg1];
        },
        __wbg_set_storage_texture_22f78b5171d1195a: function(arg0, arg1) {
            arg0.storageTexture = arg1;
        },
        __wbg_set_store_op_386596acc7bf2c16: function(arg0, arg1) {
            arg0.storeOp = __wbindgen_enum_GpuStoreOp[arg1];
        },
        __wbg_set_strip_index_format_e76748cd840ab562: function(arg0, arg1) {
            arg0.stripIndexFormat = __wbindgen_enum_GpuIndexFormat[arg1];
        },
        __wbg_set_targets_22473476afe0dabd: function(arg0, arg1) {
            arg0.targets = arg1;
        },
        __wbg_set_texture_aeea930400349204: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_timestamp_writes_0236dfc7ae2b1a03: function(arg0, arg1) {
            arg0.timestampWrites = arg1;
        },
        __wbg_set_timestamp_writes_d1259248cc80f658: function(arg0, arg1) {
            arg0.timestampWrites = arg1;
        },
        __wbg_set_topology_e18a15a717ebc912: function(arg0, arg1) {
            arg0.topology = __wbindgen_enum_GpuPrimitiveTopology[arg1];
        },
        __wbg_set_type_31b1662dd5a6144d: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuSamplerBindingType[arg1];
        },
        __wbg_set_type_719f40cf36d314f1: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuBufferBindingType[arg1];
        },
        __wbg_set_unclipped_depth_0f5d142d317e3a7c: function(arg0, arg1) {
            arg0.unclippedDepth = arg1 !== 0;
        },
        __wbg_set_usage_26861a639595cd45: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_usage_7b79a227ada2f5cc: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_usage_d9ff4b7757fac246: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_usage_e8d45decd5c483b3: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_vertex_b95705590b782671: function(arg0, arg1) {
            arg0.vertex = arg1;
        },
        __wbg_set_view_6ff951d6e3f9e337: function(arg0, arg1) {
            arg0.view = arg1;
        },
        __wbg_set_view_cf298e1e7b6ef38a: function(arg0, arg1) {
            arg0.view = arg1;
        },
        __wbg_set_view_dimension_87c95b0d987a14cd: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_dimension_e99ec138da7b8f83: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_formats_733fb624c2f2ef6b: function(arg0, arg1) {
            arg0.viewFormats = arg1;
        },
        __wbg_set_view_formats_c2b27891ca5d2740: function(arg0, arg1) {
            arg0.viewFormats = arg1;
        },
        __wbg_set_visibility_315bcac6427d0ba0: function(arg0, arg1) {
            arg0.visibility = arg1 >>> 0;
        },
        __wbg_set_width_5cda41d4d06a14dd: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_width_63034f88f9905ea3: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_width_adc925bca9c5351a: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_write_mask_0b6ca0cb1b797997: function(arg0, arg1) {
            arg0.writeMask = arg1 >>> 0;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_8cfadc87a297ca02: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_602256ae5c8f42cf: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_e445c1c7484aecc3: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_f20e8576ef1e0f17: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_submit_f39583470d95df20: function(arg0, arg1) {
            arg0.submit(arg1);
        },
        __wbg_then_792e0c862b060889: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_8e16ee11f05e4827: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_then_a50dc2689b076063: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_warn_3cc416af27dbdc02: function(arg0) {
            console.warn(arg0);
        },
        __wbg_width_5adcb07d04d08bdf: function(arg0) {
            const ret = arg0.width;
            return ret;
        },
        __wbg_writeBuffer_2384abff9a0faef7: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.writeBuffer(arg1, arg2, getArrayU8FromWasm0(arg3, arg4), arg5, arg6);
        }, arguments); },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 294, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_30888b4dce7fd75a___convert__closures_____invoke___wasm_bindgen_30888b4dce7fd75a___JsValue______true_);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 320, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_30888b4dce7fd75a___convert__closures_____invoke___wasm_bindgen_30888b4dce7fd75a___JsValue__core_ab6e094e3d388e59___result__Result_____wasm_bindgen_30888b4dce7fd75a___JsError___true_);
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./vibeboy_bg.js": import0,
    };
}

function wasm_bindgen_30888b4dce7fd75a___convert__closures_____invoke___wasm_bindgen_30888b4dce7fd75a___JsValue______true_(arg0, arg1, arg2) {
    wasm.wasm_bindgen_30888b4dce7fd75a___convert__closures_____invoke___wasm_bindgen_30888b4dce7fd75a___JsValue______true_(arg0, arg1, arg2);
}

function wasm_bindgen_30888b4dce7fd75a___convert__closures_____invoke___wasm_bindgen_30888b4dce7fd75a___JsValue__core_ab6e094e3d388e59___result__Result_____wasm_bindgen_30888b4dce7fd75a___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_30888b4dce7fd75a___convert__closures_____invoke___wasm_bindgen_30888b4dce7fd75a___JsValue__core_ab6e094e3d388e59___result__Result_____wasm_bindgen_30888b4dce7fd75a___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_30888b4dce7fd75a___convert__closures_____invoke___js_sys_12065f71e6567827___Function_fn_wasm_bindgen_30888b4dce7fd75a___JsValue_____wasm_bindgen_30888b4dce7fd75a___sys__Undefined___js_sys_12065f71e6567827___Function_fn_wasm_bindgen_30888b4dce7fd75a___JsValue_____wasm_bindgen_30888b4dce7fd75a___sys__Undefined_______true_(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen_30888b4dce7fd75a___convert__closures_____invoke___js_sys_12065f71e6567827___Function_fn_wasm_bindgen_30888b4dce7fd75a___JsValue_____wasm_bindgen_30888b4dce7fd75a___sys__Undefined___js_sys_12065f71e6567827___Function_fn_wasm_bindgen_30888b4dce7fd75a___JsValue_____wasm_bindgen_30888b4dce7fd75a___sys__Undefined_______true_(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuAddressMode = ["clamp-to-edge", "repeat", "mirror-repeat"];


const __wbindgen_enum_GpuBlendFactor = ["zero", "one", "src", "one-minus-src", "src-alpha", "one-minus-src-alpha", "dst", "one-minus-dst", "dst-alpha", "one-minus-dst-alpha", "src-alpha-saturated", "constant", "one-minus-constant", "src1", "one-minus-src1", "src1-alpha", "one-minus-src1-alpha"];


const __wbindgen_enum_GpuBlendOperation = ["add", "subtract", "reverse-subtract", "min", "max"];


const __wbindgen_enum_GpuBufferBindingType = ["uniform", "storage", "read-only-storage"];


const __wbindgen_enum_GpuCanvasAlphaMode = ["opaque", "premultiplied"];


const __wbindgen_enum_GpuCompareFunction = ["never", "less", "equal", "less-equal", "greater", "not-equal", "greater-equal", "always"];


const __wbindgen_enum_GpuCullMode = ["none", "front", "back"];


const __wbindgen_enum_GpuFilterMode = ["nearest", "linear"];


const __wbindgen_enum_GpuFrontFace = ["ccw", "cw"];


const __wbindgen_enum_GpuIndexFormat = ["uint16", "uint32"];


const __wbindgen_enum_GpuLoadOp = ["load", "clear"];


const __wbindgen_enum_GpuMipmapFilterMode = ["nearest", "linear"];


const __wbindgen_enum_GpuPowerPreference = ["low-power", "high-performance"];


const __wbindgen_enum_GpuPrimitiveTopology = ["point-list", "line-list", "line-strip", "triangle-list", "triangle-strip"];


const __wbindgen_enum_GpuSamplerBindingType = ["filtering", "non-filtering", "comparison"];


const __wbindgen_enum_GpuStencilOperation = ["keep", "zero", "replace", "invert", "increment-clamp", "decrement-clamp", "increment-wrap", "decrement-wrap"];


const __wbindgen_enum_GpuStorageTextureAccess = ["write-only", "read-only", "read-write"];


const __wbindgen_enum_GpuStoreOp = ["store", "discard"];


const __wbindgen_enum_GpuTextureAspect = ["all", "stencil-only", "depth-only"];


const __wbindgen_enum_GpuTextureDimension = ["1d", "2d", "3d"];


const __wbindgen_enum_GpuTextureFormat = ["r8unorm", "r8snorm", "r8uint", "r8sint", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint", "r32float", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint", "rg32float", "rgba16uint", "rgba16sint", "rgba16float", "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"];


const __wbindgen_enum_GpuTextureSampleType = ["float", "unfilterable-float", "depth", "sint", "uint"];


const __wbindgen_enum_GpuTextureViewDimension = ["1d", "2d", "2d-array", "cube", "cube-array", "3d"];


const __wbindgen_enum_GpuVertexFormat = ["uint8", "uint8x2", "uint8x4", "sint8", "sint8x2", "sint8x4", "unorm8", "unorm8x2", "unorm8x4", "snorm8", "snorm8x2", "snorm8x4", "uint16", "uint16x2", "uint16x4", "sint16", "sint16x2", "sint16x4", "unorm16", "unorm16x2", "unorm16x4", "snorm16", "snorm16x2", "snorm16x4", "float16", "float16x2", "float16x4", "float32", "float32x2", "float32x3", "float32x4", "uint32", "uint32x2", "uint32x3", "uint32x4", "sint32", "sint32x2", "sint32x3", "sint32x4", "unorm10-10-10-2", "unorm8x4-bgra"];


const __wbindgen_enum_GpuVertexStepMode = ["vertex", "instance"];
const WasmEmulatorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmemulator_free(ptr >>> 0, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('vibeboy_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
