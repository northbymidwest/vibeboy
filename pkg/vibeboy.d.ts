/* tslint:disable */
/* eslint-disable */

export class WasmEmulator {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Attach a Game Boy Printer to the serial port.
     */
    attach_printer(): void;
    /**
     * Downsample the internal audio buffer by an integer factor using a
     * Blackman-windowed sinc low-pass filter.
     */
    audio_downsample(factor: number): void;
    /**
     * Drain audio samples into internal buffer; read via audio_ptr/audio_len.
     */
    audio_drain(): void;
    audio_len(): number;
    audio_ptr(): number;
    /**
     * Reverse the internal audio buffer in-place (stereo pairs).
     */
    audio_reverse(): void;
    /**
     * Apply CPU scaling filter and write result to the RGBA buffer.
     * Returns [scaled_width, scaled_height]. The JS side should use these
     * dimensions for the Canvas2D ImageData.
     */
    cpu_scaled_frame_update(disp_w: number, disp_h: number): Uint32Array;
    /**
     * Get the CLI name of the current filter.
     */
    current_filter(): string;
    frame_buffer_len(): number;
    frame_buffer_ptr(): number;
    /**
     * Get a pointer to the RGBA frame buffer in wasm memory.
     * Call frame_buffer_update() first to refresh, then use
     * frame_buffer_ptr() + frame_buffer_len() to read directly.
     */
    frame_buffer_update(): void;
    /**
     * Returns true if the cartridge has an accelerometer (MBC7 / Kirby Tilt 'n' Tumble).
     */
    has_accelerometer(): boolean;
    /**
     * Returns true if the cartridge has battery-backed save RAM.
     */
    has_battery(): boolean;
    /**
     * Returns true if the loaded ROM has a camera sensor (Pocket Camera).
     */
    has_camera(): boolean;
    /**
     * Check if the printer has a completed print ready for download.
     */
    has_print(): boolean;
    /**
     * Returns true if the cartridge has a rumble motor (MBC5+Rumble).
     */
    has_rumble(): boolean;
    height(): number;
    /**
     * Initialize WebGPU rendering on the given canvas element.
     */
    init_gpu(canvas: HTMLCanvasElement): Promise<void>;
    /**
     * Load save RAM from bytes (from localStorage).
     */
    load_save(data: Uint8Array): void;
    /**
     * Load emulator state from serialized bytes into slot.
     */
    load_state_from_bytes(slot: number, data: Uint8Array): boolean;
    /**
     * Create a new emulator from ROM bytes. Call init_gpu() after to enable WebGPU rendering.
     */
    constructor(rom: Uint8Array);
    print_height(): number;
    print_width(): number;
    /**
     * Render the current frame via WebGPU.
     * Returns true if GPU rendering was used, false if fallback needed.
     */
    render_gpu(): boolean;
    /**
     * Resize the GPU surface (call when canvas size changes).
     */
    resize_gpu(width: number, height: number): void;
    /**
     * Pop one rewind frame, regenerating the display. Returns true if rewound.
     */
    rewind_one_frame(): boolean;
    /**
     * Returns true if the rumble motor was active at any point since the last call.
     */
    rumble_active(): boolean;
    /**
     * Get cartridge save RAM as bytes (for persisting to localStorage).
     */
    save_data(): Uint8Array;
    /**
     * Get a save key derived from the ROM title + checksum.
     */
    save_key(): string;
    /**
     * Save emulator state to slot and return serialized bytes for storage.
     */
    save_state_to_bytes(slot: number): Uint8Array | undefined;
    /**
     * Feed accelerometer data. gx/gy are in g-force units (±1.0 = ±1g).
     */
    set_accelerometer(gx: number, gy: number): void;
    /**
     * Enable or disable audio sample generation. Disabling saves CPU.
     */
    set_audio_enabled(enabled: boolean): void;
    set_button(button: number, pressed: boolean): void;
    /**
     * Feed a 128x112 grayscale image from a webcam into the camera sensor.
     */
    set_camera_image(grayscale: Uint8Array): void;
    /**
     * Set the active scaling filter by name.
     * Valid names: "nearest", "vectorize", "epx", "eagle", "scale3x",
     * "bicubic", "nearest-aa", "omniscale".
     * Returns true if the filter was recognized.
     */
    set_filter(name: string): boolean;
    /**
     * Set the hardware model and restart emulation.
     * Valid names: "auto", "dmg0", "dmg", "mgb", "sgb", "sgb2", "cgb0", "cgb", "agb".
     * Returns true if the model was recognized.
     */
    set_model(name: string): boolean;
    /**
     * Set rewind mode (call before rewind_one_frame).
     */
    set_rewinding(active: boolean): void;
    /**
     * Set whether to skip the boot ROM animation.
     * Takes effect on next set_model() or ROM load.
     */
    set_skip_boot(skip: boolean): void;
    /**
     * Step one frame of emulation.
     */
    step_frame(): void;
    /**
     * Take the next print as RGBA pixel data. Also stores width/height
     * for retrieval via print_width()/print_height().
     */
    take_print_rgba(): Uint8Array;
    width(): number;
}

export function btn_a(): number;

export function btn_b(): number;

export function btn_down(): number;

export function btn_left(): number;

export function btn_right(): number;

export function btn_select(): number;

export function btn_start(): number;

export function btn_up(): number;

/**
 * Get the filter registry as a JSON string for building UI dropdowns.
 * Returns an array of `{ "cli_name": "...", "display_name": "...", "group": "..." }`.
 * Group is "main", "hqx", "xbr", "xbrz", or "edge_detect".
 */
export function filter_registry_json(): string;

/**
 * Get the wasm linear memory for zero-copy buffer access from JS.
 */
export function wasm_memory(): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmemulator_free: (a: number, b: number) => void;
    readonly btn_a: () => number;
    readonly btn_b: () => number;
    readonly btn_down: () => number;
    readonly btn_left: () => number;
    readonly btn_right: () => number;
    readonly btn_select: () => number;
    readonly btn_start: () => number;
    readonly btn_up: () => number;
    readonly filter_registry_json: () => [number, number];
    readonly wasmemulator_attach_printer: (a: number) => void;
    readonly wasmemulator_audio_downsample: (a: number, b: number) => void;
    readonly wasmemulator_audio_drain: (a: number) => void;
    readonly wasmemulator_audio_len: (a: number) => number;
    readonly wasmemulator_audio_ptr: (a: number) => number;
    readonly wasmemulator_audio_reverse: (a: number) => void;
    readonly wasmemulator_cpu_scaled_frame_update: (a: number, b: number, c: number) => [number, number];
    readonly wasmemulator_current_filter: (a: number) => [number, number];
    readonly wasmemulator_frame_buffer_len: (a: number) => number;
    readonly wasmemulator_frame_buffer_ptr: (a: number) => number;
    readonly wasmemulator_frame_buffer_update: (a: number) => void;
    readonly wasmemulator_has_accelerometer: (a: number) => number;
    readonly wasmemulator_has_battery: (a: number) => number;
    readonly wasmemulator_has_camera: (a: number) => number;
    readonly wasmemulator_has_print: (a: number) => number;
    readonly wasmemulator_has_rumble: (a: number) => number;
    readonly wasmemulator_height: (a: number) => number;
    readonly wasmemulator_init_gpu: (a: number, b: any) => any;
    readonly wasmemulator_load_save: (a: number, b: number, c: number) => void;
    readonly wasmemulator_load_state_from_bytes: (a: number, b: number, c: number, d: number) => number;
    readonly wasmemulator_new: (a: number, b: number) => [number, number, number];
    readonly wasmemulator_print_height: (a: number) => number;
    readonly wasmemulator_print_width: (a: number) => number;
    readonly wasmemulator_render_gpu: (a: number) => number;
    readonly wasmemulator_resize_gpu: (a: number, b: number, c: number) => void;
    readonly wasmemulator_rewind_one_frame: (a: number) => number;
    readonly wasmemulator_rumble_active: (a: number) => number;
    readonly wasmemulator_save_data: (a: number) => [number, number];
    readonly wasmemulator_save_key: (a: number) => [number, number];
    readonly wasmemulator_save_state_to_bytes: (a: number, b: number) => [number, number];
    readonly wasmemulator_set_accelerometer: (a: number, b: number, c: number) => void;
    readonly wasmemulator_set_audio_enabled: (a: number, b: number) => void;
    readonly wasmemulator_set_button: (a: number, b: number, c: number) => void;
    readonly wasmemulator_set_camera_image: (a: number, b: number, c: number) => void;
    readonly wasmemulator_set_filter: (a: number, b: number, c: number) => number;
    readonly wasmemulator_set_model: (a: number, b: number, c: number) => number;
    readonly wasmemulator_set_rewinding: (a: number, b: number) => void;
    readonly wasmemulator_set_skip_boot: (a: number, b: number) => void;
    readonly wasmemulator_step_frame: (a: number) => void;
    readonly wasmemulator_take_print_rgba: (a: number) => [number, number];
    readonly wasmemulator_width: (a: number) => number;
    readonly wasm_memory: () => any;
    readonly wasm_bindgen_fd7fa3b53d400c1a___closure__destroy___dyn_core_ab6e094e3d388e59___ops__function__FnMut__wasm_bindgen_fd7fa3b53d400c1a___JsValue____Output_______: (a: number, b: number) => void;
    readonly wasm_bindgen_fd7fa3b53d400c1a___closure__destroy___dyn_core_ab6e094e3d388e59___ops__function__FnMut__wasm_bindgen_fd7fa3b53d400c1a___JsValue____Output___core_ab6e094e3d388e59___result__Result_____wasm_bindgen_fd7fa3b53d400c1a___JsError___: (a: number, b: number) => void;
    readonly wasm_bindgen_fd7fa3b53d400c1a___convert__closures_____invoke___wasm_bindgen_fd7fa3b53d400c1a___JsValue__core_ab6e094e3d388e59___result__Result_____wasm_bindgen_fd7fa3b53d400c1a___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_fd7fa3b53d400c1a___convert__closures_____invoke___js_sys_34353494d4ac19ba___Function_fn_wasm_bindgen_fd7fa3b53d400c1a___JsValue_____wasm_bindgen_fd7fa3b53d400c1a___sys__Undefined___js_sys_34353494d4ac19ba___Function_fn_wasm_bindgen_fd7fa3b53d400c1a___JsValue_____wasm_bindgen_fd7fa3b53d400c1a___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_fd7fa3b53d400c1a___convert__closures_____invoke___wasm_bindgen_fd7fa3b53d400c1a___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
