// VibeBoy — Main emulator module

import { setupTouchControls } from './touch.js';

// ── Shared state ──────────────────────────────────────────

const state = {
  emu: null,
  useGpu: false,
  gpuAvailable: false,
  animFrameId: null,
  ctx: null,
  audioCtx: null,
  audioWorklet: null,
  keyMap: {},
  paused: false,
  stepOneFrame: false,
  rewindHeld: false,
  fastForwardHeld: false,
  kbRewindHeld: false,
  kbFastFwdHeld: false,
  currentSlot: 0,
  emuTime: 0,
  lastTimestamp: 0,
  printCount: 0,
  cameraActive: false,
  accelActive: false,
  accelX: 0,
  accelY: 0,
  deviceMotionHandler: null,
  audioBuffered: 0,
  gpPrevState: {},
  rumbleWasOn: false,
  saveFlushScheduled: false,
};

// Lazy-loaded wasm module references
let wasmMod = null;     // the default export (init function)
let WasmEmulator = null;
let wasm_memory = null;
let filter_registry_json = null;
let btn_a = null, btn_b = null, btn_up = null, btn_down = null;
let btn_left = null, btn_right = null, btn_start = null, btn_select = null;

// ── DOM references ────────────────────────────────────────

const canvas = document.getElementById('screen');
const canvas2d = document.getElementById('screen-2d');
const dropZone = document.getElementById('canvas-wrapper');
const romPrompt = document.getElementById('rom-prompt');
const fileInput = document.getElementById('file-input');
const statusEl = document.getElementById('status');
const filterBar = document.getElementById('filter-bar');
const filterSelect = document.getElementById('filter-select');
const modelSelect = document.getElementById('model-select');
const forceCpuCheckbox = document.getElementById('force-cpu');
const skipBootCheckbox = document.getElementById('skip-boot');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = loadingOverlay ? loadingOverlay.querySelector('.loading-text') : null;
const loadingError = loadingOverlay ? loadingOverlay.querySelector('.loading-error') : null;
const camVideo = document.getElementById('cam-video');
const camCanvas = document.getElementById('cam-canvas');
const camCtx = camCanvas.getContext('2d', { willReadFrequently: true });
const camGrayscale = new Uint8Array(128 * 112);

const GB_FRAME_MS = 1000.0 / 59.7275;
const STICK_DEADZONE = 0.4;

// ── Toast notifications ───────────────────────────────────

export function showToast(msg, durationMs = 2000) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  el.style.animationDuration = '0.25s, 0.3s';
  el.style.animationDelay = `0s, ${durationMs - 300}ms`;
  container.appendChild(el);
  setTimeout(() => el.remove(), durationMs + 100);
}

// ── Loading overlay ───────────────────────────────────────

function showLoading(msg) {
  if (loadingText) loadingText.textContent = msg || 'Loading VibeBoy...';
  if (loadingError) loadingError.textContent = '';
  if (loadingOverlay) loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
  if (loadingOverlay) loadingOverlay.classList.add('hidden');
}

function showLoadingError(msg) {
  if (loadingError) loadingError.textContent = msg;
}

// ── Lazy wasm loading ─────────────────────────────────────

let wasmReady = false;

async function ensureWasm() {
  if (wasmReady) return;
  showLoading('Loading VibeBoy...');
  try {
    const mod = await import('./pkg/vibeboy.js');
    wasmMod = mod.default;       // init()
    WasmEmulator = mod.WasmEmulator;
    wasm_memory = mod.wasm_memory;
    filter_registry_json = mod.filter_registry_json;
    btn_a = mod.btn_a;
    btn_b = mod.btn_b;
    btn_up = mod.btn_up;
    btn_down = mod.btn_down;
    btn_left = mod.btn_left;
    btn_right = mod.btn_right;
    btn_start = mod.btn_start;
    btn_select = mod.btn_select;

    await wasmMod();
    wasmReady = true;

    // Build key map
    state.keyMap = {
      'ArrowUp':    btn_up(),
      'ArrowDown':  btn_down(),
      'ArrowLeft':  btn_left(),
      'ArrowRight': btn_right(),
      'KeyZ':       btn_b(),
      'KeyX':       btn_a(),
      'Enter':      btn_start(),
      'ShiftLeft':  btn_select(),
      'ShiftRight': btn_select(),
    };

    // Populate filter dropdown
    populateFilterDropdown();

    // Set up touch controls now that btn constants are available
    setupTouchControls(state, {
      btn_a: btn_a(),
      btn_b: btn_b(),
      btn_up: btn_up(),
      btn_down: btn_down(),
      btn_left: btn_left(),
      btn_right: btn_right(),
      btn_start: btn_start(),
      btn_select: btn_select(),
    });

    hideLoading();
  } catch (e) {
    console.error('Wasm init failed:', e);
    showLoadingError('Failed to load: ' + e.message);
    throw e;
  }
}

// ── Filter dropdown ───────────────────────────────────────

function populateFilterDropdown() {
  const filters = JSON.parse(filter_registry_json());
  const groupLabels = { hqx: 'HQx', xbr: 'xBR', xbrz: 'xBRZ', edge_detect: 'Edge Detect' };
  const groups = {};
  for (const f of filters) {
    const opt = document.createElement('option');
    opt.value = f.cli_name;
    opt.textContent = f.display_name;
    if (f.cli_name === 'vectorize') opt.selected = true;
    if (f.group === 'main') {
      filterSelect.appendChild(opt);
    } else {
      if (!groups[f.group]) {
        groups[f.group] = document.createElement('optgroup');
        groups[f.group].label = groupLabels[f.group] || f.group;
      }
      groups[f.group].appendChild(opt);
    }
  }
  for (const g of Object.values(groups)) filterSelect.appendChild(g);
}

// ── Audio ─────────────────────────────────────────────────

async function initAudio() {
  if (state.audioCtx) {
    try { state.audioCtx.close(); } catch (_) {}
    state.audioCtx = null;
    state.audioWorklet = null;
  }
  try {
    state.audioCtx = new AudioContext({ sampleRate: 96000, latencyHint: 'interactive' });
    await state.audioCtx.audioWorklet.addModule('./audio-processor.js');
    state.audioWorklet = new AudioWorkletNode(state.audioCtx, 'emu-audio', {
      outputChannelCount: [2],
    });
    state.audioWorklet.port.onmessage = (e) => {
      if (e.data && typeof e.data.buffered === 'number') {
        state.audioBuffered = e.data.buffered;
        state._audioReportReceived = true;
      }
    };
    state.audioWorklet.connect(state.audioCtx.destination);
    console.log('AudioWorklet initialized at', state.audioCtx.sampleRate, 'Hz');
  } catch (e) {
    console.warn('AudioWorklet init failed:', e);
  }
}

function pushAudioFromBuf() {
  if (!state.emu || !state.audioWorklet) return;
  const ptr = state.emu.audio_ptr();
  const len = state.emu.audio_len();
  if (len === 0) return;
  const samples = new Float32Array(wasm_memory().buffer, ptr, len);
  const copy = new Float32Array(len);
  copy.set(samples);
  state.audioWorklet.port.postMessage(copy, [copy.buffer]);
}

function pushAudioSamples() {
  if (!state.emu) return;
  state.emu.audio_drain();
  pushAudioFromBuf();
}

// ── Camera ────────────────────────────────────────────────

function stopCamera() {
  if (state.cameraActive && camVideo.srcObject) {
    camVideo.srcObject.getTracks().forEach(t => t.stop());
    camVideo.srcObject = null;
    state.cameraActive = false;
  }
}

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 128 }, height: { ideal: 112 }, facingMode: 'user' }
    });
    camVideo.srcObject = stream;
    state.cameraActive = true;
    console.log('Webcam active for Pocket Camera');
  } catch (e) {
    console.warn('Webcam unavailable:', e);
  }
}

function feedCameraFrame() {
  if (!state.cameraActive || !state.emu || camVideo.readyState < 2) return;
  camCtx.save();
  camCtx.translate(128, 0);
  camCtx.scale(-1, 1);
  camCtx.drawImage(camVideo, 0, 0, 128, 112);
  camCtx.restore();
  const rgba = camCtx.getImageData(0, 0, 128, 112).data;
  for (let i = 0; i < 128 * 112; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    camGrayscale[i] = (r * 77 + g * 150 + b * 29) >> 8;
  }
  state.emu.set_camera_image(camGrayscale);
}

// ── Accelerometer ─────────────────────────────────────────

function stopAccelerometer() {
  if (state.deviceMotionHandler) {
    window.removeEventListener('devicemotion', state.deviceMotionHandler);
    state.deviceMotionHandler = null;
    state.accelActive = false;
  }
}

function enableDeviceMotion() {
  if (state.deviceMotionHandler) return;
  state.deviceMotionHandler = e => {
    if (e.accelerationIncludingGravity) {
      state.accelX = (e.accelerationIncludingGravity.x || 0) / 9.81;
      state.accelY = (e.accelerationIncludingGravity.y || 0) / 9.81;
      state.accelActive = true;
    }
  };
  window.addEventListener('devicemotion', state.deviceMotionHandler);
  console.log('DeviceMotion accelerometer active for MBC7');
}

function initAccelerometer() {
  if (!state.emu || !state.emu.has_accelerometer()) return;
  if (state.deviceMotionHandler) return;
  if (window.DeviceMotionEvent) {
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
      DeviceMotionEvent.requestPermission().then(s => {
        if (s === 'granted') enableDeviceMotion();
      }).catch(() => {});
    } else {
      enableDeviceMotion();
    }
  }
}

function feedAccelerometer() {
  if (!state.emu || !state.accelActive) return;
  state.emu.set_accelerometer(state.accelX, state.accelY);
}

// ── Printer ───────────────────────────────────────────────

function checkForPrint() {
  if (!state.emu || !state.emu.has_print()) return;
  const rgba = state.emu.take_print_rgba();
  const w = state.emu.print_width();
  const h = state.emu.print_height();
  if (w === 0 || h === 0) return;

  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const cx = c.getContext('2d');
  const img = new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, rgba.byteLength), w, h);
  cx.putImageData(img, 0, 0);
  c.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `print_${String(++state.printCount).padStart(4, '0')}.png`;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

// ── Rumble ─────────────────────────────────────────────────

function updateRumble() {
  if (!state.emu || !state.emu.has_rumble()) return;
  const on = state.emu.rumble_active();
  if (on === state.rumbleWasOn) return;
  state.rumbleWasOn = on;
  const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
  for (const gp of gamepads) {
    if (!gp || !gp.vibrationActuator) continue;
    if (on) {
      gp.vibrationActuator.playEffect('dual-rumble', {
        duration: 200,
        strongMagnitude: 0.6,
        weakMagnitude: 0.4,
      }).catch(() => {});
    } else {
      gp.vibrationActuator.reset().catch(() => {});
    }
    break;
  }
}

// ── Save states ───────────────────────────────────────────

function saveStateKey(slot) {
  if (!state.emu) return null;
  return 'vibeboy_ss_' + state.emu.save_key() + '_' + slot;
}

function saveState() {
  if (!state.emu) return;
  const data = state.emu.save_state_to_bytes(state.currentSlot);
  if (data && data.length > 0) {
    const key = saveStateKey(state.currentSlot);
    let bin = '';
    for (let i = 0; i < data.length; i++) bin += String.fromCharCode(data[i]);
    try {
      localStorage.setItem(key, btoa(bin));
      showToast('State saved to slot ' + state.currentSlot);
    } catch (e) {
      showToast('Save failed: ' + e.message, 3000);
    }
  }
}

function loadState() {
  if (!state.emu) return;
  const key = saveStateKey(state.currentSlot);
  const saved = localStorage.getItem(key);
  if (saved) {
    const bytes = Uint8Array.from(atob(saved), c => c.charCodeAt(0));
    if (state.emu.load_state_from_bytes(state.currentSlot, bytes)) {
      showToast('State loaded from slot ' + state.currentSlot);
    } else {
      showToast('Failed to load state (incompatible version?)', 3000);
    }
  } else {
    showToast('Slot ' + state.currentSlot + ' is empty');
  }
}

// ── Save flushing via requestIdleCallback ─────────────────

function scheduleSaveFlush() {
  if (state.saveFlushScheduled) return;
  state.saveFlushScheduled = true;
  const cb = () => {
    state.saveFlushScheduled = false;
    if (!state.emu || !state.emu.has_battery()) return;
    const data = state.emu.save_data();
    let bin = '';
    for (let i = 0; i < data.length; i++) bin += String.fromCharCode(data[i]);
    localStorage.setItem(state.emu.save_key(), btoa(bin));
  };
  if (typeof requestIdleCallback === 'function') {
    requestIdleCallback(cb, { timeout: 2000 });
  } else {
    setTimeout(cb, 1000);
  }
}

let saveFrameCounter = 0;

// ── Render mode ───────────────────────────────────────────

function updateRenderMode() {
  const forceCpu = forceCpuCheckbox.checked;
  if (forceCpu) {
    state.useGpu = false;
    canvas.style.display = 'none';
    canvas2d.style.display = '';
    statusEl.textContent = 'Rendering: CPU (Canvas2D)';
  } else if (state.gpuAvailable) {
    state.useGpu = true;
    canvas.style.display = '';
    canvas2d.style.display = 'none';
    const sel = filterSelect.options[filterSelect.selectedIndex];
    statusEl.textContent = 'Rendering: ' + (sel ? sel.text : 'GPU');
  }
}

// ── Gamepad ───────────────────────────────────────────────

function pollGamepad() {
  if (!state.emu) return;
  const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
  let gpRewind = false, gpFastFwd = false;

  const gpBtnMap = [
    [0, btn_b()],
    [1, btn_a()],
    [8, btn_select()],
    [9, btn_start()],
    [12, btn_up()],
    [13, btn_down()],
    [14, btn_left()],
    [15, btn_right()],
  ];

  for (const gp of gamepads) {
    if (!gp) continue;

    if (gp.buttons.length > 5) {
      gpRewind = gp.buttons[4].pressed;
      gpFastFwd = gp.buttons[5].pressed;
    }

    for (const [gi, emuBtn] of gpBtnMap) {
      if (gi < gp.buttons.length) {
        const pressed = gp.buttons[gi].pressed;
        const key = gp.index + '_' + gi;
        if (state.gpPrevState[key] !== pressed) {
          state.emu.set_button(emuBtn, pressed);
          state.gpPrevState[key] = pressed;
        }
      }
    }

    if (gp.axes.length >= 2) {
      const lx = gp.axes[0];
      const ly = gp.axes[1];
      const stickLeft = lx < -STICK_DEADZONE;
      const stickRight = lx > STICK_DEADZONE;
      const stickUp = ly < -STICK_DEADZONE;
      const stickDown = ly > STICK_DEADZONE;

      const sk = gp.index + '_s';
      const prev = state.gpPrevState[sk] || {};
      if (prev.l !== stickLeft)  state.emu.set_button(btn_left(), stickLeft);
      if (prev.r !== stickRight) state.emu.set_button(btn_right(), stickRight);
      if (prev.u !== stickUp)    state.emu.set_button(btn_up(), stickUp);
      if (prev.d !== stickDown)  state.emu.set_button(btn_down(), stickDown);
      state.gpPrevState[sk] = { l: stickLeft, r: stickRight, u: stickUp, d: stickDown };
    }

    break;
  }

  if (gpRewind && !state.rewindHeld) { state.rewindHeld = true; state.emu.set_rewinding(true); }
  if (!gpRewind && state.rewindHeld && !state.kbRewindHeld) { state.rewindHeld = false; state.emu.set_rewinding(false); }
  if (gpFastFwd) state.fastForwardHeld = true;
  if (!gpFastFwd && !state.kbFastFwdHeld) state.fastForwardHeld = false;
}

// ── Frame loop ────────────────────────────────────────────

function frame(timestamp) {
  if (!state.emu) return;
  if (state.lastTimestamp === 0) state.lastTimestamp = timestamp;

  state.emuTime += timestamp - state.lastTimestamp;
  state.lastTimestamp = timestamp;

  if (state.emuTime > GB_FRAME_MS * 3) state.emuTime = GB_FRAME_MS * 3;

  pollGamepad();
  feedAccelerometer();

  let stepped = false;

  if (state.paused && !state.stepOneFrame) {
    state.emuTime = 0;
  } else if (state.paused && state.stepOneFrame) {
    if (state.cameraActive) feedCameraFrame();
    state.emu.step_frame();
    if (state.audioWorklet) pushAudioSamples();
    state.stepOneFrame = false;
    state.emuTime = 0;
    stepped = true;
    checkForPrint();
    updateRumble();
  } else if (state.rewindHeld) {
    for (let i = 0; i < 3; i++) state.emu.rewind_one_frame();
    if (state.audioWorklet) {
      state.emu.audio_drain();
      state.emu.audio_reverse();
      state.emu.audio_downsample(3);
      pushAudioFromBuf();
    }
    state.emuTime = 0;
    stepped = true;
  } else if (state.fastForwardHeld) {
    for (let i = 0; i < 4; i++) {
      if (state.cameraActive) feedCameraFrame();
      state.emu.step_frame();
    }
    if (state.audioWorklet) {
      state.emu.audio_drain();
      state.emu.audio_downsample(4);
      pushAudioFromBuf();
    }
    state.emuTime = 0;
    stepped = true;
    checkForPrint();
    updateRumble();
  } else {
    // Audio-driven timing gated by wall clock: only step when enough
    // time has elapsed (handles >60Hz displays), then use the AudioWorklet
    // buffer level to fine-tune how many frames to step.
    let framesNeeded = 0;
    if (state.emuTime >= GB_FRAME_MS) {
      // Time gate passed — at least one frame is due
      if (state.audioWorklet && state._audioReportReceived) {
        const samplesPerFrame = 96000 / 60 * 2;
        const targetFill = samplesPerFrame * 3;  // ~50ms
        const maxFill = samplesPerFrame * 8;     // ~133ms
        const queued = state.audioBuffered;

        if (queued < targetFill / 2) framesNeeded = 2;
        else if (queued < targetFill) framesNeeded = 1;
        else if (queued > maxFill) framesNeeded = 0;
        else framesNeeded = 1;
      } else {
        framesNeeded = 1;
      }
      state.emuTime -= GB_FRAME_MS;
      // Consume any extra accumulated time (don't let it pile up)
      if (state.emuTime > GB_FRAME_MS) state.emuTime = GB_FRAME_MS;
    }

    for (let i = 0; i < framesNeeded; i++) {
      if (state.cameraActive) feedCameraFrame();
      state.emu.step_frame();
      if (state.audioWorklet) pushAudioSamples();
      stepped = true;
      checkForPrint();
      updateRumble();
      saveFrameCounter++;
      if (saveFrameCounter >= 60) {
        saveFrameCounter = 0;
        scheduleSaveFlush();
      }
    }
  }

  // Resize GPU surface to match wrapper
  if (state.useGpu) {
    const wrapper = document.getElementById('canvas-wrapper');
    const cw = wrapper.clientWidth;
    const ch = wrapper.clientHeight;
    const srcW = state.emu.width();
    const srcH = state.emu.height();
    let tw, th;
    if (cw * srcH <= ch * srcW) {
      tw = cw;
      th = Math.round(cw * srcH / srcW);
    } else {
      th = ch;
      tw = Math.round(ch * srcW / srcH);
    }
    if (tw > 0 && th > 0 && (canvas.width !== tw || canvas.height !== th)) {
      canvas.width = tw;
      canvas.height = th;
      state.emu.resize_gpu(tw, th);
    }
  }

  if (stepped) {
    if (state.useGpu && state.emu.render_gpu()) {
      // GPU rendered
    } else if (state.ctx) {
      const cw = canvas2d.parentElement.clientWidth;
      const ch = canvas2d.parentElement.clientHeight;
      const dims = state.emu.cpu_scaled_frame_update(cw, ch);
      const sw = dims[0];
      const sh = dims[1];
      const ptr = state.emu.frame_buffer_ptr();
      const len = state.emu.frame_buffer_len();
      if (canvas2d.width !== sw || canvas2d.height !== sh) {
        canvas2d.width = sw;
        canvas2d.height = sh;
      }
      const rgba = new Uint8ClampedArray(wasm_memory().buffer, ptr, len);
      const img = new ImageData(rgba, sw, sh);
      state.ctx.putImageData(img, 0, 0);
    }
  }
  state.animFrameId = requestAnimationFrame(frame);
}

// ── Start emulator ────────────────────────────────────────

async function startEmulator(romBytes) {
  await ensureWasm();

  if (state.animFrameId !== null) {
    cancelAnimationFrame(state.animFrameId);
    state.animFrameId = null;
  }
  if (state.emu) {
    state.emu.free();
    state.emu = null;
  }
  state.lastTimestamp = 0;
  state.emuTime = 0;
  state.paused = false;
  state.stepOneFrame = false;
  state.rewindHeld = false;
  state.fastForwardHeld = false;
  state.kbRewindHeld = false;
  state.kbFastFwdHeld = false;

  // Hide ROM prompt, show we have a ROM loaded
  if (romPrompt) romPrompt.classList.add('hidden');
  dropZone.classList.add('has-rom');

  state.emu = new WasmEmulator(new Uint8Array(romBytes));
  if (skipBootCheckbox.checked) {
    state.emu.set_skip_boot(true);
    state.emu.set_model(modelSelect.value);
  }
  const w = state.emu.width();
  const h = state.emu.height();

  await initAudio();

  // Try WebGPU
  if (navigator.gpu) {
    try {
      canvas.width = w * 4;
      canvas.height = h * 4;
      await state.emu.init_gpu(canvas);
      state.useGpu = true;
      state.gpuAvailable = true;
      statusEl.textContent = 'Rendering: WebGPU vectorize';
      canvas2d.width = w;
      canvas2d.height = h;
      state.ctx = canvas2d.getContext('2d');
    } catch (e) {
      console.warn('WebGPU init failed, falling back to Canvas2D:', e);
      state.useGpu = false;
    }
  }

  if (!state.useGpu) {
    canvas.width = w;
    canvas.height = h;
    state.ctx = canvas.getContext('2d');
    statusEl.textContent = 'Rendering: Canvas2D';
  }

  // Load save from localStorage
  if (state.emu.has_battery()) {
    const key = state.emu.save_key();
    const saved = localStorage.getItem(key);
    if (saved) {
      const bytes = Uint8Array.from(atob(saved), c => c.charCodeAt(0));
      state.emu.load_save(bytes);
      console.log('Loaded save from localStorage:', key);
    }
  }

  stopCamera();
  stopAccelerometer();

  state.emu.attach_printer();
  if (state.emu.has_camera()) initCamera();
  initAccelerometer();

  if (state.gpuAvailable) filterBar.style.display = '';
  if (forceCpuCheckbox.checked) updateRenderMode();
  state.animFrameId = requestAnimationFrame(frame);
}

// ── Keyboard input ────────────────────────────────────────

function setupKeyboard() {
  document.addEventListener('keydown', e => {
    if (!state.emu) return;
    if (e.code >= 'Digit0' && e.code <= 'Digit9') {
      state.currentSlot = parseInt(e.code.charAt(5));
      showToast('Slot ' + state.currentSlot + ' selected');
      e.preventDefault(); return;
    }
    if (e.code === 'Space') {
      state.paused = !state.paused;
      showToast(state.paused ? 'Paused' : 'Resumed');
      e.preventDefault(); return;
    }
    if (e.code === 'Period') {
      if (state.paused) state.stepOneFrame = true;
      e.preventDefault(); return;
    }
    if (e.code === 'F5') { saveState(); e.preventDefault(); return; }
    if (e.code === 'F7') { loadState(); e.preventDefault(); return; }
    if (e.code === 'Backspace') {
      state.kbRewindHeld = true; state.rewindHeld = true;
      state.emu.set_rewinding(true); e.preventDefault(); return;
    }
    if (e.code === 'Tab') {
      state.kbFastFwdHeld = true; state.fastForwardHeld = true;
      e.preventDefault(); return;
    }
    const btn = state.keyMap[e.code];
    if (btn !== undefined) { state.emu.set_button(btn, true); e.preventDefault(); }
  });

  document.addEventListener('keyup', e => {
    if (e.code === 'Backspace') {
      state.kbRewindHeld = false; state.rewindHeld = false;
      if (state.emu) state.emu.set_rewinding(false);
      e.preventDefault(); return;
    }
    if (e.code === 'Tab') {
      state.kbFastFwdHeld = false; state.fastForwardHeld = false;
      e.preventDefault(); return;
    }
    const btn = state.keyMap[e.code];
    if (btn !== undefined && state.emu) { state.emu.set_button(btn, false); e.preventDefault(); }
  });
}

// ── UI event wiring ───────────────────────────────────────

function setupUI() {
  // File loading
  function loadFile(file) {
    const reader = new FileReader();
    reader.onload = () => startEmulator(reader.result);
    reader.readAsArrayBuffer(file);
  }

  const openRomBtn = document.getElementById('open-rom-btn');
  if (openRomBtn) openRomBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
  dropZone.addEventListener('click', () => { if (!state.emu) fileInput.click(); });
  dropZone.addEventListener('dblclick', () => fileInput.click());
  fileInput.addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('hover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('hover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('hover');
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });

  // Filter change
  filterSelect.addEventListener('change', () => {
    if (!state.emu) return;
    state.emu.set_filter(filterSelect.value);
    updateRenderMode();
    showToast('Filter: ' + filterSelect.options[filterSelect.selectedIndex].text);
  });

  // Force CPU toggle
  forceCpuCheckbox.addEventListener('change', () => updateRenderMode());

  // Skip boot toggle — applies on next model change or ROM load
  skipBootCheckbox.addEventListener('change', () => {
    if (!state.emu) return;
    state.emu.set_skip_boot(skipBootCheckbox.checked);
  });

  // Model change
  modelSelect.addEventListener('change', () => {
    if (!state.emu) return;
    state.emu.set_skip_boot(skipBootCheckbox.checked);
    state.emu.set_model(modelSelect.value);
    if (state.emu.has_battery()) {
      const saved = localStorage.getItem(state.emu.save_key());
      if (saved) {
        const bytes = Uint8Array.from(atob(saved), c => c.charCodeAt(0));
        state.emu.load_save(bytes);
      }
    }
    state.lastTimestamp = 0;
    state.emuTime = 0;
    if (state.useGpu) {
      canvas.width = 0;
      canvas.height = 0;
    }
    showToast('Model: ' + modelSelect.options[modelSelect.selectedIndex].text);
  });

  // Fullscreen
  document.getElementById('fullscreen-btn').addEventListener('click', () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      document.documentElement.requestFullscreen().catch(() => {});
    }
  });
}

// ── Initialization ────────────────────────────────────────

// Expose startEmulator for the ROM selector inline script
window._startEmulator = (buf) => startEmulator(buf);

setupKeyboard();
setupUI();

// If a ROM was selected before this module loaded, start it now
if (window._pendingRom) {
  startEmulator(window._pendingRom);
  window._pendingRom = null;
}
