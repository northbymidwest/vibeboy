// VibeBoy — AudioWorklet processor
// Loaded via audioWorklet.addModule()

class EmuAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(0);
    this.pos = 0;
    this.port.onmessage = (e) => {
      const incoming = e.data;
      const remaining = this.buf.length - this.pos;
      const total = remaining + incoming.length;
      // Cap buffer at ~4 frames to limit latency (~25600 stereo floats at 96kHz)
      const MAX_BUF = 25600;
      if (total > MAX_BUF) {
        // Drop oldest samples to stay within cap
        const keep = MAX_BUF - incoming.length;
        const start = keep > 0 ? this.buf.length - keep : this.buf.length;
        const newBuf = new Float32Array(Math.min(total, MAX_BUF));
        if (keep > 0) newBuf.set(this.buf.subarray(start));
        newBuf.set(incoming, Math.max(0, keep));
        this.buf = newBuf;
      } else {
        const newBuf = new Float32Array(total);
        newBuf.set(this.buf.subarray(this.pos));
        newBuf.set(incoming, remaining);
        this.buf = newBuf;
      }
      this.pos = 0;
    };
  }

  process(inputs, outputs) {
    const outL = outputs[0][0];
    const outR = outputs[0][1];
    if (!outL) return true;
    for (let i = 0; i < outL.length; i++) {
      if (this.pos < this.buf.length - 1) {
        outL[i] = this.buf[this.pos];
        outR[i] = this.buf[this.pos + 1];
        this.pos += 2;
      } else {
        outL[i] = 0;
        outR[i] = 0;
      }
    }
    // Compact buffer when mostly consumed
    if (this.pos > 16384) {
      this.buf = this.buf.slice(this.pos);
      this.pos = 0;
    }
    return true;
  }
}

registerProcessor('emu-audio', EmuAudioProcessor);
