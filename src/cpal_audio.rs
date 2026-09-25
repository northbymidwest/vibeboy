//! cpal audio output shared by the winit and GTK frontends: the emulator
//! pushes into a lock-free ring (see [`crate::ui_util::audio_ring`]) that the
//! cpal stream callback drains, with half-band FIR downsampling when the
//! device cannot run at the APU rate.

use crate::ui_util::{AudioProducer, audio_ring, audio_ring_capacity};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// 31-tap half-band low-pass FIR for 2:1 downsampling (96kHz -> 48kHz).
/// Blackman-windowed sinc, cutoff at Nyquist/2 (24kHz), normalized to unit DC gain.
/// Symmetric with zero-valued odd taps (half-band property).
const HALFBAND_FIR: [f32; 31] = [
    0.0000000000,
    0.0000000000,
    0.0004103229,
    0.0000000000,
    -0.0022302855,
    0.0000000000,
    0.0071008571,
    0.0000000000,
    -0.017_917_03,
    0.0000000000,
    0.040_107_418,
    0.0000000000,
    -0.090_106_92,
    0.0000000000,
    0.312_633_34,
    0.500_004_65,
    0.312_633_34,
    0.0000000000,
    -0.090_106_92,
    0.0000000000,
    0.040_107_418,
    0.0000000000,
    -0.017_917_03,
    0.0000000000,
    0.0071008571,
    0.0000000000,
    -0.0022302855,
    0.0000000000,
    0.0004103229,
    0.0000000000,
    0.0000000000,
];

/// Anti-aliased integer-ratio downsampler for interleaved stereo.
struct Downsampler {
    /// Ratio of APU sample rate to stream sample rate (e.g. 2 for 96k->48k).
    ratio: usize,
    /// Per-channel FIR filter history.
    hist_l: [f32; HALFBAND_FIR.len()],
    hist_r: [f32; HALFBAND_FIR.len()],
    pos: usize,
    /// Input frames since the last output frame.
    phase: usize,
    out: Vec<f32>,
}

impl Downsampler {
    fn new(ratio: usize) -> Self {
        Self {
            ratio,
            hist_l: [0.0; HALFBAND_FIR.len()],
            hist_r: [0.0; HALFBAND_FIR.len()],
            pos: 0,
            phase: 0,
            out: Vec::new(),
        }
    }

    /// Feed every stereo frame into the FIR history and emit one filtered
    /// frame for every `ratio` input frames.
    fn process(&mut self, samples: &[f32]) -> &[f32] {
        let fir_len = HALFBAND_FIR.len();
        self.out.clear();
        for frame in samples.as_chunks::<2>().0 {
            self.hist_l[self.pos] = frame[0];
            self.hist_r[self.pos] = frame[1];
            self.pos = (self.pos + 1) % fir_len;
            self.phase += 1;
            if self.phase == self.ratio {
                self.phase = 0;
                let mut l = 0.0f32;
                let mut r = 0.0f32;
                for (k, &tap) in HALFBAND_FIR.iter().enumerate() {
                    let idx = (self.pos + k) % fir_len;
                    l += self.hist_l[idx] * tap;
                    r += self.hist_r[idx] * tap;
                }
                self.out.push(l);
                self.out.push(r);
            }
        }
        &self.out
    }
}

/// A playing cpal output stream and the emulation side of its ring.
pub struct CpalAudio {
    producer: AudioProducer,
    /// `None` when the device runs at the APU rate.
    downsampler: Option<Downsampler>,
    _stream: cpal::Stream,
}

impl CpalAudio {
    /// Open the default output device, preferring `source_rate` (the APU
    /// rate) and falling back to 48kHz. Returns `None` if no device works.
    pub fn start(source_rate: u32) -> Option<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;

        let err_fn = |err: cpal::Error| eprintln!("Audio error: {err}");

        // Try 96kHz with fixed buffer, then default buffer, then 48kHz.
        // 1024 frames (~10.7ms at 96kHz): 512 underran continuously through ALSA's
        // PipeWire plugin, while much larger buffers drain in bursts that upset the
        // session's fill-level frame pacing.
        let configs = [
            cpal::StreamConfig {
                channels: 2,
                sample_rate: source_rate,
                buffer_size: cpal::BufferSize::Fixed(1024),
            },
            cpal::StreamConfig {
                channels: 2,
                sample_rate: source_rate,
                buffer_size: cpal::BufferSize::Default,
            },
            cpal::StreamConfig {
                channels: 2,
                sample_rate: 48_000,
                buffer_size: cpal::BufferSize::Default,
            },
        ];

        for config in &configs {
            // Each attempt gets its own ring: a failed build drops the
            // callback, and the consumer with it.
            let (producer, mut consumer) = audio_ring(audio_ring_capacity(config.sample_rate));
            let result = device.build_output_stream(
                *config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| consumer.fill(data),
                err_fn,
                None,
            );
            match result {
                Ok(stream) => {
                    if stream.play().is_ok() {
                        eprintln!(
                            "Audio: {}Hz {}ch buf={:?}",
                            config.sample_rate, config.channels, config.buffer_size
                        );
                        let ratio = (source_rate / config.sample_rate).max(1) as usize;
                        return Some(Self {
                            producer,
                            downsampler: (ratio > 1).then(|| Downsampler::new(ratio)),
                            _stream: stream,
                        });
                    }
                }
                Err(e) => eprintln!(
                    "Audio: {}Hz {:?} failed: {e}",
                    config.sample_rate, config.buffer_size
                ),
            }
        }
        eprintln!("Audio: all configurations failed");
        None
    }

    /// Queue interleaved stereo samples at the APU rate.
    pub fn push(&mut self, samples: &[f32]) {
        match self.downsampler {
            Some(ref mut ds) => self.producer.push(ds.process(samples)),
            None => self.producer.push(samples),
        }
    }

    /// Stereo frames queued ahead of the device, in APU-rate frames.
    pub fn queued_frames(&self) -> usize {
        let ratio = self.downsampler.as_ref().map_or(1, |ds| ds.ratio);
        self.producer.queued() * ratio
    }
}
