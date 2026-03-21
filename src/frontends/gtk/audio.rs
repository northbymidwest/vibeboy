use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

use super::AUDIO_SAMPLE_RATE;

/// 31-tap half-band low-pass FIR for 2:1 downsampling (96kHz -> 48kHz).
const HALFBAND_FIR: [f32; 31] = [
     0.0000000000,  0.0000000000,  0.0004103229,  0.0000000000,
    -0.0022302855,  0.0000000000,  0.0071008571,  0.0000000000,
    -0.0179170304,  0.0000000000,  0.0401074177,  0.0000000000,
    -0.0901069221,  0.0000000000,  0.3126333216,  0.5000046375,
     0.3126333216,  0.0000000000, -0.0901069221,  0.0000000000,
     0.0401074177,  0.0000000000, -0.0179170304,  0.0000000000,
     0.0071008571,  0.0000000000, -0.0022302855,  0.0000000000,
     0.0004103229,  0.0000000000,  0.0000000000,
];

pub(super) struct AudioRing {
    buf: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    capacity: usize,
    pub downsample_ratio: usize,
    fir_hist_l: Vec<f32>,
    fir_hist_r: Vec<f32>,
    fir_pos: usize,
}

impl AudioRing {
    pub fn new(capacity: usize, stream_rate: u32) -> Self {
        let ratio = (AUDIO_SAMPLE_RATE / stream_rate) as usize;
        AudioRing {
            buf: vec![0.0; capacity],
            write_pos: 0,
            read_pos: 0,
            capacity,
            downsample_ratio: ratio.max(1),
            fir_hist_l: vec![0.0; HALFBAND_FIR.len()],
            fir_hist_r: vec![0.0; HALFBAND_FIR.len()],
            fir_pos: 0,
        }
    }

    fn len(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }

    fn push_one(&mut self, s: f32) {
        let next = (self.write_pos + 1) % self.capacity;
        if next == self.read_pos {
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
        self.buf[self.write_pos] = s;
        self.write_pos = next;
    }

    pub fn push(&mut self, samples: &[f32]) {
        if self.len() > self.capacity / 2 {
            self.read_pos = self.write_pos;
        }
        if self.downsample_ratio <= 1 {
            for &s in samples {
                self.push_one(s);
            }
        } else {
            let r = self.downsample_ratio;
            let fir_len = HALFBAND_FIR.len();
            let pair_count = samples.len() / 2;
            for i in 0..pair_count {
                self.fir_hist_l[self.fir_pos] = samples[i * 2];
                self.fir_hist_r[self.fir_pos] = samples[i * 2 + 1];
                self.fir_pos = (self.fir_pos + 1) % fir_len;

                if (i + 1) % r == 0 {
                    let mut l = 0.0f32;
                    let mut rv = 0.0f32;
                    for k in 0..fir_len {
                        let idx = (self.fir_pos + k) % fir_len;
                        l += self.fir_hist_l[idx] * HALFBAND_FIR[k];
                        rv += self.fir_hist_r[idx] * HALFBAND_FIR[k];
                    }
                    self.push_one(l);
                    self.push_one(rv);
                }
            }
        }
    }

    pub fn pop(&mut self) -> f32 {
        if self.read_pos == self.write_pos {
            return 0.0;
        }
        let val = self.buf[self.read_pos];
        self.read_pos = (self.read_pos + 1) % self.capacity;
        val
    }
}

pub(super) fn start_audio(ring: Arc<Mutex<AudioRing>>) -> Option<(cpal::Stream, u32)> {
    let host = cpal::default_host();
    let device = host.default_output_device()?;

    let err_fn = |err: cpal::StreamError| eprintln!("Audio error: {err}");

    let configs = [
        cpal::StreamConfig {
            channels: 2,
            sample_rate: AUDIO_SAMPLE_RATE,
            buffer_size: cpal::BufferSize::Fixed(512),
        },
        cpal::StreamConfig {
            channels: 2,
            sample_rate: AUDIO_SAMPLE_RATE,
            buffer_size: cpal::BufferSize::Default,
        },
        cpal::StreamConfig {
            channels: 2,
            sample_rate: 48_000,
            buffer_size: cpal::BufferSize::Default,
        },
    ];

    for config in &configs {
        let ring = Arc::clone(&ring);
        let result = device.build_output_stream(
            config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut ring = ring.lock().unwrap();
                for sample in data.iter_mut() {
                    *sample = ring.pop();
                }
            },
            err_fn,
            None,
        );
        match result {
            Ok(stream) => {
                if stream.play().is_ok() {
                    eprintln!("Audio: {}Hz {}ch buf={:?}",
                        config.sample_rate, config.channels, config.buffer_size);
                    return Some((stream, config.sample_rate));
                }
            }
            Err(e) => eprintln!("Audio: {}Hz {:?} failed: {e}",
                config.sample_rate, config.buffer_size),
        }
    }
    eprintln!("Audio: all configurations failed");
    None
}
