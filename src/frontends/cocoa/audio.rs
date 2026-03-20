use std::sync::{Arc, Mutex};

use super::AUDIO_SAMPLE_RATE;

// ── CoreAudio FFI ────────────────────────────────────────────────────────────

pub(super) mod core_audio {
    use std::os::raw::c_void;

    pub type OSStatus = i32;
    pub type AudioUnit = *mut c_void;
    pub type AudioComponent = *mut c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct AudioComponentDescription {
        pub component_type: u32,
        pub component_sub_type: u32,
        pub component_manufacturer: u32,
        pub component_flags: u32,
        pub component_flags_mask: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct AudioStreamBasicDescription {
        pub sample_rate: f64,
        pub format_id: u32,
        pub format_flags: u32,
        pub bytes_per_packet: u32,
        pub frames_per_packet: u32,
        pub bytes_per_frame: u32,
        pub channels_per_frame: u32,
        pub bits_per_channel: u32,
        pub reserved: u32,
    }

    #[repr(C)]
    pub struct AudioBufferList {
        pub number_buffers: u32,
        pub buffers: [AudioBuffer; 1],
    }

    #[repr(C)]
    pub struct AudioBuffer {
        pub number_channels: u32,
        pub data_byte_size: u32,
        pub data: *mut c_void,
    }

    #[repr(C)]
    pub struct AudioTimeStamp {
        pub sample_time: f64,
        pub host_time: u64,
        pub rate_scalar: f64,
        pub word_clock_time: u64,
        pub smpte_time: [u8; 24],
        pub flags: u32,
        pub reserved: u32,
    }

    pub type AURenderCallback = unsafe extern "C" fn(
        in_ref_con: *mut c_void,
        io_action_flags: *mut u32,
        in_time_stamp: *const AudioTimeStamp,
        in_bus_number: u32,
        in_number_frames: u32,
        io_data: *mut AudioBufferList,
    ) -> OSStatus;

    #[repr(C)]
    pub struct AURenderCallbackStruct {
        pub input_proc: AURenderCallback,
        pub input_proc_ref_con: *mut c_void,
    }

    pub const K_AUDIO_UNIT_TYPE_OUTPUT: u32 = u32::from_be_bytes(*b"auou");
    pub const K_AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT: u32 = u32::from_be_bytes(*b"def ");
    pub const K_AUDIO_UNIT_MANUFACTURER_APPLE: u32 = u32::from_be_bytes(*b"appl");
    pub const K_AUDIO_FORMAT_LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
    pub const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1;
    pub const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 8;
    pub const K_AUDIO_UNIT_SCOPE_INPUT: u32 = 1;
    pub const K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT: u32 = 8;
    pub const K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: u32 = 23;

    #[link(name = "AudioToolbox", kind = "framework")]
    unsafe extern "C" {
        pub fn AudioComponentFindNext(
            component: AudioComponent,
            desc: *const AudioComponentDescription,
        ) -> AudioComponent;
        pub fn AudioComponentInstanceNew(
            component: AudioComponent,
            out: *mut AudioUnit,
        ) -> OSStatus;
        pub fn AudioUnitSetProperty(
            unit: AudioUnit,
            property_id: u32,
            scope: u32,
            element: u32,
            data: *const c_void,
            data_size: u32,
        ) -> OSStatus;
        pub fn AudioUnitInitialize(unit: AudioUnit) -> OSStatus;
        pub fn AudioOutputUnitStart(unit: AudioUnit) -> OSStatus;
        pub fn AudioOutputUnitStop(unit: AudioUnit) -> OSStatus;
        pub fn AudioComponentInstanceDispose(unit: AudioUnit) -> OSStatus;
    }
}

// ── Audio ring buffer ────────────────────────────────────────────────────────

pub(super) struct AudioRingBuffer {
    buffer: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    capacity: usize,
}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        AudioRingBuffer {
            buffer: vec![0.0; capacity],
            write_pos: 0,
            read_pos: 0,
            capacity,
        }
    }

    fn len(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }

    pub fn write(&mut self, data: &[f32]) {
        // If buffer is more than half full, skip ahead to stay low-latency
        if self.len() > self.capacity / 2 {
            self.read_pos = self.write_pos;
        }
        for &sample in data {
            let next = (self.write_pos + 1) % self.capacity;
            if next == self.read_pos {
                self.read_pos = (self.read_pos + 1) % self.capacity;
            }
            self.buffer[self.write_pos] = sample;
            self.write_pos = next;
        }
    }

    pub fn read(&mut self, out: &mut [f32]) {
        for sample in out.iter_mut() {
            if self.read_pos == self.write_pos {
                *sample = 0.0;
            } else {
                *sample = self.buffer[self.read_pos];
                self.read_pos = (self.read_pos + 1) % self.capacity;
            }
        }
    }
}

pub(super) type SharedAudioBuffer = Arc<Mutex<AudioRingBuffer>>;

pub(super) unsafe extern "C" fn audio_render_callback(
    in_ref_con: *mut std::os::raw::c_void,
    _io_action_flags: *mut u32,
    _in_time_stamp: *const core_audio::AudioTimeStamp,
    _in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut core_audio::AudioBufferList,
) -> i32 {
    unsafe {
        let ring = &*(in_ref_con as *const Mutex<AudioRingBuffer>);
        let buf_list = &mut *io_data;
        let ab = &mut buf_list.buffers[0];
        let out_ptr = ab.data as *mut f32;
        let sample_count = in_number_frames as usize * 2;
        let out_slice = std::slice::from_raw_parts_mut(out_ptr, sample_count);

        if let Ok(mut guard) = ring.lock() {
            guard.read(out_slice);
        } else {
            out_slice.fill(0.0);
        }
        0
    }
}

pub(super) fn setup_audio(ring_buffer: &SharedAudioBuffer) -> Option<core_audio::AudioUnit> {
    unsafe {
        let desc = core_audio::AudioComponentDescription {
            component_type: core_audio::K_AUDIO_UNIT_TYPE_OUTPUT,
            component_sub_type: core_audio::K_AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT,
            component_manufacturer: core_audio::K_AUDIO_UNIT_MANUFACTURER_APPLE,
            component_flags: 0,
            component_flags_mask: 0,
        };

        let comp = core_audio::AudioComponentFindNext(std::ptr::null_mut(), &desc);
        if comp.is_null() {
            eprintln!("Failed to find audio output component");
            return None;
        }

        let mut audio_unit: core_audio::AudioUnit = std::ptr::null_mut();
        if core_audio::AudioComponentInstanceNew(comp, &mut audio_unit) != 0 {
            eprintln!("Failed to create audio unit");
            return None;
        }

        let stream_desc = core_audio::AudioStreamBasicDescription {
            sample_rate: AUDIO_SAMPLE_RATE as f64,
            format_id: core_audio::K_AUDIO_FORMAT_LINEAR_PCM,
            format_flags: core_audio::K_AUDIO_FORMAT_FLAG_IS_FLOAT
                | core_audio::K_AUDIO_FORMAT_FLAG_IS_PACKED,
            bytes_per_packet: 8,
            frames_per_packet: 1,
            bytes_per_frame: 8,
            channels_per_frame: 2,
            bits_per_channel: 32,
            reserved: 0,
        };

        if core_audio::AudioUnitSetProperty(
            audio_unit,
            core_audio::K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
            core_audio::K_AUDIO_UNIT_SCOPE_INPUT,
            0,
            &stream_desc as *const _ as *const _,
            std::mem::size_of::<core_audio::AudioStreamBasicDescription>() as u32,
        ) != 0 {
            eprintln!("Failed to set audio stream format");
            core_audio::AudioComponentInstanceDispose(audio_unit);
            return None;
        }

        let callback_struct = core_audio::AURenderCallbackStruct {
            input_proc: audio_render_callback,
            input_proc_ref_con: Arc::as_ptr(ring_buffer) as *mut _,
        };

        if core_audio::AudioUnitSetProperty(
            audio_unit,
            core_audio::K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK,
            core_audio::K_AUDIO_UNIT_SCOPE_INPUT,
            0,
            &callback_struct as *const _ as *const _,
            std::mem::size_of::<core_audio::AURenderCallbackStruct>() as u32,
        ) != 0 {
            eprintln!("Failed to set audio render callback");
            core_audio::AudioComponentInstanceDispose(audio_unit);
            return None;
        }

        if core_audio::AudioUnitInitialize(audio_unit) != 0 {
            eprintln!("Failed to initialize audio unit");
            core_audio::AudioComponentInstanceDispose(audio_unit);
            return None;
        }

        if core_audio::AudioOutputUnitStart(audio_unit) != 0 {
            eprintln!("Failed to start audio unit");
            core_audio::AudioComponentInstanceDispose(audio_unit);
            return None;
        }

        Some(audio_unit)
    }
}
