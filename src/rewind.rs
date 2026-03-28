//! Delta-compressed rewind buffer.
//!
//! Stores emulator snapshots as a base keyframe (the current state) plus
//! reverse deltas going backward. Each reverse delta records the bytes that
//! changed when going from frame N-1 to frame N, storing the OLD values —
//! so applying a reverse delta to frame N's bytes recovers frame N-1.
//!
//! This means `pop()` is O(delta_size) — just apply one reverse delta.
//! No keyframe walking, no forward reconstruction.
//!
//! Delta format: packed runs of (offset: u32, length: u16, old_bytes...).

use crate::snapshot::Snapshot;
use std::collections::VecDeque;

/// A reverse delta: applying it to the next frame's bytes recovers this frame.
/// Stores the old bytes at each changed offset.
struct ReverseDelta(Vec<u8>);

/// Delta-compressed rewind buffer.
pub struct RewindBuffer {
    /// Reverse deltas, oldest at front. Each delta, applied to the state
    /// after it, recovers the state at this entry's point in time.
    deltas: VecDeque<ReverseDelta>,
    /// Serialized bytes of the most recent (tail) snapshot.
    /// This is the "current" state that deltas are relative to.
    head: Vec<u8>,
    /// Maximum number of rewind frames to keep.
    capacity: usize,
}

impl RewindBuffer {
    pub fn new(capacity: usize) -> Self {
        RewindBuffer {
            deltas: VecDeque::with_capacity(capacity),
            head: Vec::new(),
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        if self.head.is_empty() { 0 } else { self.deltas.len() + 1 }
    }

    pub fn clear(&mut self) {
        self.deltas.clear();
        self.head.clear();
    }

    /// Push a snapshot into the rewind buffer.
    pub fn push(&mut self, snap: &Snapshot) {
        let serialized = bincode::serde::encode_to_vec(snap, bincode::config::standard())
            .expect("snapshot serialization failed");

        if self.head.is_empty() {
            // First frame — just store as the head
            self.head = serialized;
            return;
        }

        // Compute reverse delta: what bytes in `head` (old) differ from
        // `serialized` (new). Store the OLD values so we can undo later.
        let reverse = encode_reverse_delta(&self.head, &serialized);

        // Evict oldest if at capacity
        if self.deltas.len() >= self.capacity {
            self.deltas.pop_front();
        }

        self.deltas.push_back(ReverseDelta(reverse));
        self.head = serialized;
    }

    /// Pop the most recent snapshot and return it.
    pub fn pop(&mut self) -> Option<Snapshot> {
        if self.head.is_empty() {
            return None;
        }

        // Deserialize the current head
        let bytes = std::mem::take(&mut self.head);

        // Apply the most recent reverse delta to recover the previous frame
        if let Some(ReverseDelta(delta)) = self.deltas.pop_back() {
            self.head = bytes.clone();
            apply_delta(&mut self.head, &delta);
        }
        // else: this was the last frame, head stays empty

        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .map(|(snap, _)| snap)
            .ok()
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.deltas.iter().map(|d| d.0.len()).sum::<usize>() + self.head.len()
    }
}

/// Encode a reverse delta: for each byte that differs between `old` and `new`,
/// store the OLD value at that offset. Applying this delta to `new` recovers `old`.
///
/// Format: sequence of (offset: u32, length: u16, old_bytes...) runs.
/// Runs merge gaps <= 8 bytes to reduce header overhead.
fn encode_reverse_delta(old: &[u8], new: &[u8]) -> Vec<u8> {
    let len = old.len().min(new.len());
    let mut result = Vec::new();

    let mut i = 0;
    while i < len {
        if old[i] == new[i] {
            i += 1;
            continue;
        }

        let run_start = i;
        i += 1;

        while i < len {
            if old[i] != new[i] {
                i += 1;
            } else {
                let gap_start = i;
                while i < len && i - gap_start < 8 && old[i] == new[i] {
                    i += 1;
                }
                if i < len && old[i] != new[i] {
                    i += 1;
                } else {
                    i = gap_start;
                    break;
                }
            }
        }

        let run_end = i;
        let mut chunk_start = run_start;
        while chunk_start < run_end {
            let chunk_len = (run_end - chunk_start).min(0xFFFF);
            result.extend_from_slice(&(chunk_start as u32).to_le_bytes());
            result.extend_from_slice(&(chunk_len as u16).to_le_bytes());
            // Store OLD bytes (not new) — this is what makes it a reverse delta
            result.extend_from_slice(&old[chunk_start..chunk_start + chunk_len]);
            chunk_start += chunk_len;
        }
    }

    // If old is longer than new, store the tail from old
    if old.len() > new.len() {
        let mut chunk_start = new.len();
        while chunk_start < old.len() {
            let chunk_len = (old.len() - chunk_start).min(0xFFFF);
            result.extend_from_slice(&(chunk_start as u32).to_le_bytes());
            result.extend_from_slice(&(chunk_len as u16).to_le_bytes());
            result.extend_from_slice(&old[chunk_start..chunk_start + chunk_len]);
            chunk_start += chunk_len;
        }
    }

    result
}

/// Apply a delta to a byte buffer in-place.
fn apply_delta(base: &mut Vec<u8>, delta: &[u8]) {
    let mut pos = 0;
    while pos + 6 <= delta.len() {
        let offset = u32::from_le_bytes([delta[pos], delta[pos+1], delta[pos+2], delta[pos+3]]) as usize;
        let length = u16::from_le_bytes([delta[pos+4], delta[pos+5]]) as usize;
        pos += 6;

        if pos + length > delta.len() {
            break;
        }

        if offset + length > base.len() {
            base.resize(offset + length, 0);
        }

        base[offset..offset + length].copy_from_slice(&delta[pos..pos + length]);
        pos += length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_delta_roundtrip() {
        let old = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let new = vec![1, 2, 99, 4, 5, 6, 7, 8, 9, 10];
        let delta = encode_reverse_delta(&old, &new);
        let mut reconstructed = new.clone();
        apply_delta(&mut reconstructed, &delta);
        assert_eq!(reconstructed, old);
    }

    #[test]
    fn test_reverse_delta_no_changes() {
        let data = vec![1, 2, 3, 4, 5];
        let delta = encode_reverse_delta(&data, &data);
        assert!(delta.is_empty());
    }

    #[test]
    fn test_reverse_delta_all_changed() {
        let old = vec![0u8; 100];
        let new = vec![1u8; 100];
        let delta = encode_reverse_delta(&old, &new);
        let mut reconstructed = new.clone();
        apply_delta(&mut reconstructed, &delta);
        assert_eq!(reconstructed, old);
    }
}
