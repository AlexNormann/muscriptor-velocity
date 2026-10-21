"""Estimate per-note MIDI velocity from the audio at each note's onset.

The MT3 tokenizer the model was trained with only predicts note on/off — its
velocity event is a binary 0/1 gate — so the MIDI was always written with
one fixed velocity (default 100) for every note. This module derives a per-note
velocity from the audio itself: the strength of the signal in the note's own
pitch band, measured over a short window at the note's onset, mapped onto
MIDI's 1..127 velocity range. It is a first dynamic pass — polyphonic
cross-talk and harmonics are ignored, but the relative loudness of each note is
preserved as it was in the recording.
"""

import math

import numpy as np

# --- measurement parameters ---
ONSET_WINDOW_S = 0.03  # audio examined at/after each onset (30 ms attack)
N_FFT = 2048  # FFT size for the short-time frame
# --- mapping ---
# The loudest note is written at this velocity; any quieter note is attenuated
# at this many velocity units per dB.
REFERENCE_VELOCITY = 115.0
VELOCITY_PER_DB = 70.0 / 30.0  # -30 dB -> 70 velocity steps quieter
MIN_VELOCITY = 1
MAX_VELOCITY = 127


def _midi_pitch_to_hz(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def _note_frequency_power(wav, sr, onset, pitch):
    """Spectral power of wav (mono, float) in pitch's band at onset."""
    start = int(round(onset * sr))
    window = int(round(ONSET_WINDOW_S * sr))
    seg = wav[start : start + window]
    if seg.shape[0] < 2:
        return 0.0
    if seg.shape[0] < window:
        seg = np.concatenate([seg, np.zeros(window - seg.shape[0])])
    seg = seg * np.hanning(window)
    spectrum = np.fft.rfft(seg, n=N_FFT)
    power = np.abs(spectrum) ** 2
    f0 = _midi_pitch_to_hz(pitch)
    bin_width = sr / N_FFT
    centre = int(round(f0 / bin_width))
    lo = max(0, centre - 1)
    hi = min(len(power), centre + 2)
    if hi <= lo:
        return 0.0
    return float(power[lo:hi].sum())


def estimate_note_velocities(wav, sample_rate, notes):
    """Return a velocity (1..127) for every note in notes."""
    if hasattr(wav, "detach"):  # torch tensor (possibly on GPU) -> CPU numpy
        arr = wav.detach().cpu().numpy()
    else:
        arr = np.asarray(wav)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    elif arr.ndim > 2:
        arr = arr.reshape(-1)
    powers = [_note_frequency_power(arr, sample_rate, n.onset, n.pitch) for n in notes]
    reference = max(powers) if powers else 0.0
    if reference <= 0.0:
        return [int(REFERENCE_VELOCITY) for _ in notes]
    velocities = []
    for p in powers:
        rel_db = 10.0 * math.log10((p + 1e-12) / reference)
        vel = int(round(REFERENCE_VELOCITY + rel_db * VELOCITY_PER_DB))
        velocities.append(max(MIN_VELOCITY, min(MAX_VELOCITY, vel)))
    return velocities


def apply_note_velocities(wav, sample_rate, notes):
    """Set each note's velocity from the audio, in place."""
    velocities = estimate_note_velocities(wav, sample_rate, notes)
    for note, vel in zip(notes, velocities):
        note.velocity = vel
