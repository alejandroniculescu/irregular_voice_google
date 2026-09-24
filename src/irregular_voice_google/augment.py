"""Training-time audio augmentation (numpy only, so it runs on a GPU box without ffmpeg).

With a few hundred utterances the adapter memorises the exact recordings
within a few epochs. Each augmentation is applied independently with
probability ``p``, so every epoch sees a slightly different copy of the clip:

- speed      0.9–1.1× (tempo and pitch, as in Kaldi speed perturbation)
- muffle     low-pass at 1–4 kHz (his voice is muffled; how much varies by day)
- reverb     synthetic room, RT60 0.15–0.6 s (kitchen, bathroom, car)
- noise      white noise at 10–30 dB SNR
- gain       ±6 dB

Applied only to training batches, never to dev/test or live audio.
"""

from __future__ import annotations

import numpy as np

from irregular_voice_google.preprocess import SAMPLE_RATE


def speed(audio: np.ndarray, factor: float) -> np.ndarray:
    n = max(1, round(len(audio) / factor))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def lowpass(audio: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    """Butterworth-shaped magnitude response, applied in the frequency domain."""
    freqs = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)
    gain = 1 / np.sqrt(1 + (freqs / cutoff) ** (2 * order))
    return np.fft.irfft(np.fft.rfft(audio) * gain, len(audio)).astype(np.float32)


def reverb(audio: np.ndarray, rt60: float, wet: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(rt60 * SAMPLE_RATE)) / SAMPLE_RATE
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir**2))
    n = len(audio) + len(rir) - 1
    tail = np.fft.irfft(np.fft.rfft(audio, n) * np.fft.rfft(rir, n), n)[: len(audio)]
    return _match_rms((1 - wet) * audio + wet * tail, audio)


def noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2)) + 1e-9
    return (audio + rng.standard_normal(len(audio)) * rms / 10 ** (snr_db / 20)).astype(np.float32)


def _match_rms(audio: np.ndarray, like: np.ndarray) -> np.ndarray:
    return (audio * (np.sqrt(np.mean(like**2)) / (np.sqrt(np.mean(audio**2)) + 1e-9))).astype(np.float32)


def augment(audio: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    if len(audio) == 0 or p <= 0:
        return audio
    if rng.random() < p:
        audio = speed(audio, rng.uniform(0.9, 1.1))
    if rng.random() < p:
        audio = _match_rms(lowpass(audio, rng.uniform(1000, 4000)), audio)
    if rng.random() < p:
        audio = reverb(audio, rng.uniform(0.15, 0.6), rng.uniform(0.2, 0.6), rng)
    if rng.random() < p:
        audio = noise(audio, rng.uniform(10, 30), rng)
    if rng.random() < p:
        audio = audio * 10 ** (rng.uniform(-6, 6) / 20)
    return np.clip(audio, -1, 1).astype(np.float32)
