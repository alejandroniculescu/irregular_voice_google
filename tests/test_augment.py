import numpy as np

from irregular_voice_google.augment import augment, lowpass, speed
from irregular_voice_google.preprocess import SAMPLE_RATE


def _tone(freq: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_speed_changes_length():
    assert len(speed(_tone(440), 1.1)) == round(SAMPLE_RATE / 1.1)


def test_lowpass_keeps_low_and_cuts_high():
    assert np.std(lowpass(_tone(300), 2000)) > 0.9 * np.std(_tone(300))
    assert np.std(lowpass(_tone(6000), 1000)) < 0.01


def test_augment_is_bounded_float32_and_seeded():
    a = augment(_tone(440), np.random.default_rng(0), p=1.0)
    b = augment(_tone(440), np.random.default_rng(0), p=1.0)
    assert a.dtype == np.float32 and np.abs(a).max() <= 1 and np.array_equal(a, b)
    assert augment(_tone(440), np.random.default_rng(0), p=0.0) is not None
