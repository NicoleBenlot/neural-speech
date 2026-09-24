import numpy as np

from src.inference.mic import EnergyVAD


def test_energy_vad_splits_synthetic_utterances():
    sample_rate = 16000
    t = np.arange(int(0.25 * sample_rate), dtype=np.float32) / sample_rate
    speech = 0.25 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    silence = np.zeros(int(0.8 * sample_rate), dtype=np.float32)
    audio = np.concatenate(
        [
            np.zeros(int(0.2 * sample_rate), dtype=np.float32),
            speech,
            silence,
            speech,
            silence,
        ]
    )

    vad = EnergyVAD(samplerate=sample_rate)
    utterances = vad.feed(audio)
    utterances.extend(vad.flush())

    assert len(utterances) == 2
    assert all(segment.size >= int(0.2 * sample_rate) for segment in utterances)
    assert all(np.max(np.abs(segment)) > 0.2 for segment in utterances)
