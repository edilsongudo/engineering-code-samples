from dataclasses import dataclass
import unittest
import numpy as np

from generate_automix import (
    SAMPLE_RATE,
    build_optimal_harmonic_chain,
    calculate_harmonic_compatibility_score,
    parse_camelot_key,
    render_dj_echo_out_transition,
    snap_to_phrase_grid_samples,
    studio_smooth_dj_transition,
)


@dataclass
class MockTrack:
    id: int
    title: str
    musical_key: str
    bpm: float
    energy_score: float = 0.7
    valence: float = 0.6
    acousticness: float = 0.1


class TestAutomixEngine(unittest.TestCase):
    def test_parse_camelot_key(self) -> None:
        self.assertEqual(parse_camelot_key("8A"), (8, "A"))
        self.assertEqual(parse_camelot_key("12b"), (12, "B"))
        self.assertEqual(parse_camelot_key("1A"), (1, "A"))
        self.assertEqual(parse_camelot_key(None), (8, "A"))
        self.assertEqual(parse_camelot_key("invalid"), (8, "A"))

    def test_camelot_harmonic_scoring_circular_wheel_and_tempo(self) -> None:
        base = MockTrack(id=1, title="Base", musical_key="12A", bpm=124.0, energy_score=0.70)
        same_key = MockTrack(id=2, title="Same", musical_key="12A", bpm=124.0, energy_score=0.75)
        wrap_adjacent = MockTrack(id=3, title="Adjacent 1A", musical_key="1A", bpm=124.0, energy_score=0.75)
        relative_major = MockTrack(id=4, title="Relative Major", musical_key="12B", bpm=124.0, energy_score=0.75)
        clashing = MockTrack(id=5, title="Clash", musical_key="6B", bpm=140.0, energy_score=0.40)

        s_same = calculate_harmonic_compatibility_score(base, same_key)
        s_rel = calculate_harmonic_compatibility_score(base, relative_major)
        s_adj = calculate_harmonic_compatibility_score(base, wrap_adjacent)
        s_clash = calculate_harmonic_compatibility_score(base, clashing)

        # Same key > relative major > circular adjacent (12A -> 1A) > distant key/tempo clash
        self.assertGreater(s_same, s_rel)
        self.assertGreater(s_rel, s_adj)
        self.assertGreater(s_adj, s_clash)

    def test_build_optimal_harmonic_chain_orders_smooth_progression(self) -> None:
        tracks = [
            MockTrack(id=1, title="Peak 126 9A", musical_key="9A", bpm=126.0, energy_score=0.85),
            MockTrack(id=2, title="Opener 122 8A", musical_key="8A", bpm=122.0, energy_score=0.60),
            MockTrack(id=3, title="Outlier 128 2B", musical_key="2B", bpm=128.0, energy_score=0.50),
            MockTrack(id=4, title="Warmup 123 8A", musical_key="8A", bpm=123.0, energy_score=0.70),
        ]
        chain = build_optimal_harmonic_chain(tracks)
        ordered_ids = [t.id for t in chain]
        # Starts with lowest BPM/energy opener (id=2, 122 BPM 8A), then transitions to 8A 123 BPM (id=4), then 9A 126 BPM (id=1)
        self.assertEqual(ordered_ids[:3], [2, 4, 1])

    def test_snap_to_phrase_grid_samples(self) -> None:
        bpm = 120.0  # 1 beat = 0.5s, 4-bar phrase (16 beats) = 8.0s
        beatgrid = [i * 0.5 for i in range(64)]  # 0.0s .. 31.5s

        # 7.6 seconds should snap to the 8.0s phrase downbeat (beat 16)
        raw_samples = int(7.6 * SAMPLE_RATE)
        snapped = snap_to_phrase_grid_samples(raw_samples, beatgrid, bpm, phrase_bars=4, sr=SAMPLE_RATE)
        self.assertEqual(snapped, int(8.0 * SAMPLE_RATE))

        # Fallback without beatgrid
        snapped_fallback = snap_to_phrase_grid_samples(raw_samples, None, bpm, phrase_bars=4, sr=SAMPLE_RATE)
        self.assertEqual(snapped_fallback, int(8.0 * SAMPLE_RATE))

    def test_studio_smooth_dj_transition_shape_and_peak_limiting(self) -> None:
        # 2 seconds of stereo audio at 44.1 kHz
        n_samples = SAMPLE_RATE * 2
        t = np.linspace(0, 2.0, n_samples, dtype=np.float32)

        # Outgoing: 60 Hz sub-bass + 440 Hz mid tone at high amplitude (0.95)
        out_ch = 0.55 * np.sin(2 * np.pi * 60.0 * t) + 0.40 * np.sin(2 * np.pi * 440.0 * t)
        outgoing = np.vstack([out_ch, out_ch]).astype(np.float32)

        # Incoming: 80 Hz sub-bass + 880 Hz mid-high tone at high amplitude (0.95)
        in_ch = 0.55 * np.sin(2 * np.pi * 80.0 * t) + 0.40 * np.sin(2 * np.pi * 880.0 * t)
        incoming = np.vstack([in_ch, in_ch]).astype(np.float32)

        blended = studio_smooth_dj_transition(outgoing, incoming, sr=SAMPLE_RATE)
        self.assertEqual(blended.shape, (2, n_samples))
        self.assertEqual(blended.dtype, np.float32)
        # Soft-knee tanh limiting prevents digital clipping above 1.0
        self.assertLessEqual(float(np.max(np.abs(blended))), 1.0)

    def test_render_dj_echo_out_transition(self) -> None:
        n_samples = SAMPLE_RATE * 3
        t = np.linspace(0, 3.0, n_samples, dtype=np.float32)
        wave = (0.7 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
        outgoing = np.vstack([wave, wave])
        incoming = np.vstack([wave * 0.8, wave * 0.8])

        transition = render_dj_echo_out_transition(outgoing, incoming, bpm_prev=120.0, sr=SAMPLE_RATE)
        self.assertEqual(transition.shape, (2, n_samples))
        self.assertEqual(transition.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(transition))), 1.0)
        # First sample starts at 0.0 due to 15ms micro-fade on incoming head (after cut) or clean sum
        self.assertTrue(np.all(np.isfinite(transition)))


if __name__ == "__main__":
    unittest.main()
