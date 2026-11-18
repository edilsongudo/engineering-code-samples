# Automated Audio Mixing (`generate_automix.py`)

## The Problem

We needed a backend command to turn a list of uploaded tracks into a continuous DJ mix. Doing that reliably required solving three things:

1. **Track ordering:** Sorting only by BPM often pairs tracks in clashing musical keys or causes abrupt drops in energy and mood.
2. **Transitions:** Standard volume crossfades sound muddy when two basslines and kick drums overlap, and tempo mismatches cause beats to drift apart.
3. **Memory usage:** Building an hour-long stereo Float32 mix as a single NumPy array in memory before encoding takes over 1 GB of RAM per worker.

## The Solution

`generate_automix.py` orders tracks using a greedy compatibility score across five musical features, renders transitions in Float32 stereo using frequency crossover filters (or an echo-out cut when tempos are far apart), and pipes each finished audio chunk directly to an `ffmpeg` subprocess.

## How It Works

- **Track sequencing (`calculate_harmonic_compatibility_score`, `build_optimal_harmonic_chain`):** Scores track pairs using circular distance on the Camelot Wheel (`1A`–`12B`), BPM difference, energy progression, valence, and acousticness.
- **Beat alignment & phrase snapping (`align_track_phase`, `snap_to_phrase_grid_samples`):** Bandpass-filters the first 6 seconds (`40–220 Hz`) to find the first kick drum onset (ignoring soft vocal/pad intros) and aligns the track head. Outro transition points are snapped to 4-bar or 8-bar phrase boundaries using the track's beatgrid.
- **Transitions (`studio_smooth_dj_transition`, `render_dj_echo_out_transition`):**
  - For tempo differences within `0.4–8.5 BPM`, only the outgoing tail is time-stretched (`dj_time_stretch` via Rubber Band or Librosa) so the incoming track enters at its native tempo. Both tracks are split into Low (`< 220 Hz`), Mid (`220–3200 Hz`), and High (`> 3200 Hz`) bands with Butterworth filters so outgoing bass fades out before incoming bass drops in.
  - If `|ΔBPM| > 8.5`, it cuts the outgoing track on the beat and generates a 4-tap decaying echo high-passed at `350 Hz`.
- **FFmpeg pipe (`Command.handle`):** As each track body and transition zone is rendered, raw `f32le` bytes are piped to `ffmpeg`'s `stdin` and freed.

## Tests & Limitations

```bash
python3 -m unittest discover -s tests -v
```

**Production vs. Sample Changes:**
- Made Django imports optional so the sequencing and DSP functions can be imported and unit-tested without a running Django setup.
- Removed inline S3 upload and HLS task-queueing boilerplate from the end of `Command.handle`.

**Limitations:**
- The sequencing and DSP functions run standalone (tested with synthetic stereo waveforms in `tests/test_automix_engine.py`), while `Command.handle` is a Django management command that expects `core.models.Audio` records pre-populated by `audio_analyzer.py` and an `ffmpeg` binary on `PATH`.
