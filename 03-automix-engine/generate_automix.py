"""
Automated DJ Audio Mixing & 5D Harmonic Sequencing Engine.

Implements:
  1. 5D Harmonic & Acoustic Track Sequencing (Camelot wheel circular distance,
     BPM similarity, energy progression, mood/valence, and acoustic texture).
  2. Float32 Stereo DSP Processing (LUFS loudness normalization, sub-bass kick
     phase alignment, Rubber Band / Phase-Vocoder time-stretching, 3-band
     Butterworth crossover transitions, and rhythmic echo-out drops).
  3. Direct Stream-to-Pipe FFmpeg Encoding (streams Float32 PCM chunks directly
     to an `ffmpeg` stdin pipe to keep memory consumption bounded during long mixes).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import gc
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

import librosa
import numpy as np
from pydub import AudioSegment
import requests
from scipy import signal

try:
    import pyrubberband as pyrb

    RUBBERBAND_AVAILABLE = True
except Exception:
    pyrb = None  # type: ignore[assignment]
    RUBBERBAND_AVAILABLE = False

try:
    from django.conf import settings as django_settings
    from django.contrib.auth import get_user_model
    from django.core.management.base import BaseCommand, CommandError
except ImportError:
    django_settings = None  # type: ignore[assignment]

    def get_user_model() -> Any:  # type: ignore[misc]
        return None

    class CommandError(Exception):  # type: ignore[no-redef]
        pass

    class BaseCommand:  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)
SAMPLE_RATE = 44100


# ── 1. 5D Harmonic & Acoustic Playlist Sequencing ──


def parse_camelot_key(key_str: Optional[str]) -> Tuple[int, str]:
    """Parses a Camelot key string (e.g. '8A', '11B') into `(number: int [1..12], mode: 'A'|'B')`."""
    if not key_str:
        return 8, "A"
    s = str(key_str).strip().upper()
    digits = "".join(filter(str.isdigit, s))
    num = max(1, min(12, int(digits) if digits else 8))
    mode = "B" if "B" in s else "A"
    return num, mode


def calculate_harmonic_compatibility_score(audio_a: Any, audio_b: Any) -> float:
    """
    Calculates a 5D transition compatibility score between two tracks based on:
      1. Camelot Wheel circular distance (0..6 steps) & Major/Minor mode relationship
      2. BPM delta penalty
      3. Energy progression bonus (maintaining or building energy)
      4. Valence (mood) continuity
      5. Acousticness (timbral texture) continuity
    """
    key_a_num, key_a_mode = parse_camelot_key(getattr(audio_a, "musical_key", None))
    key_b_num, key_b_mode = parse_camelot_key(getattr(audio_b, "musical_key", None))

    raw_diff = abs(key_a_num - key_b_num)
    hour_diff = min(raw_diff, 12 - raw_diff)
    same_mode = key_a_mode == key_b_mode

    if hour_diff == 0 and same_mode:
        harmonic_score = 100.0
    elif hour_diff == 0 and not same_mode:
        harmonic_score = 92.0
    elif hour_diff == 1 and same_mode:
        harmonic_score = 88.0
    elif hour_diff == 1 and not same_mode:
        harmonic_score = 75.0
    elif hour_diff == 2 and same_mode:
        harmonic_score = 60.0
    else:
        harmonic_score = max(10.0, 45.0 - (hour_diff * 8.0))

    bpm_a = float(audio_a.bpm) if getattr(audio_a, "bpm", None) else 120.0
    bpm_b = float(audio_b.bpm) if getattr(audio_b, "bpm", None) else 120.0
    bpm_penalty = abs(bpm_a - bpm_b) * 2.0

    energy_a = float(audio_a.energy_score) if getattr(audio_a, "energy_score", None) is not None else 0.5
    energy_b = float(audio_b.energy_score) if getattr(audio_b, "energy_score", None) is not None else 0.5
    energy_bonus = 8.0 if (energy_b >= energy_a - 0.05) else 0.0

    val_a = float(audio_a.valence) if getattr(audio_a, "valence", None) is not None else 0.5
    val_b = float(audio_b.valence) if getattr(audio_b, "valence", None) is not None else 0.5
    valence_penalty = abs(val_a - val_b) * 12.0

    ac_a = float(audio_a.acousticness) if getattr(audio_a, "acousticness", None) is not None else 0.2
    ac_b = float(audio_b.acousticness) if getattr(audio_b, "acousticness", None) is not None else 0.2
    acoustic_penalty = abs(ac_a - ac_b) * 10.0

    return max(0.0, harmonic_score - bpm_penalty + energy_bonus - valence_penalty - acoustic_penalty)


def build_optimal_harmonic_chain(audios: Sequence[Any]) -> List[Any]:
    """Constructs an ordered playlist sequence from candidate tracks using greedy 5D compatibility pathfinding."""
    if len(audios) <= 1:
        return list(audios)

    remaining = list(audios)
    remaining.sort(
        key=lambda a: (
            float(a.bpm) if getattr(a, "bpm", None) else 120.0,
            float(a.energy_score) if getattr(a, "energy_score", None) is not None else 0.5,
        )
    )
    chain = [remaining.pop(0)]

    while remaining:
        current = chain[-1]
        best_idx, best_score = 0, -1.0
        for idx, cand in enumerate(remaining):
            score = calculate_harmonic_compatibility_score(current, cand)
            if score > best_score:
                best_score = score
                best_idx = idx
        chain.append(remaining.pop(best_idx))

    return chain


# ── 2. Studio Native DSP Audio Processing (Float32) ──


def load_and_normalize_track(
    file_path: str,
    track_lufs: Optional[float] = None,
    target_lufs: float = -14.0,
    trim: bool = True,
) -> np.ndarray:
    """
    Decodes an audio file into a contiguous Float32 stereo NumPy array `(2, N)` at 44.1 kHz.
    Applies calibrated LUFS loudness gain (clamped to +/- 6 dB), trims leading silence,
    and normalizes peaks above 0.98.
    """
    seg = AudioSegment.from_file(file_path).set_frame_rate(SAMPLE_RATE).set_channels(2)
    raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    y = np.ascontiguousarray(raw.reshape((-1, 2)).T)
    del seg, raw

    if trim and y.shape[1] > SAMPLE_RATE * 4:
        head_check = y[:, : int(SAMPLE_RATE * 4.0)]
        energy = np.mean(np.abs(head_check), axis=0)
        non_silent = np.where(energy > 0.005)[0]
        if len(non_silent) > 0 and non_silent[0] > 0:
            y = y[:, non_silent[0] :]

    if track_lufs is not None:
        gain_db = float(np.clip(target_lufs - float(track_lufs), -6.0, 6.0))
        gain_linear = 10.0 ** (gain_db / 20.0)
        y = y * gain_linear

    max_peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    if max_peak > 0.98:
        y = y * (0.98 / max_peak)

    return y.astype(np.float32)


def dj_time_stretch(y: np.ndarray, source_bpm: float, target_bpm: float) -> np.ndarray:
    """
    Time-stretches a stereo audio buffer `(2, N)` from `source_bpm` to `target_bpm`
    without shifting musical pitch (Master Tempo).
    Uses the Rubber Band C++ engine when installed, falling back to Librosa's phase vocoder.
    """
    if source_bpm <= 0 or target_bpm <= 0 or abs(source_bpm - target_bpm) < 0.3:
        return y

    rate = float(np.clip(target_bpm / source_bpm, 0.85, 1.15))

    if RUBBERBAND_AVAILABLE and pyrb is not None:
        try:
            stretched = pyrb.time_stretch(y.T, SAMPLE_RATE, rate).T
            return stretched.astype(np.float32)
        except Exception as exc:
            logger.debug("Rubber Band stretch failed, falling back to phase vocoder: %s", exc)

    try:
        stretched_channels = [
            librosa.effects.time_stretch(np.ascontiguousarray(y[ch]), rate=rate)
            for ch in range(y.shape[0])
        ]
        min_len = min(len(s) for s in stretched_channels)
        return np.vstack([s[:min_len] for s in stretched_channels]).astype(np.float32)
    except Exception as exc:
        logger.warning("Phase-vocoder time stretch failed: %s", exc)
        return y


def align_track_phase(track_audio: np.ndarray, bpm: float) -> np.ndarray:
    """
    Shifts a stereo track array to lock its beat phase on the first physical kick downbeat.
    Isolates sub-bass (40 Hz - 220 Hz) via a 2nd-order Butterworth bandpass filter so soft vocal
    or pad intros do not trigger false downbeat cuts.
    """
    if bpm <= 0 or track_audio.shape[1] == 0:
        return track_audio

    try:
        beat_interval_samples = int((60.0 / bpm) * SAMPLE_RATE)
        if beat_interval_samples <= 0:
            return track_audio

        probe_len = min(track_audio.shape[1], int(SAMPLE_RATE * 6.0))
        probe_chunk = track_audio[0, :probe_len]

        sos_kick = signal.butter(2, [40.0, 220.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
        probe_kick = signal.sosfiltfilt(sos_kick, probe_chunk)

        if np.max(np.abs(probe_kick)) > 0.03:
            onset_samples = librosa.onset.onset_detect(y=probe_kick, sr=SAMPLE_RATE, units="samples")
            if len(onset_samples) > 0:
                first_kick = int(onset_samples[0])
                phase_offset = first_kick % beat_interval_samples
                if 0 < phase_offset < beat_interval_samples:
                    return track_audio[:, phase_offset:]
    except Exception as exc:
        logger.debug("Beat phase alignment skipped: %s", exc)
    return track_audio


def studio_smooth_dj_transition(
    outgoing_tail: np.ndarray,
    incoming_head: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    3-Way Multi-Band DJ Transition with Analog Mixer Crossover Curves:
      - Low-Band (< 220 Hz): Steep sub-bass handover so two kick drums never clash in the low end.
      - Mid-Band (220 Hz - 3200 Hz): Proactive vocal/harmonic ducking on outgoing track.
      - High-Band (> 3200 Hz): Airy trigonometric crossfade.
      - Soft-knee `tanh` limiting above 0.98 to prevent clipping while preserving boundary continuity.
    """
    num_samples = min(outgoing_tail.shape[1], incoming_head.shape[1])
    if num_samples <= 0:
        return np.hstack([outgoing_tail, incoming_head])

    out_crop = outgoing_tail[:, :num_samples]
    in_crop = incoming_head[:, :num_samples]

    t = np.linspace(0, np.pi / 2, num_samples, dtype=np.float32)
    fade_out = (np.maximum(0.0, np.cos(t)) ** 1.6).astype(np.float32)
    fade_in = (np.maximum(0.0, np.sin(t)) ** 2.0).astype(np.float32)

    try:
        sos_lp = signal.butter(2, 220.0, btype="lowpass", fs=sr, output="sos")
        sos_bp = signal.butter(2, [220.0, 3200.0], btype="bandpass", fs=sr, output="sos")
        sos_hp = signal.butter(2, 3200.0, btype="highpass", fs=sr, output="sos")

        out_low = signal.sosfiltfilt(sos_lp, out_crop, axis=-1)
        out_mid = signal.sosfiltfilt(sos_bp, out_crop, axis=-1)
        out_high = signal.sosfiltfilt(sos_hp, out_crop, axis=-1)

        in_low = signal.sosfiltfilt(sos_lp, in_crop, axis=-1)
        in_mid = signal.sosfiltfilt(sos_bp, in_crop, axis=-1)
        in_high = signal.sosfiltfilt(sos_hp, in_crop, axis=-1)

        bass_fade_out = (np.maximum(0.0, np.cos(t)) ** 2.4).astype(np.float32)
        bass_fade_in = (np.maximum(0.0, np.sin(t)) ** 2.8).astype(np.float32)

        mid_fade_out = (np.maximum(0.0, np.cos(t)) ** 1.8).astype(np.float32)
        mid_fade_in = (np.maximum(0.0, np.sin(t)) ** 2.2).astype(np.float32)

        out_blend = (out_low * bass_fade_out) + (out_mid * mid_fade_out) + (out_high * fade_out)
        in_blend = (in_low * bass_fade_in) + (in_mid * mid_fade_in) + (in_high * fade_in)

        blend = out_blend + in_blend
    except Exception:
        blend = (out_crop * fade_out) + (in_crop * fade_in)

    max_peak = float(np.max(np.abs(blend)))
    if max_peak > 0.98:
        blend = np.tanh(blend * 0.98) / 0.98

    return blend.astype(np.float32)


def render_dj_echo_out_transition(
    outgoing_tail: np.ndarray,
    incoming_head: np.ndarray,
    bpm_prev: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    DJ Echo-Out Drop Transition for large tempo shifts (`|BPM_A - BPM_B| > 8.5 BPM`):
      - Cuts outgoing track on the last phrase bar with a 20 ms anti-click micro-fade.
      - Synthesizes a 4-tap decaying half-beat feedback echo high-passed at 350 Hz.
      - Drops incoming track at native tempo with a 15 ms micro-fade at the head.
    """
    num_samples = min(outgoing_tail.shape[1], incoming_head.shape[1])
    if num_samples <= 0:
        return np.hstack([outgoing_tail, incoming_head])

    out_crop = outgoing_tail[:, :num_samples]
    in_crop = incoming_head[:, :num_samples]

    safe_bpm = bpm_prev if bpm_prev > 0 else 120.0
    beat_samples = int((60.0 / safe_bpm) * sr)
    half_beat_samples = max(int(0.15 * sr), beat_samples // 2)

    cut_point = max(beat_samples, num_samples - (2 * beat_samples))
    cut_point = min(cut_point, num_samples)

    echo_source = np.copy(out_crop)
    echo_tail = np.zeros_like(out_crop)

    fade_len = min(int(0.02 * sr), cut_point)
    if fade_len > 0:
        cut_fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        echo_source[:, cut_point - fade_len : cut_point] *= cut_fade
    echo_source[:, cut_point:] = 0.0

    slice_start = max(0, cut_point - beat_samples)
    trigger_slice = out_crop[:, slice_start:cut_point]

    if trigger_slice.shape[1] > 0:
        for tap, decay in [(1, 0.60), (2, 0.35), (3, 0.18), (4, 0.08)]:
            offset = cut_point + (tap * half_beat_samples)
            if offset < num_samples:
                tap_len = min(trigger_slice.shape[1], num_samples - offset)
                echo_tail[:, offset : offset + tap_len] += trigger_slice[:, :tap_len] * decay

    try:
        sos_echo = signal.butter(2, 350.0, btype="highpass", fs=sr, output="sos")
        echo_tail = signal.sosfiltfilt(sos_echo, echo_tail, axis=-1)
    except Exception:
        pass

    out_final = echo_source + echo_tail

    in_final = np.copy(in_crop)
    micro_len = min(int(0.015 * sr), num_samples)
    if micro_len > 0:
        micro_fade = np.linspace(0.0, 1.0, micro_len, dtype=np.float32)
        in_final[:, :micro_len] *= micro_fade

    blend = out_final + in_final

    max_peak = float(np.max(np.abs(blend)))
    if max_peak > 0.98:
        blend = np.tanh(blend * 0.98) / 0.98

    return blend.astype(np.float32)


def snap_to_phrase_grid_samples(
    local_samples: int,
    beatgrid: Optional[Sequence[float]],
    bpm: float,
    phrase_bars: int = 4,
    sr: int = SAMPLE_RATE,
) -> int:
    """
    Snaps a sample offset to the nearest 4-bar or 8-bar musical phrase boundary
    (16 or 32 beats) using either the track's detected beatgrid or nominal BPM.
    """
    phrase_samples = int((60.0 / (bpm if bpm > 0 else 120.0)) * (phrase_bars * 4) * sr)
    if phrase_samples <= 0:
        return local_samples

    if beatgrid and len(beatgrid) >= phrase_bars * 4:
        pos_sec = local_samples / sr
        phrase_step = phrase_bars * 4
        phrase_downbeats = [beatgrid[i] for i in range(0, len(beatgrid), phrase_step)]
        if phrase_downbeats:
            closest_phrase_sec = min(phrase_downbeats, key=lambda b: abs(b - pos_sec))
            return int(closest_phrase_sec * sr)

    return int(round(local_samples / phrase_samples) * phrase_samples)


# ── 3. Parallel Prefetching ──


def _decrypt_if_msd1(filepath: str, tmp_path: Path, audio_id: int) -> str:
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
        if magic == b"MSD1":
            from core.audio_crypto import decrypt_audio_bytes

            with open(filepath, "rb") as f_in:
                decrypted = decrypt_audio_bytes(f_in.read())
            out_path = str(tmp_path / f"dec_{audio_id}.mp3")
            with open(out_path, "wb") as f_out:
                f_out.write(decrypted)
            return out_path
    except Exception as exc:
        logger.debug("MSD1 decryption check skipped for %s: %s", filepath, exc)
    return filepath


def download_track(audio_obj: Any, tmp_path: Path, idx: int) -> Tuple[int, Optional[str]]:
    raw_file = getattr(audio_obj, "file", "") or ""
    if raw_file.startswith(("http://", "https://")):
        dest = tmp_path / f"track_{idx}_{audio_obj.id}.mp3"
        if not dest.exists():
            with requests.get(
                raw_file,
                stream=True,
                timeout=60,
                headers={"User-Agent": "MuseedAutomix/1.0"},
            ) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
        return audio_obj.id, _decrypt_if_msd1(str(dest), tmp_path, audio_obj.id)

    candidates = [raw_file]
    if django_settings is not None and getattr(django_settings, "configured", False):
        candidates.extend(
            [
                os.path.join(django_settings.MEDIA_ROOT, raw_file.lstrip("/")),
                os.path.join(django_settings.BASE_DIR, raw_file.lstrip("/")),
            ]
        )
    for p in candidates:
        if p and os.path.exists(p):
            return audio_obj.id, _decrypt_if_msd1(p, tmp_path, audio_obj.id)
    return audio_obj.id, None


def prefetch_all(ordered_audios: Sequence[Any], tmp_path: Path) -> Dict[int, str]:
    file_map: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(download_track, a, tmp_path, i): a.id for i, a in enumerate(ordered_audios)
        }
        for f in as_completed(futures):
            try:
                aid, path = f.result()
                if path:
                    file_map[aid] = path
            except Exception as exc:
                logger.warning("Failed to prefetch track #%s: %s", futures[f], exc)
    return file_map


# ── 4. Django Management Command Entrypoint (Production Integration) ──


class Command(BaseCommand):
    help = "Generates Studio-Grade DJ Automixes with 3-Band Crossover Transitions & 32-Beat Phrase-Matching"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--user-id", type=int, help="User ID to pick audios from")
        parser.add_argument("--audio-ids", type=str, required=True, help="Comma-separated list of Audio IDs")
        parser.add_argument("--output-title", type=str, help="Custom title for the mix")
        parser.add_argument("--max-minutes", type=int, default=60, help="Max duration in minutes")
        parser.add_argument("--no-save-db", action="store_true", help="Do not save to DB")

    def handle(self, *args: Any, **options: Any) -> None:
        User = get_user_model()
        if User is None or django_settings is None:
            raise CommandError("Django environment is required to run the management command entrypoint.")

        user_id = options.get("user_id")
        audio_ids_str = options.get("audio_ids")
        custom_title = options.get("output_title")
        max_minutes = options.get("max_minutes", 60)
        save_db = not options.get("no_save_db", False)

        user = (
            User.objects.filter(id=user_id).first()
            if user_id
            else User.objects.filter(is_active=True).first()
        )
        if not user:
            raise CommandError("No valid user found.")

        from core.models import Audio

        parsed_ids = [int(i.strip()) for i in str(audio_ids_str).split(",") if i.strip().isdigit()]
        audios = list(Audio.objects.filter(id__in=parsed_ids, user=user))
        if len(audios) < 2:
            raise CommandError("Please provide at least 2 valid audio IDs.")

        # 1. 5D Optimal Sequencing
        ordered = build_optimal_harmonic_chain(audios)
        raw_unmixed_total_sec = sum(float(a.duration or 0) for a in ordered)

        # 2. Real-Time Stream-to-Pipe DSP Mixing Engine
        mix_title = custom_title or f"DJ Mix — {datetime.now().strftime('%b %d, %Y')}"
        safe_filename = f"automix_{user.id}_{uuid.uuid4().hex[:8]}.mp3"
        output_file_path = Path(django_settings.MEDIA_ROOT) / "automixes" / safe_filename
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-b:a",
            "320k",
            "-metadata",
            f"title={mix_title}",
            "-metadata",
            "artist=Museed Automix Engine",
            "-metadata",
            "album=DJ Mixes",
            str(output_file_path),
        ]
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        total_master_samples = 0

        def stream_chunk(chunk_data: Optional[np.ndarray]) -> None:
            nonlocal total_master_samples
            if chunk_data is not None and chunk_data.shape[1] > 0 and ffmpeg_proc.stdin is not None:
                raw_bytes = chunk_data.T.astype(np.float32).tobytes()
                ffmpeg_proc.stdin.write(raw_bytes)
                total_master_samples += chunk_data.shape[1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_map = prefetch_all(ordered, tmp_path)

            active_body: Optional[np.ndarray] = None
            prev_audio_obj: Any = None
            prev_track_samples = 0
            loaded_count = 0
            accumulated_samples = 0
            timeline_cursor_samples = 0
            max_samples = max_minutes * 60 * SAMPLE_RATE

            for idx, audio_obj in enumerate(ordered):
                if accumulated_samples >= max_samples:
                    break

                track_id = audio_obj.id
                local_file = file_map.get(track_id)
                if not local_file or not os.path.exists(local_file):
                    continue

                bpm = float(audio_obj.bpm) if audio_obj.bpm else 120.0
                bpm_prev = float(prev_audio_obj.bpm or 120.0) if prev_audio_obj else bpm
                bpm_diff = abs(bpm_prev - bpm)

                track_audio = load_and_normalize_track(
                    local_file,
                    track_lufs=audio_obj.lufs,
                    target_lufs=-14.0,
                    trim=(loaded_count > 0),
                )
                track_audio = align_track_phase(track_audio, bpm)
                curr_samples = track_audio.shape[1]

                bar_samples = int((60.0 / bpm) * 4 * SAMPLE_RATE)
                phrase_bars = 4 if bpm >= 120.0 else 2
                phrase_samples = bar_samples * phrase_bars

                intro_sec = getattr(audio_obj, "best_mix_in_sec", None) or getattr(
                    audio_obj, "intro_end_sec", None
                )
                if intro_sec and float(intro_sec) >= 4.0:
                    raw_intro_samples = int(float(intro_sec) * SAMPLE_RATE)
                    transition_samples = int(round(raw_intro_samples / bar_samples) * bar_samples)
                    transition_samples = max(4 * SAMPLE_RATE, min(24 * SAMPLE_RATE, transition_samples))
                else:
                    transition_samples = phrase_samples

                if loaded_count == 0:
                    active_body = track_audio
                    accumulated_samples = curr_samples
                else:
                    assert active_body is not None
                    active_len = active_body.shape[1]

                    max_allowed_tail = max(
                        transition_samples,
                        min(int(prev_track_samples * 0.12), 30 * SAMPLE_RATE),
                    )
                    tail_samples = max(
                        transition_samples,
                        min(transition_samples * 2, max_allowed_tail),
                    )

                    best_out_sec = getattr(prev_audio_obj, "best_mix_out_sec", None) or getattr(
                        prev_audio_obj, "outro_start_sec", None
                    )
                    if best_out_sec and float(best_out_sec) > 0:
                        prev_dur_sec = prev_track_samples / SAMPLE_RATE
                        if prev_dur_sec > float(best_out_sec):
                            raw_tail = int((prev_dur_sec - float(best_out_sec)) * SAMPLE_RATE)
                            tail_samples = max(transition_samples, min(max_allowed_tail, raw_tail))

                    local_outro = max(0, prev_track_samples - tail_samples)
                    snapped_local_outro = snap_to_phrase_grid_samples(
                        local_outro,
                        getattr(prev_audio_obj, "beatgrid", []),
                        bpm_prev,
                        phrase_bars=phrase_bars,
                    )
                    snapped_tail = max(transition_samples, prev_track_samples - snapped_local_outro)

                    outro_pos = max(0, active_len - snapped_tail)
                    overlap = min(transition_samples, active_len - outro_pos, curr_samples)
                    if overlap <= 0:
                        outro_pos = max(0, active_len - transition_samples)
                        overlap = min(transition_samples, active_len - outro_pos, curr_samples)

                    body_prev = active_body[:, :outro_pos]
                    tail_prev = active_body[:, outro_pos : outro_pos + overlap]
                    head_curr = track_audio[:, :overlap]
                    body_curr = track_audio[:, overlap:]

                    stream_chunk(body_prev)
                    del body_prev
                    del active_body

                    if bpm_diff > 8.5:
                        transition_zone = render_dj_echo_out_transition(
                            tail_prev, head_curr, bpm_prev=bpm_prev, sr=SAMPLE_RATE
                        )
                    else:
                        if 0.4 <= bpm_diff <= 8.5:
                            tail_prev = dj_time_stretch(
                                tail_prev, source_bpm=bpm_prev, target_bpm=bpm
                            )
                        transition_zone = studio_smooth_dj_transition(
                            tail_prev, head_curr, sr=SAMPLE_RATE
                        )

                    stream_chunk(transition_zone)
                    del transition_zone
                    del tail_prev
                    del head_curr

                    timeline_cursor_samples += outro_pos + overlap
                    active_body = body_curr
                    accumulated_samples += body_curr.shape[1]

                prev_track_samples = curr_samples
                prev_audio_obj = audio_obj
                del track_audio
                gc.collect()
                loaded_count += 1

            if active_body is not None:
                stream_chunk(active_body)
                del active_body
                gc.collect()

            if loaded_count == 0:
                raise CommandError("Could not mix audio tracks.")

            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.flush()
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

            if ffmpeg_proc.returncode != 0:
                err = (
                    ffmpeg_proc.stderr.read().decode("utf-8", errors="ignore")
                    if ffmpeg_proc.stderr
                    else "Unknown error"
                )
                raise CommandError(f"FFmpeg encoding failed: {err}")

            final_duration_sec = total_master_samples / float(SAMPLE_RATE)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Generated Automix '{mix_title}' ({final_duration_sec:.1f}s from {raw_unmixed_total_sec:.1f}s raw)."
                )
            )
            sys.stdout.flush()

            if save_db:
                file_size = output_file_path.stat().st_size
                hasher = hashlib.sha256()
                with open(output_file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)

                Audio.objects.create(
                    user=user,
                    filename=safe_filename,
                    title=mix_title,
                    artist="Museed Automix Engine",
                    album="DJ Mixes",
                    genre="DJ Mix",
                    file=f"/media/automixes/{safe_filename}",
                    file_size=file_size,
                    duration=str(int(final_duration_sec)),
                    file_hash=hasher.hexdigest(),
                    imported_by_script=True,
                    analysis_status="pending",
                )
