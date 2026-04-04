"""
src/features.py
===============
Feature extraction pipeline for the attention-geometry project.

Extracts four groups of content-side features from each video:

    Group A - Audio
        Speech rate, pitch statistics, energy envelope, tempo,
        spectral centroid, zero-crossing rate, music presence indicator.
        Source: yt-dlp (audio stream only) + librosa.

    Group B - Visual
        Thumbnail colour palette (dominant hues, saturation, brightness),
        colour variance, estimated visual complexity.
        Source: thumbnail URL from API + Pillow.

    Group C - Text
        Title and description sentiment (VADER), readability (textstat),
        lexical diversity, punctuation signals, length statistics.
        Source: title and description fields from API.

    Group D - Structural
        Duration, estimated pacing label, engagement-per-second proxy.
        Source: derived from API fields.

Design principle
----------------
Every function is pure (input → output, no side effects) and independently
testable. The notebook calls them in sequence; they can also be called
individually for debugging.

Audio processing note
---------------------
yt-dlp downloads only the audio stream (m4a/webm), which is ~3-10 MB
per video rather than the full video file. This keeps storage manageable.
Librosa resamples to 22,050 Hz for all analyses.

Author: K. Cissé
Date:   April 2026
"""

import os
import io
import time
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP A - AUDIO FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def download_audio(video_id: str, out_dir: Path,
                   timeout_seconds: int = 120) -> Optional[Path]:
    """
    Download the audio stream of a YouTube video using yt-dlp.

    Downloads only the audio (no video) to keep file sizes small.
    Output format is best-available audio, converted to wav for librosa.

    Parameters
    ----------
    video_id        : YouTube video ID
    out_dir         : directory to save the audio file
    timeout_seconds : maximum time to wait for the download

    Returns
    -------
    Path to the downloaded .wav file, or None if download failed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.wav"

    if out_path.exists():
        log.info(f"  Audio already exists: {out_path}")
        return out_path

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "5",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--retries", "3",
        "--fragment-retries", "3",
        "--sleep-interval", "2",
        "--output", str(out_dir / f"{video_id}.%(ext)s"),
        url,
    ]

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.warning(f"  yt-dlp failed for {video_id}: {result.stderr[:200]}")
            return None
        if out_path.exists():
            return out_path
        # Sometimes yt-dlp saves with a slightly different extension
        candidates = list(out_dir.glob(f"{video_id}.*"))
        if candidates:
            return candidates[0]
        return None

    except subprocess.TimeoutExpired:
        log.warning(f"  Timeout downloading audio for {video_id}")
        return None
    except FileNotFoundError:
        log.error("  yt-dlp not found. Install with: pip install yt-dlp")
        return None


def extract_audio_features(audio_path: Path,
                            max_duration_sec: float = 120.0) -> dict:
    """
    Extract audio features from a downloaded audio file using librosa.

    We analyse only the first `max_duration_sec` seconds. This is
    sufficient for engagement-relevant features (hooks, speech rate,
    energy) and keeps computation fast.

    Features extracted
    ------------------
    speech_rate_proxy : zero-crossing rate (correlates with speech articulation)
    pitch_mean        : mean fundamental frequency (F0) in Hz
    pitch_std         : variability of pitch — monotone vs expressive
    energy_mean       : mean RMS energy — loud vs quiet content
    energy_std        : energy variability — dynamic vs flat
    tempo_bpm         : estimated beats per minute
    spectral_centroid : mean spectral centroid — brightness of the sound
    spectral_bandwidth: mean spectral bandwidth — richness
    music_presence    : heuristic indicator (high tempo + low ZCR → likely music)
    silence_ratio     : fraction of frames with near-zero energy

    Returns
    -------
    dict of audio features, or dict of NaN values if extraction fails.
    """
    _nan_audio = {
        'pitch_mean': np.nan, 'pitch_std': np.nan,
        'energy_mean': np.nan, 'energy_std': np.nan,
        'tempo_bpm': np.nan, 'spectral_centroid': np.nan,
        'spectral_bandwidth': np.nan, 'zcr_mean': np.nan,
        'music_presence': np.nan, 'silence_ratio': np.nan,
    }

    try:
        import librosa
    except ImportError:
        log.error("  librosa not found. Install with: pip install librosa")
        return _nan_audio

    try:
        y, sr = librosa.load(str(audio_path), sr=22050,
                             duration=max_duration_sec, mono=True)
    except Exception as e:
        log.warning(f"  librosa.load failed: {e}")
        return _nan_audio

    if len(y) < sr:  # less than 1 second — skip
        return _nan_audio

    try:
        # ── Pitch (F0) via pyin ──────────────────────────────────────────────
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else np.nan
        pitch_std  = float(np.std(voiced_f0))  if len(voiced_f0) > 0 else np.nan

        # ── Energy (RMS) ─────────────────────────────────────────────────────
        rms        = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))
        energy_std  = float(np.std(rms))
        silence_ratio = float(np.mean(rms < 0.01))

        # ── Tempo ─────────────────────────────────────────────────────────────
        onset_env  = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _   = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr) 
        tempo_bpm  = float(np.squeeze(tempo))

        # ── Spectral features ─────────────────────────────────────────────────
        spec_cent  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_bw    = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spec_centroid   = float(np.mean(spec_cent))
        spec_bandwidth  = float(np.mean(spec_bw))

        # ── Zero-crossing rate ───────────────────────────────────────────────
        zcr        = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean   = float(np.mean(zcr))

        # ── Music presence heuristic ─────────────────────────────────────────
        # Speech: high ZCR, variable pitch, moderate tempo
        # Music:  regular tempo, lower ZCR, higher spectral centroid
        # This is a rough heuristic, not a classifier.
        music_score = 0.0
        if tempo_bpm > 60:
            music_score += 0.4
        if zcr_mean < 0.08:
            music_score += 0.3
        if not np.isnan(pitch_std) and pitch_std < 30:
            music_score += 0.3
        music_presence = float(music_score)

        return {
            'pitch_mean':        pitch_mean,
            'pitch_std':         pitch_std,
            'energy_mean':       energy_mean,
            'energy_std':        energy_std,
            'tempo_bpm':         tempo_bpm,
            'spectral_centroid': spec_centroid,
            'spectral_bandwidth': spec_bandwidth,
            'zcr_mean':          zcr_mean,
            'music_presence':    music_presence,
            'silence_ratio':     silence_ratio,
        }

    except Exception as e:
        log.warning(f"  Audio feature extraction failed: {e}")
        return _nan_audio


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP B - VISUAL FEATURES (THUMBNAIL)
# ═══════════════════════════════════════════════════════════════════════════════

def download_thumbnail(thumbnail_url: str, video_id: str,
                       out_dir: Path) -> Optional[Path]:
    """
    Download a video thumbnail from its URL.

    Parameters
    ----------
    thumbnail_url : URL from YouTube API (maxres or high quality)
    video_id      : used for the output filename
    out_dir       : directory to save the image

    Returns
    -------
    Path to saved thumbnail, or None if download failed.
    """
    import urllib.request

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.jpg"

    if out_path.exists():
        return out_path

    if not thumbnail_url:
        return None

    try:
        urllib.request.urlretrieve(thumbnail_url, out_path)
        return out_path
    except Exception as e:
        log.warning(f"  Thumbnail download failed for {video_id}: {e}")
        return None


def extract_visual_features(thumbnail_path: Path) -> dict:
    """
    Extract visual features from a thumbnail image using Pillow.

    Features extracted
    ------------------
    brightness_mean   : mean luminosity (0–255) — overall brightness
    brightness_std    : brightness variability — high contrast vs flat
    saturation_mean   : mean colour saturation — vivid vs muted palette
    saturation_std    : saturation variability
    hue_mean          : mean hue in HSV space (0–360)
    hue_std           : hue variability — colourful vs monochromatic
    red_dominance     : fraction of pixels where red channel is highest
    visual_complexity : normalised pixel variance — busy vs simple thumbnail
    warm_cool_ratio   : ratio of warm (red/yellow) to cool (blue/green) pixels
    face_heuristic    : rough proxy for face presence (skin-tone pixel fraction)

    Returns
    -------
    dict of visual features, or dict of NaN values if extraction fails.
    """
    _nan_visual = {
        'brightness_mean': np.nan, 'brightness_std': np.nan,
        'saturation_mean': np.nan, 'saturation_std': np.nan,
        'hue_mean': np.nan, 'hue_std': np.nan,
        'red_dominance': np.nan, 'visual_complexity': np.nan,
        'warm_cool_ratio': np.nan, 'face_heuristic': np.nan,
    }

    try:
        from PIL import Image
    except ImportError:
        log.error("  Pillow not found. Install with: pip install Pillow")
        return _nan_visual

    try:
        img = Image.open(thumbnail_path).convert('RGB')
        # Resize to 128×72 for speed (keeps aspect ratio for thumbnails)
        img = img.resize((128, 72), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)  # shape (72, 128, 3)

        # ── Brightness (luminosity) ───────────────────────────────────────────
        luminosity = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        brightness_mean = float(luminosity.mean())
        brightness_std  = float(luminosity.std())

        # ── HSV conversion for saturation and hue ────────────────────────────
        img_hsv = img.convert('HSV') if hasattr(Image, 'HSV') else None
        # Pillow HSV not always available — compute manually
        r, g, b = arr[:,:,0]/255, arr[:,:,1]/255, arr[:,:,2]/255
        cmax   = np.maximum(np.maximum(r, g), b)
        cmin   = np.minimum(np.minimum(r, g), b)
        delta  = cmax - cmin

        saturation = np.where(cmax > 0, delta / (cmax + 1e-8), 0)
        sat_mean = float(saturation.mean())
        sat_std  = float(saturation.std())

        # Hue (0–360)
        hue = np.zeros_like(r)
        mask_r = (cmax == r) & (delta > 0)
        mask_g = (cmax == g) & (delta > 0)
        mask_b = (cmax == b) & (delta > 0)
        hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        hue[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
        hue[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)
        hue_mean = float(hue.mean())
        hue_std  = float(hue.std())

        # ── Dominant channel ──────────────────────────────────────────────────
        red_dom = float(np.mean(
            (arr[:,:,0] > arr[:,:,1]) & (arr[:,:,0] > arr[:,:,2])
        ))

        # ── Visual complexity ─────────────────────────────────────────────────
        # Normalised pixel variance — a busy thumbnail has high variance
        visual_complexity = float(arr.var() / (255**2))

        # ── Warm / cool ratio ─────────────────────────────────────────────────
        # Warm pixels: hue in [0,60] ∪ [300,360] (red, orange, yellow)
        # Cool pixels: hue in [120,270] (green, cyan, blue)
        warm = np.mean((hue < 60) | (hue > 300))
        cool = np.mean((hue > 120) & (hue < 270))
        warm_cool_ratio = float(warm / (cool + 1e-8))

        # ── Face heuristic ────────────────────────────────────────────────────
        # Skin tone in RGB: roughly r > 95, g > 40, b > 20,
        #                           r > g, r > b, |r-g| > 15
        r8, g8, b8 = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        skin_mask = (
            (r8 > 95) & (g8 > 40) & (b8 > 20) &
            (r8 > g8) & (r8 > b8) &
            (np.abs(r8 - g8) > 15)
        )
        face_heuristic = float(skin_mask.mean())

        return {
            'brightness_mean':  brightness_mean,
            'brightness_std':   brightness_std,
            'saturation_mean':  sat_mean,
            'saturation_std':   sat_std,
            'hue_mean':         hue_mean,
            'hue_std':          hue_std,
            'red_dominance':    red_dom,
            'visual_complexity': visual_complexity,
            'warm_cool_ratio':  warm_cool_ratio,
            'face_heuristic':   face_heuristic,
        }

    except Exception as e:
        log.warning(f"  Visual feature extraction failed: {e}")
        return _nan_visual


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP C - TEXT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_features(title: str, description: str) -> dict:
    """
    Extract linguistic and sentiment features from title and description.

    Features extracted
    ------------------
    title_length          : character count of title
    title_word_count      : word count of title
    title_sentiment       : VADER compound score of title (−1 to +1)
    title_has_question    : 1 if title contains '?'
    title_has_exclamation : 1 if title contains '!'
    title_all_caps_words  : count of ALL-CAPS words in title
    title_has_number      : 1 if title contains a digit
    desc_length           : character count of description (capped at 1000)
    desc_readability      : Flesch reading ease score
    desc_sentiment        : VADER compound score of first 500 chars
    lexical_diversity     : type-token ratio of title + first 200 chars of desc
    combined_sentiment    : weighted average of title and desc sentiment

    Returns
    -------
    dict of text features
    """
    _nan_text = {
        'title_length': np.nan, 'title_word_count': np.nan,
        'title_sentiment': np.nan, 'title_has_question': np.nan,
        'title_has_exclamation': np.nan, 'title_all_caps_words': np.nan,
        'title_has_number': np.nan, 'desc_length': np.nan,
        'desc_readability': np.nan, 'desc_sentiment': np.nan,
        'lexical_diversity': np.nan, 'combined_sentiment': np.nan,
    }

    if not title:
        return _nan_text

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import textstat
    except ImportError as e:
        log.error(f"  Missing text library: {e}. "
                  "Install: pip install vaderSentiment textstat")
        return _nan_text

    vader = SentimentIntensityAnalyzer()

    # ── Title features ────────────────────────────────────────────────────────
    title_words  = title.split()
    title_length = len(title)
    title_wc     = len(title_words)
    title_sent   = vader.polarity_scores(title)['compound']
    has_question = int('?' in title)
    has_exclaim  = int('!' in title)
    caps_words   = sum(1 for w in title_words
                       if w.isupper() and len(w) > 1)
    has_number   = int(any(ch.isdigit() for ch in title))

    # ── Description features ─────────────────────────────────────────────────
    desc = str(description or '')[:1000]
    desc_len  = len(desc)
    desc_read = float(textstat.flesch_reading_ease(desc)) if desc else np.nan
    desc_sent = vader.polarity_scores(desc[:500])['compound'] if desc else np.nan

    # ── Lexical diversity (type-token ratio) ──────────────────────────────────
    combined_text = (title + ' ' + desc[:200]).lower().split()
    types  = len(set(combined_text))
    tokens = len(combined_text)
    lexical_div = types / tokens if tokens > 0 else np.nan

    # ── Combined sentiment (title weighted 2×, description 1×) ───────────────
    if not np.isnan(desc_sent):
        combined_sent = (2 * title_sent + desc_sent) / 3
    else:
        combined_sent = title_sent

    return {
        'title_length':          title_length,
        'title_word_count':      title_wc,
        'title_sentiment':       title_sent,
        'title_has_question':    has_question,
        'title_has_exclamation': has_exclaim,
        'title_all_caps_words':  caps_words,
        'title_has_number':      has_number,
        'desc_length':           desc_len,
        'desc_readability':      desc_read,
        'desc_sentiment':        desc_sent,
        'lexical_diversity':     lexical_div,
        'combined_sentiment':    combined_sent,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP D - STRUCTURAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_structural_features(duration_seconds: float,
                                 view_count: int,
                                 like_count: int,
                                 comment_count: int) -> dict:
    """
    Derive structural and pacing features from metadata.

    Features extracted
    ------------------
    duration_seconds    : video length in seconds
    duration_minutes    : video length in minutes
    pacing_label        : 'short' (<3min), 'medium' (3–15min), 'long' (>15min)
    pacing_code         : numeric code for pacing label (0/1/2)
    log_duration        : log10(duration_seconds + 1) — for modelling
    engagement_density  : (likes + comments) / duration_seconds
                          — how much reaction per second of content
    log_views           : log10(view_count + 1) — for popularity control

    Returns
    -------
    dict of structural features
    """
    dur  = float(duration_seconds) if duration_seconds else np.nan
    dur_min = dur / 60 if not np.isnan(dur) else np.nan

    if np.isnan(dur):
        pacing_label = 'unknown'
        pacing_code  = -1
    elif dur < 180:
        pacing_label = 'short'
        pacing_code  = 0
    elif dur < 900:
        pacing_label = 'medium'
        pacing_code  = 1
    else:
        pacing_label = 'long'
        pacing_code  = 2

    log_dur = float(np.log10(dur + 1)) if not np.isnan(dur) else np.nan
    log_views = float(np.log10(view_count + 1)) if view_count >= 0 else np.nan

    eng_density = (
        (like_count + comment_count) / dur
        if (not np.isnan(dur) and dur > 0)
        else np.nan
    )

    return {
        'duration_seconds':   dur,
        'duration_minutes':   dur_min,
        'pacing_label':       pacing_label,
        'pacing_code':        pacing_code,
        'log_duration':       log_dur,
        'engagement_density': eng_density,
        'log_views':          log_views,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all_features(row: pd.Series,
                          audio_dir: Path,
                          thumb_dir: Path,
                          download_audio_flag: bool = False) -> dict:
    """
    Extract all feature groups for a single video row.

    Parameters
    ----------
    row                  : a row from videos_t0.csv
    audio_dir            : directory for downloaded audio files
    thumb_dir            : directory for downloaded thumbnails
    download_audio_flag  : if True, attempt audio download + librosa features.
                           Set to False for a fast run (visual + text only).

    Returns
    -------
    dict with all features from groups A, B, C, D, keyed by feature name.
    """
    video_id = row['video_id']
    features = {'video_id': video_id}

    # ── Group D: Structural (always fast) ─────────────────────────────────────
    structural = extract_structural_features(
        duration_seconds=row.get('duration_seconds', np.nan),
        view_count=int(row.get('view_count', 0)),
        like_count=int(row.get('like_count', 0)),
        comment_count=int(row.get('comment_count', 0)),
    )
    features.update({f'd_{k}': v for k, v in structural.items()})

    # ── Group C: Text (always fast) ───────────────────────────────────────────
    text_feats = extract_text_features(
        title=str(row.get('title', '') or ''),
        description=str(row.get('description', '') or ''),
    )
    features.update({f't_{k}': v for k, v in text_feats.items()})

    # ── Group B: Visual (thumbnail download + Pillow) ─────────────────────────
    thumb_url  = row.get('thumbnail_url', '')
    thumb_path = download_thumbnail(thumb_url, video_id, thumb_dir)
    if thumb_path:
        vis_feats = extract_visual_features(thumb_path)
    else:
        vis_feats = {k: np.nan for k in [
            'brightness_mean', 'brightness_std', 'saturation_mean',
            'saturation_std', 'hue_mean', 'hue_std', 'red_dominance',
            'visual_complexity', 'warm_cool_ratio', 'face_heuristic',
        ]}
    features.update({f'v_{k}': v for k, v in vis_feats.items()})

    # ── Group A: Audio (yt-dlp + librosa / slow, optional) ───────────────────
    if download_audio_flag:
        audio_path = download_audio(video_id, audio_dir)
        if audio_path:
            aud_feats = extract_audio_features(audio_path)
            # Delete immediately after extraction to save space
            try:
                audio_path.unlink()
            except Exception:
                pass
        else:
            aud_feats = {k: np.nan for k in [
                'pitch_mean', 'pitch_std', 'energy_mean', 'energy_std',
                'tempo_bpm', 'spectral_centroid', 'spectral_bandwidth',
                'zcr_mean', 'music_presence', 'silence_ratio',
            ]}
        features.update({f'a_{k}': v for k, v in aud_feats.items()})
    return features


# ── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Feature extraction module — sanity check")
    print("=" * 50)

    # Text features (no external files needed)
    title = "Why Most People Get Investing WRONG (And How to Fix It)"
    desc  = "In this video I explain the five most common mistakes new investors make."
    tf = extract_text_features(title, desc)
    print("\nText features:")
    for k, v in tf.items():
        print(f"  {k:<28} = {v}")

    # Structural features
    sf = extract_structural_features(
        duration_seconds=847,
        view_count=1_200_000,
        like_count=48_000,
        comment_count=2_300,
    )
    print("\nStructural features:")
    for k, v in sf.items():
        print(f"  {k:<28} = {v}")

    print("\nModule OK. Audio and visual require downloaded files.")
    print("Run notebook 02 for the full pipeline.")