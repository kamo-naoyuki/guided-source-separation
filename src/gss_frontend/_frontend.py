# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guided Source Separation (GSS) frontend for speech enhancement.

This is a minimal, self-contained implementation derived from FrontEnd_v1 in the
CHiME-8 DASR NeMo baseline. It processes one utterance at a time (no batch
infrastructure or lhotse dependencies).

Requires:
    torch, torchaudio, numpy, soundfile
"""

import math
import logging
import contextlib
import heapq
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch

from ._modules import (
    AudioToSpectrogram,
    SpectrogramToAudio,
    MaskBasedBeamformer,
    MaskBasedDereverbWPE,
    MaskEstimatorGSS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def samples_to_frames(samples: int, fft_length: int, hop_length: int) -> int:
    """Convert number of audio samples to STFT frame count."""
    frames = (samples - fft_length + hop_length) / hop_length
    return int(frames)


def activity_time_to_timefreq(
    a_time: torch.Tensor,
    win_length: int,
    hop_length: int,
    aggregation: str = "mean",
) -> torch.Tensor:
    """Convert time-domain speaker activity to frame-domain activity.

    Args:
        a_time: Speaker activity with shape (batch, speakers, time).
            Binary (0/1) or soft values in [0, 1].
        win_length: STFT window length (= fft_length).
        hop_length: STFT hop length.
        aggregation: Frame aggregation method for soft activity.
            - ``"mean"``: average over the STFT window (default)
            - ``"max"``: max over the STFT window
            - ``"any"``: binary occupancy (window has any activity)

    Returns:
        Float tensor with shape (batch, speakers, frames).
    """
    assert a_time.ndim == 3
    a_tf = torch.nn.functional.pad(
        a_time.unsqueeze(-1),
        pad=(0, 0, win_length // 2, win_length // 2),
    )
    a_tf = torch.nn.functional.unfold(
        a_tf, kernel_size=(win_length, 1), stride=(hop_length, 1)
    )
    a_tf = a_tf.reshape(a_time.size(0), a_time.size(1), win_length, -1)
    a_tf = a_tf.clamp(min=0.0)
    if aggregation == "mean":
        a_tf = a_tf.mean(dim=-2)
    elif aggregation == "max":
        a_tf = a_tf.max(dim=-2).values
    elif aggregation == "any":
        a_tf = a_tf.gt(0).any(dim=-2).to(dtype=a_tf.dtype)
    else:
        raise ValueError(
            f"Unsupported activity aggregation '{aggregation}'. "
            "Use one of: 'mean', 'max', 'any'."
        )
    return a_tf


def _get_int_divisors(n: int):
    """Return sorted list of integer divisors of n (used for OOM chunking)."""
    divs = [1]
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.extend([i, n // i])
    divs.append(n)
    return sorted(list(set(divs)))


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True for CUDA OOM and closely related allocator failures.

    CUDA execution is asynchronous, so memory failures sometimes surface as a
    generic RuntimeError at a later synchronization point rather than as
    torch.cuda.OutOfMemoryError at the original allocation site.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False

    msg = str(exc).lower()
    oom_markers = (
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "hip out of memory",
        "out of memory",
    )
    return any(marker in msg for marker in oom_markers)


def _try_gpu_else_cpu(module: torch.nn.Module, *args, **kwargs):
    """Run *module* on GPU; on CUDA OOM fall back to CPU for this call only.

    Moves all tensor arguments to CPU, runs the module (which must have no
    GPU-resident parameters or buffers), then moves the outputs back to the
    original device.  Non-tensor arguments are passed through unchanged.
    """
    try:
        return module(*args, **kwargs)
    except RuntimeError as exc:
        if not _is_cuda_oom_error(exc):
            raise
        torch.cuda.empty_cache()
        logger.warning(
            "CUDA OOM in %s — retrying this chunk on CPU.",
            module.__class__.__name__,
        )
        # Infer original device from the first tensor argument.
        device = next(
            (a.device for a in args if isinstance(a, torch.Tensor)),
            None,
        )
        if device is None:
            device = next(
                (v.device for v in kwargs.values() if isinstance(v, torch.Tensor)),
                None,
            )
        cpu_args = tuple(a.cpu() if isinstance(a, torch.Tensor) else a for a in args)
        cpu_kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        result = module(*cpu_args, **cpu_kwargs)
        if device is not None:
            if isinstance(result, (tuple, list)):
                return type(result)(r.to(device) if isinstance(r, torch.Tensor) else r for r in result)
            if isinstance(result, torch.Tensor):
                return result.to(device)
        return result


def _prepare_audio(
    audio: Union[np.ndarray, torch.Tensor], device: torch.device
) -> "tuple[torch.Tensor, bool]":
    """Add batch dim and move to device.  Returns (tensor, is_numpy)."""
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(audio).float().to(device).unsqueeze(0), True
    t = audio if audio.dim() == 3 else audio.unsqueeze(0)
    return t.to(device), False


def _prepare_activity(
    activity: Union[np.ndarray, torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Add batch dim and move to device."""
    if isinstance(activity, np.ndarray):
        return torch.from_numpy(activity).float().to(device).unsqueeze(0)
    t = activity if activity.dim() == 3 else activity.unsqueeze(0)
    return t.to(device)


def _append_garbage_activity_class(
    activity: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """Append one always-active garbage/background class when enabled."""
    if not enabled:
        return activity
    garbage = torch.full(
        (activity.size(0), 1, activity.size(-1)),
        fill_value=1.0,
        dtype=activity.dtype,
        device=activity.device,
    )
    return torch.cat((activity, garbage), dim=1)


def _to_sample_index(value: Union[int, float], sample_rate: int, unit: str) -> int:
    """Convert a time/sample index to integer sample index."""
    if unit == "samples":
        return int(value)
    if unit == "seconds":
        return int(round(float(value) * sample_rate))
    raise ValueError("segment_unit must be either 'samples' or 'seconds'.")


def _validate_segment_mode(mode: str) -> str:
    """Validate enhance mode for :meth:`GSS.enhance_segment`."""
    aliases = {
        "enhance": "standard",
        "auto": "oom_fallback",
    }
    if mode in aliases:
        new_mode = aliases[mode]
        logger.warning(
            "mode='%s' is deprecated and will be removed in a future release; "
            "use mode='%s' instead.",
            mode,
            new_mode,
        )
        return new_mode

    valid_modes = ("standard", "oom_fallback")
    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {valid_modes}, got '{mode}'."
        )
    return mode


def _import_meeteval_io():
    """Import ``meeteval.io`` with a clear optional-dependency error."""
    try:
        from meeteval import io as meeteval_io
    except ImportError as exc:
        raise ImportError(
            "Optional dependency 'meeteval' is required for diarization loading. "
            "Install with: pip install 'gss-frontend[diarization]'"
        ) from exc
    return meeteval_io


def _partition_segments_by_duration(
    segments: List[Dict[str, Any]],
    num_groups: int = 1,
) -> List[List[int]]:
    """Partition segment indices into groups with balanced total duration.

    Uses a greedy algorithm: repeatedly assign the next segment (in descending
    duration order) to the group with the smallest total duration.

    Args:
        segments: List of dicts with at least "start" and "end" keys.
        num_groups: Number of groups to partition into (default 1 = no partitioning).

    Returns:
        List of num_groups lists, each containing segment indices for that group.
        Each segment appears exactly once, in no particular order within its group.
    """
    import heapq

    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if not segments:
        return [[] for _ in range(num_groups)]
    if num_groups == 1:
        return [list(range(len(segments)))]

    # Compute duration for each segment
    durations = [
        seg["end"] - seg["start"]
        for seg in segments
    ]

    # Sort segment indices by descending duration
    sorted_indices = sorted(
        range(len(segments)),
        key=lambda i: durations[i],
        reverse=True,
    )

    # Initialize groups with (total_duration, group_id, [segment_indices])
    # Use heap to efficiently find the group with smallest duration
    groups: List[Tuple[float, int, List[int]]] = [
        (0.0, gid, [])
        for gid in range(num_groups)
    ]
    heapq.heapify(groups)

    # Greedily assign each segment to the group with smallest total duration
    for seg_idx in sorted_indices:
        total_dur, gid, seg_list = heapq.heappop(groups)
        seg_list.append(seg_idx)
        heapq.heappush(
            groups,
            (total_dur + durations[seg_idx], gid, seg_list),
        )

    # Extract result in group order
    result_dict: Dict[int, List[int]] = {}
    for _, gid, seg_list in groups:
        result_dict[gid] = seg_list

    return [result_dict[gid] for gid in range(num_groups)]


def _compute_group_statistics(
    segments: List[Dict[str, Any]],
    group_indices: List[int],
) -> Dict[str, Any]:
    """Compute statistics for a group of segments.

    Args:
        segments: List of all segments.
        group_indices: Indices of segments in this group.

    Returns:
        Dict with keys: "num_segments", "total_duration_seconds", "avg_duration_seconds".
    """
    if not group_indices:
        return {
            "num_segments": 0,
            "total_duration_seconds": 0.0,
            "avg_duration_seconds": 0.0,
        }

    total_duration = sum(
        segments[i]["end"] - segments[i]["start"]
        for i in group_indices
    )
    avg_duration = total_duration / len(group_indices)

    return {
        "num_segments": len(group_indices),
        "total_duration_seconds": total_duration,
        "avg_duration_seconds": avg_duration,
    }


def _segment_to_dict(segment: Any, default_session: Optional[str] = None) -> Dict[str, Any]:
    """Normalize one diarization segment object to a dict.

    Expected normalized keys are: ``session_id``, ``speaker``, ``start``, ``end``
    where time values are in seconds.
    """

    def _get(names):
        if isinstance(segment, dict):
            for name in names:
                if name in segment:
                    return segment[name]
            return None
        for name in names:
            if hasattr(segment, name):
                return getattr(segment, name)
        return None

    if isinstance(segment, (tuple, list)):
        if len(segment) == 4:
            session_id, speaker, start, end = segment
            return {
                "session_id": str(session_id) if session_id is not None else default_session,
                "speaker": str(speaker),
                "start": float(start),
                "end": float(end),
            }
        if len(segment) == 3:
            speaker, start, end = segment
            return {
                "session_id": default_session,
                "speaker": str(speaker),
                "start": float(start),
                "end": float(end),
            }

    start = _get(("start", "start_time", "begin", "begin_time", "offset"))
    end = _get(("end", "end_time", "stop"))
    duration = _get(("duration", "dur"))
    speaker = _get(("speaker", "speaker_id", "label"))
    session_id = _get(("session_id", "recording_id", "file_id", "example_id"))

    if end is None and start is not None and duration is not None:
        end = float(start) + float(duration)

    if speaker is None or start is None or end is None:
        raise ValueError(f"Could not parse diarization segment: {segment!r}")

    return {
        "session_id": str(session_id) if session_id is not None else default_session,
        "speaker": str(speaker),
        "start": float(start),
        "end": float(end),
    }


def _load_diarization_segments(
    diarization: Union[str, Sequence[str]],
    diarization_format: Optional[str] = None,
    session_id: Optional[str] = None,
    time_concat: bool = False,
    concat_gap_seconds: float = 0.0,
    diarization_offsets: Optional[Sequence[Union[int, float]]] = None,
) -> List[Dict[str, Any]]:
    """Load and normalize diarization segments via ``meeteval.io.load``."""
    meeteval_io = _import_meeteval_io()
    kwargs = {}
    if diarization_format is not None:
        kwargs["format"] = diarization_format

    if concat_gap_seconds < 0:
        raise ValueError("concat_gap_seconds must be >= 0.")

    if isinstance(diarization, str):
        paths = [diarization]
    elif isinstance(diarization, Sequence) and not isinstance(diarization, (bytes, bytearray)):
        paths = list(diarization)
        if not paths:
            raise ValueError("diarization sequence must not be empty.")
        if not all(isinstance(path, str) for path in paths):
            raise TypeError("diarization entries must be file path strings.")
    else:
        raise TypeError("diarization must be str or sequence of file paths.")

    if diarization_offsets is not None:
        if len(diarization_offsets) != len(paths):
            raise ValueError(
                "diarization_offsets length must match number of diarization files. "
                f"Got {len(diarization_offsets)} offsets for {len(paths)} files."
            )
        if time_concat:
            raise ValueError("Use either time_concat=True or diarization_offsets, not both.")

    segments: List[Dict[str, Any]] = []
    running_offset = 0.0
    for idx, path in enumerate(paths):
        loaded = meeteval_io.load(path, **kwargs)

        cur_segments: List[Dict[str, Any]] = []
        if isinstance(loaded, dict):
            for loaded_session, loaded_segments in loaded.items():
                for segment in loaded_segments:
                    cur_segments.append(_segment_to_dict(segment, default_session=str(loaded_session)))
        else:
            for segment in loaded:
                cur_segments.append(_segment_to_dict(segment))

        if session_id is not None:
            cur_segments = [s for s in cur_segments if s["session_id"] == session_id]

        shift = 0.0
        if diarization_offsets is not None:
            shift = float(diarization_offsets[idx])
        elif time_concat:
            shift = running_offset

        if shift != 0.0:
            cur_segments = [
                {
                    **s,
                    "start": float(s["start"]) + shift,
                    "end": float(s["end"]) + shift,
                }
                for s in cur_segments
            ]

        segments.extend(cur_segments)

        if time_concat:
            if cur_segments:
                local_end = max(float(s["end"]) - shift for s in cur_segments)
            else:
                local_end = 0.0
            running_offset += local_end + concat_gap_seconds

    segments = [s for s in segments if s["end"] > s["start"]]
    segments.sort(key=lambda s: (s["start"], s["end"], s["speaker"]))

    if not segments:
        if session_id is None:
            raise ValueError("No valid diarization segments were loaded.")
        raise ValueError(
            f"No valid diarization segments were found for session_id='{session_id}'."
        )
    return segments


def _interval_to_dict(interval: Any, default_session: Optional[str] = None) -> Dict[str, Any]:
    """Normalize one interval object to a dict with ``session_id``, ``start``, ``end``."""

    def _get(names):
        if isinstance(interval, dict):
            for name in names:
                if name in interval:
                    return interval[name]
            return None
        for name in names:
            if hasattr(interval, name):
                return getattr(interval, name)
        return None

    if isinstance(interval, (tuple, list)):
        if len(interval) == 3:
            session_id, start, end = interval
            return {
                "session_id": str(session_id) if session_id is not None else default_session,
                "start": float(start),
                "end": float(end),
            }
        if len(interval) == 2:
            start, end = interval
            return {
                "session_id": default_session,
                "start": float(start),
                "end": float(end),
            }

    start = _get(("start", "start_time", "begin", "offset", "begin_time"))
    end = _get(("end", "end_time", "stop"))
    duration = _get(("duration", "dur"))
    session_id = _get(("session_id", "recording_id", "file_id", "example_id", "filename"))

    if end is None and start is not None and duration is not None:
        end = float(start) + float(duration)

    if start is None or end is None:
        raise ValueError(f"Could not parse interval: {interval!r}")

    return {
        "session_id": str(session_id) if session_id is not None else default_session,
        "start": float(start),
        "end": float(end),
    }


def _load_uem_regions(
    uem: str,
    uem_format: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load and normalize UEM regions via ``meeteval.io.load``."""
    meeteval_io = _import_meeteval_io()
    kwargs = {}
    if uem_format is not None:
        kwargs["format"] = uem_format
    loaded = meeteval_io.load(uem, **kwargs)

    regions: List[Dict[str, Any]] = []
    if isinstance(loaded, dict):
        for loaded_session, loaded_regions in loaded.items():
            for region in loaded_regions:
                regions.append(_interval_to_dict(region, default_session=str(loaded_session)))
    else:
        for region in loaded:
            regions.append(_interval_to_dict(region))

    if session_id is not None:
        regions = [r for r in regions if r["session_id"] == session_id]

    regions = [r for r in regions if r["end"] > r["start"]]
    regions.sort(key=lambda r: (r["start"], r["end"]))
    if not regions:
        if session_id is None:
            raise ValueError("No valid UEM regions were loaded.")
        raise ValueError(f"No valid UEM regions were found for session_id='{session_id}'.")
    return regions


def _find_valid_region_for_segment(
    segment: Dict[str, Any],
    valid_regions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a valid region that fully contains *segment* (or None)."""
    eps = 1e-8
    candidates = [
        region
        for region in valid_regions
        if segment["start"] >= region["start"] - eps and segment["end"] <= region["end"] + eps
    ]
    if not candidates:
        return None
    # Prefer the widest allowed region to maximize available context.
    return max(candidates, key=lambda region: region["end"] - region["start"])


def _load_valid_regions_arg(
    valid_regions: Union[Sequence[Any], Dict[str, Any]],
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Normalize user-provided valid regions argument.

    Supported shapes:
    - ``[(start, end), ...]``
    - ``[{"start": ..., "end": ...}, ...]``
    - ``{session_id: [(start, end), ...], ...}``
    """
    regions: List[Dict[str, Any]] = []

    if isinstance(valid_regions, dict):
        for region_session, region_values in valid_regions.items():
            default_session = str(region_session)
            if isinstance(region_values, (list, tuple)):
                # Either one interval tuple/list or a list of interval items.
                if len(region_values) == 2 and not isinstance(region_values[0], (dict, list, tuple)):
                    regions.append(_interval_to_dict(region_values, default_session=default_session))
                else:
                    for region in region_values:
                        regions.append(_interval_to_dict(region, default_session=default_session))
            else:
                regions.append(_interval_to_dict(region_values, default_session=default_session))
    else:
        for region in valid_regions:
            regions.append(_interval_to_dict(region, default_session=session_id))

    if session_id is not None:
        regions = [r for r in regions if r["session_id"] == session_id]

    regions = [r for r in regions if r["end"] > r["start"]]
    regions.sort(key=lambda r: (r["start"], r["end"]))
    if not regions:
        if session_id is None:
            raise ValueError("No valid regions were provided.")
        raise ValueError(f"No valid regions were found for session_id='{session_id}'.")
    return regions


def _intersect_valid_regions(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return intersection of two valid-region lists."""
    intersections: List[Dict[str, Any]] = []
    for a in left:
        for b in right:
            if a.get("session_id") != b.get("session_id"):
                continue
            start = max(a["start"], b["start"])
            end = min(a["end"], b["end"])
            if end > start:
                intersections.append(
                    {
                        "session_id": a.get("session_id"),
                        "start": float(start),
                        "end": float(end),
                    }
                )
    intersections.sort(key=lambda r: (r["start"], r["end"]))
    return intersections


def _validate_channel_length_mode(mode: str) -> str:
    """Validate channel length mismatch handling mode."""
    valid = ("error", "trim", "pad")
    if mode not in valid:
        raise ValueError(f"channel_length_mode must be one of {valid}, got '{mode}'.")
    return mode


def _validate_offset_unit(unit: str) -> str:
    """Validate per-channel offset unit."""
    valid = ("samples", "seconds")
    if unit not in valid:
        raise ValueError(f"channel_offset_unit must be one of {valid}, got '{unit}'.")
    return unit


def _match_channel_lengths(
    channels: List[np.ndarray],
    channel_length_mode: str,
) -> List[np.ndarray]:
    """Match channel lengths according to requested mode."""
    lengths = [ch.shape[-1] for ch in channels]
    if len(set(lengths)) == 1:
        return channels

    if channel_length_mode == "error":
        raise ValueError(
            "Channel lengths do not match across audio inputs: "
            f"{lengths}. Set channel_length_mode='trim' or 'pad'."
        )

    if channel_length_mode == "trim":
        target_len = min(lengths)
        return [ch[:target_len] for ch in channels]

    target_len = max(lengths)
    padded = []
    for ch in channels:
        cur_len = ch.shape[-1]
        if cur_len < target_len:
            ch = np.pad(ch, (0, target_len - cur_len), mode="constant")
        padded.append(ch)
    return padded


def _apply_channel_offsets(
    channels: List[np.ndarray],
    channel_offsets: Sequence[Union[int, float]],
    sample_rate: int,
    channel_offset_unit: str,
) -> List[np.ndarray]:
    """Apply per-channel temporal shifts.

    Positive offset delays a channel (prepends zeros). Negative offset advances
    a channel (drops leading samples).
    """
    _validate_offset_unit(channel_offset_unit)
    if len(channel_offsets) != len(channels):
        raise ValueError(
            "channel_offsets length must match total channel count. "
            f"Got {len(channel_offsets)} offsets for {len(channels)} channels."
        )

    shifted = []
    for idx, (ch, raw_offset) in enumerate(zip(channels, channel_offsets)):
        if isinstance(raw_offset, bool):
            raise TypeError("channel_offsets entries must be int/float, not bool.")
        if channel_offset_unit == "samples":
            offset_samples = int(raw_offset)
        else:
            offset_samples = int(round(float(raw_offset) * sample_rate))

        if offset_samples > 0:
            ch_shifted = np.pad(ch, (offset_samples, 0), mode="constant")
        elif offset_samples < 0:
            trim = -offset_samples
            ch_shifted = ch[trim:] if trim < ch.shape[-1] else np.zeros((0,), dtype=ch.dtype)
        else:
            ch_shifted = ch

        if ch_shifted.ndim != 1:
            raise ValueError(f"Invalid channel shape after offset on channel {idx}: {ch_shifted.shape}")
        shifted.append(ch_shifted)
    return shifted


def _load_audio_channels(
    audio_path: Union[str, Sequence[str]],
    channel_length_mode: str = "error",
    channel_offsets: Optional[Sequence[Union[int, float]]] = None,
    channel_offset_unit: str = "samples",
) -> Tuple[np.ndarray, int]:
    """Load audio from a single file or multiple channel files.

    Args:
        audio_path:
            - ``str``: one audio file, can be mono or multi-channel.
            - sequence of ``str``: one or more files. All channels from all files
              are concatenated on the channel axis.
        channel_length_mode:
            Behavior when loaded channel lengths do not match.
            - ``'error'``: raise ValueError
            - ``'trim'``: trim all channels to the shortest length
            - ``'pad'``: zero-pad shorter channels to the longest length
        channel_offsets:
            Optional per-channel temporal shifts. Length must match total
            channel count after loading all files.
        channel_offset_unit:
            Unit for ``channel_offsets``: ``'samples'`` or ``'seconds'``.

    Returns:
        Tuple ``(audio, sample_rate)`` where ``audio`` has shape
        ``(channels, samples)`` and dtype ``float32``.
    """
    import torchaudio

    channel_length_mode = _validate_channel_length_mode(channel_length_mode)
    channel_offset_unit = _validate_offset_unit(channel_offset_unit)

    if isinstance(audio_path, str):
        audio_t, sample_rate = torchaudio.load(audio_path)
        channels_list = [ch.numpy().astype(np.float32) for ch in audio_t]
        channels_list = _match_channel_lengths(channels_list, channel_length_mode)
        if channel_offsets is not None:
            channels_list = _apply_channel_offsets(
                channels=channels_list,
                channel_offsets=channel_offsets,
                sample_rate=int(sample_rate),
                channel_offset_unit=channel_offset_unit,
            )
            channels_list = _match_channel_lengths(channels_list, channel_length_mode)
        return np.stack(channels_list, axis=0), int(sample_rate)

    if not isinstance(audio_path, Sequence) or isinstance(audio_path, (bytes, bytearray)):
        raise TypeError("audio_path must be str or a sequence of file paths.")

    paths = list(audio_path)
    if not paths:
        raise ValueError("audio_path sequence must not be empty.")

    channels: List[np.ndarray] = []
    sample_rate_val: Optional[int] = None

    for path in paths:
        if not isinstance(path, str):
            raise TypeError("audio_path entries must be file path strings.")
        wav_t, sr = torchaudio.load(path)
        wav = wav_t.numpy().astype(np.float32)
        if sample_rate_val is None:
            sample_rate_val = int(sr)
        elif int(sr) != sample_rate_val:
            raise ValueError(
                f"All audio files must have the same sample rate. "
                f"Expected {sample_rate_val}, got {int(sr)} for '{path}'."
            )
        for ch in wav:
            channels.append(ch)

    assert sample_rate_val is not None

    channels = _match_channel_lengths(channels, channel_length_mode)
    if channel_offsets is not None:
        channels = _apply_channel_offsets(
            channels=channels,
            channel_offsets=channel_offsets,
            sample_rate=sample_rate_val,
            channel_offset_unit=channel_offset_unit,
        )
        channels = _match_channel_lengths(channels, channel_length_mode)

    audio = np.stack(channels, axis=0).astype(np.float32)
    return audio, sample_rate_val


def _build_activity_from_diarization(
    segments: List[Dict[str, Any]],
    speakers: List[str],
    num_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """Build sample-domain activity matrix from normalized diarization segments."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    activity = np.zeros((len(speakers), num_samples), dtype=np.float32)
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(speakers)}

    for segment in segments:
        speaker = segment["speaker"]
        if speaker not in speaker_to_idx:
            continue
        start = int(round(float(segment["start"]) * sample_rate))
        end = int(round(float(segment["end"]) * sample_rate))
        start = max(0, min(num_samples, start))
        end = max(0, min(num_samples, end))
        if end <= start:
            continue
        activity[speaker_to_idx[speaker], start:end] = 1.0

    return activity


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GSS:
    """NeMo-based GSS (Guided Source Separation) front-end.

    Processes one utterance at a time.  All NeMo sub-modules run on GPU.

    Typical usage::

        frontend = GSS()

        # audio:    numpy array (channels, samples), float32
        # activity: numpy array (speakers, samples), bool or float {0,1}
        # speaker_id: 0-based index of the target speaker in `activity`
        enhanced = frontend.enhance(audio, activity, speaker_id=0)
        # enhanced: numpy array (samples,), float32

    Parameters
    ----------
    stft_fft_length : int
        FFT length for STFT analysis/synthesis (default 1024).
    stft_hop_length : int
        Hop length for STFT (default 256).
    enable_dereverb : bool
        If True (default), apply WPE dereverberation. If False, skip dereverberation.
    dereverb_prediction_delay : int
        WPE prediction delay (default 2).
    dereverb_filter_length : int
        WPE filter length (default 10).
    dereverb_num_iterations : int
        WPE iterations (default 3).
    bss_iterations : int
        GSS / FastMNMF iterations (default 20).
    mc_filter_type : str
        Multichannel filter type, e.g. ``'pmwf'`` (default).
    mc_filter_beta : float
        Beta parameter for the multichannel filter (default 0).
    mc_filter_rank : str
        Rank of the multichannel filter, ``'one'`` or ``'full'`` (default ``'one'``).
    mc_filter_postfilter : str
        Post-filter, e.g. ``'ban'`` (default).
    mc_ref_channel : str, int, or None
        Reference channel selection strategy.
        - ``'max_snr'`` (default): Auto-select best channel by SNR.
        - ``int``: Use fixed channel index (0-based).
        - ``None``: No channel selection (output all channels, MIMO mode).
    mc_mask_min_db : float
        Minimum mask value in dB for the multichannel filter (default -200).
    mc_postmask_min_db : float
        Minimum post-mask value in dB (default 0, i.e. no post-masking).
    activity_aggregation : str
        How to aggregate sample-level activity into frame-level activity.
        One of ``'mean'`` (default), ``'max'``, or ``'any'``.
    garbage_class : bool
        If True (default), append one extra background/garbage activity class
        so GSS runs with ``n_speakers + 1`` classes when using enhance() or estimate_masks().
        This option does NOT apply to estimate_unguided(), which always uses exactly
        num_sources classes without garbage class.
    use_dtype : torch.dtype
        Complex dtype used internally (default ``torch.cfloat``).
    device : str or torch.device
        Device to run on (default ``'cuda'``).
    """

    def __init__(
        self,
        stft_fft_length: int = 1024,
        stft_hop_length: int = 256,
        enable_dereverb: bool = True,
        dereverb_prediction_delay: int = 2,
        dereverb_filter_length: int = 10,
        dereverb_num_iterations: int = 3,
        bss_iterations: int = 20,
        mc_filter_type: str = "pmwf",
        mc_filter_beta: float = 0,
        mc_filter_rank: str = "one",
        mc_filter_postfilter: str = "ban",
        mc_ref_channel: str = "max_snr",
        mc_mask_min_db: float = -200,
        mc_postmask_min_db: float = 0,
        garbage_class: bool = True,
        activity_aggregation: str = "mean",
        use_dtype: torch.dtype = torch.cfloat,
        device: str = "cuda",
    ):
        self.fft_length = stft_fft_length
        self.hop_length = stft_hop_length
        self.device = torch.device(device)
        self.enable_dereverb = enable_dereverb
        if activity_aggregation not in ("mean", "max", "any"):
            raise ValueError(
                f"activity_aggregation='{activity_aggregation}' is not supported. "
                "Use one of: 'mean', 'max', 'any'."
            )
        self.activity_aggregation = activity_aggregation
        self.garbage_class = garbage_class

        self.analysis = AudioToSpectrogram(
            fft_length=stft_fft_length, hop_length=stft_hop_length
        ).to(self.device)
        self.synthesis = SpectrogramToAudio(
            fft_length=stft_fft_length, hop_length=stft_hop_length
        ).to(self.device)
        self.dereverb: Optional[MaskBasedDereverbWPE]
        if enable_dereverb:
            self.dereverb = MaskBasedDereverbWPE(
                filter_length=dereverb_filter_length,
                prediction_delay=dereverb_prediction_delay,
                num_iterations=dereverb_num_iterations,
                dtype=use_dtype,
            ).to(self.device)
        else:
            self.dereverb = None
        self.gss = MaskEstimatorGSS(
            num_iterations=bss_iterations, dtype=use_dtype
        ).to(self.device)
        self.mc = MaskBasedBeamformer(
            filter_type=mc_filter_type,
            filter_beta=mc_filter_beta,
            filter_rank=mc_filter_rank,
            filter_postfilter=mc_filter_postfilter,
            ref_channel=mc_ref_channel,
            mask_min_db=mc_mask_min_db,
            postmask_min_db=mc_postmask_min_db,
        ).to(self.device)

        logger.info("Initialized GSS on device=%s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        activity: Union[np.ndarray, torch.Tensor],
        speaker_id: Union[int, List[int]],
        left_context: int = 0,
        right_context: int = 0,
        num_chunks: int = 1,
        return_dict: bool = False,
        garbage_class: Optional[bool] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Enhance a single utterance.

        Supports both single speaker and multiple speakers enhancement.
        When multiple speakers are requested, each speaker is processed
        independently through the beamformer, and the results are returned as a list.

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor, shape (channels, samples)
            Multi-channel waveform (float32 recommended).  When a
            ``torch.Tensor`` is provided gradients are preserved and the
            output is also a ``torch.Tensor``.
        activity : np.ndarray or torch.Tensor, shape (speakers, samples)
            Speaker activity. Each row corresponds to one speaker; values may
            be binary {0, 1} or soft confidences in [0, 1].
        speaker_id : int or List[int]
            Index (0-based) of the target speaker row in ``activity``.
            - If int: single speaker enhancement
            - If List[int]: multiple speakers enhancement (results will be returned as list)
        left_context : int
            Number of leading samples that are context (will be dropped from
            the output). Typically set when the input ``audio`` was extended
            to the left before calling this function.
        right_context : int
            Number of trailing context samples (will be dropped from output).
        num_chunks : int
            Split the frequency axis into this many chunks to reduce peak GPU
            memory usage. Use ``auto`` or increase manually when you hit OOM.
        return_dict : bool
            If False (default), return enhanced audio only. If True, return dict with
            'audio', 'masks', 'eigenvalues', 'mahalanobis', 'occupancy',
            'temporal_variance', and 'condition_number' for external classification.
        garbage_class : bool or None
            If True, append one extra background/garbage activity class.
            If False, do not append garbage class.
            If None (default), use self.garbage_class.

        Returns
        -------
        np.ndarray or torch.Tensor or Dict
            If return_dict=False (default):
                Enhanced waveform(s), float32. Shape depends on speaker_id and beamformer mode:
                Single speaker (speaker_id is int):
                - Single-channel mode: shape (samples_out,)
                - Multi-channel MIMO mode: shape (num_channels, samples_out)
                Multiple speakers (speaker_id is List[int]):
                - Single-channel mode: shape (num_speakers, samples_out)
                - Multi-channel MIMO mode: shape (num_speakers, num_channels, samples_out)
                Same type as *audio*.
                ``samples_out = len(audio[0]) - left_context - right_context``.

            If return_dict=True:
                Dict with keys:
                - 'audio': enhanced waveform(s) (as above)
                - 'masks': (num_sources, freq, frames) source masks [0, 1]
                - 'eigenvalues': (num_sources, freq, num_channels) eigenvalue statistics
                - 'mahalanobis': (num_sources, freq, frames) Mahalanobis distances
                - 'occupancy': (num_sources,) time-averaged mask values [0, 1]
                - 'temporal_variance': (num_sources,) temporal variance per source
                - 'condition_number': (num_sources, freq) eigenvalue ratio per freq
        """
        audio_t, is_numpy = _prepare_audio(audio, self.device)
        activity_t = _prepare_activity(activity, self.device)
        
        # Use provided garbage_class or default to self.garbage_class
        use_garbage_class = garbage_class if garbage_class is not None else self.garbage_class

        left_context_frames = samples_to_frames(
            left_context, self.fft_length, self.hop_length
        )
        right_context_frames = samples_to_frames(
            right_context, self.fft_length, self.hop_length
        )

        ctx = torch.inference_mode() if is_numpy else contextlib.nullcontext()
        with ctx:
            # Get enhanced audio and optionally statistics
            enhance_result = self._enhance_tensor(
                audio_t,
                activity_t,
                speaker_id=speaker_id,
                left_context_frames=left_context_frames,
                right_context_frames=right_context_frames,
                num_chunks=num_chunks,
                return_dict=return_dict,
                garbage_class=use_garbage_class,
            )
            
            stats_dict: Optional[Dict[str, Any]] = None
            if return_dict:
                assert isinstance(enhance_result, dict), "enhance_result must be dict when return_dict=True"
                stats_dict = cast(Dict[str, Any], enhance_result)
                result = stats_dict['audio']
            else:
                result = enhance_result

        # Drop context from time domain
        is_multi_speaker = isinstance(speaker_id, list)

        if is_multi_speaker:
            # Multiple speakers: list of tensors
            # Each tensor is either (samples,) in single-channel mode or (num_channels, samples) in MIMO mode
            result_list = cast(List[Any], result)
            result_trimmed: List[Any] = []
            for output in result_list:
                if output.dim() == 1:
                    # Single-channel mode: (samples,)
                    trimmed = output[left_context:]
                    if right_context > 0:
                        trimmed = trimmed[:-right_context]
                else:
                    # MIMO mode: (num_channels, samples)
                    trimmed = output[:, left_context:]
                    if right_context > 0:
                        trimmed = trimmed[:, :-right_context]
                result_trimmed.append(trimmed)
            result = result_trimmed
        else:
            # Single speaker
            result_tensor = cast(torch.Tensor, result)
            if result_tensor.dim() == 1:
                # Single-channel mode: (samples,)
                result = result_tensor[left_context:]
                if right_context > 0:
                    result = result[:-right_context]
            else:
                # MIMO mode: (num_channels, samples)
                result = result_tensor[:, left_context:]
                if right_context > 0:
                    result = result[:, :-right_context]

        if is_numpy:
            if is_multi_speaker:
                # Convert list of tensors to list of numpy arrays
                result_list_final = cast(List[torch.Tensor], result)
                result = [r.detach().cpu().numpy() for r in result_list_final]
            else:
                # Convert single tensor to numpy
                result_tensor_final = cast(torch.Tensor, result)
                result = result_tensor_final.detach().cpu().numpy()
        
        if return_dict:
            assert stats_dict is not None, "stats_dict must be set when return_dict=True"
            return {
                'audio': result,
                'masks': stats_dict['masks'],
                'eigenvalues': stats_dict['eigenvalues'],
                'mahalanobis': stats_dict['mahalanobis'],
                'occupancy': stats_dict['occupancy'],
                'temporal_variance': stats_dict['temporal_variance'],
                'condition_number': stats_dict['condition_number'],
            }
        return result

    def enhance_auto(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        activity: Union[np.ndarray, torch.Tensor],
        speaker_id: int,
        left_context: int = 0,
        right_context: int = 0,
        return_dict: bool = False,
        garbage_class: Optional[bool] = None,
    ) -> Union[np.ndarray, torch.Tensor, Dict[str, torch.Tensor]]:
        """Same as :meth:`enhance` but retries with finer chunking on OOM.

        Returns the same type as *audio* (``np.ndarray`` or ``torch.Tensor``).

        Automatically splits the frequency axis into more chunks when a CUDA
        out-of-memory error occurs.  The chunk counts used are all integer
        divisors of ``fft_length // 2 + 1`` in ascending order.
        """
        import decimal

        audio_t, is_numpy = _prepare_audio(audio, self.device)
        activity_t = _prepare_activity(activity, self.device)

        left_context_frames = samples_to_frames(
            left_context, self.fft_length, self.hop_length
        )
        right_context_frames = samples_to_frames(
            right_context, self.fft_length, self.hop_length
        )

        num_chunks_list = _get_int_divisors(self.fft_length // 2 + 1)

        ctx = torch.inference_mode() if is_numpy else contextlib.nullcontext()
        result = None
        for num_chunks in num_chunks_list:
            try:
                with ctx:
                    result = self._enhance_tensor(
                        audio_t,
                        activity_t,
                        speaker_id=speaker_id,
                        left_context_frames=left_context_frames,
                        right_context_frames=right_context_frames,
                        num_chunks=num_chunks,
                        return_dict=return_dict,
                        garbage_class=garbage_class,
                    )
                break  # success
            except (RuntimeError, decimal.InvalidOperation) as exc:
                if isinstance(exc, RuntimeError) and not _is_cuda_oom_error(exc):
                    raise
                logger.warning(
                    "OOM with num_chunks=%d, retrying with more chunks.", num_chunks
                )
                torch.cuda.empty_cache()
                continue

        if result is None:
            logger.warning(
                "All GPU chunk sizes exhausted. Retrying with per-stage CPU fallback."
            )
            try:
                with ctx:
                    result = self._enhance_tensor(
                        audio_t,
                        activity_t,
                        speaker_id=speaker_id,
                        left_context_frames=left_context_frames,
                        right_context_frames=right_context_frames,
                        num_chunks=1,
                        cpu_fallback=True,
                    )
            except Exception:
                logger.exception("CPU fallback also failed. Falling back to channel 0.")
                result = audio_t[0, 0]
                if is_numpy:
                    return result.detach().cpu().numpy()
                return result

        # Handle return_dict case - return early without context dropping on audio
        if return_dict:
            # result is dict when return_dict=True
            result_dict = cast(Dict[str, Any], result)
            audio_output = result_dict['audio']
            if torch.is_tensor(audio_output):
                audio_tensor = cast(torch.Tensor, audio_output)
                if audio_tensor.dim() == 1:
                    # Single-channel mode: (samples,)
                    audio_tensor = audio_tensor[left_context:]
                    if right_context > 0:
                        audio_tensor = audio_tensor[:-right_context]
                else:
                    # MIMO mode: (num_channels, samples)
                    audio_tensor = audio_tensor[:, left_context:]
                    if right_context > 0:
                        audio_tensor = audio_tensor[:, :-right_context]
                result_dict['audio'] = audio_tensor
                if is_numpy:
                    # Convert audio to numpy but keep stats as torch tensors
                    result_dict['audio'] = audio_tensor.detach().cpu().numpy()
            return result_dict

        # Drop context from time domain - handle both STANDARD and MIMO modes
        # result is tensor when return_dict=False (speaker_id is always int in enhance_auto)
        result_tensor = cast(torch.Tensor, result)
        if result_tensor.dim() == 1:
            # Single-channel mode: (samples,)
            result = result_tensor[left_context:]
            if right_context > 0:
                result = result[:-right_context]
        else:
            # MIMO mode: (num_channels, samples)
            result = result_tensor[:, left_context:]
            if right_context > 0:
                result = result[:, :-right_context]
        
        if is_numpy:
            return result.detach().cpu().numpy()
        return result

    def estimate_masks(
        self,
        audio: np.ndarray,
        activity: np.ndarray,
        garbage_class: Optional[bool] = None,
    ) -> "Union[np.ndarray, torch.Tensor]":
        """Estimate time-frequency masks via GSS (cACGMM).

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor, shape (channels, samples)
            Multi-channel waveform (float32 recommended).  When a
            ``torch.Tensor`` is provided gradients are preserved and the
            output is also a ``torch.Tensor`` (useful for training).
        activity : np.ndarray or torch.Tensor, shape (speakers, samples)
            Speaker activity, binary {0, 1} or soft confidences in [0, 1].
        garbage_class : bool or None
            If True, append one extra background/garbage activity class.
            If False, do not append garbage class.
            If None (default), use self.garbage_class.

        Returns
        -------
        np.ndarray or torch.Tensor, shape (speakers, freq, frames)
            Soft time-frequency masks, values in [0, 1].  Same type as *audio*.
            ``shape = (speakers + 1, freq, frames)`` when ``self.garbage_class=True``,
            else ``(speakers, freq, frames)``.
        """
        audio_t, is_numpy = _prepare_audio(audio, self.device)
        activity_t = _prepare_activity(activity, self.device)
        
        # Use provided garbage_class or default to self.garbage_class
        use_garbage_class = garbage_class if garbage_class is not None else self.garbage_class
        activity_t = _append_garbage_activity_class(activity_t, use_garbage_class)
        ctx = torch.inference_mode() if is_numpy else contextlib.nullcontext()
        with ctx:
            x_enc, _ = self.analysis(input=audio_t)        # (1, ch, freq, frames)
            a_enc = activity_time_to_timefreq(
                activity_t,
                win_length=self.fft_length,
                hop_length=self.hop_length,
                aggregation=self.activity_aggregation,
            )
            masks = self.gss(x_enc, a_enc)                 # (1, spk, freq, frames) tensor
        if is_numpy:
            return masks[0].detach().cpu().numpy()
        return masks[0]

    def enhance_unguided(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        num_sources: int,
        left_context: int = 0,
        right_context: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Enhance multi-channel audio without speaker activity guidance (blind BSS).

        Performs blind source separation using uniform activity assumption across
        all sources, returning statistics for external classification (speech vs noise).

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor, shape (channels, samples)
            Multi-channel waveform (float32 recommended).
        num_sources : int
            Total number of sources (speakers + noise). Activity initialized uniformly.
        left_context : int
            Number of leading samples that are context (will be dropped).
        right_context : int
            Number of trailing context samples (will be dropped).

        Returns
        -------
        dict with keys:
            - 'masks': (num_sources, freq, frames) source masks [0, 1]
            - 'eigenvalues': (num_sources, freq, num_channels) eigenvalue statistics
            - 'mahalanobis': (num_sources, freq, frames) Mahalanobis distances
            - 'occupancy': (num_sources,) time-averaged mask values [0, 1]
            - 'temporal_variance': (num_sources,) temporal variance per source
            - 'condition_number': (num_sources, freq) eigenvalue ratio per freq
        """
        num_samples = audio.shape[-1]
        # Create uniform activity for all sources
        activity = np.ones((num_sources, num_samples), dtype=np.float32) / num_sources
        
        # Use speaker_id=0 (arbitrary choice) with return_dict=True, and garbage_class=False for blind BSS
        result = self.enhance(
            audio=audio,
            activity=activity,
            speaker_id=0,
            left_context=left_context,
            right_context=right_context,
            return_dict=True,
            garbage_class=False,
        )
        
        # Return full result dict including enhanced audio and statistics
        return cast(Dict[str, torch.Tensor], result)

    def enhance_unguided_auto(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        num_sources: int,
        speaker_id: int = 0,
        left_context: int = 0,
        right_context: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Blind source separation using enhance_auto() with OOM-aware chunking.

        Performs blind source separation using uniform activity assumption, 
        with automatic out-of-memory handling via enhance_auto()'s retry logic.

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor, shape (channels, samples)
            Multi-channel waveform (float32 recommended).
        num_sources : int
            Total number of sources (speakers + noise). Activity initialized uniformly.
        speaker_id : int
            Dummy speaker index (default 0). Not used for blind BSS but required 
            by enhance_auto() interface.
        left_context : int
            Number of leading samples that are context (will be dropped).
        right_context : int
            Number of trailing context samples (will be dropped).

        Returns
        -------
        dict with keys:
            - 'audio': separated audio (num_sources, samples)
            - 'masks': (num_sources, freq, frames) source masks [0, 1]
            - 'eigenvalues': (num_sources, freq, num_channels) eigenvalue statistics
            - 'mahalanobis': (num_sources, freq, frames) Mahalanobis distances
            - 'occupancy': (num_sources,) time-averaged mask values [0, 1]
            - 'temporal_variance': (num_sources,) temporal variance per source
            - 'condition_number': (num_sources, freq) eigenvalue ratio per freq
        """
        num_samples = audio.shape[-1]
        
        # Create uniform activity for all sources
        activity = np.ones((num_sources, num_samples), dtype=np.float32) / num_sources
        
        # Use enhance_auto() for OOM-aware chunking with uniform activity and garbage_class=False
        result = self.enhance_auto(
            audio=audio,
            activity=activity,
            speaker_id=speaker_id,
            left_context=left_context,
            right_context=right_context,
            return_dict=True,
            garbage_class=False,
        )
        
        # Return full result dict including enhanced audio and statistics
        return cast(Dict[str, torch.Tensor], result)

    def enhance_segment(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        activity: Union[np.ndarray, torch.Tensor],
        speaker_id: int,
        segment_start: Union[int, float],
        segment_end: Union[int, float],
        sample_rate: int,
        context_left_seconds: float = 15.0,
        context_right_seconds: float = 15.0,
        segment_unit: str = "seconds",
        num_chunks: int = 1,
        mode: str = "standard",
    ) -> Union[np.ndarray, torch.Tensor]:
        """Enhance one target segment from long-form audio with context.

        This API is intended for the common long-recording workflow where GSS
        mask estimation is run on ``[left context] + [target segment] +
        [right context]``, then only the target segment is returned.

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor, shape (channels, samples)
            Multi-channel long-form waveform.
        activity : np.ndarray or torch.Tensor, shape (speakers, samples)
            Speaker activity for the same full recording.
        speaker_id : int
            Target speaker index in ``activity``.
        segment_start, segment_end : int or float
            Target segment boundaries. Interpreted as seconds when
            ``segment_unit='seconds'`` and as sample indices when
            ``segment_unit='samples'``.
        sample_rate : int
            Sampling rate used for seconds-to-samples conversion.
        context_left_seconds, context_right_seconds : float
            Left/right context duration in seconds (default: 15 s each).
        segment_unit : str
            ``'seconds'`` (default) or ``'samples'``.
        num_chunks : int
            Number of frequency chunks for memory control.
        mode : str
            Enhancement mode:
            - ``'standard'``: call :meth:`enhance` with ``num_chunks``
            - ``'oom_fallback'``: call :meth:`enhance_auto` (OOM-aware retry)

            Backward-compatible aliases are accepted:
            - ``'enhance'`` -> ``'standard'``
            - ``'auto'`` -> ``'oom_fallback'``

        Returns
        -------
        np.ndarray or torch.Tensor
            Enhanced single-channel waveform for the target segment only.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if context_left_seconds < 0 or context_right_seconds < 0:
            raise ValueError("context_left_seconds and context_right_seconds must be >= 0.")
        mode = _validate_segment_mode(mode)

        segment_start_samples = _to_sample_index(segment_start, sample_rate, segment_unit)
        segment_end_samples = _to_sample_index(segment_end, sample_rate, segment_unit)
        if segment_end_samples <= segment_start_samples:
            raise ValueError("segment_end must be greater than segment_start.")

        num_samples = audio.shape[-1]
        if segment_start_samples < 0 or segment_end_samples > num_samples:
            raise ValueError(
                f"segment range [{segment_start_samples}, {segment_end_samples}) is outside "
                f"audio length {num_samples}."
            )

        left_ctx_samples = int(round(context_left_seconds * sample_rate))
        right_ctx_samples = int(round(context_right_seconds * sample_rate))

        window_start = max(0, segment_start_samples - left_ctx_samples)
        window_end = min(num_samples, segment_end_samples + right_ctx_samples)

        left_context = segment_start_samples - window_start
        right_context = window_end - segment_end_samples

        audio_window = audio[..., window_start:window_end]
        activity_window = activity[..., window_start:window_end]

        if mode == "oom_fallback":
            return self.enhance_auto(
                audio=audio_window,
                activity=activity_window,
                speaker_id=speaker_id,
                left_context=left_context,
                right_context=right_context,
            )

        return self.enhance(
            audio=audio_window,
            activity=activity_window,
            speaker_id=speaker_id,
            left_context=left_context,
            right_context=right_context,
            num_chunks=num_chunks,
        )

    def enhance_from_diarization(
        self,
        audio_path: Union[str, Sequence[str]],
        diarization: Union[str, Sequence[str]],
        speaker_id: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        diarization_format: Optional[str] = None,
        diarization_session_id: Optional[str] = None,
        diarization_time_concat: bool = False,
        diarization_concat: Optional[bool] = None,
        diarization_concat_gap_seconds: float = 0.0,
        diarization_offsets: Optional[Sequence[Union[int, float]]] = None,
        uem: Optional[str] = None,
        uem_format: Optional[str] = None,
        valid_regions: Optional[Union[Sequence[Any], Dict[str, Any]]] = None,
        channel_length_mode: str = "error",
        channel_offsets: Optional[Sequence[Union[int, float]]] = None,
        channel_offset_unit: str = "samples",
        context_left_seconds: float = 15.0,
        context_right_seconds: float = 15.0,
        mode: str = "standard",
        num_chunks: int = 1,
        num_groups: int = 1,
        group_id: int = 0,
    ):
        """Enhance all diarized target-speaker utterances from a long recording.
        
        Yields
        ------
        dict
            Enhanced segment with keys: speaker, speaker_id, segment_index, 
            segment_start, segment_end, sample_rate, enhanced_audio

        Parameters
        ----------
                audio_path : str or sequence of str
                        Input audio source.
                        - ``str``: one long-form audio file (mono or multi-channel)
                        - sequence: multiple files (e.g., separate mono channels)
                            concatenated along channel axis.
        diarization : str or sequence of str
            Path(s) to diarization file(s) readable by ``meeteval.io.load``.
        speaker_id : int, str, sequence, or None
            Target speaker selector.
            - ``int``: index in lexicographically sorted unique speaker labels
            - ``str``: exact speaker label from diarization
            - sequence of ``int``/``str``: select multiple speakers
            - ``None`` (default): process all speakers
        diarization_format : str, optional
            Explicit meeteval format hint (e.g. ``'rttm'``).
        diarization_session_id : str, optional
            Session/recording ID filter when diarization contains multiple sessions.
        diarization_time_concat : bool
            When ``diarization`` is a sequence, concatenate files in order by
            shifting each subsequent file to the end of the previous one.
        diarization_concat : bool, optional
            Deprecated alias of ``diarization_time_concat``.
        diarization_concat_gap_seconds : float
            Optional gap inserted between concatenated diarization files.
        diarization_offsets : sequence of int/float, optional
            Explicit per-file time offsets (seconds) for diarization files.
            Use this instead of ``diarization_time_concat`` when exact offsets are
            already known.
        uem : str, optional
            Path to UEM file that defines valid scoring/enhancement regions.
            Segments outside UEM are excluded, and context is clipped to stay
            inside UEM boundaries.
        uem_format : str, optional
            Explicit meeteval format hint for UEM (e.g. ``'uem'``).
        valid_regions : sequence or dict, optional
            Valid time regions given directly as argument.
            Supported examples:
            - ``[(start, end), ...]``
            - ``[{"start": ..., "end": ...}, ...]``
            - ``{session_id: [(start, end), ...], ...}``
            Regions are interpreted in seconds and combined with ``uem`` when
            both are specified (intersection).
        channel_length_mode : str
            Length mismatch handling for multi-file ``audio_path``.
            - ``'error'`` (default): raise error
            - ``'trim'``: trim all channels to shortest length
            - ``'pad'``: zero-pad to longest length
        channel_offsets : sequence of int/float, optional
            Optional per-channel temporal shift values. Length must equal
            total channel count after loading ``audio_path``.
            Positive offset delays a channel, negative offset advances it.
        channel_offset_unit : str
            Unit of ``channel_offsets``: ``'samples'`` (default) or
            ``'seconds'``.
        context_left_seconds, context_right_seconds : float
            Left/right context for each segment enhancement.
        mode : str
            ``'standard'`` or ``'oom_fallback'``; same meaning as
            :meth:`enhance_segment`.
        num_chunks : int
            Frequency chunk count used when ``mode='standard'``.
        num_groups : int
            Number of groups to partition segments into for distributed processing.
            (default: 1 = no partitioning)
        group_id : int
            Zero-based group index to process when using partitioning.
            Must be 0 <= group_id < num_groups. (default: 0)
            Segments are partitioned with balanced total duration across groups.

        Returns
        -------
        list of dict
            One dict per selected segment (time order) with keys:
            ``speaker``, ``speaker_id``, ``segment_index``, ``segment_start``,
            ``segment_end``, ``sample_rate``, ``enhanced_audio``.
        """
        if diarization_concat is not None:
            logger.warning(
                "diarization_concat is deprecated and will be removed in a future release; "
                "use diarization_time_concat instead."
            )
            diarization_time_concat = bool(diarization_concat)

        segments = _load_diarization_segments(
            diarization=diarization,
            diarization_format=diarization_format,
            session_id=diarization_session_id,
            time_concat=diarization_time_concat,
            concat_gap_seconds=diarization_concat_gap_seconds,
            diarization_offsets=diarization_offsets,
        )

        allowed_regions = None
        if uem is not None:
            allowed_regions = _load_uem_regions(
                uem=uem,
                uem_format=uem_format,
                session_id=diarization_session_id,
            )

        if valid_regions is not None:
            direct_regions = _load_valid_regions_arg(
                valid_regions=valid_regions,
                session_id=diarization_session_id,
            )
            if allowed_regions is None:
                allowed_regions = direct_regions
            else:
                allowed_regions = _intersect_valid_regions(allowed_regions, direct_regions)

        if allowed_regions is not None:
            filtered_segments = []
            for segment in segments:
                if _find_valid_region_for_segment(segment, allowed_regions) is not None:
                    filtered_segments.append(segment)
            segments = filtered_segments
            if not segments:
                raise ValueError(
                    "No diarization segments remain after applying valid-region filtering."
                )

        speakers = sorted({segment["speaker"] for segment in segments})
        if not speakers:
            raise ValueError("No speakers found in diarization.")

        if speaker_id is None:
            selected_indices = list(range(len(speakers)))
        else:
            if isinstance(speaker_id, bool):
                raise TypeError("speaker_id must be int/str/sequence or None, not bool.")
            if isinstance(speaker_id, (int, str)):
                selectors = [speaker_id]
            elif isinstance(speaker_id, Sequence) and not isinstance(speaker_id, (bytes, bytearray)):
                selectors = list(speaker_id)
                if not selectors:
                    raise ValueError("speaker_id sequence must not be empty.")
            else:
                raise TypeError("speaker_id must be int, str, sequence of them, or None.")

            selected_indices = []
            for selector in selectors:
                if isinstance(selector, bool):
                    raise TypeError("speaker_id entries must be int or str, not bool.")
                if isinstance(selector, str):
                    if selector not in speakers:
                        raise ValueError(
                            f"Unknown speaker_id='{selector}'. Available speakers: {speakers}."
                        )
                    selected_indices.append(speakers.index(selector))
                elif isinstance(selector, int):
                    if selector < 0 or selector >= len(speakers):
                        raise ValueError(
                            f"speaker_id={selector} is out of range for {len(speakers)} speakers."
                        )
                    selected_indices.append(selector)
                else:
                    raise TypeError("speaker_id entries must be int or str.")

            # Deduplicate while preserving order.
            selected_indices = list(dict.fromkeys(selected_indices))

        selected_speakers = {speakers[idx] for idx in selected_indices}
        target_segments = [segment for segment in segments if segment["speaker"] in selected_speakers]
        if not target_segments:
            raise ValueError("No segments found for selected speaker(s).")

        # Partition segments for distributed processing
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        if group_id < 0 or group_id >= num_groups:
            raise ValueError(
                f"group_id must be in range [0, {num_groups}), got {group_id}"
            )

        # Log total statistics
        total_stats = _compute_group_statistics(target_segments, list(range(len(target_segments))))
        logger.info(
            "Total segments: %d, total duration: %.1f seconds (avg: %.1f s/seg)",
            total_stats["num_segments"],
            total_stats["total_duration_seconds"],
            total_stats["avg_duration_seconds"],
        )

        # Apply group partitioning if needed
        group_segment_indices = None  # Track global segment indices for consistent file naming
        if num_groups > 1:
            all_groups = _partition_segments_by_duration(target_segments, num_groups)
            group_segment_indices = all_groups[group_id]
            target_segments = [target_segments[i] for i in group_segment_indices]
            group_stats = _compute_group_statistics(target_segments, list(range(len(target_segments))))
            logger.info(
                "Processing group %d/%d: %d segments, %.1f seconds (avg: %.1f s/seg)",
                group_id + 1,
                num_groups,
                group_stats["num_segments"],
                group_stats["total_duration_seconds"],
                group_stats["avg_duration_seconds"],
            )
        else:
            # When not partitioning, use sequential indices
            group_segment_indices = list(range(len(target_segments)))
        
        if not target_segments:
            raise ValueError("No segments in the requested group.")

        audio, sample_rate = _load_audio_channels(
            audio_path=audio_path,
            channel_length_mode=channel_length_mode,
            channel_offsets=channel_offsets,
            channel_offset_unit=channel_offset_unit,
        )
        num_samples = audio.shape[-1]

        activity = _build_activity_from_diarization(
            segments=segments,
            speakers=speakers,
            num_samples=num_samples,
            sample_rate=sample_rate,
        )


        for idx, segment in enumerate(target_segments):
            target_speaker = segment["speaker"]
            target_idx = speakers.index(target_speaker)

            left_context_seconds_eff = context_left_seconds
            right_context_seconds_eff = context_right_seconds
            if allowed_regions is not None:
                allowed_region = _find_valid_region_for_segment(segment, allowed_regions)
                if allowed_region is None:
                    continue
                left_context_seconds_eff = min(
                    context_left_seconds,
                    max(0.0, segment["start"] - allowed_region["start"]),
                )
                right_context_seconds_eff = min(
                    context_right_seconds,
                    max(0.0, allowed_region["end"] - segment["end"]),
                )

            enhanced = self.enhance_segment(
                audio=audio,
                activity=activity,
                speaker_id=target_idx,
                segment_start=segment["start"],
                segment_end=segment["end"],
                sample_rate=sample_rate,
                context_left_seconds=left_context_seconds_eff,
                context_right_seconds=right_context_seconds_eff,
                segment_unit="seconds",
                num_chunks=num_chunks,
                mode=mode,
            )
            yield {
                "speaker": target_speaker,
                "speaker_id": target_idx,
                "segment_index": group_segment_indices[idx],  # Use global index for consistent naming
                "segment_start": segment["start"],
                "segment_end": segment["end"],
                "sample_rate": sample_rate,
                "enhanced_audio": enhanced,
            }

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _enhance_tensor(
        self,
        audio: torch.Tensor,
        activity: torch.Tensor,
        speaker_id: Union[int, List[int]],
        left_context_frames: int,
        right_context_frames: int,
        num_chunks: int,
        cpu_fallback: bool = False,
        return_dict: bool = False,
        garbage_class: Optional[bool] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Core enhancement logic operating on GPU tensors.

        Supports both single and multiple speaker enhancement.

        Parameters
        ----------
        audio : torch.Tensor, shape (1, channels, samples)
        activity : torch.Tensor, shape (1, speakers, samples)
        speaker_id : int or List[int]
            - If int: single speaker index (0-based)
            - If List[int]: multiple speaker indices
        left_context_frames, right_context_frames : int
        num_chunks : int
        cpu_fallback : bool
            If True, each memory-intensive stage (dereverb, gss, mc) falls back
            to CPU on CUDA OOM instead of propagating the error.
        return_dict : bool
            If True, return dict with 'audio' and statistics ('masks', 'eigenvalues',
            'mahalanobis', 'occupancy', 'temporal_variance', 'condition_number').
        garbage_class : bool or None
            If True, append one extra background/garbage activity class.
            If False, do not append garbage class.
            If None (default), use self.garbage_class.

        Returns
        -------
        torch.Tensor or List[torch.Tensor] or Dict
            If return_dict=False (default):
                Enhanced output, full output including context frames.
                - Single speaker + single-channel mode: shape (samples,)
                - Single speaker + MIMO mode: shape (num_channels, samples)
                - Multiple speakers + single-channel mode: shape (num_speakers, samples)
                - Multiple speakers + MIMO mode: shape (num_speakers, num_channels, samples)
            
            If return_dict=True:
                Dict with keys:
                - 'audio': (enhanced output as above)
                - 'masks': (num_sources, freq, frames) source masks [0, 1]
                - 'eigenvalues': (num_sources, freq, num_channels) eigenvalue statistics
                - 'mahalanobis': (num_sources, freq, frames) Mahalanobis distances
                - 'occupancy': (num_sources,) time-averaged mask values [0, 1]
                - 'temporal_variance': (num_sources,) temporal variance per source
                - 'condition_number': (num_sources, freq) eigenvalue ratio per freq
        """
        # Validate speaker_id
        is_multi_speaker = isinstance(speaker_id, list)
        speaker_ids: List[int] = cast(List[int], speaker_id) if is_multi_speaker else [cast(int, speaker_id)]
        
        # Use provided garbage_class or default to self.garbage_class
        use_garbage_class = garbage_class if garbage_class is not None else self.garbage_class
        # Add garbage class if enabled
        activity = _append_garbage_activity_class(activity, use_garbage_class)
        num_speakers_activity = activity.size(1)
        for sid in speaker_ids:
            if not isinstance(sid, int) or sid < 0 or sid >= num_speakers_activity:
                raise ValueError(
                    f"Invalid speaker_id: {sid}. Must be in range [0, {num_speakers_activity - 1}]"
                )

        # Analysis transform → complex spectrogram
        x_enc, _ = self.analysis(input=audio)          # (1, ch, freq, frames)
        a_enc = activity_time_to_timefreq(
            activity,
            win_length=self.fft_length,
            hop_length=self.hop_length,
            aggregation=self.activity_aggregation,
        )                                               # (1, spk, frames)

        F = x_enc.size(-2)
        chunk_size = int(math.ceil(F / num_chunks))

        # ---- Dereverberation + GSS mask estimation (per chunk) ----
        mask_chunks = []
        eigenvalue_chunks = []
        mahalanobis_chunks = []
        
        for n in range(num_chunks):
            n_start = n * chunk_size
            n_end = min(F, (n + 1) * chunk_size)

            x_enc_n = x_enc[..., n_start:n_end, :]

            # WPE dereverberation (optional)
            if self.enable_dereverb and self.dereverb is not None:
                if cpu_fallback:
                    x_enc_n, _ = _try_gpu_else_cpu(self.dereverb, input=x_enc_n)
                else:
                    x_enc_n, _ = self.dereverb(input=x_enc_n)
            x_enc[..., n_start:n_end, :] = x_enc_n

            # GSS mask estimation
            if cpu_fallback:
                gss_result_n = _try_gpu_else_cpu(self.gss, x_enc_n, a_enc, return_dict=True)
            else:
                gss_result_n = self.gss(x_enc_n, a_enc, return_dict=True)
            mask_n = gss_result_n["masks"]  # Extract masks from Dict
            mask_chunks.append(mask_n)
            if return_dict:
                eigenvalue_chunks.append(gss_result_n["eigenvalues"])
                mahalanobis_chunks.append(gss_result_n["mahalanobis"])

        mask = torch.concatenate(mask_chunks, dim=-2)  # (1, spk, freq, frames)
        
        # Collect eigenvalues and mahalanobis for return_dict
        if return_dict:
            eigenvalues = torch.concatenate(eigenvalue_chunks, dim=-2)  # (1, spk, freq, channels)
            mahalanobis = torch.concatenate(mahalanobis_chunks, dim=-2)  # (1, spk, freq, frames)

        # Zero out context frames in the mask
        mask[..., :left_context_frames] = 0
        if right_context_frames > 0:
            mask[..., -right_context_frames:] = 0

        # Process each speaker independently
        outputs = []
        for sid in speaker_ids:
            # Split into target vs. undesired for this speaker
            mask_target = mask[:, sid : sid + 1, ...]
            mask_undesired = torch.sum(mask, dim=1, keepdim=True) - mask_target

            # ---- Multichannel beamforming (per chunk) ----
            target_chunks = []
            for n in range(num_chunks):
                n_start = n * chunk_size
                n_end = min(F, (n + 1) * chunk_size)

                if cpu_fallback:
                    target_enc_n, _ = _try_gpu_else_cpu(
                        self.mc,
                        input=x_enc[..., n_start:n_end, :],
                        mask=mask_target[..., n_start:n_end, :],
                        mask_undesired=mask_undesired[..., n_start:n_end, :],
                    )
                else:
                    target_enc_n, _ = self.mc(
                        input=x_enc[..., n_start:n_end, :],
                        mask=mask_target[..., n_start:n_end, :],
                        mask_undesired=mask_undesired[..., n_start:n_end, :],
                    )
                target_chunks.append(target_enc_n)

            target_enc = torch.concatenate(target_chunks, dim=-2)

            # Synthesis transform → waveform
            target, _ = self.synthesis(input=target_enc)   # (1, num_channels, samples)

            # Extract output based on MIMO mode
            if self.mc.filter.is_mimo:
                output = target[0, :, :]  # (num_channels, samples) for MIMO mode
            else:
                output = target[0, 0, :]  # (samples,) for single-channel mode

            outputs.append(output)

        # Return outputs as list for multiple speakers, single tensor for single speaker
        if is_multi_speaker:
            audio_output = outputs  # List of tensors
        else:
            audio_output = outputs[0]  # Single speaker output
        
        if return_dict:
            # Compute statistics
            masks = mask[0]  # (spk, freq, frames)
            occupancy = masks.mean(dim=(1, 2))  # (spk,)
            temporal_variance = masks.var(dim=(1, 2))  # (spk,)
            eigenvalues_0 = eigenvalues[0]  # (spk, freq, channels)
            condition_number = eigenvalues_0.amax(dim=-1) / (eigenvalues_0.amin(dim=-1) + 1e-8)
            mahalanobis_0 = mahalanobis[0]  # (spk, freq, frames)
            
            return {
                'audio': audio_output,
                'masks': masks,
                'eigenvalues': eigenvalues_0,
                'mahalanobis': mahalanobis_0,
                'occupancy': occupancy,
                'temporal_variance': temporal_variance,
                'condition_number': condition_number,
            }
        else:
            return audio_output

