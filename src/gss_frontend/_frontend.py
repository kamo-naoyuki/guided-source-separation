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
from typing import Union, Optional

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


def _try_gpu_else_cpu(module: torch.nn.Module, *args, **kwargs):
    """Run *module* on GPU; on CUDA OOM fall back to CPU for this call only.

    Moves all tensor arguments to CPU, runs the module (which must have no
    GPU-resident parameters or buffers), then moves the outputs back to the
    original device.  Non-tensor arguments are passed through unchanged.
    """
    try:
        return module(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
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
    mc_ref_channel : str or int
        Reference channel selection strategy (default ``'max_snr'``).
    mc_mask_min_db : float
        Minimum mask value in dB for the multichannel filter (default -200).
    mc_postmask_min_db : float
        Minimum post-mask value in dB (default 0, i.e. no post-masking).
    activity_aggregation : str
        How to aggregate sample-level activity into frame-level activity.
        One of ``'mean'`` (default), ``'max'``, or ``'any'``.
    use_dtype : torch.dtype
        Complex dtype used internally (default ``torch.cfloat``).
    device : str or torch.device
        Device to run on (default ``'cuda'``).
    """

    def __init__(
        self,
        stft_fft_length: int = 1024,
        stft_hop_length: int = 256,
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
        activity_aggregation: str = "mean",
        use_dtype: torch.dtype = torch.cfloat,
        device: str = "cuda",
    ):
        self.fft_length = stft_fft_length
        self.hop_length = stft_hop_length
        self.device = torch.device(device)
        if activity_aggregation not in ("mean", "max", "any"):
            raise ValueError(
                f"activity_aggregation='{activity_aggregation}' is not supported. "
                "Use one of: 'mean', 'max', 'any'."
            )
        self.activity_aggregation = activity_aggregation

        self.analysis = AudioToSpectrogram(
            fft_length=stft_fft_length, hop_length=stft_hop_length
        ).to(self.device)
        self.synthesis = SpectrogramToAudio(
            fft_length=stft_fft_length, hop_length=stft_hop_length
        ).to(self.device)
        self.dereverb = MaskBasedDereverbWPE(
            filter_length=dereverb_filter_length,
            prediction_delay=dereverb_prediction_delay,
            num_iterations=dereverb_num_iterations,
            dtype=use_dtype,
        ).to(self.device)
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
        audio: np.ndarray,
        activity: np.ndarray,
        speaker_id: int,
        left_context: int = 0,
        right_context: int = 0,
        num_chunks: int = 1,
    ) -> np.ndarray:
        """Enhance a single utterance.

        Parameters
        ----------
        audio : np.ndarray, shape (channels, samples)
            Multi-channel waveform (float32 recommended).
        activity : np.ndarray, shape (speakers, samples)
            Speaker activity. Each row corresponds to one speaker; values may
            be binary {0, 1} or soft confidences in [0, 1].
        speaker_id : int
            Index (0-based) of the target speaker row in ``activity``.
        left_context : int
            Number of leading samples that are context (will be dropped from
            the output). Typically set when the input ``audio`` was extended
            to the left before calling this function.
        right_context : int
            Number of trailing context samples (will be dropped from output).
        num_chunks : int
            Split the frequency axis into this many chunks to reduce peak GPU
            memory usage. Use ``auto`` or increase manually when you hit OOM.

        Returns
        -------
        np.ndarray, shape (samples_out,)
            Single-channel enhanced waveform, float32.
            ``samples_out = len(audio[0]) - left_context - right_context``.
        """
        audio_t = torch.from_numpy(audio).float().to(self.device)
        activity_t = torch.from_numpy(activity).float().to(self.device)

        # Add batch dimension: (1, channels, samples) / (1, speakers, samples)
        audio_t = audio_t.unsqueeze(0)
        activity_t = activity_t.unsqueeze(0)

        left_context_frames = samples_to_frames(
            left_context, self.fft_length, self.hop_length
        )
        right_context_frames = samples_to_frames(
            right_context, self.fft_length, self.hop_length
        )

        with torch.inference_mode():
            result = self._enhance_tensor(
                audio_t,
                activity_t,
                speaker_id=speaker_id,
                left_context_frames=left_context_frames,
                right_context_frames=right_context_frames,
                num_chunks=num_chunks,
            )

        # Drop context from time domain
        result = result[left_context:]
        if right_context > 0:
            result = result[:-right_context]
        return result

    def enhance_auto(
        self,
        audio: np.ndarray,
        activity: np.ndarray,
        speaker_id: int,
        left_context: int = 0,
        right_context: int = 0,
    ) -> np.ndarray:
        """Same as :meth:`enhance` but retries with finer chunking on OOM.

        Automatically splits the frequency axis into more chunks when a CUDA
        out-of-memory error occurs.  The chunk counts used are all integer
        divisors of ``fft_length // 2 + 1`` in ascending order.
        """
        import decimal

        audio_t = torch.from_numpy(audio).float().to(self.device)
        activity_t = torch.from_numpy(activity).float().to(self.device)
        audio_t = audio_t.unsqueeze(0)
        activity_t = activity_t.unsqueeze(0)

        left_context_frames = samples_to_frames(
            left_context, self.fft_length, self.hop_length
        )
        right_context_frames = samples_to_frames(
            right_context, self.fft_length, self.hop_length
        )

        num_chunks_list = _get_int_divisors(self.fft_length // 2 + 1)

        result = None
        for num_chunks in num_chunks_list:
            try:
                with torch.inference_mode():
                    result = self._enhance_tensor(
                        audio_t,
                        activity_t,
                        speaker_id=speaker_id,
                        left_context_frames=left_context_frames,
                        right_context_frames=right_context_frames,
                        num_chunks=num_chunks,
                    )
                break  # success
            except (torch.cuda.OutOfMemoryError, decimal.InvalidOperation):
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
                with torch.inference_mode():
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
                result = audio_t[0, 0].cpu().numpy()
                return result

        result = result[left_context:]
        if right_context > 0:
            result = result[:-right_context]

        return result

    def dereverberate(
        self,
        audio: np.ndarray,
    ) -> "Union[np.ndarray, torch.Tensor]":
        """Apply WPE dereverberation to multi-channel audio.

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor, shape (channels, samples)
            Multi-channel waveform (float32 recommended).  When a
            ``torch.Tensor`` is provided gradients are preserved and the
            output is also a ``torch.Tensor`` (useful for training).

        Returns
        -------
        np.ndarray or torch.Tensor, shape (channels, samples)
            Dereverberated multi-channel waveform, same type as input.
        """
        audio_t, is_numpy = _prepare_audio(audio, self.device)
        ctx = torch.inference_mode() if is_numpy else contextlib.nullcontext()
        with ctx:
            x_enc, _ = self.analysis(input=audio_t)       # (1, ch, freq, frames)
            x_enc, _ = self.dereverb(input=x_enc)
            output, _ = self.synthesis(input=x_enc)        # (1, ch, samples)
        if is_numpy:
            return output[0].detach().cpu().numpy()
        return output[0]

    def beamform(
        self,
        audio: np.ndarray,
        mask_target: np.ndarray,
        mask_undesired: np.ndarray,
    ) -> "Union[np.ndarray, torch.Tensor]":
        """Apply mask-based multichannel beamforming.

        Parameters
        ----------
        audio : np.ndarray or torch.Tensor, shape (channels, samples)
            Multi-channel waveform (float32 recommended).  When a
            ``torch.Tensor`` is provided gradients are preserved and the
            output is also a ``torch.Tensor`` (useful for training).
        mask_target : np.ndarray or torch.Tensor, shape (freq, frames)
            Time-frequency mask for the target speaker.
            Obtain from :meth:`estimate_masks` or any external mask estimator.
        mask_undesired : np.ndarray or torch.Tensor, shape (freq, frames)
            Time-frequency mask for undesired signals.

        Returns
        -------
        np.ndarray or torch.Tensor, shape (samples,)
            Single-channel beamformed waveform, same type as *audio*.
        """
        audio_t, is_numpy = _prepare_audio(audio, self.device)
        # Add batch + speaker dims: (1, 1, freq, frames)
        def _to_mask(m):
            if isinstance(m, np.ndarray):
                return torch.from_numpy(m).float().to(self.device)[None, None]
            return m.to(self.device)[None, None] if m.dim() == 2 else m.to(self.device)
        mask_t = _to_mask(mask_target)
        mask_u = _to_mask(mask_undesired)
        ctx = torch.inference_mode() if is_numpy else contextlib.nullcontext()
        with ctx:
            x_enc, _ = self.analysis(input=audio_t)        # (1, ch, freq, frames)
            target_enc, _ = self.mc(
                input=x_enc, mask=mask_t, mask_undesired=mask_u
            )                                               # (1, 1, freq, frames)
            output, _ = self.synthesis(input=target_enc)   # (1, 1, samples)
        if is_numpy:
            return output[0, 0].detach().cpu().numpy()
        return output[0, 0]

    def estimate_masks(
        self,
        audio: np.ndarray,
        activity: np.ndarray,
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

        Returns
        -------
        np.ndarray or torch.Tensor, shape (speakers, freq, frames)
            Soft time-frequency masks, values in [0, 1].  Same type as *audio*.
        """
        audio_t, is_numpy = _prepare_audio(audio, self.device)
        activity_t = _prepare_activity(activity, self.device)
        ctx = torch.inference_mode() if is_numpy else contextlib.nullcontext()
        with ctx:
            x_enc, _ = self.analysis(input=audio_t)        # (1, ch, freq, frames)
            a_enc = activity_time_to_timefreq(
                activity_t,
                win_length=self.fft_length,
                hop_length=self.hop_length,
                aggregation=self.activity_aggregation,
            )
            masks = self.gss(x_enc, a_enc)                  # (1, spk, freq, frames)
        if is_numpy:
            return masks[0].detach().cpu().numpy()
        return masks[0]

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _enhance_tensor(
        self,
        audio: torch.Tensor,
        activity: torch.Tensor,
        speaker_id: int,
        left_context_frames: int,
        right_context_frames: int,
        num_chunks: int,
        cpu_fallback: bool = False,
    ) -> np.ndarray:
        """Core enhancement logic operating on GPU tensors.

        Parameters
        ----------
        audio : torch.Tensor, shape (1, channels, samples)
        activity : torch.Tensor, shape (1, speakers, samples)
        speaker_id : int
        left_context_frames, right_context_frames : int
        num_chunks : int
        cpu_fallback : bool
            If True, each memory-intensive stage (dereverb, gss, mc) falls back
            to CPU on CUDA OOM instead of propagating the error.

        Returns
        -------
        np.ndarray, shape (samples,)  — full output including context.
        """
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
        for n in range(num_chunks):
            n_start = n * chunk_size
            n_end = min(F, (n + 1) * chunk_size)

            x_enc_n = x_enc[..., n_start:n_end, :]

            # WPE dereverberation
            if cpu_fallback:
                x_enc_n, _ = _try_gpu_else_cpu(self.dereverb, input=x_enc_n)
            else:
                x_enc_n, _ = self.dereverb(input=x_enc_n)
            x_enc[..., n_start:n_end, :] = x_enc_n

            # GSS mask estimation
            if cpu_fallback:
                mask_n = _try_gpu_else_cpu(self.gss, x_enc_n, a_enc)
            else:
                mask_n = self.gss(x_enc_n, a_enc)
            mask_chunks.append(mask_n)

        mask = torch.concatenate(mask_chunks, dim=-2)  # (1, spk, freq, frames)

        # Zero out context frames in the mask
        mask[..., :left_context_frames] = 0
        if right_context_frames > 0:
            mask[..., -right_context_frames:] = 0

        # Split into target vs. undesired
        mask_target = mask[:, speaker_id : speaker_id + 1, ...]
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
        target, _ = self.synthesis(input=target_enc)   # (1, 1, samples)

        return target[0, 0].detach().cpu().numpy().squeeze()
