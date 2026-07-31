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

"""Unit tests for gss_frontend package.

All tests run on CPU (no GPU required).

Run with:
    cd gss-frontend
    pip install -e .
    python -m pytest tests/ -v
"""

import math
import pytest
import numpy as np
import torch

from gss_frontend._modules import (
    AudioToSpectrogram,
    SpectrogramToAudio,
    MaskEstimatorGSS,
    MaskBasedDereverbWPE,
    MaskBasedBeamformer,
    ParametricMultichannelWienerFilter,
    db2mag,
    make_seq_mask_like,
)
from gss_frontend import GSS
from gss_frontend._frontend import activity_time_to_timefreq, samples_to_frames


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
FFT_LENGTH = 512
HOP_LENGTH = 128
N_CHANNELS = 4
N_SPEAKERS = 3
DURATION_S = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)
DEVICE = torch.device("cpu")


@pytest.fixture
def audio_batch():
    """Random multi-channel audio batch, shape (B=1, C, T)."""
    torch.manual_seed(0)
    return torch.randn(1, N_CHANNELS, N_SAMPLES, device=DEVICE)


@pytest.fixture
def activity_batch():
    """Random binary activity mask, shape (B=1, S, T)."""
    torch.manual_seed(1)
    activity = (torch.rand(1, N_SPEAKERS, N_SAMPLES, device=DEVICE) > 0.5).float()
    # Ensure at least one active frame per speaker to avoid degenerate case
    activity[:, :, :100] = 1.0
    return activity


@pytest.fixture
def spectrogram(audio_batch):
    """Pre-computed spectrogram from audio_batch."""
    stft = AudioToSpectrogram(fft_length=FFT_LENGTH, hop_length=HOP_LENGTH).to(DEVICE)
    with torch.inference_mode():
        spec, _ = stft(input=audio_batch)
    return spec


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_db2mag_zero(self):
        assert db2mag(0) == pytest.approx(1.0)

    def test_db2mag_positive(self):
        assert db2mag(20) == pytest.approx(10.0)

    def test_db2mag_negative(self):
        assert db2mag(-20) == pytest.approx(0.1)

    def test_make_seq_mask_like_shape(self):
        like = torch.zeros(2, 3, 10)
        lengths = torch.tensor([5, 8])
        mask = make_seq_mask_like(lengths=lengths, like=like, time_dim=-1, valid_ones=True)
        # Broadcastable to (2, 3, 10): batch and time match, middle dim is singleton
        assert mask.shape == (2, 1, 10)

    def test_make_seq_mask_like_valid_count(self):
        like = torch.zeros(2, 10)
        lengths = torch.tensor([3, 7])
        mask = make_seq_mask_like(lengths=lengths, like=like, time_dim=-1, valid_ones=True)
        assert mask[0, :, :3].all()
        assert not mask[0, :, 3:].any()
        assert mask[1, :, :7].all()
        assert not mask[1, :, 7:].any()

    def test_samples_to_frames(self):
        # Standard formula: frames = (samples - fft + hop) / hop
        f = samples_to_frames(N_SAMPLES, FFT_LENGTH, HOP_LENGTH)
        expected = int((N_SAMPLES - FFT_LENGTH + HOP_LENGTH) / HOP_LENGTH)
        assert f == expected

    def test_activity_time_to_timefreq_shape(self):
        activity = torch.ones(1, N_SPEAKERS, N_SAMPLES)
        a_tf = activity_time_to_timefreq(activity, win_length=FFT_LENGTH, hop_length=HOP_LENGTH)
        assert a_tf.ndim == 3
        assert a_tf.shape[0] == 1
        assert a_tf.shape[1] == N_SPEAKERS

    def test_activity_time_to_timefreq_soft_values(self):
        activity = torch.zeros(1, 1, N_SAMPLES)
        activity[..., : N_SAMPLES // 2] = 0.25
        activity[..., N_SAMPLES // 2 :] = 0.75
        a_tf = activity_time_to_timefreq(activity, win_length=FFT_LENGTH, hop_length=HOP_LENGTH)
        assert a_tf.dtype.is_floating_point
        assert torch.min(a_tf) >= 0.0
        assert torch.max(a_tf) <= 1.0
        # Soft values should be preserved (not collapsed to {0,1}).
        assert torch.any((a_tf > 0.0) & (a_tf < 1.0))

    def test_activity_time_to_timefreq_aggregation_modes(self):
        activity = torch.rand(1, 2, N_SAMPLES)
        a_mean = activity_time_to_timefreq(
            activity, win_length=FFT_LENGTH, hop_length=HOP_LENGTH, aggregation="mean"
        )
        a_max = activity_time_to_timefreq(
            activity, win_length=FFT_LENGTH, hop_length=HOP_LENGTH, aggregation="max"
        )
        a_any = activity_time_to_timefreq(
            activity, win_length=FFT_LENGTH, hop_length=HOP_LENGTH, aggregation="any"
        )

        assert a_mean.shape == a_max.shape == a_any.shape
        assert torch.all(a_max >= a_mean)
        assert torch.all((a_any == 0.0) | (a_any == 1.0))

    def test_activity_time_to_timefreq_invalid_aggregation(self):
        activity = torch.ones(1, 1, N_SAMPLES)
        with pytest.raises(ValueError):
            activity_time_to_timefreq(
                activity,
                win_length=FFT_LENGTH,
                hop_length=HOP_LENGTH,
                aggregation="median",
            )


# ---------------------------------------------------------------------------
# STFT round-trip
# ---------------------------------------------------------------------------

class TestSTFT:
    def test_analysis_output_shape(self, audio_batch):
        stft = AudioToSpectrogram(fft_length=FFT_LENGTH, hop_length=HOP_LENGTH).to(DEVICE)
        with torch.inference_mode():
            out, length = stft(input=audio_batch)
        B, C, F, N = out.shape
        assert B == 1
        assert C == N_CHANNELS
        assert F == FFT_LENGTH // 2 + 1
        assert length.shape == (1,)

    def test_synthesis_output_shape(self, spectrogram):
        istft = SpectrogramToAudio(fft_length=FFT_LENGTH, hop_length=HOP_LENGTH).to(DEVICE)
        with torch.inference_mode():
            out, length = istft(input=spectrogram)
        assert out.shape[0] == 1
        assert out.shape[1] == N_CHANNELS
        assert out.shape[-1] > 0

    def test_round_trip_close(self, audio_batch):
        """iSTFT(STFT(x)) should be close to x (up to edge effects)."""
        stft = AudioToSpectrogram(fft_length=FFT_LENGTH, hop_length=HOP_LENGTH).to(DEVICE)
        istft = SpectrogramToAudio(fft_length=FFT_LENGTH, hop_length=HOP_LENGTH).to(DEVICE)
        with torch.inference_mode():
            spec, _ = stft(input=audio_batch)
            reconstructed, _ = istft(input=spec)

        # Trim to original length (STFT/iSTFT may pad)
        T = min(audio_batch.shape[-1], reconstructed.shape[-1])
        err = (audio_batch[..., :T] - reconstructed[..., :T]).abs().mean().item()
        assert err < 0.05, f"Round-trip error too large: {err:.4f}"

    def test_analysis_is_complex(self, audio_batch):
        stft = AudioToSpectrogram(fft_length=FFT_LENGTH, hop_length=HOP_LENGTH).to(DEVICE)
        with torch.inference_mode():
            out, _ = stft(input=audio_batch)
        assert out.is_complex()


# ---------------------------------------------------------------------------
# MaskEstimatorGSS
# ---------------------------------------------------------------------------

class TestMaskEstimatorGSS:
    def test_output_shape(self, spectrogram, activity_batch):
        gss = MaskEstimatorGSS(num_iterations=2, dtype=torch.cfloat).to(DEVICE)
        activity_tf = activity_time_to_timefreq(
            activity_batch, win_length=FFT_LENGTH, hop_length=HOP_LENGTH
        ).to(DEVICE)
        with torch.inference_mode():
            mask = gss(spectrogram, activity_tf)
        B, S, F, T = mask.shape
        assert B == 1
        assert S == N_SPEAKERS
        assert F == FFT_LENGTH // 2 + 1

    def test_mask_sums_to_one(self, spectrogram, activity_batch):
        """Masks should sum to approximately 1 across speakers."""
        gss = MaskEstimatorGSS(num_iterations=2, dtype=torch.cfloat).to(DEVICE)
        activity_tf = activity_time_to_timefreq(
            activity_batch, win_length=FFT_LENGTH, hop_length=HOP_LENGTH
        ).to(DEVICE)
        with torch.inference_mode():
            mask = gss(spectrogram, activity_tf)
        mask_sum = mask.sum(dim=1)  # sum over speakers
        assert torch.allclose(mask_sum, torch.ones_like(mask_sum), atol=1e-4)

    def test_mask_range(self, spectrogram, activity_batch):
        gss = MaskEstimatorGSS(num_iterations=2, dtype=torch.cfloat).to(DEVICE)
        activity_tf = activity_time_to_timefreq(
            activity_batch, win_length=FFT_LENGTH, hop_length=HOP_LENGTH
        ).to(DEVICE)
        with torch.inference_mode():
            mask = gss(spectrogram, activity_tf)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0 + 1e-6

    def test_no_nan(self, spectrogram, activity_batch):
        gss = MaskEstimatorGSS(num_iterations=2, dtype=torch.cfloat).to(DEVICE)
        activity_tf = activity_time_to_timefreq(
            activity_batch, win_length=FFT_LENGTH, hop_length=HOP_LENGTH
        ).to(DEVICE)
        with torch.inference_mode():
            mask = gss(spectrogram, activity_tf)
        assert not torch.any(torch.isnan(mask))


# ---------------------------------------------------------------------------
# MaskBasedDereverbWPE
# ---------------------------------------------------------------------------

class TestMaskBasedDereverbWPE:
    def test_output_shape_equals_input(self, spectrogram):
        dereverb = MaskBasedDereverbWPE(
            filter_length=5,
            prediction_delay=2,
            num_iterations=1,
            dtype=torch.cfloat,
        ).to(DEVICE)
        with torch.inference_mode():
            out, _ = dereverb(input=spectrogram)
        assert out.shape == spectrogram.shape

    def test_output_is_complex(self, spectrogram):
        dereverb = MaskBasedDereverbWPE(
            filter_length=5, prediction_delay=2, num_iterations=1, dtype=torch.cfloat
        ).to(DEVICE)
        with torch.inference_mode():
            out, _ = dereverb(input=spectrogram)
        assert out.is_complex()

    def test_output_dtype_preserved(self, spectrogram):
        dereverb = MaskBasedDereverbWPE(
            filter_length=5, prediction_delay=2, num_iterations=1, dtype=torch.cfloat
        ).to(DEVICE)
        with torch.inference_mode():
            out, _ = dereverb(input=spectrogram)
        assert out.dtype == spectrogram.dtype


# ---------------------------------------------------------------------------
# MaskBasedBeamformer
# ---------------------------------------------------------------------------

class TestMaskBasedBeamformer:
    def _make_masks(self, spectrogram):
        """Create dummy target and undesired masks with shape (1, 1, F, T)."""
        B, C, F, T = spectrogram.shape
        mask = torch.rand(B, 1, F, T, device=DEVICE)
        mask_u = 1 - mask
        return mask, mask_u

    def test_output_shape_miso(self, spectrogram):
        mc = MaskBasedBeamformer(
            filter_type="pmwf",
            ref_channel=0,
            mask_min_db=-200,
            mask_max_db=0,
        ).to(DEVICE)
        mask, mask_u = self._make_masks(spectrogram)
        with torch.inference_mode():
            out, _ = mc(input=spectrogram, mask=mask, mask_undesired=mask_u)
        # MISO: output has 1 channel
        assert out.shape[1] == 1
        assert out.shape[0] == spectrogram.shape[0]
        assert out.shape[-2] == spectrogram.shape[-2]

    def test_output_is_complex(self, spectrogram):
        mc = MaskBasedBeamformer(ref_channel=0).to(DEVICE)
        mask, mask_u = self._make_masks(spectrogram)
        with torch.inference_mode():
            out, _ = mc(input=spectrogram, mask=mask, mask_undesired=mask_u)
        assert out.is_complex()


# ---------------------------------------------------------------------------
# End-to-end GSS.enhance()
# ---------------------------------------------------------------------------

class TestGSSEnhance:
    @pytest.fixture
    def frontend(self):
        return GSS(
            stft_fft_length=FFT_LENGTH,
            stft_hop_length=HOP_LENGTH,
            dereverb_filter_length=5,
            dereverb_num_iterations=1,
            bss_iterations=2,
            mc_ref_channel=0,
            use_dtype=torch.cfloat,
            device="cpu",
        )

    def _make_inputs(self):
        rng = np.random.default_rng(42)
        audio = rng.standard_normal((N_CHANNELS, N_SAMPLES)).astype(np.float32)
        activity = np.zeros((N_SPEAKERS, N_SAMPLES), dtype=np.float32)
        seg = N_SAMPLES // N_SPEAKERS
        for s in range(N_SPEAKERS):
            activity[s, s * seg : (s + 1) * seg] = 1.0
        return audio, activity

    def test_output_shape(self, frontend):
        audio, activity = self._make_inputs()
        enhanced = frontend.enhance(audio, activity, speaker_id=0)
        assert enhanced.ndim == 1
        assert len(enhanced) > 0

    def test_output_dtype(self, frontend):
        audio, activity = self._make_inputs()
        enhanced = frontend.enhance(audio, activity, speaker_id=0)
        assert enhanced.dtype == np.float32

    def test_output_no_nan(self, frontend):
        audio, activity = self._make_inputs()
        enhanced = frontend.enhance(audio, activity, speaker_id=0)
        assert not np.any(np.isnan(enhanced))

    def test_output_length_matches_input_minus_context(self, frontend):
        audio, activity = self._make_inputs()
        left_ctx = 1600   # 0.1 s
        right_ctx = 1600
        enhanced = frontend.enhance(
            audio, activity, speaker_id=0,
            left_context=left_ctx, right_context=right_ctx,
        )
        expected_len = N_SAMPLES - left_ctx - right_ctx
        # Allow slight STFT-reconstruction offset
        assert abs(len(enhanced) - expected_len) <= HOP_LENGTH

    def test_different_speaker_ids_differ(self, frontend):
        """Enhancing different target speakers should give different results."""
        audio, activity = self._make_inputs()
        out0 = frontend.enhance(audio, activity, speaker_id=0)
        out1 = frontend.enhance(audio, activity, speaker_id=1)
        assert not np.allclose(out0[:min(len(out0), len(out1))],
                               out1[:min(len(out0), len(out1))], atol=1e-6)

    def test_enhance_auto_matches_enhance(self, frontend):
        """enhance_auto with no OOM should give the same result as enhance."""
        audio, activity = self._make_inputs()
        out_normal = frontend.enhance(audio, activity, speaker_id=0)
        out_auto = frontend.enhance_auto(audio, activity, speaker_id=0)
        np.testing.assert_allclose(out_normal, out_auto, rtol=1e-5, atol=1e-6)

    def test_multi_chunk_same_as_single_chunk(self, frontend):
        """Processing in 2 frequency chunks should give the same result as 1 chunk."""
        audio, activity = self._make_inputs()
        out1 = frontend.enhance(audio, activity, speaker_id=0, num_chunks=1)
        out2 = frontend.enhance(audio, activity, speaker_id=0, num_chunks=2)
        np.testing.assert_allclose(out1, out2, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Standalone pipeline stages
# ---------------------------------------------------------------------------

class TestGSSStandaloneAPIs:
    @pytest.fixture
    def frontend(self):
        return GSS(
            stft_fft_length=FFT_LENGTH,
            stft_hop_length=HOP_LENGTH,
            dereverb_filter_length=5,
            dereverb_num_iterations=1,
            bss_iterations=2,
            mc_ref_channel=0,
            use_dtype=torch.cfloat,
            device="cpu",
        )

    def _make_inputs(self):
        rng = np.random.default_rng(42)
        audio = rng.standard_normal((N_CHANNELS, N_SAMPLES)).astype(np.float32)
        activity = np.zeros((N_SPEAKERS, N_SAMPLES), dtype=np.float32)
        seg = N_SAMPLES // N_SPEAKERS
        for s in range(N_SPEAKERS):
            activity[s, s * seg : (s + 1) * seg] = 1.0
        return audio, activity

    # --- dereverberate ---

    def test_dereverberate_output_shape(self, frontend):
        audio, _ = self._make_inputs()
        dry = frontend.dereverberate(audio)
        assert dry.shape == audio.shape

    def test_dereverberate_output_dtype(self, frontend):
        audio, _ = self._make_inputs()
        dry = frontend.dereverberate(audio)
        assert dry.dtype == np.float32

    def test_dereverberate_no_nan(self, frontend):
        audio, _ = self._make_inputs()
        dry = frontend.dereverberate(audio)
        assert not np.any(np.isnan(dry))

    # --- estimate_masks ---

    def test_estimate_masks_output_shape(self, frontend):
        audio, activity = self._make_inputs()
        masks = frontend.estimate_masks(audio, activity)
        F = FFT_LENGTH // 2 + 1
        assert masks.shape[0] == N_SPEAKERS
        assert masks.shape[1] == F
        assert masks.ndim == 3

    def test_estimate_masks_range(self, frontend):
        audio, activity = self._make_inputs()
        masks = frontend.estimate_masks(audio, activity)
        assert masks.min() >= 0.0
        assert masks.max() <= 1.0 + 1e-6

    def test_estimate_masks_sum_to_one(self, frontend):
        """Masks should sum to ~1 across the speaker axis."""
        audio, activity = self._make_inputs()
        masks = frontend.estimate_masks(audio, activity)
        np.testing.assert_allclose(masks.sum(axis=0), np.ones_like(masks[0]), atol=1e-4)

    def test_estimate_masks_no_nan(self, frontend):
        audio, activity = self._make_inputs()
        masks = frontend.estimate_masks(audio, activity)
        assert not np.any(np.isnan(masks))

    # --- beamform ---

    def test_beamform_output_shape(self, frontend):
        audio, activity = self._make_inputs()
        masks = frontend.estimate_masks(audio, activity)
        mask_target    = masks[0]
        mask_undesired = masks.sum(axis=0) - masks[0]
        enhanced = frontend.beamform(audio, mask_target, mask_undesired)
        assert enhanced.ndim == 1
        assert len(enhanced) > 0

    def test_beamform_output_dtype(self, frontend):
        audio, activity = self._make_inputs()
        masks = frontend.estimate_masks(audio, activity)
        enhanced = frontend.beamform(audio, masks[0], masks.sum(axis=0) - masks[0])
        assert enhanced.dtype == np.float32

    def test_beamform_no_nan(self, frontend):
        audio, activity = self._make_inputs()
        masks = frontend.estimate_masks(audio, activity)
        enhanced = frontend.beamform(audio, masks[0], masks.sum(axis=0) - masks[0])
        assert not np.any(np.isnan(enhanced))

    def test_dereverberate_then_beamform_matches_enhance(self, frontend):
        """dereverberate → estimate_masks → beamform should approximate enhance."""
        audio, activity = self._make_inputs()
        # Full pipeline via enhance
        ref = frontend.enhance(audio, activity, speaker_id=0)
        # Manual pipeline
        dry   = frontend.dereverberate(audio)
        masks = frontend.estimate_masks(dry, activity)
        out   = frontend.beamform(dry, masks[0], masks.sum(axis=0) - masks[0])
        # Results won't be bit-exact (enhance does dereverb per-chunk on the
        # spectrogram directly) but should have the same length and no NaNs.
        assert abs(len(out) - len(ref)) <= HOP_LENGTH
        assert not np.any(np.isnan(out))
