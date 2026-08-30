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
import gss_frontend._frontend as frontend_module

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
from gss_frontend._frontend import (
    activity_time_to_timefreq,
    samples_to_frames,
    _build_activity_from_diarization,
    _load_diarization_segments,
    _load_uem_regions,
)


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
        # like is 2D so mask is also 2D (B, T)
        assert mask[0, :3].all()
        assert not mask[0, 3:].any()
        assert mask[1, :7].all()
        assert not mask[1, 7:].any()

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


class TestDiarizationHelpers:
    def test_load_diarization_segments_from_dict(self, monkeypatch):
        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                assert kwargs.get("format") == "rttm"
                return {
                    "sessionA": [
                        {"speaker": "spk1", "start": 0.0, "end": 0.5},
                        {"speaker": "spk2", "start_time": 0.5, "end_time": 1.0},
                    ]
                }

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        segments = _load_diarization_segments(
            diarization="dummy.rttm",
            diarization_format="rttm",
            session_id="sessionA",
        )
        assert len(segments) == 2
        assert segments[0]["speaker"] == "spk1"
        assert segments[0]["session_id"] == "sessionA"
        assert segments[1]["speaker"] == "spk2"

    def test_load_diarization_segments_multi_file_concat(self, monkeypatch):
        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                if path == "part1.rttm":
                    return [{"speaker": "spk1", "start": 0.1, "end": 0.5}]
                if path == "part2.rttm":
                    return [{"speaker": "spk1", "start": 0.2, "end": 0.4}]
                raise AssertionError(path)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        segments = _load_diarization_segments(
            diarization=["part1.rttm", "part2.rttm"],
            time_concat=True,
        )

        assert len(segments) == 2
        assert segments[0]["start"] == pytest.approx(0.1)
        assert segments[0]["end"] == pytest.approx(0.5)
        # part2 is shifted by part1 local end (=0.5)
        assert segments[1]["start"] == pytest.approx(0.7)
        assert segments[1]["end"] == pytest.approx(0.9)

    def test_load_diarization_segments_multi_file_offsets(self, monkeypatch):
        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                if path == "part1.rttm":
                    return [{"speaker": "spk1", "start": 0.1, "end": 0.5}]
                if path == "part2.rttm":
                    return [{"speaker": "spk1", "start": 0.2, "end": 0.4}]
                raise AssertionError(path)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        segments = _load_diarization_segments(
            diarization=["part1.rttm", "part2.rttm"],
            diarization_offsets=[0.0, 2.0],
        )

        assert len(segments) == 2
        assert segments[0]["start"] == pytest.approx(0.1)
        assert segments[1]["start"] == pytest.approx(2.2)

    def test_build_activity_from_diarization(self):
        segments = [
            {"speaker": "spk1", "start": 0.0, "end": 0.25},
            {"speaker": "spk2", "start": 0.25, "end": 0.5},
        ]
        speakers = ["spk1", "spk2"]
        activity = _build_activity_from_diarization(
            segments=segments,
            speakers=speakers,
            num_samples=100,
            sample_rate=100,
        )
        assert activity.shape == (2, 100)
        assert np.all(activity[0, :25] == 1.0)
        assert np.all(activity[0, 25:] == 0.0)
        assert np.all(activity[1, :25] == 0.0)
        assert np.all(activity[1, 25:50] == 1.0)

    def test_load_uem_regions_from_dict(self, monkeypatch):
        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.uem"
                assert kwargs.get("format") == "uem"
                return {
                    "sessionA": [
                        {"start_time": 0.1, "end_time": 0.9},
                    ]
                }

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        regions = _load_uem_regions(
            uem="dummy.uem",
            uem_format="uem",
            session_id="sessionA",
        )
        assert len(regions) == 1
        assert regions[0]["session_id"] == "sessionA"
        assert regions[0]["start"] == pytest.approx(0.1)
        assert regions[0]["end"] == pytest.approx(0.9)

    def test_enhance_from_diarization_calls_segment_enhancer(self, monkeypatch):
        frontend = GSS(
            stft_fft_length=FFT_LENGTH,
            stft_hop_length=HOP_LENGTH,
            dereverb_filter_length=5,
            dereverb_num_iterations=1,
            bss_iterations=2,
            mc_ref_channel=0,
            use_dtype=torch.cfloat,
            device="cpu",
        )

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                return [
                    {"speaker": "spkA", "start": 0.1, "end": 0.2},
                    {"speaker": "spkB", "start": 0.2, "end": 0.3},
                    {"speaker": "spkA", "start": 0.3, "end": 0.4},
                ]

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            seg_len = int(round((kwargs["segment_end"] - kwargs["segment_start"]) * kwargs["sample_rate"]))
            return np.zeros((seg_len,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization="dummy.rttm",
            speaker_id="spkA",
            context_left_seconds=0.5,
            context_right_seconds=0.5,
        ))

        assert len(outputs) == 2
        assert len(calls) == 2
        assert all(out["speaker"] == "spkA" for out in outputs)
        assert all(call["speaker_id"] == 0 for call in calls)

    def test_enhance_from_diarization_accepts_int_speaker_id(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                return [
                    {"speaker": "spkA", "start": 0.1, "end": 0.2},
                    {"speaker": "spkB", "start": 0.2, "end": 0.3},
                ]

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization="dummy.rttm",
            speaker_id=1,
        ))

        assert len(outputs) == 1
        assert outputs[0]["speaker"] == "spkB"
        assert len(calls) == 1
        assert calls[0]["speaker_id"] == 1

    def test_enhance_from_diarization_all_speakers_when_omitted(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                return [
                    {"speaker": "spkA", "start": 0.1, "end": 0.2},
                    {"speaker": "spkB", "start": 0.2, "end": 0.3},
                    {"speaker": "spkA", "start": 0.3, "end": 0.4},
                ]

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization="dummy.rttm",
        ))

        assert len(outputs) == 3
        assert [out["speaker"] for out in outputs] == ["spkA", "spkB", "spkA"]
        assert [call["speaker_id"] for call in calls] == [0, 1, 0]

    def test_enhance_from_diarization_supports_multiple_speakers(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                return [
                    {"speaker": "spkA", "start": 0.1, "end": 0.2},
                    {"speaker": "spkB", "start": 0.2, "end": 0.3},
                    {"speaker": "spkC", "start": 0.3, "end": 0.4},
                ]

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization="dummy.rttm",
            speaker_id=["spkC", 0],
        ))

        assert len(outputs) == 2
        assert [out["speaker"] for out in outputs] == ["spkA", "spkC"]
        assert [call["speaker_id"] for call in calls] == [0, 2]

    def test_enhance_from_diarization_filters_segments_with_uem(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                if path == "dummy.rttm":
                    return [
                        {"speaker": "spkA", "start": 0.05, "end": 0.12},
                        {"speaker": "spkA", "start": 0.20, "end": 0.28},
                        {"speaker": "spkA", "start": 0.33, "end": 0.42},
                    ]
                if path == "dummy.uem":
                    return [
                        {"start": 0.10, "end": 0.35},
                    ]
                raise AssertionError(path)

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization="dummy.rttm",
            uem="dummy.uem",
            speaker_id="spkA",
        ))

        assert len(outputs) == 1
        assert len(calls) == 1
        assert outputs[0]["segment_start"] == pytest.approx(0.20)
        assert outputs[0]["segment_end"] == pytest.approx(0.28)

    def test_enhance_from_diarization_caps_context_by_uem(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                if path == "dummy.rttm":
                    return [
                        {"speaker": "spkA", "start": 0.20, "end": 0.25},
                    ]
                if path == "dummy.uem":
                    return [
                        {"start": 0.18, "end": 0.26},
                    ]
                raise AssertionError(path)

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization="dummy.rttm",
            uem="dummy.uem",
            speaker_id="spkA",
            context_left_seconds=0.5,
            context_right_seconds=0.5,
        ))

        assert len(calls) == 1
        assert calls[0]["context_left_seconds"] == pytest.approx(0.02)
        assert calls[0]["context_right_seconds"] == pytest.approx(0.01)

    def test_enhance_from_diarization_filters_with_direct_valid_regions(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                if path == "dummy.rttm":
                    return [
                        {"speaker": "spkA", "start": 0.05, "end": 0.12},
                        {"speaker": "spkA", "start": 0.20, "end": 0.28},
                        {"speaker": "spkA", "start": 0.33, "end": 0.42},
                    ]
                raise AssertionError(path)

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization="dummy.rttm",
            speaker_id="spkA",
            valid_regions=[(0.10, 0.35)],
        ))

        assert len(outputs) == 1
        assert len(calls) == 1
        assert outputs[0]["segment_start"] == pytest.approx(0.20)
        assert outputs[0]["segment_end"] == pytest.approx(0.28)

    def test_enhance_from_diarization_caps_context_by_direct_valid_regions(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                if path == "dummy.rttm":
                    return [
                        {"speaker": "spkA", "start": 0.20, "end": 0.25},
                    ]
                raise AssertionError(path)

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization="dummy.rttm",
            speaker_id="spkA",
            valid_regions=[{"start": 0.18, "end": 0.26}],
            context_left_seconds=0.5,
            context_right_seconds=0.5,
        ))

        assert len(calls) == 1
        assert calls[0]["context_left_seconds"] == pytest.approx(0.02)
        assert calls[0]["context_right_seconds"] == pytest.approx(0.01)

    def test_enhance_from_diarization_intersects_uem_and_direct_valid_regions(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                if path == "dummy.rttm":
                    return [
                        {"speaker": "spkA", "start": 0.20, "end": 0.24},
                        {"speaker": "spkA", "start": 0.24, "end": 0.27},
                    ]
                if path == "dummy.uem":
                    return [
                        {"start": 0.15, "end": 0.26},
                    ]
                raise AssertionError(path)

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization="dummy.rttm",
            speaker_id="spkA",
            uem="dummy.uem",
            valid_regions=[(0.18, 0.25)],
        ))

        assert len(outputs) == 1
        assert len(calls) == 1
        assert outputs[0]["segment_start"] == pytest.approx(0.20)
        assert outputs[0]["segment_end"] == pytest.approx(0.24)

    def test_enhance_from_diarization_multi_file_audio_trim_mode(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                return [{"speaker": "spkA", "start": 0.1, "end": 0.2}]

        def _fake_torchaudio_load(path):
            if path == "ch0.wav":
                return torch.zeros((1, 16000), dtype=torch.float32), SAMPLE_RATE
            if path == "ch1.wav":
                return torch.zeros((1, 14000), dtype=torch.float32), SAMPLE_RATE
            raise AssertionError(path)

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            # Trim mode should align both channels to shortest length.
            assert kwargs["audio"].shape == (2, 14000)
            assert kwargs["activity"].shape[-1] == 14000
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path=["ch0.wav", "ch1.wav"],
            diarization="dummy.rttm",
            speaker_id="spkA",
            channel_length_mode="trim",
        ))

        assert len(outputs) == 1
        assert len(calls) == 1

    def test_enhance_from_diarization_multi_file_audio_error_mode(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                return [{"speaker": "spkA", "start": 0.1, "end": 0.2}]

        def _fake_torchaudio_load(path):
            if path == "ch0.wav":
                return torch.zeros((1, 16000), dtype=torch.float32), SAMPLE_RATE
            if path == "ch1.wav":
                return torch.zeros((1, 14000), dtype=torch.float32), SAMPLE_RATE
            raise AssertionError(path)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)

        with pytest.raises(ValueError):
            list(frontend.enhance_from_diarization(
                audio_path=["ch0.wav", "ch1.wav"],
                diarization="dummy.rttm",
                speaker_id="spkA",
                channel_length_mode="error",
            ))

    def test_enhance_from_diarization_multi_file_audio_pad_mode(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                return [{"speaker": "spkA", "start": 0.1, "end": 0.2}]

        def _fake_torchaudio_load(path):
            if path == "ch0.wav":
                return torch.zeros((1, 16000), dtype=torch.float32), SAMPLE_RATE
            if path == "ch1.wav":
                return torch.zeros((1, 14000), dtype=torch.float32), SAMPLE_RATE
            raise AssertionError(path)

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            # Pad mode should align both channels to longest length.
            assert kwargs["audio"].shape == (2, 16000)
            assert kwargs["activity"].shape[-1] == 16000
            return np.zeros((16,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path=["ch0.wav", "ch1.wav"],
            diarization="dummy.rttm",
            speaker_id="spkA",
            channel_length_mode="pad",
        ))

        assert len(outputs) == 1
        assert len(calls) == 1

    def test_enhance_from_diarization_applies_channel_offsets_samples(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                return [{"speaker": "spkA", "start": 0.0, "end": 0.002}]

        def _fake_torchaudio_load(path):
            if path == "ch0.wav":
                return torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32), SAMPLE_RATE
            if path == "ch1.wav":
                return torch.tensor([[2.0, 2.0, 2.0, 2.0]], dtype=torch.float32), SAMPLE_RATE
            raise AssertionError(path)

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            audio = kwargs["audio"]
            # ch1 delayed by +2 samples then trimmed to match shortest length.
            np.testing.assert_allclose(audio[0], np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
            np.testing.assert_allclose(audio[1], np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32))
            return np.zeros((8,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path=["ch0.wav", "ch1.wav"],
            diarization="dummy.rttm",
            speaker_id="spkA",
            channel_length_mode="trim",
            channel_offsets=[0, 2],
            channel_offset_unit="samples",
        ))

        assert len(outputs) == 1
        assert len(calls) == 1

    def test_enhance_from_diarization_multi_rttm_concat(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                if path == "part1.rttm":
                    return [{"speaker": "spkA", "start": 0.1, "end": 0.2}]
                if path == "part2.rttm":
                    return [{"speaker": "spkA", "start": 0.1, "end": 0.2}]
                raise AssertionError(path)

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((8,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization=["part1.rttm", "part2.rttm"],
            speaker_id="spkA",
            diarization_time_concat=True,
        ))

        assert len(outputs) == 2
        assert calls[0]["segment_start"] == pytest.approx(0.1)
        # part2 is shifted by part1 end (=0.2)
        assert calls[1]["segment_start"] == pytest.approx(0.3)

    def test_enhance_from_diarization_multi_rttm_default_is_merge(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                if path == "spkA.rttm":
                    return [{"speaker": "spkA", "start": 0.1, "end": 0.2}]
                if path == "spkB.rttm":
                    return [{"speaker": "spkB", "start": 0.1, "end": 0.2}]
                raise AssertionError(path)

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((N_CHANNELS, N_SAMPLES), dtype=torch.float32), SAMPLE_RATE

        calls = []

        def _fake_enhance_segment(**kwargs):
            calls.append(kwargs)
            return np.zeros((8,), dtype=np.float32)

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)
        monkeypatch.setattr(frontend, "enhance_segment", _fake_enhance_segment)

        outputs = list(frontend.enhance_from_diarization(
            audio_path="dummy.wav",
            diarization=["spkA.rttm", "spkB.rttm"],
            speaker_id=None,
        ))

        assert len(outputs) == 2
        # merge mode keeps original timestamps (no sequential shift)
        assert calls[0]["segment_start"] == pytest.approx(0.1)
        assert calls[1]["segment_start"] == pytest.approx(0.1)

    def test_enhance_from_diarization_channel_offsets_length_mismatch_raises(self, monkeypatch):
        frontend = GSS(device="cpu")

        class _FakeMeetevalIO:
            @staticmethod
            def load(path, **kwargs):
                assert path == "dummy.rttm"
                return [{"speaker": "spkA", "start": 0.0, "end": 0.01}]

        def _fake_torchaudio_load(path):
            assert path == "dummy.wav"
            return torch.zeros((2, 16), dtype=torch.float32), SAMPLE_RATE

        monkeypatch.setattr(frontend_module, "_import_meeteval_io", lambda: _FakeMeetevalIO)
        monkeypatch.setattr("torchaudio.load", _fake_torchaudio_load)

        with pytest.raises(ValueError):
            list(frontend.enhance_from_diarization(
                audio_path="dummy.wav",
                diarization="dummy.rttm",
                speaker_id="spkA",
                channel_offsets=[0],
            ))


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
            bss_iterations=10,
            mc_ref_channel=0,
            use_dtype=torch.cfloat,
            device="cpu",
        )

    def _make_inputs(self):
        # Two overlapping sinusoidal speakers with opposite spatial signatures.
        # Overlap is required so GSS has a real separation task and produces
        # distinct masks for the two speaker IDs.
        t = np.arange(N_SAMPLES, dtype=np.float32) / SAMPLE_RATE
        seg = N_SAMPLES // 3
        gains0 = np.array([1.0, 0.8, 0.4, 0.1], dtype=np.float32)
        gains1 = np.array([0.1, 0.4, 0.8, 1.0], dtype=np.float32)
        src0 = np.sin(2 * np.pi * 300.0 * t).astype(np.float32)
        src0[2 * seg :] = 0.0                      # active in [0, 2/3)
        src1 = np.sin(2 * np.pi * 800.0 * t).astype(np.float32)
        src1[: seg] = 0.0                          # active in [1/3, 1)
        audio = (gains0[:, None] * src0[None, :] +
                 gains1[:, None] * src1[None, :])
        rng = np.random.default_rng(42)
        audio += rng.standard_normal(audio.shape).astype(np.float32) * 0.01
        activity = np.zeros((2, N_SAMPLES), dtype=np.float32)
        activity[0, : 2 * seg] = 1.0
        activity[1, seg :] = 1.0
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

    def test_enhance_segment_default_context_matches_manual_window(self, frontend):
        audio, activity = self._make_inputs()
        seg_start_s = 0.2
        seg_end_s = 0.7

        out_segment = frontend.enhance_segment(
            audio,
            activity,
            speaker_id=0,
            segment_start=seg_start_s,
            segment_end=seg_end_s,
            sample_rate=SAMPLE_RATE,
        )

        seg_start = int(round(seg_start_s * SAMPLE_RATE))
        seg_end = int(round(seg_end_s * SAMPLE_RATE))
        left_ctx = int(round(15.0 * SAMPLE_RATE))
        right_ctx = int(round(15.0 * SAMPLE_RATE))
        win_start = max(0, seg_start - left_ctx)
        win_end = min(audio.shape[-1], seg_end + right_ctx)

        out_manual = frontend.enhance(
            audio[..., win_start:win_end],
            activity[..., win_start:win_end],
            speaker_id=0,
            left_context=seg_start - win_start,
            right_context=win_end - seg_end,
        )

        np.testing.assert_allclose(out_segment, out_manual, rtol=1e-5, atol=1e-6)

    def test_enhance_segment_seconds_and_samples_match(self, frontend):
        audio, activity = self._make_inputs()
        seg_start = int(0.25 * SAMPLE_RATE)
        seg_end = int(0.75 * SAMPLE_RATE)

        out_seconds = frontend.enhance_segment(
            audio,
            activity,
            speaker_id=1,
            segment_start=seg_start / SAMPLE_RATE,
            segment_end=seg_end / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            context_left_seconds=0.1,
            context_right_seconds=0.2,
            segment_unit="seconds",
        )
        out_samples = frontend.enhance_segment(
            audio,
            activity,
            speaker_id=1,
            segment_start=seg_start,
            segment_end=seg_end,
            sample_rate=SAMPLE_RATE,
            context_left_seconds=0.1,
            context_right_seconds=0.2,
            segment_unit="samples",
        )

        np.testing.assert_allclose(out_seconds, out_samples, rtol=1e-5, atol=1e-6)

    def test_enhance_segment_oom_fallback_mode_matches_manual_auto_window(self, frontend):
        audio, activity = self._make_inputs()
        seg_start_s = 0.2
        seg_end_s = 0.7

        out_segment = frontend.enhance_segment(
            audio,
            activity,
            speaker_id=0,
            segment_start=seg_start_s,
            segment_end=seg_end_s,
            sample_rate=SAMPLE_RATE,
            mode="oom_fallback",
        )

        seg_start = int(round(seg_start_s * SAMPLE_RATE))
        seg_end = int(round(seg_end_s * SAMPLE_RATE))
        left_ctx = int(round(15.0 * SAMPLE_RATE))
        right_ctx = int(round(15.0 * SAMPLE_RATE))
        win_start = max(0, seg_start - left_ctx)
        win_end = min(audio.shape[-1], seg_end + right_ctx)

        out_manual = frontend.enhance_auto(
            audio[..., win_start:win_end],
            activity[..., win_start:win_end],
            speaker_id=0,
            left_context=seg_start - win_start,
            right_context=win_end - seg_end,
        )

        np.testing.assert_allclose(out_segment, out_manual, rtol=1e-5, atol=1e-6)

    def test_enhance_segment_legacy_mode_aliases_are_supported(self, frontend):
        audio, activity = self._make_inputs()

        out_new_standard = frontend.enhance_segment(
            audio,
            activity,
            speaker_id=0,
            segment_start=0.2,
            segment_end=0.7,
            sample_rate=SAMPLE_RATE,
            mode="standard",
        )
        out_old_enhance = frontend.enhance_segment(
            audio,
            activity,
            speaker_id=0,
            segment_start=0.2,
            segment_end=0.7,
            sample_rate=SAMPLE_RATE,
            mode="enhance",
        )
        np.testing.assert_allclose(out_new_standard, out_old_enhance, rtol=1e-5, atol=1e-6)

        out_new_fallback = frontend.enhance_segment(
            audio,
            activity,
            speaker_id=0,
            segment_start=0.2,
            segment_end=0.7,
            sample_rate=SAMPLE_RATE,
            mode="oom_fallback",
        )
        out_old_auto = frontend.enhance_segment(
            audio,
            activity,
            speaker_id=0,
            segment_start=0.2,
            segment_end=0.7,
            sample_rate=SAMPLE_RATE,
            mode="auto",
        )
        np.testing.assert_allclose(out_new_fallback, out_old_auto, rtol=1e-5, atol=1e-6)

    def test_enhance_segment_invalid_mode_raises(self, frontend):
        audio, activity = self._make_inputs()

        with pytest.raises(ValueError):
            frontend.enhance_segment(
                audio,
                activity,
                speaker_id=0,
                segment_start=0.1,
                segment_end=0.2,
                sample_rate=SAMPLE_RATE,
                mode="unknown",
            )


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

    # --- estimate_masks ---

    def test_estimate_masks_output_shape(self, frontend):
        audio, activity = self._make_inputs()
        masks = frontend.estimate_masks(audio, activity)
        F = FFT_LENGTH // 2 + 1
        assert masks.shape[0] == N_SPEAKERS + 1  # +1 for garbage_class (default=True)
        assert masks.shape[1] == F
        assert masks.ndim == 3

    def test_estimate_masks_output_shape_without_garbage_class(self):
        frontend = GSS(
            stft_fft_length=FFT_LENGTH,
            stft_hop_length=HOP_LENGTH,
            dereverb_filter_length=5,
            dereverb_num_iterations=1,
            bss_iterations=2,
            mc_ref_channel=0,
            use_dtype=torch.cfloat,
            garbage_class=False,
            device="cpu",
        )
        audio, activity = self._make_inputs()
        masks = frontend.estimate_masks(audio, activity)
        assert masks.shape[0] == N_SPEAKERS  # No garbage class

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

    # --- enhance_unguided (method) ---

    def test_enhance_unguided_output_keys(self, frontend):
        """Test that enhance_unguided returns all expected keys."""
        rng = np.random.default_rng(42)
        audio = rng.standard_normal((N_CHANNELS, N_SAMPLES)).astype(np.float32)
        result = frontend.enhance_unguided(audio, num_sources=N_SPEAKERS)
        
        expected_keys = {"audio", "masks", "eigenvalues", "mahalanobis", "occupancy", 
                         "temporal_variance", "condition_number"}
        assert set(result.keys()) == expected_keys

    def test_enhance_unguided_masks_shape(self, frontend):
        """Test that masks have correct shape: (num_sources, freq, frames)."""
        rng = np.random.default_rng(42)
        audio = rng.standard_normal((N_CHANNELS, N_SAMPLES)).astype(np.float32)
        result = frontend.enhance_unguided(audio, num_sources=N_SPEAKERS)
        masks = result["masks"]
        F = FFT_LENGTH // 2 + 1
        
        assert masks.shape[0] == N_SPEAKERS
        assert masks.shape[1] == F
        assert masks.ndim == 3

    def test_enhance_unguided_masks_in_range(self, frontend):
        """Test that masks are in [0, 1]."""
        rng = np.random.default_rng(42)
        audio = rng.standard_normal((N_CHANNELS, N_SAMPLES)).astype(np.float32)
        result = frontend.enhance_unguided(audio, num_sources=N_SPEAKERS)
        masks = result["masks"]
        
        assert masks.min() >= 0.0
        assert masks.max() <= 1.0 + 1e-6

    def test_enhance_unguided_auto(self, frontend):
        """Test enhance_unguided_auto method."""
        rng = np.random.default_rng(42)
        audio = rng.standard_normal((N_CHANNELS, N_SAMPLES)).astype(np.float32)
        result = frontend.enhance_unguided_auto(audio, num_sources=N_SPEAKERS)
        
        # Should return dict with expected keys
        expected_keys = {"audio", "masks", "eigenvalues", "mahalanobis", "occupancy", 
                         "temporal_variance", "condition_number"}
        assert set(result.keys()) == expected_keys
        
        # num_sources should match input
        assert result["masks"].shape[0] == N_SPEAKERS


# ---------------------------------------------------------------------------
# Backprop through audio and activity
# ---------------------------------------------------------------------------

class TestBackprop:
    """Verify that gradients flow through audio and activity (mean/max aggregation)."""

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

    def _make_tensors(self):
        torch.manual_seed(0)
        audio = torch.randn(N_CHANNELS, N_SAMPLES, dtype=torch.float32, requires_grad=True)
        activity = torch.zeros(N_SPEAKERS, N_SAMPLES, dtype=torch.float32)
        seg = N_SAMPLES // N_SPEAKERS
        for s in range(N_SPEAKERS):
            activity.data[s, s * seg : (s + 1) * seg] = 1.0
        activity.requires_grad_(True)
        return audio, activity

    def _run_pipeline(self, frontend, audio_t, activity_t):
        audio_3d    = audio_t.unsqueeze(0)
        activity_3d = activity_t.unsqueeze(0)
        x_enc, _    = frontend.analysis(audio_3d)
        x_enc, _    = frontend.dereverb(input=x_enc)
        a_enc = activity_time_to_timefreq(
            activity_3d,
            win_length=frontend.fft_length,
            hop_length=frontend.hop_length,
            aggregation=frontend.activity_aggregation,
        )
        masks       = frontend.gss(x_enc, a_enc)
        mask_t      = masks[:, :1]
        mask_u      = masks.sum(dim=1, keepdim=True) - mask_t
        target_enc, _ = frontend.mc(input=x_enc, mask=mask_t, mask_undesired=mask_u)
        out, _      = frontend.synthesis(input=target_enc)
        return out[0, 0]

    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    def test_audio_grad_flows(self, frontend, aggregation):
        frontend.activity_aggregation = aggregation
        audio, activity = self._make_tensors()
        out = self._run_pipeline(frontend, audio, activity)
        out.abs().mean().backward()
        assert audio.grad is not None
        assert not torch.all(audio.grad == 0)

    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    def test_activity_grad_flows(self, frontend, aggregation):
        frontend.activity_aggregation = aggregation
        audio, activity = self._make_tensors()
        out = self._run_pipeline(frontend, audio, activity)
        out.abs().mean().backward()
        assert activity.grad is not None
        assert not torch.all(activity.grad == 0)

    def test_activity_grad_blocked_by_any(self, frontend):
        """aggregation='any' uses boolean ops so activity gradient must be zero."""
        frontend.activity_aggregation = "any"
        audio, activity = self._make_tensors()
        out = self._run_pipeline(frontend, audio, activity)
        out.abs().mean().backward()
        assert activity.grad is None or torch.all(activity.grad == 0)


# ---------------------------------------------------------------------------
# Distributed Processing / Group Partitioning
# ---------------------------------------------------------------------------

class TestDistributedProcessing:
    """Test group partitioning for distributed processing (SLURM, etc.)."""

    def test_partition_single_group(self):
        """Single group (no partitioning) returns all segments."""
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "spkA"},
            {"start": 5.0, "end": 10.0, "speaker": "spkB"},
            {"start": 10.0, "end": 15.0, "speaker": "spkA"},
        ]
        groups = frontend_module._partition_segments_by_duration(segments, num_groups=1)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_partition_balanced_groups(self):
        """Multiple groups should have balanced total duration."""
        segments = [
            {"start": 0.0, "end": 10.0, "speaker": "spkA"},    # 10s
            {"start": 10.0, "end": 15.0, "speaker": "spkB"},   # 5s
            {"start": 15.0, "end": 25.0, "speaker": "spkA"},   # 10s
            {"start": 25.0, "end": 30.0, "speaker": "spkB"},   # 5s
        ]
        groups = frontend_module._partition_segments_by_duration(segments, num_groups=2)
        
        # Each group should have total duration ~15s
        durations = []
        for group in groups:
            total_dur = sum(segments[i]["end"] - segments[i]["start"] for i in group)
            durations.append(total_dur)
        
        assert len(groups) == 2
        assert sum(durations) == 30.0  # Total duration preserved
        # Check balance: difference should be small
        assert abs(durations[0] - durations[1]) <= 10.0  # Allow some imbalance

    def test_partition_empty_segments(self):
        """Partitioning empty segment list should return empty groups."""
        groups = frontend_module._partition_segments_by_duration([], num_groups=3)
        assert len(groups) == 3
        assert all(len(g) == 0 for g in groups)

    def test_compute_group_statistics(self):
        """Statistics computation should work correctly."""
        segments = [
            {"start": 0.0, "end": 5.0},
            {"start": 5.0, "end": 12.0},
            {"start": 12.0, "end": 17.0},
        ]
        stats = frontend_module._compute_group_statistics(segments, [0, 1, 2])
        
        assert stats["num_segments"] == 3
        assert stats["total_duration_seconds"] == 17.0
        assert stats["avg_duration_seconds"] == pytest.approx(17.0 / 3)

    def test_compute_group_statistics_partial(self):
        """Statistics for a subset of segments."""
        segments = [
            {"start": 0.0, "end": 5.0},
            {"start": 5.0, "end": 12.0},
            {"start": 12.0, "end": 17.0},
        ]
        stats = frontend_module._compute_group_statistics(segments, [0, 2])
        
        assert stats["num_segments"] == 2
        assert stats["total_duration_seconds"] == 10.0  # 5 + 5
        assert stats["avg_duration_seconds"] == pytest.approx(5.0)

    def test_compute_group_statistics_empty(self):
        """Statistics for empty group."""
        segments = [
            {"start": 0.0, "end": 5.0},
        ]
        stats = frontend_module._compute_group_statistics(segments, [])
        
        assert stats["num_segments"] == 0
        assert stats["total_duration_seconds"] == 0.0
        assert stats["avg_duration_seconds"] == 0.0

    def test_partition_invalid_num_groups(self):
        """Invalid num_groups should raise error."""
        segments = [{"start": 0.0, "end": 5.0}]
        with pytest.raises(ValueError):
            frontend_module._partition_segments_by_duration(segments, num_groups=0)
        with pytest.raises(ValueError):
            frontend_module._partition_segments_by_duration(segments, num_groups=-1)


