"""Tests for multiple speaker enhancement functionality."""

import pytest
import numpy as np
import torch
from gss_frontend import GSS


class TestMultiSpeakerEnhance:
    """Test suite for multiple speaker enhancement."""

    @pytest.fixture
    def gss_instance(self):
        """Create a GSS instance for testing."""
        return GSS(
            stft_fft_length=512,
            stft_hop_length=128,
            device="cpu",
        )

    @pytest.fixture
    def dummy_audio(self):
        """Create dummy multi-channel audio."""
        sample_rate = 16000
        duration = 1.0  # 1 second
        num_channels = 2
        samples = int(sample_rate * duration)

        # Create audio with different frequencies for each channel
        t = np.arange(samples) / sample_rate
        audio = np.zeros((num_channels, samples), dtype=np.float32)

        for ch in range(num_channels):
            freq = 400 + ch * 100
            audio[ch] = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.1

        return audio

    @pytest.fixture
    def dummy_activity(self):
        """Create dummy speaker activity."""
        sample_rate = 16000
        duration = 1.0
        num_speakers = 3
        samples = int(sample_rate * duration)

        # Create activity with different patterns for each speaker
        activity = np.zeros((num_speakers, samples), dtype=np.float32)

        # Speaker 0: active in first 1/3
        activity[0, :samples // 3] = 1.0
        # Speaker 1: active in middle 1/3
        activity[1, samples // 3 : 2 * samples // 3] = 1.0
        # Speaker 2: active in last 1/3
        activity[2, 2 * samples // 3 :] = 1.0

        return activity

    def test_single_speaker_enhancement_int(self, gss_instance, dummy_audio, dummy_activity):
        """Test enhancement with single speaker (int speaker_id)."""
        result = gss_instance.enhance(
            audio=dummy_audio,
            activity=dummy_activity,
            speaker_id=0,  # Single speaker as int
            num_chunks=1,
        )

        # For single speaker, result should be 1D (single-channel) or 2D (MIMO)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim in [1, 2], f"Expected 1D or 2D result, got shape {result.shape}"
        
        # Output length should match input
        assert result.shape[-1] == dummy_audio.shape[-1]

    def test_multi_speaker_enhancement_list(self, gss_instance, dummy_audio, dummy_activity):
        """Test enhancement with multiple speakers (list speaker_id)."""
        speaker_ids = [0, 1, 2]
        result = gss_instance.enhance(
            audio=dummy_audio,
            activity=dummy_activity,
            speaker_id=speaker_ids,  # Multiple speakers as list
            num_chunks=1,
        )

        # For multiple speakers, result should be a list of tensors/arrays
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == len(speaker_ids)
        
        # Check each element in the list
        for i, output in enumerate(result):
            assert isinstance(output, np.ndarray)
            assert output.dtype == np.float32
            assert output.ndim in [1, 2], f"Expected 1D or 2D output for speaker {i}, got shape {output.shape}"
            assert output.shape[-1] == dummy_audio.shape[-1]

    def test_multi_speaker_enhancement_tensor(self, gss_instance, dummy_audio, dummy_activity):
        """Test enhancement with torch.Tensor input and multiple speakers."""
        audio_t = torch.from_numpy(dummy_audio)
        activity_t = torch.from_numpy(dummy_activity)
        speaker_ids = [0, 1]

        result = gss_instance.enhance(
            audio=audio_t,
            activity=activity_t,
            speaker_id=speaker_ids,
            num_chunks=1,
        )

        # Result should be a list of torch.Tensor when input is torch.Tensor
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == len(speaker_ids)
        
        # Check each element in the list
        for i, output in enumerate(result):
            assert isinstance(output, torch.Tensor)
            assert output.dtype == audio_t.dtype
            assert output.shape[-1] == audio_t.shape[-1]

    def test_multi_speaker_partial_ids(self, gss_instance, dummy_audio, dummy_activity):
        """Test enhancement with subset of speaker IDs."""
        speaker_ids = [1, 2]  # Only speakers 1 and 2
        result = gss_instance.enhance(
            audio=dummy_audio,
            activity=dummy_activity,
            speaker_id=speaker_ids,
            num_chunks=1,
        )

        assert isinstance(result, list)
        assert len(result) == len(speaker_ids)
        for output in result:
            assert output.shape[-1] == dummy_audio.shape[-1]

    def test_invalid_speaker_id_out_of_range(self, gss_instance, dummy_audio, dummy_activity):
        """Test that invalid speaker_id raises error."""
        # Test with out-of-range single speaker
        with pytest.raises(ValueError, match="Invalid speaker_id"):
            gss_instance.enhance(
                audio=dummy_audio,
                activity=dummy_activity,
                speaker_id=99,  # Out of range (only 3 speakers)
                num_chunks=1,
            )

        # Test with out-of-range in list
        with pytest.raises(ValueError, match="Invalid speaker_id"):
            gss_instance.enhance(
                audio=dummy_audio,
                activity=dummy_activity,
                speaker_id=[0, 99],  # 99 is out of range
                num_chunks=1,
            )

    def test_invalid_speaker_id_negative(self, gss_instance, dummy_audio, dummy_activity):
        """Test that negative speaker_id raises error."""
        with pytest.raises(ValueError, match="Invalid speaker_id"):
            gss_instance.enhance(
                audio=dummy_audio,
                activity=dummy_activity,
                speaker_id=-1,
                num_chunks=1,
            )

    def test_multi_speaker_with_context(self, gss_instance, dummy_audio, dummy_activity):
        """Test multiple speakers with left/right context."""
        # Extend audio to allow context
        extended_audio = np.concatenate([
            dummy_audio * 0.01,  # Left context
            dummy_audio,
            dummy_audio * 0.01,  # Right context
        ], axis=-1)
        
        extended_activity = np.concatenate([
            dummy_activity * 0.1,  # Low activity for context
            dummy_activity,
            dummy_activity * 0.1,  # Low activity for context
        ], axis=-1)

        left_context = 16000 // 4  # 0.25 seconds (4000 samples)
        right_context = 16000 // 4  # 0.25 seconds (4000 samples)

        result = gss_instance.enhance(
            audio=extended_audio,
            activity=extended_activity,
            speaker_id=[0, 1],
            left_context=left_context,
            right_context=right_context,
            num_chunks=1,
        )

        # extended_audio has shape (2, 16000 + 16000 + 16000) = (2, 48000)
        # After removing left_context (4000) and right_context (4000):
        # output should be (2, 48000 - 4000 - 4000) = (2, 40000)
        expected_samples = 48000 - left_context - right_context  # 40000
        assert isinstance(result, list)
        assert len(result) == 2  # 2 speakers
        for output in result:
            assert output.shape[-1] == expected_samples

    def test_single_vs_multi_speaker_consistency(self, gss_instance, dummy_audio, dummy_activity):
        """Test that single speaker processing is consistent in single vs multi mode."""
        # Process speaker 0 as single speaker
        result_single = gss_instance.enhance(
            audio=dummy_audio,
            activity=dummy_activity,
            speaker_id=0,
            num_chunks=1,
        )

        # Process speaker 0 in multi-speaker mode
        result_multi = gss_instance.enhance(
            audio=dummy_audio,
            activity=dummy_activity,
            speaker_id=[0],
            num_chunks=1,
        )

        # Extract single speaker from multi result (which is a list)
        assert isinstance(result_multi, list) and len(result_multi) == 1
        result_multi_single = result_multi[0]

        # Results should be close (not necessarily identical due to numerical precision)
        np.testing.assert_allclose(
            result_single,
            result_multi_single,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Single speaker mode should produce consistent results"
        )
