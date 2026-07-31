"""Example: separate two overlapping speakers from a simulated multi-channel recording.

This script demonstrates how to use the GSS frontend to separate two speakers
from a mixture.  No real audio files are needed — synthetic sinusoid-based
"speech" is generated on the fly so the example can be run immediately.

Usage
-----
    cd gss-frontend
    pip install -e .
    python examples/separate_speakers.py [--device cpu] [--out-dir /tmp/gss_out]

For real audio, replace the ``simulate_mixture`` function with code that loads
your own multi-channel recordings and diarization / VAD activity masks.
"""

import argparse
import logging
import os

import numpy as np
import soundfile as sf

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def simulate_mixture(
    sample_rate: int = 16000,
    duration_s: float = 3.0,
    n_channels: int = 4,
    n_speakers: int = 2,
    seed: int = 0,
):
    """Generate a synthetic multi-channel mixture of two speakers.

    Each "speaker" is a band-limited sinusoid at a different frequency.  An
    independent room impulse response (random FIR filter) is applied per
    channel to simulate spatial diversity.

    Returns
    -------
    audio : np.ndarray, shape (n_channels, n_samples)  float32
        Multi-channel mixture.
    activity : np.ndarray, shape (n_speakers, n_samples)  float32
        Ground-truth activity masks (1 = active).
    sources : list of np.ndarray, each shape (n_samples,)
        Ground-truth single-channel clean signals (for SNR evaluation).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples) / sample_rate

    # --- Clean sources: simple amplitude-modulated sinusoids ---
    # Speaker 0: 300 Hz carrier, active during first 2/3
    # Speaker 1: 900 Hz carrier, active during last 2/3  (overlap in middle 1/3)
    seg = n_samples // 3

    def make_source(freq_hz, start, end):
        s = np.sin(2 * np.pi * freq_hz * t)
        # Slight AM modulation to make it more speech-like
        s *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
        envelope = np.zeros(n_samples)
        envelope[start:end] = np.ones(end - start)
        # Smooth edges
        fade = min(int(0.02 * sample_rate), (end - start) // 4)
        if fade > 0:
            ramp = np.linspace(0, 1, fade)
            envelope[start : start + fade] = ramp
            envelope[end - fade : end] = ramp[::-1]
        return (s * envelope).astype(np.float32)

    sources = [
        make_source(300, 0,       2 * seg),   # speaker 0: segments 0–1
        make_source(900, seg,     3 * seg),   # speaker 1: segments 1–2
    ]

    # --- Activity masks ---
    activity = np.zeros((n_speakers, n_samples), dtype=np.float32)
    activity[0, :2 * seg] = 1.0
    activity[1, seg:] = 1.0

    # --- Room impulse responses: short random FIR per channel per speaker ---
    rir_len = int(0.02 * sample_rate)  # 20 ms
    mixture = np.zeros((n_channels, n_samples), dtype=np.float32)
    for spk_idx, src in enumerate(sources):
        for ch in range(n_channels):
            rir = rng.standard_normal(rir_len).astype(np.float32)
            rir /= np.linalg.norm(rir) + 1e-8
            convolved = np.convolve(src, rir)[:n_samples]
            mixture[ch] += convolved

    # Add a little noise
    noise = rng.standard_normal((n_channels, n_samples)).astype(np.float32) * 0.01
    mixture += noise

    # Normalise mixture to [-1, 1]
    peak = np.abs(mixture).max()
    if peak > 0:
        mixture /= peak

    return mixture, activity, sources


# ---------------------------------------------------------------------------
# SNR helper
# ---------------------------------------------------------------------------

def signal_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-noise ratio between a reference and an estimate (dB).

    SNR = 10 * log10( ||reference||^2 / ||reference - estimate||^2 )
    """
    T = min(len(reference), len(estimate))
    ref = reference[:T]
    est = estimate[:T]
    # Align scale
    scale = np.dot(ref, est) / (np.dot(est, est) + 1e-12)
    est_scaled = scale * est
    noise = ref - est_scaled
    snr = 10 * np.log10(np.dot(ref, ref) / (np.dot(noise, noise) + 1e-12))
    return float(snr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GSS speaker separation demo")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--out-dir", default="/tmp/gss_out", help="Directory for output WAVs")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=3.0, help="Mixture duration (s)")
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--fft-length", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--bss-iter", type=int, default=10)
    args = parser.parse_args()

    # Lazy import so --help doesn't require torch
    import torch
    from gss_frontend import GSS

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate (or load) the multi-channel mixture
    # ------------------------------------------------------------------
    logger.info("Generating synthetic %d-channel, 2-speaker mixture ...", args.n_channels)
    mixture, activity, sources = simulate_mixture(
        sample_rate=args.sample_rate,
        duration_s=args.duration,
        n_channels=args.n_channels,
    )
    logger.info("  mixture shape : %s", mixture.shape)
    logger.info("  activity shape: %s", activity.shape)

    # Save mixture (channel 0) for reference
    sf.write(
        os.path.join(args.out_dir, "mixture_ch0.wav"),
        mixture[0],
        args.sample_rate,
    )

    # ------------------------------------------------------------------
    # 2. Initialise the GSS frontend
    # ------------------------------------------------------------------
    logger.info("Initialising GSS on device=%s ...", args.device)
    frontend = GSS(
        stft_fft_length=args.fft_length,
        stft_hop_length=args.hop_length,
        dereverb_filter_length=5,
        dereverb_prediction_delay=2,
        dereverb_num_iterations=1,
        bss_iterations=args.bss_iter,
        mc_filter_type="pmwf",
        mc_filter_rank="one",
        mc_filter_postfilter="ban",
        mc_ref_channel="max_snr",
        use_dtype=torch.cfloat,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 3. Separate each speaker
    # ------------------------------------------------------------------
    n_speakers = activity.shape[0]
    for spk_id in range(n_speakers):
        logger.info("Enhancing speaker %d / %d ...", spk_id, n_speakers - 1)

        enhanced = frontend.enhance_auto(
            audio=mixture,         # (channels, samples)  float32
            activity=activity,     # (speakers, samples)  float32
            speaker_id=spk_id,
        )

        out_path = os.path.join(args.out_dir, f"enhanced_spk{spk_id}.wav")
        sf.write(out_path, enhanced, args.sample_rate)
        logger.info("  saved → %s  (length: %d samples)", out_path, len(enhanced))

        # Evaluate SNR against the synthetic ground-truth
        snr = signal_snr(sources[spk_id], enhanced)
        logger.info("  SNR vs. clean source: %.1f dB", snr)

    # ------------------------------------------------------------------
    # 4. Also save the clean sources for comparison
    # ------------------------------------------------------------------
    for spk_id, src in enumerate(sources):
        sf.write(
            os.path.join(args.out_dir, f"clean_spk{spk_id}.wav"),
            src,
            args.sample_rate,
        )

    logger.info("Done. Output files written to: %s", args.out_dir)
    logger.info(
        "Files:\n  mixture_ch0.wav  — channel 0 of the multi-channel input\n"
        + "\n".join(
            f"  enhanced_spk{i}.wav — GSS output for speaker {i}\n"
            f"  clean_spk{i}.wav    — synthetic ground truth for speaker {i}"
            for i in range(n_speakers)
        )
    )


if __name__ == "__main__":
    main()
