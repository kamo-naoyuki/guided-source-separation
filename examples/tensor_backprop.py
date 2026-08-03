"""Example: use standalone GSS stages with torch.Tensor and backprop.

This script demonstrates training-friendly usage of:
  - GSS.dereverberate
  - GSS.estimate_masks
  - GSS.beamform

All three methods accept torch.Tensor input and preserve gradients for both
audio and activity (requires aggregation="mean" or "max"; "any" breaks the
activity gradient via boolean ops).

Usage
-----
    cd gss-frontend
    pip install -e .
    python examples/tensor_backprop.py [--device cpu]
"""

import argparse

import torch

from gss_frontend import GSS


def make_toy_batch(
    sample_rate: int = 16000,
    duration_s: float = 1.0,
    n_channels: int = 4,
    n_speakers: int = 2,
    device: str = "cpu",
):
    """Create toy multi-channel audio and activity masks as tensors."""
    n_samples = int(sample_rate * duration_s)
    t = torch.arange(n_samples, device=device, dtype=torch.float32) / sample_rate

    # Two simple sinusoid sources with overlap.
    src0 = 0.5 * torch.sin(2 * torch.pi * 300.0 * t)
    src1 = 0.5 * torch.sin(2 * torch.pi * 700.0 * t)

    seg = n_samples // 3
    env0 = torch.zeros(n_samples, device=device)
    env1 = torch.zeros(n_samples, device=device)
    env0[: 2 * seg] = 1.0
    env1[seg:] = 1.0

    src0 = src0 * env0
    src1 = src1 * env1

    # Mix into multi-channel observations via per-channel gains.
    gains0 = torch.tensor([1.0, 0.9, 0.7, 0.5], device=device)
    gains1 = torch.tensor([0.6, 0.8, 1.0, 0.7], device=device)
    audio = gains0[:, None] * src0[None, :] + gains1[:, None] * src1[None, :]

    # Add tiny noise and enable gradient on both audio and activity.
    audio = audio + 0.005 * torch.randn_like(audio)
    audio.requires_grad_(True)

    activity = torch.zeros(n_speakers, n_samples, device=device, dtype=torch.float32)
    activity[0, : 2 * seg] = 1.0
    activity[1, seg:] = 1.0
    activity.requires_grad_(True)

    return audio, activity


def main():
    parser = argparse.ArgumentParser(description="GSS Tensor/backprop demo")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    args = parser.parse_args()

    frontend = GSS(
        stft_fft_length=512,
        stft_hop_length=128,
        dereverb_filter_length=5,
        dereverb_prediction_delay=2,
        dereverb_num_iterations=1,
        bss_iterations=4,
        mc_ref_channel=0,
        use_dtype=torch.cfloat,
        device=args.device,
    )

    audio_t, activity_t = make_toy_batch(device=args.device)

    # Standalone pipeline with autograd.
    dry_t = frontend.dereverberate(audio_t)
    masks_t = frontend.estimate_masks(dry_t, activity_t)
    out_t = frontend.beamform(dry_t, masks_t[0], masks_t.sum(dim=0) - masks_t[0])

    # Dummy objective for demonstration.
    loss = out_t.abs().mean()
    loss.backward()

    print(f"loss: {loss.item():.6f}")
    if audio_t.grad is None:
        print("No gradient on input audio (unexpected).")
    else:
        print(f"audio    grad norm: {audio_t.grad.norm().item():.6f}")
    if activity_t.grad is None:
        print("No gradient on input activity (unexpected).")
    else:
        print(f"activity grad norm: {activity_t.grad.norm().item():.6f}")


if __name__ == "__main__":
    main()
