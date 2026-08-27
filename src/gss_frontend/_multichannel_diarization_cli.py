"""Command-line tool for multi-channel speaker diarization using pyannote + dover-lap."""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


def main():
    """CLI entry point for multi-channel diarization with merging."""
    # Check for required dependencies
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.error(
            "pyannote.audio is required. Install it with: pip install pyannote.audio"
        )
        sys.exit(1)

    try:
        from dover_lap import DiariaziationComparator
    except ImportError:
        logger.error(
            "dover-lap is required for merging. Install it with: pip install dover-lap"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Multi-channel speaker diarization using pyannote + dover-lap merging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Process pre-separated channel files
  gss-multichannel-diarization \\
    --audio ch0.wav ch1.wav ch2.wav \\
    --output meeting.rttm \\
    --hf-token <your_hf_token>

  # Process single multi-channel file
  gss-multichannel-diarization \\
    --audio meeting_multichannel.wav \\
    --output meeting.rttm \\
    --hf-token <your_hf_token>

  # Custom model
  gss-multichannel-diarization \\
    --audio ch0.wav ch1.wav \\
    --model pyannote/speaker-diarization \\
    --output meeting.rttm \\
    --hf-token <your_hf_token>

  # With device selection
  gss-multichannel-diarization \\
    --audio ch0.wav ch1.wav \\
    --output meeting.rttm \\
    --device cuda:1 \\
    --hf-token <your_hf_token>
""",
    )

    parser.add_argument(
        "--audio",
        nargs="+",
        required=True,
        help="Audio file(s): if one file with multiple channels, extract them; "
        "if multiple files, treat as individual channels.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output RTTM file path.",
    )

    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote diarization model (default: pyannote/speaker-diarization-3.1).",
    )

    parser.add_argument(
        "--hf-token",
        required=False,
        help="HuggingFace user access token (get from https://huggingface.co/settings/tokens). "
             "Required for diarization (not needed for --merge-only).",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for pyannote (default: cuda:0).",
    )

    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=None,
        help="For single multi-channel file: select specific channels (0-based indexing). "
             "If not specified, all channels are used. Example: --channels 0 2 (use channels 0 and 2).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional: use fixed threshold instead of optimal_threshold. "
        "dover-lap will find optimal by default.",
    )

    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge only mode: merge pre-computed RTTM files from individual channels. "
        "Use --audio to specify RTTM files to merge.",
    )

    args = parser.parse_args()

    # Handle merge-only mode
    if args.merge_only:
        # Merge-only: use --audio to specify RTTM files
        try:
            from dover_lap import DiariaziationComparator
        except ImportError:
            logger.error(
                "dover-lap is required for merging. Install it with: pip install dover-lap"
            )
            sys.exit(1)

        rttm_files = args.audio
        logger.info(f"Merging {len(rttm_files)} RTTM files...")

        # Read RTTM files
        rttm_strings = []
        for rttm_file in rttm_files:
            with open(rttm_file, "r") as f:
                rttm_content = f.read()
                rttm_strings.append(rttm_content)
                logger.debug(f"Loaded RTTM from {rttm_file}")

        # Merge using dover-lap
        comparator = DiariaziationComparator()

        if args.threshold is not None:
            logger.info(f"Using fixed threshold {args.threshold}")
            merged = comparator.merge(rttm_strings, threshold=args.threshold)
        else:
            logger.info("Finding optimal threshold...")
            merged = comparator.optimal_threshold(rttm_strings)

        # Save result
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            merged.write_rttm(f)

        # Log merged stats
        num_speakers = len(set(label for _, _, label in merged.itertracks(yield_label=True)))
        num_turns = len(list(merged.itertracks()))
        logger.info(
            f"Merged result: {num_speakers} speakers, {num_turns} turns"
        )
        return

    # Load pipeline
    logger.info(f"Loading model {args.model}...")
    
    if not args.hf_token:
        logger.error("--hf-token is required for diarization. Get it from https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=args.hf_token)
    pipeline.to(torch.device(args.device))

    # Prepare channel audio files
    channel_files = []

    if len(args.audio) == 1:
        # Single multi-channel file: extract channels
        audio_file = args.audio[0]
        audio, sr = sf.read(audio_file)

        if audio.ndim == 1:
            # Already mono
            logger.warning(
                f"Input file {audio_file} is mono. Consider using single-channel diarization."
            )
            channel_files = [audio_file]
        else:
            # Determine which channels to extract
            num_channels = audio.shape[1]
            if args.channels is not None:
                channels_to_extract = args.channels
                # Validate channel indices
                for ch_idx in channels_to_extract:
                    if ch_idx < 0 or ch_idx >= num_channels:
                        logger.error(
                            f"Channel index {ch_idx} out of range [0, {num_channels-1}]"
                        )
                        sys.exit(1)
                logger.info(
                    f"Extracting {len(channels_to_extract)} selected channels from {audio_file}: {channels_to_extract}..."
                )
            else:
                channels_to_extract = list(range(num_channels))
                logger.info(
                    f"Extracting {num_channels} channels from {audio_file}..."
                )

            # Extract channels
            with tempfile.TemporaryDirectory() as tmpdir:
                for ch_idx in channels_to_extract:
                    ch_file = Path(tmpdir) / f"ch{ch_idx}.wav"
                    sf.write(ch_file, audio[:, ch_idx], sr)
                    channel_files.append(str(ch_file))

                # Process channels and merge
                _process_and_merge_channels(
                    channel_files, pipeline, args.threshold, args.output
                )
    else:
        # Multiple channel files
        if args.channels is not None:
            logger.warning("--channels is ignored when multiple audio files are provided")
        channel_files = args.audio
        logger.info(f"Processing {len(channel_files)} channel files...")
        _process_and_merge_channels(
            channel_files, pipeline, args.threshold, args.output
        )

    logger.info(f"Merged diarization saved to {args.output}")


def _process_and_merge_channels(
    channel_files: List[str],
    pipeline,
    threshold: Optional[float],
    output_path: str,
) -> None:
    """Process each channel and merge results."""
    from dover_lap import DiariaziationComparator

    diarizations = []
    rttm_strings = []

    for ch_idx, ch_file in enumerate(channel_files):
        logger.info(f"[Ch{ch_idx}] Running diarization on {ch_file}...")
        diarization = pipeline(ch_file)
        diarizations.append(diarization)

        # Convert to RTTM string for merging
        rttm_str = str(diarization)
        rttm_strings.append(rttm_str)

        # Log some stats
        num_turns = len(list(diarization.itertracks()))
        logger.debug(f"[Ch{ch_idx}] Found {num_turns} speaker turns")

    # Merge using dover-lap
    logger.info("Merging diarizations using dover-lap...")
    comparator = DiariaziationComparator()

    if threshold is not None:
        # Use fixed threshold
        logger.info(f"Using fixed threshold {threshold}")
        merged = comparator.merge(rttm_strings, threshold=threshold)
    else:
        # Find optimal threshold
        logger.info("Finding optimal threshold...")
        merged = comparator.optimal_threshold(rttm_strings)

    # Save result
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        merged.write_rttm(f)

    # Log merged stats
    num_speakers = len(set(label for _, _, label in merged.itertracks(yield_label=True)))
    num_turns = len(list(merged.itertracks()))
    logger.info(
        f"Merged result: {num_speakers} speakers, {num_turns} turns"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
