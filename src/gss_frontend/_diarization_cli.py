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
        logger.error("pyannote.audio is required. Install it with: pip install pyannote.audio")
        sys.exit(1)

    try:
        from dover_lap import DiariaziationComparator
    except ImportError:
        logger.error("dover-lap is required for merging. Install it with: pip install dover-lap")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Multi-channel speaker diarization using pyannote + dover-lap merging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Process pre-separated channel files
  gss-diarize \\
    --audio ch0.wav ch1.wav ch2.wav \\
    --output meeting.rttm \\
    --hf-token <your_hf_token>

  # Process single multi-channel file
  gss-diarize \\
    --audio meeting_multichannel.wav \\
    --output meeting.rttm \\
    --hf-token <your_hf_token>

  # Custom model
  gss-diarize \\
    --audio ch0.wav ch1.wav \\
    --model pyannote/speaker-diarization \\
    --output meeting.rttm \\
    --hf-token <your_hf_token>

  # With device selection
  gss-diarize \\
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

    parser.add_argument(
        "--output-format",
        choices=["rttm", "seglst"],
        default="rttm",
        help="Output format: 'rttm' (default) or 'seglst' (meeteval format).",
    )

    parser.add_argument(
        "--uem",
        default=None,
        help="UEM (Universal English Mask) file: speech segments to consider during merging. "
        "RTTM format file specifying speech activity regions.",
    )

    parser.add_argument(
        "--label-mapping",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Label mapping algorithm for merging: 'hungarian' (optimal, slower) or 'greedy' (faster, default: hungarian).",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (sets numpy, torch, and python random seeds).",
    )

    args = parser.parse_args()

    # Set random seed if specified
    if args.random_seed is not None:
        import random

        logger.info(f"Setting random seed to {args.random_seed}")
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)

    # Auto-detect output format from file extension
    output_ext = Path(args.output).suffix.lower()
    if output_ext == ".seglst":
        output_format = "seglst"
    elif output_ext == ".rttm":
        output_format = "rttm"
    else:
        output_format = args.output_format  # Use specified format or default

    # Handle merge-only mode
    if args.merge_only:
        # Merge-only: use --audio to specify RTTM or JSON files
        try:
            from dover_lap import DiariaziationComparator
        except ImportError:
            logger.error(
                "dover-lap is required for merging. Install it with: pip install dover-lap"
            )
            sys.exit(1)

        input_files = args.audio

        logger.info("=" * 60)
        logger.info("Diarization Merging Task")
        logger.info("=" * 60)
        logger.info(f"  Input files: {len(input_files)}")
        for i, f in enumerate(input_files, 1):
            logger.info(f"    [{i}] {f}")
        logger.info(f"  Output file: {args.output}")
        logger.info(f"  Output format: {output_format}")
        logger.info(f"  Label mapping: {args.label_mapping}")
        if args.uem:
            logger.info(f"  UEM file: {args.uem}")
        if args.threshold is not None:
            logger.info(f"  Threshold: {args.threshold}")
        else:
            logger.info(f"  Threshold: automatic (optimal)")
        logger.info("=" * 60)

        logger.info(f"Merging {len(input_files)} files...")

        # Load diarization data from files (supports JSON and RTTM formats)
        try:
            from meeteval.io import load
        except ImportError:
            logger.error(
                "meeteval is required for JSON/RTTM loading. Install it with: pip install meeteval"
            )
            sys.exit(1)

        rttm_strings = []
        for input_file in input_files:
            file_ext = Path(input_file).suffix.lower()

            logger.info(f"Loading {file_ext} file: {input_file}")

            try:
                # Use meeteval to load JSON or RTTM
                diarization_data = load(input_file)
                # Convert to RTTM string
                rttm_str = str(diarization_data)
                rttm_strings.append(rttm_str)
                logger.debug(
                    f"Loaded {len(list(diarization_data.itertracks()))} speaker turns from {input_file}"
                )
            except Exception as e:
                logger.error(f"Failed to load {input_file}: {e}")
                sys.exit(1)

        # Load UEM if specified
        uem_data = None
        if args.uem:
            logger.info(f"Loading UEM file: {args.uem}")
            try:
                uem_data = load(args.uem)
            except Exception as e:
                logger.error(f"Failed to load UEM file {args.uem}: {e}")
                sys.exit(1)

        # Merge using dover-lap
        comparator = DiariaziationComparator()

        # Set algorithm based on --label-mapping
        algorithm = args.label_mapping

        if args.threshold is not None:
            logger.info(f"Using fixed threshold {args.threshold}")
            merged = comparator.merge(
                rttm_strings, threshold=args.threshold, uem=uem_data, algorithm=algorithm
            )
        else:
            logger.info(f"Finding optimal threshold using {algorithm} algorithm...")
            merged = comparator.optimal_threshold(rttm_strings, uem=uem_data, algorithm=algorithm)

        # Save result
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        _save_diarization(merged, output_file, output_format)

        # Log merged stats
        num_speakers = len(set(label for _, _, label in merged.itertracks(yield_label=True)))
        num_turns = len(list(merged.itertracks()))

        # Calculate total duration
        total_duration = 0.0
        for turn, _, _ in merged.itertracks(yield_label=True):
            total_duration = max(total_duration, turn.end)

        logger.info("=" * 60)
        logger.info("Diarization Merging Complete")
        logger.info("=" * 60)
        logger.info(f"  Number of speakers: {num_speakers}")
        logger.info(f"  Number of turns: {num_turns}")
        logger.info(f"  Total duration: {total_duration:.2f}s")
        logger.info(f"  Output file: {output_file}")
        logger.info("=" * 60)
        return

    # Load pipeline
    logger.info(f"Loading model {args.model}...")

    if not args.hf_token:
        logger.error(
            "--hf-token is required for diarization. Get it from https://huggingface.co/settings/tokens"
        )
        sys.exit(1)

    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=args.hf_token)
    pipeline.to(torch.device(args.device))

    # Load UEM if specified
    uem_data = None
    if args.uem:
        logger.info(f"Loading UEM file: {args.uem}")
        try:
            from meeteval.io import load

            uem_data = load(args.uem)
        except Exception as e:
            logger.error(f"Failed to load UEM file {args.uem}: {e}")
            sys.exit(1)

    # Print task information
    logger.info("=" * 60)
    logger.info("Diarization Task")
    logger.info("=" * 60)
    logger.info(f"  Input audio files: {len(args.audio)}")
    for i, f in enumerate(args.audio, 1):
        logger.info(f"    [{i}] {f}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Device: {args.device}")
    if args.channels:
        logger.info(f"  Channels: {args.channels}")
    if args.uem:
        logger.info(f"  UEM file: {args.uem}")
    logger.info(f"  Label mapping: {args.label_mapping}")
    if args.threshold is not None:
        logger.info(f"  Threshold: {args.threshold}")
    else:
        logger.info(f"  Threshold: automatic (optimal)")
    if args.random_seed is not None:
        logger.info(f"  Random seed: {args.random_seed}")
    logger.info(f"  Output file: {args.output}")
    logger.info(f"  Output format: {output_format}")
    logger.info("=" * 60)

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
                        logger.error(f"Channel index {ch_idx} out of range [0, {num_channels-1}]")
                        sys.exit(1)
                logger.info(
                    f"Extracting {len(channels_to_extract)} selected channels from {audio_file}: {channels_to_extract}..."
                )
            else:
                channels_to_extract = list(range(num_channels))
                logger.info(f"Extracting {num_channels} channels from {audio_file}...")

            # Extract channels
            with tempfile.TemporaryDirectory() as tmpdir:
                for ch_idx in channels_to_extract:
                    ch_file = Path(tmpdir) / f"ch{ch_idx}.wav"
                    sf.write(ch_file, audio[:, ch_idx], sr)
                    channel_files.append(str(ch_file))

                # Process channels and merge
                _process_and_merge_channels(
                    channel_files,
                    pipeline,
                    args.threshold,
                    args.output,
                    output_format,
                    uem=uem_data,
                    args=args,
                )
    else:
        # Multiple channel files
        if args.channels is not None:
            logger.warning("--channels is ignored when multiple audio files are provided")
        channel_files = args.audio
        logger.info(f"Processing {len(channel_files)} channel files...")
        _process_and_merge_channels(
            channel_files,
            pipeline,
            args.threshold,
            args.output,
            output_format,
            uem=uem_data,
            args=args,
        )

    logger.info(f"Merged diarization saved to {args.output}")


def _process_and_merge_channels(
    channel_files: List[str],
    pipeline,
    threshold: Optional[float],
    output_path: str,
    output_format: str = "rttm",
    uem=None,
    args=None,
) -> None:
    """Process each channel and merge results.

    Args:
        channel_files: List of channel audio files.
        pipeline: pyannote diarization pipeline.
        threshold: Optional fixed threshold for merging.
        output_path: Path to save output file.
        output_format: Output format: 'rttm' or 'seglst'.
        uem: Optional UEM (Universal English Mask) data for diarization.
        args: Parsed command-line arguments (for label_mapping, etc.).
    """
    from dover_lap import DiariaziationComparator

    diarizations = []
    rttm_strings = []

    for ch_idx, ch_file in enumerate(channel_files):
        logger.info(f"[Ch{ch_idx}] Running diarization on {ch_file}...")
        # Pass UEM to pipeline if available
        if uem is not None:
            diarization = pipeline(ch_file, uem=uem)
        else:
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

    # Prepare merge parameters
    merge_kwargs = {}
    if uem is not None:
        merge_kwargs["uem"] = uem
    if args and hasattr(args, "label_mapping"):
        merge_kwargs["algorithm"] = args.label_mapping

    if threshold is not None:
        # Use fixed threshold
        logger.info(f"Using fixed threshold {threshold}")
        merged = comparator.merge(rttm_strings, threshold=threshold, **merge_kwargs)
    else:
        # Find optimal threshold
        algorithm_str = (
            args.label_mapping if args and hasattr(args, "label_mapping") else "hungarian"
        )
        logger.info(f"Finding optimal threshold using {algorithm_str} algorithm...")
        merged = comparator.optimal_threshold(rttm_strings, **merge_kwargs)

    # Save result
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    _save_diarization(merged, output_file, output_format)

    # Log merged stats
    num_speakers = len(set(label for _, _, label in merged.itertracks(yield_label=True)))
    num_turns = len(list(merged.itertracks()))

    # Calculate total duration
    total_duration = 0.0
    for turn, _, _ in merged.itertracks(yield_label=True):
        total_duration = max(total_duration, turn.end)

    logger.info("=" * 60)
    logger.info("Diarization Processing Complete")
    logger.info("=" * 60)
    logger.info(f"  Channels processed: {len(channel_files)}")
    logger.info(f"  Number of speakers: {num_speakers}")
    logger.info(f"  Number of turns: {num_turns}")
    logger.info(f"  Total duration: {total_duration:.2f}s")
    logger.info(f"  Output file: {output_file}")
    logger.info("=" * 60)


def _save_diarization(diarization, output_file: Path, output_format: str) -> None:
    """Save diarization result in the specified format.

    Args:
        diarization: Diarization object.
        output_file: Path to save the output file.
        output_format: Output format: 'rttm' or 'seglst'.
    """
    if output_format.lower() == "seglst":
        # Convert RTTM to SegLST format using meeteval
        try:
            from meeteval.io import load_rttm, write_seglst
        except ImportError:
            logger.error(
                "meeteval is required for SegLST format. Install it with: pip install meeteval"
            )
            sys.exit(1)

        # Write RTTM to temporary file, then convert to SegLST
        import tempfile as tmp_module

        with tmp_module.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as tmp_rttm:
            tmp_rttm_path = tmp_rttm.name
            diarization.write_rttm(tmp_rttm)

        try:
            # Load RTTM and write as SegLST
            rttm_data = load_rttm(tmp_rttm_path)
            write_seglst(rttm_data, str(output_file))
            logger.info(f"Saved output in SegLST format: {output_file}")
        finally:
            # Clean up temporary file
            Path(tmp_rttm_path).unlink(missing_ok=True)
    else:
        # Default: RTTM format
        with open(output_file, "w") as f:
            diarization.write_rttm(f)
        logger.info(f"Saved output in RTTM format: {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
