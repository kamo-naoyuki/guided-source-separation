"""Command-line tool for embedding enhanced speech segments back into original audio."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def main():
    """CLI entry point for embedding enhanced segments."""
    parser = argparse.ArgumentParser(
        description="Embed enhanced speech segments back into original audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single segment file
  gss-embed --segments enhanced/segments.json \\
    --audio meeting.wav \\
    --output-dir ./embedded

  # Multiple segment files (from distributed processing)
  gss-embed --segments enhanced/seg_group0.json enhanced/seg_group1.json \\
    --audio meeting.wav \\
    --output-dir ./embedded

  # With channel offsets (must match gss-enhance)
  gss-embed --segments enhanced/segments.json \\
    --audio ch0.wav ch1.wav ch2.wav \\
    --channel-offsets 0 -0.1 0.05 \\
    --channel-offset-unit seconds \\
    --output-dir ./embedded
""",
    )

    parser.add_argument(
        "--segments",
        nargs="+",
        required=True,
        help="Segment metadata file(s): SegLST format (.seglst) or JSON (.json). "
             "Can specify multiple files: --segments seg0.json seg1.json seg2.json",
    )

    parser.add_argument(
        "--audio",
        nargs="+",
        required=True,
        help="Original audio file path(s): single multi-channel file or multiple mono files "
             "(must match the input to gss-enhance in order and sample rate).",
    )

    parser.add_argument(
        "--channel-length-mode",
        type=str,
        default="error",
        choices=["error", "trim", "pad"],
        help="If multiple audio files have different lengths: 'error' (raise), "
             "'trim' (to shortest), 'pad' (zero-pad to longest). "
             "Must match gss-enhance --channel-length-mode (default: error).",
    )

    parser.add_argument(
        "--channel-offset-unit",
        type=str,
        default="samples",
        choices=["samples", "seconds"],
        help="Unit for channel offset: 'samples' (default) or 'seconds'. "
             "Must match the unit used in gss-enhance --channel-offset-unit.",
    )

    parser.add_argument(
        "--channel-offsets",
        type=float,
        nargs="+",
        default=None,
        help="Per-channel time offsets (must match gss-enhance input). "
             "Example: --channel-offsets 0 -0.1 0.05 (in samples or seconds). "
             "Required if gss-enhance used --channel-offsets.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for embedded audio files.",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default="wav",
        help="Output audio format: 'wav', 'flac', 'ogg', etc. (default: wav).",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Load all segment metadata files
    all_segments = []
    for seg_file in args.segments:
        seglst_file = Path(seg_file)
        if not seglst_file.exists():
            logger.error(f"Segment metadata file not found: {seglst_file}")
            sys.exit(1)
        
        seglst_data = _load_segments(seglst_file)
        if seglst_data is None:
            sys.exit(1)
        all_segments.extend(seglst_data)
        logger.info(f"Loaded {len(seglst_data)} segments from {seglst_file}")
    
    logger.info(f"Total {len(all_segments)} segments loaded")

    # Load original audio(s)
    audio_files = [Path(f) for f in args.audio]
    for af in audio_files:
        if not af.exists():
            logger.error(f"Audio file not found: {af}")
            sys.exit(1)

    original_audio, sr = _load_audio_files(
        audio_files, args.channel_length_mode
    )
    if original_audio is None:
        sys.exit(1)
    
    logger.info(
        f"Loaded {len(audio_files)} audio file(s): {original_audio.shape[0]} samples, "
        f"{original_audio.shape[1]} channels, {sr} Hz"
    )

    # Group segments by speaker
    segments_by_speaker: Dict[str, List[Dict]] = {}
    for seg in all_segments:
        speaker = seg["speaker"]
        if speaker not in segments_by_speaker:
            segments_by_speaker[speaker] = []
        segments_by_speaker[speaker].append(seg)

    logger.info(f"Found {len(segments_by_speaker)} speakers")

    # Print task information
    logger.info("=" * 60)
    logger.info("Embedding Task")
    logger.info("=" * 60)
    logger.info(f"  Segment metadata files: {len(args.segments)}")
    for i, f in enumerate(args.segments, 1):
        logger.info(f"    [{i}] {f}")
    logger.info(f"  Total segments: {len(all_segments)}")
    logger.info(f"  Speakers: {len(segments_by_speaker)}")
    for speaker in sorted(segments_by_speaker.keys()):
        count = len(segments_by_speaker[speaker])
        logger.info(f"    - {speaker}: {count} segments")
    logger.info(f"  Audio files: {len(args.audio)}")
    for i, f in enumerate(args.audio, 1):
        logger.info(f"    [{i}] {f}")
    logger.info(f"  Audio shape: {original_audio.shape[0]} samples, {original_audio.shape[1]} channels, {sr} Hz")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Output format: {args.output_format}")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each speaker
    num_saved = 0
    for speaker, segments in segments_by_speaker.items():
        logger.info(f"Processing speaker: {speaker}")
        embedded_audio = _embed_speaker_segments(
            original_audio,
            sr,
            segments,
            speaker,
            args.output_format,
            channel_offsets=args.channel_offsets,
            channel_offset_unit=args.channel_offset_unit,
        )

        if embedded_audio is None:
            logger.warning(f"Skipping speaker {speaker} due to errors")
            continue

        # Save embedded audio
        output_format = args.output_format.lower().lstrip(".")
        output_file = output_dir / f"{speaker}.{output_format}"
        subtype = "PCM_16" if output_format in {"wav", "flac", "aiff"} else None
        sf.write(
            str(output_file),
            embedded_audio,
            sr,
            subtype=subtype,
            format=output_format.upper(),
        )
        logger.info(f"  Saved: {output_file}")
        num_saved += 1

    logger.info("=" * 60)
    logger.info("Embedding Complete")
    logger.info("=" * 60)
    logger.info(f"  Speakers processed: {len(segments_by_speaker)}")
    logger.info(f"  Files saved: {num_saved}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)

    logger.info("Done!")


def _load_audio_files(
    audio_files: List[Path],
    channel_length_mode: str = "error",
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Load audio file(s) and combine into multi-channel array.

    Args:
        audio_files: List of audio file paths
        channel_length_mode: How to handle length mismatches: "error", "trim", or "pad"

    Returns:
        (audio_array, sample_rate) or (None, None) if error
    """
    if len(audio_files) == 1:
        # Single file: load as-is
        audio, sr = sf.read(str(audio_files[0]))
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        return audio, sr
    else:
        # Multiple files: load each as a channel
        channels = []
        sr = None
        
        for i, af in enumerate(audio_files):
            try:
                ch_audio, ch_sr = sf.read(str(af))
            except Exception as e:
                logger.error(f"Failed to load {af}: {e}")
                return None, None
            
            # Check sample rate consistency
            if sr is None:
                sr = ch_sr
            elif ch_sr != sr:
                logger.error(
                    f"Sample rate mismatch: {af} has {ch_sr} Hz, "
                    f"but first file has {sr} Hz"
                )
                return None, None
            
            # Ensure 1D
            if ch_audio.ndim > 1:
                logger.error(
                    f"Multi-channel audio file {af} not supported in multi-file mode. "
                    f"Use single multi-channel file or split into mono files."
                )
                return None, None
            
            channels.append(ch_audio)
        
        # Align lengths
        lengths = [len(ch) for ch in channels]
        min_len = min(lengths)
        max_len = max(lengths)
        
        if min_len != max_len:
            if channel_length_mode == "error":
                logger.error(
                    f"Audio files have different lengths: {lengths}. "
                    f"Use --channel-length-mode trim or pad, or provide same-length files."
                )
                return None, None
            elif channel_length_mode == "trim":
                logger.warning(f"Trimming to shortest: {lengths} -> {min_len}")
                channels = [ch[:min_len] for ch in channels]
            elif channel_length_mode == "pad":
                logger.warning(f"Padding to longest: {lengths} -> {max_len}")
                channels = [
                    np.pad(ch, (0, max_len - len(ch)), mode="constant", constant_values=0)
                    for ch in channels
                ]
        
        # Stack as columns (samples, channels)
        return np.column_stack(channels), sr


def _load_segments(seglst_file: Path) -> list | None:
    """Load segments from SegLST (.seglst) or JSON (.json) file.

    Args:
        seglst_file: Path to segment metadata file

    Returns:
        List of segment dicts or None if error
    """
    if seglst_file.suffix == ".json":
        # Load JSON format (has audio_path field)
        with open(seglst_file, "r") as f:
            return json.load(f)
    
    elif seglst_file.suffix == ".seglst":
        # Load meeteval SegLST format
        try:
            import meeteval
        except ImportError:
            logger.error("meeteval is required for .seglst format. Install with: pip install meeteval")
            return None
        
        # Load SegLST via meeteval
        seg_list = meeteval.io.load(str(seglst_file))
        
        # Look for accompanying JSON with audio paths
        json_file = seglst_file.parent / "segments.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                json_data = json.load(f)
            # Merge SegLST with JSON audio paths
            return json_data
        else:
            logger.error(
                f"SegLST file {seglst_file} found, but corresponding "
                f"segments.json not found. Both are generated by gss-enhance --output-seglst."
            )
            return None
    else:
        logger.error(f"Unsupported segment file format: {seglst_file.suffix}")
        return None


def _embed_speaker_segments(
    original_audio: np.ndarray,
    sr: int,
    segments: List[Dict],
    speaker: str,
    output_format: str,
    channel_offsets: List[float] | None = None,
    channel_offset_unit: str = "samples",
) -> np.ndarray | None:
    """Embed enhanced segments for a single speaker into original audio.

    Args:
        original_audio: Original audio (samples, channels)
        sr: Sample rate
        segments: List of segment dicts with start, end, audio_path
        speaker: Speaker label
        output_format: Output format name (for validation)
        channel_offsets: Per-channel time offsets (must match gss-enhance input)
        channel_offset_unit: Unit of channel_offsets ("samples" or "seconds")

    Returns:
        Embedded audio (samples, channels) or None if error
    """
    # Validate channel_offsets
    if channel_offsets is not None:
        if len(channel_offsets) != original_audio.shape[1]:
            logger.error(
                f"Channel offset mismatch: {len(channel_offsets)} offsets provided, "
                f"but audio has {original_audio.shape[1]} channels. "
                f"Must match gss-enhance --channel-offsets."
            )
            return None
    
    # Convert channel offsets to samples if needed
    if channel_offsets is not None and channel_offset_unit == "seconds":
        channel_offsets_samples = [int(offset * sr) for offset in channel_offsets]
    else:
        channel_offsets_samples = channel_offsets or [0] * original_audio.shape[1]
    
    # Start with a copy of original audio
    result = original_audio.copy()

    for seg in segments:
        start_time = seg["start"]
        end_time = seg["end"]
        audio_path = seg["audio_path"]
        seg_sr = seg.get("sample_rate", sr)

        # Convert time to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        duration_samples = end_sample - start_sample

        # Load enhanced segment
        try:
            enhanced_seg, _ = sf.read(audio_path)
        except Exception as e:
            logger.error(f"Failed to load enhanced segment {audio_path}: {e}")
            return None

        # Handle channel dimension
        if enhanced_seg.ndim == 1:
            enhanced_seg = enhanced_seg[:, np.newaxis]

        # Validate
        if enhanced_seg.shape[1] != result.shape[1]:
            logger.error(
                f"Channel mismatch for segment {audio_path}: "
                f"enhanced has {enhanced_seg.shape[1]} channels, "
                f"but original has {result.shape[1]} channels. "
                f"Verify gss-enhance input audio had {result.shape[1]} channels."
            )
            return None

        if enhanced_seg.shape[0] != duration_samples:
            logger.warning(
                f"Segment {audio_path} has {enhanced_seg.shape[0]} samples, "
                f"expected {duration_samples}. Truncating/padding."
            )
            if enhanced_seg.shape[0] < duration_samples:
                # Pad with silence
                pad_width = ((0, duration_samples - enhanced_seg.shape[0]), (0, 0))
                enhanced_seg = np.pad(enhanced_seg, pad_width, mode="constant", constant_values=0)
            else:
                # Truncate
                enhanced_seg = enhanced_seg[:duration_samples, :]

        # Embed into result (accounting for channel offsets)
        for ch_idx in range(result.shape[1]):
            ch_offset = channel_offsets_samples[ch_idx]
            start_in_original = start_sample + ch_offset
            end_in_original = start_in_original + duration_samples
            
            # Validate boundaries
            if start_in_original < 0 or end_in_original > result.shape[0]:
                logger.warning(
                    f"Channel {ch_idx}: segment [{start_in_original}, {end_in_original}) "
                    f"exceeds audio bounds [0, {result.shape[0]}). "
                    f"Clipping to valid region."
                )
                valid_start = max(0, start_in_original)
                valid_end = min(result.shape[0], end_in_original)
                valid_len = valid_end - valid_start
                result[valid_start:valid_end, ch_idx] = enhanced_seg[:valid_len, ch_idx]
            else:
                result[start_in_original:end_in_original, ch_idx] = enhanced_seg[:, ch_idx]

    return result


if __name__ == "__main__":
    main()
