"""Command-line interface for GSS diarization-based enhancement."""

import argparse
import json
import sys
import logging
import tempfile
from pathlib import Path
from typing import Optional, Sequence, List

import numpy as np
import soundfile as sf

from ._frontend import GSS

logger = logging.getLogger(__name__)


def _merge_overlapping_segments(segments: List) -> List:
    """Merge overlapping speaker segments into continuous denoising regions.
    
    Args:
        segments: List of meeteval Segment objects or dicts
        
    Returns:
        List of merged dicts with normalized format
    """
    if not segments:
        return []
    
    # Helper to get segment properties regardless of type
    def get_start(seg):
        if isinstance(seg, dict):
            return seg.get("start", 0)
        return getattr(seg, "start", getattr(seg, "begin_time", 0))
    
    def get_end(seg):
        if isinstance(seg, dict):
            return seg.get("end", 0)
        end = getattr(seg, "end", None)
        if end is None and hasattr(seg, "duration"):
            end = float(get_start(seg)) + float(seg.duration)
        return end
    
    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: get_start(s))
    
    # Merge overlapping segments
    merged = []
    current_start = get_start(sorted_segs[0])
    current_end = get_end(sorted_segs[0])
    
    for seg in sorted_segs[1:]:
        seg_start = get_start(seg)
        seg_end = get_end(seg)
        if seg_start <= current_end:
            # Overlapping or adjacent: extend current region
            current_end = max(current_end, seg_end)
        else:
            # No overlap: save current region and start new one
            merged.append({
                "segment": f"{current_start:.2f}-{current_end:.2f}",
                "speaker": "all_speakers",
                "start": float(current_start),
                "end": float(current_end),
            })
            current_start = seg_start
            current_end = seg_end
    
    # Don't forget the last segment
    merged.append({
        "segment": f"{current_start:.2f}-{current_end:.2f}",
        "speaker": "all_speakers",
        "start": float(current_start),
        "end": float(current_end),
    })
    
    return merged


def main():
    """CLI entry point for enhance_from_diarization."""
    # Check for required dependency: meeteval
    try:
        import meeteval
    except ImportError:
        logger.error(
            "meeteval is required for diarization file parsing.\n"
            "Install it with: pip install meeteval"
        )
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Enhance speech in diarized segments using GSS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single speaker
  gss-enhance-diarization \\
    --audio meeting.wav \\
    --diarization meeting.rttm \\
    --speaker-id 0 \\
    --output-dir ./enhanced

  # All speakers
  gss-enhance-diarization \\
    --audio meeting.wav \\
    --diarization meeting.rttm \\
    --output-dir ./enhanced

  # Multiple audio channels with multiple diarization files
  gss-enhance-diarization \\
    --audio ch0.wav ch1.wav ch2.wav \\
    --diarization part1.rttm part2.rttm \\
    --diarization-time-concat \\
    --channel-length-mode trim \\
    --output-dir ./enhanced

  # With UEM and custom context
  gss-enhance-diarization \\
    --audio meeting.wav \\
    --diarization meeting.rttm \\
    --uem meeting.uem \\
    --context-left 10.0 \\
    --context-right 10.0 \\
    --output-dir ./enhanced
""",
    )

    # Audio files
    parser.add_argument(
        "--audio",
        nargs="+",
        required=True,
        help="Audio file path(s): *.wav, *.flac, *.mp3, etc.",
    )

    # Diarization files
    parser.add_argument(
        "--diarization",
        nargs="+",
        required=True,
        help="Diarization file path(s): *.rttm, *.ctm, *.stm, etc.",
    )

    # Speaker selection
    parser.add_argument(
        "--speaker-id",
        type=str,
        default=None,
        help="Target speaker: int (index), str (label), or None (all speakers, default). "
             "Ignored if --denoising-only is set.",
    )

    parser.add_argument(
        "--denoising-only",
        action="store_true",
        help="Enable denoising-only mode: remove background noise while keeping all speakers. "
             "Outputs one denoised segment per time interval where any speaker is active, "
             "regardless of speaker identity. Useful for iterative denoising: "
             "denoise -> re-diarize (higher confidence) -> separate speakers. "
             "Ignores --speaker-id. Useful for meeting recording preprocessing.",
    )

    # Diarization options
    parser.add_argument(
        "--diarization-format",
        type=str,
        default=None,
        help="Diarization format hint (e.g. 'rttm'). Auto-detect if omitted.",
    )

    parser.add_argument(
        "--diarization-session-id",
        type=str,
        default=None,
        help="Session ID filter for multi-session diarization files.",
    )

    parser.add_argument(
        "--diarization-time-concat",
        action="store_true",
        help="When multiple diarization files: concatenate in time order.",
    )

    parser.add_argument(
        "--diarization-concat-gap",
        type=float,
        default=0.0,
        help="Gap (seconds) between concatenated diarization files.",
    )

    parser.add_argument(
        "--diarization-offsets",
        type=float,
        nargs="+",
        default=None,
        help="Explicit per-file time offsets (seconds) for diarization files.",
    )

    # UEM and valid regions
    parser.add_argument(
        "--uem",
        type=str,
        default=None,
        help="UEM file path for valid scoring regions.",
    )

    parser.add_argument(
        "--uem-format",
        type=str,
        default=None,
        help="UEM format hint (e.g. 'uem').",
    )

    parser.add_argument(
        "--valid-regions",
        type=float,
        nargs="+",
        default=None,
        help="Valid region boundaries: start1 end1 [start2 end2 ...] in seconds.",
    )

    # Audio channel handling
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=None,
        help="For single multi-channel file: select specific channels (0-based indexing). "
             "If not specified, all channels are used. Example: --channels 0 2 (use channels 0 and 2).",
    )

    parser.add_argument(
        "--channel-length-mode",
        type=str,
        choices=["error", "trim", "pad"],
        default="error",
        help="Handle mismatched channel lengths when using multiple audio files.",
    )

    parser.add_argument(
        "--channel-offsets",
        type=float,
        nargs="+",
        default=None,
        help="Per-channel temporal shifts (samples or seconds).",
    )

    parser.add_argument(
        "--channel-offset-unit",
        type=str,
        choices=["samples", "seconds"],
        default="samples",
        help="Unit for channel-offsets.",
    )

    # Enhancement options
    parser.add_argument(
        "--context-left",
        type=float,
        default=15.0,
        help="Left context in seconds (default: 15.0).",
    )

    parser.add_argument(
        "--context-right",
        type=float,
        default=15.0,
        help="Right context in seconds (default: 15.0).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "oom_fallback"],
        default="oom_fallback",
        help="Enhancement mode.",
    )

    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="Frequency axis chunk count for memory control.",
    )

    # Distributed processing (slurm, etc.)
    parser.add_argument(
        "--num-groups",
        type=int,
        default=1,
        help="Number of groups to partition segments into for distributed processing (default: 1 = no partitioning).",
    )

    parser.add_argument(
        "--group-id",
        type=int,
        default=0,
        help="Zero-based group index to process (default: 0). Must be 0 <= group_id < num_groups.",
    )

    # GSS parameters
    parser.add_argument(
        "--stft-fft-length",
        type=int,
        default=1024,
        help="FFT length for STFT.",
    )

    parser.add_argument(
        "--stft-hop-length",
        type=int,
        default=256,
        help="Hop length for STFT.",
    )

    parser.add_argument(
        "--bss-iterations",
        type=int,
        default=20,
        help="GSS iterations.",
    )

    parser.add_argument(
        "--enable-dereverb",
        action="store_true",
        default=True,
        help="Enable WPE dereverberation (default: True).",
    )

    parser.add_argument(
        "--disable-dereverb",
        action="store_false",
        dest="enable_dereverb",
        help="Disable WPE dereverberation.",
    )

    parser.add_argument(
        "--mc-ref-channel",
        type=str,
        default="max_snr",
        help="Channel selection for beamformer output: 'max_snr' (auto-select by SNR, default), "
             "'none' (output all channels, MIMO mode), or integer channel index (0-based).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu'.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for enhanced audio files.",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default="wav",
        help="Output audio format: 'wav', 'flac', 'ogg', etc. (default: wav). "
             "Supported formats depend on installed soundfile backends.",
    )

    parser.add_argument(
        "--output-seglst",
        type=str,
        default=None,
        nargs="?",
        const="segments",
        help="Save segment metadata as SegLST and JSON formats for later embedding. "
             "Optionally specify filename prefix (default: 'segments'). "
             "Useful for distributed processing: prefix can include placeholders like {group_id}. "
             "Example: --output-seglst segments_group{group_id} "
             "Will generate: segments_group0.seglst, segments_group0.json, etc.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Handle channel selection for single multi-channel files
    audio_path = list(args.audio)  # Convert to list to unify handling
    
    if args.channels is not None:
        if len(args.audio) > 1:
            logger.warning("--channels is ignored when multiple audio files are provided")
        elif len(args.audio) == 1:
            # Single multi-channel file: extract selected channels
            audio_file = args.audio[0]
            try:
                audio, sr = sf.read(audio_file)
            except Exception as exc:
                logger.error(f"Failed to read audio file {audio_file}: {exc}")
                sys.exit(1)

            if audio.ndim == 1:
                logger.warning(
                    f"Input file {audio_file} is mono. --channels is ignored."
                )
            else:
                # Validate channel indices
                num_channels = audio.shape[1]
                for ch_idx in args.channels:
                    if ch_idx < 0 or ch_idx >= num_channels:
                        logger.error(
                            f"Channel index {ch_idx} out of range [0, {num_channels-1}]"
                        )
                        sys.exit(1)

                logger.info(
                    f"Extracting {len(args.channels)} selected channels from {audio_file}: {args.channels}..."
                )

                # Extract selected channels to separate temporary files
                tmpdir = tempfile.mkdtemp()
                audio_path = []
                for ch_idx in args.channels:
                    ch_file = Path(tmpdir) / f"ch{ch_idx}.wav"
                    sf.write(str(ch_file), audio[:, ch_idx], sr)
                    audio_path.append(str(ch_file))
                
                # Note: temporary directory will persist for processing
                logger.debug(f"Temporary channel files stored in {tmpdir}")

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Normalize to single item or list
    audio_path = args.audio[0] if len(args.audio) == 1 else args.audio
    diarization = args.diarization[0] if len(args.diarization) == 1 else args.diarization

    # Process valid_regions if provided
    valid_regions = None
    if args.valid_regions:
        if len(args.valid_regions) % 2 != 0:
            parser.error("--valid-regions must have even number of values (start1 end1 start2 end2 ...)")
        valid_regions = [
            (args.valid_regions[i], args.valid_regions[i + 1])
            for i in range(0, len(args.valid_regions), 2)
        ]

    # Parse speaker_id
    speaker_id: Optional[int | str] = None
    if args.denoising_only:
        logger.info("Denoising-only mode: ignoring --speaker-id and processing all speakers")
        speaker_id = None
    elif args.speaker_id is not None:
        if args.speaker_id.lower() == "none":
            speaker_id = None
        elif args.speaker_id.isdigit():
            speaker_id = int(args.speaker_id)
        else:
            speaker_id = args.speaker_id

    # Parse mc_ref_channel
    mc_ref_channel = args.mc_ref_channel
    if mc_ref_channel.lower() == "none":
        mc_ref_channel = None
    elif mc_ref_channel.isdigit():
        mc_ref_channel = int(mc_ref_channel)
    # else: keep as string (e.g., 'max_snr')

    # Initialize GSS
    logger.info("Initializing GSS frontend...")
    frontend = GSS(
        stft_fft_length=args.stft_fft_length,
        stft_hop_length=args.stft_hop_length,
        enable_dereverb=args.enable_dereverb,
        mc_ref_channel=mc_ref_channel,
        bss_iterations=args.bss_iterations,
        device=args.device,
    )

    # Run enhancement
    logger.info("Running enhancement from diarization...")
    
    # Print basic information about the task
    try:
        import meeteval
        
        # Load diarization to get statistics
        diar_file = str(diarization) if isinstance(diarization, (str, Path)) else str(diarization[0])
        diar_data = meeteval.io.load(diar_file)
        
        # Get unique speakers
        speakers = set()
        total_duration = 0.0
        for segment in diar_data:
            # Handle both dict and Segment object formats
            if isinstance(segment, dict):
                speaker = segment.get("speaker", segment.get("spk", "unknown"))
                end = segment.get("end", segment.get("end_time", segment.get("end_sample", 0)))
            else:
                speaker = segment.speaker
                end = segment.end
            
            speakers.add(speaker)
            total_duration = max(total_duration, end)
        
        num_speakers = len(speakers)
        num_segments = len(diar_data)
        
        logger.info("=" * 60)
        logger.info("Enhancement Task Summary")
        logger.info("=" * 60)
        logger.info(f"  Audio file: {audio_path}")
        logger.info(f"  Diarization file: {diar_file}")
        logger.info(f"  Number of speakers: {num_speakers}")
        logger.info(f"  Speaker labels: {sorted(speakers)}")
        logger.info(f"  Number of segments: {num_segments}")
        logger.info(f"  Total duration: {total_duration:.2f}s")
        if args.denoising_only:
            logger.info(f"  Mode: DENOISING-ONLY (merging all speakers)")
        elif speaker_id is not None:
            logger.info(f"  Mode: Single speaker extraction (speaker_id={speaker_id})")
        else:
            logger.info(f"  Mode: Multi-speaker enhancement")
        logger.info("=" * 60)
    except Exception as e:
        logger.warning(f"Could not load diarization for stats: {type(e).__name__}: {e}")
        import traceback
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
    
    # For denoising-only mode, merge all speaker segments
    enhancement_diarization = diarization
    if args.denoising_only:
        logger.info("Denoising-only mode: merging all speaker segments...")
        try:
            import meeteval
            
            # Load diarization
            diar_loaded = meeteval.io.load(
                str(diarization) if isinstance(diarization, (str, Path)) else str(diarization[0])
            )
            
            # If multiple diarization files, load and merge them
            if isinstance(diarization, (list, tuple)) and len(diarization) > 1:
                for diar_file in diarization[1:]:
                    diar_next = meeteval.io.load(str(diar_file))
                    diar_loaded = diar_loaded.union(diar_next)
            
            # Convert all segments to use a single unified speaker label
            unified_segments = []
            for segment in diar_loaded:
                # Handle both RTTMLine and dict formats
                if isinstance(segment, dict):
                    seg_dict = {
                        "segment": segment.get("segment", f"{segment.get('start', 0):.2f}-{segment.get('end', 0):.2f}"),
                        "speaker": "all_speakers",
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                    }
                else:
                    # RTTMLine or similar object
                    start = getattr(segment, "start", getattr(segment, "begin_time", 0))
                    end = getattr(segment, "end", None)
                    if end is None and hasattr(segment, "duration"):
                        end = float(start) + float(segment.duration)
                    seg_dict = {
                        "segment": f"{start:.2f}-{end:.2f}",
                        "speaker": "all_speakers",
                        "start": float(start),
                        "end": float(end),
                    }
                unified_segments.append(seg_dict)
            
            # Merge overlapping segments
            # This creates continuous denoised regions
            merged_segments = _merge_overlapping_segments(unified_segments)
            
            # Create temporary diarization file for processing
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rttm', delete=False) as f:
                # Write in RTTM format
                for idx, seg in enumerate(merged_segments):
                    # RTTM format: <type> <file_id> <chnl> <begin_time> <duration> <ortho> <stype> <speaker_id> <confidence> <lookahead>
                    session_id = seg.get("session_id", "meeting")
                    start = float(seg["start"])
                    end = float(seg["end"])
                    duration = end - start
                    speaker = seg.get("speaker", "all_speakers")
                    f.write(f"SPEAKER {session_id} 1 {start:.2f} {duration:.2f} <NA> <NA> {speaker} <NA> <NA>\n")
                
                enhancement_diarization = f.name
                logger.info(f"Merged diarization: {len(merged_segments)} denoised regions")
                for seg in merged_segments:
                    logger.debug(f"  Region: {seg['start']:.2f}s - {seg['end']:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to merge diarization for denoising: {e}")
            sys.exit(1)
    
    try:
        segments = frontend.enhance_from_diarization(
            audio_path=audio_path,
            diarization=enhancement_diarization,
            speaker_id=speaker_id,
            diarization_format=args.diarization_format,
            diarization_session_id=args.diarization_session_id,
            diarization_time_concat=args.diarization_time_concat,
            diarization_concat_gap_seconds=args.diarization_concat_gap,
            diarization_offsets=args.diarization_offsets,
            uem=args.uem,
            uem_format=args.uem_format,
            valid_regions=valid_regions,
            channel_length_mode=args.channel_length_mode,
            channel_offsets=args.channel_offsets,
            channel_offset_unit=args.channel_offset_unit,
            context_left_seconds=args.context_left,
            context_right_seconds=args.context_right,
            mode=args.mode,
            num_chunks=args.num_chunks,
            num_groups=args.num_groups,
            group_id=args.group_id,
        )
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Print processing results summary
    logger.info("=" * 60)
    logger.info("Enhancement Processing")
    logger.info("=" * 60)

    # Process segments and save incrementally
    logger.info("Saving enhanced segments...")
    seglst_segments = []
    speakers_summary = {}
    total_segments = 0
    
    for idx, item in enumerate(segments):
        speaker = item["speaker"]
        seg_start = item["segment_start"]
        seg_end = item["segment_end"]
        sample_rate = item["sample_rate"]
        enhanced_audio = item["enhanced_audio"]
        segment_index = item["segment_index"]  # Use global index for consistent naming across groups

        total_segments += 1

        # Display progress
        progress = f"[segment {idx+1}]"
        logger.info(f"{progress} Processing segment: speaker={speaker}, time={seg_start:.2f}s-{seg_end:.2f}s")

        # Format output filename with specified extension
        output_format = args.output_format.lower().lstrip('.')  # Remove leading dot if present
        
        # In denoising-only mode, use "denoised" instead of speaker name
        label = "denoised" if args.denoising_only else speaker
        filename = f"{segment_index:03d}_{label}_{seg_start:.2f}_{seg_end:.2f}.{output_format}"
        filepath = output_dir / filename

        # Write with explicit subtype for lossless formats
        # soundfile expects (samples,) for mono or (samples, channels) for multi-channel
        # If multi-channel (channels, samples), need to transpose to (samples, channels)
        import torch
        audio_to_write = enhanced_audio
        
        # Convert torch tensor to numpy if needed
        if isinstance(audio_to_write, torch.Tensor):
            audio_to_write = audio_to_write.cpu().numpy()
        
        # Transpose if multi-channel
        if audio_to_write.ndim > 1:
            audio_to_write = audio_to_write.T  # (channels, samples) -> (samples, channels)
        
        # Set format and subtype
        subtype = None
        if output_format in {"wav", "flac", "aiff"}:
            subtype = "PCM_16"
        
        # soundfile.write requires format in lowercase or uppercase depending on libsndfile version
        sf.write(str(filepath), audio_to_write, sample_rate, subtype=subtype)
        logger.info(f"  Saved: {filepath}")
        
        # Track speaker counts for summary
        if speaker not in speakers_summary:
            speakers_summary[speaker] = 0
        speakers_summary[speaker] += 1
        
        # Collect metadata for SegLST using meeteval
        seglst_segments.append({
            "segment": f"{seg_start:.2f}-{seg_end:.2f}",
            "speaker": speaker,
            "start": seg_start,
            "end": seg_end,
            "audio_path": str(filepath.relative_to(output_dir.parent)),
            "sample_rate": sample_rate,
        })

    # Save SegLST if requested
    if args.output_seglst:
        import meeteval
        
        # Format filename with placeholders
        seglst_prefix = args.output_seglst.format(group_id=args.group_id)
        seglst_file = output_dir / f"{seglst_prefix}.seglst"
        seglst_json = output_dir / f"{seglst_prefix}.json"
        
        # Save as SegLST text format (meeteval standard)
        # Format: "segment_spec speaker_label"
        with open(seglst_file, "w") as f:
            for s in seglst_segments:
                f.write(f"{s['segment']} {s['speaker']}\n")
        
        # Also save JSON with audio paths + metadata for gss-embed
        with open(seglst_json, "w") as f:
            json.dump(seglst_segments, f, indent=2)
        
        logger.info(f"SegLST metadata saved: {seglst_file}")
        logger.info(f"SegLST JSON (for gss-embed) saved: {seglst_json}")

    # Print processing completion summary
    logger.info("=" * 60)
    logger.info("Enhancement Processing Complete")
    logger.info("=" * 60)
    logger.info(f"  Segments processed: {total_segments}")
    
    for speaker, count in sorted(speakers_summary.items()):
        logger.info(f"    Speaker '{speaker}': {count} segments")
    logger.info("=" * 60)

    logger.info("Done!")


if __name__ == "__main__":
    main()
