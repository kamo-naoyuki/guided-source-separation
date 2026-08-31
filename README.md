# gss-frontend

[![Test](https://github.com/kamo-naoyuki/guided-source-separation/workflows/Test/badge.svg)](https://github.com/kamo-naoyuki/guided-source-separation/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/kamo-naoyuki/guided-source-separation/main.svg)](https://results.pre-commit.ci/latest/github/kamo-naoyuki/guided-source-separation/main)

A minimal, self-contained implementation of the NeMo-based GSS (Guided Source
Separation) front-end for speech enhancement.

Extracted from the [CHiME-8 DASR NeMo baseline](https://github.com/chimechallenge/C8DASR-Baseline-NeMo),
`GSS` has been stripped of its batch-processing infrastructure, lhotse
dependencies, and DataLoader boilerplate so that it can be called **one utterance
at a time** with a straightforward numpy API.

All operations are differentiable: when `audio` and `activity` are
`torch.Tensor`, gradients propagate through dereverberation, mask estimation,
and beamforming, making the modules suitable for **end-to-end training**
(see [`examples/tensor_backprop.py`](examples/tensor_backprop.py)).

## What is GSS?

Guided Source Separation (GSS) is a multichannel speech-enhancement front end
for separating a target speaker from overlapping speakers and background noise.
It is typically used in far-field meeting transcription, diarization-assisted
ASR, or other multi-microphone speech pipelines where speaker activity
information is available.

The method became widely used in the CHiME distant-speech / meeting-recognition
line of work and later appeared in the CHiME-8 DASR NeMo baseline, from which
this repository was extracted.

At a high level, GSS uses speaker activity to guide time-frequency mask
estimation, then applies multichannel front-end processing such as WPE
dereverberation and mask-based beamforming to enhance one speaker at a time.

## Dependencies

```
torch
torchaudio
numpy
soundfile
meeteval  # for diarization text/file loading
```

No NeMo dependency required. All necessary modules are bundled in `src/gss_frontend/_modules.py`.

Optional

```
pyannote  # for diarization
dover-lap # for combining diarization results
```

## Installation

```bash
pip install -e .

# For diarization support (pyannote + dover-lap)
pip install -e ".[diarization]"
```

Run the tests:

```bash
pip install -e .
python -m pytest tests/ -v
```

Run the example:

```bash
python examples/separate_speakers.py --device cpu --out-dir /tmp/gss_out
python examples/tensor_backprop.py --device cpu
```

## Usage

### Guided Enhancement

```python
import numpy as np
import soundfile as sf
from gss_frontend import GSS

# --- Initialize once ---
frontend = GSS(
    stft_fft_length=1024,
    stft_hop_length=256,
    bss_iterations=20,
    garbage_class=True,           # add extra always-active background/garbage class
    activity_aggregation="mean",  # "mean" | "max" | "any"
    device="cuda",
)

# --- Prepare inputs ---
# audio:    (channels, samples)  float32 numpy array
# activity: (speakers, samples) float array
#           binary {0,1} or soft confidence in [0,1]
#           each row marks the activity of one speaker
audio, sr = sf.read("meeting.wav", always_2d=True)   # (samples, ch)
audio = audio.T.astype(np.float32)                    # -> (ch, samples)

# Build activity from VAD output, diarization labels, or soft posteriors
num_speakers = 2
activity = np.zeros((num_speakers, audio.shape[1]), dtype=np.float32)
# activity[0, start_sample:end_sample] = 1.0  ...

# --- Enhance target speaker ---
enhanced = frontend.enhance(audio, activity, speaker_id=0)
# enhanced: (samples,) float32

# Long-form recording + one target segment:
# automatically uses left/right context (default: 15s each) for mask estimation
segment = frontend.enhance_segment(
    audio,
    activity,
    speaker_id=0,
    segment_start=125.4,
    segment_end=132.1,
    sample_rate=sr,
)
# segment: enhanced waveform for [125.4, 132.1] seconds only

# Segment mode with automatic OOM retry:
# segment = frontend.enhance_segment(
#     audio, activity, speaker_id=0,
#     segment_start=125.4, segment_end=132.1, sample_rate=sr,
#     mode="oom_fallback",
# )

# With leading/trailing context already prepended to audio:
# enhanced = frontend.enhance(audio_with_context, activity_with_context,
#                             speaker_id=0,
#                             left_context=left_samples,
#                             right_context=right_samples)

# Automatic OOM retry (splits frequency axis into finer chunks;
# falls back to per-stage CPU execution as a last resort):
# enhanced = frontend.enhance_auto(audio, activity, speaker_id=0)

sf.write("enhanced.wav", enhanced, sr)
```

Long-form file + diarization text/file workflow (via `meeteval.io.load`):

```python
from gss_frontend import GSS

frontend = GSS(device="cuda")

# diarization can be RTTM/UEM/etc. supported by meeteval.
# speaker_id can be:
# - int: speaker index (0-based; in lexicographically sorted unique labels)
# - str: exact diarization speaker label
# - list[int|str]: multiple speakers
# - omitted/None: all speakers
segments = frontend.enhance_from_diarization(
    audio_path="meeting.wav",      # or ["ch0.wav", "ch1.wav", ...]
    diarization="meeting.rttm",    # or ["spkA.rttm", "spkB.rttm", ...] (merged by default)
    # speaker_id=0,
    # diarization_format="rttm",   # optional (omit to let meeteval auto-detect)
    # diarization_session_id=None,   # optional for multi-session diarization
    # diarization_time_concat=True,  # optional: sequential time-concatenation
    # diarization_offsets=[0.0, 300.0], # optional explicit per-file time offsets (seconds)
    # uem="meeting.uem",            # optional: valid regions only
    # valid_regions=[(10.0, 120.0)], # optional: direct valid regions (seconds)
    # channel_length_mode="trim",   # for multi-file input: "error" | "trim" | "pad"
    # channel_offsets=[0, -120, 35], # optional: per-channel shift
    # channel_offset_unit="samples",# "samples" | "seconds"
    context_left_seconds=15.0,
    context_right_seconds=15.0,
    mode="standard",             # or "oom_fallback"
    # For distributed processing: partition into 4 groups, process group 0
    # num_groups=4,
    # group_id=0,                  # This job processes segments in group 0
)

# segments is a generator that yields results incrementally (memory-efficient)
for item in segments:
    print(
        item["speaker"],
        item["segment_start"],
        item["segment_end"],
        item["sample_rate"],
        item["enhanced_audio"].shape,
    )
```

### Blind Source Separation

When speaker activity annotations are unavailable, use **blind BSS mode** to separate all
sources using a uniform activity assumption. The method cannot internally distinguish
speakers from noise, so external classification based on statistical properties is recommended.

```python
# Basic blind BSS with fixed num_sources
result = frontend.enhance_unguided(audio, num_sources=3)

# Blind BSS with automatic OOM retry (recommended for large audio)
result = frontend.enhance_unguided_auto(audio, num_sources=3)

# Both return a dict with enhanced audio and statistical properties:
audio_separated = result['audio']          # separated audio (num_sources, samples)
masks = result['masks']                    # (num_sources, freq, frames)
eigenvalues = result['eigenvalues']        # (num_sources, freq, channels)
mahalanobis = result['mahalanobis']        # (num_sources, freq, frames)
occupancy = result['occupancy']            # (num_sources,)
temporal_variance = result['temporal_variance']  # (num_sources,)
condition_number = result['condition_number']    # (num_sources, freq)
```

#### Distinguishing Speakers from Noise

CACGMM clusters the time-frequency space but cannot label clusters as "speech" or "noise".
Use the returned statistics to classify each source:

| Statistic | Interpretation |
|-----------|-----------------|
| **condition_number** | λ_max / λ_min per frequency bin; high values (> 10) indicate concentrated subspace → likely **speech** |
| **occupancy** | Time-averaged mask value [0, 1]; high (> 0.3) → likely **speech**; low → likely **noise** |
| **temporal_variance** | Mask variance over time; high → on-off activation pattern → likely **speech**; low → background noise |

**Example heuristic classifier**:
```python
# Average statistics across frequency (eigenvalues → condition_number averaged per freq)
mean_condition = condition_number.mean(dim=1)  # (num_sources,)
is_speech = (mean_condition > 10) & (occupancy > 0.3) & (temporal_variance > 0.01)

speech_sources = [i for i in range(len(is_speech)) if is_speech[i]]
noise_sources = [i for i in range(len(is_speech)) if not is_speech[i]]
```

See [Complex Angular Central Gaussian Mixture Model (CACGMM)](https://ieeexplore.ieee.org/document/7760429)
for the underlying clustering algorithm and why eigenvalue analysis reveals speech structure.



### Command-line usage

The `gss-enhance` CLI tool processes audio files with diarization directly from the shell.
Use `--audio` for audio file(s) and `--diarization` for diarization file(s):

```bash
gss-enhance --audio meeting.wav --diarization meeting.rttm --device cuda --output-dir ./enhanced

# Process with explicit speaker ID
gss-enhance --audio meeting.wav --diarization meeting.rttm --speaker-id 0 --device cuda --output-dir ./enhanced

# Multiple speakers
gss-enhance --audio meeting.wav --diarization meeting.rttm --speaker-id spkA spkB --device cuda --output-dir ./enhanced

# Multi-file audio and diarization (merged by default)
gss-enhance --audio ch0.wav ch1.wav \
  --diarization meeting_part1.rttm meeting_part2.rttm \
  --channel-length-mode trim \
  --device cuda \
  --output-dir ./enhanced

# Output in different audio formats (WAV, FLAC, OGG)
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --output-format wav \
  --device cuda \
  --output-dir ./enhanced_wav

gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --output-format flac \
  --device cuda \
  --output-dir ./enhanced_flac

gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --output-format ogg \
  --device cuda \
  --output-dir ./enhanced_ogg

# With UEM and custom context
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --uem meeting.uem \
  --context-left 10.0 \
  --context-right 10.0 \
  --output-dir ./enhanced

# Use GPU with custom STFT settings
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --device cuda \
  --stft-fft-length 1024 \
  --stft-hop-length 256 \
  --bss-iterations 20 \
  --output-dir ./enhanced

# Skip WPE dereverberation (faster, no dereverberation processing)
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --disable-dereverb \
  --output-dir ./enhanced_no_dereverb

# Denoising-only mode
# Beamformer treats all speakers as targets and suppresses only background noise
# (instead of isolating each speaker individually)
# Useful for meeting preprocessing when speaker separation is not needed
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --denoising-only \
  --device cuda \
  --output-dir ./enhanced_denoised

# Input channel selection (for single multi-channel file)
# Process only specific channels from a multi-channel audio file
gss-enhance --audio meeting_4ch.wav --diarization meeting.rttm \
  --channels 0 2 \
  --device cuda \
  --output-dir ./enhanced

# Useful for microphone array selection or processing specific mics

# Beamformer reference channel selection (default: max_snr)
# By default, GSS automatically selects the channel with the highest output SNR
# (signal-to-noise ratio) after beamforming. Use --mc-ref-channel to override:

# Output all channels from beamformer (MIMO mode, no channel selection)
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --mc-ref-channel none \
  --output-dir ./enhanced_mimo

# Use specific channel (e.g., channel 0) instead of auto-selection
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --mc-ref-channel 0 \
  --output-dir ./enhanced_ch0

# Distributed processing (SLURM/Condor): partition into 4 groups, process group 0
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --num-groups 4 --group-id 0 \
  --output-dir ./enhanced/group0
# Run in parallel on 4 nodes/jobs with --group-id 1, 2, 3
```

#### Parallel processing with SLURM

Create a SLURM array job script `run_gss_slurm.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=gss-enhance
#SBATCH --array=0-3                    # 4 parallel jobs (groups 0-3)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1              # 1 GPU per job
#SBATCH --time=02:00:00
#SBATCH --output=logs/gss_%a.log

# Number of parallel groups
NUM_GROUPS=4
GROUP_ID=$SLURM_ARRAY_TASK_ID

# Run gss-enhance for this group
gss-enhance \
  --audio meeting.wav \
  --diarization meeting.rttm \
  --num-groups $NUM_GROUPS \
  --group-id $GROUP_ID \
  --device cuda \
  --output-dir ./enhanced/group_${GROUP_ID}

echo "Group $GROUP_ID completed at $(date)"
```

Submit and run:

```bash
mkdir -p logs
sbatch run_gss_slurm.sh

# Check status
squeue -u $USER

# After completion, merge results from all groups
ls -la enhanced/group_*/
```

Output files from all groups will be named consistently (e.g., `000_spkA_10.50_15.75.wav`, 
`001_spkB_20.00_28.30.wav`, etc.), making them easy to combine or process further.

See `gss-enhance --help` for all options.

### Denoising-only mode

By default, `gss-enhance` processes each speaker individually: the beamformer treats
each speaker as a target and suppresses other speakers + background noise simultaneously.
For meeting preprocessing, you may want to keep all speakers while removing only background
noise. Use `--denoising-only` to change the beamformer behavior: treat **all speakers as
targets** and suppress **only background noise**:

```bash
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --denoising-only \
  --output-dir ./enhanced_denoised
```

**How it works:**

- **Beamformer target**: All speakers combined (not individual speakers)
- **Suppression**: Background noise only (keeps all speech overlaps unchanged)
- Implementation: Merges all overlapping speaker regions into continuous denoising intervals,
  then applies beamforming with the merged regions as activity
- Output: One denoised segment per merged interval (combining all speakers)
- Ignores `--speaker-id` if provided
- Can be combined with distributed processing (`--num-groups`)

Output segments will be labeled `denoised` instead of speaker names.

### Embedding enhanced segments back into original audio

By default, `gss-enhance` outputs individual enhanced segments per speaker. To embed these
segments back into the original audio (with enhanced speech regions replacing originals),
use the `gss-embed` tool:

```bash
# Step 1: Generate enhanced segments + metadata (MIMO mode: all channels needed for embedding)
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --mc-ref-channel none \
  --output-seglst segments \
  --output-dir ./enhanced

# Step 2: Embed enhanced segments back into original audio
gss-embed --segments enhanced/segments.json \
  --audio meeting.wav \
  --output-dir ./embedded

# Output: embedded/spk0.wav, embedded/spk1.wav, etc.
```

**Output filename scheme:**

gss-embed generates output filenames based on speaker labels from the segments metadata:
- `{speaker}.{format}` (default format: wav)
- Speaker names come from segments.json (e.g., `spk0`, `spk1`, `denoised` for denoising-only mode)
- Output format can be specified with `--output-format` (wav, flac, ogg, etc.)

Examples:
```
embedded/spk0.wav
embedded/spk1.wav
embedded/denoised.wav  (if using --denoising-only mode)
```

**For distributed processing:**

When using `--num-groups` with `gss-enhance`, each group generates its own segment files:

```bash
# Run in parallel: group 0, 1, 2, 3 (e.g., 4 SLURM jobs)
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --mc-ref-channel none \
  --output-seglst segments_group{group_id} \
  --num-groups 4 --group-id 0 \
  --output-dir ./enhanced

# After all groups complete, embed using all segment files:
gss-embed --segments enhanced/segments_group0.json \
          enhanced/segments_group1.json \
          enhanced/segments_group2.json \
          enhanced/segments_group3.json \
  --audio meeting.wav \
  --output-dir ./embedded
```

**Generated segment metadata:**

`gss-enhance --output-seglst [prefix]` produces (per group):

- `{prefix}.seglst` — meeteval SegLST format (standard diarization format)
- `{prefix}.json` — JSON format with audio paths (for `gss-embed`)

Default prefix is `segments`. Placeholders like `{group_id}` are replaced.

**Channel offset synchronization:**

If `gss-enhance` was run with `--channel-offsets`, use the same offsets in `gss-embed`:

```bash
# Single multi-channel file
gss-enhance --audio meeting_stereo.wav \
  --diarization meeting.rttm \
  --mc-ref-channel none \
  --channel-offsets 0 -0.1 \
  --channel-offset-unit seconds \
  --output-seglst segments \
  --output-dir ./enhanced

gss-embed --segments enhanced/segments.json \
  --audio meeting_stereo.wav \
  --channel-offsets 0 -0.1 \
  --channel-offset-unit seconds \
  --output-dir ./embedded

# Multiple audio files (each as a channel)
gss-enhance --audio ch0.wav ch1.wav ch2.wav \
  --diarization meeting.rttm \
  --mc-ref-channel none \
  --channel-offsets 0 -0.1 0.05 \
  --channel-offset-unit seconds \
  --output-seglst segments \
  --output-dir ./enhanced

gss-embed --segments enhanced/segments.json \
  --audio ch0.wav ch1.wav ch2.wav \
  --channel-offsets 0 -0.1 0.05 \
  --channel-offset-unit seconds \
  --output-dir ./embedded
```

**Multi-channel input handling:**

If gss-enhance used multiple audio files with mismatched lengths, specify the same resolution mode in gss-embed:

```bash
# If files differ in length, must specify mode (must match gss-enhance)
gss-enhance --audio ch0.wav ch1.wav \
  --diarization meeting.rttm \
  --mc-ref-channel none \
  --channel-length-mode trim \
  --output-seglst segments \
  --output-dir ./enhanced

gss-embed --segments enhanced/segments.json \
  --audio ch0.wav ch1.wav \
  --channel-length-mode trim \
  --output-dir ./embedded
```

**Key points:**

- Original audio file(s) must match the input to `gss-enhance` (same sample rate, channels, order, and lengths)
- Enhanced output channels must match original channels (error otherwise)
- Each speaker receives one full-length audio file with only their enhanced segments embedded
- Non-enhanced regions preserve the original audio
- Channel offsets and length handling must be identical between `gss-enhance` and `gss-embed`
- Single multi-channel file OR multiple single-channel files, but not mixed

See `gss-embed --help` for all options.

## Diarization (Optional)

To generate speaker activity from an audio file, use external diarization tools.

### Single-channel diarization

Generate diarization using [pyannote](https://github.com/pyannote/pyannote-audio):

```python
from pyannote.audio import Pipeline
import torch

# Load pretrained diarization model (requires HuggingFace token)
# Get token at https://huggingface.co/settings/tokens
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="<your_hf_token>"
)

# Move to GPU for faster processing
pipeline.to(torch.device("cuda:0"))

# Run diarization on audio file
diarization = pipeline("meeting.wav")

# Save as RTTM format (compatible with GSS)
with open("meeting.rttm", "w") as f:
    diarization.write_rttm(f)

# Inspect diarization output
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.2f}s - {turn.end:.2f}s: {speaker}")
```

Then use the generated RTTM file with GSS as shown in the [Usage](#usage) examples.

### Multi-channel diarization

For multi-channel audio, run diarization on each channel independently and merge
the results using [dover-lap](https://github.com/desh2608/dover-lap):

```python
from pyannote.audio import Pipeline
import soundfile as sf
import torch
import numpy as np

# Load pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="<your_hf_token>"
)
pipeline.to(torch.device("cuda:0"))

# Load multi-channel audio
audio_multichannel, sr = sf.read("meeting.wav")  # shape: (samples, channels)

# Run diarization on each channel
diarizations = []
for ch_idx in range(audio_multichannel.shape[1]):
    channel_audio = audio_multichannel[:, ch_idx]
    
    # Write temporary mono audio
    temp_file = f"_temp_ch{ch_idx}.wav"
    sf.write(temp_file, channel_audio, sr)
    
    # Run diarization
    diarization = pipeline(temp_file)
    diarizations.append(diarization)

# Merge using dover-lap
from dover_lap import DiariaziationComparator

# Convert to RTTM strings for merging
rttms = [str(d) for d in diarizations]
merged = DiariaziationComparator().optimal_threshold(rttms)

# Save merged result
with open("meeting_merged.rttm", "w") as f:
    merged.write_rttm(f)
```

Or use the command-line tool:

```bash
gss-diarize \
  --audio ch0.wav ch1.wav ch2.wav \
  --output meeting.rttm \
  --hf-token <your_hf_token>
```

**Distributed multi-channel diarization:**

For parallel processing of multiple channels, run diarization on each channel separately and merge:

```bash
# Process each channel in parallel (e.g., 3 SLURM jobs)
gss-diarize --audio ch0.wav --output ch0.rttm --hf-token <token>
gss-diarize --audio ch1.wav --output ch1.rttm --hf-token <token>
gss-diarize --audio ch2.wav --output ch2.rttm --hf-token <token>

# After all channels complete, merge the results
gss-diarize --merge-only --audio ch0.rttm ch1.rttm ch2.rttm \
  --output merged.rttm
```

**Channel selection:**

For single multi-channel files, select specific channels to process:

```bash
# 4-channel audio: process only channels 0 and 2
gss-diarize --audio meeting_4ch.wav \
  --channels 0 2 \
  --output meeting.rttm \
  --hf-token <token>

# Useful for microphone array selection or debugging specific mics
```

**Citations for diarization:**

If you use `gss-diarize`, please cite the following works:

For `pyannote.audio`:
```
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```

```
@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```

For `dover-lap` (multi-channel merging):
```
@article{Raj2021Doverlap,
  title={{DOVER-Lap}: A Method for Combining Overlap-aware Diarization Outputs},
  author={D.Raj and P.Garcia and Z.Huang and S.Watanabe and D.Povey and A.Stolcke and S.Khudanpur},
  journal={2021 IEEE Spoken Language Technology Workshop (SLT)},
  year={2021}
}

@article{Raj2021ReformulatingDL,
  title={Reformulating {DOVER-Lap} Label Mapping as a Graph Partitioning Problem},
  author={Desh Raj and S. Khudanpur},
  journal={INTERSPEECH},
  year={2021},
}
```

## Iterative denoising workflow

For higher-quality speaker separation, combine denoising-only mode with diarization and re-processing:

```bash
# First pass: denoise to improve diarization (MIMO mode: all channels needed for embedding)
gss-enhance --audio meeting.wav --diarization meeting.rttm \
  --mc-ref-channel none \
  --denoising-only \
  --output-seglst segments \
  --output-dir ./pass1_denoised

# Combine denoised segments back into audio
gss-embed --segments ./pass1_denoised/segments.json \
  --audio meeting.wav \
  --output-dir ./pass1_embedded

# Re-run diarization on cleaner audio (higher confidence)
gss-diarize --audio pass1_embedded/denoised.wav \
  --output meeting_rttm_v2 \
  --hf-token <token>

# Second pass: speaker separation with better diarization
gss-enhance --audio meeting.wav \
  --diarization meeting_rttm_v2 \
  --output-dir ./pass2_speakers
```

This approach might yield better speaker separation by first removing noise, then performing
diarization on the cleaner signal with higher confidence.

## Low-level Python API (composing modules directly)

The individual sub-modules are also exported and can be composed directly in
spectrogram domain, avoiding redundant STFT/iSTFT round-trips:

```python
from gss_frontend import (
    AudioToSpectrogram, SpectrogramToAudio,
    MaskBasedDereverbWPE, MaskEstimatorGSS, MaskBasedBeamformer,
    activity_time_to_timefreq,
)
import torch

FFT, HOP = 1024, 256

analysis  = AudioToSpectrogram(fft_length=FFT, hop_length=HOP).cuda()
synthesis = SpectrogramToAudio(fft_length=FFT, hop_length=HOP).cuda()
dereverb  = MaskBasedDereverbWPE(filter_length=10, prediction_delay=2).cuda()
gss       = MaskEstimatorGSS(num_iterations=20).cuda()
mc        = MaskBasedBeamformer().cuda()

audio_t    = torch.from_numpy(audio).float().cuda().requires_grad_(True)
activity_t = torch.from_numpy(activity).float().cuda().requires_grad_(True)

audio_3d    = audio_t.unsqueeze(0)     # (1, ch, T)
activity_3d = activity_t.unsqueeze(0)  # (1, spk, T)

x_enc, _    = analysis(audio_3d)      # (1, ch, F, N)
x_enc, _    = dereverb(input=x_enc)
a_enc       = activity_time_to_timefreq(activity_3d, win_length=FFT, hop_length=HOP)
masks       = gss(x_enc, a_enc)       # (1, spk, F, N)
mask_t      = masks[:, :1]            # target speaker
mask_u      = masks.sum(1, keepdim=True) - mask_t
target_enc, _ = mc(input=x_enc, mask=mask_t, mask_undesired=mask_u)
out, _      = synthesis(input=target_enc)
enhanced    = out[0, 0]               # (T,) — still a Tensor, gradients intact

loss = enhanced.abs().mean()
loss.backward()
# audio_t.grad and activity_t.grad are now populated
```

Because operations stay in spectrogram domain throughout, `GSS.estimate_masks`
can still be used as a convenience helper that handles the numpy/tensor
conversion and `activity_time_to_timefreq`.

Gradients flow through both `audio` and `activity` when using `torch.Tensor`
(with `aggregation="mean"` or `"max"`; `"any"` uses boolean ops and blocks the
activity gradient). See `examples/tensor_backprop.py` for a full example.

## API

### `GSS.__init__`

| Parameter | Default | Description |
|---|---|---|
| `stft_fft_length` | 1024 | FFT size for STFT analysis/synthesis |
| `stft_hop_length` | 256 | Hop length for STFT |
| `dereverb_prediction_delay` | 2 | WPE prediction delay |
| `dereverb_filter_length` | 10 | WPE filter length |
| `dereverb_num_iterations` | 3 | WPE iterations |
| `bss_iterations` | 20 | GSS iterations |
| `garbage_class` | `True` | Add one extra always-active background/noise class (`n_spk + 1`). Set `False` to disable class addition |
| `activity_aggregation` | `'mean'` | Frame aggregation for activity (`'mean'`, `'max'`, `'any'`) |
| `mc_filter_type` | `'pmwf'` | Multichannel filter type |
| `mc_ref_channel` | `'max_snr'` | Reference channel selection |
| `device` | `'cuda'` | PyTorch device |

### `GSS.enhance(audio, activity, speaker_id, ...)`

Enhance a single utterance.

- **`audio`** — `(channels, samples)` float32 numpy array
- **`activity`** — `(speakers, samples)` float numpy array; binary {0,1} or soft [0,1]
- **`speaker_id`** — 0-based index of the target speaker row in `activity`
- **`left_context`** / **`right_context`** — number of context samples prepended/appended to `audio` (will be trimmed from the output)
- **`num_chunks`** — split the frequency axis to reduce peak GPU memory (default 1)
- **Returns** `(samples_out,)` float32 numpy array

### `GSS.enhance_segment(audio, activity, speaker_id, segment_start, segment_end, sample_rate, ...)`

Enhance one target segment from long-form audio with automatic context handling.

- **`audio`** — full recording, shape `(channels, samples)`
- **`activity`** — full-recording speaker activity, shape `(speakers, samples)`
- **`segment_start` / `segment_end`** — target segment boundaries
- **`segment_unit`** — `'seconds'` (default) or `'samples'`
- **`sample_rate`** — used for seconds-to-samples conversion
- **`context_left_seconds` / `context_right_seconds`** — context for mask estimation (default `15.0`, `15.0`)
- **`mode`** — `'standard'` (default, uses `num_chunks`) or `'oom_fallback'` (uses `enhance_auto` OOM retry)
- **Legacy aliases** — `'enhance'` and `'auto'` are still accepted for backward compatibility
- **Returns** enhanced waveform for the target segment only (context is removed)

### `GSS.enhance_from_diarization(audio_path, diarization, speaker_id, ...)`

Enhance diarized segments from a long recording.

- **`audio_path`** — one audio path (`str`) or a list of paths (e.g., separate mono channels)
- **`diarization`** — diarization path (`str`) or list of diarization files (list input is merged by default without time-shift)
- **`speaker_id`** — target speaker selector; supports `int`, `str`, list/tuple of them, or `None` (default: all speakers)
- **`speaker_id` numeric rule** — `0`-based index over lexicographically sorted unique speaker labels in the diarization
- **`diarization_format`** — optional explicit format (e.g. `'rttm'`); when omitted, `meeteval.io.load` auto-detects by extension
- **Supported format labels** — `'rttm'`, `'stm'`, `'ctm'`, `'uem'`, `'seglst'`
- **`diarization_session_id`** — optional session filter for multi-session diarization files
- **`diarization_time_concat`** — when `diarization` is a list, shift each file sequentially and concatenate in time
- **`diarization_concat`** — deprecated alias of `diarization_time_concat`
- **`diarization_concat_gap_seconds`** — optional gap inserted between concatenated diarization files
- **`diarization_offsets`** — optional explicit per-file time offsets (seconds); use instead of `diarization_concat` when known
- **`uem`** — optional UEM path; diarization segments outside UEM are excluded, and context is clipped to UEM-valid boundaries
- **`uem_format`** — optional explicit UEM format (e.g. `'uem'`)
- **`valid_regions`** — optional valid regions provided directly (seconds), e.g. `[(start, end), ...]`; combined with `uem` by intersection when both are given
- **`channel_length_mode`** — when `audio_path` is a list and lengths mismatch: `'error'` (raise), `'trim'` (to shortest), `'pad'` (zero-pad to longest)
- **`channel_offsets`** — optional per-channel shifts; positive delays a channel (prepend zeros), negative advances a channel (drop leading samples)
- **`channel_offset_unit`** — unit for `channel_offsets`: `'samples'` (default) or `'seconds'`
- **`num_groups`** — number of groups to partition segments into for distributed processing (default: 1 = no partitioning). Useful for SLURM/distributed environments where each job processes a subset of segments
- **`group_id`** — zero-based group index to process (must satisfy `0 <= group_id < num_groups`; default: 0). Segments are partitioned with balanced total duration across groups using a greedy algorithm
- **Returns** generator of dicts (yields one at a time for memory efficiency), each containing:
  - `speaker` — speaker label (str)
  - `speaker_id` — speaker index (int, 0-based in lexicographic order)
  - `segment_start` / `segment_end` — segment timing (seconds)
  - `sample_rate` — audio sample rate (Hz)
  - `enhanced_audio` — enhanced waveform `(samples,)` float32

### `GSS.enhance_auto(...)`

Same as `enhance`, but automatically retries with a finer frequency-axis split
whenever a CUDA out-of-memory error occurs.  If all chunk sizes are exhausted,
falls back to per-stage CPU execution (dereverb / GSS / beamforming individually).

### `GSS.enhance_unguided(audio, num_sources, left_context=0, right_context=0, ...)`

Blind source separation without speaker activity guidance. Uses uniform activity
assumption across all sources, making it suitable when diarization is unavailable.

- **`audio`** — `(channels, samples)` float32 `numpy.ndarray` or `torch.Tensor`
- **`num_sources`** — number of sources to separate (default: `num_channels + 1`)
- **Returns** — dict with keys:
  - `'audio'`: separated audio `(num_sources, samples)` same type as input
  - `'masks'`: time-frequency masks `(num_sources, freq, frames)` in [0, 1]
  - `'eigenvalues'`: covariance eigenvalues `(num_sources, freq, channels)` for condition number computation
  - `'mahalanobis'`: Mahalanobis distances `(num_sources, freq, frames)`
  - `'occupancy'`: time-averaged mask per source `(num_sources,)`
  - `'temporal_variance'`: per-source mask variance over time `(num_sources,)`
  - `'condition_number'`: eigenvalue condition number λ_max/λ_min `(num_sources, freq)`

Use statistics (especially `condition_number`, `occupancy`, `temporal_variance`)
to classify speech vs. noise via external heuristics.

### `GSS.enhance_unguided_auto(audio, num_sources, ...)`

Same as `enhance_unguided`, but uses the OOM retry logic from `enhance_auto`.
Automatically falls back to per-stage CPU execution if CUDA memory is exhausted.
Recommended for large audio files.

- **`audio`** — `(channels, samples)` float32 `numpy.ndarray` or `torch.Tensor`
- **`num_sources`** — number of sources to separate
- **Returns** — dict with same keys as `enhance_unguided()`: `'audio'`, `'masks'`, 
  `'eigenvalues'`, `'mahalanobis'`, `'occupancy'`, `'temporal_variance'`, `'condition_number'`

### `GSS.estimate_masks(audio, activity, garbage_class=None)`

Estimate time-frequency masks via GSS (cACGMM EM).

- **`audio`** — `(channels, samples)` float32 `numpy.ndarray` or `torch.Tensor`
- **`activity`** — `(speakers, samples)` `numpy.ndarray` or `torch.Tensor`; binary {0,1} or soft [0,1]
- **`garbage_class`** — bool or None; if None, uses `self.garbage_class` (default: True)
- **Returns** `(classes, freq, frames)` same type as `audio`, values in [0, 1]
- `classes = speakers + 1` when `garbage_class=True`, else `classes = speakers`
- When `audio` is a `torch.Tensor`, gradients are preserved (training-friendly)

### Building-block modules

The following classes are exported directly from `gss_frontend` and operate on
spectrograms (shape `(B, C, F, N)` complex):

| Class | Description |
|---|---|
| `AudioToSpectrogram` | STFT: waveform → complex spectrogram |
| `SpectrogramToAudio` | iSTFT: complex spectrogram → waveform |
| `MaskBasedDereverbWPE` | Iterative WPE dereverberation |
| `MaskEstimatorGSS` | cACGMM mask estimation |
| `MaskBasedBeamformer` | Mask-based multichannel beamformer |

`activity_time_to_timefreq(activity, win_length, hop_length)` converts
sample-level activity `(B, spk, T)` to frame-level `(B, spk, N)`.

## Citation

If you use this repository in academic work, please cite the GSS/front-end
processing papers it is based on:

```bibtex
@inproceedings{boeddecker18_chime,
  title     = {{Front-end processing for the CHiME-5 dinner party scenario}},
  author    = {Christoph Boeddecker and Jens Heitkaemper and Joerg Schmalenstroeer and Lukas Drude and Jahn Heymann and Reinhold Haeb-Umbach},
  year      = {2018},
  booktitle = {{5th International Workshop on Speech Processing in Everyday Environments (CHiME 2018)}},
  pages     = {35--40},
  doi       = {10.21437/CHiME.2018-8},
}

@inproceedings{raj23_interspeech,
    title     = {{GPU-accelerated Guided Source Separation for Meeting Transcription}},
    author    = {Desh Raj and Daniel Povey and Sanjeev Khudanpur},
    year      = {2023},
    booktitle = {{Interspeech 2023}},
    pages     = {3507--3511},
    doi       = {10.21437/Interspeech.2023-42},
    issn      = {2958-1796}
}
```