# gss-frontend

A minimal, self-contained implementation of the NeMo-based GSS (Guided Source
Separation) front-end for speech enhancement.

Extracted from the [CHiME-8 DASR NeMo baseline](https://github.com/NVIDIA-NeMo/Speech),
`GSS` has been stripped of its batch-processing infrastructure, lhotse
dependencies, and DataLoader boilerplate so that it can be called **one utterance
at a time** with a straightforward numpy API.

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
soundfile  # only needed for file I/O in your own code
```

No NeMo dependency required. All necessary modules are bundled in `src/gss_frontend/_modules.py`.

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
import soundfile as sf
from gss_frontend import GSS

# --- Initialize once ---
frontend = GSS(
    stft_fft_length=1024,
    stft_hop_length=256,
    bss_iterations=20,
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

The three pipeline stages can also be called independently:

```python
# WPE dereverberation only → (channels, samples)
dry = frontend.dereverberate(audio)

# GSS mask estimation only → (speakers, freq, frames)
masks = frontend.estimate_masks(audio, activity)

# Mask-based beamforming only → (samples,)
mask_target    = masks[0]
mask_undesired = masks.sum(axis=0) - masks[0]
enhanced = frontend.beamform(audio, mask_target, mask_undesired)
```

The standalone stage APIs (`dereverberate`, `estimate_masks`, `beamform`) accept
both `numpy.ndarray` and `torch.Tensor`.

When inputs are `torch.Tensor`, the computation graph is preserved, so these APIs
are backpropagatable and can be used inside neural-network training loops.

Gradients flow through both `audio` and `activity` (with `aggregation="mean"` or
`"max"`; `"any"` uses boolean ops and breaks the activity gradient).

```python
import torch

# Example: training-friendly use (Tensor input -> Tensor output)
audio_t = torch.randn(8, 16000, device="cuda", dtype=torch.float32, requires_grad=True)
activity_t = torch.zeros(2, 16000, device="cuda", dtype=torch.float32, requires_grad=True)
activity_t.data[0, 2000:8000] = 1.0

dry_t = frontend.dereverberate(audio_t)
masks_t = frontend.estimate_masks(dry_t, activity_t)
out_t = frontend.beamform(dry_t, masks_t[0], masks_t.sum(dim=0) - masks_t[0])

loss = out_t.abs().mean()
loss.backward()  # gradients flow back to both audio_t and activity_t
```

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

### `GSS.enhance_auto(...)`

Same as `enhance`, but automatically retries with a finer frequency-axis split
whenever a CUDA out-of-memory error occurs.  If all chunk sizes are exhausted,
falls back to per-stage CPU execution (dereverb / GSS / beamforming individually).

### `GSS.dereverberate(audio)`

Apply WPE dereverberation to multi-channel audio.

- **`audio`** — `(channels, samples)` float32 `numpy.ndarray` or `torch.Tensor`
- **Returns** `(channels, samples)` same type as input (`numpy.ndarray` / `torch.Tensor`)
- When `audio` is a `torch.Tensor`, gradients are preserved (training-friendly)

### `GSS.estimate_masks(audio, activity)`

Estimate time-frequency masks via GSS (cACGMM EM).

- **`audio`** — `(channels, samples)` float32 `numpy.ndarray` or `torch.Tensor`
- **`activity`** — `(speakers, samples)` `numpy.ndarray` or `torch.Tensor`; binary {0,1} or soft [0,1]
- **Returns** `(speakers, freq, frames)` same type as `audio`, values in [0, 1]
- When `audio` is a `torch.Tensor`, gradients are preserved (training-friendly)

### `GSS.beamform(audio, mask_target, mask_undesired)`

Apply mask-based multichannel beamforming with pre-computed masks.

- **`audio`** — `(channels, samples)` float32 `numpy.ndarray` or `torch.Tensor`
- **`mask_target`** — `(freq, frames)` float32 `numpy.ndarray` or `torch.Tensor`
- **`mask_undesired`** — `(freq, frames)` float32 `numpy.ndarray` or `torch.Tensor`
- **Returns** `(samples,)` same type as `audio` (`numpy.ndarray` / `torch.Tensor`)
- When `audio` is a `torch.Tensor`, gradients are preserved (training-friendly)

## Repository layout

```
gss-frontend/
├── src/
│   └── gss_frontend/
│       ├── __init__.py      # exposes GSS
│       ├── _frontend.py     # GSS class
│       └── _modules.py      # standalone PyTorch modules (no NeMo)
├── tests/
│   └── test_modules.py      # CPU-only unit tests
├── examples/
│   ├── separate_speakers.py  # 2-speaker separation demo
│   └── tensor_backprop.py    # torch.Tensor + backprop demo
├── pyproject.toml
└── LICENSE
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

## Citation

If you use this repository in academic work, please cite the GSS/front-end
processing papers it is based on:

```bibtex
@inproceedings{boeddeker2018chime5_frontend,
    author    = {Christoph Boeddeker and Jens Heitkaemper and Johannes Schmalenstroeer and Lukas Drude and Jahn Heymann and Reinhold Haeb-Umbach},
    title     = {Front-end processing for the CHiME-5 dinner party scenario},
    booktitle = {The 6th CHiME Workshop},
    year      = {2018}
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

## Mapping to the original code

| Original | This repo |
|---|---|
| `CutEnhancer.enhance_cuts()` | Removed — write your own loop if needed |
| `FrontEnd_v1.enhance_batch()` | `GSS._enhance_tensor()` (internal) |
| `lhotse.CutSet` | Removed — pass numpy arrays directly |
| `Activity` class | Removed — pass `activity` as a numpy array |
| `GssDataset` / `create_sampler` | Removed |
| `save_worker` | Removed — handle file I/O in the caller |
