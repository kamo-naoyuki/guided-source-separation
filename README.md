# gss-frontend

A minimal, self-contained implementation of the NeMo-based GSS (Guided Source
Separation) front-end for speech enhancement.

Extracted from the [CHiME-8 DASR NeMo baseline](https://github.com/NVIDIA-NeMo/Speech),
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

### `GSS.estimate_masks(audio, activity)`

Estimate time-frequency masks via GSS (cACGMM EM).

- **`audio`** — `(channels, samples)` float32 `numpy.ndarray` or `torch.Tensor`
- **`activity`** — `(speakers, samples)` `numpy.ndarray` or `torch.Tensor`; binary {0,1} or soft [0,1]
- **Returns** `(speakers, freq, frames)` same type as `audio`, values in [0, 1]
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
