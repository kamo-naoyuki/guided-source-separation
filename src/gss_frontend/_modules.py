# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone re-implementation of NeMo audio-processing modules used by FrontEnd_v1.

All NeMo-specific infrastructure (NeuralModule, typecheck, NeuralType) has been
removed.  The only runtime dependencies are torch and torchaudio.

Extracted from:
  nemo/collections/asr/modules/audio_preprocessing.py
  nemo/collections/asr/modules/audio_modules.py
  nemo/collections/asr/parts/submodules/multichannel_modules.py
  nemo/collections/asr/parts/utils/audio_utils.py
  nemo/collections/asr/parts/preprocessing/features.py
"""

import logging
from typing import Optional, Tuple

import torch
import torchaudio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def db2mag(db: float) -> float:
    """Convert a value in dB to a linear magnitude ratio."""
    return 10 ** (db / 20)


def make_seq_mask_like(
    lengths: torch.Tensor,
    like: torch.Tensor,
    time_dim: int = -1,
    valid_ones: bool = True,
) -> torch.Tensor:
    """Build a boolean sequence mask with the same shape as *like*.

    Args:
        lengths: Valid-length counts, shape (B,).
        like: Reference tensor — the mask will match its number of dimensions
              and its size along *time_dim*.
        time_dim: Time dimension index (zero-based).
        valid_ones: If True, valid positions are 1 and padding is 0; else inverted.

    Returns:
        Boolean tensor broadcastable to *like*.
    """
    mask = (
        torch.arange(like.shape[time_dim], device=like.device)
        .repeat(lengths.shape[0], 1)
        .lt(lengths.view(-1, 1))
    )
    # Insert singleton dims so the mask is broadcastable to *like*
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    if time_dim != -1 and time_dim != mask.dim() - 1:
        mask = mask.transpose(-1, time_dim)
    if not valid_ones:
        mask = ~mask
    return mask


# ---------------------------------------------------------------------------
# STFT analysis / synthesis
# ---------------------------------------------------------------------------

class AudioToSpectrogram(torch.nn.Module):
    """Transform a batch of multi-channel waveforms into complex STFT spectrograms.

    Args:
        fft_length: FFT size (must be even).
        hop_length: STFT hop length.
        power: Spectrogram power exponent.  ``None`` (default) returns complex output.

    Input:  (B, C, T)  float
    Output: (B, C, F, N)  complex  +  output_length (B,)
    """

    def __init__(self, fft_length: int, hop_length: int, power: Optional[float] = None):
        super().__init__()
        if fft_length % 2 != 0:
            raise ValueError(f"fft_length={fft_length} must be divisible by 2")
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=fft_length, hop_length=hop_length, power=power, pad_mode="constant"
        )
        self.F = fft_length // 2 + 1

    def forward(
        self,
        input: torch.Tensor,
        input_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input.size(0), input.size(-1)
        input = input.view(B, -1, T)

        with torch.amp.autocast('cuda', enabled=False):
            output = self.stft(input.float())  # (B, C, F, N)

        if input_length is not None:
            output_length = input_length.div(self.stft.hop_length, rounding_mode="floor").add(1).long()
            length_mask = make_seq_mask_like(lengths=output_length, like=output, time_dim=-1, valid_ones=False)
            output = output.masked_fill(length_mask, 0.0)
        else:
            output_length = output.size(-1) * torch.ones(B, device=output.device).long()

        return output, output_length


class SpectrogramToAudio(torch.nn.Module):
    """Transform a batch of complex STFT spectrograms back to waveforms.

    Input:  (B, C, F, N)  complex
    Output: (B, C, T)  float  +  output_length (B,)
    """

    def __init__(self, fft_length: int, hop_length: int):
        super().__init__()
        if fft_length % 2 != 0:
            raise ValueError(f"fft_length={fft_length} must be divisible by 2")
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=fft_length, hop_length=hop_length, pad_mode="constant"
        )
        self.F = fft_length // 2 + 1

    def forward(
        self,
        input: torch.Tensor,
        input_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, F, N = input.size(0), input.size(-2), input.size(-1)
        assert F == self.F, f"Number of subbands F={F} does not match self.F={self.F}"
        input = input.view(B, -1, F, N)

        with torch.amp.autocast('cuda', enabled=False):
            output = self.istft(input.cfloat())  # (B, C, T)

        if input_length is not None:
            output_length = (input_length - 1) * self.istft.hop_length
            length_mask = make_seq_mask_like(lengths=output_length, like=output, time_dim=-1, valid_ones=False)
            output = output.masked_fill(length_mask, 0.0)
        else:
            output_length = output.size(-1) * torch.ones(B, device=output.device).long()

        return output, output_length


# ---------------------------------------------------------------------------
# GSS mask estimator
# ---------------------------------------------------------------------------

class MaskEstimatorGSS(torch.nn.Module):
    """Estimate time-frequency masks using Guided Source Separation (GSS).

    Uses a complex Angular Central Gaussian Mixture Model (cACGMM) with EM.

    Args:
        num_iterations: Number of EM iterations.
        eps: Small regularisation constant.
        dtype: Complex dtype for internal calculations (``torch.cfloat`` or
               ``torch.cdouble``).

    Input:
        input:    (B, num_inputs, F, T)  complex spectrogram
        activity: (B, num_outputs, T)   float activity mask in [0, 1]

    Output: (B, num_outputs, F, T) float masks
    """

    def __init__(
        self,
        num_iterations: int = 3,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.cdouble,
    ):
        super().__init__()
        assert dtype in (torch.cfloat, torch.cdouble), f"Unsupported dtype {dtype}"
        self.num_iterations = num_iterations
        self.eps = eps
        self.dtype = dtype

    def normalize(self, x: torch.Tensor, dim: int = -3) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
        return x / (norm + self.eps)

    def update_masks(self, alpha, activity, log_pdf):
        gamma = log_pdf - torch.max(log_pdf, dim=-3, keepdim=True)[0]
        gamma = torch.exp(gamma)
        gamma = alpha[..., None] * gamma * activity[..., None, :]
        gamma = gamma / (torch.sum(gamma, dim=-3, keepdim=True) + self.eps)
        return gamma

    def update_weights(self, gamma):
        return torch.mean(gamma, dim=-1)

    def update_pdf(self, z, gamma, zH_invBM_z):
        num_inputs = z.size(-3)
        scale = gamma / (zH_invBM_z + self.eps)
        BM = num_inputs * torch.einsum("bmft,bift,bjft->bmfij", scale.to(z.dtype), z, z.conj())
        denom = torch.sum(gamma, dim=-1)
        BM = BM / (denom[..., None, None] + self.eps)
        BM = (BM + BM.conj().transpose(-1, -2)) / 2

        L, Q = torch.linalg.eigh(BM)
        L = torch.clamp(L.real, min=self.eps)
        L = L / (torch.max(L, dim=-1, keepdim=True)[0] + self.eps)
        L = L + self.eps

        log_detBM = torch.sum(torch.log(L), dim=-1)

        zH_invBM_z = torch.einsum("bmfj,bmfkj,bkft->bmftj", (1 / L.sqrt()).to(Q.dtype), Q.conj(), z)
        zH_invBM_z = zH_invBM_z.abs().pow(2).sum(-1) + self.eps

        log_pdf = -num_inputs * torch.log(zH_invBM_z) - log_detBM[..., None]
        return log_pdf, zH_invBM_z

    def forward(self, input: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        B, num_inputs, F, T = input.shape
        num_outputs = activity.size(1)
        assert activity.size(0) == B and activity.size(-1) == T

        with torch.amp.autocast('cuda', enabled=False):
            input = input.to(dtype=self.dtype)
            z = self.normalize(input, dim=-3)

            gamma = torch.clamp(activity, min=self.eps)
            gamma = gamma / torch.sum(gamma, dim=-2, keepdim=True)
            gamma = gamma.unsqueeze(2).expand(-1, -1, F, -1)

            zH_invBM_z = torch.ones(B, num_outputs, F, T, dtype=input.dtype, device=input.device)

            for _ in range(self.num_iterations):
                alpha = self.update_weights(gamma)
                log_pdf, zH_invBM_z = self.update_pdf(z, gamma, zH_invBM_z)
                gamma = self.update_masks(alpha, activity, log_pdf)

        if torch.any(torch.isnan(gamma)):
            raise RuntimeError("gamma contains NaNs")

        return gamma


# ---------------------------------------------------------------------------
# WPE dereverberation filter
# ---------------------------------------------------------------------------

class WPEFilter(torch.nn.Module):
    """Weighted Prediction Error filter for MIMO dereverberation.

    Args:
        filter_length: Prediction filter length (frames per channel).
        prediction_delay: Prediction delay (frames).
        diag_reg: Diagonal regularisation coefficient for correlation matrix.
        eps: Small regularisation constant.
    """

    def __init__(
        self,
        filter_length: int,
        prediction_delay: int,
        diag_reg: Optional[float] = 1e-6,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.filter_length = filter_length
        self.prediction_delay = prediction_delay
        self.diag_reg = diag_reg
        self.eps = eps

    def forward(
        self,
        input: torch.Tensor,
        power: torch.Tensor,
        input_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight = torch.mean(power, dim=1)
        weight = 1 / (weight + self.eps)

        tilde_input = self.convtensor(input, filter_length=self.filter_length, delay=self.prediction_delay)
        Q, R = self.estimate_correlations(input=input, weight=weight, tilde_input=tilde_input, input_length=input_length)
        G = self.estimate_filter(Q=Q, R=R)
        undesired = self.apply_filter(filter=G, tilde_input=tilde_input)
        desired = input - undesired

        if input_length is not None:
            length_mask = make_seq_mask_like(lengths=input_length, like=desired, time_dim=-1, valid_ones=False)
            desired = desired.masked_fill(length_mask, 0.0)

        return desired, input_length

    @classmethod
    def convtensor(cls, x: torch.Tensor, filter_length: int, delay: int = 0, n_steps: Optional[int] = None):
        if x.ndim != 4:
            raise RuntimeError(f"Expecting 4-D input, got {x.shape}")
        B, C, F, N = x.shape
        if n_steps is None:
            n_steps = N
        x = torch.nn.functional.pad(x, (filter_length - 1 + delay, 0))
        tilde_X = x.unfold(-1, filter_length, 1)
        return tilde_X[:, :, :, :n_steps, :]

    def estimate_correlations(self, input, weight, tilde_input, input_length=None):
        if input_length is not None:
            length_mask = make_seq_mask_like(lengths=input_length, like=weight, time_dim=-1, valid_ones=False)
            weight = weight.masked_fill(length_mask, 0.0)
        Q = torch.einsum("bjfik,bmfin->bfjkmn", tilde_input.conj(), weight[:, None, :, :, None] * tilde_input)
        R = torch.einsum("bjfik,bmfi->bfjkm", tilde_input.conj(), weight[:, None, :, :] * input)
        return Q, R

    def estimate_filter(self, Q, R):
        B, F, C, filter_length, _, _ = Q.shape
        Q = Q.reshape(B, F, C * filter_length, C * filter_length)
        R = R.reshape(B, F, C * filter_length, C)

        if self.diag_reg:
            diag_reg = self.diag_reg * torch.diagonal(Q, dim1=-2, dim2=-1).sum(-1).real + self.eps
            Q = Q + torch.diag_embed(diag_reg.unsqueeze(-1) * torch.ones(Q.shape[-1], device=Q.device))

        fail = False
        try:
            QL = torch.linalg.cholesky(Q)
            G = torch.linalg.solve_triangular(QL, R, upper=False)
            G = torch.linalg.solve_triangular(QL.conj().transpose(-2, -1), G, upper=True)
        except torch.linalg.LinAlgError:
            fail = True

        if fail:
            logger.warning("Cholesky failed, falling back to linalg.solve")
            G = torch.linalg.solve(Q, R)

        G = G.reshape(B, F, C, filter_length, C)
        G = G.permute(0, 4, 1, 2, 3)
        return G

    def apply_filter(self, filter, input=None, tilde_input=None):
        if input is None and tilde_input is None:
            raise RuntimeError("Both input and tilde_input cannot be None")
        if input is not None and tilde_input is not None:
            raise RuntimeError("Provide either input or tilde_input, not both")
        if tilde_input is None:
            tilde_input = self.convtensor(input, filter_length=self.filter_length, delay=self.prediction_delay)
        return torch.einsum("bjfik,bmfjk->bmfi", tilde_input, filter)


# ---------------------------------------------------------------------------
# WPE-based dereverberation
# ---------------------------------------------------------------------------

class MaskBasedDereverbWPE(torch.nn.Module):
    """Iterative WPE dereverberation with optional TF mask.

    Args:
        filter_length: WPE filter length per channel (frames).
        prediction_delay: WPE prediction delay (frames).
        num_iterations: Number of reweighting iterations.
        mask_min_db: Lower mask clip threshold (dB).
        mask_max_db: Upper mask clip threshold (dB).
        diag_reg: Diagonal regularisation for WPE.
        eps: Small regularisation constant.
        dtype: Complex dtype (``torch.cfloat`` or ``torch.cdouble``).
    """

    def __init__(
        self,
        filter_length: int,
        prediction_delay: int,
        num_iterations: int = 1,
        mask_min_db: float = -200,
        mask_max_db: float = 0,
        diag_reg: Optional[float] = 1e-6,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.cdouble,
    ):
        super().__init__()
        self.filter = WPEFilter(filter_length=filter_length, prediction_delay=prediction_delay, diag_reg=diag_reg, eps=eps)
        self.num_iterations = num_iterations
        self.mask_min = db2mag(mask_min_db)
        self.mask_max = db2mag(mask_max_db)
        if dtype not in (torch.cfloat, torch.cdouble):
            raise ValueError(f"Unsupported dtype {dtype}")
        self.dtype = dtype

    def forward(
        self,
        input: torch.Tensor,
        input_length: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        io_dtype = input.dtype

        with torch.amp.autocast('cuda', enabled=False):
            output = input.to(dtype=self.dtype)
            if not output.is_complex():
                raise RuntimeError(f"Expecting complex input, got {output.dtype}")

            for i in range(self.num_iterations):
                magnitude = torch.abs(output)
                if i == 0 and mask is not None:
                    mask = torch.clamp(mask, min=self.mask_min, max=self.mask_max)
                    magnitude = mask * magnitude
                power = magnitude ** 2
                output, output_length = self.filter(input=output, input_length=input_length, power=power)

        return output.to(io_dtype), output_length


# ---------------------------------------------------------------------------
# Reference channel estimator
# ---------------------------------------------------------------------------

class ReferenceChannelEstimatorSNR(torch.nn.Module):
    """Select (or softly weight) the reference channel by maximising output SNR.

    Args:
        hard: Return hard (one-hot) reference if True, soft otherwise.
        hard_use_grad: Use straight-through estimator for the hard reference.
        subband_weighting: Weight subbands when aggregating SNR.
        num_subbands: Required when *subband_weighting* is True.
        eps: Small regularisation constant.
    """

    def __init__(
        self,
        hard: bool = True,
        hard_use_grad: bool = True,
        subband_weighting: bool = False,
        num_subbands: Optional[int] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.hard = hard
        self.hard_use_grad = hard_use_grad
        self.subband_weighting = subband_weighting
        self.eps = eps

        if subband_weighting and num_subbands is None:
            raise ValueError("num_subbands must be set when subband_weighting=True")
        self.weight_s = torch.nn.Parameter(torch.ones(num_subbands)) if subband_weighting else None
        self.weight_n = torch.nn.Parameter(torch.ones(num_subbands)) if subband_weighting else None

    def forward(self, W: torch.Tensor, psd_s: torch.Tensor, psd_n: torch.Tensor) -> torch.Tensor:
        if self.subband_weighting:
            pow_s = torch.einsum("...jm,...jk,...km->...m", W.conj(), psd_s, W).abs()
            pow_n = torch.einsum("...jm,...jk,...km->...m", W.conj(), psd_n, W).abs()
            pow_s = torch.sum(pow_s * self.weight_s.softmax(dim=0).unsqueeze(1), dim=-2)
            pow_n = torch.sum(pow_n * self.weight_n.softmax(dim=0).unsqueeze(1), dim=-2)
        else:
            pow_s = torch.einsum("...fjm,...fjk,...fkm->...m", W.conj(), psd_s, W).abs()
            pow_n = torch.einsum("...fjm,...fjk,...fkm->...m", W.conj(), psd_n, W).abs()

        snr = 10 * torch.log10(pow_s / (pow_n + self.eps) + self.eps)
        ref_soft = snr.softmax(dim=-1)

        if self.hard:
            _, idx = ref_soft.max(dim=-1, keepdim=True)
            ref_hard = torch.zeros_like(snr).scatter(-1, idx, 1.0)
            if self.hard_use_grad:
                ref = ref_hard - ref_soft.detach() + ref_soft
            else:
                ref = ref_hard
        else:
            ref = ref_soft

        return ref


# ---------------------------------------------------------------------------
# Parametric Multichannel Wiener Filter
# ---------------------------------------------------------------------------

class ParametricMultichannelWienerFilter(torch.nn.Module):
    """Parametric multichannel Wiener filter (PMWF / MVDR).

    Args:
        beta: Trade-off between noise reduction (1 = MWF) and
              distortion minimisation (0 = MVDR).
        rank: Rank assumption for the speech covariance matrix.
              ``'one'`` (default) or ``'full'``.
        postfilter: Optional post-filter; ``None`` or ``'ban'``.
        ref_channel: Fixed reference channel index, or ``'max_snr'`` for
                     automatic selection, or ``None`` for MIMO output.
        ref_hard / ref_hard_use_grad / ref_subband_weighting / num_subbands:
            Options forwarded to :class:`ReferenceChannelEstimatorSNR`.
        diag_reg: Diagonal regularisation coefficient.
        eps: Small regularisation constant.
    """

    def __init__(
        self,
        beta: float = 1.0,
        rank: str = "one",
        postfilter: Optional[str] = None,
        ref_channel=None,
        ref_hard: bool = True,
        ref_hard_use_grad: bool = True,
        ref_subband_weighting: bool = False,
        num_subbands: Optional[int] = None,
        diag_reg: Optional[float] = 1e-6,
        eps: float = 1e-8,
    ):
        super().__init__()
        if postfilter not in (None, "ban"):
            raise ValueError(f"Postfilter '{postfilter}' is not supported")
        if rank == "full" and beta == 0:
            raise ValueError(f"rank='full' is incompatible with beta=0")

        self.beta = beta
        self.rank = rank
        self.postfilter = postfilter
        self.diag_reg = diag_reg
        self.eps = eps
        self.psd = torchaudio.transforms.PSD()
        self.ref_channel = ref_channel
        self.is_mimo = ref_channel is None
        self.ref_estimator: Optional[ReferenceChannelEstimatorSNR] = None

        if self.ref_channel == "max_snr":
            self.ref_estimator = ReferenceChannelEstimatorSNR(
                hard=ref_hard,
                hard_use_grad=ref_hard_use_grad,
                subband_weighting=ref_subband_weighting,
                num_subbands=num_subbands,
                eps=eps,
            )

    @staticmethod
    def trace(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        t = torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)
        if keepdim:
            t = t.unsqueeze(-1).unsqueeze(-1)
        return t

    def apply_diag_reg(self, psd: torch.Tensor) -> torch.Tensor:
        diag_reg = self.diag_reg * self.trace(psd).real + self.eps
        return psd + torch.diag_embed(diag_reg.unsqueeze(-1) * torch.ones(psd.shape[-1], device=psd.device))

    def apply_filter(self, input: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bfcm,bcft->bmft", filter.conj(), input)

    def apply_ban(self, input: torch.Tensor, filter: torch.Tensor, psd_n: torch.Tensor) -> torch.Tensor:
        num_inputs = filter.size(-2)
        numerator = torch.einsum("bfcm,bfci,bfij,bfjm->bmf", filter.conj(), psd_n, psd_n, filter)
        numerator = torch.sqrt(numerator.abs() / num_inputs)
        denominator = torch.einsum("bfcm,bfci,bfim->bmf", filter.conj(), psd_n, filter).abs()
        ban = numerator / (denominator + self.eps)
        return ban[..., None] * input

    def forward(
        self,
        input: torch.Tensor,
        mask_s: torch.Tensor,
        mask_n: torch.Tensor,
    ) -> torch.Tensor:
        iodtype = input.dtype

        with torch.amp.autocast('cuda', enabled=False):
            input = input.cdouble()
            mask_s = mask_s.double()
            mask_n = mask_n.double()

            psd_s = self.psd(input, mask_s)
            psd_n = self.psd(input, mask_n)

            if self.rank == "one":
                if self.diag_reg:
                    psd_n = self.apply_diag_reg(psd_n)
                W = torch.linalg.solve(psd_n, psd_s)
                lam = self.trace(W, keepdim=True).real
                W = W / (self.beta + lam + self.eps)
            elif self.rank == "full":
                psd_sn = psd_s + self.beta * psd_n
                if self.diag_reg:
                    psd_sn = self.apply_diag_reg(psd_sn)
                W = torch.linalg.solve(psd_sn, psd_s)
            else:
                raise RuntimeError(f"Unexpected rank '{self.rank}'")

            if torch.jit.isinstance(self.ref_channel, int):
                W = W[..., self.ref_channel].unsqueeze(-1)
            elif self.ref_estimator is not None:
                ref = self.ref_estimator(W=W, psd_s=psd_s, psd_n=psd_n).to(W.dtype)
                W = torch.sum(W * ref[:, None, None, :], dim=-1, keepdim=True)

            output = self.apply_filter(input=input, filter=W)

            if self.postfilter == "ban":
                output = self.apply_ban(input=output, filter=W, psd_n=psd_n)

        return output.to(iodtype)


# ---------------------------------------------------------------------------
# Mask-based beamformer
# ---------------------------------------------------------------------------

class MaskBasedBeamformer(torch.nn.Module):
    """Multi-channel beamformer driven by time-frequency masks.

    Args:
        filter_type: ``'pmwf'`` (default) or ``'mvdr_souden'``.
        filter_beta / filter_rank / filter_postfilter:
            Forwarded to :class:`ParametricMultichannelWienerFilter`.
        ref_channel: Reference channel (int, ``'max_snr'``, or ``None``).
        ref_hard / ref_hard_use_grad / ref_subband_weighting / num_subbands:
            Forwarded to :class:`ReferenceChannelEstimatorSNR`.
        mask_min_db / mask_max_db: Mask clip thresholds.
        postmask_min_db / postmask_max_db: Post-mask clip thresholds.
        diag_reg / eps: Regularisation parameters.
    """

    def __init__(
        self,
        filter_type: str = "mvdr_souden",
        filter_beta: float = 0.0,
        filter_rank: str = "one",
        filter_postfilter: Optional[str] = None,
        ref_channel=0,
        ref_hard: bool = True,
        ref_hard_use_grad: bool = False,
        ref_subband_weighting: bool = False,
        num_subbands: Optional[int] = None,
        mask_min_db: float = -200,
        mask_max_db: float = 0,
        postmask_min_db: float = 0,
        postmask_max_db: float = 0,
        diag_reg: Optional[float] = 1e-6,
        eps: float = 1e-8,
    ):
        super().__init__()
        if filter_type not in ("pmwf", "mvdr_souden"):
            raise ValueError(f"Unknown filter_type '{filter_type}'")
        self.filter_type = filter_type
        if filter_type == "mvdr_souden" and filter_beta != 0:
            logger.warning("mvdr_souden: forcing beta=0 and rank='one'")
            filter_beta = 0.0
            filter_rank = "one"

        self.filter = ParametricMultichannelWienerFilter(
            beta=filter_beta,
            rank=filter_rank,
            postfilter=filter_postfilter,
            ref_channel=ref_channel,
            ref_hard=ref_hard,
            ref_hard_use_grad=ref_hard_use_grad,
            ref_subband_weighting=ref_subband_weighting,
            num_subbands=num_subbands,
            diag_reg=diag_reg,
            eps=eps,
        )

        if mask_min_db >= mask_max_db:
            raise ValueError(f"mask_min_db ({mask_min_db}) must be < mask_max_db ({mask_max_db})")
        self.mask_min = db2mag(mask_min_db)
        self.mask_max = db2mag(mask_max_db)

        if postmask_min_db > postmask_max_db:
            raise ValueError(f"postmask_min_db ({postmask_min_db}) must be <= postmask_max_db ({postmask_max_db})")
        self.postmask_min = db2mag(postmask_min_db)
        self.postmask_max = db2mag(postmask_max_db)

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        mask_undesired: Optional[torch.Tensor] = None,
        input_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_length is not None:
            length_mask = make_seq_mask_like(lengths=input_length, like=mask[:, 0, ...], time_dim=-1, valid_ones=False)

        output = []
        num_masks = mask.size(1)
        for m in range(num_masks):
            mask_d = mask[:, m, ...]
            if mask_undesired is not None:
                mask_u = mask_undesired[:, m, ...]
            elif num_masks == 1:
                mask_u = 1 - mask_d
            else:
                mask_u = torch.sum(mask, dim=1) - mask_d

            mask_d = torch.clamp(mask_d, min=self.mask_min, max=self.mask_max)
            mask_u = torch.clamp(mask_u, min=self.mask_min, max=self.mask_max)

            if input_length is not None:
                mask_d = mask_d.masked_fill(length_mask, 0.0)
                mask_u = mask_u.masked_fill(length_mask, 0.0)

            output_m = self.filter(input=input, mask_s=mask_d, mask_n=mask_u)

            if self.postmask_min < self.postmask_max:
                postmask_m = torch.clamp(mask[:, m, ...], min=self.postmask_min, max=self.postmask_max)
                output_m = output_m * postmask_m.unsqueeze(1)

            output.append(output_m)

        output = torch.concatenate(output, dim=1)

        if input_length is not None:
            output = output.masked_fill(length_mask[:, None, ...], 0.0)

        return output, input_length
