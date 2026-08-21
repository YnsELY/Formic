"""Numerical comparisons used by the formal identity gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TensorComparison:
    exact: bool
    max_abs_delta: float
    first_coordinate: tuple[int, ...] | None
    reference_value: float | int | bool | None
    candidate_value: float | int | bool | None
    shape: tuple[int, ...]
    dtype_reference: str
    dtype_candidate: str
    device_reference: str
    device_candidate: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["first_coordinate"] = (
            list(self.first_coordinate) if self.first_coordinate is not None else None
        )
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True)
class LogitComparison:
    tensor: TensorComparison
    kl_next_token: float
    reference_top1: int
    candidate_top1: int

    @property
    def top1_agreement(self) -> bool:
        return self.reference_top1 == self.candidate_top1

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor.to_dict(),
            "kl_next_token": self.kl_next_token,
            "reference_top1": self.reference_top1,
            "candidate_top1": self.candidate_top1,
            "top1_agreement": self.top1_agreement,
        }


def _coordinate(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinate: list[int] = []
    remainder = flat_index
    for size in reversed(shape):
        coordinate.append(remainder % size)
        remainder //= size
    return tuple(reversed(coordinate))


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> TensorComparison:
    """Compare shape/dtype/content and identify the first differing coordinate."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    if reference.numel() == 0:
        exact = reference.dtype == candidate.dtype and reference.device == candidate.device
        return TensorComparison(
            exact, 0.0, None, None, None, tuple(reference.shape),
            str(reference.dtype), str(candidate.dtype), str(reference.device), str(candidate.device),
        )

    same_dtype = reference.dtype == candidate.dtype
    same_device = reference.device == candidate.device
    same_values = bool(torch.equal(reference, candidate))
    ref64 = reference.detach().to(device="cpu", dtype=torch.float64)
    cand64 = candidate.detach().to(device="cpu", dtype=torch.float64)
    delta = torch.abs(ref64 - cand64)
    max_abs_delta = float(torch.max(delta).item())
    different = torch.ne(reference.detach().cpu(), candidate.detach().cpu()).reshape(-1)
    first_coordinate: tuple[int, ...] | None = None
    reference_value: float | int | bool | None = None
    candidate_value: float | int | bool | None = None
    if bool(torch.any(different)):
        flat_index = int(torch.nonzero(different, as_tuple=False)[0].item())
        first_coordinate = _coordinate(flat_index, tuple(reference.shape))
        reference_value = reference.detach().cpu()[first_coordinate].item()
        candidate_value = candidate.detach().cpu()[first_coordinate].item()
    return TensorComparison(
        same_dtype and same_device and same_values,
        max_abs_delta,
        first_coordinate,
        reference_value,
        candidate_value,
        tuple(reference.shape),
        str(reference.dtype),
        str(candidate.dtype),
        str(reference.device),
        str(candidate.device),
    )


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> LogitComparison:
    """Compare one next-token logit vector; KL is FP64 and diagnostic only."""
    if reference.ndim != 1 or candidate.ndim != 1:
        raise ValueError("next-token logits must be one-dimensional")
    tensor = compare_tensors(reference, candidate)
    ref = reference.detach().to(device="cpu", dtype=torch.float64)
    cand = candidate.detach().to(device="cpu", dtype=torch.float64)
    ref_logp = torch.log_softmax(ref, dim=-1)
    cand_logp = torch.log_softmax(cand, dim=-1)
    kl = torch.sum(torch.exp(ref_logp) * (ref_logp - cand_logp))
    return LogitComparison(
        tensor=tensor,
        kl_next_token=max(0.0, float(kl.item())),
        reference_top1=int(torch.argmax(ref).item()),
        candidate_top1=int(torch.argmax(cand).item()),
    )
