"""Bounded Gaussian Blue Noise sampler using PyTorch.

This is a CUDA-free Python translation of the bounded-domain optimizer in
``gbn-bounded.cu`` from the Gaussian Blue Noise code release.  It samples inside
an axis-aligned rectangular domain in arbitrary dimension and can run on the
integrated GPU in a MacBook through PyTorch's MPS backend.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class GBNResult:
    """Generated points plus the normalized optimizer state."""

    points: np.ndarray
    unit_points: np.ndarray
    bounds: np.ndarray
    device: str


def _as_bounds(bounds: Iterable[Iterable[float]], dims: int) -> np.ndarray:
    parsed = np.asarray(bounds, dtype=np.float64)
    if parsed.shape == (2,):
        parsed = np.repeat(parsed[None, :], dims, axis=0)
    if parsed.shape != (dims, 2):
        raise ValueError(
            f"bounds must have shape ({dims}, 2), or be a single [min, max] pair"
        )
    if np.any(parsed[:, 1] <= parsed[:, 0]):
        raise ValueError("each bounds row must be [min, max] with max > min")
    return parsed


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str) and device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    requested = torch.device(device)
    if requested.type == "auto":
        raise ValueError("device must be 'auto' or a valid PyTorch device")
    return requested


def _domain_attraction_torch(
    points: torch.Tensor,
    sigma_unit: float,
    sigma_optimizer: float,
    n_per_axis: float,
) -> torch.Tensor:
    """Generalized rectangular-domain attraction force.

    For D=2 this reduces to the force in ``gbn-bounded.cu``:
    grad_x = a * (erf(bottom) + erf(top)) * (exp(-left^2) - exp(-right^2)).
    """

    s2_inv = 1.0 / (2.0 * sigma_unit)
    lower = s2_inv * points
    upper = s2_inv * (1.0 - points)
    erf_sums = torch.erf(lower) + torch.erf(upper)
    exp_diff = torch.exp(-(lower * lower)) - torch.exp(-(upper * upper))

    dims = points.shape[1]
    if dims == 1:
        other_dims_product = torch.ones_like(points)
    else:
        product_all = torch.prod(erf_sums, dim=1, keepdim=True)
        other_dims_product = product_all / torch.clamp(erf_sums, min=1e-30)

    dims = points.shape[1]
    attraction_scale = (
        2.0
        * sigma_optimizer ** (dims + 1)
        * math.pi ** ((dims - 1) / 2.0)
        / n_per_axis
    )
    return attraction_scale * other_dims_product * exp_diff


def _repulsion_torch(
    points: torch.Tensor,
    sigma_unit: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute pairwise Gaussian repulsion in chunks on the selected device."""

    count = points.shape[0]
    grad = torch.zeros_like(points)
    neg_inv_4_sigma2 = -1.0 / (4.0 * sigma_unit * sigma_unit)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        diff = points[start:stop, None, :] - points[None, :, :]
        squared_dist = torch.sum(diff * diff, dim=2)
        weights = torch.exp(neg_inv_4_sigma2 * squared_dist)
        rows = torch.arange(stop - start, device=points.device)
        weights[rows, start + rows] = 0.0
        grad[start:stop] = torch.sum(diff * weights[:, :, None], dim=1)
    return grad


def optimize_unit_points(
    initial_points: np.ndarray | torch.Tensor,
    sigma: float,
    iterations: int,
    *,
    step_scale: float | None = None,
    chunk_size: int = 1024,
    project: bool = True,
    progress: bool = False,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Optimize points in the unit hypercube ``[0, 1]^D`` using PyTorch.

    Args:
        initial_points: Array or tensor with shape ``(point_count, dims)``.
        sigma: Gaussian filter sigma, matching the C++ command-line meaning.
        iterations: Number of relaxation iterations.
        step_scale: Time-step multiplier. If omitted, uses the original 2D
            value for 2D and a smaller dimension-aware value above 2D.
        chunk_size: Number of points processed per pairwise-repulsion chunk.
        project: Clip points back to the unit hypercube after each iteration.
        progress: Print iteration progress to stderr.
        device: ``"auto"``, ``"mps"``, ``"cpu"``, or any valid PyTorch device.
        dtype: Floating dtype for optimization. ``float32`` is best for MPS.
    """

    selected_device = _resolve_device(device)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    if selected_device.type == "mps" and dtype == torch.float64:
        raise ValueError("PyTorch MPS does not support float64; use torch.float32")

    points = torch.as_tensor(initial_points, dtype=dtype, device=selected_device).clone()
    if points.ndim != 2:
        raise ValueError("initial_points must have shape (point_count, dims)")
    if torch.any((points < 0.0) | (points > 1.0)).item():
        raise ValueError("initial_points must lie in the unit hypercube")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    count, dims = points.shape
    if step_scale is None:
        step_scale = 1.0 if dims <= 2 else 1.0 / dims
    if step_scale <= 0:
        raise ValueError("step_scale must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    n_per_axis = count ** (1.0 / dims)
    optimizer_sigma = sigma / math.sqrt(2.0)
    sigma_unit = optimizer_sigma / n_per_axis

    scale = (0.5)**dims * step_scale
    if sigma > 1.0:
        scale /= sigma * sigma

    with torch.no_grad():
        for iteration in range(iterations):
            if progress:
                print(
                    f"\rIteration {iteration + 1}/{iterations} on {selected_device}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            grad = _domain_attraction_torch(
                points,
                sigma_unit,
                optimizer_sigma,
                n_per_axis,
            )
            grad += _repulsion_torch(points, sigma_unit, chunk_size)
            points += scale * grad
            if project:
                points.clamp_(0.0, 1.0)

    if progress and iterations:
        print(file=sys.stderr)
    return points.cpu().numpy().astype(np.float64)


def sample_bounded_gbn(
    point_count: int,
    dims: int,
    bounds: Iterable[Iterable[float]] | Iterable[float],
    sigma: float,
    iterations: int,
    *,
    seed: int | None = None,
    step_scale: float | None = None,
    chunk_size: int = 1024,
    project: bool = True,
    progress: bool = False,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float32,
) -> GBNResult:
    """Generate bounded Gaussian blue noise points.

    ``bounds`` is an axis-aligned box.  Pass either one ``[min, max]`` pair to
    use for every dimension, or one pair per dimension, for example
    ``[[0, 1], [-2, 2], [10, 20]]`` for 3D.
    """

    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if dims <= 0:
        raise ValueError("dims must be positive")

    selected_device = _resolve_device(device)
    parsed_bounds = _as_bounds(bounds, dims)
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    initial = torch.rand((point_count, dims), generator=generator, dtype=dtype)
    unit_points = optimize_unit_points(
        initial,
        sigma,
        iterations,
        step_scale=step_scale,
        chunk_size=chunk_size,
        project=project,
        progress=progress,
        device=selected_device,
        dtype=dtype,
    )

    lower = parsed_bounds[:, 0]
    widths = parsed_bounds[:, 1] - parsed_bounds[:, 0]
    points = lower + unit_points * widths
    return GBNResult(
        points=points,
        unit_points=unit_points,
        bounds=parsed_bounds,
        device=str(selected_device),
    )


def save_points(path: str | Path, points: np.ndarray) -> None:
    """Save points in the same simple text format used by the C++ code."""

    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must have shape (point_count, dims)")
    header = (
        str(points.shape[0])
        if points.shape[1] == 2
        else f"{points.shape[0]} {points.shape[1]}"
    )
    np.savetxt(path, points, fmt="%.17g", header=header, comments="")


def _parse_bounds(text: str, dims: int) -> np.ndarray:
    values = [float(part) for part in text.replace(",", " ").split()]
    if len(values) == 2:
        return _as_bounds(values, dims)
    if len(values) != 2 * dims:
        raise ValueError(
            f"expected 2 values, or {2 * dims} values for {dims}D bounds"
        )
    return _as_bounds(np.asarray(values).reshape(dims, 2), dims)


def _parse_dtype(text: str) -> torch.dtype:
    if text == "float32":
        return torch.float32
    if text == "float64":
        return torch.float64
    raise argparse.ArgumentTypeError("dtype must be float32 or float64")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bounded Gaussian blue noise samples with PyTorch."
    )
    parser.add_argument("point_count", type=int)
    parser.add_argument("iterations", type=int)
    parser.add_argument("--dims", "-d", type=int, default=2)
    parser.add_argument("--sigma", "-g", type=float, default=1.0)
    parser.add_argument(
        "--bounds",
        default="0 1",
        help="Either 'min max' for all axes, or 'x0 x1 y0 y1 ...'.",
    )
    parser.add_argument("--output", "-o", type=Path, default=Path("gbn_points.txt"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--step-scale",
        "-s",
        type=float,
        default=None,
        help="Time-step multiplier. Default is 1 in 2D and 1/dims above 2D.",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, mps, cpu, cuda, etc. Default prefers MPS.",
    )
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.float32)
    parser.add_argument(
        "--no-project",
        action="store_true",
        help="Do not clip points to the bounded domain after each iteration.",
    )
    args = parser.parse_args()

    try:
        bounds = _parse_bounds(args.bounds, args.dims)
        result = sample_bounded_gbn(
            args.point_count,
            args.dims,
            bounds,
            args.sigma,
            args.iterations,
            seed=args.seed,
            step_scale=args.step_scale,
            chunk_size=args.chunk_size,
            project=not args.no_project,
            progress=True,
            device=args.device,
            dtype=args.dtype,
        )
    except ValueError as exc:
        parser.error(str(exc))

    save_points(args.output, result.points)
    print(f"Saved {len(result.points)} points to {args.output} using {result.device}")


if __name__ == "__main__":
    main()
