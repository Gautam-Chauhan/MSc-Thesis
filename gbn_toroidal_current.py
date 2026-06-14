"""Toroidal Gaussian Blue Noise sampler using PyTorch.

This is a CUDA-free Python translation of ``gbn-toroidal.cu`` from the
Gaussian Blue Noise code release. It optimizes points in the unit torus and
then maps them into a user-supplied axis-aligned box. The optimization can run
on CUDA, MPS, or CPU through PyTorch.
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


MAX_DIST_FLOAT32 = 5.8
MAX_DIST_FLOAT64 = 8.5
MAX_PERIODS = 9


@dataclass(frozen=True)
class GBNResult:
    """Generated points plus the normalized torus state."""

    points: np.ndarray
    unit_points: np.ndarray
    bounds: np.ndarray
    device: str
    periods: int


def _as_bounds(bounds: Iterable[Iterable[float]] | Iterable[float], dims: int) -> np.ndarray:
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
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    requested = torch.device(device)
    if requested.type == "auto":
        raise ValueError("device must be 'auto' or a valid PyTorch device")
    return requested


def _period_count(point_count: int, dims: int, sigma: float, dtype: torch.dtype) -> int:
    max_dist = MAX_DIST_FLOAT64 if dtype == torch.float64 else MAX_DIST_FLOAT32
    periods = int(max_dist * sigma / (point_count ** (1.0 / dims)))
    return min(periods, MAX_PERIODS)


def _nearest_torus_grad(
    points: torch.Tensor,
    ss2_inv_neg: float,
    chunk_size: int,
) -> torch.Tensor:
    """Gradient from the nearest wrapped copy of every other point."""

    count = points.shape[0]
    grad = torch.zeros_like(points)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        diff = points[start:stop, None, :] - points[None, :, :]
        diff = torch.remainder(diff + 0.5, 1.0) - 0.5
        squared_dist = torch.sum(diff * diff, dim=2)
        weights = torch.exp(ss2_inv_neg * squared_dist)
        rows = torch.arange(stop - start, device=points.device)
        weights[rows, start + rows] = 0.0
        grad[start:stop] = torch.sum(diff * weights[:, :, None], dim=1)

    return grad


def _periodic_axis_grad(
    points: torch.Tensor,
    ss2_inv_neg: float,
    periods: int,
    chunk_size: int,
) -> torch.Tensor:
    """Gradient from the separable multi-period Gaussian sum in gbnND<K>."""

    count, dims = points.shape
    grad = torch.zeros_like(points)
    offsets = torch.arange(
        -periods,
        periods,
        device=points.device,
        dtype=points.dtype,
    )

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        chunk = points[start:stop]
        g_dims = []
        grad_dims = []

        for dim in range(dims):
            dx = chunk[:, None, dim] - points[None, :, dim]
            dx = torch.where(dx < 0.0, dx + 1.0, dx)
            dx_images = dx[:, :, None] + offsets
            weights = torch.exp(ss2_inv_neg * dx_images * dx_images)
            g_dims.append(weights.sum(dim=2))
            grad_dims.append((dx_images * weights).sum(dim=2))

        g = torch.stack(g_dims, dim=2)
        grad_per_dim = torch.stack(grad_dims, dim=2)
        rows = torch.arange(stop - start, device=points.device)
        g[rows, start + rows, :] = 0.0
        grad_per_dim[rows, start + rows, :] = 0.0

        if dims == 1:
            grad[start:stop] = grad_per_dim[:, :, 0].sum(dim=1, keepdim=True)
            continue

        for dim in range(dims):
            product = grad_per_dim[:, :, dim]
            for other_dim in range(dims):
                if other_dim != dim:
                    product = product * g[:, :, other_dim]
            grad[start:stop, dim] = product.sum(dim=1)

    return grad


def optimize_unit_points(
    initial_points: np.ndarray | torch.Tensor,
    sigma: float,
    iterations: int,
    *,
    step_scale: float = 0.25,
    chunk_size: int = 1024,
    progress: bool = False,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float32,
    return_periods: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Optimize points in the unit torus ``[0, 1)^D`` using PyTorch.

    Args:
        initial_points: Array or tensor with shape ``(point_count, dims)``.
        sigma: Gaussian filter sigma, matching the C++ ``-g`` parameter.
        iterations: Number of optimizer iterations.
        step_scale: Base time-step. The default matches ``gbn-toroidal.cu``.
        chunk_size: Number of source points processed per pairwise chunk.
        progress: Print iteration progress to stderr.
        device: ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``, or any PyTorch device.
        dtype: Floating dtype for optimization. ``float32`` is best for MPS.
        return_periods: Return the selected image-period count with the points.
    """

    selected_device = _resolve_device(device)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    if selected_device.type == "mps" and dtype == torch.float64:
        raise ValueError("PyTorch MPS does not support float64; use torch.float32")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if step_scale <= 0:
        raise ValueError("step_scale must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    points = torch.as_tensor(initial_points, dtype=dtype, device=selected_device).clone()
    if points.ndim != 2:
        raise ValueError("initial_points must have shape (point_count, dims)")
    if torch.any((points < 0.0) | (points >= 1.0)).item():
        raise ValueError("initial_points must lie in the unit torus [0, 1)")

    count, dims = points.shape
    if count <= 0:
        raise ValueError("initial_points must contain at least one point")
    if dims <= 0:
        raise ValueError("initial_points must contain at least one dimension")
    if dims > 32:
        raise ValueError("gbn-toroidal.cu supports at most 32 dimensions")

    ss2_inv_neg = -(count ** (2.0 / dims)) / (2.0 * sigma * sigma)
    scale = step_scale * (count ** (2.0 / dims)) / count
    if sigma > 1.0:
        scale /= sigma * sigma
    periods = _period_count(count, dims, sigma, dtype)

    with torch.no_grad():
        for iteration in range(iterations):
            if progress:
                print(
                    f"\rIteration {iteration + 1}/{iterations} on {selected_device}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            if periods == 0:
                grad = _nearest_torus_grad(points, ss2_inv_neg, chunk_size)
            else:
                grad = _periodic_axis_grad(points, ss2_inv_neg, periods, chunk_size)
            points = torch.remainder(points + scale * grad, 1.0)

    if progress and iterations:
        print(file=sys.stderr)

    unit_points = points.cpu().numpy().astype(np.float64)
    if return_periods:
        return unit_points, periods
    return unit_points


def sample_toroidal_gbn(
    point_count: int,
    dims: int,
    bounds: Iterable[Iterable[float]] | Iterable[float],
    sigma: float,
    iterations: int,
    *,
    seed: int | None = None,
    step_scale: float = 0.25,
    chunk_size: int = 1024,
    progress: bool = False,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float32,
) -> GBNResult:
    """Generate Gaussian blue noise points in a rectangular toroidal domain.

    ``bounds`` can be either one ``[min, max]`` pair reused for every dimension,
    or one pair per dimension, for example ``[[0, 1], [-2, 2], [10, 20]]``.
    The optimizer runs in normalized torus coordinates and the returned
    ``points`` are mapped into the requested bounds.
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
    unit_points, periods = optimize_unit_points(
        initial,
        sigma,
        iterations,
        step_scale=step_scale,
        chunk_size=chunk_size,
        progress=progress,
        device=selected_device,
        dtype=dtype,
        return_periods=True,
    )

    lower = parsed_bounds[:, 0]
    widths = parsed_bounds[:, 1] - parsed_bounds[:, 0]
    points = lower + unit_points * widths
    return GBNResult(
        points=points,
        unit_points=unit_points,
        bounds=parsed_bounds,
        device=str(selected_device),
        periods=periods,
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
        description="Generate toroidal Gaussian blue noise samples with PyTorch."
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
    parser.add_argument("--output", "-o", type=Path, default=Path("gbn_toroidal_points.txt"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--step-scale",
        "-s",
        type=float,
        default=0.25,
        help="Base time-step before the original dimension/N scaling.",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cuda, mps, cpu, etc. Default prefers CUDA then MPS.",
    )
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.float32)
    args = parser.parse_args()

    try:
        bounds = _parse_bounds(args.bounds, args.dims)
        result = sample_toroidal_gbn(
            args.point_count,
            args.dims,
            bounds,
            args.sigma,
            args.iterations,
            seed=args.seed,
            step_scale=args.step_scale,
            chunk_size=args.chunk_size,
            progress=True,
            device=args.device,
            dtype=args.dtype,
        )
    except ValueError as exc:
        parser.error(str(exc))

    save_points(args.output, result.points)
    print(
        f"Saved {len(result.points)} points to {args.output} "
        f"using {result.device}; periods = {result.periods}"
    )


if __name__ == "__main__":
    main()
