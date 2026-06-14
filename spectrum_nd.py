"""N-dimensional power spectrum for point sets.

This is a PyTorch translation of ``spectrum-nd.h`` from the Gaussian Blue
Noise code release.  Given one point set with shape ``(N, D)`` or a batch of
sets with shape ``(S, N, D)``, it evaluates the Fourier power on the integer
frequency grid ``[-size, size]^D`` and returns an array shaped
``(2 * size + 1,) * D``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to an available PyTorch accelerator or CPU."""

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


def frequency_grid(
    size: int,
    dims: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return all integer frequency vectors in the C++ header's order.

    The first dimension varies fastest in the flattened order, matching:
    ``freq[dim] = tmp % base - size; tmp /= base``.
    """

    if size < 0:
        raise ValueError("size must be non-negative")
    if dims <= 0:
        raise ValueError("dims must be positive")

    selected_device = torch.device(device)
    base = 2 * size + 1
    flat_ids = torch.arange(base**dims, device=selected_device)
    frequencies = []
    tmp = flat_ids
    for _ in range(dims):
        frequencies.append((tmp % base).to(dtype) - size)
        tmp = torch.div(tmp, base, rounding_mode="floor")
    return torch.stack(frequencies, dim=1)


def _frequency_grid_from_flat_ids(
    flat_ids: torch.Tensor,
    size: int,
    dims: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return integer frequency vectors for a chunk of flattened indices."""

    base = 2 * size + 1
    frequencies = []
    tmp = flat_ids
    for _ in range(dims):
        frequencies.append((tmp % base).to(dtype) - size)
        tmp = torch.div(tmp, base, rounding_mode="floor")
    return torch.stack(frequencies, dim=1)


def power_spectrum(
    points: np.ndarray | torch.Tensor,
    size: int,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float32,
    frequency_chunk_size: int = 8192,
    return_frequencies: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the Gaussian Blue Noise reference power spectrum.

    Args:
        points: Point coordinates in the unit hypercube with shape ``(N, D)``,
            or a batch of point sets with shape ``(S, N, D)``.
        size: Maximum integer frequency along each dimension.  The output grid
            has shape ``(2 * size + 1,) * D``.
        device: ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``, or any valid
            PyTorch device.  ``"auto"`` prefers CUDA, then MPS, then CPU.
        dtype: ``torch.float32`` or ``torch.float64``.  MPS requires float32.
        frequency_chunk_size: Number of frequencies evaluated per chunk.
        return_frequencies: If true, also return the integer frequency vectors
            with shape ``((2 * size + 1) ** D, D)``.

    Returns:
        A NumPy array containing the power spectrum.  If ``return_frequencies``
        is true, returns ``(spectrum, frequencies)``.  ``frequencies`` follows
        the flattened CUDA-header order where dimension 0 varies fastest.
    """

    if size < 0:
        raise ValueError("size must be non-negative")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    if frequency_chunk_size <= 0:
        raise ValueError("frequency_chunk_size must be positive")

    selected_device = resolve_device(device)
    if selected_device.type == "mps" and dtype == torch.float64:
        raise ValueError("PyTorch MPS does not support float64; use torch.float32")

    point_tensor = torch.as_tensor(points, dtype=dtype, device=selected_device)
    if point_tensor.ndim == 2:
        point_tensor = point_tensor.unsqueeze(0)
    if point_tensor.ndim != 3:
        raise ValueError(
            "points must have shape (point_count, dims) "
            "or (sets, point_count, dims)"
        )

    set_count, point_count, dims = point_tensor.shape
    if set_count <= 0 or point_count <= 0:
        raise ValueError("points must contain at least one set and one point")
    if dims <= 0:
        raise ValueError("points must contain at least one dimension")
    if dims > 16:
        raise ValueError("spectrum-nd.h supports at most 16 dimensions")

    base = 2 * size + 1
    volume = base**dims
    output_dtype = np.float32 if dtype == torch.float32 else np.float64
    spectrum_flat = np.empty(volume, dtype=output_dtype)
    angular_scale = -2.0 * math.pi

    with torch.no_grad():
        for start in range(0, volume, frequency_chunk_size):
            stop = min(start + frequency_chunk_size, volume)
            flat_ids = torch.arange(start, stop, device="cpu")
            frequencies = _frequency_grid_from_flat_ids(
                flat_ids,
                size,
                dims,
                dtype,
            ).to(selected_device)
            angular_frequencies = angular_scale * frequencies
            angles = torch.einsum(
                "fd,snd->sfn",
                angular_frequencies,
                point_tensor,
            )
            real = torch.cos(angles).sum(dim=2)
            imaginary = torch.sin(angles).sum(dim=2)
            power = real * real + imaginary * imaginary
            spectrum_chunk = power.sum(dim=0) / (set_count * point_count)
            spectrum_flat[start:stop] = spectrum_chunk.cpu().numpy()

    spectrum_np = spectrum_flat.reshape((base,) * dims, order="F")
    if not return_frequencies:
        return spectrum_np

    frequencies_np = frequency_grid(size, dims, device="cpu", dtype=dtype).cpu().numpy()
    return spectrum_np, frequencies_np


def _load_points(path: Path) -> np.ndarray:
    """Load points saved by ``gbn_bounded.save_points`` or plain whitespace data."""

    lines = path.read_text(encoding="utf-8").splitlines()
    first = lines[0].split() if lines else []
    if len(first) in (1, 2):
        try:
            header_values = [int(value) for value in first]
            integer_header = all(
                str(value) == token for value, token in zip(header_values, first)
            )
        except ValueError:
            integer_header = False
        if integer_header:
            try:
                header_data = np.loadtxt(path, skiprows=1)
            except ValueError:
                header_data = np.array([])
            if header_data.ndim == 1 and header_data.size > 0:
                header_data = header_data[None, :]
            if (
                header_data.ndim == 2
                and header_values[0] == header_data.shape[0]
                and (
                    len(header_values) == 1
                    or header_values[1] == header_data.shape[1]
                )
            ):
                return header_data

    data = np.loadtxt(path)
    if data.ndim != 2:
        raise ValueError("input file must contain a 2D table of point coordinates")
    return data


def _parse_dtype(text: str) -> torch.dtype:
    if text == "float32":
        return torch.float32
    if text == "float64":
        return torch.float64
    raise argparse.ArgumentTypeError("dtype must be float32 or float64")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute an N-dimensional point-set power spectrum with PyTorch."
    )
    parser.add_argument("input", type=Path, help="Whitespace-delimited point file.")
    parser.add_argument("size", type=int, help="Maximum integer frequency per axis.")
    parser.add_argument("--output", "-o", type=Path, default=Path("spectrum.npy"))
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, cpu, etc.")
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.float32)
    parser.add_argument("--frequency-chunk-size", type=int, default=8192)
    args = parser.parse_args()

    try:
        points = _load_points(args.input)
        spectrum = power_spectrum(
            points,
            args.size,
            device=args.device,
            dtype=args.dtype,
            frequency_chunk_size=args.frequency_chunk_size,
        )
    except ValueError as exc:
        parser.error(str(exc))

    np.save(args.output, spectrum)
    print(f"Saved spectrum with shape {spectrum.shape} to {args.output}")


if __name__ == "__main__":
    main()
