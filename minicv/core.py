"""Core image operations such as normalization, clipping, padding, and convolution."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .utils import ensure_float32, parse_padding, validate_grayscale_image, validate_image_array, validate_kernel


def normalize_image(image: np.ndarray, mode: str = "minmax", eps: float = 1e-8) -> np.ndarray:
    """Normalize image intensities using one of several modes.

    Parameters
    ----------
    image:
        Input grayscale or RGB image.
    mode:
        One of ``"minmax"``, ``"zscore"``, or ``"unit"``.
    eps:
        Small value to avoid division by zero.
    """
    validate_image_array(image)
    image_f = ensure_float32(image)
    if mode == "minmax":
        min_val = float(np.min(image_f))
        max_val = float(np.max(image_f))
        return ((image_f - min_val) / max(max_val - min_val, eps) * 255.0).astype(np.float32)
    if mode == "zscore":
        mean = float(np.mean(image_f))
        std = float(np.std(image_f))
        return ((image_f - mean) / max(std, eps)).astype(np.float32)
    if mode == "unit":
        min_val = float(np.min(image_f))
        max_val = float(np.max(image_f))
        return ((image_f - min_val) / max(max_val - min_val, eps)).astype(np.float32)
    raise ValueError(f"unsupported normalization mode {mode!r}; use 'minmax', 'zscore', or 'unit'.")


def clip_pixels(image: np.ndarray, min_value: float = 0.0, max_value: float = 255.0) -> np.ndarray:
    """Clip image intensities to a target interval."""
    validate_image_array(image)
    if min_value > max_value:
        raise ValueError(f"min_value must be <= max_value, got {min_value} and {max_value}.")
    return np.clip(ensure_float32(image), min_value, max_value).astype(np.float32)


def pad_image(image: np.ndarray, pad_width: int | tuple[int, int], mode: str = "constant", constant_values: float = 0.0) -> np.ndarray:
    """Pad an image using constant, edge, or reflect mode."""
    validate_image_array(image)
    if mode not in {"constant", "edge", "reflect"}:
        raise ValueError(f"padding mode must be 'constant', 'edge', or 'reflect', got {mode!r}.")
    pad_y, pad_x = parse_padding(pad_width)
    pad_spec = ((pad_y, pad_y), (pad_x, pad_x)) if image.ndim == 2 else ((pad_y, pad_y), (pad_x, pad_x), (0, 0))
    kwargs = {"mode": mode}
    if mode == "constant":
        kwargs["constant_values"] = constant_values
    return np.pad(ensure_float32(image), pad_spec, **kwargs)


def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: str = "reflect", constant_values: float = 0.0) -> np.ndarray:
    """Perform true 2D convolution on a grayscale image."""
    validate_grayscale_image(image)
    kernel_f = validate_kernel(kernel)
    image_f = ensure_float32(image)
    ky, kx = kernel_f.shape
    pad_y, pad_x = ky // 2, kx // 2
    padded = pad_image(image_f, (pad_y, pad_x), mode=padding, constant_values=constant_values)
    windows = sliding_window_view(padded, (ky, kx))
    flipped = np.flip(kernel_f, axis=(0, 1))
    return np.einsum("ijkl,kl->ij", windows, flipped, optimize=True).astype(np.float32)
