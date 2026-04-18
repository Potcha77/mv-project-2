"""Feature extraction methods for milestone one."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .filtering import sobel_gradients
from .utils import grayscale_view, validate_image_array


def intensity_statistics(image: np.ndarray) -> pd.Series:
    """Return a global intensity descriptor with common summary statistics."""
    validate_image_array(image)
    gray = grayscale_view(image)
    flat = gray.ravel().astype(np.float32)
    return pd.Series({
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "energy": float(np.mean(flat**2)),
    })


def grayscale_histogram_descriptor(image: np.ndarray, bins: int = 16, normalize: bool = True) -> pd.Series:
    """Return a global histogram descriptor from a grayscale image."""
    validate_image_array(image)
    gray = grayscale_view(image)
    hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0.0, 255.0))
    hist = hist.astype(np.float32)
    if normalize:
        hist /= max(hist.sum(), 1.0)
    names = [f"hist_bin_{idx:02d}" for idx in range(bins)]
    return pd.Series(hist, index=names)


def gradient_statistics(image: np.ndarray, magnitude_bins: int = 8, orientation_bins: int = 8) -> pd.Series:
    """Return a gradient-based descriptor using Sobel magnitude and orientation histograms."""
    validate_image_array(image)
    _, _, magnitude, direction = sobel_gradients(image)
    mag_hist, _ = np.histogram(magnitude.ravel(), bins=magnitude_bins, range=(0.0, float(np.max(magnitude) + 1e-6)))
    dir_hist, _ = np.histogram(direction.ravel(), bins=orientation_bins, range=(-180.0, 180.0))
    mag_hist = mag_hist.astype(np.float32)
    dir_hist = dir_hist.astype(np.float32)
    mag_hist /= max(mag_hist.sum(), 1.0)
    dir_hist /= max(dir_hist.sum(), 1.0)
    data = {}
    for idx, value in enumerate(mag_hist):
        data[f"grad_mag_bin_{idx:02d}"] = float(value)
    for idx, value in enumerate(dir_hist):
        data[f"grad_dir_bin_{idx:02d}"] = float(value)
    return pd.Series(data)


def hog_descriptor(image: np.ndarray, cell_size: int = 8, bins: int = 9) -> pd.Series:
    """Compute a simple Histogram of Oriented Gradients descriptor."""
    validate_image_array(image)
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}.")
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}.")
    gray = grayscale_view(image)
    _, _, magnitude, direction = sobel_gradients(gray)
    direction = np.mod(direction, 180.0)
    h, w = gray.shape
    usable_h = (h // cell_size) * cell_size
    usable_w = (w // cell_size) * cell_size
    if usable_h == 0 or usable_w == 0:
        raise ValueError(f"image is too small for cell_size={cell_size}; got image shape {gray.shape}.")
    magnitude = magnitude[:usable_h, :usable_w]
    direction = direction[:usable_h, :usable_w]
    bin_edges = np.linspace(0.0, 180.0, bins + 1, dtype=np.float32)
    features: list[float] = []
    names: list[str] = []
    cell_id = 0
    for row in range(0, usable_h, cell_size):
        for col in range(0, usable_w, cell_size):
            mag_cell = magnitude[row:row+cell_size, col:col+cell_size]
            dir_cell = direction[row:row+cell_size, col:col+cell_size]
            hist, _ = np.histogram(dir_cell, bins=bin_edges, weights=mag_cell)
            hist = hist.astype(np.float32)
            norm = np.linalg.norm(hist) + 1e-8
            hist /= norm
            for b, value in enumerate(hist):
                features.append(float(value))
                names.append(f"hog_cell_{cell_id:03d}_bin_{b:02d}")
            cell_id += 1
    return pd.Series(features, index=names)
