"""Filtering and classic image-processing techniques."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .core import clip_pixels, convolve2d, pad_image
from .utils import ensure_float32, grayscale_view, validate_grayscale_image, validate_image_array, validate_kernel


def apply_filter(image: np.ndarray, kernel: np.ndarray, padding: str = "reflect") -> np.ndarray:
    """Apply a convolution-based filter to grayscale or RGB images."""
    validate_image_array(image)
    kernel_f = validate_kernel(kernel)
    if image.ndim == 2:
        return convolve2d(image, kernel_f, padding=padding)
    channels = [convolve2d(image[..., c], kernel_f, padding=padding) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1).astype(np.float32)


def mean_filter(image: np.ndarray, kernel_size: int = 3, padding: str = "reflect") -> np.ndarray:
    """Blur an image using a box filter."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
    kernel = np.full((kernel_size, kernel_size), 1.0 / (kernel_size * kernel_size), dtype=np.float32)
    return apply_filter(image, kernel, padding=padding)


def gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Generate a normalized 2D Gaussian kernel."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    radius = kernel_size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, padding: str = "reflect") -> np.ndarray:
    """Blur an image using a Gaussian kernel."""
    return apply_filter(image, gaussian_kernel(kernel_size, sigma), padding=padding)


def median_filter(image: np.ndarray, kernel_size: int = 3, padding: str = "reflect") -> np.ndarray:
    """Apply a median filter to grayscale or RGB images.

    Notes
    -----
    Median filtering is not linear, so it cannot be expressed as a convolution.
    This implementation uses sliding windows and a median reduction, which is a controlled and justified use of window operations.
    """
    validate_image_array(image)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

    def _median_2d(channel: np.ndarray) -> np.ndarray:
        padded = pad_image(channel, kernel_size // 2, mode=padding)
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        return np.median(windows, axis=(-2, -1)).astype(np.float32)

    if image.ndim == 2:
        return _median_2d(ensure_float32(image))
    channels = [_median_2d(image[..., c]) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def global_threshold(image: np.ndarray, threshold: float, max_value: float = 255.0) -> np.ndarray:
    """Apply global thresholding to a grayscale image."""
    validate_grayscale_image(image)
    image_f = ensure_float32(image)
    return np.where(image_f >= threshold, max_value, 0.0).astype(np.float32)


def otsu_threshold(image: np.ndarray, max_value: float = 255.0) -> tuple[float, np.ndarray]:
    """Compute Otsu's threshold and return the thresholded image."""
    validate_grayscale_image(image)
    img = clip_pixels(image, 0.0, 255.0).astype(np.uint8)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    threshold = float(np.argmax(sigma_b))
    return threshold, global_threshold(img.astype(np.float32), threshold, max_value=max_value)


def adaptive_threshold(image: np.ndarray, block_size: int = 11, c: float = 2.0, method: str = "mean", max_value: float = 255.0) -> np.ndarray:
    """Apply adaptive thresholding using local mean or Gaussian weighting."""
    validate_grayscale_image(image)
    if block_size <= 0 or block_size % 2 == 0:
        raise ValueError(f"block_size must be a positive odd integer, got {block_size}.")
    if method not in {"mean", "gaussian"}:
        raise ValueError(f"method must be 'mean' or 'gaussian', got {method!r}.")
    image_f = ensure_float32(image)
    if method == "mean":
        local = mean_filter(image_f, kernel_size=block_size)
    else:
        sigma = max(block_size / 6.0, 1.0)
        local = gaussian_filter(image_f, kernel_size=block_size, sigma=sigma)
    return np.where(image_f > (local - c), max_value, 0.0).astype(np.float32)


def sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel x-gradient, y-gradient, magnitude, and direction in degrees."""
    gray = grayscale_view(image)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve2d(gray, kx)
    gy = convolve2d(gray, ky)
    mag = np.sqrt(gx**2 + gy**2).astype(np.float32)
    direction = np.degrees(np.arctan2(gy, gx)).astype(np.float32)
    return gx, gy, mag, direction


def bit_plane_slicing(image: np.ndarray) -> np.ndarray:
    """Return all 8 bit planes of a grayscale image as an array of shape (8, H, W)."""
    validate_grayscale_image(image)
    img = clip_pixels(image, 0.0, 255.0).astype(np.uint8)
    planes = [(img >> bit) & 1 for bit in range(8)]
    return np.stack(planes, axis=0).astype(np.uint8)


def histogram(image: np.ndarray, bins: int = 256, value_range: tuple[float, float] = (0.0, 255.0)) -> tuple[np.ndarray, np.ndarray]:
    """Compute a grayscale histogram."""
    gray = grayscale_view(image)
    hist, edges = np.histogram(gray.ravel(), bins=bins, range=value_range)
    return hist.astype(np.int64), edges.astype(np.float32)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Equalize the histogram of a grayscale image."""
    validate_grayscale_image(image)
    img = clip_pixels(image, 0.0, 255.0).astype(np.uint8)
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = hist.cumsum().astype(np.float64)
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return img.astype(np.float32)
    cdf_min = nonzero[0]
    lut = np.round((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[img].astype(np.float32)


def laplacian_filter(image: np.ndarray, padding: str = "reflect") -> np.ndarray:
    """Apply a Laplacian edge-enhancement filter as one additional technique."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return apply_filter(image, kernel, padding=padding)


def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Sharpen an image using unsharp masking as one additional technique."""
    if amount < 0:
        raise ValueError(f"amount must be non-negative, got {amount}.")
    image_f = ensure_float32(image)
    blurred = gaussian_filter(image_f, kernel_size=kernel_size, sigma=sigma)
    sharp = image_f + amount * (image_f - blurred)
    return clip_pixels(sharp, 0.0, 255.0)
