"""Image I/O and color conversion routines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_float32, validate_image_array

RGB_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


def read_image(path: str | Path, mode: str = "unchanged") -> np.ndarray:
    """Read an image from disk into a NumPy array.

    Parameters
    ----------
    path:
        Path to an image file supported by Matplotlib backends.
    mode:
        One of ``"unchanged"``, ``"rgb"``, or ``"grayscale"``.

    Returns
    -------
    numpy.ndarray
        Image array with dtype ``float32``. RGB images use shape ``(H, W, 3)``.
        Grayscale images use shape ``(H, W)``. Intensity range is standardized to ``[0, 255]``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image file was not found: {path}")
    if mode not in {"unchanged", "rgb", "grayscale"}:
        raise ValueError(f"mode must be 'unchanged', 'rgb', or 'grayscale', got {mode!r}.")

    image = mpimg.imread(path)
    image = ensure_float32(np.asarray(image))

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    if image.max(initial=0.0) <= 1.0:
        image = image * 255.0

    if mode == "grayscale":
        if image.ndim == 2:
            return image
        return rgb_to_grayscale(image)
    if mode == "rgb":
        if image.ndim == 2:
            return grayscale_to_rgb(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected grayscale or RGB image after loading, got shape {image.shape}.")
        return image

    if image.ndim not in (2, 3):
        raise ValueError(f"unsupported loaded image shape {image.shape}; expected 2D or 3D image array.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"RGB image must have 3 channels, got shape {image.shape}.")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an image array to disk.

    Parameters
    ----------
    path:
        Output file path, such as ``.png`` or ``.jpg``.
    image:
        Grayscale or RGB image.
    """
    validate_image_array(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_f32 = ensure_float32(image)
    clipped = np.clip(image_f32, 0.0, 255.0)
    out = clipped.astype(np.uint8)
    if out.ndim == 2:
        plt.imsave(path, out, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imsave(path, out)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using luminance weights."""
    validate_image_array(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb_to_grayscale expects shape (H, W, 3), got {image.shape}.")
    return np.tensordot(ensure_float32(image), RGB_WEIGHTS, axes=([-1], [0])).astype(np.float32)


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to a 3-channel RGB image."""
    validate_image_array(image, allow_rgb=False)
    if image.ndim != 2:
        raise ValueError(f"grayscale_to_rgb expects a 2D image, got {image.shape}.")
    gray = ensure_float32(image)
    return np.stack([gray, gray, gray], axis=-1)
