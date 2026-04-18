"""Utility helpers for the minicv package."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

ArrayLike = np.ndarray
ALLOWED_IMAGE_DIMENSIONS = {2, 3}

_BITMAP_FONT = {
    " ": ["00000","00000","00000","00000","00000","00000","00000"],
    "A": ["01110","10001","10001","11111","10001","10001","10001"],
    "B": ["11110","10001","10001","11110","10001","10001","11110"],
    "C": ["01111","10000","10000","10000","10000","10000","01111"],
    "D": ["11110","10001","10001","10001","10001","10001","11110"],
    "E": ["11111","10000","10000","11110","10000","10000","11111"],
    "F": ["11111","10000","10000","11110","10000","10000","10000"],
    "G": ["01110","10001","10000","10111","10001","10001","01110"],
    "H": ["10001","10001","10001","11111","10001","10001","10001"],
    "I": ["11111","00100","00100","00100","00100","00100","11111"],
    "J": ["11111","00010","00010","00010","10010","10010","01100"],
    "K": ["10001","10010","10100","11000","10100","10010","10001"],
    "L": ["10000","10000","10000","10000","10000","10000","11111"],
    "M": ["10001","11011","10101","10101","10001","10001","10001"],
    "N": ["10001","11001","10101","10011","10001","10001","10001"],
    "O": ["01110","10001","10001","10001","10001","10001","01110"],
    "P": ["11110","10001","10001","11110","10000","10000","10000"],
    "Q": ["01110","10001","10001","10001","10101","10010","01101"],
    "R": ["11110","10001","10001","11110","10100","10010","10001"],
    "S": ["01111","10000","10000","01110","00001","00001","11110"],
    "T": ["11111","00100","00100","00100","00100","00100","00100"],
    "U": ["10001","10001","10001","10001","10001","10001","01110"],
    "V": ["10001","10001","10001","10001","10001","01010","00100"],
    "W": ["10001","10001","10001","10101","10101","10101","01010"],
    "X": ["10001","10001","01010","00100","01010","10001","10001"],
    "Y": ["10001","10001","01010","00100","00100","00100","00100"],
    "Z": ["11111","00001","00010","00100","01000","10000","11111"],
    "0": ["01110","10001","10011","10101","11001","10001","01110"],
    "1": ["00100","01100","00100","00100","00100","00100","01110"],
    "2": ["01110","10001","00001","00010","00100","01000","11111"],
    "3": ["11110","00001","00001","01110","00001","00001","11110"],
    "4": ["00010","00110","01010","10010","11111","00010","00010"],
    "5": ["11111","10000","10000","11110","00001","00001","11110"],
    "6": ["01110","10000","10000","11110","10001","10001","01110"],
    "7": ["11111","00001","00010","00100","01000","01000","01000"],
    "8": ["01110","10001","10001","01110","10001","10001","01110"],
    "9": ["01110","10001","10001","01111","00001","00001","01110"],
    ":": ["00000","00100","00100","00000","00100","00100","00000"],
    ".": ["00000","00000","00000","00000","00000","00110","00110"],
    "-": ["00000","00000","00000","11111","00000","00000","00000"],
    "_": ["00000","00000","00000","00000","00000","00000","11111"],
    "/": ["00001","00010","00100","01000","10000","00000","00000"],
}


def validate_image_array(image: ArrayLike, allow_rgb: bool = True) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a NumPy array, got {type(image).__name__}.")
    if image.ndim not in ALLOWED_IMAGE_DIMENSIONS:
        raise ValueError(f"image must have 2 dimensions (grayscale) or 3 dimensions (RGB), got shape {image.shape}.")
    if image.ndim == 3:
        if not allow_rgb:
            raise ValueError("RGB images are not allowed in this function; expected a 2D grayscale image.")
        if image.shape[2] != 3:
            raise ValueError(f"RGB image must have exactly 3 channels in the last dimension, got shape {image.shape}.")
    if image.size == 0:
        raise ValueError("image must be non-empty.")


def validate_grayscale_image(image: ArrayLike) -> None:
    validate_image_array(image, allow_rgb=False)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {image.shape}.")


def validate_numeric_array(array: ArrayLike, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(array).__name__}.")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values, got dtype {array.dtype}.")


def ensure_float32(array: ArrayLike) -> np.ndarray:
    validate_numeric_array(array, "array")
    return array.astype(np.float32, copy=False)


def normalize_color_value(color: int | float | Sequence[int | float], channels: int) -> np.ndarray:
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}.")
    if isinstance(color, np.ndarray):
        color = color.tolist() if color.ndim > 0 else float(color)
    if channels == 1:
        if isinstance(color, Sequence) and not isinstance(color, (str, bytes)):
            if len(color) != 1:
                raise ValueError("grayscale drawing expects a scalar color or a sequence of length 1.")
            return np.array(color[0], dtype=np.float32)
        return np.array(color, dtype=np.float32)
    if not isinstance(color, Sequence) or isinstance(color, (str, bytes)):
        raise ValueError("RGB drawing expects a color sequence with exactly 3 values.")
    if len(color) != 3:
        raise ValueError(f"RGB drawing expects 3 color values, got {len(color)}.")
    return np.asarray(color, dtype=np.float32)


def parse_padding(pad_width: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(pad_width, int):
        if pad_width < 0:
            raise ValueError("pad_width must be non-negative.")
        return pad_width, pad_width
    if isinstance(pad_width, Sequence):
        values = tuple(int(v) for v in pad_width)
        if len(values) != 2:
            raise ValueError("pad_width sequence must contain exactly 2 integers: (pad_y, pad_x).")
        if values[0] < 0 or values[1] < 0:
            raise ValueError("pad_width values must be non-negative.")
        return values
    raise TypeError(f"pad_width must be an int or a 2-value sequence, got {type(pad_width).__name__}.")


def validate_kernel(kernel: ArrayLike) -> np.ndarray:
    validate_numeric_array(kernel, "kernel")
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D, got shape {kernel.shape}.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(f"kernel dimensions must be odd, got shape {kernel.shape}.")
    return kernel.astype(np.float32, copy=False)


def grayscale_view(image: ArrayLike) -> np.ndarray:
    validate_image_array(image)
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    rgb = image.astype(np.float32, copy=False)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def get_bitmap(character: str) -> list[str]:
    return _BITMAP_FONT.get(character.upper(), _BITMAP_FONT[" "])


def iter_polygon_edges(points: np.ndarray) -> Iterable[tuple[tuple[int, int], tuple[int, int]]]:
    for idx in range(len(points)):
        start = tuple(points[idx])
        end = tuple(points[(idx + 1) % len(points)])
        yield start, end
