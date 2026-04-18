"""Geometric image transformations implemented from scratch."""

from __future__ import annotations

import numpy as np

from .utils import ensure_float32, validate_image_array


def _sample_at(image: np.ndarray, x: np.ndarray, y: np.ndarray, interpolation: str, fill_value: float = 0.0) -> np.ndarray:
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError(f"interpolation must be 'nearest' or 'bilinear', got {interpolation!r}.")
    image_f = ensure_float32(image)
    h, w = image_f.shape[:2]
    channels = 1 if image_f.ndim == 2 else image_f.shape[2]
    if interpolation == "nearest":
        xi = np.rint(x).astype(int)
        yi = np.rint(y).astype(int)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out_shape = x.shape if channels == 1 else x.shape + (channels,)
        out = np.full(out_shape, fill_value, dtype=np.float32)
        out[valid] = image_f[yi[valid], xi[valid]]
        return out
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    Ia = image_f[y0c, x0c]
    Ib = image_f[y1c, x0c]
    Ic = image_f[y0c, x1c]
    Id = image_f[y1c, x1c]
    if channels == 1:
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id
        out = np.where(valid, out, fill_value)
    else:
        out = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id
        out = np.where(valid[..., None], out, fill_value)
    return out.astype(np.float32)


def resize(image: np.ndarray, new_size: tuple[int, int], interpolation: str = "nearest") -> np.ndarray:
    """Resize an image to ``(new_height, new_width)`` using nearest or bilinear interpolation."""
    validate_image_array(image)
    new_h, new_w = new_size
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"new_size values must be positive, got {new_size}.")
    src = ensure_float32(image)
    old_h, old_w = src.shape[:2]
    y = np.linspace(0, old_h - 1, new_h, dtype=np.float32)
    x = np.linspace(0, old_w - 1, new_w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return _sample_at(src, xx, yy, interpolation=interpolation)


def translate(image: np.ndarray, tx: float, ty: float, interpolation: str = "nearest", fill_value: float = 0.0) -> np.ndarray:
    """Translate an image by ``tx`` pixels horizontally and ``ty`` pixels vertically."""
    validate_image_array(image)
    src = ensure_float32(image)
    h, w = src.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src_x = xx - tx
    src_y = yy - ty
    return _sample_at(src, src_x, src_y, interpolation=interpolation, fill_value=fill_value)


def rotate(image: np.ndarray, angle_degrees: float, interpolation: str = "nearest", fill_value: float = 0.0) -> np.ndarray:
    """Rotate an image about its center by a specified angle in degrees."""
    validate_image_array(image)
    src = ensure_float32(image)
    h, w = src.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    theta = np.deg2rad(angle_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_shift = xx - cx
    y_shift = yy - cy
    src_x = cos_t * x_shift + sin_t * y_shift + cx
    src_y = -sin_t * x_shift + cos_t * y_shift + cy
    return _sample_at(src, src_x, src_y, interpolation=interpolation, fill_value=fill_value)
