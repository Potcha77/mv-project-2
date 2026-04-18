"""Canvas operations and drawing primitives on NumPy arrays."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .utils import get_bitmap, iter_polygon_edges, normalize_color_value, validate_image_array


def create_canvas(height: int, width: int, channels: int = 3, color: int | float | Sequence[int | float] = 0) -> np.ndarray:
    """Create a blank grayscale or RGB canvas."""
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}.")
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}.")
    color_arr = normalize_color_value(color, channels)
    if channels == 1:
        return np.full((height, width), float(color_arr), dtype=np.float32)
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    canvas[...] = color_arr
    return canvas


def _paint_disk(image: np.ndarray, x: int, y: int, color: np.ndarray, thickness: int) -> None:
    radius = max(0, thickness // 2)
    h, w = image.shape[:2]
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            if 0 <= yy < h and 0 <= xx < w:
                if image.ndim == 2:
                    image[yy, xx] = float(color)
                else:
                    image[yy, xx] = color


def draw_point(image: np.ndarray, x: int, y: int, color: int | float | Sequence[int | float], thickness: int = 1) -> np.ndarray:
    """Draw a point on a grayscale or RGB image with boundary clipping."""
    validate_image_array(image)
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}.")
    out = image.astype(np.float32, copy=True)
    color_arr = normalize_color_value(color, 1 if out.ndim == 2 else 3)
    _paint_disk(out, int(x), int(y), color_arr, thickness)
    return out


def draw_line(image: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: int | float | Sequence[int | float], thickness: int = 1) -> np.ndarray:
    """Draw a line using Bresenham's algorithm."""
    validate_image_array(image)
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}.")
    out = image.astype(np.float32, copy=True)
    color_arr = normalize_color_value(color, 1 if out.ndim == 2 else 3)
    x0, y0 = map(int, pt1)
    x1, y1 = map(int, pt2)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _paint_disk(out, x0, y0, color_arr, thickness)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return out


def draw_rectangle(image: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], color: int | float | Sequence[int | float], thickness: int = 1, filled: bool = False) -> np.ndarray:
    """Draw a rectangle outline or a filled rectangle."""
    validate_image_array(image)
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}.")
    x0, y0 = map(int, top_left)
    x1, y1 = map(int, bottom_right)
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))
    out = image.astype(np.float32, copy=True)
    color_arr = normalize_color_value(color, 1 if out.ndim == 2 else 3)
    h, w = out.shape[:2]
    xmin = max(0, xmin)
    xmax = min(w - 1, xmax)
    ymin = max(0, ymin)
    ymax = min(h - 1, ymax)
    if filled:
        if out.ndim == 2:
            out[ymin:ymax+1, xmin:xmax+1] = float(color_arr)
        else:
            out[ymin:ymax+1, xmin:xmax+1] = color_arr
        return out
    out = draw_line(out, (xmin, ymin), (xmax, ymin), color_arr, thickness)
    out = draw_line(out, (xmax, ymin), (xmax, ymax), color_arr, thickness)
    out = draw_line(out, (xmax, ymax), (xmin, ymax), color_arr, thickness)
    out = draw_line(out, (xmin, ymax), (xmin, ymin), color_arr, thickness)
    return out


def draw_polygon(image: np.ndarray, points: Sequence[tuple[int, int]], color: int | float | Sequence[int | float], thickness: int = 1, filled: bool = False) -> np.ndarray:
    """Draw a polygon outline, with optional scanline filling."""
    validate_image_array(image)
    if len(points) < 3:
        raise ValueError("polygon requires at least 3 points.")
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}.")
    pts = np.asarray(points, dtype=int)
    out = image.astype(np.float32, copy=True)
    color_arr = normalize_color_value(color, 1 if out.ndim == 2 else 3)
    for start, end in iter_polygon_edges(pts):
        out = draw_line(out, start, end, color_arr, thickness)
    if not filled:
        return out
    h, w = out.shape[:2]
    ymin = max(0, int(np.min(pts[:, 1])))
    ymax = min(h - 1, int(np.max(pts[:, 1])))
    for y in range(ymin, ymax + 1):
        intersections: list[int] = []
        for (x0, y0), (x1, y1) in iter_polygon_edges(pts):
            if y0 == y1:
                continue
            if (y >= min(y0, y1)) and (y < max(y0, y1)):
                x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                intersections.append(int(round(x)))
        intersections.sort()
        for i in range(0, len(intersections) - 1, 2):
            x_start = max(0, intersections[i])
            x_end = min(w - 1, intersections[i + 1])
            if out.ndim == 2:
                out[y, x_start:x_end+1] = float(color_arr)
            else:
                out[y, x_start:x_end+1] = color_arr
    return out


def put_text(image: np.ndarray, text: str, position: tuple[int, int], font_scale: int = 1, color: int | float | Sequence[int | float] = 255, spacing: int = 1) -> np.ndarray:
    """Draw simple bitmap text on an image."""
    validate_image_array(image)
    if font_scale <= 0:
        raise ValueError(f"font_scale must be positive, got {font_scale}.")
    if spacing < 0:
        raise ValueError(f"spacing must be non-negative, got {spacing}.")
    out = image.astype(np.float32, copy=True)
    color_arr = normalize_color_value(color, 1 if out.ndim == 2 else 3)
    start_x, start_y = map(int, position)
    cursor_x = start_x
    for character in text:
        bitmap = get_bitmap(character)
        for row_idx, row in enumerate(bitmap):
            for col_idx, value in enumerate(row):
                if value != "1":
                    continue
                for yy in range(font_scale):
                    for xx in range(font_scale):
                        x = cursor_x + col_idx * font_scale + xx
                        y = start_y + row_idx * font_scale + yy
                        if 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
                            if out.ndim == 2:
                                out[y, x] = float(color_arr)
                            else:
                                out[y, x] = color_arr
        cursor_x += 5 * font_scale + spacing
    return out
