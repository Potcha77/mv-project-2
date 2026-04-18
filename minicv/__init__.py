"""minicv: a small educational image-processing library built from scratch."""

from .core import clip_pixels, convolve2d, normalize_image, pad_image
from .drawing import create_canvas, draw_line, draw_point, draw_polygon, draw_rectangle, put_text
from .features import gradient_statistics, grayscale_histogram_descriptor, hog_descriptor, intensity_statistics
from .filtering import adaptive_threshold, apply_filter, bit_plane_slicing, gaussian_filter, gaussian_kernel, global_threshold, histogram, histogram_equalization, laplacian_filter, mean_filter, median_filter, otsu_threshold, sobel_gradients, unsharp_mask
from .io import grayscale_to_rgb, read_image, rgb_to_grayscale, save_image
from .transforms import resize, rotate, translate

__all__ = [
    "read_image","save_image","rgb_to_grayscale","grayscale_to_rgb",
    "normalize_image","clip_pixels","pad_image","convolve2d",
    "apply_filter","mean_filter","gaussian_kernel","gaussian_filter","median_filter",
    "global_threshold","otsu_threshold","adaptive_threshold","sobel_gradients",
    "bit_plane_slicing","histogram","histogram_equalization","laplacian_filter","unsharp_mask",
    "resize","rotate","translate",
    "intensity_statistics","grayscale_histogram_descriptor","gradient_statistics","hog_descriptor",
    "create_canvas","draw_point","draw_line","draw_rectangle","draw_polygon","put_text",
]
