"""utils/coordinate_scaling.py : for scaling coordinates between raw image space 
and model input space, with smart resizing to respect min/max pixel constraints."""

import math
from typing import List, Tuple, Optional

FACTOR = 28  # Qwen2.5-VL patch factor


def _round_by_factor(x: float, factor: int) -> int:
    """Round a dimension to the nearest patch-factor multiple."""
    return round(x / factor) * factor


def _ceil_by_factor(x: float, factor: int) -> int:
    """Ceil a dimension to a patch-factor multiple."""
    return math.ceil(x / factor) * factor


def _floor_by_factor(x: float, factor: int) -> int:
    """Floor a dimension to a patch-factor multiple."""
    return math.floor(x / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    """Compute resized dimensions that satisfy min/max pixel budget and patch alignment."""
    if min_pixels is None or max_pixels is None:
        raise ValueError("min_pixels/max_pixels are required parameters.")

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def resized_wh(orig_width: int, orig_height: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """Return adjusted width and height for the resized image."""
    h_bar, w_bar = smart_resize(orig_height, orig_width, FACTOR, min_pixels, max_pixels)
    return w_bar, h_bar


def scale_gt_coords_to_model_space(
    box: List[float],
    orig_width: int,
    orig_height: int,
    min_pixels: int,
    max_pixels: int,
) -> List[int]:
    """Map raw image bbox coordinates into resized model input space."""
    x1, y1, x2, y2 = box
    resized_w, resized_h = resized_wh(orig_width, orig_height, min_pixels, max_pixels)

    mx1 = max(0, min(round((x1 / orig_width) * resized_w), resized_w))
    my1 = max(0, min(round((y1 / orig_height) * resized_h), resized_h))
    mx2 = max(0, min(round((x2 / orig_width) * resized_w), resized_w))
    my2 = max(0, min(round((y2 / orig_height) * resized_h), resized_h))
    
    return [mx1, my1, mx2, my2]


def scale_predicted_coords_to_raw_image(
    pred_box: List[float],
    orig_width: int,
    orig_height: int,
    min_pixels: int,
    max_pixels: int,
) -> List[int]:
    """Map predicted model-space coords back to raw image pixels."""
    px1, py1, px2, py2 = pred_box
    resized_w, resized_h = resized_wh(orig_width, orig_height, min_pixels, max_pixels)

    rx1 = max(0, min(round((px1 / resized_w) * orig_width), orig_width))
    ry1 = max(0, min(round((py1 / resized_h) * orig_height), orig_height))
    rx2 = max(0, min(round((px2 / resized_w) * orig_width), orig_width))
    ry2 = max(0, min(round((py2 / resized_h) * orig_height), orig_height))
    
    return [rx1, ry1, rx2, ry2]


def pad_model_space_bbox(
    box: List[int], resized_w: int, resized_h: int, pad_pct: float
) -> List[int]:
    """Pad a model-space bounding box by a fraction of the resized image size.
    Always returns clean ints, even if `box` came in as floats (e.g. from an
    albumentations bbox resize)."""
    if pad_pct <= 0:
        return [int(round(v)) for v in box]

    pad_x = round(resized_w * pad_pct)
    pad_y = round(resized_h * pad_pct)
    x1, y1, x2, y2 = box

    return [
        int(max(0, round(x1) - pad_x)),
        int(max(0, round(y1) - pad_y)),
        int(min(resized_w, round(x2) + pad_x)),
        int(min(resized_h, round(y2) + pad_y)),
    ]