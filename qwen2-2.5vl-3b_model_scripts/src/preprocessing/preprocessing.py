"""preprocessing.py — config-driven albumentations augmentation chain."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import albumentations as A
from PIL import Image


class Preprocessor:
    """Builds a photometric augmentation pipeline from config and applies it (plus an
    optional bbox-synced resize) to a single image per call."""

    # cfg keys mapped to their albumentations class. All other keys under a step
    # (besides "enabled") are passed straight through as kwargs to the transform,
    # so this file doesn't need to hardcode/track each transform's exact param names.
    PHOTOMETRIC_OPS = {
        "brightness_contrast": A.RandomBrightnessContrast,
        "sharpen": A.Sharpen,
        "noise": A.GaussNoise,
        "blur": A.GaussianBlur,
    }

    def __init__(self, cfg: Dict[str, Any]):
        """Store augmentation config and build the photometric transform pipeline."""
        self.cfg = cfg
        self.photometric = self._build_photometric(cfg)

    def _build_photometric(self, cfg: Dict[str, Any]) -> Optional[A.Compose]:
        """Build a Compose of enabled photometric ops, or None if none are enabled."""
        ops = []
        for step_name, transform_cls in self.PHOTOMETRIC_OPS.items():
            step_cfg = cfg.get(step_name, {})
            if not step_cfg.get("enabled", False):
                continue
            kwargs = {k: v for k, v in step_cfg.items() if k != "enabled"}
            ops.append(transform_cls(**kwargs))
        return A.Compose(ops) if ops else None

    def run(
        self,
        image: Image.Image,
        bbox: Optional[list] = None,
        target_size: Optional[Tuple[int, int]] = None,
        augment: bool = True,
    ) -> Tuple[Image.Image, Optional[list]]:
        """Resize (bbox-synced if a bbox is given) and, when augment=True, apply
        photometric jitter. Always returns (image, bbox) — bbox is None if not given.

        target_size is (width, height) in already patch-aligned model-space pixels
        (e.g. from coordinate_scaling.resized_wh) — this method does not compute it.
        """
        arr = np.array(image.convert("RGB"))

        if target_size is not None:
            resized_w, resized_h = target_size
            if bbox is not None:
                resize = A.Compose(
                    [A.Resize(height=resized_h, width=resized_w)],
                    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
                )
                out = resize(image=arr, bboxes=[bbox], labels=[0])
                arr = out["image"]
                bbox = list(out["bboxes"][0]) if out["bboxes"] else None
            else:
                arr = A.Resize(height=resized_h, width=resized_w)(image=arr)["image"]

        if augment and self.photometric is not None:
            arr = self.photometric(image=arr)["image"]

        return Image.fromarray(arr), bbox

    def run_debug(
        self,
        image: Image.Image,
        bbox: Optional[list] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Image.Image]:
        """Run each enabled photometric step individually (one stochastic draw each)
        for visual inspection. Resize (if target_size given) is applied first."""
        from collections import OrderedDict

        stages = OrderedDict()
        img = image.convert("RGB")
        stages["original"] = img.copy()

        arr = np.array(img)
        if target_size is not None:
            resized_w, resized_h = target_size
            if bbox is not None:
                resize = A.Compose(
                    [A.Resize(height=resized_h, width=resized_w)],
                    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
                )
                out = resize(image=arr, bboxes=[bbox], labels=[0])
                arr = out["image"]
                # Update bbox to the rescaled coordinates so any downstream
                # use of bbox (e.g. overlaying it on the debug image) is correct.
                bbox = list(out["bboxes"][0]) if out["bboxes"] else bbox
            else:
                arr = A.Resize(height=resized_h, width=resized_w)(image=arr)["image"]
            stages["after_resize"] = Image.fromarray(arr)

        for step_name, transform_cls in self.PHOTOMETRIC_OPS.items():
            step_cfg = self.cfg.get(step_name, {})
            if not step_cfg.get("enabled", False):
                continue
            kwargs = {k: v for k, v in step_cfg.items() if k != "enabled"}
            transform = transform_cls(p=1.0, **{k: v for k, v in kwargs.items() if k != "p"})
            out_arr = transform(image=arr)["image"]
            stages[f"after_{step_name}"] = Image.fromarray(out_arr)

        return stages


if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--save_dir", default="debug_preprocessing")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pp = Preprocessor(config["augmentation"])
    stages = pp.run_debug(Image.open(args.image))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    for name, si in stages.items():
        out = Path(args.save_dir) / f"{Path(args.image).stem}_{name}.png"
        si.save(out)
        print(f"Saved: {out}")