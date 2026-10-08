"""utils/qwen_processor.py"""

from transformers import AutoProcessor


def build_processor(base_id: str, min_pixels: int, max_pixels: int) -> AutoProcessor:
    """
    Constructs a Qwen2.5-VL AutoProcessor with an explicit resolution window and
    resizing disabled.

    KNOWN TRANSFORMERS BUG: AutoProcessor.from_pretrained updates .min_pixels / .max_pixels
    but does NOT reliably update the internal .size dict that smart_resize actually reads
    (confirmed against transformers as of Oct 2025, huggingface/transformers#41955). We
    explicitly set .size here to prevent the image from being resized against the
    checkpoint's default preprocessor_config.json bounds.
    """
    proc = AutoProcessor.from_pretrained(
        base_id, min_pixels=min_pixels, max_pixels=max_pixels, do_resize=False
    )
    proc.image_processor.min_pixels = min_pixels
    proc.image_processor.max_pixels = max_pixels
    proc.image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    proc.image_processor.do_resize = False
    return proc


def debug_check_processor(proc: AutoProcessor, base_id: str, min_pixels: int, max_pixels: int) -> None:
    """Fails loudly if the .size bug is silently active, or if do_resize has drifted back on."""
    size = getattr(proc.image_processor, "size", None)
    actual_min = getattr(proc.image_processor, "min_pixels", None)
    actual_max = getattr(proc.image_processor, "max_pixels", None)
    actual_do_resize = getattr(proc.image_processor, "do_resize", None)

    if actual_min != min_pixels or actual_max != max_pixels:
        raise RuntimeError(
            f"[{base_id}] processor.image_processor min/max_pixels ({actual_min}, {actual_max}) "
            f"don't match the requested ({min_pixels}, {max_pixels})."
        )

    if isinstance(size, dict) and "shortest_edge" in size:
        if size.get("shortest_edge") != min_pixels or size.get("longest_edge") != max_pixels:
            raise RuntimeError(
                f"[{base_id}] processor.image_processor.size={size} doesn't "
                f"match min/max_pixels=({min_pixels},{max_pixels}). Override failed."
            )

    if actual_do_resize is not False:
        raise RuntimeError(
            f"[{base_id}] processor.image_processor.do_resize={actual_do_resize}, expected False. "
            f"Images are pre-resized upstream (dataset/preprocessing) for per-sample resolution augmentation;"
            f" leaving do_resize enabled lets the processor silently override whatever resolution was chosen upstream."
        ) 