"""Local compatibility helpers for the CleanDIFT import path."""


def patch_transformers_for_diffusers_compat() -> None:
    """Patch missing transformer symbols expected by diffusers 0.37.

    diffusers 0.37 imports AutoencoderRAE at package import time, which expects
    Dinov2-with-registers classes that are absent from the current
    transformers build. CleanDIFT does not use that RAE path, so aliasing the
    standard DINOv2 classes is sufficient to unblock imports without touching
    the shared environment.
    """

    try:
        import transformers
    except ImportError:
        return

    alias_pairs = (
        ("Dinov2WithRegistersConfig", "Dinov2Config"),
        ("Dinov2WithRegistersModel", "Dinov2Model"),
    )
    for missing_name, fallback_name in alias_pairs:
        if not hasattr(transformers, missing_name) and hasattr(transformers, fallback_name):
            setattr(transformers, missing_name, getattr(transformers, fallback_name))
