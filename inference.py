"""Checkpoint loading and embedding extraction for SomaAlign.

This module accepts already-preprocessed tensors and keeps dataset-specific
loading separate from the embedding API.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F

from ImageBind.models.x2touch_alignment import ImageBindModelForAlignment, ModalityType


MODALITY_TO_IMAGEBIND = {
    "rgb": ModalityType.VISION,
    "photo": ModalityType.VISION,
    "depth": ModalityType.DEPTH,
    "tactile": ModalityType.TOUCH,
    "imu": ModalityType.IMU,
    "text": ModalityType.TEXT,
}

CHECKPOINT_PREFIXES = (
    "module.",
    "model.imagebind_model.",
    "imagebind_model.",
    "model.",
)


def build_model(device: torch.device | str = "cpu") -> ImageBindModelForAlignment:
    """Construct the architecture used by SomaAlign checkpoints."""
    model = ImageBindModelForAlignment(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
        modality_list=["vision", "text", "audio", "thermal", "depth", "imu", "touch"],
    )
    return model.to(device).eval()


def normalize_checkpoint_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove common checkpoint wrapper prefixes."""
    cleaned = {}
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in CHECKPOINT_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        if key in cleaned:
            raise ValueError(f"checkpoint keys collide after prefix removal: {original_key!r}")
        cleaned[key] = value
    return cleaned


LEGACY_EXTRA_KEY_PREFIXES = (
    "stage2_loss_fn.",
    "loss_scaler.",
)


def load_checkpoint(
    model: ImageBindModelForAlignment,
    checkpoint_path: str | Path,
    weights_only: bool = True,
    allow_extra_keys: bool = False,
) -> None:
    """Load model weights, retaining an opt-in legacy compatibility interface.

    The safe default uses ``weights_only=True`` and requires an exact model-key
    match. Trusted historical checkpoints may opt into non-weights loading and
    removal of known auxiliary-state prefixes; arbitrary unexpected keys are
    still rejected.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=weights_only
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a state dictionary")
    raw = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint state_dict must be a mapping")

    cleaned = normalize_checkpoint_keys(raw)
    if allow_extra_keys:
        cleaned = {
            key: value
            for key, value in cleaned.items()
            if not key.startswith(LEGACY_EXTRA_KEY_PREFIXES)
        }
    result = model.load_state_dict(cleaned, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "checkpoint does not exactly match the SomaAlign model: "
            f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}"
        )


def prepare_tensor(data: torch.Tensor, modality: str, device: torch.device) -> torch.Tensor:
    """Convert one sample or a temporal batch to the model's expected shape."""
    if modality not in MODALITY_TO_IMAGEBIND:
        raise KeyError(f"unsupported modality: {modality!r}")
    if modality == "text":
        data = data.to(device)
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        data = data.to(device, dtype=dtype)
    if modality in ("rgb", "photo", "depth", "tactile"):
        if data.dim() == 4:              # [frames, C, H, W]
            data = data.unsqueeze(0)     # [1, frames, C, H, W]
        elif data.dim() == 3:            # [C, H, W]
            data = data.unsqueeze(0).unsqueeze(0)
    elif modality == "imu" and data.dim() == 2:
        data = data.unsqueeze(0)
    elif modality == "text" and data.dim() == 1:
        data = data.unsqueeze(0)
    return data


@torch.no_grad()
def encode(
    model: ImageBindModelForAlignment,
    inputs: Mapping[str, torch.Tensor],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Return one L2-normalized embedding tensor per named SomaAlign modality.

    Text outputs use the end-of-text representation selected by the model head.
    """
    device = torch.device(device)
    outputs = {}
    for modality, tensor in inputs.items():
        model_key = MODALITY_TO_IMAGEBIND[modality]
        prepared = prepare_tensor(tensor, modality, device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            features = model({model_key: prepared}, normalize=True)[model_key]
        if features.dim() == 3:
            features = features[:, 0]
        if features.dim() != 2:
            raise RuntimeError(f"unexpected {modality} embedding shape: {tuple(features.shape)}")
        # ImageBind heads may apply modality-specific logit scales after their
        # internal normalization.  The public SomaAlign embedding API promises
        # unit vectors, so enforce that contract at the API boundary.
        outputs[modality] = F.normalize(features.float(), dim=-1)
    return outputs
