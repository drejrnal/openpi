"""Data transforms for a Franka Panda with an external and wrist camera."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(value: np.ndarray, *, key: str) -> np.ndarray:
    """Convert LeRobot CHW float images or runtime HWC images to HWC uint8."""
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{key} must be a 3-D RGB image, got {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"{key} must have three RGB channels, got {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)) or np.any((image < 0) | (image > 1)):
            raise ValueError(f"{key} float values must be finite and in [0, 1].")
        image = np.round(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"{key} must use uint8 or floating-point values in [0, 1], got {image.dtype}.")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Convert canonical Franka observations to π0/π0.5 model inputs.

    The canonical interface is:
      - ``observation/image``: right-side external RGB camera
      - ``observation/wrist_image``: wrist RGB camera
      - ``observation/state``: seven joint positions followed by gripper state
      - ``actions``: optional absolute joint-position targets and gripper command
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (8,) or not np.all(np.isfinite(state)):
            raise ValueError(f"observation/state must be a finite vector with shape (8,), got {state.shape}.")

        base_image = _parse_image(data["observation/image"], key="observation/image")
        wrist_image = _parse_image(data["observation/wrist_image"], key="observation/wrist_image")

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        result = {
            "state": state,
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim < 1 or actions.shape[-1] != 8 or not np.all(np.isfinite(actions)):
                raise ValueError(f"actions must be finite with final dimension 8, got {actions.shape}.")
            result["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        return result


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Return the seven joint-position targets and one gripper command."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :8])}
