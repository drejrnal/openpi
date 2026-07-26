"""Run joint-position inference with the pi0.5-DROID-JointPos checkpoint."""

import pathlib

import numpy as np

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

CHECKPOINT = "/home/mxy/openpi/checkpoints/pi05_franka_finetune/franka_object_full/5000"


def make_example() -> dict:
    """Build a deterministic synthetic two-camera Franka observation."""
    height, width = 224, 224
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)

    right_image = np.zeros((height, width, 3), dtype=np.uint8)
    right_image[..., 0] = x[None, :]
    right_image[..., 1] = y[:, None]
    right_image[80:144, 80:144, 0] = 255  # Synthetic red object.

    wrist_image = np.zeros((height, width, 3), dtype=np.uint8)
    wrist_image[..., 1] = x[None, :]
    wrist_image[..., 2] = y[:, None]
    wrist_image[96:128, 96:128, 0] = 255
    

    return {
        "observation/image": right_image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate(
            [
                np.array(
                    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                    dtype=np.float32,
                ),
                np.array([1.0], dtype=np.float32),  # Gripper position.
            ]
        ),
        "observation/joint_position": np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            dtype=np.float32,
        ),
        # DROID convention: one normalized gripper-position value.
        "observation/gripper_position": np.array([1.0], dtype=np.float32),
        "prompt": "pick up the red object",
    }


def main() -> None:
    config = _config.get_config("pi05_franka_finetune")

    print(f"resolving checkpoint: {CHECKPOINT}", flush=True)
    checkpoint_dir = pathlib.Path(download.maybe_download(CHECKPOINT))
    print(f"checkpoint ready at: {checkpoint_dir}", flush=True)

    print("loading franka fine-tuned policy with the Franka input adapter...", flush=True)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)

    example = make_example()
    print("joint position:", example["observation/joint_position"])
    print("gripper position:", example["observation/gripper_position"])
    print("prompt:", example["prompt"])
    print("running inference...", flush=True)

    result = policy.infer(example)
    actions = np.asarray(result["actions"])

    np.set_printoptions(precision=5, suppress=True, linewidth=180)
    print("\naction semantics: [target_joint_position_1 ... target_joint_position_7, gripper_position]")
    print("actions shape:", actions.shape)
    print("actions dtype:", actions.dtype)
    print("per-dimension min:", actions.min(axis=0))
    print("per-dimension max:", actions.max(axis=0))
    print("per-dimension mean:", actions.mean(axis=0))
    print("\nfull action chunk:")
    print(actions)


if __name__ == "__main__":
    main()
