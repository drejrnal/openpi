"""Local inference smoke test for the fine-tuned pi05_libero_object checkpoint.

Loads the full-FT checkpoint and runs one inference on a synthetic LIBERO example.
Like the pi05_base test, the input is random, so the ACTIONS ARE NOT MEANINGFUL
behavior -- this only verifies the checkpoint loads, the libero transforms apply,
its own norm stats un-normalize the output, and the shape is (action_horizon, 7).
"""

import numpy as np

from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

# Final training step. Use .../5000 to test the earlier checkpoint instead.
CHECKPOINT = "/home/mxy/openpi/worktree/pi_inference_test/checkpoints/pi05_libero_object/libero_object_full_ft/9999"


def main() -> None:
    config = _config.get_config("pi05_libero_object")
    ckpt = download.maybe_download(CHECKPOINT)  # local path -> returned as-is
    print(f"loading fine-tuned policy from {ckpt}", flush=True)
    policy = policy_config.create_trained_policy(config, ckpt)

    example = libero_policy.make_libero_example()
    print("running inference...", flush=True)
    result = policy.infer(example)

    actions = np.asarray(result["actions"])
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print("\nactions shape:", actions.shape, "(expect (10, 7))")
    print("actions dtype:", actions.dtype)
    print("per-dim min:", actions.min(axis=0))
    print("per-dim max:", actions.max(axis=0))
    print()
    print(actions)


if __name__ == "__main__":
    main()
