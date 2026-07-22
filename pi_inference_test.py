"""Smoke test: load the pi0.5 base checkpoint and run one inference on a synthetic example."""

import numpy as np

from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base"


def main() -> None:
    config = _config.get_config("pi05_droid")

    print(f"downloading {CHECKPOINT} (12.4 GB on first run)...", flush=True)
    ckpt = download.maybe_download(CHECKPOINT)
    print(f"checkpoint ready at {ckpt}", flush=True)

    print("loading policy (pi05_base params)...", flush=True)
    policy = policy_config.create_trained_policy(config, ckpt)

    example = droid_policy.make_droid_example()
    print("running inference...", flush=True)
    result = policy.infer(example)

    actions = np.asarray(result["actions"])
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print("\nactions shape:", actions.shape)
    print("actions dtype:", actions.dtype)
    print("per-dim min:", actions.min(axis=0))
    print("per-dim max:", actions.max(axis=0))
    print()
    print(actions)


if __name__ == "__main__":
    main()
