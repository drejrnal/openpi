import numpy as np

from openpi.models import model as _model
from openpi.policies import franka_policy
from openpi.training import config as _config


def test_franka_inputs_accept_lerobot_v3_sample():
    sample = {
        "observation/image": np.ones((3, 12, 16), dtype=np.float32),
        "observation/wrist_image": np.zeros((3, 12, 16), dtype=np.float32),
        "observation/state": np.arange(8, dtype=np.float32),
        "actions": np.zeros((15, 8), dtype=np.float32),
        "prompt": "pick up the apple",
    }

    result = franka_policy.FrankaInputs(model_type=_model.ModelType.PI05)(sample)

    np.testing.assert_array_equal(result["state"], sample["observation/state"])
    assert result["image"]["base_0_rgb"].shape == (12, 16, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert np.all(result["image"]["base_0_rgb"] == 255)
    assert not np.any(result["image"]["right_wrist_0_rgb"])
    assert result["image_mask"]["right_wrist_0_rgb"] == np.False_
    assert result["actions"].shape == (15, 8)


def test_franka_outputs_keep_robot_action_dimensions():
    actions = np.arange(15 * 32).reshape(15, 32)

    result = franka_policy.FrankaOutputs()({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :8])


def test_franka_finetune_config_uses_full_model_and_wandb():
    config = _config.get_config("pi05_franka_finetune")

    assert config.model.paligemma_variant == "gemma_2b"
    assert config.model.action_expert_variant == "gemma_300m"
    assert config.ema_decay == 0.999
    assert config.fsdp_devices == 8
    assert config.wandb_enabled
    assert config.project_name == "openpi-franka"
