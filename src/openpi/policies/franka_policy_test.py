import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import franka_policy
from openpi.training import config as _config
from openpi.training import weight_loaders


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


def test_franka_inputs_select_joint_velocity_actions():
    joint_position = np.ones((15, 8), dtype=np.float32)
    joint_position[:, -1] = 0.75
    joint_velocity = np.full((15, 7), 0.25, dtype=np.float32)
    sample = {
        "observation/image": np.zeros((12, 16, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((12, 16, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "actions": {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "gripper_position": joint_position,
        },
    }

    result = franka_policy.FrankaInputs(
        model_type=_model.ModelType.PI05,
        action_space=franka_policy.FrankaActionSpace.JOINT_VELOCITY,
    )(sample)

    np.testing.assert_array_equal(result["actions"][..., :7], joint_velocity)
    np.testing.assert_array_equal(result["actions"][..., 7], joint_position[..., 7])


def test_franka_finetune_config_uses_pi05_droid_full_model_and_wandb():
    config = _config.get_config("pi05_franka_finetune")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert config.model.paligemma_variant == "gemma_2b"
    assert config.model.action_expert_variant == "gemma_300m"
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert config.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_droid/params"
    assert isinstance(config.data, _config.LeRobotFrankaDataConfig)
    assert config.data.action_space == franka_policy.FrankaActionSpace.JOINT_VELOCITY
    assert config.data.assets.assets_dir == "gs://openpi-assets/checkpoints/pi05_droid/assets"
    assert config.data.assets.asset_id == "droid"
    assert data_config.lerobot_root == "/data/mxy/lerobot/franka_object"
    assert data_config.action_sequence_keys == ("data.actions.joint_velocity", "action")
    assert not any(isinstance(transform, transforms.DeltaActions) for transform in data_config.data_transforms.inputs)
    assert not any(isinstance(transform, transforms.AbsoluteActions) for transform in data_config.data_transforms.outputs)

    velocity_actions = np.full((15, 7), 0.25, dtype=np.float32)
    position_actions = np.ones((15, 8), dtype=np.float32)
    repacked = data_config.repack_transforms.inputs[0](
        {
            "observation.images.exterior_image": np.zeros((12, 16, 3), dtype=np.uint8),
            "observation.images.wrist_image": np.zeros((12, 16, 3), dtype=np.uint8),
            "observation.state": np.zeros(8, dtype=np.float32),
            "data.actions.joint_velocity": velocity_actions,
            "action": position_actions,
            "prompt": "pick up the apple",
        }
    )
    np.testing.assert_array_equal(repacked["actions"]["joint_velocity"], velocity_actions)
    np.testing.assert_array_equal(repacked["actions"]["gripper_position"], position_actions)

    assert config.ema_decay == 0.999
    assert config.fsdp_devices == 8
    assert config.wandb_enabled
    assert config.project_name == "openpi-franka"


def test_franka_joint_position_action_space_adds_delta_transforms():
    config = _config.get_config("pi05_franka_finetune")
    assert isinstance(config.data, _config.LeRobotFrankaDataConfig)
    position_factory = dataclasses.replace(
        config.data,
        action_space=franka_policy.FrankaActionSpace.JOINT_POSITION,
        assets=_config.AssetsConfig(),
    )

    data_config = position_factory.create(config.assets_dirs, config.model)

    assert data_config.action_sequence_keys == ("action",)
    assert any(isinstance(transform, transforms.DeltaActions) for transform in data_config.data_transforms.inputs)
    assert any(isinstance(transform, transforms.AbsoluteActions) for transform in data_config.data_transforms.outputs)
