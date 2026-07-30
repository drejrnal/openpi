from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

dataset = LeRobotDataset(
    "lerobot/droid_1.0.1",
    root="/data/mxy/cache/lerobot/droid_1.0.1",
)
frame = dataset[0]

print("Dataset length:", len(dataset))
print("FPS:", dataset.meta.fps)

for key in sorted(frame):
    value = frame[key]
    shape = getattr(value, "shape", None)
    print(f"{key:50s} {shape}")

state = frame["observation.state"]
action = frame["action"]
joint_target = frame["action.joint_position"]
gripper_target = frame["action.gripper_position"]

print("state:", state)
print("action:", action)
print("joint target:", joint_target)
print("gripper target:", gripper_target)

assert state.shape[-1] == 8
assert action.shape[-1] == 8
torch.testing.assert_close(action[:7], joint_target)
torch.testing.assert_close(action[7], gripper_target)

print("Verified: action = 7 joint-position targets + gripper target")
