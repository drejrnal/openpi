"""Convert local LIBERO *_demo.hdf5 files into a LeRobot dataset for openpi fine-tuning.

These are ORIGINAL LIBERO (robomimic) HDF5 files, not the RLDS format that
examples/libero/convert_libero_data_to_lerobot.py expects -- so this is a custom reader.

Field mapping mirrors exactly what examples/libero/main.py sends at eval time, so the
training and inference distributions line up:

  image       <- obs/agentview_rgb[::-1, ::-1]      # opengl->image orientation; main.py does the same 180-deg flip
  wrist_image <- obs/eye_in_hand_rgb[::-1, ::-1]
  state (8)   <- concat(ee_pos[3], ee_ori[3], gripper_states[2])
                 # == eval's concat(eef_pos, quat2axisangle(eef_quat), gripper_qpos); ee_ori is already axis-angle
  actions (7) <- actions
  task        <- data.attrs['problem_info'].language_instruction

Corrupt/truncated HDF5 files are skipped with a warning (e.g. an interrupted download).

Usage:
  python convert_libero_object.py --data-dir /home/mxy/robot-data/libero_object
  python convert_libero_object.py --data-dir ... --repo-id libero_object   # output name under HF_LEROBOT_HOME
"""

import dataclasses
import json
import pathlib
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro


@dataclasses.dataclass
class Args:
    data_dir: str = "/home/mxy/robot-data/libero_object"
    repo_id: str = "libero_object"  # output dataset dir under HF_LEROBOT_HOME
    fps: int = 10
    push_to_hub: bool = False


def _flip(img: np.ndarray) -> np.ndarray:
    # 180-degree flip to match examples/libero/main.py (obs[::-1, ::-1]); ascontiguous for the image writer.
    return np.ascontiguousarray(img[::-1, ::-1])


def main(args: Args) -> None:
    data_dir = pathlib.Path(args.data_dir)
    files = sorted(data_dir.glob("*_demo.hdf5"))
    if not files:
        raise FileNotFoundError(f"no *_demo.hdf5 files under {data_dir}")

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        print(f"removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_episodes = n_frames = 0
    act_lo = state_lo = np.full(0, np.inf)
    tasks_seen: dict[str, int] = {}

    for f in files:
        try:
            h5 = h5py.File(f, "r")
        except OSError as e:
            print(f"  SKIP (corrupt) {f.name}: {e}")
            continue

        with h5:
            data = h5["data"]
            task = json.loads(data.attrs["problem_info"])["language_instruction"]
            demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
            print(f"  {f.name}: {len(demos)} demos | task={task!r}")

            for demo in tqdm.tqdm(demos, desc=f.stem[:30], leave=False):
                d = data[demo]
                obs = d["obs"]
                actions = np.asarray(d["actions"], dtype=np.float32)
                agent = np.asarray(obs["agentview_rgb"])
                wrist = np.asarray(obs["eye_in_hand_rgb"])
                state = np.concatenate(
                    [np.asarray(obs["ee_pos"]), np.asarray(obs["ee_ori"]), np.asarray(obs["gripper_states"])],
                    axis=1,
                ).astype(np.float32)

                assert state.shape[1] == 8, f"{demo}: state dim {state.shape[1]} != 8"
                assert actions.shape[1] == 7, f"{demo}: action dim {actions.shape[1]} != 7"
                T = len(actions)

                for t in range(T):
                    dataset.add_frame(
                        {
                            "image": _flip(agent[t]),
                            "wrist_image": _flip(wrist[t]),
                            "state": state[t],
                            "actions": actions[t],
                            "task": task,
                        }
                    )
                dataset.save_episode()

                n_episodes += 1
                n_frames += T
                tasks_seen[task] = tasks_seen.get(task, 0) + 1
                act_lo = actions.min(0) if act_lo.size == 0 else np.minimum(act_lo, actions.min(0))
                state_lo = state.min(0) if state_lo.size == 0 else np.minimum(state_lo, state.min(0))

    print("\n=== conversion summary ===")
    print(f"output:   {output_path}")
    print(f"episodes: {n_episodes}")
    print(f"frames:   {n_frames}")
    print(f"tasks:    {len(tasks_seen)}")
    for t, c in tasks_seen.items():
        print(f"   {c:3d} demos  {t!r}")
    np.set_printoptions(precision=3, suppress=True)
    print(f"state min (per-dim): {state_lo}")
    print(f"action min (per-dim): {act_lo}")

    if args.push_to_hub:
        dataset.push_to_hub(tags=["libero", "panda"], private=False, license="apache-2.0")


if __name__ == "__main__":
    main(tyro.cli(Args))
