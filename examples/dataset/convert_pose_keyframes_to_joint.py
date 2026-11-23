#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replace pose keyframe targets with their exact source-trajectory joint actions.

The pose keyframe dataset remains the source of images, observations, timestamps,
episode splits, and keyframe selection. ``manifest.json`` maps every keyframe back
to an episode/frame in the full trajectory. This script uses that mapping to copy
the corresponding joint-space ``action`` without running inverse kinematics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ACTION_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
ACTION_ENCODING = "joint_radians_gripper_0to1_abs"
TASK = "predict ordered press-button commanded joint keyframes v1"
TASK_NAME = "press_three_buttons_joint_keyframes_v1"
ROBOT_TYPE = "piper_d405_keyframes_v1_joint_gripper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-keyframes", type=Path, required=True)
    parser.add_argument("--full-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def fixed_list_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    values = column.combine_chunks()
    return np.asarray(values.values, dtype=np.float32).reshape(len(values), values.type.list_size)


def numpy_to_fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def read_all_parquet(directory: Path) -> pa.Table:
    paths = sorted(directory.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {directory}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing column {name!r}")
    return table.set_column(index, name, values)


def convert(pose_root: Path, full_root: Path, output: Path) -> dict[str, Any]:
    pose_root = pose_root.resolve()
    full_root = full_root.resolve()
    output = output.resolve()
    for root, label in ((pose_root, "pose keyframes"), (full_root, "full dataset")):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {label} directory: {root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    manifest = load_json(pose_root / "manifest.json")
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise ValueError("The pose keyframe manifest contains no episodes")

    full_table = read_all_parquet(full_root / "data")
    full_episode = np.asarray(full_table["episode_index"], dtype=np.int64)
    full_frame = np.asarray(full_table["frame_index"], dtype=np.int64)
    full_action = fixed_list_to_numpy(full_table["action"])
    full_state = fixed_list_to_numpy(full_table["observation.state"])
    source_rows = {
        (int(episode_index), int(frame_index)): row
        for row, (episode_index, frame_index) in enumerate(zip(full_episode, full_frame, strict=True))
    }
    if len(source_rows) != len(full_table):
        raise ValueError("Full dataset contains duplicate (episode_index, frame_index) pairs")

    staging = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    all_actions: list[np.ndarray] = []
    episode_action_stats: list[dict[str, list[float] | list[int]]] = []
    max_state_error = 0.0
    try:
        shutil.copytree(pose_root, staging)
        for expected_episode_index, episode in enumerate(episodes):
            episode_index = int(episode["episode_index"])
            if episode_index != expected_episode_index:
                raise ValueError(
                    f"Manifest episodes must be contiguous: expected {expected_episode_index}, got {episode_index}"
                )
            source_episode_index = int(episode["source_episode_index"])
            source_frame_indices = [int(index) for index in episode["source_frame_indices"]]
            data_path = (
                staging
                / "data"
                / f"chunk-{episode_index // 1000:03d}"
                / f"file-{episode_index % 1000:03d}.parquet"
            )
            table = pq.read_table(data_path)
            if len(source_frame_indices) != len(table):
                raise ValueError(
                    f"Episode {episode_index}: {len(source_frame_indices)} mapped frames for {len(table)} rows"
                )
            rows: list[int] = []
            for source_frame_index in source_frame_indices:
                key = (source_episode_index, source_frame_index)
                if key not in source_rows:
                    raise KeyError(f"Episode {episode_index}: source row {key} does not exist")
                rows.append(source_rows[key])

            actions = full_action[rows]
            states = full_state[rows]
            pose_states = fixed_list_to_numpy(table["observation.state"])
            state_error = float(np.max(np.abs(states - pose_states)))
            max_state_error = max(max_state_error, state_error)
            if state_error > 1e-6:
                raise ValueError(
                    f"Episode {episode_index}: selected source states differ from pose keyframes by {state_error}"
                )

            output_episode = np.asarray(table["episode_index"], dtype=np.int64)
            output_frame = np.asarray(table["frame_index"], dtype=np.int64)
            if not np.all(output_episode == episode_index) or not np.array_equal(
                output_frame, np.arange(len(table), dtype=np.int64)
            ):
                raise ValueError(f"Episode {episode_index}: keyframe data indices are inconsistent")

            table = replace_column(table, "action", numpy_to_fixed_list(actions))
            pq.write_table(table, data_path, compression="snappy")
            all_actions.append(actions)
            episode_action_stats.append(stats(actions))

        episode_meta_path = staging / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episode_meta = pq.read_table(episode_meta_path)
        if len(episode_meta) != len(episodes):
            raise ValueError(f"Episode metadata has {len(episode_meta)} rows, expected {len(episodes)}")
        episode_meta = replace_column(
            episode_meta,
            "tasks",
            pa.array([[TASK] for _ in episodes], type=episode_meta.schema.field("tasks").type),
        )
        for stat_name in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            column_name = f"stats/action/{stat_name}"
            column_type = episode_meta.schema.field(column_name).type
            episode_meta = replace_column(
                episode_meta,
                column_name,
                pa.array(
                    [episode_stats[stat_name] for episode_stats in episode_action_stats], type=column_type
                ),
            )
        pq.write_table(episode_meta, episode_meta_path, compression="snappy")

        tasks_path = staging / "meta" / "tasks.parquet"
        tasks_table = pq.read_table(tasks_path)
        tasks_table = replace_column(tasks_table, "task", pa.array([TASK], type=pa.string()))
        pq.write_table(tasks_table, tasks_path, compression="snappy")

        combined_actions = np.concatenate(all_actions, axis=0)
        global_stats = load_json(staging / "meta" / "stats.json")
        global_stats["action"] = stats(combined_actions)
        write_json(staging / "meta" / "stats.json", global_stats)

        info = load_json(staging / "meta" / "info.json")
        full_info = load_json(full_root / "meta" / "info.json")
        info["robot_type"] = ROBOT_TYPE
        info["features"]["action"]["names"] = ACTION_NAMES
        if "piper_gripper_encoding" in full_info:
            info["piper_gripper_encoding"] = full_info["piper_gripper_encoding"]
        if "piper_camera_mapping" in full_info:
            info["piper_camera_mapping"] = full_info["piper_camera_mapping"]
        write_json(staging / "meta" / "info.json", info)

        keyframes = load_json(staging / "meta" / "keyframes.json")
        keyframes["joint_action_schema_version"] = 1
        keyframes["action_encoding"] = ACTION_ENCODING
        keyframes["action_names"] = ACTION_NAMES
        keyframes["source_pose_keyframe_dataset"] = str(pose_root)
        keyframes["source_datasets"] = [str(full_root)]
        keyframes["derived_from"] = (
            "exact joint actions from the full trajectory at manifest source_frame_indices; no IK applied"
        )
        keyframes["rotation_convention"] = None
        keyframes["gripper_encoding"] = "binary commanded opening, 0 closed and 1 fully open"
        write_json(staging / "meta" / "keyframes.json", keyframes)

        manifest["repo_id"] = TASK_NAME
        manifest["robot_type"] = ROBOT_TYPE
        manifest["source_pose_keyframe_dataset"] = str(pose_root)
        for episode in manifest["episodes"]:
            episode["task"] = TASK
            episode["task_name"] = TASK_NAME
            episode["action_encoding"] = ACTION_ENCODING
            episode["source_pose_keyframe_dataset"] = str(pose_root)
        write_json(staging / "manifest.json", manifest)

        parquet_count = len(list((staging / "data").rglob("*.parquet")))
        video_count = len(list((staging / "videos").rglob("*.mp4")))
        if parquet_count != len(episodes):
            raise ValueError(f"Output has {parquet_count} data files, expected {len(episodes)}")
        if video_count != len(episodes) * len(manifest["camera_names"]):
            raise ValueError(
                f"Output has {video_count} videos, expected {len(episodes) * len(manifest['camera_names'])}"
            )

        staging.rename(output)
        return {
            "output": str(output),
            "episodes": len(episodes),
            "frames": len(combined_actions),
            "frames_per_episode": len(combined_actions) // len(episodes),
            "videos": video_count,
            "max_state_alignment_error": max_state_error,
            "action_min": combined_actions.min(axis=0).tolist(),
            "action_max": combined_actions.max(axis=0).tolist(),
            "initial_action_abs_max": float(
                np.max(np.abs(combined_actions.reshape(len(episodes), -1, 7)[:, 0, :6]))
            ),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    args = parse_args()
    result = convert(args.pose_keyframes, args.full_dataset, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
