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

import numpy as np

from lerobot.inference import FeatureSpec, ObservationSnapshot, PolicyCapabilities
from lerobot.inference.policy_runner import PolicyRunner


def test_policy_runner_builds_batched_pi05_mem_history():
    image = FeatureSpec(
        "observation.images.head", (3, 8, 8, 3), "uint8", kind="rgb_sequence", semantics="rgb"
    )
    state = FeatureSpec("observation.state", (3, 7), "float32", semantics="radians")
    action = FeatureSpec("action", (7,), "float32", semantics="radians")
    runner = PolicyRunner.__new__(PolicyRunner)
    runner.capabilities = PolicyCapabilities(
        modes=("chunk",),
        prediction_steps=50,
        execution_steps=10,
        action_interval=0.1,
        features=(image, state),
        action_feature=action,
        current_observation_only=False,
        history_frames=3,
        history_stride=5,
        history_padding=True,
    )
    runner.max_text_input = 200
    runner.robot_type = "piper"
    observation = ObservationSnapshot(
        {
            image.name: np.zeros(image.shape, dtype=np.uint8),
            state.name: np.zeros(state.shape, dtype=np.float32),
        },
        capture_time=0.0,
        task="press buttons",
        padding={
            image.name: np.array([True, True, False]),
            state.name: np.array([True, True, False]),
        },
    )
    batch = runner._batch(observation)
    assert batch[image.name].shape == (1, 3, 3, 8, 8)
    assert batch[image.name].dtype.is_floating_point
    assert batch[state.name].shape == (1, 3, 7)
    assert batch[f"{image.name}_is_pad"].tolist() == [[True, True, False]]
    assert batch[f"{state.name}_is_pad"].tolist() == [[True, True, False]]
