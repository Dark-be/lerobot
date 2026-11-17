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

from dataclasses import dataclass

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("akt")
@dataclass
class AKTConfig(ACTConfig):
    """Action Keyframe Transformer configuration.

    AKT is the state-free, open-loop keyframe specialization of ACT. An image at
    frame ``t`` predicts the complete action sequence beginning at frame ``t+1``.
    This prevents the observation-aligned initial action from becoming the first
    action executed on the robot.
    """

    action_delta_start_index: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_delta_start_index != 1:
            raise ValueError(
                "AKT always predicts future keyframes starting at t+1; "
                f"action_delta_start_index must be 1, got {self.action_delta_start_index}."
            )
        if self.n_action_steps != self.chunk_size:
            raise ValueError(
                "AKT executes the complete open-loop keyframe sequence; "
                f"n_action_steps ({self.n_action_steps}) must equal chunk_size ({self.chunk_size})."
            )

    def validate_features(self) -> None:
        image_features = self.image_features
        if len(image_features) != 1:
            raise ValueError(f"AKT requires exactly one visual input, got {sorted(image_features)}.")

        non_visual_features = [
            name
            for name, feature in (self.input_features or {}).items()
            if feature.type is not FeatureType.VISUAL
        ]
        if non_visual_features:
            raise ValueError(
                "AKT is state-free and accepts only the first image; "
                f"remove non-visual input features {sorted(non_visual_features)}."
            )
