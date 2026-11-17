#!/usr/bin/env python

import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies import get_policy_class
from lerobot.policies.akt.configuration_akt import AKTConfig
from lerobot.policies.akt.modeling_akt import AKTPolicy


def _visual_features() -> dict[str, PolicyFeature]:
    return {
        "observation.images.cam_head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }


def test_akt_is_registered() -> None:
    assert get_policy_class("akt") is AKTPolicy


def test_akt_targets_only_future_keyframes() -> None:
    config = AKTConfig(chunk_size=4, n_action_steps=4, device="cpu", push_to_hub=False)
    assert config.action_delta_start_index == 1
    assert config.action_delta_indices == [1, 2, 3, 4]


def test_akt_rejects_observation_aligned_action() -> None:
    with pytest.raises(ValueError, match="must be 1"):
        AKTConfig(
            chunk_size=4,
            n_action_steps=4,
            action_delta_start_index=0,
            device="cpu",
            push_to_hub=False,
        )


def test_akt_rejects_partial_chunk_execution() -> None:
    with pytest.raises(ValueError, match="must equal chunk_size"):
        AKTConfig(chunk_size=4, n_action_steps=3, device="cpu", push_to_hub=False)


def test_akt_accepts_one_image_without_state() -> None:
    config = AKTConfig(
        chunk_size=4,
        n_action_steps=4,
        input_features=_visual_features(),
        device="cpu",
        push_to_hub=False,
    )
    config.validate_features()


def test_akt_rejects_state_input() -> None:
    features = _visual_features()
    features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
    config = AKTConfig(
        chunk_size=4,
        n_action_steps=4,
        input_features=features,
        device="cpu",
        push_to_hub=False,
    )
    with pytest.raises(ValueError, match="state-free"):
        config.validate_features()
