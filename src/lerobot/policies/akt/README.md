# AKT: Action Keyframe Transformer

AKT is LeRobot ACT specialized for state-free, open-loop keyframe prediction.
It uses one image at frame `t` to predict and execute every future keyframe
action from `t+1` onward.

The policy enforces these invariants:

- exactly one visual input and no robot-state input;
- `action_delta_start_index=1`, so the initial action at `t` is never a target;
- `n_action_steps=chunk_size`, so one inference executes the full keyframe chunk.

For an episode containing one initial frame plus four future keyframes:

```bash
lerobot-train \
  --dataset.episode_start_only=true \
  --policy.type=akt \
  --policy.chunk_size=4 \
  --policy.n_action_steps=4 \
  --policy.input_features='{"observation.images.cam_head":{"type":"VISUAL","shape":[3,480,640]}}'
```

`action_delta_start_index` does not need to be passed: AKT fixes it to `1` and
rejects attempts to change it.
