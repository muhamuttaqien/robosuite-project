# Robosuite Project

Robotic manipulation experiments using **robosuite**, **robomimic**, and the **Franka Panda** robot in MuJoCo.

This project investigates imitation learning for robotic manipulation, starting with **Behavioral Cloning (BC)** and extending toward ACT and Diffusion Policy.

## Task Overview

The environment contains three cubes: red, blue, and green. The robot must grasp and lift the **red cube**, while the other two act as distractors.

Both the robot's initial arm configuration and the cube positions are randomized across episodes.

## Behavioral Cloning (BC)

The BC policy is trained using successful trajectories collected by a scripted expert.

| Configuration | Value |
|---|---|
| Robot | Franka Panda |
| Algorithm | BC (MLP) |
| Demonstrations | 100 |
| Training | 200 epochs |
| Control | 7D OSC_POSE |
| Observations | Low-dimensional robot and cube states |
| Evaluation | 5 seeds × 50 episodes |

### Evaluation Results

| Metric | Result |
|---|---|
| Success Rate | **92.4% (231/250)** |
| Mean ± SD across seeds | **92.4% ± 8.65 pp** |
| Failed Episodes | 19 |

Failure breakdown: 9 grasp failures, 7 insufficient-lift failures, and 3 insufficient-hold failures (heuristic classifications).

[View evaluation results, images, and videos on W&B](https://wandb.ai/muhamuttaqien/robosuite-project/runs/atyi069p)

### Example Rollouts

**Successful demonstrations**

![Successful rollout 1](assets/success_1.gif)
![Successful rollout 2](assets/success_2.gif)
![Successful rollout 3](assets/success_3.gif)

**Failed demonstrations**

![Failed rollout 1](assets/failure_1.gif)
![Failed rollout 2](assets/failure_2.gif)
![Failed rollout 3](assets/failure_3.gif)

*GIF previews require generating the files from the recorded evaluation videos.*

## Installation

```bash
git clone https://github.com/muhamuttaqien/robosuite-project.git
cd robosuite-project

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**1. Collect expert demonstrations**

```bash
python scripts/collect_demos.py \
    --num-demos 100 \
    --max-attempts 600 \
    --output datasets/red_cube_lift/expert_demos_100.hdf5
```

**2. Train BC**

```bash
python scripts/train_bc.py \
    --dataset datasets/red_cube_lift/expert_demos_100.hdf5 \
    --epochs 200 \
    --batch-size 64
```

**3. Evaluate BC**

Replace `<checkpoint.pth>` with your trained checkpoint path.

```bash
python scripts/evaluate_bc.py \
    --checkpoint <checkpoint.pth> \
    --episodes 50 \
    --seeds 2000 3000 4000 5000 6000 \
    --save-videos \
    --videos-per-outcome 3 \
    --wandb \
    --wandb-project robosuite-project
```

## Experiment Tracking

Training metrics, evaluation results, failure analysis, and rollout videos are tracked using [Weights & Biases](https://wandb.ai/muhamuttaqien/robosuite-project).

## Future Work

- Action Chunking Transformer (ACT)
- Diffusion Policy
- World Model-guided manipulation

## References

- [robosuite](https://github.com/ARISE-Initiative/robosuite)
- [robomimic](https://github.com/ARISE-Initiative/robomimic)
