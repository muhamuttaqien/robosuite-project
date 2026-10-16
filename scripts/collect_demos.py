#!/usr/bin/env python3
"""Collect successful scripted-expert red-cube lifts with randomized Panda/cubes.

Run from project root: python scripts/collect_demos.py --num-demos 20 --render
Requires robosuite==1.5.2; API behavior should be verified in your installation.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor


class ThreeCubeRedLift(Lift):
    """Standard Lift red target with green/blue distractors."""

    def _load_model(self):
        super()._load_model()
        arena = self.model.mujoco_arena
        self.green_cube = BoxObject(name="green_cube", size_min=[.02]*3,
                                    size_max=[.02]*3, rgba=[.1,.9,.2,1])
        self.blue_cube = BoxObject(name="blue_cube", size_min=[.02]*3,
                                   size_max=[.02]*3, rgba=[.1,.3,1.,1])
        self.placement_initializer = UniformRandomSampler(
            name="ThreeCubeSampler", mujoco_objects=[self.cube, self.green_cube, self.blue_cube],
            x_range=[-.12,.12], y_range=[-.12,.12], rotation=None,
            ensure_object_boundary_in_range=True, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=.01, rng=self.rng,
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[self.cube, self.green_cube, self.blue_cube],
        )

    def _setup_references(self):
        super()._setup_references()
        self.green_body_id = self.sim.model.body_name2id(self.green_cube.root_body)
        self.blue_body_id = self.sim.model.body_name2id(self.blue_cube.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            @sensor(modality="object")
            def green_cube_pos(cache):
                return self.sim.data.body_xpos[self.green_body_id].copy()

            @sensor(modality="object")
            def blue_cube_pos(cache):
                return self.sim.data.body_xpos[self.blue_body_id].copy()

            for fn in (green_cube_pos, blue_cube_pos):
                observables[fn.__name__] = Observable(
                    name=fn.__name__, sensor=fn, sampling_rate=self.control_freq
                )
        return observables

    def _check_success(self):
        # Match the named target instead of letting a distractor count as success.
        return bool(self.sim.data.body_xpos[self.cube_body_id][2] >
                    self.table_offset[2] + .12)


def xyz(env, body_id):
    return np.array(env.sim.data.body_xpos[body_id], dtype=np.float64)


def eef_xyz(env):
    """Read end-effector site position in world coordinates (robosuite 1.5.x)."""
    robot = env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id.get("right", next(iter(site_id.values())))
    return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()


def panda_gripper(env):
    """Resolve single-arm gripper under robosuite 1.5.x."""
    gripper = env.robots[0].gripper
    if isinstance(gripper, dict):
        gripper = gripper.get("right", next(iter(gripper.values())))
    return gripper


def obs_for_bc(obs):
    """Stable low-dimensional features for the first BC baseline.

    Keep exact keys across demos and train/eval; images can be added later.
    """
    keys = (
        "robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos",
        "robot0_eef_quat", "robot0_gripper_qpos", "cube_pos",
        "green_cube_pos", "blue_cube_pos",
    )
    missing = [k for k in keys if k not in obs]
    if missing:
        raise KeyError(f"Missing observable keys: {missing}; available: {sorted(obs)}")
    return {k: np.asarray(obs[k], dtype=np.float32).copy() for k in keys}


class ScriptedExpert:
    """Finite-state OSC position controller, with fixed initial tool orientation."""

    def __init__(self, gain=8.0, max_cmd=.75, hover=.16, grasp=.012):
        self.gain = gain
        self.max_cmd = max_cmd
        self.hover = hover
        self.grasp = grasp
        self.state = "approach"
        self.ticks = 0
        self.stable = 0

    def act(self, env):
        target = xyz(env, env.cube_body_id)
        current = eef_xyz(env)
        self.ticks += 1
        # The Panda grasp frame can have a z offset relative to cube center.
        # These values are tunable; inspect failures before scaling collection.
        if self.state == "approach":
            goal = target + [0., 0., self.hover]
            if np.linalg.norm(goal - current) < .025:
                self.stable += 1
                if self.stable >= 4:
                    self.state, self.stable = "descend", 0
            else:
                self.stable = 0
        elif self.state == "descend":
            goal = target + [0., 0., self.grasp]
            if np.linalg.norm(goal - current) < .014:
                self.stable += 1
                if self.stable >= 4:
                    self.state, self.stable = "close", 0
            else:
                self.stable = 0
        elif self.state == "close":
            goal = target + [0., 0., self.grasp]
            if self.ticks > 0:
                self.stable += 1
            if self.stable >= 16:
                self.state = "lift"
        else:
            # Keep XY fixed over the red cube; the target cube moves with the hand.
            # Maintain lift goal in absolute table-frame height.
            goal = np.array([target[0], target[1], env.table_offset[2] + .28])

        action = np.zeros(env.action_dim, dtype=np.float64)
        delta = (goal - current) * self.gain
        action[:3] = np.clip(delta, -self.max_cmd, self.max_cmd)
        action[6] = 1. if self.state in ("close", "lift") else -1.
        return action


def create_env(args):
    env = ThreeCubeRedLift(
        robots="Panda", has_renderer=args.render,
        has_offscreen_renderer=bool(getattr(args, "save_videos", False) or getattr(args, "wandb", False)), use_camera_obs=False,
        use_object_obs=True, reward_shaping=False, ignore_done=False,
        horizon=args.max_steps, control_freq=20, hard_reset=False,
        initialization_noise={"magnitude": args.arm_noise, "type": "uniform"},
        seed=args.seed,
    )
    if env.action_dim != 7:
        raise RuntimeError(f"Expected 7D OSC_POSE, got {env.action_dim}. "
                           "Configure Panda OSC_POSE controller explicitly.")
    return env


def rollout(env, args):
    obs = env.reset()
    initial_red_z = xyz(env, env.cube_body_id)[2]
    expert = ScriptedExpert(gain=args.gain, hover=args.hover, grasp=args.grasp)
    states, actions, rewards, dones, observations, next_observations = [], [], [], [], [], []
    held = 0
    success = False
    for t in range(args.max_steps):
        state = env.sim.get_state().flatten().copy()
        action = expert.act(env)
        old_obs = obs_for_bc(obs)
        obs, _, done, _ = env.step(action)
        new_obs = obs_for_bc(obs)
        red_z = xyz(env, env.cube_body_id)[2]
        # Distinguish a real grasp/lift from one-frame contact or a falling block.
        raised = (red_z - initial_red_z >= args.lift_height)
        grasped = bool(env._check_grasp(gripper=panda_gripper(env),
                                        object_geoms=env.cube))
        held = held + 1 if raised and grasped else 0
        success = held >= args.hold_steps
        states.append(state)
        actions.append(action.copy())
        rewards.append(float(success))
        dones.append(int(success or done or t == args.max_steps - 1))
        observations.append(old_obs)
        next_observations.append(new_obs)
        if args.render:
            env.render()
        if success or done:
            break
    return dict(success=success, states=states, actions=actions,
                rewards=rewards, dones=dones, obs=observations,
                next_obs=next_observations, final_state=expert.state,
                lift=float(xyz(env, env.cube_body_id)[2] - initial_red_z))


def write_demo(data_group, index, trajectory, xml):
    ep = data_group.create_group(f"demo_{index}")
    ep.attrs["num_samples"] = len(trajectory["actions"])
    ep.attrs["model_file"] = xml
    for name in ("states", "actions", "rewards", "dones"):
        ep.create_dataset(name, data=np.asarray(trajectory[name]), compression="gzip")
    for group_name in ("obs", "next_obs"):
        group = ep.create_group(group_name)
        for key in trajectory[group_name][0]:
            group.create_dataset(key, data=np.stack([
                entry[key] for entry in trajectory[group_name]
            ]), compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-demos", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--arm-noise", type=float, default=.04)
    parser.add_argument("--gain", type=float, default=8.)
    parser.add_argument("--hover", type=float, default=.16)
    parser.add_argument("--grasp", type=float, default=.012)
    parser.add_argument("--lift-height", type=float, default=.10)
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", default="datasets/red_cube_lift/expert_demos.hdf5")
    args = parser.parse_args()
    np.random.seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = create_env(args)
    try:
        with h5py.File(output, "w") as file:
            group = file.create_group("data")
            group.attrs["env_args"] = json.dumps({
                "env_name": "ThreeCubeRedLift", "type": 1,
                "env_kwargs": {"robots": "Panda", "control_freq": 20,
                               "initialization_noise": {"type": "uniform", "magnitude": args.arm_noise}}
            })
            group.attrs["collector_config"] = json.dumps(vars(args))
            successful = 0
            total_steps = 0
            for attempt in range(1, args.max_attempts + 1):
                traj = rollout(env, args)
                print(f"Attempt {attempt:03d} | {traj['final_state']:>8} | "
                      f"red lift {traj['lift']:+.3f} m | "
                      f"steps {len(traj['actions'])} | "
                      f"{'SUCCESS' if traj['success'] else 'FAILED'}", flush=True)
                if traj["success"]:
                    write_demo(group, successful, traj, env.sim.model.get_xml())
                    successful += 1
                    total_steps += len(traj["actions"])
                    group.attrs["total"] = total_steps
                    file.flush()
                if successful >= args.num_demos:
                    break
            print(f"Saved {successful}/{args.num_demos} successful demos to {output}")
            print(f"Success among attempts: {successful}/{attempt}")
            if successful == 0:
                print("No successful trajectories: tune grasp offset, controller and workspace first.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
