#!/usr/bin/env python3
"""Multi-seed robomimic BC evaluation with W&B media and heuristic failure analysis."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from collect_demos import create_env, obs_for_bc, xyz, panda_gripper


def failure_reason(result, lift_height):
    """Coarse post-hoc diagnostic (NOT a verified physical failure mechanism)."""
    if result['success']:
        return 'success'
    if result['min_eef_red_distance_m'] > 0.085:
        return 'did_not_reach_target'
    if result['grasped_steps'] == 0:
        return 'reached_but_no_grasp_detected'
    if result['max_red_lift_m'] < lift_height:
        return 'grasp_detected_insufficient_lift'
    return 'lifted_but_not_held_long_enough'


def frame_from_env(env, camera, width, height):
    # Rendering directly via robosuite's MuJoCo simulation avoids adding RGB
    # observations to a low-dimensional BC policy.
    frame = env.sim.render(camera_name=camera, width=width, height=height, depth=False)
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise RuntimeError(f'Unexpected camera frame shape: {frame.shape}')
    return np.ascontiguousarray(np.flipud(frame[..., :3]).astype(np.uint8))


def evaluate_episode(env, policy, args, video_path=None):
    obs = env.reset()
    initial_red_z = float(xyz(env, env.cube_body_id)[2])
    held = 0
    max_held = 0
    max_lift = 0.0
    grasped_steps = 0
    success = False
    sum_reward = 0.0
    min_distance = float('inf')
    policy.start_episode()
    writer = None
    try:
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video_path), fps=args.video_fps, codec='libx264',
                                        quality=7, macro_block_size=1)
            writer.append_data(frame_from_env(env, args.camera, args.video_width, args.video_height))
        for step in range(1, args.max_steps + 1):
            with torch.no_grad():
                action = np.asarray(policy(ob=obs_for_bc(obs)), dtype=np.float64).reshape(-1)
            if action.shape != (env.action_dim,) or not np.isfinite(action).all():
                raise RuntimeError(f'Invalid policy output: shape={action.shape}, values={action}')
            obs, reward, done, _ = env.step(np.clip(action, -1., 1.))
            sum_reward += float(reward)
            red_pos = xyz(env, env.cube_body_id)
            lift = float(red_pos[2] - initial_red_z)
            max_lift = max(max_lift, lift)
            min_distance = min(min_distance, float(np.linalg.norm(np.asarray(obs['robot0_eef_pos']) - red_pos)))
            grasped = bool(env._check_grasp(gripper=panda_gripper(env), object_geoms=env.cube))
            grasped_steps += int(grasped)
            held = held + 1 if (lift >= args.lift_height and grasped) else 0
            max_held = max(max_held, held)
            success = held >= args.hold_steps
            if writer is not None and step % args.video_stride == 0:
                writer.append_data(frame_from_env(env, args.camera, args.video_width, args.video_height))
            if args.render:
                env.render()
            if success or done:
                break
    finally:
        if writer is not None:
            writer.close()
    result = dict(success=int(success), steps=step,
                  max_red_lift_m=round(max_lift, 5),
                  min_eef_red_distance_m=round(min_distance, 5),
                  grasped_steps=grasped_steps, max_consecutive_lift_grasp_steps=max_held,
                  return_value=round(sum_reward, 5))
    result['failure_reason'] = failure_reason(result, args.lift_height)
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_charts(per_seed, failed, media_dir):
    media_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(x['seed']) for x in per_seed], [100*x['success_rate'] for x in per_seed])
    ax.set_ylim(0, 100)
    ax.set_xlabel('Evaluation seed')
    ax.set_ylabel('Success rate (%)')
    ax.set_title('BC success by random seed')
    fig.tight_layout()
    rates_path = media_dir / 'success_by_seed.png'
    fig.savefig(rates_path, dpi=150)
    plt.close(fig)
    counts = Counter(r['failure_reason'] for r in failed)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = list(counts) or ['no_failures']
    values = [counts[x] for x in labels] if counts else [0]
    ax.barh(labels, values)
    ax.set_xlabel('Failed episodes')
    ax.set_title('Heuristic failure categories')
    fig.tight_layout()
    failures_path = media_dir / 'failure_categories.png'
    fig.savefig(failures_path, dpi=150)
    plt.close(fig)
    return rates_path, failures_path, counts


def main():
    p = argparse.ArgumentParser(description='Evaluate low-dimensional BC and log results to W&B')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--episodes', type=int, default=50, help='Episodes per seed')
    p.add_argument('--seed', type=int, default=2000)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--max-steps', type=int, default=300)
    p.add_argument('--arm-noise', type=float, default=.04)
    p.add_argument('--lift-height', type=float, default=.10)
    p.add_argument('--hold-steps', type=int, default=10)
    p.add_argument('--render', action='store_true')
    p.add_argument('--output', default='results/bc_eval_multiseed.json')
    p.add_argument('--wandb', action='store_true', help='Enable online/offline W&B logging')
    p.add_argument('--wandb-project', default='robosuite-project')
    p.add_argument('--run-name', '--wandb-run-name', dest='run_name', default='bc-red-cube-eval',
                   help='W&B run name (default: bc-red-cube-eval)')
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-mode', choices=['online','offline','disabled'], default='online')
    p.add_argument('--save-videos', action='store_true', help='Keep up to 3 success + 3 failure clips')
    p.add_argument('--videos-per-outcome', type=int, default=3)
    p.add_argument('--camera', default='agentview')
    p.add_argument('--video-width', type=int, default=640)
    p.add_argument('--video-height', type=int, default=480)
    p.add_argument('--video-fps', type=int, default=20)
    p.add_argument('--video-stride', type=int, default=2)
    args = p.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.hold_steps < 1 or args.videos_per_outcome < 0 or args.video_stride < 1:
        p.error('Invalid episodes/max-steps/hold-steps/videos-per-outcome/video-stride')
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        p.error(f'Checkpoint not found: {ckpt}')
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        p.error('Seeds must be unique')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    media = output.parent / (output.stem + '_media')
    media.mkdir(parents=True, exist_ok=True)
    videos_dir = media / 'videos'
    want_video = args.save_videos or args.wandb
    if want_video:
        videos_dir.mkdir(exist_ok=True)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         mode=args.wandb_mode, name=args.run_name, job_type='evaluation',
                         config={**vars(args), 'checkpoint': str(ckpt.resolve())})
    try:
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(ckpt.resolve()), device=device, verbose=False)
        print('Checkpoint observations:', list(ckpt_dict['shape_metadata']['all_shapes']))
        all_rows, per_seed, chosen_videos = [], [], {'success': [], 'failure': []}
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            # create_env takes the same args, only these two attributes determine rendering.
            args.seed = seed
            env = create_env(args)
            # Offscreen rendering is opt-in; enable it before creating env in collect_demos.
            rows = []
            try:
                for episode in range(1, args.episodes + 1):
                    need_capture = want_video and (len(chosen_videos['success']) < args.videos_per_outcome or len(chosen_videos['failure']) < args.videos_per_outcome)
                    tentative = videos_dir / f'_candidate_seed{seed}_ep{episode:03d}.mp4' if need_capture else None
                    res = evaluate_episode(env, policy, args, tentative)
                    res.update(seed=seed, episode=episode)
                    outcome = 'success' if res['success'] else 'failure'
                    if tentative is not None:
                        if len(chosen_videos[outcome]) < args.videos_per_outcome:
                            selected = videos_dir / f'{outcome}_{len(chosen_videos[outcome])+1}_seed{seed}_ep{episode:03d}.mp4'
                            tentative.rename(selected)
                            chosen_videos[outcome].append({'path': str(selected), 'seed': seed, 'episode': episode})
                            res['video'] = str(selected)
                        else:
                            tentative.unlink(missing_ok=True)
                    rows.append(res)
                    all_rows.append(res)
                    print(f"Seed {seed} | Episode {episode:03}/{args.episodes}: {'SUCCESS' if res['success'] else 'FAILURE'} | lift={res['max_red_lift_m']:.3f} m | {res['failure_reason']}", flush=True)
            finally:
                env.close()
            rate = sum(r['success'] for r in rows) / len(rows)
            item = dict(seed=seed, episodes=len(rows), successes=sum(r['success'] for r in rows),
                        failures=sum(not r['success'] for r in rows), success_rate=rate)
            per_seed.append(item)
            print(f'Seed {seed}: {item["successes"]}/{len(rows)} ({rate*100:.1f}%)')
            if args.seeds is not None:
                seed_path = output.with_name(f'{output.stem}_seed{seed}.json')
                seed_path.write_text(json.dumps({**item, 'episodes_detail': rows}, indent=2) + '\n')
                write_csv(seed_path.with_suffix('.csv'), rows)
            if run is not None:
                run.log({'seed/success_rate': rate, 'seed/successes': item['successes'], 'seed/value': seed})
        rates = np.array([x['success_rate'] for x in per_seed], dtype=float)
        failures = [r for r in all_rows if not r['success']]
        success_rate = float(np.mean([r['success'] for r in all_rows]))
        mean_rate = float(np.mean(rates))
        sample_std = float(np.std(rates, ddof=1)) if len(rates) > 1 else None
        rates_png, failures_png, counts = make_charts(per_seed, failures, media)
        report = {
            'checkpoint': str(ckpt), 'seeds': seeds, 'episodes_per_seed': args.episodes,
            'total_episodes': len(all_rows), 'successes': sum(r['success'] for r in all_rows),
            'failures': len(failures), 'success_rate': success_rate,
            'mean_seed_success_rate': mean_rate,
            'std_seed_success_rate': sample_std,
            'std_definition': 'sample standard deviation of per-seed rates (ddof=1); null for one seed',
            'per_seed': per_seed, 'episodes_detail': all_rows,
            'video_examples': chosen_videos,
        }
        output.write_text(json.dumps(report, indent=2) + '\n')
        write_csv(output.with_suffix('.csv'), all_rows)
        failure_report = {
            'method': 'heuristic categorization from measured proximity, grasp flag, lift and hold duration; not verified physical root causes',
            'thresholds': {'near_distance_m': .085, 'lift_height_m': args.lift_height, 'hold_steps': args.hold_steps},
            'total_failures': len(failures), 'failure_categories': dict(counts),
            'failed_episodes': failures,
            'recommendations': {
                'did_not_reach_target': 'Inspect end-effector trajectories and workspace coverage.',
                'reached_but_no_grasp_detected': 'Inspect approach angle, gripper timing and closing width.',
                'grasp_detected_insufficient_lift': 'Inspect post-grasp motion and slippage.',
                'lifted_but_not_held_long_enough': 'Inspect hold duration, premature release and oscillation.'
            }
        }
        analysis_path = output.parent / (output.stem + '_analysis_failures.json')
        analysis_path.write_text(json.dumps(failure_report, indent=2) + '\n')
        lines = ['# BC Failure Analysis', '', f'- Episodes: {len(all_rows)}', f'- Failed: {len(failures)}',
                 '- Labels are heuristic diagnoses, not confirmed physical causes.', '', '## Categories', '']
        lines.extend(f'- {key}: {value}' for key, value in sorted(counts.items()))
        lines += ['', '## Failed episodes', '', '| Seed | Episode | Category | Max lift (m) | Min EE distance (m) |', '|---|---:|---|---:|---:|']
        lines.extend(f"| {r['seed']} | {r['episode']} | {r['failure_reason']} | {r['max_red_lift_m']:.3f} | {r['min_eef_red_distance_m']:.3f} |" for r in failures)
        (output.parent / (output.stem + '_analysis_failures.md')).write_text('\n'.join(lines) + '\n')
        print(f"\nOverall: {report['successes']}/{len(all_rows)} = {100*success_rate:.1f}%")
        print(f"Mean per-seed success: {100*mean_rate:.1f}% | sample SD: {100*sample_std:.2f} pp" if sample_std is not None else f'Mean per-seed success: {100*mean_rate:.1f}% | SD: N/A (one seed)')
        print('Failure categories:', dict(counts))
        print(f'Saved: {output}, {output.with_suffix(".csv")}, {analysis_path}, {media}')
        if run is not None:
            run.log({'evaluation/success_rate': success_rate, 'evaluation/mean_seed_success_rate': mean_rate,
                     'evaluation/std_seed_success_rate': sample_std if sample_std is not None else 0.0,
                     'evaluation/total_episodes': len(all_rows), 'evaluation/failures': len(failures),
                     'charts/success_by_seed': wandb.Image(str(rates_png)),
                     'charts/failure_categories': wandb.Image(str(failures_png)),
                     'episodes/table': wandb.Table(dataframe=__import__('pandas').DataFrame(all_rows))})
            for label, clips in chosen_videos.items():
                for i, clip in enumerate(clips, 1):
                    run.log({f'videos/{label}_{i}': wandb.Video(clip['path'], format='mp4',
                                                              caption=f"{label}: seed {clip['seed']} episode {clip['episode']}")})
            artifact = wandb.Artifact(name='bc-evaluation-results', type='evaluation')
            for fpath in [output, output.with_suffix('.csv'), analysis_path,
                          output.parent / (output.stem + '_analysis_failures.md'), rates_png, failures_png]:
                artifact.add_file(str(fpath))
            run.log_artifact(artifact)
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    main()
