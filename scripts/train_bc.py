from pathlib import Path
import argparse

from robomimic.config import config_factory
from robomimic.scripts.train import train
import robomimic.utils.torch_utils as TorchUtils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="datasets/red_cube_lift/expert_demos_100.hdf5",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--wandb", action="store_true", help="Log training/validation losses to W&B")
    parser.add_argument("--wandb-project", default="robosuite-red-cube")
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    dataset = str(Path(args.dataset).resolve())
    output_dir = str(Path("checkpoints").resolve())

    config = config_factory("bc")

    # Experiment
    config.experiment.name = "red_cube_bc"
    config.experiment.validate = True
    config.experiment.rollout.enabled = False
    config.experiment.logging.log_wandb = args.wandb
    if args.wandb:
        config.experiment.logging.wandb_proj_name = args.wandb_project
        if args.wandb_entity:
            import robomimic.macros as macros
            macros.WANDB_ENTITY = args.wandb_entity

    # Save checkpoints
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 50
    config.experiment.save.on_best_validation = True

    # Dataset
    config.train.data = [{"path": dataset}]
    config.train.output_dir = output_dir
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.train.hdf5_cache_mode = "all"
    config.train.num_data_workers = 0

    # Training
    config.train.num_epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.seq_length = 1

    # Low-dimensional observations
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
        "robot0_joint_vel",
        "cube_pos",
        "blue_cube_pos",
        "green_cube_pos",
    ]
    config.observation.modalities.obs.rgb = []

    # Deterministic MLP BC
    config.algo.gmm.enabled = False
    config.algo.rnn.enabled = False
    config.algo.transformer.enabled = False

    device = TorchUtils.get_torch_device(
        try_to_use_cuda=True
    )

    print("Dataset:", dataset)
    print("Device:", device)

    train(config, device=device)


if __name__ == "__main__":
    main()
