"""Script to train model with specified config file path, optionally resuming from checkpoint.

This script is used to train a model with specified config file path, optionally resuming from checkpoint.
We use PyTorch Lighting for training and wandb for logging / checkpointing.

Examples:
    $ python train.py -c configs/configs/gan.py
    $ python train.py -c configs/configs/gan.py -C checkpoints/last.ckpt
"""

from pathlib import Path
import argparse
import importlib

import diffusion_downscaling.lightning.utils as lightning_utils

from lightning.pytorch.loggers import WandbLogger

import wandb
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("medium")


def parse_module(path):
    return path.replace(".py", "").replace("/", ".")


def main(config_path, checkpoint_path=None):
    """
    Train model with specified config file path, optionally resuming from checkpoint.
    :param config_path: Path to the config file
    :param checkpoint_path: Checkpoint path in order to resume training

    """

    # Import config dynamically using provided config path
    config_module = importlib.import_module(parse_module(config_path))
    config = config_module.get_config()

    data_path = Path(config.data.dataset_path)
    config = lightning_utils.configure_location_args(config, data_path)
    use_josh_pipeline = getattr(config.data, "use_josh_pipeline", False)

    if use_josh_pipeline:
        datamodule = lightning_utils.build_josh_datamodule(config, num_workers=16)
        datamodule.setup("fit")
        training_dataloader, eval_dataloader = (
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
        )
    else:
        datamodule = None
        data_scaler = lightning_utils.build_or_load_data_scaler(config)
        training_dataloader, eval_dataloader = lightning_utils.build_dataloaders(
            config, data_scaler.transform, num_workers=16
        )

    model = lightning_utils.build_model(config)

    base_output_dir = Path(config.base_output_dir)
    run_name = config.run_name
    project_name = config.project_name
    output_dir = base_output_dir / project_name / run_name
    output_dir.mkdir(exist_ok=True, parents=True)
    if not use_josh_pipeline:
        data_scaler.save_scaler_parameters(output_dir / 'scaler_parameters.pkl')

    checkpoint_base_path = f"{str(output_dir)}/chkpts"
    callback_config = {
        "checkpoint_dir": checkpoint_base_path,
        "lr_monitor": "step",
        "ema_rate": config.model.ema_rate,
        "save_n_epochs": config.training.save_n_epochs
    }
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_base_path) / checkpoint_path

    wandb_logger = WandbLogger(project=project_name, name=run_name, save_dir=str(output_dir))
    trainer = lightning_utils.build_trainer(
        config.training,
        config.optim.grad_clip,
        callback_config,
        config.precision,
        config.device,
        wandb_logger,
    )

    if datamodule is not None:
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, training_dataloader, eval_dataloader, ckpt_path=checkpoint_path)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model with specified config file path"
    )
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        default="configs/configs/gan.py",
        help="Path to the config file",
    )
    parser.add_argument(
        "-C",
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path in order to resume training",
    )
    args = parser.parse_args()

    main(args.config_path, args.checkpoint)
