"""Script to train model with specified config file path, optionally resuming from checkpoint.

This script is used to train a model with specified config file path, optionally resuming from checkpoint.
We use PyTorch Lighting for training and wandb for logging / checkpointing.

Examples:
    $ python train.py -c configs/configs/gan.py
    $ python train.py -c configs/configs/gan.py -C checkpoints/last.ckpt
"""
import sys
sys.dont_write_bytecode=True
import os
from pathlib import Path
import argparse
import importlib
import logging
from dotenv import load_dotenv, dotenv_values, find_dotenv

sys.path.append(os.path.dirname(os.getcwd()))
from src.diffusion_downscaling.lightning.utils import build_trainer, configure_location_args, build_josh_datamodule, build_or_load_data_scaler, build_dataloaders, build_model, LossOnlyProgressBar, save_training_config

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

import wandb
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("medium")
#====================================================================
log_dir = os.path.join(os.path.dirname(os.getcwd()), "Outputs")

os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.log")
open(log_file, 'w').close()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Overwrite mode
        logging.StreamHandler(),  # Also logs to stdout
    ],
)
# Create logger
logger = logging.getLogger(__name__)
logger.info(" << <<< <<<< Logging setup complete. See %s >>>> >>> >>>", log_file)
print(" << <<< <<<< Logging setup complete. See", log_file, " >>>> >>> >>>")
#====================================================================
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
    logger.info(f" >> INSIDE main | config_path: {config_path}")
    
    data_path = Path(config.data.dataset_path)
    logger.info(f" >> INSIDE main | data_path: {data_path}")

    config = configure_location_args(config, data_path)
    use_josh_pipeline = getattr(config.data, "use_josh_pipeline", False)

    if use_josh_pipeline:
        datamodule = build_josh_datamodule(config, num_workers=1)
        datamodule.setup("fit")
        training_dataloader, eval_dataloader = (
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
        )
        logger.info(f" >> INSIDE main | built_josh_datamodule")
    else:
        datamodule = None
        data_scaler = build_or_load_data_scaler(config)
        training_dataloader, eval_dataloader = build_dataloaders(
            config, data_scaler.transform, num_workers=1
        )
        logger.info(f" >> INSIDE main | built_old_datamodule")

    logger.info(f" >> INSIDE main | building_model ...")
    model = build_model(config)
    logger.info(f" >> INSIDE main | built_model")

    #----------------------------------------------------------------
    # Get variables from .env file

    load_dotenv()

    config_ = dotenv_values(find_dotenv(usecwd=True))
    #----------------------------------------------------------------
    # Place to put Asaoulis' transforms

    scalar_params_dir = os.path.join(config_['WORK_DIR'], config.data.dataset)
    os.makedirs(scalar_params_dir, exist_ok=True)
    if not use_josh_pipeline:
        data_scaler.save_scaler_parameters(os.path.join(scalar_params_dir, 'scaler_parameters.pkl'))
    #----------------------------------------------------------------
    # Checkpoint path

    checkpoint_base_path = os.path.join(config_['WORK_DIR'], "checkpoints", config.data.dataset, config.run_name)
    logger.info(f" >> INSIDE train | checkpoint_base_path {checkpoint_base_path}")
    os.makedirs(checkpoint_base_path, exist_ok=True)
    saved_config_path = save_training_config(config, checkpoint_base_path)
    logger.info(" >> INSIDE train | saved training config to %s", saved_config_path)
    #----------------------------------------------------------------
    # Callbacks

    callback_config = {
        "checkpoint_dir": checkpoint_base_path,
        "lr_monitor": "step",
        "ema_rate": config.model.ema_rate,
        "save_n_epochs": config.training.save_n_epochs
    }
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_base_path) / checkpoint_path
    logger.info(f" >> INSIDE train | checkpoint_path: {checkpoint_path}")

    #----------------------------------------------------------------
    # Logger

    #wandb_logger = WandbLogger(project=project_name, name=run_name, save_dir=str(output_dir))
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(config_['WORK_DIR'], 'tensorboard'),  # root directory
        name=config.data.dataset,              # subdirectory = project
        version=config.run_name,               # sub-subdirectory = run
        default_hp_metric=False,        # optional, avoids extra "hp_metric" plot
    )

    logger.info(f" >> INSIDE train | tensorboard path {os.path.join(config_['WORK_DIR'], 'tensorboard')}")
    #----------------------------------------------------------------
    # Put all together in the Trainer object

    trainer = build_trainer(
        config.training,
        config.optim.grad_clip,
        callback_config,
        config.precision,
        config.device,
        tb_logger,
    )

    #pbar = LossOnlyProgressBar()
    # trainer = Trainer(
    #     default_root_dir=os.path.join("lightning_logs", config.data.dataset_name),
    #     max_epochs=config.training.n_epochs,
    #     accelerator="gpu",
    #     devices=torch.cuda.device_count(),
    #     strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
    #     use_distributed_sampler=True,
    #     log_every_n_steps=10,
    #     val_check_interval=1.0, # Run validation at the end of every epoch
    #     callbacks=[pbar, checkpoint_cb]
    # )

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
