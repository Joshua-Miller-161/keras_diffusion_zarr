from pathlib import Path
import torch
import xarray as xr
import numpy as np
import lightning as pl

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint


from ..data.scaling import DataScaler
from ..data.data_loading import get_dataloader, prepare_and_scale_data
from ..data.constants import TRAINING_COORDS_LOOKUP
from ..data_Josh.data_module import LightningDataModule as JoshDataModule

from .ema import EMA

from ..yang_cm.cwgan_patch import cWGAN_GP
from ..yang_cm.utils import create_model as create_cm_model

from .models.diffusion import LightningDiffusion, setup_edm_model, setup_vp_model
from .models.deterministic import LightningDeterministic
from .models.gan import LightningGAN

_MODELS_DICT = {
    "diffusion": LightningDiffusion,
    "gan": LightningGAN,
    "deterministic": LightningDeterministic,
}


def _get_output_variables(config):
    if hasattr(config.data, "predictands"):
        return list(config.data.predictands.variables)
    return config.data.variables[1]


def build_model(config, checkpoint_name=None):
    model_type = config.model_type
    model_class = _MODELS_DICT[model_type]
    output_variables = _get_output_variables(config)
    model_kwargs = {
        "output_channels": output_variables,
        "weights": config.training.loss_weights,
    }

    if model_type == "diffusion":
        base_model = create_cm_model(config)

        diffusion_type = config.diffusion_type

        if diffusion_type == "karras":
            base_model, loss_config = setup_edm_model(config, base_model, config.device)
        elif diffusion_type == "vpsde":
            base_model, loss_config = setup_vp_model(config, base_model, config.device)

    elif model_type == "gan":
        base_model = cWGAN_GP(config)
        loss_config = config.training.loss_config
    elif model_type == "deterministic":
        base_model = create_cm_model(config)
        loss_config = {"loss_type": config.training.loss}

    # introducing the location specific parameters led to weird issues with the compiled model
    # ONLY during inference (training worked fine)
    # I could only fix them using an eager backend, after which the model behaves as expected.
    # Should investigate at some point what's going wrong with the LocationParameterField
    # in yang_cm/unet, and how to fix it for different compilation backends.
    base_model = torch.compile(base_model)

    if checkpoint_name is None:
        model = model_class(base_model, loss_config, config.optim, **model_kwargs)
    else:
        if not Path(checkpoint_name).is_file():
            base_output_dir = Path(config.base_output_dir)
            run_name = config.run_name
            project_name = config.project_name
            output_dir = str(base_output_dir / project_name / run_name)
            checkpoint_name = output_dir + f"/chkpts/{checkpoint_name}"

        model = model_class.load_from_checkpoint(
            checkpoint_name,
            model=base_model,
            loss_config=loss_config,
            optimizer_config=config.optim,
            **model_kwargs,
        )

    return model



def convert_precision(precision_string):
    conversion = {"32": "32", "bf16": "bf16-mixed", "16": "16-mixed"}
    return conversion[precision_string]


def build_trainer(
    training_config, gradient_clip_val, callback_config, precision, device, logger
):
    callbacks = get_callbacks(callback_config)
    trainer_args = get_training_config(training_config, gradient_clip_val, device)
    trainer_args["callbacks"] = callbacks
    trainer = pl.Trainer(
        precision=convert_precision(precision), logger=logger, **trainer_args
    )
    return trainer


def build_or_load_data_scaler(config, data_scaler_parameters_path = None):
    if data_scaler_parameters_path is None:
        ds_args = (
            Path(config.data.dataset_path),
            config.data.variable_scaler_map,
            config.data.location_config,
            config.data.train_indices
        )
        data_scaler = build_data_scaler(*ds_args)
    else:
        data_scaler = load_data_scaler(data_scaler_parameters_path)
    return data_scaler

def build_data_scaler(data_path, variable_scaler_map, variable_location_config, split_config):
    split = create_indices(split_config)
    ds = prepare_and_scale_data(data_path, split, variable_location_config, data_transform=None)
    data_scaler = DataScaler(variable_scaler_map)
    data_scaler.fit(ds)
    ds.close()

    return data_scaler

def load_data_scaler(parameters_path):
    data_scaler = DataScaler({})
    data_scaler.load_scaler_parameters(parameters_path)
    return data_scaler
    
def get_training_config(training_config, gradient_clip_val, device):
    trainer_args = {}
    trainer_args["accelerator"] = "gpu"  # TODO: FIX
    trainer_args["max_epochs"] = training_config.n_epochs
    trainer_args["gradient_clip_val"] = gradient_clip_val
    return trainer_args


def configure_location_args(config, data_path):

    if config.model.location_parameters is None:
        config.model.location_parameter_config = None
        return config
    ds = xr.open_dataset(data_path)
    coords = ds.lat.values, ds.lon.values
    config.model.location_parameter_config = coords, config.model.location_parameters
    return config


def get_callbacks(callback_args):
    args = dict(callback_args)

    callbacks = []
    ema_rate = args.get("ema_rate")
    if ema_rate is not None:
        callbacks.append(EMA(ema_rate))
    lr_monitor_interval = args.get("lr_monitor")
    if lr_monitor_interval is not None:

        lr_monitor = LearningRateMonitor(logging_interval=lr_monitor_interval)
        callbacks.append(lr_monitor)

    checkpoint_dir = args.pop("checkpoint_dir")
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.4f}",
            every_n_epochs=1,
            save_top_k=10,
            monitor="val_loss",
        )
    )
    save_n_epochs = args.get('save_n_epochs')
    if save_n_epochs is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch}-{val_loss:.4f}",
                every_n_epochs=save_n_epochs,
                save_top_k=-1,  # -1 means save all checkpoints
                save_last=False,  # Avoids saving an additional last checkpoint
                monitor=None,  # No monitoring, just saving every n epochs
            )
        )
    return callbacks

def setup_custom_training_coords(config, sampling_config):

    if config.model.location_parameters is None:
        training_coords = None
    else:
        training_coords = TRAINING_COORDS_LOOKUP[sampling_config.training_dataset]
    config.data.training_coords = training_coords
    return config


def create_custom_indices(indices_config: dict):
    indices = []
    for indices_type, config in indices_config.items():
        if indices_type == "date_range":
            start_date, end_date = config
            new_indices = xr.date_range(
                np.datetime64(start_date), np.datetime64(end_date)
            )
        elif indices_type == "isel":
            new_indices = config
    indices.append(new_indices)
    return np.concatenate(indices, axis=0)


def create_indices(full_indices_config):
    full_indices_config = dict(full_indices_config)
    split = np.array(full_indices_config.pop("split"))
    if len(full_indices_config) > 0:
        return split, create_custom_indices(full_indices_config)
    else:
        return split


def build_dataloaders(config, transform, num_workers):

    if getattr(config.data, "use_josh_pipeline", False):
        datamodule = build_josh_datamodule(config, num_workers=num_workers)
        datamodule.setup("fit")
        return datamodule.train_dataloader(), datamodule.val_dataloader()

    dl_configs = [
        (config.data.train_indices, True, False),
        (config.data.eval_indices, False, True),
    ]
    dls = (
        build_dataloader(
            config.data.dataset_path,
            config.data.variables,
            indices,
            transform,
            config.data.variable_location,
            config.data.location_config,
            config.data.image_size,
            config.training.loss_buffer_width,
            config.data.training_coords,
            config.training.batch_size,
            shuffle=shuffle,
            evaluation=evaluation,
            num_workers=num_workers,
        )
        for indices, shuffle, evaluation in dl_configs
    )

    return dls


def build_dataloader(
    data_path,
    variables,
    indices,
    transform,
    variable_location,
    location_config,
    image_size,
    buffer_width,
    training_coords,
    batch_size,
    shuffle,
    evaluation,
    num_workers,
):
    formatted_indices = create_indices(indices)
    dloader_kwargs = dict(        
        include_time_inputs=False,
        variable_location=variable_location,
        location_config=location_config,
        image_size=image_size,
        buffer_width=buffer_width,
        training_coords=training_coords,
        batch_size=batch_size,
        split=formatted_indices,
        evaluation=evaluation,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    
    dataloader = get_dataloader(
        data_path,
        variables,
        transform,
        **dloader_kwargs
    )
    return dataloader


def build_josh_datamodule(config, num_workers=0):
    data_cfg = config.data
    return JoshDataModule(
        config=config,
        active_dataset_name=data_cfg.dataset,
        model_src_dataset_name=getattr(data_cfg, "dataset_name", data_cfg.dataset),
        input_transform_dataset_name=getattr(
            data_cfg, "input_transform_dataset", getattr(data_cfg, "dataset_name", data_cfg.dataset)
        ),
        transform_dir=getattr(data_cfg, "transform_dir", None),
        batch_size=config.training.batch_size,
        filename=getattr(data_cfg, "filename", getattr(data_cfg, "train_filename", None)),
        val_filename=getattr(data_cfg, "val_filename", None),
        include_time_inputs=getattr(data_cfg, "time_inputs", False),
        evaluation=False,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=getattr(data_cfg, "prefetch_factor", None),
    )
