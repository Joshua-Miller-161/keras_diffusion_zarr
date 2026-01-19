"""Model sampling, prediction generation and evaluation.

This module contains the Sampler class, which is a high-level API for sampling 
from all the different models, including diffusion, GANs, regression models. 

The Sampler class will generate predictions given a model, sampling configuration,
and dataloader configuration. It can then run a custom evaluation callable on the
saved predictions folder.
"""

from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import xarray as xr
from pathlib import Path
from functools import reduce
import json
import os
from typing import List, Tuple, Callable, Dict
import logging

logger = logging.getLogger(__name__)

from ..lightning import utils as lightning_utils
from ..data.scaling import DataScaler
from ..data.data_loading import  select_custom_coordinates


def _open_dataset(path: Path):
    path = Path(path)
    if path.suffix == ".zarr" or path.is_dir():
        try:
            return xr.open_zarr(path, consolidated=True)
        except (KeyError, ValueError, OSError):
            return xr.open_zarr(path)
    return xr.open_dataset(path)


def make_predictions_filename(directory, config, prefix="predictions"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    schedule_config = config[0] if isinstance(config, tuple) and len(config) > 0 else {}
    n_steps = schedule_config.get("n", "unknown") if isinstance(schedule_config, dict) else "unknown"

    existing_files = list(directory.glob(f"{prefix}_n={n_steps}_sample*.nc"))
    next_index = len(existing_files)

    print(
        " >> >> INSIDE make_predictions_filename next_index,",
        next_index,
        ", ",
        len(list(directory.glob("*.nc"))),
    )

    new_filename = f"{prefix}_n={n_steps}_sample{next_index}.nc"
    return os.path.join(directory, new_filename)


def diffusion_sampling(
    model, cond, config, callables, output_variables, extra_args=None
):
    """Diffusion model sampling."""
    schedule_kwargs, sampling_kwargs, _ = config
    schedule_callable, sampling_callable = callables
    sigmas = schedule_callable(**schedule_kwargs)
    shape_cond = cond[0].shape if isinstance(cond, list) else cond.shape
    output_shape = (shape_cond[0],) + (len(output_variables),) + (shape_cond[2:])
    initial_scale = sigmas[0].clone()
    if sampling_kwargs.get("sde_type") == "vp":
        initial_scale *= 1 / (1 + initial_scale**2).sqrt()
    # Initialise with random fields.
    starting_noise = torch.randn(output_shape, device="cuda") * initial_scale

    samples = sampling_callable(
        model,
        starting_noise,
        cond,
        sigmas,
        extra_args=extra_args,
        **sampling_kwargs,
    )
    return samples

def forward_pass_sampling(model, cond, *args, **kwargs):
    """Sampling using implicit stochasticity in the model's forward pass
    (e.g. GANs, VAEs)
    """
    samples = model(None, cond)
    return samples

def dropout_forward_pass_sampling(model, cond, *args, **kwargs):
    """Sampling using dropout from a regression based model."""
    model.eval() # THIS IS BAD; just doing it now for the sake of deterministic sampling
    return forward_pass_sampling(model, cond, *args, **kwargs)

class Sampler:
    """High-level API for sampling from all the different models, incl. diffusion, GANs, CNNs.

    evaluate_model_on_all_configs is the outer loop that goes through all the different configurations,
    samples using the model, saves the predictions then calls the evaluation callable.

    generate_predictions is the inner loop that samples from the model and saves the predictions.

    run_sampling is the innermost loop that samples from the model.

    There are several static models that are used to sample from the different model types.
    """

    model_sampling_callables = {
        "diffusion": diffusion_sampling,
        "gan": forward_pass_sampling,
        "deterministic": dropout_forward_pass_sampling,
    }

    precision_lookup = {
        "32": torch.float32,
        "bf16": torch.bfloat16,
        "16": torch.float16,
    }

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        data_scaler: DataScaler,
        output_variables: List[str],
        output_string_format: str,
    ):
        """Sampling setup.

        :param model: PyTorch model to sample from
        :param model_type: str, model type (diffusion, gan, deterministic)
        :param data_scaler: DataScaler, data scaler to rescale the model outputs. Must have a transform
            method that acts on a xarray object.
        :param output_variables: List[str], output variable names.
        :param output_string_format: str, formattable string that can include config-specific parameters
            to generate the name of the output directory.
        """
        self.model = model
        self.eval_dl = None
        self.data_scaler = data_scaler
        self.output_variables = output_variables
        self.output_string_format = output_string_format
        self.model_specific_sampling = self.model_sampling_callables[model_type]
        self.precision = torch.float32

    def evaluate_model_on_all_configs(
        self,
        main_config,
        eval_args: Tuple,
        num_samples: int,
        base_output: str,
        evaluator: Callable = None,
        output_variables: List[str] = None,
    ):
        """Evaluate the model on all configurations provided via eval_args.

        This is the outer loop that goes through all the different configurations, sampling from the model,
        generating predictions and running evaluation over those predictions.

        :param main_config: Config, main configuration object.
        :param eval_args: 2-tuple containing the sampling kwargs and a list of all the configurations to evaluate.
        :param num_samples: int, number of samples to generate.
        :param base_output: str, base output directory to save the predictions and evaluation results.
        :param evaluator: Callable, evaluation function to run on the predictions.
        :param output_variables: List[str], output variable names.
        """
        self.precision = self.precision_lookup[main_config.precision]

        xr_data = _open_dataset(main_config.data.dataset_path)
        buffer_width = main_config.training.loss_buffer_width

        sampling_args, all_configs = eval_args

        for config in all_configs:
            location_config = config[2]["location_config"]
            coords = select_custom_coordinates(xr_data, location_config, buffer_width)
            main_config.data.location_config = location_config
            _, self.eval_dl = lightning_utils.build_dataloaders(
                main_config, self.data_scaler.transform, num_workers=20
            )

            output_string = self.format_output_dir_name(config)
            output_path, predictions_dir, results_dir = self.setup_output_dirs(
                base_output, output_string
            )
            self.save_config(config, output_path)

            print(f"Beginning predictions on {output_string}", flush=True)
            self.generate_predictions(
                config,
                coords,
                num_samples,
                predictions_dir,
                sampling_args,
                output_variables,
            )
            print(f"{num_samples} predictions on {output_string} completed.", flush=True)

            if evaluator is not None:
                print(f"Beginning evaluation on {output_string}", flush=True)
                evaluator(predictions_dir, results_dir, coords, buffer_width)
                print(f"Finished evaluation on {output_string}", flush=True)

    def setup_output_dirs(self, base_output: str, output_string: str):
        output_path = Path(base_output) / output_string
        #predictions_dir = output_path / "predictions"
        predictions_dir = output_path
        results_dir = output_path / "plots"
        predictions_dir.mkdir(exist_ok=True, parents=True)
        results_dir.mkdir(exist_ok=True, parents=True)
        return output_path, predictions_dir, results_dir

    def generate_predictions(
        self,
        config: Dict,
        coords,
        num_samples: int,
        output_dirpath: str,
        sampling_args: Dict,
        output_variables: List[str],
    ):
        """Run model specific sampling, passing all the relevant config forward.

        Once the predictions have been generated, they are saved to netcdf in the output
        directory that has been specified. Each sample is saved in a separate predictions file.

        :param config: Dict, scheduling configurations dictionary.
        :param coords: 2-tuple of lat, lon coordinates.
        :param num_samples: int, number of samples to generate.
        :param output_dirpath: str, formattable output directory to save the predictions.
        :param sampling_args: Dict, additional arguments to pass to the sampling function.
        :param output_variables: List[str], output variable names.
        """
        for sample_id in range(num_samples):
            print(f"Sample run {sample_id}...")
            logger.info(f" >> >> INSIDE sampling.generate_predictions | sampling {sample_id}")
            xr_samples = self.run_sampling(
                coords, config, sampling_args, output_variables=output_variables
            )

            output_filepath = Path(
                make_predictions_filename(output_dirpath, config, prefix="predictions")
            )

            print(f"Saving samples to {output_filepath}...")
            logger.info(f" >> >> INSIDE sampling.generate_predictions | output_filepath {output_filepath}")
            xr_samples.to_netcdf(output_filepath)

    def run_sampling(self, coords, *args, **kwargs):
        """Loop over dataloader and return scaled predictions for the data.

        This is the interface for handling the data, moving it over to the correct
        datatype and device, passing it to the model and running the sampling algorithms.

        Once the predictions have been computed, they are rescaled before being returned.

        :param coords: Tuple or None, a 2-tuple containing the lat, lon coordinates.
        :param *args: Additional arguments to pass to the sampling function.
        :param **kwargs: Additional keyword arguments to pass to the sampling function.
        """
        device = "cuda"
        dims = self.get_dims(coords)

        preds = []
        with logging_redirect_tqdm():
            with tqdm(
                total=len(self.eval_dl.dataset),
                desc="Sampling",
                unit=" timesteps",
            ) as pbar:
                with torch.no_grad():
                    self.model.eval()
                    for batch in self.eval_dl:
                        cond = batch[0]
                        if isinstance(cond, list):
                            cond_shape = cond[0].shape
                            cond = [
                                cond_i.to(device, dtype=self.precision)
                                for cond_i in cond
                            ]
                        else:
                            cond = cond.to(device)
                            cond_shape = cond.shape
                        with torch.autocast(device, dtype=self.precision):
                            samples = self.model_specific_sampling(
                                self.model, cond, *args, **kwargs
                            )

                        preds.append(samples.cpu().float().numpy())

                        pbar.update(cond_shape[0])

        stacked_preds = np.vstack(preds)
        preds_ds = self.build_rescaled_ds(dims, stacked_preds)

        return preds_ds

    def get_dims(self, coords):
        """Extract lat lon and time dimensions from the evaluation dataset.
        """
        dataset = self.eval_dl.dataset
        if hasattr(dataset, "ds"):
            times = dataset.ds.time.values
            if coords is None:
                lat = dataset.ds.lat
                lon = dataset.ds.lon
            else:
                lat, lon = coords
            return times, lat, lon

        if hasattr(dataset, "_ensure_open"):
            dataset._ensure_open()

        times = getattr(dataset, "time_values", None)
        if times is None:
            raise AttributeError(
                "Evaluation dataset does not expose time values. "
                "Expected a .ds.time array or a time_values attribute."
            )
        if coords is None:
            lat = getattr(dataset, "lat_values", None)
            lon = getattr(dataset, "lon_values", None)
            if lat is None or lon is None:
                raise AttributeError(
                    "Evaluation dataset does not expose lat/lon values. "
                    "Expected .ds.lat/.ds.lon or lat_values/lon_values attributes."
                )
        else:
            lat, lon = coords
        return times,lat,lon

    def build_rescaled_ds(self, dims, stacked_preds):
        """Rescale predictions before packing them in an xarray.

        :param dims: tuple, (times, lat, lon) dimensions of the evaluation dataset.
        :param dims: np.ndarray, stacked predictions.

        :return: xr.Dataset, rescaled predictions.
        """
        times, lat, lon = dims
        # standard inference processes the whole field at once
        rescaled_preds = self.rescale_outputs(stacked_preds)
        var_dims = ["time", "lat", "lon"]
        preds_ds = xr.Dataset(
            data_vars={
                variable: (var_dims, rescaled_preds[variable])
                for variable in self.output_variables
            },
            coords={"time": times, "lat": lat, "lon": lon},
        )

            
        return preds_ds

    def rescale_outputs(self, preds: np.ndarray):
        """Rescale outputs back to true data units using the data_scaler.

        All predictions are done in scaled units; we rescale back to true
        data units before saving them to file.

        :param preds: np.ndarray, model predictions.
        :return: Dict, rescaled outputs.
        """
        scaled_outputs = {}
        target_transforms = self._get_target_transforms()
        if target_transforms:
            for var_idx, variable in enumerate(self.output_variables):
                transform = target_transforms.get(variable)
                if transform is None:
                    if hasattr(self.data_scaler, "variable_scaler_map") and (
                        variable in self.data_scaler.variable_scaler_map
                    ):
                        scaled_outputs[variable] = self.data_scaler.variable_scaler_map[
                            variable
                        ].inverse_transform(preds[:, var_idx])
                    else:
                        scaled_outputs[variable] = preds[:, var_idx]
                else:
                    scaled_outputs[variable] = self._invert_with_transform(
                        transform, variable, preds[:, var_idx]
                    )
            return scaled_outputs
        for var_idx, variable in enumerate(self.output_variables):
            scaled_outputs[variable] = self.data_scaler.variable_scaler_map[
                variable
            ].inverse_transform(preds[:, var_idx])
        return scaled_outputs

    def format_output_dir_name(self, config):

        # combine all configuration dicts
        combined_dict = reduce(lambda x, y: {**x, **y}, config, {}) if config else {}
        return self.output_string_format.format(**combined_dict)

    def save_config(self, config, output_path):
        with open(output_path / "sampling_config.json", "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)

    def _get_target_transforms(self):
        """Extract target transforms from the evaluation dataloader if available."""
        collate_fn = getattr(self.eval_dl, "collate_fn", None)
        if collate_fn is None:
            return None
        target_transforms = getattr(collate_fn, "target_transforms", None)
        if isinstance(target_transforms, dict) and len(target_transforms) > 0:
            return target_transforms
        return None

    def _invert_with_transform(self, transform, variable, arr: np.ndarray) -> np.ndarray:
        """Safely apply a transform's invert method to a prediction slice."""
        if transform is None:
            return arr

        candidates = [arr]
        if arr.ndim == 3:
            candidates.append(arr[:, None, ...])

        for candidate in candidates:
            try:
                if hasattr(transform, "invert") and callable(transform.invert):
                    inverted = transform.invert(candidate)
                elif hasattr(transform, "inverse_transform") and callable(
                    transform.inverse_transform
                ):
                    inverted = transform.inverse_transform(candidate)
                else:
                    continue
            except Exception:
                continue

            if isinstance(inverted, dict):
                if variable in inverted:
                    inverted = inverted[variable]
                elif len(inverted) > 0:
                    inverted = next(iter(inverted.values()))

            inverted = (
                inverted.cpu().numpy() if hasattr(inverted, "cpu") else np.asarray(inverted)
            )
            if inverted.ndim == 4 and inverted.shape[1] == 1:
                inverted = np.squeeze(inverted, axis=1)
            return inverted.astype(np.float32)

        return arr
