"""Grid search sampling parameters and score with CRPS."""
import sys
sys.dont_write_bytecode = True

import argparse
import importlib
import logging
import os
import re
from functools import reduce
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import warnings
import xarray as xr
import xskillscore as xs
from dask.diagnostics import ProgressBar
from dotenv import load_dotenv, dotenv_values, find_dotenv
from tqdm import tqdm

from src.diffusion_downscaling.data.scaling import DataScaler
from src.diffusion_downscaling.lightning import utils as lightning_utils
from src.diffusion_downscaling.lightning.utils import (
    build_model,
    build_or_load_data_scaler,
    configure_location_args,
    setup_custom_training_coords,
)
from src.diffusion_downscaling.sampling.sampling import Sampler
from src.diffusion_downscaling.sampling.utils import create_sampling_configurations
from src.diffusion_downscaling.data.data_loading import select_custom_coordinates
from src.diffusion_downscaling.evaluation.utils import (
    _open_dataset,
    prepare_prediction_data,
)

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")

sys.path.append(os.getcwd())


def parse_module(path: str) -> str:
    return path.replace(".py", "").replace("/", ".")


def round_time(da: xr.DataArray, freq: str = "s") -> xr.DataArray:
    if "time" not in da.coords:
        return da
    time_values = da["time"].values
    if np.issubdtype(time_values.dtype, np.datetime64):
        rounded = pd.to_datetime(time_values).round(freq)
        return da.assign_coords(time=("time", rounded))
    return da


def round_lat_lon(da: xr.DataArray, decimals: int = 6) -> xr.DataArray:
    for coord in ("lat", "lon"):
        if coord in da.coords:
            da = da.assign_coords({coord: np.round(da[coord].values, decimals)})
    return da


def compute_crps(
    simulation_da: xr.DataArray,
    ground_truth_da: xr.DataArray,
    member_dim: str = "member",
    time_dim: str = "time",
    spatial_dims: Sequence[str] = ("lat", "lon"),
    use_dask: bool = True,
    rechunk_members: bool = True,
    verbose: bool = True,
):
    """
    Updated compute_crps: deterministic-case uses (climatology_da - obs).abs() instead of xr.apply_ufunc(...)
    Returns dict with: crps_time, crps_space, overall_crps, fig_time, fig_space
    """

    def _is_dask_backed(x: xr.DataArray) -> bool:
        try:
            import dask.array as da_mod  # type: ignore
            return isinstance(x.data, da_mod.Array)
        except Exception:
            return False

    def _ensure_time_and_space_in_obs(obs_x: xr.DataArray):
        for d in (time_dim, *spatial_dims):
            if d not in obs_x.dims:
                raise ValueError(f"Dimension '{d}' not found in ground_truth_da.dims: {obs_x.dims}")

    def _xs_crps_safe(obs_x, sim_x, member_dim_local, dim):
        try:
            return xs.crps_ensemble(obs_x, sim_x, member_dim=member_dim_local, dim=dim)
        except ValueError as e:
            txt = str(e)
            if "consists of multiple chunks" in txt and rechunk_members and _is_dask_backed(sim_x):
                if verbose:
                    warnings.warn(
                        "apply_ufunc complained about multiple chunks on the core dimension --- rechunking members and retrying."
                    )
                sim_x = sim_x.chunk({member_dim_local: -1})
                return xs.crps_ensemble(obs_x, sim_x, member_dim=member_dim_local, dim=dim)
            raise

    simulation_da = round_time(round_lat_lon(simulation_da))
    ground_truth_da = round_time(round_lat_lon(ground_truth_da))

    _ensure_time_and_space_in_obs(ground_truth_da)

    # align
    sim, obs = xr.align(simulation_da, ground_truth_da, join="inner")
    print("_______________________________")
    print(" >> inside_compute_crps: sim", sim)
    print("_______________________________")
    print(" >> inside_compute_crps: obs", obs)
    print("_______________________________")

    # defensive: check non-empty after align
    if time_dim not in sim.dims or sim.sizes.get(time_dim, 0) == 0:
        raise ValueError("After alignment, time dimension is empty. Check coordinates/overlap between sim and obs.")

    member_present = member_dim in sim.dims
    n_members = int(sim.sizes.get(member_dim, 0)) if member_present else None

    deterministic_case = (not member_present) or (n_members == 1)
    zero_members_case = member_present and (n_members == 0)

    if zero_members_case:
        warnings.warn(
            f"Ensemble dimension '{member_dim}' exists but has length 0. Returning NaN reductions."
        )
        crps_time = xr.DataArray(
            np.full(sim.sizes[time_dim], np.nan),
            coords={time_dim: sim[time_dim]},
            dims=(time_dim,),
        )
        crps_space = xr.DataArray(
            np.full((sim.sizes[spatial_dims[0]], sim.sizes[spatial_dims[1]]), np.nan),
            coords={spatial_dims[0]: sim[spatial_dims[0]], spatial_dims[1]: sim[spatial_dims[1]]},
            dims=spatial_dims,
        )
        overall_crps_val = float("nan")

    elif deterministic_case:
        if verbose:
            if not member_present:
                warnings.warn(
                    f"Ensemble dimension '{member_dim}' not found: treating simulation as deterministic forecast (CRPS = |f - o|)."
                )
            else:
                warnings.warn(
                    f"Ensemble dimension '{member_dim}' has length {n_members}: treating forecast as deterministic (CRPS = |f - o|)."
                )

        # ---------- FIX: use xarray arithmetic (dask-safe) instead of xr.apply_ufunc(...)
        # elementwise absolute error (CRPS for deterministic forecast)
        abs_err = xr.ufuncs.abs(sim - obs)

        # reductions (still lazy if inputs are dask-backed)
        crps_time = abs_err.mean(dim=list(spatial_dims))
        crps_space = abs_err.mean(dim=time_dim)

        if use_dask and (_is_dask_backed(sim) or _is_dask_backed(obs)):
            overall_crps_val = float(abs_err.mean(dim=[time_dim, *list(spatial_dims)]).compute().item())
        else:
            overall_da = abs_err.mean(dim=[time_dim, *list(spatial_dims)])
            overall_crps_val = float(overall_da.item()) if hasattr(overall_da, "item") else float(overall_da.values)

    else:
        # ensemble case: keep original logic for xskillscore with safe rechunk
        if use_dask and _is_dask_backed(sim) and rechunk_members:
            try:
                axis = tuple(sim.dims).index(member_dim)
                chunks = sim.data.chunks
                member_chunks = chunks[axis]
                if len(member_chunks) > 1:
                    if verbose:
                        warnings.warn(
                            f"Detected {len(member_chunks)} chunks along '{member_dim}'. Rechunking that dim into a single chunk."
                        )
                    sim = sim.chunk({member_dim: -1})
            except Exception:
                if verbose:
                    warnings.warn(
                        f"Could not inspect dask chunks for '{member_dim}', attempting a safe rechunk."
                    )
                sim = sim.chunk({member_dim: -1})

        crps_time = _xs_crps_safe(obs, sim, member_dim, dim=list(spatial_dims))
        crps_space = _xs_crps_safe(obs, sim, member_dim, dim=time_dim)
        overall_da = _xs_crps_safe(obs, sim, member_dim, dim=[time_dim, *list(spatial_dims)])

        try:
            if use_dask:
                with ProgressBar():
                    crps_time = crps_time.compute()
                    crps_space = crps_space.compute()
                    overall_crps_val = float(overall_da.compute().item())
            else:
                if _is_dask_backed(sim) or _is_dask_backed(obs):
                    if verbose:
                        warnings.warn(
                            "Converting input arrays to memory (compute) because use_dask=False."
                        )
                    sim = sim.compute()
                    obs = obs.compute()
                    crps_time = xs.crps_ensemble(obs, sim, member_dim=member_dim, dim=list(spatial_dims))
                    crps_space = xs.crps_ensemble(obs, sim, member_dim=member_dim, dim=time_dim)
                    overall_crps_val = float(
                        xs.crps_ensemble(obs, sim, member_dim=member_dim, dim=[time_dim, *list(spatial_dims)]).item()
                    )
                else:
                    overall_crps_val = float(overall_da.item()) if hasattr(overall_da, "item") else float(overall_da.values)
        except Exception as e:
            warnings.warn(
                f"Encountered error when computing reduced outputs: {e}. Attempting .compute() fallback."
            )
            with ProgressBar():
                crps_time = crps_time.compute()
                crps_space = crps_space.compute()
                overall_crps_val = float(overall_da.compute().item())

    return crps_time, crps_space, overall_crps_val


def _extract_checkpoint_epoch(checkpoint_name: str) -> str:
    checkpoint_base = os.path.basename(checkpoint_name)
    match = re.search(r"epoch=(\d+)", checkpoint_base)
    if match:
        return f"epoch={match.group(1)}"
    return Path(checkpoint_base).stem


def _merge_config_dicts(config_tuple):
    return reduce(lambda x, y: {**x, **y}, config_tuple, {}) if config_tuple else {}


def run_grid_search(config, sampling_config):
    load_dotenv()
    config_ = dotenv_values(find_dotenv(usecwd=True))

    config.data.eval_indices = sampling_config.eval_indices
    output_variables = config.data.variables[1]

    use_josh_pipeline = getattr(config.data, "use_josh_pipeline", False)

    data_path = Path(config.data.dataset_path) if config.data.dataset_path else None
    location_config = dict(sampling_config.eval).get("location_config")
    if not hasattr(config.model, "location_parameter_config"):
        config.model.location_parameter_config = None

    custom_dset = dict(sampling_config).get("eval_dataset")
    if custom_dset is not None:
        data_path = Path(custom_dset)
        config.data.dataset_path = str(data_path)
        if use_josh_pipeline:
            config.data.filename = str(data_path)
            config.data.val_filename = str(data_path)

    eval_filename = sampling_config.eval_dataset.rsplit("/", 1)[-1].split(".", 1)[0]
    checkpoint_epoch = _extract_checkpoint_epoch(sampling_config.eval.checkpoint_name)
    output_dir = os.path.join(
        config_["WORK_DIR"],
        "samples",
        config.data.dataset_name,
        eval_filename,
        config.run_name,
        checkpoint_epoch,
    )

    if use_josh_pipeline:
        data_scaler = DataScaler({})
    else:
        if data_path is None:
            data_path = Path(config.data.dataset_path)
        config = configure_location_args(config, data_path)
        data_scaler_path = sampling_config.get("data_scaler_path") or Path(output_dir) / "scaler_parameters.pkl"
        data_scaler = build_or_load_data_scaler(config, data_scaler_path)

    eval_config = sampling_config.eval
    checkpoint_name = eval_config.checkpoint_name
    logger.info(" >> >> INSIDE sampling_grid_search | checkpoint_name %s", checkpoint_name)
    print(f" >> >> INSIDE sampling_grid_search | checkpoint_name {checkpoint_name}")

    model = build_model(config, checkpoint_name)

    config.training.batch_size = sampling_config.batch_size
    config = setup_custom_training_coords(config, sampling_config)

    evaluation_sampler = Sampler(
        model,
        config.model_type,
        data_scaler,
        output_variables,
        sampling_config.output_format,
    )

    num_samples = eval_config.n_samples

    sampling_config.sampling.schedule.device = str(config.device)
    eval_args = create_sampling_configurations(sampling_config.sampling, location_config)

    xr_data = _open_dataset(config.data.dataset_path)
    buffer_width = config.training.loss_buffer_width

    sampling_args, all_configs = eval_args

    results = []
    for config_tuple in tqdm(all_configs, desc="Grid-search configs", unit="config"):
        combined_config = _merge_config_dicts(config_tuple)
        logger.info("Grid-search config: %s", combined_config)
        print(f"\n==> Grid-search config: {combined_config}")
        location_config = config_tuple[2]["location_config"]
        coords = select_custom_coordinates(xr_data, location_config, buffer_width)
        config.data.location_config = location_config
        logger.info("Location config: %s", location_config)
        print(f" >> Location config: {location_config}")
        _, evaluation_sampler.eval_dl = lightning_utils.build_dataloaders(
            config, data_scaler.transform, num_workers=20
        )

        output_string = evaluation_sampler.format_output_dir_name(config_tuple)
        output_path, predictions_dir, _ = evaluation_sampler.setup_output_dirs(
            output_dir, output_string
        )
        evaluation_sampler.save_config(config_tuple, output_path)
        logger.info("Output path: %s", output_path)
        print(f" >> Output path: {output_path}")

        print(f"Beginning predictions on {output_string}", flush=True)
        logger.info("Beginning predictions on %s", output_string)
        evaluation_sampler.generate_predictions(
            config_tuple,
            coords,
            num_samples,
            predictions_dir,
            sampling_args,
            output_variables,
        )
        print(f"{num_samples} predictions on {output_string} completed.", flush=True)
        logger.info("%s predictions on %s completed.", num_samples, output_string)
        print(" >> Preparing prediction data for CRPS evaluation.")

        sample_xrs, eval_data = prepare_prediction_data(
            data_path,
            predictions_dir,
            sampling_config.eval_indices,
            coords,
            buffer_width,
        )
        logger.info("Prepared predictions and eval data for CRPS evaluation.")

        used_output_variables = output_variables
        if getattr(config, "residuals", False):
            used_output_variables = []
            for output_variable in output_variables:
                raw_variable = "_".join(output_variable.split("_")[1:])
                reconstituted_preds = sample_xrs[output_variable] + eval_data["regression_" + raw_variable]
                sample_xrs[raw_variable] = reconstituted_preds
                used_output_variables.append(raw_variable)

        variable_scores = {}
        overall_scores = []
        for output_variable in used_output_variables:
            logger.info("Computing CRPS for variable: %s", output_variable)
            print(f" >> Computing CRPS for variable: {output_variable}")
            crps_time, crps_space, overall_crps_val = compute_crps(
                sample_xrs[output_variable],
                eval_data[output_variable],
                member_dim="member",
            )
            logger.info("CRPS overall for %s: %s", output_variable, overall_crps_val)
            print(f" >> CRPS overall for {output_variable}: {overall_crps_val}")
            variable_scores[output_variable] = {
                "overall_crps": overall_crps_val,
                "crps_time_mean": float(crps_time.mean().item()),
                "crps_space_mean": float(crps_space.mean().item()),
            }
            overall_scores.append(overall_crps_val)

        mean_overall_crps = float(np.mean(overall_scores)) if overall_scores else float("nan")
        logger.info("Mean overall CRPS: %s", mean_overall_crps)
        print(f" >> Mean overall CRPS: {mean_overall_crps}")
        result_entry = {
            "output_dir": str(output_path),
            "config": combined_config,
            "per_variable": variable_scores,
            "mean_overall_crps": mean_overall_crps,
        }
        results.append(result_entry)

        results_path = Path(output_dir) / "grid_search_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as handle:
            import json

            json.dump(results, handle, indent=2)

    if results:
        best_result = min(results, key=lambda x: x["mean_overall_crps"])
        print("Best config:", best_result)


def main(config_path: str, sampling_config_path: str):
    config_module = importlib.import_module(parse_module(config_path))
    config = config_module.get_config()

    sampling_config_module = importlib.import_module(parse_module(sampling_config_path))
    sampling_config = sampling_config_module.get_sampling_config()

    print("Loaded configuration files.", flush=True)

    run_grid_search(config, sampling_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search sampling parameters with CRPS evaluation"
    )
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        default="training/configs/configs/gan.py",
        help="Path to the config file",
    )
    parser.add_argument(
        "-s",
        "--sampling-path",
        type=str,
        default="training/configs/configs/gan.py",
        help="Path to the sampling config file",
    )
    args = parser.parse_args()

    main(args.config_path, args.sampling_path)
