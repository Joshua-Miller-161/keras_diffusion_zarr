"""High-level utilities to load predictions and observations, and running evaluation.
"""
import numpy as np
import xarray as xr
import torch
import pandas as pd
from pprint import pprint
from pathlib import Path

from .eval import Evaluation
from .plotting import EvaluationPlotter
from .metrics import Metrics
from .consts import HIST_LIMITS


def _open_dataset(path: Path):
    path = Path(path)
    if path.suffix == ".zarr" or path.is_dir():
        try:
            return xr.open_zarr(path, consolidated=True)
        except (KeyError, ValueError, OSError):
            return xr.open_zarr(path)
    return xr.open_dataset(path)

def remove_leap_days(dates_list):

    dates_pd = pd.to_datetime(dates_list)
    mask = (dates_pd.month == 2) & (dates_pd.day == 29)

    # Apply mask to filter out leap years
    dates_pd_masked = dates_pd[~mask]

    # Convert back to numpy datetime64
    dates_np_masked = np.array(dates_pd_masked, dtype="datetime64[D]")
    return dates_np_masked


def build_evaluation_callable(
    data_path, eval_indices, metrics_config, output_variables, predictions_only, residuals
):
    """Builds a callable that can be used to evaluate a model in the Sampler.

    :param data_path: Path, the path to the observation data file.
    :param eval_indices: dict, the indices to evaluate the model on.
    :param metrics_config: tuple, metrics configurations to compute.
    :param output_variables: list, the output variables to evaluate.
    :param predictions_only: bool, whether to only generate predictions. If True,
        evaluation is skipped completely. 
    
    :return evaluation_callable: callable, the evaluation function.

    """
    def evaluation_callable(
        predictions_path, results_output_path, coords, buffer_width
    ):
        """Callable for computing metrics and generating output plots.

        :param predictions_path: Path, directory containing prediction sample files.
        :param results_output_path: Path, directory to save evaluation results.
        :param coords: tuple, the coordinates to evaluate the model on.
        :param buffer_width: int, the buffer width to apply to the evaluation data.
            This will trim the output fields if used.
        
        :return metrics: dict, dictionary containing quantitative metrics to be saved.
        """
        if predictions_only:
            return
        metrics = {}
        sample_xrs, eval_data = prepare_prediction_data(
            data_path, predictions_path, eval_indices, coords, buffer_width
        )
        used_output_variables = output_variables
        if residuals:
            used_output_variables = []
            for output_variable in output_variables:
                raw_variable = '_'.join(output_variable.split('_')[1:])
                reconstituted_preds = sample_xrs[output_variable] + eval_data['regression_' + raw_variable]
                sample_xrs[raw_variable] = reconstituted_preds
                used_output_variables.append(raw_variable)
        if len(output_variables) > 1:
            plotter = EvaluationPlotter(results_output_path, used_output_variables[0])
            plotter.plot_multivariate_timelapse(
                sample_xrs,
                eval_data,
                used_output_variables,
                num_plots=5,
                num_days=5,
                num_samples=4,
            )
        for output_variable in used_output_variables:
            metrics[output_variable] = run_variable_evaluation(
                sample_xrs,
                eval_data,
                metrics_config,
                output_variable,
                results_output_path,
            )
        return metrics

    return evaluation_callable


def run_variable_evaluation(
    sample_xrs, eval_data, metrics_config, output_variable, results_output_path
):
    """Generate and save plots, compute metrics given predictions, observations, and some extra config.

    :param sample_xrs: xr.Dataset, the samples xarray with dims (member, time, lat, lon)
        and data_vars for each outout variable.
    :param eval_data: xr.Dataset, the evaluation data xarray with dims (time, lat, lon)
        and data_vars for each outout variable.
    :param metrics_config: tuple, metrics configurations to compute.
    :param output_variable: str, the output variable to evaluate.
    :param results_output_path: Path, the directory to save evaluation results.

    :return metrics: dict, dictionary containing quantitative metrics to be saved.
    """

    generate_all_plots(sample_xrs, eval_data, results_output_path, output_variable)
    
    observation_array = eval_data[output_variable].values
    sample_arrays = sample_xrs[output_variable].values.transpose(1, 0, 2, 3)
    metrics = compute_and_save_metrics(
        sample_arrays,
        observation_array,
        *metrics_config,
        output_variable,
        results_output_path,
    )
    return metrics


def prepare_eval_data(data_path, eval_data_split, coords=None):
    """Load in evaluation data given a data path, the data split and coordinates.
    """
    xarray_original = _open_dataset(data_path)
    indices = slice(
        *(np.array(eval_data_split["split"]) * len(xarray_original.time)).astype(int)
    )
    eval_data = xarray_original.isel(time=indices)
    eval_data = eval_data.drop_duplicates("time")
    if eval_data_split.get("date_range") is not None:
        year_indices = xr.date_range(*eval_data_split["date_range"])
        year_indices = remove_leap_days(year_indices)
        eval_data = eval_data.sel(time=year_indices)
    if coords is not None:
        eval_data = eval_data.sel(lon=coords[1], lat=coords[0])
    return eval_data


def get_predictions(predictions_path : Path):
    """Loads in samples as an xarray given prediction path.

    :param predictions_path: Path, the directory containing the prediction xarrays

    :return sample_xrs: xr.Dataset, samples xarray with dims (member, time, lat, lon)
        and data_vars for each outout variable.
    """
    predictions = list((predictions_path).glob("*.nc"))
    sample_xrs = [xr.open_dataset(p) for p in predictions]

    sample_xrs = xr.concat(sample_xrs, "member")
    return sample_xrs


def prepare_prediction_data(
    data_path, predictions_path, eval_data_split, coords, buffer_width = None,
):
    """Load in evaluation data and predictions, align them and return.

    :param data_path: Path, the path to the observation data file.
    :param predictions_path: Path, directory containing prediction sample files.
    :param eval_data_split: dict, the indices and dates to evaluate the model on.
    :param coords: tuple, the coordinates to evaluate the model on.
    :param buffer_width: int, the buffer width to apply to the evaluation data.

    :return sample_xrs: xr.Dataset, samples xarray with dims (member, time, lat, lon)
    :return eval_data: xr.Dataset, the evaluation data xarray with dims (time, lat, lon)
    """

    sample_xrs = get_predictions(predictions_path)
    eval_data = prepare_eval_data(data_path, eval_data_split, coords)
    if buffer_width is not None:
        sample_xrs = sample_xrs.isel(
            lat=slice(buffer_width, -buffer_width),
            lon=slice(buffer_width, -buffer_width),
        )
        eval_data = eval_data.isel(
            lat=slice(buffer_width, -buffer_width),
            lon=slice(buffer_width, -buffer_width),
        )
    # xr.align may be risky - maybe come up with a better way to ensure coord
    # consistency. We do this because different data types/floating point precision
    # leads to unaligned xarrays that cannot be subtracted etc.
    try:
        sample_xrs, eval_data = xr.align(sample_xrs, eval_data, join="override")
    except ValueError:
        sample_xrs, eval_data = xr.align(sample_xrs, eval_data, join="left")
    return sample_xrs, eval_data



def compute_and_save_metrics(
    sample_arrays,
    observation_array,
    spatial_aggregations,
    simple_metrics,
    fss_configs,
    output_variable,
    output_path,
):
    """Compute specified metrics given predictions and observations.
    
    :param sample_arrays: np.ndarray, the samples array with dims (time, member, lat, lon)
    :param observation_array: np.ndarray, the observation array with dims (time, lat, lon)
    :param spatial_aggregations: dict, the spatial aggregations to compute.
    :param simple_metrics: list, the metric types (mse, bias, crps) and spatial aggregation scales.
    :param fss_configs: list, the fss configurations to compute.
    :param output_variable: str, the output variable to evaluate.
    :param output_path: Path, the directory to save evaluation results.

    :return metrics: dict, dictionary containing quantitative metrics to be saved.
    """
    evaluator = Evaluation(spatial_aggregations, simple_metrics, fss_configs)
    with torch.no_grad():
        metrics = evaluator.compute_metrics(
            sample_arrays, observation_array, output_variable
        )
    metrics_red = {
        key: value for key, value in metrics.items() if "pixelwise" not in key
    }

    output_path.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(metrics["pixelwise"])
    df.to_csv(output_path / f"{output_variable}_mse_metrics.csv", sep="\t")
    with open(output_path / f"{output_variable}_metrics.txt", "w") as outfile:
        pprint(metrics_red, outfile)

    return metrics

def generate_all_plots(sample_xrs, eval_data, output_path, output_variable):
    """
    Generate all plots for a given output variable.

    Long list of plots to generate, including:
    - Rank histograms
    - Empirical CDFs
    - Various spatial bias plots
    - Radially average log spectral density plots
    - Histogram comparisons
    - Bulk property scatter plots

    """
    observation_array = eval_data[output_variable].values
    sample_arrays = sample_xrs[output_variable].values.transpose(1, 0, 2, 3)

    output_path = output_path / output_variable

    grid_space_units = abs((eval_data.lon.values[1] - eval_data.lon.values[0]) * 111.3)
    psd_true, psd_samples, freqs = Metrics.compute_radially_averaged_spec_density(
        torch.Tensor(observation_array), torch.Tensor(sample_arrays)
    )
    mean_psd_observations = np.mean(psd_true, axis=1)
    mean_psd_samples = np.mean(psd_samples, axis=1)
    spectra_observations = {
        "Observations": 10 * np.log10(mean_psd_observations),
        "Mean over samples": 10 * np.log10(mean_psd_samples),
    }

    plotter = EvaluationPlotter(output_path / "hists", output_variable)

    for thresholds in HIST_LIMITS[output_variable]:

        name = f"{output_variable}_{thresholds[0]}_thresh"
        try:
            hist = Metrics.compute_rank_histogram(
                observation_array, sample_arrays.transpose(1, 0, 2, 3), thresholds
            )
            plotter.plot_rank_histogram(hist, name=name)
            plotter.plot_empirical_cdf(hist, name=name)
        except:
            pass

    plotter = EvaluationPlotter(output_path, output_variable)

    plotter.plot_mean_spatial_bias(sample_xrs.mean(dim="time"), eval_data)
    plotter.plot_amax_spatial_bias(sample_xrs, eval_data, agg_type="mean")
    plotter.plot_amax_spatial_bias(sample_xrs, eval_data, agg_type="max")

    plotter.plot_power_spectra(spectra_observations, grid_space_units * 1 / freqs)
    plotter.plot_amax_spatial_sigmas(sample_xrs, eval_data)

    mean_precip_obs = np.mean(observation_array, axis=(1, 2))
    mean_precip_samples = np.mean(
        sample_arrays.reshape((-1,) + sample_arrays.shape[-2:]), axis=(1, 2)
    )
    hist_logx = output_variable == "precipitation"
    all_flattened = sample_arrays.flatten()
    if hist_logx:
        all_flattened = all_flattened.clip(0)
    plotter.plot_histogram_comparison(
        mean_precip_samples, mean_precip_obs, f"mean_{output_variable}", hist_logx
    )
    plotter.plot_histogram_comparison(
        all_flattened, observation_array.flatten(), f"all_{output_variable}", hist_logx
    )

    scatter_mean_precip_obs = np.repeat(mean_precip_obs, sample_arrays.shape[1])
    scatter_mean_precip_samples = mean_precip_samples
    plotter.scatter_plot(scatter_mean_precip_samples, scatter_mean_precip_obs)

    plotter.plot_samples(sample_xrs, eval_data, num_plots=10, num_samples=4)
    plotter.plot_timelapse(
        sample_xrs, eval_data, num_plots=5, num_days=5, num_samples=4
    )

    evaluator = Evaluation()
    amax_predictions, amax_top_observations = evaluator.compute_amax_stats(
        sample_xrs[output_variable], eval_data[output_variable]
    )
    try:
        plotter.plot_annual_maximum_distributions(
            amax_predictions, amax_top_observations
        )
    except:  
        # only bit of the code that breaks if we manually duplicate samples (KDE linalg error)
        # in order to test the plotting.
        pass
