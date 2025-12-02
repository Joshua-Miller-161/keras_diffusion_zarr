# Data Josh Module Reference

Summaries of functions and classes in `src/diffusion_downscaling/data_Josh`.

## `custom_collate.py`
- **FastCollate**: Batches `(condition, target, time)` samples that may arrive as dictionaries or stacked arrays, optionally applies per-variable input/target transforms safely, appends encoded time channels, and returns tensors with the batched timestamps. Handles variable ordering overrides and transform fallbacks.

## `data_module.py`
- **_worker_init_fn**: Worker initializer that limits math library threads to avoid oversubscription.
- **LightningDataModule**: PyTorch Lightning data module that configures training/validation/test datasets from Zarr files, builds appropriate transforms, and constructs data loaders with configurable worker, shuffling, and prefetch settings.

## `dataset.py`
- **DownscalingDataset**: Lazy-opening dataset that reads predictor and target variables from consolidated Zarr arrays, decodes time metadata when present, and returns dictionaries of NumPy arrays with associated timestamps. Includes helpers for converting arrays to tensors and encoding timestamps into normalized seasonal channels.

## `get_xr_dataset.py`
- **get_xr_dataset**: Builds or loads per-variable transforms from configuration, resolves the active dataset path, and returns the Zarr path plus the input/target transform dictionaries without materializing the data.

## `transforms.py`
- **save_transform / load_transform**: Persist and reload transform objects via pickle with logging.
- **_is_numpy_array / _array_channel_info / _stack_param_dict_to_array / _param_broadcast_for_arr**: Utility helpers for validating NumPy arrays, inferring channel layouts, stacking parameter dictionaries, and broadcasting parameters to match array shapes.
- **_build_transform / _find_or_create_transforms**: Create dataset-wide transforms (optionally cached on disk) by fitting builders against active and model-source Zarr data.
- **_build_transform_per_variable_from_config / _find_or_create_transforms_per_variable_from_config**: Fit or load one transform per variable based on configuration-specified keys, persisting them individually.
- **register_transform / get_transform**: Registry utilities for naming and retrieving transform classes.
- **_ensure_numpy_dict / _close_dataset_if_possible / _dim_index_map_for_ndim / _axes_for_dims / _maybe_reduce**: Helpers for converting datasets to NumPy dictionaries, closing datasets, mapping dimension names to axes, selecting axes, and preparing reduction instructions.
- **CropT**: Crops arrays or dict values to a square of the configured size.
- **Standardize**: Fits mean and std per variable and standardizes data with an invert method.
- **PixelStandardize**: Standardizes each pixel (optionally over time) per variable with pixel-wise means/stds.
- **NoopT**: Pass-through transform returning data unchanged with optional invert symmetry.
- **PixelMatchModelSrcStandardize**: Standardizes target data to match model-source pixel statistics and global normalization.
- **MinMax**: Scales variables to the `[0, 1]` range and can invert back to the original scale.
- **UnitRangeT**: Normalizes using per-variable maxima and supports inversion to original scale.
- **ClipT**: Identity transform with an invert step that clips values to zero minimum.
- **PercentToPropT**: Converts percentage values to proportions (and back) per variable.
- **RecentreT**: Recenters values from `[0, 1]` to `[-1, 1]` (and invertible).
- **SqrtT / RootT / RawMomentT**: Apply root-based rescaling (square, arbitrary root, or raw-moment normalization) with invert methods.
- **LogT**: Applies `log1p` transformation with invert via `expm1`.
- **ComposeT**: Chains multiple transforms with forward and inverse application order.
- **build_input_transform / build_target_transform**: Factory functions mapping string keys to composed transform pipelines for predictors and predictands, respectively.

## `utils.py`
- **is_main_process**: Determines if the current process is rank zero for logging.
- **dataset_path / datafile_path**: Resolve dataset directory and specific file paths using configurable base directories.
- **open_zarr**: Opens a consolidated Zarr dataset with logging and fallbacks.
- **build_DataLoader**: Convenience wrapper to create a DataLoader with `DownscalingDataset` and custom collate behavior.
- **get_variables / get_variables_per_var**: Read predictor/target variable lists from configuration or provided config object.
- **_build_transform / _find_or_create_transforms**: Fit or load dataset-level input/target transforms, optionally caching them.
- **generate_output_filepath**: Create an incremented NetCDF output path based on existing files in a directory.
- **TIME_RANGE**: Default datetime bounds used for time feature encoding.
- **custom_collate**: Collates batches of (input, target, time) into stacked tensors and concatenated timestamps.
- **_get_zarr_length**: Returns the length of a Zarr dataset along the time dimension.
- **_parse_cf_time_units / decode_zarr_time_array**: Parse CF-compliant time units and decode Zarr time arrays into datetime representations.
- **input_to_list**: Normalizes variable identifiers into a flat string list.
