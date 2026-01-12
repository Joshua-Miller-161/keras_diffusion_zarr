import ml_collections


def get_sampling_config():

    config = ml_collections.ConfigDict()
    config.batch_size = 32
    # config.output_format = "n_20_churn_10_snoise_1.005_rho_3.5_fixed"
    config.output_format = "n_{n}_churn_{s_churn}_snoise_{s_noise}_smin{s_tmin}_rho_{rho}"
    config.eval_indices = {
        "split": [0, 1],
    }

    # config.eval_indices = {
    #     "split": [0, 1],
    #     # "date_range": ("2012-01-01", "2013-12-31"),
    #     # "date_range": ("2015-10-01", "2015-10-30"),
    #     "date_range": ("2015-01-01", "2018-12-31"),
    # }

    config.eval = ml_collections.ConfigDict()
    eval = config.eval
    eval.eval_output_dir = "diff"
    config.eval_dataset = "/gang/gang/j_miller/data/zarr/val_consolidated_time1_elev.zarr"

    eval.checkpoint_name = "/gang/gang/j_miller/temp/checkpoints/zarr/josh_zarr_example/epoch=4-val_loss=0.0204.ckpt" # diffusion_

    eval.n_samples = 5
    # eval.location_config = 'colorado'

    config.sampling = ml_collections.ConfigDict()
    sampling = config.sampling

    sampling.sampler = ml_collections.ConfigDict()
    sampler = sampling.sampler
    sampler.integrator = "dpm2_heun"
    sampler.s_churn = 10
    sampler.s_noise = 1.005
    sampler.s_tmin = 0.04
    sampler.s_tmax = 50

    sampling.schedule = ml_collections.ConfigDict()
    schedule = sampling.schedule
    schedule.type = "karras"
    schedule.n = 20
    schedule.rho = 7
    schedule.sigma_min = 0.02
    schedule.sigma_max = 80

    return config
