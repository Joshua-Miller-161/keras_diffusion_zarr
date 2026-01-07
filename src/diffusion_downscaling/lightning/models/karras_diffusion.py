import torch.nn as nn
import torch
import logging

logger = logging.getLogger(__name__)

from ...sampling.k_sampling import append_dims


class AbstractKarrasDenoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models.
    
    This has two key methods, loss and forward, which provide the EDM
    training scheme for diffusion models and the preconditioning wrapper
    for the forward pass of the EDM model. Both of these are directly adapted
    from the EDM paper, following code from github.com/crowsonkb/k-diffusion.
    """

    def __init__(self, inner_model, device="cpu"):
        super().__init__()
        self.inner_model = inner_model
        self.register_buffer("ignore", torch.tensor([1.0], device=device))

        self.trim_output_fields = lambda x: x
        self.sigma_converter = lambda x: x

    def set_buffer_width(self, pixel_width):
        """Sets trim width of output fields during training.

        This is motivated by the problem of edge effects and the training boundary
        potentially biasing the model performance. 

        :param pixel_width: int, trim buffer around output fields in pixels. 
        """
        self.trim_output_fields = lambda x: x[
            :, :, slice(pixel_width, -pixel_width), slice(pixel_width, -pixel_width)
        ]

    def _weighting_soft_min_snr(self, sigma):
        """
        Alternative loss weighting scheme (not explored so far)/
        """
        return (sigma * self.sigma_data) ** 2 / (sigma**2 + self.sigma_data**2) ** 2

    def _weighting_snr(self, sigma):
        """
        Alternative loss weighting scheme (not explored so far)/
        """
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    @torch.compile
    def loss(self, input, condition, **kwargs):
        """Key function for training EDM-type diffusion models.

        :param input: Batch of noise-free target images, shape (bs, out_chan, H, W)
        :param condition: Corresponding of conditioning information, either a tensor of input fields
            or more generally a tuple containing the input fields and additional conditional info

        :returns loss: Tensor of losses per output-channel, dim (bs, out_chan)
        """
        sigma = self.sample_training_sigmas(input)
        c_skip, c_out, c_in = [
            append_dims(x, input.ndim) for x in self.get_scalings(sigma)
        ]
        c_weight = self.weighting(sigma)

        noised_input = input + torch.randn_like(input) * append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, condition, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out

        img_flattened = (
            (
                (
                    self.trim_output_fields(model_output)
                    - self.trim_output_fields(target)
                )
                ** 2
            )
            .flatten(2)
            .mean(2)
        )
        return (img_flattened * c_weight.unsqueeze(1)).mean(dim=0)

    @torch.compile
    def forward(self, input, cond, sigma, **kwargs):
        c_skip, c_out, c_in = [
            append_dims(x, input.ndim) for x in self.get_scalings(sigma)
        ]
        c_noise = self.sigma_converter(sigma)

        out = self.inner_model(input * c_in, cond, c_noise, **kwargs) * c_out + input * c_skip

        logger.info(f" >> >> INSIDE AbstractKarrasDenoiser forward {out.detach().cpu().numpy()}")
        # return (
        #     self.inner_model(input * c_in, cond, c_noise, **kwargs) * c_out
        #     + input * c_skip
        # )

        return out


class EDMDenoiser(AbstractKarrasDenoiser):
    """Implements the Karras denoiser parameters set out in the EDM paper.

    The only modification here is a custom training noise sampling scheme,
    which is more closely aligned with the specific task of pixel-level
    diffusion for downscaling. See sample_training_sigmas, which combines
    a uniform and normal distribution to focus training on the lower end
    of the noise-level spectrum.
    """
    def __init__(self, inner_model, sigma_data=1.0, weighting="karras", device="cpu"):
        super().__init__(inner_model, device)
        self.inner_model = inner_model
        self.register_buffer("sigma_data", torch.tensor([sigma_data], device=device))
        self.get_scalings = self.get_edm_scalings
        self.register_buffer("P_std", torch.tensor([0.8], device=device))
        self.register_buffer("P_mean", torch.tensor([-1.6], device=device))
        self.register_buffer("ignore", torch.tensor([1.0], device=device))

        self.register_buffer("log_uniform_u", torch.log(torch.tensor([80.1], device=device)))
        self.register_buffer("log_uniform_l", torch.log(torch.tensor([0.02], device=device)))

        if callable(weighting):
            self.weighting = weighting
        if weighting == "karras":
            self.weighting = torch.ones_like
        elif weighting == "soft-min-snr":
            self.weighting = self._weighting_soft_min_snr
        elif weighting == "snr":
            self.weighting = self._weighting_snr
        else:
            raise ValueError(f"Unknown weighting type {weighting}")

    def get_edm_scalings(self, sigma):
        """Scaling taken directly from the EDM paper.
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def sample_training_sigmas(self, input):
        """Custom noise level sampling during training.
        """
        rnd_normal = torch.randn([input.shape[0] // 2], device=input.device)
        sigma_1 = (rnd_normal * self.P_std + self.P_mean).exp()

        log_sigma = (self.log_uniform_u - self.log_uniform_l) * torch.rand(
            [input.shape[0] - sigma_1.shape[0]], device=input.device
        ) + self.log_uniform_l
        sigma_2 = log_sigma.exp()
        sigma = torch.cat((sigma_1, sigma_2))
        return sigma


class VPDenoiser(AbstractKarrasDenoiser):
    """Implements the Variance-Preserving (VP) SDE in the EDM formulation.

    This gives suboptimal results but we reimplement it to make comparisons
    with prior diffusion work.
    """
    def __init__(self, inner_model, device="cpu"):
        super().__init__(inner_model, device)
        self.inner_model = inner_model
        self.get_scalings = self.get_vpsde_scalings

        self.eps_t = torch.Tensor([10e-5]).to(device)
        self.M = torch.Tensor([1000]).to(device)

        self.uniform_u = torch.Tensor([1.0]).to(device)
        self.uniform_l = self.eps_t

        beta_min = torch.Tensor([0.1]).to(device)
        beta_d = torch.Tensor([19.9]).to(device)

        self.convert_sigma_to_t = (
            lambda sigma: (
                ((beta_min**2 + 2 * beta_d * (1 + sigma**2).log())).sqrt() - beta_min
            )
            / beta_d
        )
        self.convert_t_to_sigma = (
            lambda t: ((0.5 * beta_d * t**2 + beta_min * t).exp() - 1) ** 0.5
        )

        self.weighting = torch.ones_like

    def get_vpsde_scalings(self, sigma):
        c_skip = self.ignore
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1) ** 0.5
        return c_skip, c_out, c_in

    def sample_training_sigmas(self, input):
        t = (self.uniform_u - self.uniform_l) * torch.rand(
            [input.shape[0]], device=input.device
        ) + self.uniform_l
        return self.convert_t_to_sigma(t)
