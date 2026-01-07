"""
Base class for pytorch lightning generative modelling.
"""

import torch.nn as nn
import torch
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    OneCycleLR,
    ExponentialLR,
    StepLR,
    LambdaLR,
)
import pytorch_lightning as pl
import numpy as np
import logging
import torch.distributed as dist

logger = logging.getLogger(__name__)

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

class LightningBase(pl.LightningModule):
    """
    Base class for pytorch lightning generative modelling.

    PL is a high-level interface for PyTorch that allows for easy training and
    evaluation of models. This class is a base class for generative models
    that provides the main shared training and validation steps for the models.

    In addition, this base class provides utils for logging and weighting the loss
    according to each output variable. It also includes a method for configuring 
    optmizers, which is just pytorch lightning boilerplate code.

    Child models should implement the loss_function method, which is called in the
    training and validation steps. This method should return the loss tensor for each
    output channel, which is then reduced and logged by the shared_step method.

    """
    def __init__(self, output_channels=None, weights=None, **kwargs):
        """
        :param output_channels: list of strings, names of output channels
        :param weights: list of floats, weights for each output channel
            to multiply the loss by during training. If None, no weighting is applied.
        """
        super().__init__(**kwargs)
        self.output_channels = output_channels
        # just using nn.Parameter so pytorch lightning handles device correctly
        if weights is not None:
            self.weights = nn.Parameter(torch.Tensor(weights), requires_grad=False)
        else:
            self.weights = None

    def reduce_loss(self, unreduced_loss):
        """
        Apply output-dependent loss weighting to the loss tensor.
        """
        losses = {
            output_channel[:3] + "_loss": unreduced_loss[i]
            for i, output_channel in enumerate(self.output_channels)
        }
        scaled_loss = unreduced_loss * self.weights
        losses["loss"] = scaled_loss.sum()
        return losses

    def shared_step(self, batch, eval_type=""):
        """
        Key function for training and validation steps.

        We provide the skeleton for training children models, which should
        implement the loss_function method. This method is called in the
        training and validation steps, and the loss is logged via the chosen logger.

        :param batch: tuple of input and target tensors
        :param eval_type: str, prefix for logging keys during validation

        :returns: None, dict of losses
        """
        unreduced_loss = self.loss_function(self, batch[1], batch[0])
        if self.weights is not None:
            losses = self.reduce_loss(unreduced_loss)
        else:
            losses = {"loss": unreduced_loss.mean()}
        return None, {f"{eval_type}{name}": loss for name, loss in losses.items()}

    def get_batchsize(self, batch):
        conds = batch[0]
        if isinstance(conds, list):
            bs = conds[0].shape[0]
        else:
            bs = conds.shape[0]
        return bs

    def training_step(self, batch, batch_idx):
        #logger.info(f" >> >> INSIDE base training_step | batch[0] {batch[0].detach().cpu().numpy()} batch[1] {batch[1].detach().cpu().numpy()}")
        bs = self.get_batchsize(batch)
        _, loss = self.shared_step(batch)
        
        if is_main_process():
            logger.info(f" >> >> INSIDE base training_step | loss {loss['loss'].detach().cpu().numpy()}")
        self._log_loss(loss, batch_size=bs)

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        bs = self.get_batchsize(batch)
        _, loss = self.shared_step(batch, "val_")
        self.model.train()
        
        if is_main_process():
            logger.info(f" >> >> INSIDE base validation_step | val_loss {loss['val_loss'].detach().cpu().numpy()}")
        self._log_loss(loss, "val_", batch_size=bs)

        return loss["val_loss"]

    def _log_loss(self, loss, eval_type: str = "", batch_size=None):

        for l in loss.keys():
            self.log(l, loss[l], batch_size=batch_size)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        import copy

        config = copy.copy(self.optimizer_config)
        optimizer_type = config.get("optimizer")
        lr = config.get("lr")
        if optimizer_type == "Adam":
            opt = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                betas=(config.get("beta1"), 0.999),
                eps=config.get("eps"),
                weight_decay=config.get("weight_decay"),
            )
        else:
            raise NotImplementedError()

        if "lr_schedule" not in config.keys() or config["lr_schedule"] is None:
            return opt
        learning_rate_sched = config.get("lr_schedule")
        schedules = []
        for sched_type, options in learning_rate_sched.items():
            if sched_type == "one_cycle":
                print("Using one cycle LR: ", options)
                sch = OneCycleLR(opt, **options)
            elif sched_type == "reduce_on_plateau":
                sch = ReduceLROnPlateau(opt, **options)
            elif sched_type == "exponential":
                sch = ExponentialLR(opt, **options)
            elif sched_type == "step":
                sch = StepLR(opt, **options)
                sch = {"scheduler": sch, "interval": "epoch", "monitor": "val_loss"}
            elif sched_type == "warmup":
                warmup = options["n_steps"]

                def lr_foo(step):
                    lr_scale = lr * np.minimum(step / warmup, 1.0)
                    return lr_scale

                sch = LambdaLR(opt, lr_lambda=lr_foo)
                sch = {"scheduler": sch, "interval": "step"}
            else:
                continue

            schedules.append(sch)
        if len(schedules) > 0:
            return [opt], schedules
        else:
            return [opt]
