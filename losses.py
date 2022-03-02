# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""


import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):

        super().__init__()

        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):

            # Assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs

        # Base loss (first term of equations 2 & 3 in the paper)
        base_loss = self.base_criterion(outputs, labels)

        # No distillation case (no teacher)
        if self.distillation_type == "none":

            return base_loss

        if outputs_kd is None:

            err_msg = "When knowledge distillation is enabled, the model is expected to return a"
            err_msg += " Tuple[Tensor, Tensor] with the output of the class_token and the"
            err_msg += " dist_token."
            raise ValueError(err_msg)

        # Do not backprop through the teacher
        with torch.no_grad():

            teacher_outputs = self.teacher_model(inputs)

        # Soft distillation case (second term of equation 2 in the paper)
        if self.distillation_type == "soft":

            T = self.tau  # temperature
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/
            # model/net.py#L100
            # with slight modifications

            distillation_loss = F.kl_div(
                # We provide the teacher"s targets in log probability because we use
                # log_target=True
                # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/
                # 9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                # but it is possible to give just the probabilities and set log_target=False. In
                # our experiments we tried both.
                F.log_softmax(outputs_kd / T, dim=1),  # student predictions
                F.log_softmax(teacher_outputs / T, dim=1),  # teacher predictions
                reduction="sum",
                log_target=True
            ) * (T * T) / outputs_kd.numel()

            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details

        # Hard distillation case (second term of equation 3 in the paper)
        elif self.distillation_type == "hard":

            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        # Reconstruction of the global distillation loss (equations 2 & 3 in the paper)
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss
