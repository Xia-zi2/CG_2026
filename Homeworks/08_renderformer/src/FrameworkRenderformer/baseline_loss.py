from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRenderFormerLoss(nn.Module):
    """
    A lighter training loss for course use.

    By default it only uses log-L1, which is much cheaper than the original
    LPIPS-based loss and is friendlier to low-VRAM GPUs.
    """

    def __init__(
        self,
        loss_type: str = "log_l1",
        use_lpips: bool = False,
        lpips_weight: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.use_lpips = use_lpips
        self.lpips_weight = lpips_weight
        self.device_name = device

        self.lpips_model = None
        if use_lpips:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError("LPIPS is not installed. Disable --use_lpips or install lpips.") from exc

            self.lpips_model = lpips.LPIPS(net="vgg").to(device)
            self.lpips_model.eval()
            for parameter in self.lpips_model.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _log_transform(image: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(image, min=0.0))

    @staticmethod
    def _tone_map(image: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.log2(torch.clamp(image, min=1e-6)), 0.0, 1.0)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.loss_type == "mse":
            base_loss = F.mse_loss(prediction, target)
        elif self.loss_type == "l1":
            base_loss = F.l1_loss(prediction, target)
        elif self.loss_type == "log_l1":
            base_loss = torch.mean(torch.abs(self._log_transform(prediction) - self._log_transform(target)))
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        total_loss = base_loss
        lpips_loss = torch.zeros_like(base_loss)
        if self.lpips_model is not None:
            pred_lpips = self._tone_map(prediction) * 2.0 - 1.0
            target_lpips = self._tone_map(target) * 2.0 - 1.0
            lpips_loss = self.lpips_model(pred_lpips, target_lpips).mean()
            total_loss = total_loss + self.lpips_weight * lpips_loss

        metrics = {
            "total_loss": total_loss.detach(),
            "base_loss": base_loss.detach(),
            "lpips_loss": lpips_loss.detach(),
        }
        return total_loss, metrics
