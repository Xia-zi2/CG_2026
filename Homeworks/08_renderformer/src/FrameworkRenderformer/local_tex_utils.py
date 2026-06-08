from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_texture_patches(
    texture_map: torch.Tensor,
    uv_coordinates: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor | None:
    """
    Sample one square texture patch per triangle by mapping a unit square
    to triangle UV space with an affine transform.
    """
    if texture_map is None or uv_coordinates is None:
        return None

    device = texture_map.device
    num_triangles = uv_coordinates.shape[0]

    y_range = torch.linspace(0.0, 1.0, patch_size, device=device)
    x_range = torch.linspace(0.0, 1.0, patch_size, device=device)
    u_base, v_base = torch.meshgrid(x_range, y_range, indexing="xy")
    u_base = u_base.unsqueeze(0)
    v_base = v_base.unsqueeze(0)

    p0 = uv_coordinates[:, 0, :].unsqueeze(1).unsqueeze(1)
    p1 = uv_coordinates[:, 1, :].unsqueeze(1).unsqueeze(1)
    p2 = uv_coordinates[:, 2, :].unsqueeze(1).unsqueeze(1)

    vec_u = p1 - p0
    vec_v = p2 - p0
    grid = p0 + u_base.unsqueeze(-1) * vec_u + v_base.unsqueeze(-1) * vec_v

    grid = grid * 2.0 - 1.0
    flat_grid = grid.view(1, num_triangles * patch_size, patch_size, 2)

    sampled_strip = F.grid_sample(
        texture_map,
        flat_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled_strip.view(3, num_triangles, patch_size, patch_size).permute(1, 0, 2, 3)
