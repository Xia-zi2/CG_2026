from __future__ import annotations

import torch


def rearrange(tensor: torch.Tensor, pattern: str, **sizes) -> torch.Tensor:
    pattern = pattern.strip()

    if pattern == "... (d r) -> ... d r":
        r = sizes["r"]
        last_dim = tensor.shape[-1]
        if last_dim % r != 0:
            raise ValueError(f"Last dim {last_dim} is not divisible by r={r}.")
        return tensor.reshape(*tensor.shape[:-1], last_dim // r, r)

    if pattern == "... d r -> ... (d r)":
        return tensor.reshape(*tensor.shape[:-2], tensor.shape[-2] * tensor.shape[-1])

    if pattern == "batch n_tris n_verts d -> batch 1 n_tris (n_verts d)":
        batch, n_tris, n_verts, dim = tensor.shape
        return tensor.reshape(batch, n_tris, n_verts * dim).unsqueeze(1)

    if pattern == "b h s d -> (b s) h d":
        b, h, s, d = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(b * s, h, d)

    if pattern == "(b s) h d -> b s h d":
        b = sizes["b"]
        bs, h, d = tensor.shape
        if bs % b != 0:
            raise ValueError(f"First dim {bs} is not divisible by b={b}.")
        s = bs // b
        return tensor.reshape(b, s, h, d)

    if pattern == "b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)":
        p1 = sizes["p1"]
        p2 = sizes["p2"]
        b, hp, wp, c = tensor.shape
        if hp % p1 != 0 or wp % p2 != 0:
            raise ValueError(f"Spatial shape {(hp, wp)} is not divisible by patch {(p1, p2)}.")
        h1 = hp // p1
        w1 = wp // p2
        tensor = tensor.reshape(b, h1, p1, w1, p2, c)
        tensor = tensor.permute(0, 1, 3, 5, 2, 4)
        return tensor.reshape(b, h1 * w1, c * p1 * p2)

    if pattern == "b (h1 w1) (c p1 p2) -> b c (h1 p1) (w1 p2)":
        p1 = sizes["p1"]
        p2 = sizes["p2"]
        h1 = sizes["h1"]
        w1 = sizes["w1"]
        b, n, cp = tensor.shape
        if n != h1 * w1:
            raise ValueError(f"Token count {n} does not match h1*w1={h1 * w1}.")
        if cp % (p1 * p2) != 0:
            raise ValueError(f"Last dim {cp} is not divisible by p1*p2={p1 * p2}.")
        c = cp // (p1 * p2)
        tensor = tensor.reshape(b, h1, w1, c, p1, p2)
        tensor = tensor.permute(0, 3, 1, 4, 2, 5)
        return tensor.reshape(b, c, h1 * p1, w1 * p2)

    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")


def repeat(tensor: torch.Tensor, pattern: str, **sizes) -> torch.Tensor:
    pattern = pattern.strip()

    if pattern == "... f -> ... (f r)":
        r = sizes["r"]
        repeats = [1] * tensor.ndim + [r]
        expanded = tensor.unsqueeze(-1).repeat(*repeats)
        return expanded.reshape(*tensor.shape[:-1], tensor.shape[-1] * r)

    raise NotImplementedError(f"Unsupported repeat pattern: {pattern}")
