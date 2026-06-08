from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

# Force SDPA branch by default, which is more stable and consistent with the course baseline.
os.environ.setdefault("ATTN_IMPL", "sdpa")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_data import H5TriangleDataset, PtSceneDataset, renderformer_baseline_collate
from baseline_model import CourseRenderFormerWrapper, build_baseline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate RenderFormer with vis-space PSNR.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_format", choices=["pt", "h5"], default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--max_items", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--image_size", type=int, default=64, help="Only used for H5 data.")

    parser.add_argument("--save_error", action="store_true", help="Also save abs error visualization.")
    parser.add_argument("--error_scale", type=float, default=5.0)

    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def build_dataset(dataset_format: str, data_path: str, max_items: int | None, image_size: int):
    if dataset_format == "pt":
        return PtSceneDataset(data_path, max_items=max_items)
    return H5TriangleDataset(data_path, render_resolution=image_size)


def make_config_from_saved_args(saved_args: dict[str, Any]):
    return build_baseline_config(
        latent_dim=saved_args.get("latent_dim", 256),
        num_layers=saved_args.get("num_layers", 4),
        num_heads=saved_args.get("num_heads", 4),
        view_layers=saved_args.get("view_layers", 4),
        view_num_heads=saved_args.get("view_num_heads", 4),
        use_dpt_decoder=saved_args.get("use_dpt_decoder", False),
        num_register_tokens=saved_args.get("num_register_tokens", 4),
        patch_size=saved_args.get("patch_size", 8),
        texture_patch_size=saved_args.get("texture_patch_size", 1),
        vertex_pe_num_freqs=saved_args.get("vertex_pe_num_freqs", 6),
        vn_pe_num_freqs=saved_args.get("vn_pe_num_freqs", 6),
        use_vn_encoder=not saved_args.get("no_vn", False),
        ffn_opt=saved_args.get("ffn_opt", "checkpoint"),
    )


def tensor_to_display_rgb_float(image_chw: torch.Tensor) -> np.ndarray:
    """
    Same display transform as train_course_baseline.py visualization, but kept as float [0,1].

    Input:
        image_chw: [3, H, W], HDR / linear RGB tensor.

    Output:
        image_hwc: [H, W, 3], display RGB float in [0,1].
    """
    image = image_chw.detach().to(device="cpu", dtype=torch.float32)
    image = torch.clamp(image, 0.0, 1.0)
    image = torch.pow(image, 1.0 / 2.2)
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    return image


def float_rgb_to_uint8(image_hwc: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image_hwc * 255.0), 0, 255).astype(np.uint8)


def psnr_from_display_rgb(pred_rgb: np.ndarray, target_rgb: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    """
    Compute PSNR in the same display space used by saved vis images.

    pred_rgb, target_rgb:
        [H, W, 3], float in [0,1].

    PSNR assumes I_max = 1 in display space:
        PSNR = -10 log10(MSE)
    """
    mse = float(np.mean((pred_rgb.astype(np.float32) - target_rgb.astype(np.float32)) ** 2))
    psnr = -10.0 * math.log10(max(mse, eps))
    return psnr, mse


def save_vis_pair(target_rgb: np.ndarray, pred_rgb: np.ndarray, save_path: Path) -> None:
    """
    Save the same layout as course vis:
        left  = target / GT
        right = prediction
    """
    canvas = np.concatenate([target_rgb, pred_rgb], axis=1)
    Image.fromarray(float_rgb_to_uint8(canvas)).save(save_path)


def save_error_map(target_rgb: np.ndarray, pred_rgb: np.ndarray, save_path: Path, scale: float = 5.0) -> None:
    err = np.abs(pred_rgb - target_rgb) * scale
    err = np.clip(err, 0.0, 1.0)
    Image.fromarray(float_rgb_to_uint8(err)).save(save_path)


def get_sample_name(batch: dict[str, Any], local_index: int, global_index: int) -> str:
    value = batch.get("sample_name", None)
    if isinstance(value, list) and local_index < len(value):
        return str(value[local_index])
    if isinstance(value, tuple) and local_index < len(value):
        return str(value[local_index])
    if isinstance(value, str):
        return value
    return f"sample_{global_index:04d}"


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    err_dir = out_dir / "error"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_error:
        err_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})

    dataset_format = args.dataset_format or saved_args.get("dataset_format")
    data_path = args.data_path or saved_args.get("data_path")

    if dataset_format is None:
        raise ValueError("dataset_format is not specified and not found in checkpoint args.")
    if data_path is None:
        raise ValueError("data_path is not specified and not found in checkpoint args.")

    config = make_config_from_saved_args(saved_args)
    model = CourseRenderFormerWrapper(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = build_dataset(
        dataset_format=dataset_format,
        data_path=data_path,
        max_items=args.max_items,
        image_size=args.image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=renderformer_baseline_collate,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {data_path}")
    print(f"Dataset format: {dataset_format}")
    print(f"Eval samples: {len(dataset)}")
    print(f"Device: {device}")
    print(f"Output dir: {out_dir}")
    print("PSNR space: display RGB = clamp(linear RGB, 0, 1)^(1/2.2)")
    print("=" * 72)

    records: list[dict[str, Any]] = []
    global_index = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = move_to_device(batch, device)
            target = batch["gt_image"].to(dtype=torch.float32)

            prediction = model(batch).to(dtype=torch.float32)

            batch_size = prediction.shape[0]

            for i in range(batch_size):
                sample_name = get_sample_name(batch, i, global_index)

                target_rgb = tensor_to_display_rgb_float(target[i])
                pred_rgb = tensor_to_display_rgb_float(prediction[i])

                psnr_value, mse_value = psnr_from_display_rgb(
                    pred_rgb=pred_rgb,
                    target_rgb=target_rgb,
                )

                safe_name = Path(sample_name).stem
                image_name = f"{global_index:04d}_{safe_name}.png"
                save_vis_pair(target_rgb, pred_rgb, vis_dir / image_name)

                if args.save_error:
                    save_error_map(
                        target_rgb,
                        pred_rgb,
                        err_dir / f"{global_index:04d}_{safe_name}_error_x{args.error_scale:g}.png",
                        scale=args.error_scale,
                    )

                records.append(
                    {
                        "index": global_index,
                        "sample_name": sample_name,
                        "psnr_display": psnr_value,
                        "mse_display": mse_value,
                        "vis_path": str(vis_dir / image_name),
                    }
                )

                print(
                    f"[{global_index + 1:04d}/{len(dataset):04d}] "
                    f"PSNR_vis={psnr_value:.4f} dB "
                    f"MSE_vis={mse_value:.8f} "
                    f"name={sample_name}"
                )

                global_index += 1

    mean_psnr = float(np.mean([r["psnr_display"] for r in records]))
    mean_mse = float(np.mean([r["mse_display"] for r in records]))

    result = {
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "dataset_format": dataset_format,
        "num_samples": len(records),
        "psnr_definition": "display RGB PSNR after clamp(x,0,1)^(1/2.2), I_max=1; no per-image min-max normalization",
        "mean_psnr_display": mean_psnr,
        "mean_mse_display": mean_mse,
        "records": records,
    }

    json_path = out_dir / "psnr_vis_eval.json"
    csv_path = out_dir / "psnr_vis_eval.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "sample_name", "psnr_display", "mse_display", "vis_path"],
        )
        writer.writeheader()
        writer.writerows(records)

    print("=" * 72)
    print(f"Mean PSNR_vis: {mean_psnr:.4f} dB")
    print(f"Mean MSE_vis : {mean_mse:.8f}")
    print(f"Saved JSON   : {json_path}")
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved vis    : {vis_dir}")
    if args.save_error:
        print(f"Saved errors : {err_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()