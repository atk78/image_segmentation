"""推論用スクリプト

学習時の [configs/params.yaml] に合わせてモデルを復元し、
1枚またはディレクトリ内の画像に対してセグメンテーション推論を行います。

前提
- `train.py` が保存した重みは `model.model.state_dict()`（= smp のモデル本体）
- クラスは `CLASSES` の定義順をクラスindexとして扱う

CLI例
    python -m image_segmentation.predict \
        --input ./data/test/img \
        --weights ./models/model1/best_model.pth \
        --config ./configs/params.yaml \
        --out ./outputs/pred
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from .augm import get_preprocessing
from .model import SegmentationModel
from .train import load_params
from .visualize import visualize


SUPPORTED_IMAGE_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class PredictConfig:
    class_info: dict
    class_values: list[int]
    fig_h: int
    fig_w: int
    arch: str
    encoder: str
    encoder_weights: str


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _iter_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    image_paths: list[Path] = []
    for ext in SUPPORTED_IMAGE_EXTS:
        image_paths.extend(path.glob(f"*{ext}"))
        image_paths.extend(path.glob(f"*{ext.upper()}"))
    return sorted(set(image_paths))


def _load_predict_config(config_path: Path) -> PredictConfig:
    params = load_params(config_path)

    class_info = params.get(
        "CLASSES",
        {"background": 0, "object": 255},
    )
    class_values = [int(v) for _, v in class_info.items()]

    figure_params = params.get("FIGURE", {})
    figure_size = figure_params.get("SIZE", [512, 512])
    fig_h, fig_w = int(figure_size[0]), int(figure_size[1])

    model_params = params.get("MODEL", {})
    model_type = model_params.get("TYPE", "model1")
    hparams = model_params.get(model_type, {})
    arch = hparams.get("ARCHITECTURE", "DeepLabV3Plus")
    encoder = hparams.get("ENCODER", "resnet34")
    encoder_weights = hparams.get("ENCODER_WEIGHTS", "imagenet")

    return PredictConfig(
        class_info=class_info,
        class_values=class_values,
        fig_h=fig_h,
        fig_w=fig_w,
        arch=arch,
        encoder=encoder,
        encoder_weights=encoder_weights,
    )


def _build_model(
    cfg: PredictConfig,
    device: str,
    weights_path: Path,
) -> SegmentationModel:
    model = SegmentationModel(
        arch=cfg.arch,
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=3,
        out_classes=len(cfg.class_values),
        learning_rate=1e-4,
    )

    state = torch.load(weights_path, map_location=device)
    try:
        model.model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def _build_preprocessing(cfg: PredictConfig):
    preprocessing_fn = None
    try:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            cfg.encoder,
            cfg.encoder_weights,
        )
    except Exception:
        preprocessing_fn = None

    if preprocessing_fn is None:
        return None
    return get_preprocessing(preprocessing_fn)


def _preprocess_image(
    image_rgb: np.ndarray,
    cfg: PredictConfig,
    preprocessing,
) -> tuple[np.ndarray, torch.Tensor]:
    resized = cv2.resize(image_rgb, (cfg.fig_w, cfg.fig_h))
    if preprocessing is None:
        x = resized.astype("float32") / 255.0
        x = np.transpose(x, (2, 0, 1))
    else:
        dummy_mask = np.zeros((cfg.fig_h, cfg.fig_w, 1), dtype="float32")
        sample = preprocessing(image=resized, mask=dummy_mask)
        x = sample["image"]
    x_tensor = torch.from_numpy(x).unsqueeze(0)
    return resized, x_tensor


def _postprocess_mask(
    prob: torch.Tensor,
    cfg: PredictConfig,
    threshold: float,
) -> np.ndarray:
    if len(cfg.class_values) == 1:
        mask01 = (prob.squeeze(0).squeeze(0) > threshold).to(torch.uint8)
        mask = (mask01.cpu().numpy() * cfg.class_values[0]).astype("uint8")
        return mask

    pred_idx = prob.argmax(dim=1).squeeze(0).to(torch.long)
    idx_np = pred_idx.cpu().numpy()

    values = np.asarray(cfg.class_values, dtype="uint8")
    mask = values[idx_np]
    return mask


def predict(
    input_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    device: str | None = None,
    threshold: float = 0.5,
    visualize_result: bool = False,
    # --- backward compatible aliases (used by older notebooks) ---
    image_path_or_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
):
    """推論を実行してマスクを返す。

    - `out_dir` を指定すると `<stem>_pred.png` を保存
    - 返り値は (image_path, resized_rgb, pred_mask_uint8) のリスト
    """
    # Backward compatible mapping
    if input_path is None and image_path_or_dir is not None:
        input_path = image_path_or_dir

    if weights_path is None or config_path is None:
        if model_dir is None:
            raise TypeError(
                "predict() requires either (weights_path, config_path) "
                "or legacy model_dir."
            )
        model_dir = Path(model_dir)
        if weights_path is None:
            weights_path = model_dir / "best_model.pth"
        if config_path is None:
            cfg_in_model = model_dir / "params.yaml"
            if cfg_in_model.exists():
                config_path = cfg_in_model
            else:
                # Fallback: project-level config
                config_path = (
                    Path(__file__).parents[2] / "configs" / "params.yaml"
                )

    if input_path is None:
        raise TypeError("predict() missing required input_path")

    input_path = Path(input_path)
    weights_path = Path(weights_path)
    config_path = Path(config_path)
    out_dir = Path(out_dir) if out_dir is not None else None

    device = _resolve_device(device)
    cfg = _load_predict_config(config_path)
    model = _build_model(cfg, device=device, weights_path=weights_path)
    preprocessing = _build_preprocessing(cfg)

    image_paths = _iter_image_paths(input_path)
    if not image_paths:
        raise ValueError(f"No images found under: {input_path}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for image_path in image_paths:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        resized_rgb, x_tensor = _preprocess_image(rgb, cfg, preprocessing)
        x_tensor = x_tensor.to(device)

        with torch.no_grad():
            prob = model(x_tensor)

        pred_mask = _postprocess_mask(prob, cfg, threshold=threshold)

        if out_dir is not None:
            out_path = out_dir / f"{image_path.stem}_pred.png"
            cv2.imwrite(str(out_path), pred_mask)

        results.append((image_path, resized_rgb, pred_mask))

    if visualize_result and results:
        _, rgb0, mask0 = results[0]
        visualize(image=rgb0, mask=mask0)

    return results


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Image segmentation prediction"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image file or directory",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to best_model.pth (saved model.model.state_dict())",
    )
    parser.add_argument("--config", required=True, help="Path to params.yaml")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (optional)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu or cuda (optional)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args(argv)

    predict(
        input_path=args.input,
        weights_path=args.weights,
        config_path=args.config,
        out_dir=args.out,
        device=args.device,
        threshold=args.threshold,
        visualize_result=args.visualize,
    )


if __name__ == "__main__":
    main()
