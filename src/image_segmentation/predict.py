"""推論用スクリプト

model/best_model.pth と model/params.yaml を読み込み、
1 枚またはディレクトリ内の画像に対してセグメンテーション推論を行う。

使用例 (モジュール実行):

    python -m image_segmentation.predict ./data/test/img ./model

使用例 (Python から呼び出し):

    from image_segmentation.predict import predict
    predict(
        image_path_or_dir="./data/test/img/case225.bmp",
        model_dir="./model",
        device="cuda",
    )
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor

from .train import load_params
from .model import SegmentationModel
from .augm import get_preprocessing
from .visualize import visualize
import segmentation_models_pytorch as smp


def _build_model(
    model_dir: Path, device: str = "cpu"
) -> tuple[SegmentationModel, dict]:
    """model_dir 内の params.yaml と best_model.pth からモデルを構築する"""
    config_path = model_dir.joinpath("params.yaml")
    params = load_params(config_path)
    model_params = params.get("MODEL", {})
    figure_params = params.get("FIGURE", {})
    figure_size = figure_params.get("SIZE", [256, 256])

    model_type = model_params.get("TYPE", "DeepLabV3Plus")

    if model_type == "DeepLabV3Plus":
        dl_params = model_params.get("DeepLabV3Plus", {})
        encoder = dl_params.get("ENCODER", "efficientnet-b0")
        encoder_weights = dl_params.get("ENCODER_WEIGHTS", "imagenet")
        classes = dl_params.get("CLASSES", ["lung"])

        hparams = {
            "ENCODER": encoder,
            "ENCODER_WEIGHTS": encoder_weights,
            "IN_CHANNELS": 3,
            "OUT_CLASSES": len(classes),
        }
    elif model_type == "SegFormer":
        sf_params = model_params.get("SegFormer", {})
        model_name = sf_params.get(
            "NAME", "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        classes = sf_params.get("CLASSES", ["lung"])

        hparams = {
            "NAME": model_name,
            "IN_CHANNELS": 3,
            "OUT_CLASSES": len(classes),
        }

    elif model_type == "Mask2Former":
        m2f_params = model_params.get("Mask2Former", {})
        model_name = m2f_params.get(
            "NAME", "facebook/mask2former-swin-small-ade-semantic"
        )
        classes = m2f_params.get("CLASSES", ["lung"])

        hparams = {
            "NAME": model_name,
            "IN_CHANNELS": 3,
            "OUT_CLASSES": len(classes),
        }
    else:
        raise ValueError(f"Unsupported model TYPE in config: {model_type}")

    # SegmentationModel を構築
    model = SegmentationModel(
        model_type=model_type,
        hparams=hparams,
    )

    # 重みをロード
    state_dict = torch.load(
        model_dir.joinpath("best_model.pth"), map_location=device
    )
    # 学習時に保存したのは内部 model の state_dict なので、そのまま適用
    model.model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, {
        "model_type": model_type,
        "hparams": hparams,
        "classes": classes,
        "figure_size": figure_size,
    }


def _build_preprocessing(meta: dict):
    """推論時用の前処理関数を構築

    学習時の train.py と同じポリシーで、DeepLabV3+ では smp の前処理、
    SegFormer では 0-1 スケーリングのみを行い、get_preprocessing で
    HWC->CHW/float32 変換を行う。
    """
    model_type = meta["model_type"]
    hparams = meta["hparams"]

    preprocessing_fn = None
    if model_type == "DeepLabV3Plus":
        encoder = hparams["ENCODER"]
        encoder_weights = hparams["ENCODER_WEIGHTS"]
        try:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(
                encoder, encoder_weights
            )
        except Exception:
            preprocessing_fn = None
    elif model_type == "SegFormer":
        def preprocessing_fn(x, **kwargs):
            return x.astype("float32") / 255.0

    elif model_type == "Mask2Former":
        model_name = hparams["NAME"]
        processor = AutoImageProcessor.from_pretrained(model_name)
        mean = processor.image_mean
        std = processor.image_std

        def preprocessing_fn(x, **kwargs):
            x = x.astype("float32") / 255.0
            x = (x - mean) / std
            return x

    if preprocessing_fn is None:
        return None
    return get_preprocessing(preprocessing_fn)


def _predict_single(
    model: SegmentationModel,
    preprocessing,
    figure_size,
    image_path: Path,
    device: str = "cpu",
    threshold: float = 0.5,
):
    """1枚の画像に対して推論し、元画像とマスクを返す"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # FIGURE.SIZE にリサイズ（[H, W]）
    if figure_size is not None:
        fig_h, fig_w = int(figure_size[0]), int(figure_size[1])
    else:
        fig_h, fig_w = 256, 256

    image_resized = cv2.resize(image, (fig_w, fig_h))

    if preprocessing is not None:
        sample = preprocessing(
            image=image_resized,
            mask=np.zeros((fig_h, fig_w, 1), dtype="float32"),
        )
        input_image = sample["image"]  # CHW, float32
    else:
        # HWC -> CHW, 0-1 正規化
        input_image = image_resized.astype("float32") / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))

    x_tensor = torch.from_numpy(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        pr = model(x_tensor)
        if model.out_classes == 1:
            pr_mask = (
                (pr.squeeze().cpu().numpy() > threshold).astype("float32")
            )
        else:
            # 多クラスの場合は argmax で 1 チャネルに縮約
            pr_mask = (
                pr.squeeze(0)
                .cpu()
                .numpy()
                .argmax(axis=0)
                .astype("float32")
            )

    return image_resized, pr_mask


def predict(
    image_path_or_dir: Union[str, Path],
    model_dir: Union[str, Path] = "./model",
    device: str | None = None,
    visualize_result: bool = True,
    threshold: float = 0.5,
):
    """保存済みモデルで推論を行うヘルパー関数

    Parameters
    ----------
    image_path_or_dir : str or Path
        単一画像パス、または画像ディレクトリ
    model_dir : str or Path, optional
        params.yaml と best_model.pth が置かれているディレクトリ
    device : {"cpu", "cuda"}, optional
        推論デバイス（None の場合は自動判定）
    visualize_result : bool, optional
        True の場合、最初の画像について visualize() で表示
    """
    model_dir = Path(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, meta = _build_model(model_dir, device=device)
    preprocessing = _build_preprocessing(meta)

    image_path_or_dir = Path(image_path_or_dir)
    if image_path_or_dir.is_dir():
        image_paths = sorted(image_path_or_dir.glob("*.bmp"))
    else:
        image_paths = [image_path_or_dir]

    results = []
    for img_path in image_paths:
        img, mask = _predict_single(
            model,
            preprocessing,
            figure_size=meta.get("figure_size", [256, 256]),
            image_path=img_path,
            device=device,
            threshold=threshold,
        )
        results.append((img_path, img, mask))

    if visualize_result and results:
        _, img0, mask0 = results[0]
        visualize(image=img0, mask=mask0)

    return results


def main():
    import sys

    args = sys.argv
    if not (2 <= len(args) <= 3):
        print(
            "Usage: python -m image_segmentation.predict "
            "<image_path_or_dir> [model_dir]"
        )
        raise SystemExit(1)

    image_path_or_dir = args[1]
    model_dir = args[2] if len(args) == 3 else "./model"

    predict(image_path_or_dir=image_path_or_dir, model_dir=model_dir)


if __name__ == "__main__":
    main()
