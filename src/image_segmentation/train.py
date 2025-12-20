"""\
学習スクリプト (PyTorch Lightning を利用)

ノートブックの流れに合わせ、データセット・拡張・Trainer を使って学習を行います。
設定は `configs/params.toml` から読み込みます。
"""

from pathlib import Path
import shutil
import sys
import random

import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from transformers import AutoImageProcessor

from .augm import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
)
from .model import SegmentationModel
from .dataset import Dataset
from .visualize import metrics_plot


# ランダムシード固定
pl.seed_everything(42)
random.seed(42)
torch.manual_seed(42)


def load_params(path: Path):
    # YAML を優先的に読み込む。拡張子に応じて処理を切り替える
    if path.suffix in (".yaml", ".yml"):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def main(data_dir: str, config_path: str):
    data_dir = Path(data_dir)
    train_img_dir = data_dir.joinpath("train/img")
    train_mask_dir = data_dir.joinpath("train/mask")
    valid_img_dir = data_dir.joinpath("valid/img")
    valid_mask_dir = data_dir.joinpath("valid/mask")
    test_img_dir = data_dir.joinpath("test/img")
    test_mask_dir = data_dir.joinpath("test/mask")

    if Path().cwd().joinpath("lightning_logs").exists():
        shutil.rmtree(Path().cwd().joinpath("lightning_logs"))

    params = load_params(Path(config_path))
    model_params = params.get("MODEL", {})
    train_params = params.get("TRAIN", {})
    figure_params = params.get("FIGURE", {})
    figure_size = figure_params.get("SIZE", [256, 256])
    fig_h, fig_w = int(figure_size[0]), int(figure_size[1])

    # モデル種別とクラス定義を設定ファイルから読み込む
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

        # SegFormer では smp のエンコーダ前処理は使わないため encoder_* は定義しない
        encoder = None
        encoder_weights = None

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

        # Mask2Former では smp のエンコーダ前処理は使わない
        encoder = None
        encoder_weights = None

    else:
        raise ValueError(f"Unsupported model TYPE in config: {model_type}")

    # device = params.get("DEVICE", "cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = int(train_params.get("BATCH_SIZE", 8))
    lr = float(train_params.get("LR", 1e-4))
    epochs = int(train_params.get("EPOCHS", 40))

    # モデル作成（DeepLabV3+ / SegFormer を切り替え）
    model = SegmentationModel(
        model_type=model_type,
        hparams=hparams,
        learning_rate=lr,
    )

    # データセット読み込み (既定の data フォルダ構成に合わせる)
    # train_img_dir = Path("data/train/img")
    # train_mask_dir = Path("data/train/mask")
    # valid_img_dir = Path("data/valid/img")
    # valid_mask_dir = Path("data/valid/mask")

    preprocessing_fn = None
    # モデル種別ごとに前処理関数を設定
    if model_type == "DeepLabV3Plus" and encoder is not None:
        # smp のエンコーダに合わせた正規化を実施
        try:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(
                encoder, encoder_weights
            )
        except Exception:
            preprocessing_fn = None
    elif model_type == "SegFormer":
        # SegFormer 用: 0-255 -> 0-1 の float32 変換のみ行い、
        # チャネル順変換や型変換は get_preprocessing(to_tensor) に任せる
        def preprocessing_fn(x, **kwargs):
            return x.astype("float32") / 255.0

    elif model_type == "Mask2Former":
        processor = AutoImageProcessor.from_pretrained(model_name)
        mean = processor.image_mean
        std = processor.image_std

        def preprocessing_fn(x, **kwargs):
            x = x.astype("float32") / 255.0
            x = (x - mean) / std
            return x

    train_dataset = Dataset(
        train_img_dir,
        train_mask_dir,
        classes=classes,
        augmentation=get_training_augmentation(size=(fig_h, fig_w)),
        preprocessing=(
            get_preprocessing(preprocessing_fn) if preprocessing_fn else None
        ),
    )

    valid_dataset = Dataset(
        valid_img_dir,
        valid_mask_dir,
        classes=classes,
        augmentation=get_validation_augmentation(size=(fig_h, fig_w)),
        preprocessing=(
            get_preprocessing(preprocessing_fn) if preprocessing_fn else None
        ),
    )

    test_dataset = Dataset(
        test_img_dir,
        test_mask_dir,
        classes=classes,
        augmentation=get_validation_augmentation(size=(fig_h, fig_w)),
        preprocessing=(
            get_preprocessing(preprocessing_fn) if preprocessing_fn else None
        ),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Trainer の準備

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Trainer の準備
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=1,
        accelerator="gpu" if device == "cuda" else "cpu",
    )

    # 学習開始
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )

    valid_metrics = trainer.validate(
        model,
        dataloaders=valid_loader,
        verbose=False,
    )
    print(f"valid metrics: {valid_metrics}")

    test_metrics = trainer.test(
        model,
        dataloaders=test_loader,
        verbose=False,
    )
    print(f"test metrics: {test_metrics}")

    shutil.copy(
        Path()
        .cwd()
        .joinpath("lightning_logs")
        .joinpath("version_0")
        .joinpath("metrics.csv"),
        Path(__file__).parents[2].joinpath("logs").joinpath("metrics.csv"),
    )
    shutil.copy(
        Path(__file__).parents[2].joinpath("configs").joinpath("params.yaml"),
        Path(__file__).parents[2].joinpath("model").joinpath("params.yaml"),
    )
    metrics_df = pd.read_csv(
        Path()
        .cwd()
        .joinpath("lightning_logs")
        .joinpath("version_0")
        .joinpath("metrics.csv")
    )
    train_loss = metrics_df["train_loss"].dropna().tolist()
    valid_loss = metrics_df["valid_loss"].dropna().tolist()
    train_iou_score = metrics_df["train_dataset_iou"].dropna().tolist()
    valid_iou_score = metrics_df["valid_dataset_iou"].dropna().tolist()

    history = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_iou_score": train_iou_score,
        "valid_iou_score": valid_iou_score,
    }
    metrics_plot(
        history,
        save_file_path=Path(__file__).parents[2]
        .joinpath("logs")
        .joinpath("metrics.png"),
    )

    shutil.rmtree(Path().cwd().joinpath("lightning_logs"))

    best_model = model.model
    torch.save(
        best_model.state_dict(),
        Path(__file__).parents[2].joinpath("model").joinpath("best_model.pth")
    )


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print(
            "Usage: python -m image_segmentation.train "
            "<data_dir> <config_path>"
        )
        sys.exit(1)
    main(
        data_dir=args[1],
        config_path=args[2]
    )
