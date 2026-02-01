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

from .augm import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
)
from .model import SegmentationModel
from .dataset import Dataset
from .visualize import metrics_plot


# ランダムシード固定


def load_params(path: Path) -> dict:
    # YAML を優先的に読み込む。拡張子に応じて処理を切り替える
    if path.suffix in (".yaml", ".yml"):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def main(data_dir: Path, config_path: str | Path):
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    config_path = Path(config_path)
    train_img_dir = data_dir.joinpath("train/img")
    train_mask_dir = data_dir.joinpath("train/mask")
    valid_img_dir = data_dir.joinpath("valid/img")
    valid_mask_dir = data_dir.joinpath("valid/mask")
    test_img_dir = data_dir.joinpath("test/img")
    test_mask_dir = data_dir.joinpath("test/mask")

    if Path().cwd().joinpath("lightning_logs").exists():
        shutil.rmtree(Path().cwd().joinpath("lightning_logs"))

    params = load_params(config_path)
    seed = int(params.get("SEED", 42))
    pl.seed_everything(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    model_params = params.get("MODEL", {})
    train_params = params.get("TRAIN", {})
    figure_params = params.get("FIGURE", {})
    figure_size = figure_params.get("SIZE", [512, 512])
    fig_h, fig_w = int(figure_size[0]), int(figure_size[1])
    class_info = params.get(
        "CLASSES",
        {"background": 0, "object": 255}
    )
    log_dir = params["SAVE_RESULTS"].get("LOG", "logs")
    model_dir = params["SAVE_RESULTS"].get("MODEL", "models")
    if Path(__file__).parents[2].joinpath(log_dir).exists():
        shutil.rmtree(Path(__file__).parents[2].joinpath(log_dir))
    if not Path(__file__).parents[2].joinpath(model_dir).exists():
        shutil.rmtree(Path(__file__).parents[2].joinpath(model_dir))
    Path(__file__).parents[2].joinpath(log_dir).mkdir(parents=True, exist_ok=True)
    Path(__file__).parents[2].joinpath(model_dir).mkdir(parents=True, exist_ok=True)
    # モデル種別とクラス定義を設定ファイルから読み込む
    model_type = model_params.get("TYPE", "model1")
    hparams = model_params.get(
        model_type,
        {
            "ARCHITECTURE": "DeepLabV3Plus",
            "ENCODER": "resnet34",
            "ENCODER_WEIGHTS": "imagenet",
        }
    )

    # device = params.get("DEVICE", "cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = int(train_params.get("BATCH_SIZE", 8))
    lr = float(train_params.get("LR", 1e-4))
    epochs = int(train_params.get("EPOCHS", 40))

    # モデル作成（DeepLabV3+ / SegFormer を切り替え）
    model = SegmentationModel(
        arch=hparams.get("ARCHITECTURE", "DeepLabV3Plus"),
        encoder_name=hparams.get("ENCODER", "resnet34"),
        encoder_weights=hparams.get("ENCODER_WEIGHTS", "imagenet"),
        in_channels=hparams.get("IN_CHANNELS", 3),
        out_classes=len(class_info),
        learning_rate=lr,
    )

    preprocessing_fn = None
    # モデル種別ごとに前処理関数を設定
    encoder = hparams.get("ENCODER")
    encoder_weights = hparams.get("ENCODER_WEIGHTS")
    # smp のエンコーダに合わせた正規化を実施
    try:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder, encoder_weights
        )
    except Exception:
        preprocessing_fn = None

    train_dataset = Dataset(
        train_img_dir,
        train_mask_dir,
        class_info=class_info,
        augmentation=get_training_augmentation(size=(fig_h, fig_w)),
        preprocessing=(
            get_preprocessing(preprocessing_fn) if preprocessing_fn else None
        ),
    )

    valid_dataset = Dataset(
        valid_img_dir,
        valid_mask_dir,
        class_info=class_info,
        augmentation=get_validation_augmentation(size=(fig_h, fig_w)),
        preprocessing=(
            get_preprocessing(preprocessing_fn) if preprocessing_fn else None
        ),
    )

    test_dataset = Dataset(
        test_img_dir,
        test_mask_dir,
        class_info=class_info,
        augmentation=get_validation_augmentation(size=(fig_h, fig_w)),
        preprocessing=(
            get_preprocessing(preprocessing_fn) if preprocessing_fn else None
        ),
    )

    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty. "
            f"Checked: {train_img_dir} (images) and {train_mask_dir} (masks). "
            "Ensure the folders exist and contain matching filenames. "
            "Supported image extensions: .bmp/.png/.jpg/.jpeg/.tif/.tiff"
        )
    if len(valid_dataset) == 0:
        raise ValueError(
            "Validation dataset is empty. "
            f"Checked: {valid_img_dir} (images) and {valid_mask_dir} (masks)."
        )
    if len(test_dataset) == 0:
        raise ValueError(
            "Test dataset is empty. "
            f"Checked: {test_img_dir} (images) and {test_mask_dir} (masks)."
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
        Path(__file__).parents[2].joinpath(log_dir).joinpath("metrics.csv"),
    )
    shutil.copy(
        Path(__file__).parents[2].joinpath("configs").joinpath("params.yaml"),
        Path(__file__).parents[2].joinpath(model_dir).joinpath("params.yaml"),
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
        .joinpath(log_dir)
        .joinpath("metrics.png"),
    )

    shutil.rmtree(Path().cwd().joinpath("lightning_logs"))

    best_model = model.model
    torch.save(
        best_model.state_dict(),
        Path(__file__).parents[2]
        .joinpath(model_dir)
        .joinpath("best_model.pth")
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
        data_dir=Path(args[1]),
        config_path=Path(args[2])
    )
