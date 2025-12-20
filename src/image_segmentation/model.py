"""統合 LightningModule 実装

このモジュールは、共通の `SegmentationModel` クラスとして

- segmentation_models_pytorch の DeepLabV3+ ベース
- Hugging Face Transformers の SegFormer ベース
- Hugging Face Transformers の Mask2Former ベース

の両方を切り替えて利用できる PyTorch Lightning の `LightningModule` を提供します。

使い方のイメージ:

    model = SegmentationModel(
        model_type="DeepLabV3Plus",  # or "SegFormer"
        hparams={...},                # アーキテクチャ固有の設定
        learning_rate=1e-4,
    )

`hparams` には、例えば以下のような情報を持たせます。

- DeepLabV3Plus の場合:
    - ENCODER: "efficientnet-b0" など
    - ENCODER_WEIGHTS: "imagenet" など
    - IN_CHANNELS: 3
    - OUT_CLASSES: クラス数

- SegFormer の場合:
    - NAME: "nvidia/segformer-b0-finetuned-ade-512-512" など
    - IN_CHANNELS: 3
    - OUT_CLASSES: クラス数

- Mask2Former の場合:
    - NAME: "facebook/mask2former-swin-small-ade-semantic" など
    - IN_CHANNELS: 3
    - OUT_CLASSES: クラス数
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
)


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model_type: str,
        hparams: dict,
        learning_rate: float = 2e-4,
        T_MAX: int = 100,
        **kwargs,
    ) -> None:
        """SegmentationModel の初期化

        Parameters
        ----------
        model_type : str
            使用するモデル種類。
            - "DeepLabV3Plus"
            - "SegFormer"
        hparams : dict
            モデル固有のハイパーパラメータを保持する辞書。
            例)
                DeepLabV3Plus:
                    {
                        "ENCODER": "efficientnet-b0",
                        "ENCODER_WEIGHTS": "imagenet",
                        "IN_CHANNELS": 3,
                        "OUT_CLASSES": 1,
                    }
                SegFormer:
                    {
                        "NAME": "nvidia/segformer-b0-finetuned-ade-512-512",
                        "IN_CHANNELS": 3,
                        "OUT_CLASSES": 1,
                    }
        learning_rate : float, optional
            学習率（デフォルトは 2e-4）
        T_MAX : int, optional
            CosineAnnealingLR の T_max（デフォルトは 100）
        """
        super().__init__()

        self.model_type = model_type
        self.learning_rate = learning_rate
        self.T_MAX = T_MAX

        # DeepLabV3+ / SegFormer / Mask2Former で分岐して内部モデルを構築
        if model_type == "DeepLabV3Plus":
            encoder_name = hparams.get("ENCODER", "efficientnet-b0")
            encoder_weights = hparams.get("ENCODER_WEIGHTS", "imagenet")
            in_channels = int(hparams.get("IN_CHANNELS", 3))
            out_classes = int(hparams.get("OUT_CLASSES", 1))

            self.model = smp.create_model(
                "deeplabv3plus",
                encoder_name=encoder_name,
                in_channels=in_channels,
                classes=out_classes,
                encoder_weights=encoder_weights,
                **kwargs,
            )

            # 正規化パラメータ（smp のエンコーダに合わせる）
            params = smp.encoders.get_preprocessing_params(encoder_name)
            mean = params["mean"]
            std = params["std"]

        elif model_type == "SegFormer":
            model_name = hparams.get(
                "NAME", "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            in_channels = int(hparams.get("IN_CHANNELS", 3))
            out_classes = int(hparams.get("OUT_CLASSES", 1))

            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=out_classes,
                ignore_mismatched_sizes=True,
            )

            # ImageProcessor から正規化パラメータを取得（失敗時は ImageNet 標準値）
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
                mean = processor.image_mean
                std = processor.image_std
            except Exception:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

        elif model_type == "Mask2Former":
            model_name = hparams.get(
                "NAME", "facebook/mask2former-swin-small-ade-semantic"
            )
            in_channels = int(hparams.get("IN_CHANNELS", 3))
            out_classes = int(hparams.get("OUT_CLASSES", 1))

            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name,
                num_labels=out_classes,
                ignore_mismatched_sizes=True,
            )

            # 画像の正規化パラメータを ImageProcessor から取得
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
                mean = processor.image_mean
                std = processor.image_std
            except Exception:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.out_classes = out_classes

        # 正規化パラメータを buffer として登録
        self.register_buffer(
            "std", torch.tensor(std).view(1, in_channels, 1, 1)
        )
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, in_channels, 1, 1)
        )

        # DiceLoss の設定（確率値を入力とするので from_logits=False）
        if self.out_classes == 1:
            self.loss_fn = smp.losses.DiceLoss(
                smp.losses.BINARY_MODE,
                from_logits=False,
            )
        else:
            self.loss_fn = smp.losses.DiceLoss(
                smp.losses.MULTICLASS_MODE,
                from_logits=False,
            )

        # 学習・検証・テストステップの出力を保存するリスト
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """モデル推論

        DeepLabV3+ / SegFormer いずれの場合も、確率マップを返す。
        """
        if self.model_type == "DeepLabV3Plus":
            logits = self.model(image)
            if self.out_classes == 1:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=1)

        if self.model_type == "SegFormer":
            outputs = self.model(pixel_values=image)
            logits = outputs.logits
            # SegFormer の logits は一般に入力サイズの 1/4 解像度なので、
            # 入力画像と同じ解像度にアップサンプルしてから確率化する
            if logits.shape[2:] != image.shape[2:]:
                logits = F.interpolate(
                    logits,
                    size=image.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if self.out_classes == 1:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=1)

        # Mask2Former
        # Query 出力から semantic segmentation の確率マップ (B, C, H, W) を合成する。
        #   class_probs: softmax over (C + 1) then drop "no-object"
        #   mask_probs: sigmoid
        #   semseg_scores = sum_q class_probs[q,c] * mask_probs[q,h,w]
        outputs = self.model(pixel_values=image)
        class_logits = outputs.class_queries_logits  # (B, Q, C+1)
        mask_logits = outputs.masks_queries_logits  # (B, Q, H', W')

        # NOTE:
        #   - class_queries_logits は (C + 1) で最後が no-object
        #   - masks_queries_logits は query ごとのマスク
        # ここでは推論/学習で共通利用できるよう、query 合成して
        # (B, C, H, W) の「確率」っぽい値に正規化して返す。
        class_probs_full = torch.softmax(class_logits, dim=-1)  # (B,Q,C+1)
        mask_probs = torch.sigmoid(mask_logits)  # (B,Q,H',W')

        eps = 1e-6
        if self.out_classes == 1:
            # foreground: index 0, background(no-object): last index
            fg_q = class_probs_full[..., 0]
            bg_q = class_probs_full[..., -1]

            fg_score = torch.einsum("bq,bqhw->bhw", fg_q, mask_probs)
            bg_score = torch.einsum("bq,bqhw->bhw", bg_q, mask_probs)

            fg_prob = fg_score / (fg_score + bg_score + eps)
            fg_prob = fg_prob.unsqueeze(1)  # (B,1,H',W')

            if fg_prob.shape[2:] != image.shape[2:]:
                fg_prob = F.interpolate(
                    fg_prob,
                    size=image.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return fg_prob.clamp(0.0, 1.0)

        # multi-class
        fg_class_probs = class_probs_full[..., : self.out_classes]
        bg_q = class_probs_full[..., -1]

        semseg_scores = torch.einsum(
            "bqc,bqhw->bchw", fg_class_probs, mask_probs
        )
        bg_score = torch.einsum("bq,bqhw->bhw", bg_q, mask_probs).unsqueeze(1)
        denom = semseg_scores.sum(dim=1, keepdim=True) + bg_score + eps
        semseg_probs = semseg_scores / denom

        if semseg_probs.shape[2:] != image.shape[2:]:
            semseg_probs = F.interpolate(
                semseg_probs,
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return semseg_probs.clamp(0.0, 1.0)

    def _resize_to_mask(
        self, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """予測マップをマスクの解像度にリサイズ (主に SegFormer 用)"""
        if pred.shape[2:] != mask.shape[2:]:
            pred = F.interpolate(
                pred,
                size=mask.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return pred

    def shared_step(self, batch, stage):
        """ 共通のステップ処理

        Parameters
        ----------
        batch : tuple
            入力画像とマスクのタプル
        stage : str
            "train", "valid", "test" のいずれか

        Returns
        -------
        dict
            損失とメトリクスの辞書
        """
        # バッチの検証
        image = batch[0]
        h, w = image.shape[2:]

        # 入力サイズの検証
        # DeepLabV3+ は一般に 32 の倍数を前提とするため制約を残す。
        # SegFormer は任意サイズを内部で処理できるため、ここでは制約しない。
        if self.model_type == "DeepLabV3Plus":
            assert (
                h % 32 == 0 and w % 32 == 0
            ), "Input height and width must be divisible by 32 for DeepLabV3+"

        # マスクの検証
        mask = batch[1]
        assert mask.ndim == 4
        assert torch.all((mask >= 0) & (mask <= 1)), (
            "Mask values must be in [0, 1]"
        )

        # モデルの推論（確率マップ）
        probs = self.forward(image)
        # SegFormer は logits 解像度と入力解像度が異なる場合があるのでマスクサイズに揃える
        if self.model_type == "SegFormer":
            probs = self._resize_to_mask(probs, mask)

        # 損失の計算（DiceLoss は確率値を想定）
        loss = self.loss_fn(probs, mask)
        # metrics の計算
        if self.out_classes == 1:
            # 2値分類
            pred_mask = (probs > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(), mask.long(), mode="binary"
            )
        else:
            # 多クラス分類
            pred_mask = probs.argmax(dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(), mask.long(), mode="multiclass"
            )
        # tp: true positives (正解ラベルと予測ラベルが一致した画素数)
        # fp: false positives (正解ラベルが背景で予測ラベルが物体の画素数)
        # fn: false negatives (正解ラベルが物体で予測ラベルが背景の画素数)
        # tn: true negatives (正解ラベルと予測ラベルが一致した背景の画素数)
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs: list, stage: str):
        """共通のエポック終了処理

        Parameters
        ----------
        outputs : list
            各ステップの出力を格納したリスト
        stage : str
            "train", "valid", "test" のいずれか
        """
        # 各ステップの出力を集約してメトリクスを計算
        # loss は平均を計算
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        # 各画素のIoUを計算
        # IoU = TP / (TP + FP + FN) → 1 - DiceLoss に相当
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        # 全体のIoUを計算
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_loss": avg_loss,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        # self.log_dict は LightningModule のメソッド
        # 進捗バーにメトリクスを表示
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # outputのリストを初期化
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        # Trainer の setup 時に呼ばれる
        # optimizer と lr_scheduler を設定して返す
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.T_MAX,
            eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
