"""
LightningModule 実装

このモジュールは notebooks/test.ipynb の学習ループに合わせた
PyTorch Lightning の `LightningModule` を提供します。

機能:
- smp.DeepLabV3Plus を内部モデルに利用
- DiceLoss を損失に使用（smp の API に合わせて参照）
- IoU を評価指標として計算（smp の存在しない場合は簡易版を利用）
"""

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        arch,
        encoder_name,
        in_channels,
        out_classes,
        learning_rate=2e-4,
        T_MAX=100,
        **kwargs
    ):
        """SegmentationModel の初期化

        Parameters
        ----------
        arch : str
            使用するアーキテクチャ名（例: "deeplabv3plus"）
        encoder_name : str
            使用するエンコーダ名（例: "resnet34"）
        in_channels : int
            入力画像のチャネル数（通常は 3）
        out_classes : int
            出力クラス数（1 の場合は二値セグメンテーション、>1 の場合は多クラス）
        learning_rate : float, optional
            学習率（デフォルトは 2e-4）
        T_MAX : int, optional
            CosineAnnealingLR の T_max（デフォルトは 100）
        """
        super().__init__()
        # モデルの作成
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.learning_rate = learning_rate
        self.T_MAX = T_MAX
        self.out_classes = out_classes
        # 正規化パラメータの登録
        # 画像の前処理に使用
        params = smp.encoders.get_preprocessing_params(encoder_name)
        # 標準偏差と平均をバッファとして登録
        # pretrained = "imagenet"の場合のパラメータ
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        self.register_buffer(
            "std", torch.tensor(params["std"]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "mean", torch.tensor(params["mean"]).view(1, 3, 1, 1)
        )

        # 損失関数の設定
        if out_classes == 1:
            self.loss_fn = smp.losses.DiceLoss(
                smp.losses.BINARY_MODE,
                from_logits=True
            )
        else:
            self.loss_fn = smp.losses.DiceLoss(
                smp.losses.MULTICLASS_MODE,
                from_logits=True
            )

        # 学習・検証・テストステップの出力を保存するリスト
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        mask = self.model(image)
        # 出力を確率に変換
        if self.out_classes == 1:
            mask = torch.sigmoid(mask)
        else:
            mask = torch.softmax(mask, dim=1)
        return mask

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
        assert h % 32 == 0 and w % 32 == 0, "Input height and width must be divisible by 32"

        # マスクの検証
        mask = batch[1]
        assert mask.ndim == 4
        assert torch.all((mask >= 0) & (mask <= 1)), "Mask values must be in [0, 1]"

        # モデルの推論
        logits_mask = self.forward(image)
        # 損失の計算
        loss = self.loss_fn(logits_mask, mask)
        # metrics の計算
        if self.out_classes == 1:
            # 2値分類
            pred_mask = (logits_mask > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(), mask.long(), mode="binary"
            )
        else:
            # 多クラス分類
            pred_mask = logits_mask.argmax(dim=1)
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
