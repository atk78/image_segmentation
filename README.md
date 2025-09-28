# image_segmentation

シンプルな画像セグメンテーション用リポジトリのREADMEです。
このリポジトリでは、画像データからマスク画像を作成し（前処理）、そのマスクを用いてモデルの学習を行います。

## ディレクトリ構成
- src/
    ソースコード（モデル定義、ヘルパー関数など）
- preprocessing.ipynb
    画像データからマスク画像を作成するノートブック
- training.ipynb
    学習（training）を実行するノートブック
- data/  (推奨)
    - raw/ : 元画像
        - masks/ : 前処理で作成されたマスク
        - splits/ : 訓練/検証の分割情報

## 計算環境
以下の環境で実行している。
- バージョン管理ツール: uv を利用。
- GPU: NVIDIA RTX 4060 Ti（学習に使用）
- CUDA: 12.9（学習用ビルドは CUDA 12.9 を想定）
- NVIDIA ドライバ: CUDA 12.9 をサポートするドライバ
- 主なライブラリ:
    - torch（CUDA 12.9 ビルド）、torchvision、torchaudio
    - numpy、pandas、opencv-python、matplotlib
    - albumentations

## セットアップ（簡潔な手順例）
1. uv で Python を準備／同期:
     - 例:
         - uv python install 3.12
         - uv python pin 3.12
         - uv sync

2. 仮想環境を作成・有効化:
     - python -m venv .venv
     - macOS / Linux: source .venv/bin/activate
     - Windows (PowerShell): .venv\Scripts\Activate.ps1

補足:
- pyproject.toml に依存バージョンが明記されている場合は必ずそちらに従ってください。
- ドライバや CUDA ランタイムの不一致で GPU が使えないことがあるため、nvidia-smi と torch の出力を照らし合わせて確認してください。
- 学習時は GPU メモリに合わせてバッチサイズや画像サイズを調整してください。
- Windows の場合、PowerShell の実行ポリシーや PATH 設定に注意してください。
- 必要であれば requirements.txt や主要なノートブックセルを README に追記します。希望があれば教えてください。

## 使い方
1. preprocessing.ipynb を開く
     - data/raw に元画像を配置します。
     - ノートブック内のセルを順に実行すると、マスク画像が data/masks に出力されます。
     - マスク生成のルール（閾値処理、輪郭抽出、アノテーション変換など）はノートブック内で編集できます。

2. training.ipynb を開く
     - data/masks と data/raw（および splits）を参照してデータローダーを構築します。
     - ハイパーパラメータ（バッチサイズ、学習率、エポック数など）を設定して実行します。
     - 学習済みモデルの保存先はノートブック内で確認・変更してください（例: runs/ または checkpoints/）。

## 実行上の注意
- 大きな画像を扱う場合はメモリに注意し、バッチサイズやリサイズを調整してください。
- 再現性のためにランダムシードを固定することを推奨します。
