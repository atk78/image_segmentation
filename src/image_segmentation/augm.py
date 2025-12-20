import albumentations as albu


def get_training_augmentation(size: tuple[int, int] | None = None):
    """画像の水増し
    処理： 画像の反転、回転、平行移動、スケーリング、ノイズ付加、パースペクティブ変換、
           画質変化、明るさ・コントラスト変化、色調変化、シャープ化、ぼかし

    Returns
    -------
    _type_
        _description_
    """
    if size is None:
        height, width = 256, 256
    else:
        height, width = size

    train_transform = [
        albu.HorizontalFlip(p=0.5),  # 確率0.5で水平反転
        # 確率1で回転、平行移動、スケーリング
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=1, border_mode=0
        ),
        # 処理: 画像のパディングとクロップ
        albu.PadIfNeeded(
            min_height=height,
            min_width=width,
            always_apply=True,
            border_mode=0
        ),
        # 高さ・幅を FIGURE.SIZE に揃えてクロップ
        albu.RandomCrop(height=height, width=width, always_apply=True),
        # ノイズ付加
        albu.GaussNoise(p=0.2),
        # パースペクティブ変換：画像の視点を変える変換
        # 画像の四隅をランダムに変形
        albu.Perspective(p=0.5),
        # 画質変化
        albu.OneOf(
            [
                albu.CLAHE(p=1),  # コントラスト制限適応ヒストグラム平坦化
                albu.RandomBrightnessContrast(p=1),  # 明るさとコントラストのランダム変化
                albu.RandomGamma(p=1)  # ガンマ補正
            ], p=0.9
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),  # シャープ化
                albu.Blur(blur_limit=3, p=1),  # ぼかし
                albu.MotionBlur(blur_limit=3, p=1)  # 動きのあるぼかし
            ], p=0.9
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),  # 明るさとコントラストのランダム変化
                albu.HueSaturationValue(p=1)  # 色相、彩度、明度の変化
            ], p=0.9
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(size: tuple[int, int] | None = None):
    if size is None:
        height, width = 256, 256
    else:
        height, width = size
    return albu.Compose([albu.PadIfNeeded(height, width)])


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    """前処理
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
