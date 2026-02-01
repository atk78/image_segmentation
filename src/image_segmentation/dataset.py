from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        class_info: dict,
        augmentation=None,
        preprocessing=None
    ):
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)

        exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        image_files = []
        for ext in exts:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))
        paired = [
            (img_path, masks_dir / img_path.name)
            for img_path in image_files
            if (masks_dir / img_path.name).exists()
        ]

        self.images_fps = [p[0] for p in paired]
        self.masks_fps = [p[1] for p in paired]
        self.ids = [p.name for p in self.images_fps]
        self.class_info = class_info
        self.class_values = [
            v for k, v in class_info.items()
        ]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        image = cv2.imread(str(self.images_fps[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_fps[index]), 0)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)
