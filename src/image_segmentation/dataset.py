from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    CLASSES = ["outside", "lung", "heart", "body"]

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        classes: list[str] = None,
        augmentation=None,
        preprocessing=None
    ):
        self.ids = [file.name for file in Path(images_dir).glob("*.bmp")]
        self.images_fps = [
            Path(images_dir).joinpath(image_id) for image_id in self.ids
        ]
        self.masks_fps = [
            Path(masks_dir).joinpath(image_id) for image_id in self.ids
        ]

        self.class_values = [
            self.CLASSES.index(cls.lower()) for cls in classes
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
