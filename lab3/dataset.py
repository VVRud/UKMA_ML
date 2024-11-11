from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def generate_labels(data_folder: Path):
    data = pd.read_csv(data_folder / "labels.csv", delimiter="|", usecols=["image_name", "comment", "label"])
    train, val = train_test_split(data, test_size=0.2, stratify=data["label"])
    train.to_csv(data_folder / "labels_train.csv", index=False)
    val.to_csv(data_folder / "labels_val.csv", index=False)


class FlickrDataset(Dataset):
    def __init__(self, root_dir: Path, labels_file: Path, eos_token: str = "<|endoftext|>"):
        self.root_dir = root_dir
        self.labels_file = labels_file
        self.eos_token = eos_token
        self.images_dir = root_dir / "images"
        self.data = pd.read_csv(self.labels_file)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[Image, str]:
        img_name = self.images_dir / self.data.iloc[idx]["image_name"]
        image = Image.open(img_name)
        image = self._transform_image(image)
        comment = self._transform_comment(self.data.iloc[idx]["comment"])
        return image, comment

    def _transform_image(self, image: Image) -> Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _transform_comment(self, comment: str) -> str:
        return comment + self.eos_token