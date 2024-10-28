import os

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_data(logger, image_folder: str, label_file: str):
    ''' Loads images and labels from the specified folder and file.'''
    # load labels file
    logger.debug("Loading data...")
    labels_df = pd.read_csv(label_file, delimiter="|")
    labels_df["class"] = (labels_df["label"] == "animal").astype(np.int32)
    labels_df = labels_df.sort_values(by="image_name").reset_index(drop=True)

    image_names = labels_df["image_name"].tolist()
    labels = labels_df["class"].to_numpy()
    labels_text = labels_df["label"].tolist()
    comments = labels_df["comment"].tolist()

    # load corresponding images
    images = []
    for image_name in image_names:
        image_file = Image.open(os.path.join(image_folder, image_name))
        if image_file.mode != "RGB":
            image_file = image_file.convert("RGB")
        images.append(image_file)

    return images, labels, labels_text, comments,


def validation_split(X: np.ndarray, y: np.ndarray, test_size: float):
    ''' Splits data into train and test.'''
    return train_test_split(X, y, np.arange(X.shape[0]), test_size=test_size)
