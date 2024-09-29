import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def process_resnet(feature: torch.Tensor) -> torch.Tensor:
    ''' Processes the features from the ResNet model.'''
    return feature.mean(-1).mean(-1)


def process_vit(feature: torch.Tensor) -> torch.Tensor:
    ''' Processes the features from the ViT model.'''
    return feature[:, 0, :]


PROCESSORS = {
    "google/vit-base-patch16-224": process_vit,
    "microsoft/resnet-50": process_resnet
}

MODEL_ARGS = {
    "google/vit-base-patch16-224": {"add_pooling_layer": False}
}


def vectorize_images(logger, images: list[Image], model="microsoft/resnet-50"):
    ''' Vectorizes images into a matrix of size (N, D), where N is the number of images, and D is the dimensionality of the image.'''
    logger.info("Vectorizing images...")
    feature_extractor = pipeline(
        'image-feature-extraction',
        model=model,
        use_fast=True,
        framework="pt",
        return_tensors="pt",
        device="cuda:0",
        model_kwargs=MODEL_ARGS.get(model, {})
    )
    features = feature_extractor(images)
    processor = PROCESSORS.get(model, None)
    if processor is None:
        raise ValueError(f"Model {model} not supported.")
    features = [processor(feature).detach().numpy().flatten() for feature in features]
    return np.vstack(features)