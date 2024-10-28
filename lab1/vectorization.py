import numpy as np
import torch
from PIL import Image
from transformers import pipeline, CLIPModel, CLIPProcessor


def process_resnet(feature: torch.Tensor) -> torch.Tensor:
    ''' Processes the features from the ResNet model.'''
    return feature.mean(-1).mean(-1)


def process_vit(feature: torch.Tensor) -> torch.Tensor:
    ''' Processes the features from the ViT model.'''
    return feature[:, 0, :]


PROCESSORS = {
    "google/vit-base-patch32-224": process_vit,
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


def vectorize_images_with_clip(logger, images: list[Image]):
    ''' Vectorizes images using the CLIP model.'''
    logger.info("Vectorizing images with CLIP...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    image_features = clip_model.get_image_features(**inputs)
    return image_features.detach().numpy()


def vectorize_texts_with_clip(logger, texts: list[str]):
    ''' Vectorizes images using the CLIP model.'''
    logger.info("Vectorizing texts with CLIP...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True)
    text_features = clip_model.get_text_features(**inputs)
    return text_features.detach().numpy()

