import random

import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from lab3.model import CaptioningModel
import matplotlib.pyplot as plt


def visualize_samples(data_loader: DataLoader, num_samples: int = 6):
    cols = int(num_samples ** 0.5)
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(15, 30))

    for i in range(num_samples):
        image, caption = data_loader.dataset[random.randint(0, len(data_loader.dataset) - 1)]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(caption.replace("<|endoftext|>", ""))
        plt.axis('off')
    plt.savefig("lab3/samples.png")
    plt.show()


def plot_metrics(train_losses: list[float], val_losses: list[float]):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig("lab3/losses.png")
    plt.show()


def plot_sample_predictions(model: CaptioningModel, tokenizer: GPT2TokenizerFast, device: str, data_loader: DataLoader, num_samples: int = 6):
    model.eval()
    samples = [data_loader.dataset[random.randint(0, len(data_loader.dataset) - 1)] for _ in range(num_samples)]
    images, captions = data_loader.collate_fn(samples)
    images = images.to(device)
    captions = captions.to(device)

    bos_tokens = torch.full((num_samples, 1), tokenizer.bos_token_id, dtype=torch.long).to(images.device)
    predictions = model.generate(images, sequence=bos_tokens)
    generated_captions = []
    for pred, target in zip(predictions, captions):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        generated_captions.append(pred_text)

    cols = int(num_samples ** 0.5)
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(15, 30))
    for i, (image, target_caption) in enumerate(samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(generated_captions[i].replace("<|endoftext|>", ""))
        plt.axis('off')
    plt.savefig("lab3/predictions.png")
    plt.show()
