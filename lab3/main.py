from functools import partial
from pathlib import Path

import click
from types import SimpleNamespace

import numpy as np
import timm
import torch
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup

from lab3.dataset import FlickrDataset, generate_labels
from lab3.model import CaptioningModel
from lab3.utils import visualize_samples, plot_metrics, plot_sample_predictions


model_config = SimpleNamespace(
    vocab_size = 50_257,
    embed_dim = 768,
    num_heads = 12,
    seq_len = 1024,
    depth = 12,
    attention_dropout = 0.1,
    residual_dropout = 0.1,
    mlp_ratio = 4,
    mlp_dropout = 0.1,
    emb_dropout = 0.1,
)


def collate_fn(batch: list[tuple[Image, str]], tokenizer, transforms):
    images, captions = zip(*batch)
    images = torch.stack([transforms(image) for image in images]).to(torch.float32)
    captions = tokenizer(captions, return_tensors='pt', padding=True, truncation=False)
    return images, captions.input_ids


@click.command()
@click.option('--data_folder', type=str, default='./dataset')
@click.option('--bs', type=int, default=128)
@click.option('--device', type=str, default='cuda')
@click.option('--n_epochs', type=int, default=1000)
@click.option('--lr', type=float, default=1e-3)
def main(data_folder, bs, device, n_epochs, lr):
    torch.set_float32_matmul_precision('high')
    data_folder = Path(data_folder)
    # Create datasets
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    labels_train = data_folder / "labels_train.csv"
    labels_val = data_folder / "labels_val.csv"
    if not labels_train.exists() or not labels_val.exists():
        print("Generating labels...")
        generate_labels(data_folder)

    train_dataset = FlickrDataset(data_folder, labels_train)
    val_dataset = FlickrDataset(data_folder, labels_val)

    # Create model
    model = CaptioningModel(model_config).to(device)
    # model.compile()
    model.pretrained_layers_trainable(trainable=False)
    print(f'trainable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

    # Create tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, num_workers=24, persistent_workers=True, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer, transforms=transforms))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, num_workers=24, persistent_workers=True, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer, transforms=transforms))

    # Visualize a few images and captions
    visualize_samples(train_dataloader)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * n_epochs
    warmup_steps = total_steps // 50
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Define training loop
    train_losses = list()
    train_perplexities = list()
    val_losses = list()
    val_perplexities = list()
    for epoch in range(n_epochs):
        try:
            model.train()
            epoch_loss, epoch_perplexity = 0, 0
            for images, captions in train_dataloader:
                images, captions = images.to(device), captions.to(device)
                bos_tokens = torch.full(captions.shape, tokenizer.bos_token_id, dtype=torch.long).to(device)
                optimizer.zero_grad()

                loss = model(images, captions, labels=captions)
                # loss = model(images, bos_tokens, labels=captions)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_perplexity += torch.exp(loss).item()

            train_losses.append(epoch_loss / len(train_dataloader))
            train_perplexities.append(epoch_perplexity / len(train_dataloader))
            print(f"Epoch {epoch + 1}")
            print(f"\tTrain Loss: {train_losses[-1]:.4f}, Train Perplexity: {train_perplexities[-1]:.4f}")
            # if epoch % 50 != 0:
            #     continue

            model.eval()
            val_loss, val_perplexity = 0, 0
            # texts_train = list()
            # texts_val = list()
            with torch.no_grad():
                # for images, _ in train_dataloader:
                #     if texts_train:
                #         continue
                #     images = images.to(device)
                #     predictions = model.generate(
                #         images[:5],
                #         sequence=torch.full((5, 1), tokenizer.bos_token_id, dtype=torch.long).to(device)
                #     )
                #     texts_train.extend(tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions)

                for images, captions in val_dataloader:
                    images, captions = images.to(device), captions.to(device)
                    bos_tokens = torch.full(captions.shape, tokenizer.bos_token_id, dtype=torch.long).to(device)
                    # loss = model(images, captions, labels=captions)
                    loss = model(images, bos_tokens, labels=captions)

                    val_loss += loss.item()
                    val_perplexity += torch.exp(loss).item()
                    # if not texts_val:
                    #     predictions = model.generate(
                    #         images[:5],
                    #         sequence=torch.full((5, 1), tokenizer.bos_token_id, dtype=torch.long).to(device)
                    #     )
                    #     texts_val.extend(tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions)

            val_losses.append(val_loss / len(val_dataloader))
            val_perplexities.append(val_perplexity / len(val_dataloader))
            print(f"\tVal Loss: {val_losses[-1]:.4f}, Val Perplexity: {val_perplexities[-1]:.4f}")
            # print("\t", texts_train)
            # print("\t", texts_val)
        except KeyboardInterrupt:
            print("Training interrupted.")
            break

    bleu_scores = list()
    generated_captions = list()
    target_captions = list()
    with torch.no_grad():
        for images, captions in val_dataloader:
            images, captions = images.to(device), captions.to(device)
            bos_tokens = torch.full((captions.shape[0], 1), tokenizer.bos_token_id, dtype=torch.long).to(device)

            predictions = model.generate(images, sequence=bos_tokens)
            for pred, target in zip(predictions, captions):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                target_text = tokenizer.decode(target, skip_special_tokens=True)
                generated_captions.append(pred_text)
                target_captions.append(target_text)
                bleu_scores.append(sentence_bleu(pred_text, target_text))


    print(f"Average BLEU score: {np.mean(bleu_scores):.4f}")
    print(generated_captions[:5])
    print(target_captions[:5])

    # Plot metrics
    plot_metrics(train_losses, val_losses)

    # Plot sample predictions
    plot_sample_predictions(model, tokenizer, device, val_dataloader)

if __name__ == '__main__':
    main()