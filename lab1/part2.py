import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from lab1.data_manipulation import load_data, validation_split
from lab1.logger import get_logger
from lab1.models import create_model
from lab1.predictors import voting_predictor
from lab1.validators import create_validator
from lab1.vectorization import vectorize_images


base_path = Path('logs') / 'part2'
imgs_dir = base_path / 'imgs'
imgs_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("part2", base_path)


def print_metrics(y_true, y_pred):
    ''' Prints accuracy and F1 score.'''
    accuracy = accuracy_score(y_true, y_pred)
    logger.debug(f"Accuracy: {accuracy:.2f}")
    f1 = f1_score(y_true, y_pred)
    logger.debug(f"F1: {f1:.2f}")


@click.command()
@click.option("--image_folder", type=str, help="Path to the folder containing images")
@click.option("--label_file", type=str, help="Path to the file containing labels")
@click.option("--embeddings_type", type=str, help="Features generator name")
@click.option("--model_name", type=str, help="Name of the model to use")
@click.option("--validator_name", type=str, help="Name of the validator to use")
@click.option("--voting_method", type=str, help="Method of the voting to use")
@click.option("--test_size", type=float, default=0.2, help="Size of the test split")
def main(image_folder: str, label_file: str, embeddings_type: str, model_name: str, validator_name:str, voting_method: str, test_size: float):
    image_folder = "../dataset/images"
    label_file = "../dataset/labels.csv"
    embeddings_type = "google/vit-base-patch16-224"
    model_name = "logistic_regression_sklearn"
    validator_name = "stratified_k_fold"
    voting_method = "soft"
    test_size = 0.2

    images, labels, _, _ = load_data(logger, image_folder, label_file)

    X = vectorize_images(logger, images, embeddings_type)
    y = labels

    X_train, X_test, y_train, y_test, train_idx, test_idx = validation_split(X, y, test_size)

    validator = create_validator(validator_name)

    logger.debug("=" * 50)
    logger.debug("Training models...")
    result_models = []
    for i, (val_train_idx, val_test_idx) in enumerate(validator(X_train, y_train)):
        logger.debug("*" * 25)
        logger.debug(f"Fold {i + 1}")
        X_val_train_split, y_val_train_split, X_val_test, y_val_test = (
            X_train[val_train_idx], y_train[val_train_idx], X_train[val_test_idx], y_train[val_test_idx]
        )

        model = create_model(model_name)
        model.fit(X_val_train_split, y_val_train_split)
        result_models.append(model)
        y_pred_val = model.predict(X_val_test)
        print_metrics(y_val_test, y_pred_val)

    if len(result_models) == 0:
        raise ValueError("No model was trained.")
    if len(result_models) == 1:
        y_pred = result_models[0].predict(X_test)
    else:
        y_pred = voting_predictor(result_models, X_test, method=voting_method)

    logger.debug("=" * 50)
    logger.debug("Final results:")
    print_metrics(y_test, y_pred)

    # 1. Plot the first 10 test images, and on each image plot the corresponding prediction
    candidates = np.where(y_pred != y_test)[0][:10]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        if len(candidates) <= i:
            break
        candidate_idx = candidates[i]
        original_image = images[test_idx[candidate_idx]]
        filename = os.path.basename(original_image.filename)
        curr_image = original_image.resize((224, 224))
        ax.imshow(curr_image)
        ax.set_title(f"IMG {filename}\nTrue: {y_test[candidate_idx]}, Pred: {y_pred[candidate_idx]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(imgs_dir / "errors.png")
    plt.show()

    # 2. Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    plt.matshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, f'{z}', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    tick_marks = [0, 1]
    labels = ["human", "animal"]
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.yticks(tick_marks, labels, rotation=90)
    plt.xlabel('Predicted label')
    plt.xticks(tick_marks, labels)
    plt.savefig(imgs_dir / "confusion_matrix.png")
    plt.show()



if __name__ == "__main__":
    main()