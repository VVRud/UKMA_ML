import click
import numpy as np
from sklearn.metrics import accuracy_score


def load_data(image_folder: str, label_file: str):
    ''' Loads images and labels from the specified folder and file.'''
    # load labels file
    labels = None 

    # load corresponding images
    images = None 

    return images, labels


def vectorize_images(images: np.ndarray):
    ''' Vectorizes images into a matrix of size (N, D), where N is the number of images, and D is the dimensionality of the image.'''
    X = None
    return X


def validation_split(X: np.ndarray, y: np.ndarray):
    ''' Splits data into train and test.'''
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    return X_train, X_test, y_train, y_test


def create_model(model_name: str):
    ''' Creates a model of the specified name. 
    1. Use your LinearRegression implementation,
    2. TODO
    3. 
    Args:
        model_name (str): Name of the model to use.
    Returns:
        model (object): Model of the specified name.
    '''
    model = None
    return model


@click.command()
@click.option("--image_folder", type=str, help="Path to the folder containing images")
@click.option("--label_file", type=str, help="Path to the file containing labels")
@click.option("--model_name", type=str, help="Name of the model to use")
@click.option("--test_size", type=float, default=0.2, help="Size of the test split")
def main(image_folder: str, label_file: str, model_name: str, test_size: float):

    # Create dataset of image <-> label pairs
    images, labels = load_data(image_folder, label_file)

    # preprocess images and labels
    X = vectorize_images(images)
    y = None 

    # split data into train and test
    X_train, X_test, y_train, y_test = validation_split(X, y, test_size)

    # create model
    model = create_model(model_name)

    # Train model using different validation strategies (refere to https://scikit-learn.org/stable/modules/cross_validation.html)
    # 1. Train, validation, test splits: so you need to split train into train and validation 
    # 2. K-fold cross-validation: apply K-fold cross-validation on train data
    # 3. Leave-one-out cross-validation: apply Leave-one-out cross-validation on train data

    # Make a prediction on test data
    y_pred = model.predict(X_test) # in the simpliest validation strategy, in K-fold you will have multiple predictions

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Make error analysis
    # 1. Plot the first 10 test images, and on each image plot the corresponding prediction
    # 2. Plot the confusion matrix



if __name__ == "__main__":
    main()