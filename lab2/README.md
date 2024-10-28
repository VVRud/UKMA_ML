# Homework 2

You are given a Python file with either blank spaces that need to be filled in or bullet points outlining what needs to be implemented. Feel free to modify it as you wish.

1. Load the Flickr30k dataset.
2. Vectorize the images using a method of your choice (preferably the CLIP neural network).
3. Dimensionality reduction:
    - Implement a PCA class with `fit` and `transform` methods.
    - Visualize PCA with 2 and 3 components using Plotly (in interactive mode).
    - Visualize t-SNE (from sklearn) with 2 and 3 components using Plotly.
    - Analyze what you observed in 2-4 sentences.
4. Clustering:
    - Implement a K-means clustering class.
    - Perform K-means clustering on the original vectors and after PCA with 3 components.
    - Visualize, side-by-side, sample vectors with color-coded cluster labels for both the original vectors and those after PCA with 3 components.
    - Select the optimal number of clusters using a method of your choice.
    - Compare the results and determine which approach is better.
    - Perform hierarchical clustering on the samples after PCA and visualize the results.
    - Compare the hierarchical clustering results with the K-means clusters.
5. Outlier detection:
    - Use a sample training split from Homework 1 and apply the DBSCAN algorithm to detect outliers.
    - Visualize the cluster labels of the samples.
    - Remove outliers from the training dataset and retrain your models.
    - Compare the results with those from Homework 1 and determine whether they improved.
6. Contrastive search:
    - Vectorize a few textual descriptions.
    - Apply the same dimensionality reduction techniques used for the image vectors (PCA, t-SNE).
    - Search for the nearest 5-10 vectors in the space.
    - Analyze the nearest images in the space relative to a given text input and compare them with the ground truth image descriptions.
    - Save a figure of a few text requests, along with the nearest neighbor results and corresponding descriptions.

Note: For Plotly, either convert `main.py` to a Jupyter Notebook or create an HTML page.


# Answers

1. Dimensionality reduction
   - PCA just reduces dimensionality without losing much information
   - t-SNE is better for visualization because it tries to preserve the local structure of the data 
   - PCA is better for clustering because it tries to preserve the global structure of the data
2. Clustering. Optimal clusters:
   - **Original:** 5 
   - **PCA with 3 components:** 6 
   - **t-SNE with 3 components:** 2
3. Clustering. Hierarchical clustering vs KMeans:
   - Agglomerative Clustering is more flexible and can be used with different linkage methods
   - KMeans is faster and more scalable
   - KMeans is sensitive to initialization and can converge to local optima
   - Agglomerative Clustering is more sensitive to noise and outliers
   - It looks like Agglomerative clustering tries to "merge" clusters, while KMeans tries to "split" them. It wa observed on cluster size = 3 and 4, where Agglomerative clustering splits clusters not evenly and KMeans splits them more evenly
   - KMeans is more sensitive to the number of clusters, while Agglomerative Clustering is more robust
4. Outlier detection
   - DBSCAN is more robust to noise and outliers
   - DBSCAN can detect clusters of arbitrary shapes
   - DBSCAN is sensitive to the epsilon and min_samples parameters
   - DBSCAN is not suitable for high-dimensional data
   - Filtering out outliers with DBSCAN improves the performance of the model
5. Contrastive search
   - PCA performs poorly if trying to find nearest neighbours on 2, 3 and even 5 components
   - t-SNE performs better, but still not as good as on the original data
   - The original data is the best for finding nearest neighbours
   - The most likely reason is that PCA and t-SNE lose some information during dimensionality reduction and it is harder to find nearest neighbours in the reduced space
   - Mostly, the nearest neighbours are found in the same class, but sometimes they are not
   - It is a bit better with 20 components, but still not as good as the original data
