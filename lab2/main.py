from pathlib import Path

import click
import numpy as np
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
import textwrap

from lab1.data_manipulation import load_data
from lab1.logger import get_logger
from lab1.vectorization import vectorize_images_with_clip, vectorize_texts_with_clip

BASE_LOGS_PATH = Path("logs")
BASE_IMGS_PATH = BASE_LOGS_PATH / "imgs"
BASE_IMGS_PATH.mkdir(parents=True, exist_ok=True)
logger = get_logger("lab2", BASE_LOGS_PATH)

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

        self.components = None
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> None:
        self.calc_mean_std(X)
        X_std = self.standardize(X)
        cov_matrix = np.cov(X_std.T)
        eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
        idxs = np.argsort(np.abs(eig_values))[::-1]
        self.components = eig_vectors[idxs[:self.n_components]]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.dot(self.standardize(X), self.components.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def calc_mean_std(self, X: np.ndarray) -> None:
        ''' Calculate mean and std of the data '''
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def standardize(self, X: np.ndarray) -> np.ndarray:
        ''' Normalize data by substracting mean and divide by std '''
        return (X - self.mean) / self.std


class KMeans:
    def __init__(self, n_clusters: int, max_iterations: int = 100_000):
        self.n_clusters = n_clusters
        self.max_iter = max_iterations

        self.centroids = None

    def fit(self, X: np.ndarray) -> None:
        self.init_centroids(X)
        for _ in range(self.max_iter):
            clusters = self.assign_clusters(self.centroids, X)
            new_centroids = self.compute_means(clusters, X)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.assign_clusters(self.centroids, X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    def init_centroids(self, X: np.ndarray) -> None:
        centroids = [X[0]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids.append(X[i])

        self.centroids = np.array(centroids)

    def assign_clusters(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        ''' given input data X and cluster centroids assign clusters to samples '''
        clusters = np.zeros(X.shape[0])
        for idx, sample in enumerate(X):
            distances = [self.euclidean_distance(sample, point) for point in centroids]
            clusters[idx] = np.argmin(distances)
        return clusters

    def compute_means(self, clusters: np.ndarray, X: np.ndarray) -> np.ndarray:
        ''' recompute cluster centroids'''
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            new_centroids[k, :] = np.mean(X[clusters == k], axis=0)
        return new_centroids

    def sum_squared_error(self, X: np.ndarray, clusters: np.ndarray) -> float:
        ''' Calculate sum of squared distances between samples and their cluster centroids '''
        sse = 0
        for k in range(self.n_clusters):
            errors = X[clusters == k] - self.centroids[k]
            sse += np.sum(np.sqrt(np.sum(np.power(errors, 2), axis=1)))
        return sse

    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """ Calculates the euclidean distance between two vectors a and b """
        return np.sqrt(np.sum(np.power(a - b, 2)))


def neighbour_search(
    text_req: np.ndarray,
    vImages: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    ''' Search for top_k nearest neighbours in the space '''
    distances = np.linalg.norm(vImages - text_req, axis=1)
    top_k_idx = np.argsort(distances)[:top_k]
    return distances[top_k_idx], top_k_idx


@click.command()
@click.option('--input_path', type=str, help='Path to the input data')
@click.option('--n_components', type=int, help='Number of components')
@click.option('--n_clusters', type=str, help='Number of clusters')
def main(input_path, n_components, n_clusters):
    image_folder = "../dataset/images"
    label_file = "../dataset/labels.csv"
    tsne_kwargs = {"perplexity": 10, "method": "exact"}
    # load image data and text labels
    images, labels, labels_text, comments = load_data(logger, image_folder, label_file)

    vImages = vectorize_images_with_clip(logger, images)
    vText = vectorize_texts_with_clip(logger, comments)

    # PCA or t-SNE on images
    for cls_name, cls, cls_kwargs in [('PCA', PCA, None), ('t-SNE', TSNE, tsne_kwargs)]:
        cls_kwargs = cls_kwargs or {}
        dimred = cls(n_components=2, **cls_kwargs)
        drvImages = dimred.fit_transform(vImages)

        # Visualize 2D and 3D embeddings of images and color points based on labels
        fig = px.scatter(
            x=drvImages[:, 0],
            y=drvImages[:, 1],
            color=labels_text,
            title=f"{cls_name} Embeddings of Images",
            labels={'color': 'Label'}
        )
        fig.show()
        fig.write_html(BASE_IMGS_PATH / f"images_{cls_name}.html")

    # Perform clustering on the embeddings and visualize the results
    pca3_images = PCA(n_components=3).fit_transform(vImages)
    tsne3_images = TSNE(n_components=3, **tsne_kwargs).fit_transform(vImages)

    embeddings_with_names = [
        ("Original", vImages),
        ("PCA3", pca3_images),
        ("t-SNE3", tsne3_images)
    ]

    clusterer = KMeans(n_clusters=3)
    predicted_clusters = clusterer.fit_predict(vImages)
    for view_name, embeddings in embeddings_with_names[1:]:
        fig = px.scatter_3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            color=predicted_clusters.astype(int).astype(str),
            title=f"KMeans Clustering on Original Embeddings (view of {view_name})",
            labels={'color': 'Cluster'}
        )
        fig.show()
        fig.write_html(BASE_IMGS_PATH / f"clusters_original_view_of_{view_name.lower()}.html")

    for embeddings_name, X_values in embeddings_with_names[1:]:
        clusterer = KMeans(n_clusters=3)
        predicted_clusters = clusterer.fit_predict(X_values)

        fig = px.scatter_3d(
            x=X_values[:, 0],
            y=X_values[:, 1],
            z=X_values[:, 2],
            color=predicted_clusters.astype(int).astype(str),
            title=f"KMeans Clustering on {embeddings_name}",
            labels={'color': 'Cluster'}
        )
        fig.show()
        fig.write_html(BASE_IMGS_PATH / f"clusters_{embeddings_name.replace(' ', '_')}.html")

    # Optimal number of clusters
    for embeddings_name, X_values in embeddings_with_names:
        clusters = list(range(1, 21))
        squared_distances = []
        silhouette_scores = [None]
        for n_clusters in clusters:
            clusterer = KMeans(n_clusters=n_clusters)
            predicted_clusters = clusterer.fit_predict(X_values)
            squared_distances.append(clusterer.sum_squared_error(X_values, predicted_clusters))
            if n_clusters > 1:
                silhouette_scores.append(silhouette_score(X_values, predicted_clusters))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=clusters, y=squared_distances, name="Sum of Squared Distances"),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=clusters, y=silhouette_scores, name="Silhouette Score"),
            secondary_y=True
        )
        fig.update_layout(title=f"Elbow Method for {embeddings_name}")
        fig.update_xaxes(title_text="Number of Clusters", range=[clusters[0] - 1, clusters[-1] + 1])
        fig.update_yaxes(title_text="Sum of Squared Distances", secondary_y=False)
        fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
        fig.show()
        fig.write_html(BASE_IMGS_PATH / f"elbow_method_{embeddings_name.lower()}.html")

    # Hierarchical clustering vs KMeans
    # Note: Using PCA3 embeddings for visualization and 6 clusters as from optimal clusters
    for n_clusters_comp in [2, 3, 5, 6]:
        for clusterer_name, clusterer_cls in [("Hierarchical", AgglomerativeClustering), ("KMeans", KMeans)]:
            clusterer = clusterer_cls(n_clusters=n_clusters_comp)
            predicted_clusters = clusterer.fit_predict(pca3_images)
            logger.info(f"{clusterer_name} Clustering with {n_clusters_comp} clusters:")
            for i, count in zip(*np.unique(predicted_clusters, return_counts=True)):
                logger.info(f"\tCluster {i}: {count} samples")
            fig = px.scatter_3d(
                x=pca3_images[:, 0],
                y=pca3_images[:, 1],
                z=pca3_images[:, 2],
                color=predicted_clusters.astype(int).astype(str),
                title=f"{clusterer_name} Clustering with {n_clusters_comp} clusters"
            )
            fig.show()
            fig.write_html(BASE_IMGS_PATH / f"{clusterer_name.lower()}_clustering_pca3_{n_clusters_comp}_clusters.html")
        logger.info("-" * 50)

    # 1. Agglomerative Clustering is more flexible and can be used with different linkage methods
    # 2. KMeans is faster and more scalable
    # 3. KMeans is sensitive to initialization and can converge to local optima
    # 4. Agglomerative Clustering is more sensitive to noise and outliers
    # 5. It looks like Agglomerative clustering tries to "merge" clusters, while KMeans tries to "split" them
    # It is seen on cluster size = 3 and 4, where Agglomerative clustering splits clusters not evenly
    # and KMeans splits them more evenly
    # 6. KMeans is more sensitive to the number of clusters, while Agglomerative Clustering is more robust

    # DBSCAN outlier detection
    clusterer = DBSCAN(eps=8.75, min_samples=5)
    predicted_clusters = clusterer.fit_predict(vImages)
    logger.info("DBSCAN Clustering:")
    for i, count in zip(*np.unique(predicted_clusters, return_counts=True)):
        logger.info(f"\tCluster {i}: {count} samples")
    fig = px.scatter_3d(
        x=pca3_images[:, 0],
        y=pca3_images[:, 1],
        z=pca3_images[:, 2],
        color=predicted_clusters.astype(int).astype(str),
        title="DBSCAN Clustering"
    )
    fig.show()
    fig.write_html(BASE_IMGS_PATH / "dbscan_clustering.html")

    X_train, X_test, y_train, y_test = train_test_split(
        vImages,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"Original data:")
    logger.info(f"\tAccuracy: {accuracy:.2f}")
    logger.info(f"\tF1: {f1:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        vImages[predicted_clusters != -1],
        labels[predicted_clusters != -1],
        test_size=0.2,
        stratify=labels[predicted_clusters != -1]
    )
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"Filtered data:")
    logger.info(f"\tAccuracy: {accuracy:.2f}")
    logger.info(f"\tF1: {f1:.2f}")

    n_closest = 9 - 1
    text_req_idx = np.random.choice(len(comments))
    title_text = '<br>'.join(textwrap.wrap(comments[text_req_idx], 100))
    for name, cls, cls_kwargs in [
        ("Original", None, None),
        ("PCA", PCA, None),
        ("t-SNE", TSNE, tsne_kwargs)
    ]:
        if cls is None:
            distances, nearest_idx = neighbour_search(vText[text_req_idx], vImages, n_closest)
            ground_truth_distance, _ = neighbour_search(vText[text_req_idx], vImages[text_req_idx].reshape(1, -1), 1)
            distances = np.concatenate((ground_truth_distance, distances))
            nearest_idx = np.concatenate(([text_req_idx], nearest_idx))
        else:
            cls_kwargs = cls_kwargs or {}
            dimred = cls(n_components=20, **cls_kwargs)
            drvData = dimred.fit_transform(np.vstack((vImages, vText)))
            drvImages = drvData[:vImages.shape[0]]
            drvText = drvData[vImages.shape[0]:]

            distances, nearest_idx = neighbour_search(drvText[text_req_idx], drvImages, n_closest)
            ground_truth_distance, _ = neighbour_search(drvText[text_req_idx], drvImages[text_req_idx].reshape(1, -1), 1)
            distances = np.concatenate((ground_truth_distance, distances))
            nearest_idx = np.concatenate(([text_req_idx], nearest_idx))

        original_comments = ['<br>'.join(textwrap.wrap(comments[i], 60)) for i in nearest_idx]
        original_labels = [labels_text[i] for i in nearest_idx]
        original_images = [images[i].resize((300, 300)) for i in nearest_idx]
        annotations = list(zip(original_comments, original_labels, distances))
        fig = px.imshow(
            np.array([np.array(img) for img in original_images]),
            facet_col=0,
            facet_col_wrap=3,
            facet_row_spacing=0.2,
            facet_col_spacing=0.2,
            binary_string=True,
            labels={'facet_col': 'text'},
            title=f"{name} Nearest Images for<br>'{title_text}'"
        )
        for annotation in fig.layout.annotations:
            annotation_idx = int(annotation.text.split("=")[-1])
            comment, label, distance = annotations[annotation_idx]
            if annotation_idx == 0:
                annotation.text = f"<b>GROUND TRUTH</b><br>{comment}<br>({label} : {distance:.2f})"
            else:
                annotation.text = f"{comment}<br>({label} : {distance:.2f})"
        fig.update_layout(
            title={
                "x": 0.5,
                "y": 0.95,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 16}
            },
            margin={"t": 150}
        )
        fig.show()
        fig.write_html(BASE_IMGS_PATH / f"nearest_images_{name.lower()}.html")


if __name__ == "__main__":
    main()