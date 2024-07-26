import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def visualize_clusters(X, labels, kmeans_labels):
    svd = TruncatedSVD(n_components=2, random_state=42)
    principal_components = svd.fit_transform(X)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=kmeans_labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-means Clustering of 20 Newsgroups')
    plt.legend(handles=scatter.legend_elements()[0], labels=set(labels), title="Clusters")
    plt.show()

if __name__ == "__main__":
    X = np.load('tfidf_vectors.npy')
    labels = np.load('labels.npy', allow_pickle=True)
    kmeans_labels = np.load('kmeans_labels.npy')

    visualize_clusters(X, labels, kmeans_labels)
