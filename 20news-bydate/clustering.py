import numpy as np
from sklearn.cluster import KMeans

def apply_kmeans(X, num_clusters, random_state=42):
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans

if __name__ == "__main__":
    X = np.load('tfidf_vectors.npy')
    labels = np.load('labels.npy', allow_pickle=True)

    num_clusters = len(set(labels))
    kmeans = apply_kmeans(X, num_clusters)

    # Lưu nhãn phân cụm vào file để sử dụng sau
    with open('kmeans_labels.npy', 'wb') as f:
        np.save(f, kmeans.labels_)
    with open('kmeans_model.pkl', 'wb') as f:
        import pickle
        pickle.dump(kmeans, f)
