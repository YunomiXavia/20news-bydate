import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_texts(texts, max_features=5000):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X

if __name__ == "__main__":
    texts = np.load('texts.npy', allow_pickle=True)
    X = preprocess_texts(texts)

    # Lưu dữ liệu đã tiền xử lý vào file để sử dụng sau
    with open('tfidf_vectors.npy', 'wb') as f:
        np.save(f, X.toarray())
