import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Hàm để đọc tất cả các tệp văn bản từ một thư mục và các thư mục con
def read_data_from_folder(folder_path):
    data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Chỉ thêm vào danh sách nếu nội dung không trống
                        data.append(content)
                    else:
                        print(f"File '{file_path}' is empty or contains only whitespace.")
    return data

# Đường dẫn tới các thư mục dữ liệu
train_folder = './20news-bydate-train'
test_folder = './20news-bydate-test'

# Đọc dữ liệu từ các thư mục
train_data = read_data_from_folder(train_folder)
test_data = read_data_from_folder(test_folder)

# In thông tin chi tiết về dữ liệu
print(f"Number of training documents: {len(train_data)}")
print(f"Number of testing documents: {len(test_data)}")

# Kiểm tra xem dữ liệu có bị trống không
if not train_data:
    raise ValueError("Train data is empty. Please check your dataset.")
if not test_data:
    raise ValueError("Test data is empty. Please check your dataset.")

# Biến đổi văn bản thành ma trận TF-IDF, giới hạn số lượng từ vựng
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

# Áp dụng K-means clustering trên tập train
num_clusters = 20  # Số lượng clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_train)

# Giảm chiều dữ liệu để trực quan hóa (sử dụng TruncatedSVD)
svd = TruncatedSVD(n_components=2, random_state=42)
X_train_svd = svd.fit_transform(X_train)

# Lấy nhãn của từng cluster
labels_train = kmeans.labels_

# Trực quan hóa kết quả
plt.figure(figsize=(10, 7))
for i in range(num_clusters):
    points = X_train_svd[labels_train == i]
    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i}')

plt.title("K-means clustering on custom train dataset (SVD-reduced data)")
plt.xlabel("SVD component 1")
plt.ylabel("SVD component 2")
plt.legend()
plt.show()
