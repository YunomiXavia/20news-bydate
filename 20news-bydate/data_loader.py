import os
import numpy as np

def read_files_from_folder(folder_path):
    texts = []
    labels = []
    for folder_name in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_full_path):
            for file_name in os.listdir(folder_full_path):
                file_path = os.path.join(folder_full_path, file_name)
                with open(file_path, 'r', encoding='latin-1') as file:
                    texts.append(file.read())
                    labels.append(folder_name)
    return texts, labels

if __name__ == "__main__":
    train_folder_path = './20news-bydate-train'
    test_folder_path = './20news-bydate-test'

    train_texts, train_labels = read_files_from_folder(train_folder_path)
    test_texts, test_labels = read_files_from_folder(test_folder_path)

    texts = train_texts + test_texts
    labels = train_labels + test_labels

    # Lưu dữ liệu đã tải vào file để sử dụng sau
    with open('texts.npy', 'wb') as f:
        np.save(f, texts)
    with open('labels.npy', 'wb') as f:
        np.save(f, labels)
