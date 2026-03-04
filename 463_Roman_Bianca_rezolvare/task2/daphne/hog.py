import numpy as np
from skimage.feature import hog
from tqdm import tqdm


WIN_W = 64
WIN_H = 80

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True
)

X_pos = np.load(
    "../../dataset/task2/daphne/pos_images.npy"
)   

X_neg = np.load(
    "../../dataset/task2/daphne/neg_images.npy"
)   

print("Pozitive Daphne:", X_pos.shape)
print("Negative Daphne:", X_neg.shape)


def extract_hog(X):
    feats = []
    for img in tqdm(X):
      
        if img is None or img.shape != (WIN_H, WIN_W):
            continue
        feats.append(hog(img, **HOG_PARAMS))
    return np.array(feats, dtype=np.float32)

print("Extract HOG positives (Daphne)")
F_pos = extract_hog(X_pos)
y_pos = np.ones(len(F_pos), dtype=np.int32)

print("Extract HOG negatives (non-Daphne)")
F_neg = extract_hog(X_neg)
y_neg = np.zeros(len(F_neg), dtype=np.int32)

X = np.vstack((F_pos, F_neg)).astype(np.float32)
y = np.concatenate((y_pos, y_neg)).astype(np.int32)

print("Final feature matrix:", X.shape)
print("Labels:", y.shape)

np.save("../../dataset/task2/daphne/hog_features_daphne.npy", X)
np.save("../../dataset/task2/daphne/hog_labels_daphne.npy", y)

print("Saved:")
print(" - ../../dataset/task2/daphne/hog_features_daphne.npy")
print(" - ../../dataset/task2/daphne/hog_labels_daphne.npy")
