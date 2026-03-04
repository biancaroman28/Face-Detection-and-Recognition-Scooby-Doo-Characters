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


X_pos = np.load("../dataset/task1/pos_images.npy")    
X_neg = np.load("../dataset/task1/neg_images.npy")    

print("Pozitive:", X_pos.shape)
print("Negative:", X_neg.shape)

def extract_hog(X):
    feats = []
    for img in tqdm(X):
        # guard strict
        if img is None or img.shape != (WIN_H, WIN_W):
            continue

        f = hog(img, **HOG_PARAMS)
        feats.append(f)

    return np.array(feats, dtype=np.float32)

print("Extract HOG positives")
F_pos = extract_hog(X_pos)
y_pos = np.ones(len(F_pos), dtype=np.int32)

print("Extract HOG negatives")
F_neg = extract_hog(X_neg)
y_neg = np.zeros(len(F_neg), dtype=np.int32)


X = np.vstack((F_pos, F_neg)).astype(np.float32)
y = np.concatenate((y_pos, y_neg)).astype(np.int32)

print("Final feature matrix:", X.shape)
print("Labels:", y.shape)

np.save("../dataset/task1/hog_features.npy", X)
np.save("../dataset/task1/hog_labels.npy", y)
