import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


X = np.load("../dataset/task1/hog_features.npy", mmap_mode="r")
y = np.load("../dataset/task1/hog_labels.npy")

print("Original:", X.shape)

X_hard = np.load("../dataset/task1/hard_neg_feats.npy")
y_hard = np.zeros(len(X_hard), dtype=np.int32)

print("Hard neg:", X_hard.shape)


X_all = np.concatenate([X, X_hard], axis=0).astype(np.float32)
y_all = np.concatenate([y, y_hard], axis=0)

print("Final dataset:", X_all.shape)


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        C=0.5,
        dual=False,          
        max_iter=20000
    ))
])


print("Training SVM with hard negatives")
svm_pipeline.fit(X_all, y_all)

Path("../models").mkdir(exist_ok=True)
joblib.dump(svm_pipeline, "../models/model_task1_final.joblib")

print("Saved: ../models/model_task1_final.joblib")
