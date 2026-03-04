# import numpy as np
# from sklearn.linear_model import SGDClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# import joblib
# from pathlib import Path

# # =========================
# # LOAD FEATURES – DAPHNE
# # =========================
# X = np.load(
#     "dataset/task2/daphne/hog_features_daphne.npy",
#     mmap_mode="r"     # <<< NU încarcă totul în RAM
# )
# y = np.load(
#     "dataset/task2/daphne/hog_labels_daphne.npy"
# )

# print("Features shape:", X.shape)
# print("Pozitive:", int((y == 1).sum()))
# print("Negative:", int((y == 0).sum()))

# # =========================
# # PIPELINE – SVM LINIAR (SGD)
# # =========================
# svm_pipeline = Pipeline([
#     # with_mean=False este CRUCIAL pentru memorie (date dense, multe)
#     ("scaler", StandardScaler(with_mean=False)),
#     ("svm", SGDClassifier(
#         loss="hinge",        # SVM clasic
#         alpha=1e-4,          # ~ echivalent C ≈ 0.5
#         max_iter=30,         # suficient pt convergență
#         tol=1e-3,
#         n_jobs=-1,
#         random_state=42
#     ))
# ])

# # =========================
# # TRAIN
# # =========================
# print("Training SGD SVM (Daphne)...")
# svm_pipeline.fit(X, y)

# # =========================
# # SAVE MODEL
# # =========================
# Path("models").mkdir(exist_ok=True)
# joblib.dump(svm_pipeline, "models/model_daphne.joblib")

# print("\nModel salvat: models/model_daphne.joblib")

# # =========================
# # QUICK SANITY CHECK
# # =========================
# svm = svm_pipeline.named_steps["svm"]
# print("Coef shape:", svm.coef_.shape)
# print("Bias:", svm.intercept_)

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


X = np.load(
    "../../dataset/task2/daphne/hog_features_daphne.npy"
)
y = np.load(
    "../../dataset/task2/daphne/hog_labels_daphne.npy"
)

print("Initial features:", X.shape)
print("Pozitive:", int((y == 1).sum()))
print("Negative:", int((y == 0).sum()))


X_hard = np.load(
    "../../dataset/task2/daphne/hard_neg_feats_daphne.npy"
)
y_hard = np.zeros(len(X_hard), dtype=np.int32)

print("Hard negatives:", X_hard.shape)


X_all = np.vstack([X, X_hard]).astype(np.float32)
y_all = np.concatenate([y, y_hard])

print("FINAL dataset:", X_all.shape)
print("Final positives:", int((y_all == 1).sum()))
print("Final negatives:", int((y_all == 0).sum()))


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),  
    ("svm", LinearSVC(
        C=0.5,         
        max_iter=15000,
        dual=False      
    ))
])


print("Training Daphne + hard negatives")
svm_pipeline.fit(X_all, y_all)


Path("../../models").mkdir(exist_ok=True)
joblib.dump(
    svm_pipeline,
    "../../models/model_daphne_final.joblib"
)

print("\nModel FINAL salvat: models/model_daphne_final.joblib")

svm = svm_pipeline.named_steps["svm"]
print("Coef shape:", svm.coef_.shape)
print("Bias:", svm.intercept_)
