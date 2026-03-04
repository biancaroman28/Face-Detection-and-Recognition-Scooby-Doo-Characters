import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


X = np.load(
    "../../dataset/task2/fred/hog_features_fred.npy"
)
y = np.load(
    "../../dataset/task2/fred/hog_labels_fred.npy"
)

print("Initial features:", X.shape)
print("Pozitive:", int((y == 1).sum()))
print("Negative:", int((y == 0).sum()))


X_hard = np.load(
    "../../dataset/task2/fred/hard_neg_feats_fred.npy"
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


print("Training Fred + hard negatives")
svm_pipeline.fit(X_all, y_all)


Path("../../models").mkdir(exist_ok=True)
joblib.dump(
    svm_pipeline,
    "../../models/model_fred_final.joblib"
)

print("\nModel FINAL salvat: models/model_fred_final.joblib")

svm = svm_pipeline.named_steps["svm"]
print("Coef shape:", svm.coef_.shape)
print("Bias:", svm.intercept_)
