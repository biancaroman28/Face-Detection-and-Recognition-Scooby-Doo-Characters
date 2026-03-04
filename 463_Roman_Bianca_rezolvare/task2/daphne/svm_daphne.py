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


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        C=0.5,         
        max_iter=15000,
        dual=True       
    ))
])


print("Training initial Daphne")
svm_pipeline.fit(X, y)


Path("../../models").mkdir(exist_ok=True)
joblib.dump(
    svm_pipeline,
    "../../models/model_daphne.joblib"
)

print("\nModel initial salvat: models/model_daphne.joblib")


svm = svm_pipeline.named_steps["svm"]
print("Coef shape:", svm.coef_.shape)
print("Bias:", svm.intercept_)
