import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


X = np.load("../dataset/task1/hog_features.npy")    
y = np.load("../dataset/task1/hog_labels.npy")     

print("Features:", X.shape)
print("Labels:", y.shape)


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        C=0.5,           
        max_iter=15000,
        dual=True        
    ))
])


print("First training")
svm_pipeline.fit(X, y)


Path("../models").mkdir(exist_ok=True)
joblib.dump(svm_pipeline, "../models/model_task1.joblib")

print("\nModel initial saved to ../models/model_task1.joblib")
