import os
import cv2
import glob
import numpy as np
from skimage.feature import hog
import joblib

TEST_ROOT = "../testare/"

OUT_ROOT = "../evaluare/fisiere_solutie/463_Roman_Bianca"

OUT_TASK1 = os.path.join(OUT_ROOT, "task1")
OUT_TASK2 = os.path.join(OUT_ROOT, "task2")


os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(OUT_TASK1, exist_ok=True)
os.makedirs(OUT_TASK2, exist_ok=True)

MODELS = {
    "task1_all": "models/model_task1_final.joblib",
    "daphne":    "models/model_daphne_final.joblib",
    "fred":      "models/model_fred_final.joblib",
    "shaggy":    "models/model_shaggy_final.joblib",
    "velma":     "models/model_velma.joblib",
}

WIN_W, WIN_H = 64, 80
STEP = 8
SCALES = [1.6, 1.3, 1.0, 0.8, 0.65]

TOP_K_PER_IMAGE = 120
TOP_K_PRENMS = TOP_K_PER_IMAGE * 6
NMS_IOU = 0.3
SCORE_FLOOR = -1.0


HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True
)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter) if area_a + area_b - inter > 0 else 0.0


def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        order = np.array([j for j in order[1:] if iou(boxes[i], boxes[j]) < thr])
    return keep


models = {}
for name, path in MODELS.items():
    pipe = joblib.load(path)
    models[name] = {
        "svm": pipe.named_steps["svm"],
        "scaler": pipe.named_steps["scaler"]
    }
    print(f"Loaded model: {name}")


results = {
    name: {"boxes": [], "scores": [], "files": []}
    for name in MODELS
}


img_list = sorted(glob.glob(os.path.join(TEST_ROOT, "*.jpg")))
print("Images:", len(img_list))


for img_path in img_list:
    img = cv2.imread(img_path)
    if img is None:
        continue

    h0, w0 = img.shape[:2]
    img_name = os.path.basename(img_path)

    per_model = {name: {"boxes": [], "scores": []} for name in MODELS}

    for scale in SCALES:
        scaled = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
        sh, sw = scaled.shape[:2]

        for y in range(0, sh - WIN_H, STEP):
            for x in range(0, sw - WIN_W, STEP):
                crop = scaled[y:y+WIN_H, x:x+WIN_W]
                if crop.shape[:2] != (WIN_H, WIN_W):
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                feat = hog(gray, **HOG_PARAMS).reshape(1, -1)

                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + WIN_W) / scale)
                y2 = int((y + WIN_H) / scale)
                box = [x1, y1, x2, y2]

                for name, mdl in models.items():
                    f = mdl["scaler"].transform(feat)
                    score = float(mdl["svm"].decision_function(f)[0])
                    if score < SCORE_FLOOR:
                        continue
                    per_model[name]["boxes"].append(box)
                    per_model[name]["scores"].append(score)

 
    for name in MODELS:
        boxes = per_model[name]["boxes"]
        scores = per_model[name]["scores"]

        if len(scores) > TOP_K_PRENMS:
            idx = np.argsort(scores)[::-1][:TOP_K_PRENMS]
            boxes = [boxes[i] for i in idx]
            scores = [scores[i] for i in idx]

        keep = nms(boxes, scores, NMS_IOU)
        keep = sorted(keep, key=lambda i: scores[i], reverse=True)[:TOP_K_PER_IMAGE]

        for i in keep:
            results[name]["boxes"].append(boxes[i])
            results[name]["scores"].append(scores[i])
            results[name]["files"].append(img_name)

        print(f"{img_name} | {name}: {len(keep)}")



# Task 1
np.save(os.path.join(OUT_TASK1, "detections_all_faces.npy"),
        np.array(results["task1_all"]["boxes"], dtype=np.int32))
np.save(os.path.join(OUT_TASK1, "scores_all_faces.npy"),
        np.array(results["task1_all"]["scores"], dtype=np.float32))
np.save(os.path.join(OUT_TASK1, "file_names_all_faces.npy"),
        np.array(results["task1_all"]["files"]))

# Task 2
for name in ["daphne", "fred", "shaggy", "velma"]:
    np.save(os.path.join(OUT_TASK2, f"detections_{name}.npy"),
            np.array(results[name]["boxes"], dtype=np.int32))
    np.save(os.path.join(OUT_TASK2, f"scores_{name}.npy"),
            np.array(results[name]["scores"], dtype=np.float32))
    np.save(os.path.join(OUT_TASK2, f"file_names_{name}.npy"),
            np.array(results[name]["files"]))

print("\nDONE")
