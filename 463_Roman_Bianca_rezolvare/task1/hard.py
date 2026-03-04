import cv2
import os
import glob
import random
import numpy as np
from skimage.feature import hog
import joblib
from collections import defaultdict
from tqdm import tqdm


TRAIN_ROOT = "../../antrenare"

GT_FILES = {
    "daphne": "daphne_annotations.txt",
    "fred": "fred_annotations.txt",
    "shaggy": "shaggy_annotations.txt",
    "velma": "velma_annotations.txt",
}

MODEL_PATH = "../models/model_task1.joblib"

OUT_DIR = "../dataset/task1/"
os.makedirs(OUT_DIR, exist_ok=True)

WIN_W, WIN_H = 64, 80
STEP = 12                         
SCALES = [1.2, 1.0, 0.8, 0.6]

MIN_FACE_H = 15
IOU_FP_MAX = 0.2

TOP_FP_PER_IMAGE = 6              
MAX_HARD_NEG = 30000             
IMG_FRACTION = 0.30              


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
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


gt_map = defaultdict(list)

for folder, ann_file in GT_FILES.items():
    with open(os.path.join(TRAIN_ROOT, ann_file)) as f:
        for line in f:
            img, x1, y1, x2, y2, _ = line.strip().split()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if (y2 - y1) >= MIN_FACE_H:
                gt_map[(folder, img)].append((x1, y1, x2, y2))


pipeline = joblib.load(MODEL_PATH)
svm = pipeline.named_steps["svm"]
scaler = pipeline.named_steps["scaler"]

hard_feats = []

print("HARD NEGATIVE MINING START")


for folder in ["daphne", "fred", "shaggy", "velma"]:
    img_dir = os.path.join(TRAIN_ROOT, folder)
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    random.shuffle(img_files)

    img_files = img_files[:int(len(img_files) * IMG_FRACTION)]

    for img_path in tqdm(img_files, desc=f"HM {folder}"):
        if len(hard_feats) >= MAX_HARD_NEG:
            break

        img = cv2.imread(img_path)
        if img is None:
            continue

        h0, w0 = img.shape[:2]
        img_name = os.path.basename(img_path)
        gts = gt_map.get((folder, img_name), [])

        fp_scores = []

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
                    feat = scaler.transform(feat)
                    score = float(svm.decision_function(feat)[0])

                    if score < 0:
                        continue   

                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + WIN_W) / scale)
                    y2 = int((y + WIN_H) / scale)

                    max_iou = max([iou((x1,y1,x2,y2), gt) for gt in gts], default=0.0)

                    if max_iou < IOU_FP_MAX:
                        fp_scores.append((score, gray.copy()))

        fp_scores.sort(key=lambda x: x[0], reverse=True)

        for score, patch in fp_scores[:TOP_FP_PER_IMAGE]:
            hard_feats.append(hog(patch, **HOG_PARAMS))
            if len(hard_feats) >= MAX_HARD_NEG:
                break

    if len(hard_feats) >= MAX_HARD_NEG:
        break


hard_feats = np.array(hard_feats, dtype=np.float32)
np.save(os.path.join(OUT_DIR, "hard_neg_feats.npy"), hard_feats)

print("Hard negatives:", hard_feats.shape)
print("Saved to:", OUT_DIR)
