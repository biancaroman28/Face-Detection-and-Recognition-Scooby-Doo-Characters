import cv2
import os
import glob
import random
import numpy as np


TRAIN_ROOT = "../../../antrenare"

TARGET_CHAR = "daphne"
OTHER_CHARS = ["fred", "shaggy", "velma"]

OUT_DIR = "../../dataset/task2/daphne"
os.makedirs(OUT_DIR, exist_ok=True)

WIN_W = 64
WIN_H = 80

SCALES = [0.75, 1.0, 1.5]

NEG_PER_IMAGE = 40
NEAR_MISS_RATIO = 0.3
MAX_NEGATIVES = 120000   

STD_THRESH = 10
MIN_FACE_H = 15

IOU_RANDOM_MAX = 0.1
IOU_NEAR_MIN = 0.1
IOU_NEAR_MAX = 0.35


def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


gt_daphne = {}   
gt_others = {}   


with open(os.path.join(TRAIN_ROOT, "daphne_annotations.txt")) as f:
    for line in f:
        img, x1, y1, x2, y2, label = line.strip().split()
        if label != TARGET_CHAR:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if y2 - y1 >= MIN_FACE_H:
            gt_daphne.setdefault(img, []).append((x1, y1, x2, y2))

# Other characters annotations
for ch in OTHER_CHARS:
    with open(os.path.join(TRAIN_ROOT, f"{ch}_annotations.txt")) as f:
        for line in f:
            img, x1, y1, x2, y2, label = line.strip().split()
            if label == TARGET_CHAR:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if y2 - y1 >= MIN_FACE_H:
                gt_others.setdefault(img, []).append((x1, y1, x2, y2))


negatives = []

for sub in [TARGET_CHAR] + OTHER_CHARS:
    img_dir = os.path.join(TRAIN_ROOT, sub)
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

    for img_path in img_files:
        if len(negatives) >= MAX_NEGATIVES:
            break

        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        gts_daphne = gt_daphne.get(img_name, [])
        gts_other = gt_others.get(img_name, [])

        target_near = int(NEG_PER_IMAGE * NEAR_MISS_RATIO)
        target_random = NEG_PER_IMAGE - target_near

        got_near = 0
        got_random = 0
        tries = 0

        while (got_near < target_near or got_random < target_random) \
              and tries < 300 \
              and len(negatives) < MAX_NEGATIVES:

            tries += 1

            scale = random.choice(SCALES)
            w = int(WIN_W * scale)
            h = int(WIN_H * scale)
            if w >= W or h >= H:
                continue

            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            x2 = x1 + w
            y2 = y1 + h
            cand = (x1, y1, x2, y2)

            
            if any(compute_iou(cand, gt) >= IOU_NEAR_MIN for gt in gts_daphne):
                continue

        
            max_iou_other = max(
                [compute_iou(cand, gt) for gt in gts_other],
                default=0.0
            )

            is_near = IOU_NEAR_MIN <= max_iou_other < IOU_NEAR_MAX
            is_random = max_iou_other < IOU_RANDOM_MAX

            if is_near and got_near < target_near:
                pass
            elif is_random and got_random < target_random:
                pass
            else:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (WIN_W, WIN_H))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            if gray.std() < STD_THRESH:
                continue

            negatives.append(gray)

            if is_near:
                got_near += 1
            else:
                got_random += 1


negatives = np.array(negatives, dtype=np.uint8)
labels = np.zeros(len(negatives), dtype=np.int32)

np.save(os.path.join(OUT_DIR, "neg_images.npy"), negatives)
# np.save(os.path.join(OUT_DIR, "neg_labels.npy"), labels)

print("Negative Daphne:", negatives.shape)
print("Saved in:", OUT_DIR)
