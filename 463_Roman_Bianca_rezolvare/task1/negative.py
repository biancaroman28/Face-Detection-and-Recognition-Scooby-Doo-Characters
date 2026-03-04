import cv2
import os
import glob
import random
import numpy as np

TRAIN_ROOT = "../../antrenare"
OUT_DIR = "../dataset/task1/"
os.makedirs(OUT_DIR, exist_ok=True)

WIN_W = 64
WIN_H = 80

SCALES = [0.75, 1.0, 1.5]

NEG_PER_IMAGE = 40
NEAR_MISS_RATIO = 0.3   
MAX_NEGATIVES = 180000

STD_THRESH = 10
MIN_FACE_H = 15

IOU_RANDOM_MAX = 0.1
IOU_NEAR_MIN = 0.1
IOU_NEAR_MAX = 0.3


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



ann_files = glob.glob(os.path.join(TRAIN_ROOT, "*_annotations.txt"))


gt_map = {}

for ann in ann_files:
    with open(ann, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue

            img, x1, y1, x2, y2, _ = parts
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            bh = y2 - y1

            bbox = (x1, y1, x2, y2)
            gt_map.setdefault(img, []).append((bbox, bh))


negatives = []

for sub in ["daphne", "fred", "shaggy", "velma"]:
    img_dir = os.path.join(TRAIN_ROOT, sub)
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

    for img_path in img_files:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        gts = gt_map.get(img_name, [])

        target_near = int(NEG_PER_IMAGE * NEAR_MISS_RATIO)
        target_random = NEG_PER_IMAGE - target_near

        got_near = 0
        got_random = 0
        tries = 0

        while (got_near < target_near or got_random < target_random) \
              and len(negatives) < MAX_NEGATIVES \
              and tries < 300:
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

            max_iou = 0.0
            valid = True

            for (gt, bh) in gts:
                iou = compute_iou(cand, gt)

                if bh < MIN_FACE_H:
                    if iou > 0:
                        valid = False
                        break
                else:
                    max_iou = max(max_iou, iou)

            if not valid:
                continue

            is_near = IOU_NEAR_MIN <= max_iou < IOU_NEAR_MAX
            is_random = max_iou < IOU_RANDOM_MAX

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

        if len(negatives) >= MAX_NEGATIVES:
            break

    if len(negatives) >= MAX_NEGATIVES:
        break


negatives = np.array(negatives, dtype=np.uint8)
labels = np.zeros(len(negatives), dtype=np.int32)

np.save(os.path.join(OUT_DIR, "neg_images.npy"), negatives)
# np.save(os.path.join(OUT_DIR, "neg_labels.npy"), labels)

print("Negative extrase:", negatives.shape)
print("Saved in:", OUT_DIR)
