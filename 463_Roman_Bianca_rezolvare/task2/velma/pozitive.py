import cv2
from pathlib import Path
import random
import numpy as np


BASE_DIR = Path("../../../antrenare/")

TARGET_CHAR = "velma"

ANN_FILES = {
    "daphne": "daphne_annotations.txt",
    "fred": "fred_annotations.txt",
    "shaggy": "shaggy_annotations.txt",
    "velma": "velma_annotations.txt",
}

OUT_DIR = Path("../../dataset/task2/velma/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_W = 64
WINDOW_H = 80

JITTERS_PER_BOX = 4
SHIFT_FRAC = 0.08
SCALE_FRAC = 0.10

MIN_FACE_H = 15
MIN_IOU_JITTER = 0.6


def compute_iou(a, b):
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


positives = []
pos_heights = []

for folder, ann_file in ANN_FILES.items():
    img_dir = BASE_DIR / folder
    ann_path = BASE_DIR / ann_file

    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue

            img_name, x1, y1, x2, y2, label = parts

          
            if label.lower() != TARGET_CHAR:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            bh = y2 - y1
            bw = x2 - x1
            if bh < MIN_FACE_H:
                continue

            img_path = img_dir / img_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            H, W = img.shape[:2]

        
            if bh < 40:
                PAD_FRAC = 0.18
            elif bh < 80:
                PAD_FRAC = 0.14
            else:
                PAD_FRAC = 0.10

            pad_w = bw * PAD_FRAC
            pad_h = bh * PAD_FRAC

            cx = x1 + bw / 2
            cy = y1 + bh / 2

            bw_p = bw + 2 * pad_w
            bh_p = bh + 2 * pad_h

            ax1 = int(cx - bw_p / 2)
            ay1 = int(cy - bh_p / 2)
            ax2 = int(cx + bw_p / 2)
            ay2 = int(cy + bh_p / 2)

            ax1 = max(0, min(ax1, W - 1))
            ay1 = max(0, min(ay1, H - 1))
            ax2 = max(0, min(ax2, W))
            ay2 = max(0, min(ay2, H))

            if ax2 > ax1 and ay2 > ay1:
                crop = img[ay1:ay2, ax1:ax2]
                resized = cv2.resize(crop, (WINDOW_W, WINDOW_H))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                positives.append(gray)
                pos_heights.append(bh)

            gt_box = (x1, y1, x2, y2)

        
            added = 0
            tries = 0
            while added < JITTERS_PER_BOX and tries < 20:
                tries += 1

                scale = 1.0 + random.uniform(-SCALE_FRAC, SCALE_FRAC)
                new_w = bw_p * scale
                new_h = bh_p * scale

                dx = random.uniform(-SHIFT_FRAC, SHIFT_FRAC) * bw
                dy = random.uniform(-SHIFT_FRAC, SHIFT_FRAC) * bh

                nx1 = int(cx + dx - new_w / 2)
                ny1 = int(cy + dy - new_h / 2)
                nx2 = int(cx + dx + new_w / 2)
                ny2 = int(cy + dy + new_h / 2)

                nx1 = max(0, min(nx1, W - 1))
                ny1 = max(0, min(ny1, H - 1))
                nx2 = max(0, min(nx2, W))
                ny2 = max(0, min(ny2, H))

                if nx2 <= nx1 or ny2 <= ny1:
                    continue

                if compute_iou((nx1, ny1, nx2, ny2), gt_box) < MIN_IOU_JITTER:
                    continue

                crop = img[ny1:ny2, nx1:nx2]
                resized = cv2.resize(crop, (WINDOW_W, WINDOW_H))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                positives.append(gray)
                pos_heights.append(bh)
                added += 1


np.save(OUT_DIR / "pos_images.npy", np.array(positives, dtype=np.uint8))
# np.save(OUT_DIR / "pos_heights.npy", np.array(pos_heights))

print("Pozitive Velma:", len(positives))
print("Saved in:", OUT_DIR.resolve())
