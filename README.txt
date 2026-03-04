# Face Detection and Recognition – Scooby-Doo Characters

This project implements face detection and character recognition for Scooby-Doo characters (Daphne, Fred, Shaggy, Velma) using sliding window detection, HOG descriptors, and linear SVM classifiers.

Task 1 focuses on generic face detection, distinguishing between face regions and background. Task 2 extends this to individual character recognition, training a separate binary classifier for each character using a one-vs-rest approach. The two tasks do not share data or models, but use the same general detection paradigm.

---

## Approach

The pipeline is based on sliding windows scanned at multiple scales. For each window, a HOG descriptor is extracted and passed to a LinearSVC classifier. Results are refined using Non-Maximum Suppression with NMS_IOU = 0.3. A score floor of SCORE_FLOOR = -1.0 is applied to eliminate clearly negative windows early, reducing computational cost.

### Positive Examples

The window size is fixed at 64x80, chosen based on bounding box statistics to minimize distortion. Bounding boxes smaller than MIN_FACE_H = 15 are discarded. Adaptive padding is applied depending on face size: 0.18 for small faces, 0.14 for medium, and 0.10 for large. For each face, 4 additional jittered variants are generated with limited shifts (SHIFT_FRAC = 0.08) and scale variations (SCALE_FRAC = 0.10), keeping only variants with IoU >= 0.6.

### Negative Examples

40 negative examples are extracted per image at scales [0.75, 1.0, 1.5]. 30% are near-miss windows with partial overlap (0.1 <= IoU < 0.3), the rest are fully random (IoU < 0.1). Nearly uniform windows are discarded using STD_THRESH = 10.

### HOG Descriptor

All windows are described using HOG computed on grayscale images resized to 64x80:
- orientations = 9
- pixels_per_cell = (8, 8)
- cells_per_block = (2, 2)
- block_norm = L2-Hys
- transform_sqrt = True

### SVM Classifier

Classification uses LinearSVC with C = 0.5, max_iter = 15000, with StandardScaler normalization applied before training.

### Hard Negative Mining

After initial training, the detector is run on 30% of training images. Windows with score > 0 and IoU < 0.2 are collected as hard negatives, keeping the top 6 per image up to a maximum of 30000 total. Their HOG descriptors are added to the training set for retraining.

---

## Task 2 – Character Recognition

Task 2 uses the same architecture as Task 1. The difference is that positive examples come exclusively from the target character's bounding boxes, while the other characters' faces are used as negatives. Hard negative mining explicitly excludes windows overlapping with the target character's face. For hard negative mining, IMG_FRACTION is reduced from 0.30 to 0.15 to reduce processing time. For Velma, hard negative mining was not applied.

---

## Running Detection
```bash
python detectie.py
```

---

## Requirements
```bash
pip install -r requirements.txt
```