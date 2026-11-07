import os, sys
import cv2
import numpy as np

# usage: python -m tests.gt_brightness LABEL1 FOLDER1 LABEL2 FOLDER2 ...
args = sys.argv[1:]
for label, folder in zip(args[0::2], args[1::2]):
    means = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_COLOR)
            if img is not None:
                means.append(float(img.mean()))
    if not means:
        print(label, ": no images found in", folder)
        continue
    m = np.array(means)
    print(f"{label}: n={len(m)}  mean={m.mean():.1f}  median={np.median(m):.1f}  std={m.std():.1f}  (0-255 scale)")