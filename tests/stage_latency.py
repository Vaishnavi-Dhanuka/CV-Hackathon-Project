import os, sys
import cv2
import numpy as np
from classical.classical_pipeline import enhance_classical

folder = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
names = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg")))[:n]
acc = {}
for f in names:
    img = cv2.cvtColor(cv2.imread(os.path.join(folder, f)), cv2.COLOR_BGR2RGB)
    _, inter = enhance_classical(img)
    for k, v in inter["meta"]["latency_ms"].items():
        acc.setdefault(k, []).append(float(v))
print(len(names), "images")
for k, v in acc.items():
    print(f"{k}: mean {np.mean(v):.1f} ms  p95 {np.percentile(v, 95):.1f} ms")