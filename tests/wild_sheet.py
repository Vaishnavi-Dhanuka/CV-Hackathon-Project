import os, sys, cv2
import numpy as np

in_dir, out_dir, dst = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst, exist_ok=True)
W, H = 320, 240

def cell(path, label):
    img = cv2.imread(path)
    canvas = np.zeros((H, W, 3), np.uint8)
    if img is None:
        return canvas
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    y, x = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(canvas, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas

names = sorted(f for f in os.listdir(in_dir)
               if f.lower().endswith(".png") and os.path.exists(os.path.join(out_dir, f)))
print(len(names), "pairs found")
pairs = [np.hstack([cell(os.path.join(in_dir, f), "IN " + f[:22]),
                    cell(os.path.join(out_dir, f), "OUT")]) for f in names]
for k in range(0, len(pairs), 8):
    chunk = pairs[k:k + 8]
    if len(chunk) % 2:
        chunk.append(np.zeros_like(chunk[0]))
    rows = [np.hstack(chunk[i:i + 2]) for i in range(0, len(chunk), 2)]
    path = os.path.join(dst, f"wild_comparison_{k // 8 + 1}.png")
    cv2.imwrite(path, np.vstack(rows))
    print("saved", path)