import os, sys, cv2

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
n = 0
for f in sorted(os.listdir(src)):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        print("skipped", f)
        continue
    img = cv2.imread(os.path.join(src, f))
    if img is None:
        print("unreadable", f)
        continue
    h, w = img.shape[:2]
    s = 1024 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(dst, os.path.splitext(f)[0] + ".png"), img)
    n += 1
print("saved", n, "images to", dst)