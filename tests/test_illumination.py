import sys
import numpy as np
from classical.preprocessing import prepare_image
from classical.illumination import illumination_analysis, visualize

img, _ = prepare_image(sys.argv[1])
out = illumination_analysis(img)

for k in ["illum_mean", "illum_cv", "dark_region_fraction", "uneven_fraction"]:
    print(f"{k:22s} {out[k]:.3f}")

# sanity: L * R should roughly rebuild the image
recon = out["illumination_map"][:, :, None] * out["reflectance"].astype(np.float32)
err = np.abs(recon - img.astype(np.float32)).mean()
print(f"mean |L*R - I| (0-255) : {err:.2f}")

visualize(img, out, "results/illumination_demo.png")