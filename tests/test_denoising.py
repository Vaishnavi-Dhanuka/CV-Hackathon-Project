import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from classical.preprocessing import load_image
from classical.denoising import denoise

clean = load_image(sys.argv[1])          # use a HIGH (well-lit) image
rng = np.random.default_rng(0)

print(f"{'noise':>6} {'method':10s} {'PSNR':>6} {'SSIM':>6} {'ms':>6}")
for noise in (10, 25, 40):
    noisy = np.clip(clean + rng.normal(0, noise, clean.shape), 0, 255).astype(np.uint8)
    outs = {"noisy": noisy}
    for name in ("bilateral", "guided"):
        t = time.perf_counter()
        outs[name] = denoise(noisy, name, sigma=noise)
        ms = (time.perf_counter() - t) * 1000
        print(f"{noise:>6} {name:10s} {psnr(clean, outs[name]):6.2f} "
              f"{ssim(clean, outs[name], channel_axis=2):6.3f} {ms:6.1f}")
    print(f"{noise:>6} {'(noisy)':10s} {psnr(clean, noisy):6.2f} {ssim(clean, noisy, channel_axis=2):6.3f}")
    if noise == 25:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        for a, (k, im) in zip(ax, {"clean": clean, **outs}.items()):
            a.imshow(im[100:300, 150:400]); a.set_title(f"{k} (crop)"); a.axis("off")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/denoise_demo.png", dpi=110)
        plt.show()