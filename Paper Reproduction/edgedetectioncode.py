"""
Reproduction of:
Ganesan, P. & Sajiv, G. "A Comprehensive Study of Edge Detection for
Image Processing Applications," ICIIECS 2017.

This script implements the six edge detectors described in the paper
(Sobel, Prewitt, Roberts, Laplacian of Gaussian, Canny, Wavelet/DWT-based)
using the exact kernels given in the paper, runs them on three test
images that mirror the paper's test set, reproduces the qualitative
comparison figures (paper's Fig. 6-8) and Table I, and adds the
quantitative evaluation (edge density, runtime, Pratt's Figure of Merit,
noise robustness) that the original paper does not provide.

"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import pywt
from skimage import data, img_as_float, img_as_ubyte
from skimage.util import random_noise
from skimage.color import rgb2gray

rng = np.random.default_rng(42)


# 1. TEST IMAGES  (stand-ins for the paper's cameraman.tif / onion.png /
#    retina fundus.jpg -- all built into scikit-image, no internet needed)

def load_test_images(onion_path=None, retina_path=None):
    
    
    
    
    imgs = {}
    imgs["cameraman"] = img_as_ubyte(data.camera())  # identical to the paper's cameraman.tif

    if onion_path and os.path.exists(onion_path):
        onion = cv2.imread(onion_path, cv2.IMREAD_GRAYSCALE)
        imgs["onion"] = img_as_ubyte(onion)
    else:
        imgs["cat_photo"] = img_as_ubyte(rgb2gray(data.chelsea()))  # natural-scene stand-in for onion.png (this is scikit-image's built-in cat photo, not an onion)

    if retina_path and os.path.exists(retina_path):
        retina = cv2.imread(retina_path, cv2.IMREAD_GRAYSCALE)
        imgs["retina"] = img_as_ubyte(retina)
    else:
        imgs["medical_like"] = img_as_ubyte(rgb2gray(data.immunohistochemistry()))  # stand-in, substitution disclosed

    return imgs



# 2. EDGE DETECTORS  -- kernels copied exactly from the paper (Figs 1-4)

def sobel(img):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    gx = ndimage.convolve(img.astype(float), Gx)
    gy = ndimage.convolve(img.astype(float), Gy)
    mag = np.sqrt(gx**2 + gy**2)
    return normalize(mag)


def prewitt(img):
    Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    Gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=float)
    gx = ndimage.convolve(img.astype(float), Gx)
    gy = ndimage.convolve(img.astype(float), Gy)
    mag = np.sqrt(gx**2 + gy**2)
    return normalize(mag)


def roberts(img):
    Gx = np.array([[1, 0], [0, -1]], dtype=float)
    Gy = np.array([[0, 1], [-1, 0]], dtype=float)
    gx = ndimage.convolve(img.astype(float), Gx)
    gy = ndimage.convolve(img.astype(float), Gy)
    mag = np.sqrt(gx**2 + gy**2)
    return normalize(mag)


def log_edge(img, sigma=1.4):
    """Laplacian of Gaussian: Gaussian-smooth first, then apply the
    paper's Laplacian kernel (Fig. 3)."""
    smoothed = ndimage.gaussian_filter(img.astype(float), sigma=sigma)
    L = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=float)
    lap = ndimage.convolve(smoothed, L)
    return normalize(np.abs(lap))


def canny_edge(img, low=50, high=150):
    return cv2.Canny(img, low, high).astype(float)


def dwt_edge(img, wavelet="haar", level=1):
    """2D discrete wavelet transform based edge detection (paper Fig. 5):
    decompose the image and recombine the horizontal, vertical and
    diagonal detail sub-bands, which capture the high-frequency (edge)
    information."""
    coeffs = pywt.wavedec2(img.astype(float), wavelet=wavelet, level=level)
    _, (cH, cV, cD) = coeffs[0], coeffs[1]
    detail = np.sqrt(cH**2 + cV**2 + cD**2)
    detail = cv2.resize(detail, (img.shape[1], img.shape[0]))
    return normalize(detail)


def normalize(x):
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return (x * 255).astype(np.uint8)


DETECTORS = {
    "Sobel": sobel,
    "Prewitt": prewitt,
    "Roberts": roberts,
    "LoG": log_edge,
    "Canny": canny_edge,
    "DWT": dwt_edge,
}



# 3. QUANTITATIVE METRICS (the paper only gives qualitative figures --
#    this is the added critical-analysis contribution)

def edge_density(edge_img, thresh=30):
    return float(np.mean(edge_img > thresh) * 100)  # % of pixels marked as edges


def pratt_fom(edge_img, ref_img, thresh=30, alpha=1.0/9):
    """Pratt's Figure of Merit against a reference edge map.
    FOM = (1/max(Ne,Nr)) * sum( 1 / (1 + alpha*d_i^2) )
    where d_i is the distance of each detected edge pixel to the
    nearest reference edge pixel. FOM is in [0,1], higher = better."""
    edges = edge_img > thresh
    ref = ref_img > thresh
    Ne = edges.sum()
    Nr = ref.sum()
    if Ne == 0 or Nr == 0:
        return 0.0
    dist_to_ref = ndimage.distance_transform_edt(~ref)
    d = dist_to_ref[edges]
    fom = np.sum(1.0 / (1.0 + alpha * d**2)) / max(Ne, Nr)
    return float(fom)


def run_all(img, ref_for_fom=None):
    results = {}
    timings = {}
    for name, fn in DETECTORS.items():
        t0 = time.perf_counter()
        out = fn(img)
        timings[name] = time.perf_counter() - t0
        results[name] = out
    return results, timings



# 4. MAIN EXPERIMENT

def make_figure(img, results, title, outpath):
    n = len(results) + 1
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i, (name, out) in enumerate(results.items(), start=1):
        axes[i].imshow(out, cmap="gray")
        axes[i].set_title(name)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    


    outpath = os.path.abspath(outpath)
    last_err = None
    for attempt in range(4):
        try:
            fig.savefig(outpath, dpi=150)
            last_err = None
            break
        except OSError as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    plt.close(fig)
    if last_err is not None:
        raise RuntimeError(
            f"Failed to save '{outpath}' after retries: {last_err}\n"
            "This is almost always Windows locking the file transiently "
            "(OneDrive/cloud-sync on this folder, or antivirus real-time "
            "scanning). Try: (1) moving the project out of any synced "
            "folder, e.g. to C:\\edge_project, (2) temporarily pausing "
            "OneDrive sync, or (3) adding an antivirus exclusion for this "
            "folder."
        ) from last_err


def main():
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    
    # here to use the exact/closest-possible images instead of the
    # scikit-image stand-ins:
    ONION_PATH = None    # e.g. r"C:\Program Files\MATLAB\R2023b\toolbox\images\imdata\onion.png"
    RETINA_PATH = None   # e.g. "drive_dataset/01_test.tif"

    images = load_test_images(onion_path=ONION_PATH, retina_path=RETINA_PATH)
    all_rows = []

    for img_name, img in images.items():
        results, timings = run_all(img)
        make_figure(img, results, f"Edge Detection Results: {img_name}",
                    f"{out_dir}/fig_{img_name}.png")

        # Use Canny as the pseudo-reference edge map for Pratt's FOM
        # (no hand-labeled ground truth is available for these images;
        # this is stated explicitly as a limitation in the report)
        ref = results["Canny"]

        # --- noisy version of the same image (robustness test) ---
        noisy = img_as_ubyte(random_noise(img, mode="gaussian", var=0.01, rng=rng))
        noisy_results, noisy_timings = run_all(noisy)
        make_figure(noisy, noisy_results, f"Edge Detection under Gaussian Noise: {img_name}",
                    f"{out_dir}/fig_{img_name}_noisy.png")

        for name in DETECTORS:
            all_rows.append({
                "Image": img_name,
                "Method": name,
                "EdgeDensity(%)": round(edge_density(results[name]), 2),
                "PrattFOM_vs_Canny": round(pratt_fom(results[name], ref), 3),
                "Runtime(ms)": round(timings[name] * 1000, 3),
                "EdgeDensity_Noisy(%)": round(edge_density(noisy_results[name]), 2),
                "Runtime_Noisy(ms)": round(noisy_timings[name] * 1000, 3),
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{out_dir}/metrics_table.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved figures and metrics_table.csv to '{out_dir}/'")


if __name__ == "__main__":
    main()