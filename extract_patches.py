import os
from spectral import open_image
import numpy as np
import cv2
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ---------------- USER CONFIG ----------------
root_dataset_dir = "Tulsi"   # Just one "Tulsi"
class_folders = [r"Fresh\Fresh leaves", r"Disease\Disease leaves", r"Black\Black Leaves"]
label_map = {r"Fresh\Fresh leaves": 0, r"Disease\Disease leaves": 1, r"Black\Black Leaves": 2}
out_dir = "output_dataset_patches_3d"     # NEW output folder for 3D patches
patch_size = 5
stride = 5  # Non-overlapping
morph_kernel_primary = (7,7)
# ----------------------------------------------

os.makedirs(out_dir, exist_ok=True)

def load_hs(hdr_path):
    try:
        img = open_image(hdr_path)
        hs = img.load()
        hs = hs.astype(np.float32)
        bands_obj = getattr(img, "bands", None)
        return hs, bands_obj
    except Exception as e:
        print(f"[ERROR] Loading {hdr_path}: {e}")
        return None, None

def pick_best_band_index(hs):
    B = hs.shape[2]
    band_scores = [np.std(hs[:, :, b]) for b in range(B)]
    best_idx = int(np.argmax(band_scores))
    return best_idx

def normalize_to_uint8(band):
    eps = 1e-12
    bn = (band - band.min()) / (band.max() - band.min() + eps)
    return (bn * 255).astype(np.uint8)

def segment_using_band(band_uint8, kernel_size=(7,7)):
    _, binary = cv2.threshold(band_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

# Lists to store data
X_all = []
y_all = []
meta_all = []

# Walk dataset
print(f"[INFO] Starting 3D Patch Extraction...")
print(f"[INFO] Output Directory: {out_dir}")
print(f"[INFO] Patch Size: {patch_size}x{patch_size}xBands")

for cls in class_folders:
    folder = os.path.join(root_dataset_dir, cls)
    if not os.path.isdir(folder):
        continue

    # find .hdr files
    files = sorted(os.listdir(folder))
    hdr_paths = []
    for f in files:
        if f.lower().endswith(".hdr"):
            hdr_paths.append(os.path.join(folder,f))
        elif f.lower().endswith(".bil"):
            hdr = os.path.join(folder, os.path.splitext(f)[0] + ".hdr")
            if os.path.exists(hdr):
                hdr_paths.append(hdr)
    hdr_paths = sorted(list(set(hdr_paths)))

    label_int = label_map.get(cls, -1)

    for hdr_path in tqdm(hdr_paths, desc=f"Processing {cls}"):
        hs, bands_obj = load_hs(hdr_path)
        if hs is None: continue

        H, W, B = hs.shape
        
        # 1. Pick Best Band & Segment
        best_band_idx = pick_best_band_index(hs)
        band_uint8 = normalize_to_uint8(hs[:, :, best_band_idx])
        
        binary = segment_using_band(band_uint8, kernel_size=morph_kernel_primary)

        # Auto-Invert Mask Logic
        num_labels_tmp, labels_tmp, stats_tmp, _ = cv2.connectedComponentsWithStats(binary)
        max_area = 0
        for i in range(1, num_labels_tmp):
            a = int(stats_tmp[i, cv2.CC_STAT_AREA])
            if a > max_area: max_area = a
        if max_area > 0.5 * (H * W):
            binary = 255 - binary

        # 2. Patch Extraction (3D)
        
        for r in range(0, H - patch_size + 1, stride):
            for c in range(0, W - patch_size + 1, stride):
                
                # Check mask validity
                mask_patch = binary[r:r+patch_size, c:c+patch_size]
                
                if np.all(mask_patch == 255):
                    # Extract 3D Patch
                    hs_patch = hs[r:r+patch_size, c:c+patch_size, :]
                    
                    # Store
                    X_all.append(hs_patch)
                    y_all.append(label_int)
                    meta_all.append(f"{os.path.basename(hdr_path)}_r{r}_c{c}")

# Convert to Numpy Arrays
print("[INFO] Converting to Numpy Arrays...")
X_arr = np.array(X_all, dtype=np.float32)
y_arr = np.array(y_all, dtype=np.int32)
meta_arr = np.array(meta_all)

print(f"[SUCCESS] X Check shape: {X_arr.shape}")
print(f"[SUCCESS] y Check shape: {y_arr.shape}")

# Save
np.save(os.path.join(out_dir, "X.npy"), X_arr)
np.save(os.path.join(out_dir, "y.npy"), y_arr)
np.save(os.path.join(out_dir, "meta.npy"), meta_arr)

print(f"[DONE] Saved locally to {out_dir}")
