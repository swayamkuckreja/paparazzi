import cv2
import numpy as np
import glob
import os
import json
import ssl
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════
#  TREE PARAMETER LEARNER
#  Gates: size → shape → colour → colour variance → texture → JSON
#  Key insight: mat and trees overlap in edge density BUT
#  mat has uniform colour (low variance), trees don't
# ═══════════════════════════════════════════════════════════════

ssl._create_default_https_context = ssl._create_unverified_context

script_folder   = os.path.dirname(os.path.abspath(__file__))
YOLO_CONFIDENCE = 0.15
CANNY_LOW       = 30
CANNY_HIGH      = 90

# Colour gate
MIN_GREEN_PCT = 15.0
GREEN_LOWER   = np.array([25, 45, 45])
GREEN_UPPER   = np.array([95, 255, 255])

# Size gate
MIN_BOX_WIDTH  = 40
MIN_BOX_HEIGHT = 40

# Shape gate — reject poles/banners
MAX_ASPECT_RATIO = 3.5

# Colour variance gate
# Mat: uniform flat green → low std dev in H and V channels
# Tree: varied leaf shades, shadows → high std dev
# Empirically: mat stdH < 6, stdV < 25 — trees always higher
MIN_HUE_STD   = 4.0    # std dev of hue among green pixels
MIN_VAL_STD   = 22.0   # std dev of value (brightness) among green pixels

# Texture gate — both must pass
MIN_EDGE_DENSITY = 4.0   # lowered — variance gate now handles mat
MIN_DIR_SPREAD   = 70.0

# Blob size — cap at 100k to reject giant mat blobs
MIN_BLOB_AREA = 300
MAX_BLOB_AREA = 200000

# Blob merging
BLOB_KERNEL_SIZE = 15

# ── Load model ────────────────────────────────────────────────
print("Loading YOLOv8...")
model_path = os.path.join(script_folder, "yolov8n.pt")
if not os.path.exists(model_path):
    print(f"ERROR: yolov8n.pt not found in {script_folder}")
    print(f"  curl -L --insecure https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt -o {model_path}")
    exit()
model = YOLO(model_path)
print("Model loaded.\n")

# ── Gate 1: size ──────────────────────────────────────────────
def is_large_enough(x1, y1, x2, y2):
    return (x2 - x1) >= MIN_BOX_WIDTH and (y2 - y1) >= MIN_BOX_HEIGHT

# ── Gate 2: shape ─────────────────────────────────────────────
def is_compact_shape(x1, y1, x2, y2):
    w = x2 - x1; h = y2 - y1
    if h == 0: return False
    aspect = w / float(h)
    return (1.0 / MAX_ASPECT_RATIO) <= aspect <= MAX_ASPECT_RATIO

# ── Gate 3: colour ────────────────────────────────────────────
def is_green_region(hsv_img, x1, y1, x2, y2):
    roi = hsv_img[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0
    green_mask = cv2.inRange(roi, GREEN_LOWER, GREEN_UPPER)
    green_pct  = (np.count_nonzero(green_mask) / green_mask.size) * 100
    return green_pct >= MIN_GREEN_PCT, round(green_pct, 1)

# ── Gate 4: colour variance ───────────────────────────────────
# This is the KEY gate that separates mat from trees
# Mat: uniform colour → low std dev
# Tree: varied leaves + shadows → high std dev
def has_colour_variance(hsv_img, x1, y1, x2, y2):
    roi = hsv_img[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0, 0.0

    green_mask   = cv2.inRange(roi, GREEN_LOWER, GREEN_UPPER)
    green_pixels = roi[green_mask > 0]

    if len(green_pixels) < 30:
        return False, 0.0, 0.0

    hue_std = float(np.std(green_pixels[:, 0]))
    val_std = float(np.std(green_pixels[:, 2]))

    passes = hue_std >= MIN_HUE_STD or val_std >= MIN_VAL_STD
    return passes, round(hue_std, 1), round(val_std, 1)

# ── Gate 5: texture ───────────────────────────────────────────
def extract_texture_stats(bgr_img, x1, y1, x2, y2):
    roi = bgr_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    edge_density = (np.count_nonzero(edges) / edges.size) * 100

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angles  = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    mag     = np.sqrt(sobel_x**2 + sobel_y**2)
    strong  = mag > np.percentile(mag, 70)

    if np.count_nonzero(strong) == 0:
        return None

    hist, _ = np.histogram(angles[strong], bins=8, range=(-180, 180))
    hist    = hist / (hist.sum() + 1e-6)
    spread  = -np.sum(hist * np.log(hist + 1e-6))
    spread_pct = min((spread / 2.08) * 100, 100)

    w = x2 - x1
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=15,
                             minLineLength=max(w // 3, 5),
                             maxLineGap=5)
    straight_count = len(lines) if lines is not None else 0

    return {
        "edge_density":   round(edge_density,  1),
        "dir_spread":     round(spread_pct,    1),
        "straight_lines": straight_count,
    }

def passes_texture_gate(tex):
    return (tex["edge_density"] >= MIN_EDGE_DENSITY and
            tex["dir_spread"]   >= MIN_DIR_SPREAD)

# ── Extract HSV stats ─────────────────────────────────────────
def extract_hsv_stats(hsv_img, x1, y1, x2, y2):
    roi = hsv_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    green_mask   = cv2.inRange(roi, GREEN_LOWER, GREEN_UPPER)
    green_pixels = roi[green_mask > 0]

    if len(green_pixels) < 30:
        green_pixels = roi.reshape(-1, 3)
    if len(green_pixels) == 0:
        return None

    return {
        "h_min": int(np.percentile(green_pixels[:, 0],  5)),
        "h_max": int(np.percentile(green_pixels[:, 0], 95)),
        "s_min": int(np.percentile(green_pixels[:, 1],  5)),
        "s_max": int(np.percentile(green_pixels[:, 1], 95)),
        "v_min": int(np.percentile(green_pixels[:, 2],  5)),
        "v_max": int(np.percentile(green_pixels[:, 2], 95)),
    }

# ── Green blob fallback ───────────────────────────────────────
def find_green_blobs(hsv_img):
    green_mask = cv2.inRange(hsv_img, GREEN_LOWER, GREEN_UPPER)
    k    = np.ones((BLOB_KERNEL_SIZE, BLOB_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask,       cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=2)
    mask = cv2.erode(mask,  np.ones((11, 11), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_BLOB_AREA < area < MAX_BLOB_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            blobs.append((x, y, x + w, y + h))
    return blobs

# ── IoU ───────────────────────────────────────────────────────
def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0

# ── Find all images ───────────────────────────────────────────
image_files = sorted([
    f for f in glob.glob(os.path.join(script_folder, '*.jpg'))
    if not f.endswith('_m.jpg')
    and not f.endswith('_tree_mask.jpg')
    and not f.endswith('_yolo.jpg')
])

if not image_files:
    print(f"No .jpg images found in {script_folder}")
    exit()

print(f"Found {len(image_files)} images.\n")

all_hsv     = []
all_texture = []

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue

    img       = cv2.resize(img, (640, 480))
    hsv_img   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    annotated = img.copy()

    candidates = []

    results = model(img, conf=YOLO_CONFIDENCE, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            candidates.append((x1, y1, x2, y2, "yolo"))

    for (x1, y1, x2, y2) in find_green_blobs(hsv_img):
        candidates.append((x1, y1, x2, y2, "blob"))

    unique = []
    for c in candidates:
        if not any(iou(c[:4], u[:4]) > 0.5 for u in unique):
            unique.append(c)

    detections_this_img = 0

    for (x1, y1, x2, y2, source) in unique:
        x1 = max(x1 - 5, 0);    y1 = max(y1 - 5, 0)
        x2 = min(x2 + 5, 640);  y2 = min(y2 + 5, 480)

        # Gate 1: size
        if not is_large_enough(x1, y1, x2, y2):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 30, 30), 1)
            cv2.putText(annotated, "tiny", (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (30, 30, 30), 1)
            continue

        # Gate 2: shape
        if not is_compact_shape(x1, y1, x2, y2):
            w = x2-x1; h = y2-y1
            ar = round(w/float(h) if h > 0 else 0, 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 50, 150), 1)
            cv2.putText(annotated, f"pole AR:{ar}", (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 50, 150), 1)
            continue

        # Gate 3: colour
        green_ok, green_pct = is_green_region(hsv_img, x1, y1, x2, y2)
        if not green_ok:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 60, 60), 1)
            cv2.putText(annotated, f"no-green {green_pct}%",
                        (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 60), 1)
            continue

        # Gate 4: colour variance — KEY mat rejection gate
        var_ok, hue_std, val_std = has_colour_variance(hsv_img, x1, y1, x2, y2)
        if not var_ok:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 200), 1)
            cv2.putText(annotated,
                        f"uniform H_std:{hue_std} V_std:{val_std}",
                        (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 200), 1)
            continue

        # Gate 5: texture
        tex_stats = extract_texture_stats(img, x1, y1, x2, y2)
        if tex_stats is None:
            continue

        if not passes_texture_gate(tex_stats):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 100, 200), 1)
            cv2.putText(annotated,
                        f"flat E:{tex_stats['edge_density']}% D:{tex_stats['dir_spread']}%",
                        (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 200), 1)
            continue

        # All gates passed
        hsv_stats = extract_hsv_stats(hsv_img, x1, y1, x2, y2)
        if not hsv_stats:
            continue

        all_hsv.append(hsv_stats)
        all_texture.append(tex_stats)
        detections_this_img += 1

        color = (0, 0, 255) if source == "yolo" else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated,
            f"[{source}] G:{green_pct}% "
            f"E:{tex_stats['edge_density']}% "
            f"D:{tex_stats['dir_spread']}% "
            f"Hs:{hue_std} Vs:{val_std}",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

    status = f"Trees: {detections_this_img}  |  Total so far: {len(all_hsv)}"
    cv2.rectangle(annotated, (0, 0), (640, 22), (0, 0, 0), -1)
    cv2.putText(annotated, status, (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if detections_this_img > 0 else (100, 100, 100), 1)

    annotated_path = img_path.replace('.jpg', '_yolo.jpg')
    cv2.imwrite(annotated_path, annotated)

    if detections_this_img > 0:
        print(f"  ✓ {os.path.basename(img_path):40s} → {detections_this_img} trees")
    else:
        print(f"    {os.path.basename(img_path):40s} → no trees")

# ── Aggregate into final parameters ───────────────────────────
print(f"\nTotal confirmed tree regions: {len(all_hsv)}")

if not all_hsv:
    print("\nNo trees passed all gates.")
    print(f"  Lower MIN_EDGE_DENSITY (currently {MIN_EDGE_DENSITY})")
    print(f"  Lower MIN_HUE_STD      (currently {MIN_HUE_STD})")
    print(f"  Lower MIN_VAL_STD      (currently {MIN_VAL_STD})")
    exit()

print("Computing parameter ranges...\n")
MARGIN = 0.10

def pct(vals, p):
    return float(np.percentile(vals, p))

params = {
    "hsv_lower": [
        max(int(pct([s["h_min"] for s in all_hsv], 10) * (1 - MARGIN)), 0),
        max(int(pct([s["s_min"] for s in all_hsv], 10) * (1 - MARGIN)), 0),
        max(int(pct([s["v_min"] for s in all_hsv], 10) * (1 - MARGIN)), 0),
    ],
    "hsv_upper": [
        min(int(pct([s["h_max"] for s in all_hsv], 90) * (1 + MARGIN)), 179),
        min(int(pct([s["s_max"] for s in all_hsv], 90) * (1 + MARGIN)), 255),
        min(int(pct([s["v_max"] for s in all_hsv], 90) * (1 + MARGIN)), 255),
    ],
    "edge_density_min":   round(pct([t["edge_density"]   for t in all_texture], 15) * (1 - MARGIN), 1),
    "dir_spread_min":     round(pct([t["dir_spread"]     for t in all_texture], 15) * (1 - MARGIN), 1),
    "straight_lines_max": int(pct([t["straight_lines"]   for t in all_texture], 85)) + 2,
    "canny_low":          CANNY_LOW,
    "canny_high":         CANNY_HIGH,
    "min_blob_area":      MIN_BLOB_AREA,
    "max_blob_area":      MAX_BLOB_AREA,
    "min_green_pct":      MIN_GREEN_PCT,
    "min_edge_density":   MIN_EDGE_DENSITY,
    "min_dir_spread":     MIN_DIR_SPREAD,
    "min_hue_std":        MIN_HUE_STD,
    "min_val_std":        MIN_VAL_STD,
    "max_aspect_ratio":   MAX_ASPECT_RATIO,
    "min_box_width":      MIN_BOX_WIDTH,
    "min_box_height":     MIN_BOX_HEIGHT,
    "sample_count":       len(all_hsv),
    "images_processed":   len(image_files),
}

out_path = os.path.join(script_folder, "learned_tree_params.json")
with open(out_path, "w") as f:
    json.dump(params, f, indent=2)

print("=== learned_tree_params.json ===")
print(json.dumps(params, indent=2))
print(f"\nSaved to: {out_path}")
print("Ready for C implementation.")
