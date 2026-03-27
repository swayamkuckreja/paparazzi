import cv2
import numpy as np
import json
import os
import glob

# ═══════════════════════════════════════════════════════════════
#  TREE DETECTOR — drone-side logic
#  Loads learned_tree_params.json and detects trees using only
#  colour + colour variance + texture + shape. No YOLO.
#  This is what runs on the drone (translated to C later).
# ═══════════════════════════════════════════════════════════════

script_folder = os.path.dirname(os.path.abspath(__file__))

# ── Load learned parameters ───────────────────────────────────
params_path = os.path.join(script_folder, "learned_tree_params.json")
if not os.path.exists(params_path):
    print(f"ERROR: learned_tree_params.json not found in {script_folder}")
    print("Run the learner script first to generate it.")
    exit()

with open(params_path, "r") as f:
    P = json.load(f)

print("Loaded parameters:")
print(json.dumps(P, indent=2))
print()

# ── Unpack parameters ─────────────────────────────────────────
GREEN_LOWER   = np.array(P["hsv_lower"])
GREEN_UPPER   = np.array(P["hsv_upper"])
MIN_GREEN_PCT = P["min_green_pct"]

MIN_EDGE_DENSITY = P["min_edge_density"]
MIN_DIR_SPREAD   = P["dir_spread_min"]

MIN_HUE_STD = P["min_hue_std"]
MIN_VAL_STD = P["min_val_std"]

MAX_ASPECT_RATIO = P["max_aspect_ratio"]
MIN_BOX_WIDTH    = P["min_box_width"]
MIN_BOX_HEIGHT   = P["min_box_height"]

CANNY_LOW  = P["canny_low"]
CANNY_HIGH = P["canny_high"]

MIN_BLOB_AREA    = P["min_blob_area"]
MAX_BLOB_AREA    = P["max_blob_area"]
BLOB_KERNEL_SIZE = 15

# ═══════════════════════════════════════════════════════════════
#  DETECTION FUNCTIONS
#  These are what get translated to C for the drone
# ═══════════════════════════════════════════════════════════════

# ── Find candidate regions via green blob detection ───────────
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

# ── Gate 1: size ──────────────────────────────────────────────
def gate_size(x1, y1, x2, y2):
    return (x2 - x1) >= MIN_BOX_WIDTH and (y2 - y1) >= MIN_BOX_HEIGHT

# ── Gate 2: shape ─────────────────────────────────────────────
def gate_shape(x1, y1, x2, y2):
    w = x2 - x1; h = y2 - y1
    if h == 0: return False
    aspect = w / float(h)
    return (1.0 / MAX_ASPECT_RATIO) <= aspect <= MAX_ASPECT_RATIO

# ── Gate 3: colour ────────────────────────────────────────────
def gate_colour(hsv_img, x1, y1, x2, y2):
    roi = hsv_img[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0
    mask      = cv2.inRange(roi, GREEN_LOWER, GREEN_UPPER)
    green_pct = (np.count_nonzero(mask) / mask.size) * 100
    return green_pct >= MIN_GREEN_PCT, round(green_pct, 1)

# ── Gate 4: colour variance ───────────────────────────────────
def gate_variance(hsv_img, x1, y1, x2, y2):
    roi = hsv_img[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0, 0.0

    mask         = cv2.inRange(roi, GREEN_LOWER, GREEN_UPPER)
    green_pixels = roi[mask > 0]

    if len(green_pixels) < 30:
        return False, 0.0, 0.0

    hue_std = float(np.std(green_pixels[:, 0]))
    val_std = float(np.std(green_pixels[:, 2]))

    passes = hue_std >= MIN_HUE_STD or val_std >= MIN_VAL_STD
    return passes, round(hue_std, 1), round(val_std, 1)

# ── Gate 5: texture ───────────────────────────────────────────
def gate_texture(bgr_img, x1, y1, x2, y2):
    roi = bgr_img[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0, 0.0

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
        return False, round(edge_density, 1), 0.0

    hist, _ = np.histogram(angles[strong], bins=8, range=(-180, 180))
    hist    = hist / (hist.sum() + 1e-6)
    spread  = -np.sum(hist * np.log(hist + 1e-6))
    spread_pct = min((spread / 2.08) * 100, 100)

    passes = (edge_density >= MIN_EDGE_DENSITY and
              spread_pct   >= MIN_DIR_SPREAD)
    return passes, round(edge_density, 1), round(spread_pct, 1)

# ═══════════════════════════════════════════════════════════════
#  MAIN DETECTION FUNCTION
#  Input:  BGR frame
#  Output: list of confirmed tree bounding boxes + annotated frame
# ═══════════════════════════════════════════════════════════════

def detect_trees(frame):
    """
    Returns:
        trees     — list of dicts {x1,y1,x2,y2,cx,cy,green_pct,edge,spread}
        annotated — BGR image with boxes drawn
        status    — "TREES DETECTED" or "CLEAR"
    """
    img     = cv2.resize(frame, (640, 480))
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    overlay = img.copy()
    trees   = []

    blobs = find_green_blobs(hsv_img)

    for (x1, y1, x2, y2) in blobs:
        x1 = max(x1 - 5, 0);    y1 = max(y1 - 5, 0)
        x2 = min(x2 + 5, 640);  y2 = min(y2 + 5, 480)

        # Gate 1: size
        if not gate_size(x1, y1, x2, y2):
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), 1)
            continue

        # Gate 2: shape
        if not gate_shape(x1, y1, x2, y2):
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 50, 150), 1)
            continue

        # Gate 3: colour
        col_ok, green_pct = gate_colour(hsv_img, x1, y1, x2, y2)
        if not col_ok:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 60), 1)
            continue

        # Gate 4: colour variance
        var_ok, hue_std, val_std = gate_variance(hsv_img, x1, y1, x2, y2)
        if not var_ok:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 200), 1)
            cv2.putText(overlay, f"uniform Hs:{hue_std} Vs:{val_std}",
                        (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 200), 1)
            continue

        # Gate 5: texture
        tex_ok, edge_d, spread = gate_texture(img, x1, y1, x2, y2)
        if not tex_ok:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 100, 200), 1)
            cv2.putText(overlay, f"flat E:{edge_d}% D:{spread}%",
                        (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 200), 1)
            continue

        # ── Confirmed tree ─────────────────────────────────────
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w  = x2 - x1
        h  = y2 - y1

        trees.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy,
            "w": w, "h": h,
            "green_pct": green_pct,
            "edge_density": edge_d,
            "dir_spread": spread,
        })

        # Draw confirmed detection
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(overlay,
            f"TREE G:{green_pct}% E:{edge_d}% D:{spread}%",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 2)

        # Centroid crosshair — drone aims to avoid this point
        cv2.drawMarker(overlay, (cx, cy), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)

        # Danger zone fill
        zone     = overlay[y1:y2, x1:x2]
        red_fill = np.full_like(zone, (0, 0, 160))
        overlay[y1:y2, x1:x2] = cv2.addWeighted(zone, 0.7,
                                                  red_fill, 0.3, 0)

    # Status bar
    n      = len(trees)
    color  = (0, 0, 255) if n > 0 else (0, 200, 0)
    status = f"TREES: {n} — {'AVOID!' if n > 0 else 'CLEAR'}"
    cv2.rectangle(overlay, (0, 0), (640, 26), (0, 0, 0), -1)
    cv2.putText(overlay, status, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return trees, overlay, "TREES DETECTED" if n > 0 else "CLEAR"


# ═══════════════════════════════════════════════════════════════
#  RUN ON TEST IMAGES  (swap this section for live camera feed)
# ═══════════════════════════════════════════════════════════════

image_files = sorted([
    f for f in glob.glob(os.path.join(script_folder, '*.jpg'))
    if not f.endswith('_m.jpg')
    and not f.endswith('_tree_mask.jpg')
    and not f.endswith('_yolo.jpg')
    and not f.endswith('_detect.jpg')
])

if not image_files:
    print("No .jpg images found. Running on webcam instead...")

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        trees, annotated, status = detect_trees(frame)

        if trees:
            print(f"  {status}: {len(trees)} tree(s) detected")
            for i, t in enumerate(trees):
                print(f"    Tree {i+1}: centre=({t['cx']},{t['cy']}) "
                      f"size={t['w']}x{t['h']} "
                      f"E:{t['edge_density']}% D:{t['dir_spread']}%")

        cv2.imshow("Tree Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print(f"Found {len(image_files)} test images. Running detector...\n")

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        trees, annotated, status = detect_trees(frame)

        out_path = img_path.replace('.jpg', '_detect.jpg')
        cv2.imwrite(out_path, annotated)

        if trees:
            print(f"  ✓ {os.path.basename(img_path):40s} → {len(trees)} tree(s)")
            for i, t in enumerate(trees):
                print(f"      Tree {i+1}: centre=({t['cx']},{t['cy']}) "
                      f"size={t['w']}x{t['h']} "
                      f"G:{t['green_pct']}% "
                      f"E:{t['edge_density']}% "
                      f"D:{t['dir_spread']}%")
        else:
            print(f"    {os.path.basename(img_path):40s} → clear")

    print("\nDone. Check *_detect.jpg files for results.")
