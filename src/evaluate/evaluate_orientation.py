"""
evaluate_orientation.py
-----------------------
Оцінка кутової точності векторів орієнтації, передбачених YOLOv11-pose.

Читає параметри з params.yaml (розділ evaluate + prepare.dataset).

Label format (YOLO-pose, Sledevič dataset):
  class cx cy w h px1 py1 px2 py2   (all normalised 0-1, NO visibility flag)
  kp0 = head = (px1, py1)
  kp1 = tail  = (px2, py2)

Orientation vector: head – tail   (kp0 – kp1)

Запуск:
    dvc repro evaluate                              # через DVC пайплайн
    python src/evaluate/evaluate_orientation.py    # з params.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import yaml

# Корінь проєкту: evaluate/ → src/ → project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── helpers ──────────────────────────────────────────────────────────────────

def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Compute IoU of two boxes in [x1, y1, x2, y2] format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def angular_error_deg(v_pred: np.ndarray, v_gt: np.ndarray) -> float:
    """
    Angular error between two 2-D vectors in degrees.
    Uses min(angle, 180-angle) for head/tail symmetry.
    """
    norm_p = np.linalg.norm(v_pred)
    norm_g = np.linalg.norm(v_gt)
    if norm_p < 1e-6 or norm_g < 1e-6:
        return float("nan")
    cos_a = np.clip(np.dot(v_pred, v_gt) / (norm_p * norm_g), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_a))
    return min(angle, 180.0 - angle)


def load_gt_annotations(label_path: Path, img_w: int, img_h: int):
    """
    Parse a YOLO-pose label file.

    Returns list of dicts:
      { 'box_abs': [x1,y1,x2,y2],          # pixel coords
        'kp0': (x, y),                      # head – pixel coords
        'kp1': (x, y) }                     # tail – pixel coords
    """
    anns = []
    if not label_path.exists():
        return anns
    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            # class cx cy w h px1 py1 px2 py2  – all normalised
            cx, cy, w, h = (float(p) for p in parts[1:5])
            px1, py1     = float(parts[5]), float(parts[6])
            px2, py2     = float(parts[7]), float(parts[8])

            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h

            anns.append({
                "box_abs": [x1, y1, x2, y2],
                "kp0":     (px1 * img_w, py1 * img_h),   # head
                "kp1":     (px2 * img_w, py2 * img_h),   # tail
            })
    return anns


def greedy_match(pred_boxes: np.ndarray, gt_boxes: list, iou_thresh: float):
    """
    Greedy IoU matching: each GT box matched to at most one pred box.

    Returns list of (pred_idx, gt_idx) pairs with IoU >= iou_thresh.
    """
    matched = []
    used_pred = set()
    used_gt   = set()

    # Build IoU matrix
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matched

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou_matrix[pi, gi] = iou(pb, np.array(gb["box_abs"]))

    # Sort by descending IoU
    idxs = np.dstack(np.unravel_index(
        np.argsort(iou_matrix, axis=None)[::-1],
        iou_matrix.shape
    ))[0]

    for pi, gi in idxs:
        if iou_matrix[pi, gi] < iou_thresh:
            break
        if pi in used_pred or gi in used_gt:
            continue
        matched.append((int(pi), int(gi)))
        used_pred.add(pi)
        used_gt.add(gi)

    return matched


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    weights: str,
    val_images: str,
    val_labels: str,
    kp_conf_thresh: float,
    iou_thresh: float,
    output_dir: str,
) -> dict:

    from ultralytics import YOLO

    model = YOLO(weights)

    images_dir = Path(val_images)
    labels_dir = Path(val_labels)
    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in image_exts
    )

    if not image_paths:
        sys.exit(f"[ERROR] No images found in {images_dir}")

    print(f"[INFO] Found {len(image_paths)} images.")
    print(f"[INFO] Model weights : {weights}")
    print(f"[INFO] kp-conf thresh: {kp_conf_thresh}  iou thresh: {iou_thresh}")
    print()

    all_errors          = []
    total_skipped       = 0
    per_image_results   = []

    for img_path in image_paths:
        # ── inference ──────────────────────────────────────────────────────
        results = model(str(img_path), verbose=False)
        result  = results[0]

        img_h, img_w = result.orig_shape  # (H, W)

        # predicted boxes: xyxy in pixel coords
        pred_boxes_px = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None and len(result.boxes) > 0
            else np.zeros((0, 4))
        )

        # predicted keypoints: shape (N, K, 2) or (N, K, 3) — x, y[, conf]
        if (result.keypoints is not None
                and result.keypoints.data is not None
                and len(result.keypoints.data) > 0):
            kpts = result.keypoints.data.cpu().numpy()   # (N, 2, 2) or (N, 2, 3)
        else:
            kpts = np.zeros((len(pred_boxes_px), 2, 3))

        has_conf = kpts.shape[-1] == 3

        # ── ground truth ───────────────────────────────────────────────────
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_anns    = load_gt_annotations(label_path, img_w, img_h)

        if not gt_anns:
            per_image_results.append({
                "image":    img_path.name,
                "matched":  0,
                "skipped":  0,
                "errors":   [],
            })
            continue

        # ── matching ───────────────────────────────────────────────────────
        matched_pairs = greedy_match(pred_boxes_px, gt_anns, iou_thresh)

        img_errors  = []
        img_skipped = 0

        for pi, gi in matched_pairs:
            if has_conf:
                kp0_x, kp0_y, kp0_c = kpts[pi, 0]   # head – predicted
                kp1_x, kp1_y, kp1_c = kpts[pi, 1]   # tail – predicted
            else:
                kp0_x, kp0_y = kpts[pi, 0, :2]
                kp1_x, kp1_y = kpts[pi, 1, :2]
                kp0_c = kp1_c = 1.0   # no confidence stored; treat as visible

            if kp0_c < kp_conf_thresh or kp1_c < kp_conf_thresh:
                img_skipped += 1
                continue

            gt = gt_anns[gi]
            head_gt = np.array(gt["kp0"])
            tail_gt = np.array(gt["kp1"])

            v_pred = np.array([kp0_x - kp1_x, kp0_y - kp1_y])   # head – tail
            v_gt   = head_gt - tail_gt

            err = angular_error_deg(v_pred, v_gt)
            if not np.isnan(err):
                img_errors.append(float(err))

        all_errors.extend(img_errors)
        total_skipped += img_skipped

        per_image_results.append({
            "image":    img_path.name,
            "matched":  len(matched_pairs),
            "skipped":  img_skipped,
            "errors":   img_errors,
        })

    # ── aggregate stats ────────────────────────────────────────────────────
    errors_arr = np.array(all_errors)

    def pct_within(threshold: float) -> float:
        if len(errors_arr) == 0:
            return 0.0
        return float(np.mean(errors_arr <= threshold) * 100)

    stats = {
        "mean_angular_error_deg":   float(np.mean(errors_arr))   if len(errors_arr) else None,
        "median_angular_error_deg": float(np.median(errors_arr)) if len(errors_arr) else None,
        "std_angular_error_deg":    float(np.std(errors_arr))    if len(errors_arr) else None,
        "pct_within_15deg":  pct_within(15),
        "pct_within_30deg":  pct_within(30),
        "pct_within_45deg":  pct_within(45),
        "total_matched_bees":       len(all_errors) + total_skipped,
        "total_skipped_low_conf":   total_skipped,
        "per_image":                per_image_results,
    }

    # ── print ──────────────────────────────────────────────────────────────
    print("=" * 50)
    print("  ORIENTATION EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Total matched bees  : {stats['total_matched_bees']}")
    print(f"  Skipped (low conf)  : {stats['total_skipped_low_conf']}")
    print(f"  Evaluated bees      : {len(all_errors)}")
    print()
    if len(errors_arr):
        print(f"  Mean error          : {stats['mean_angular_error_deg']:.2f}°")
        print(f"  Median error        : {stats['median_angular_error_deg']:.2f}°")
        print(f"  Std dev             : {stats['std_angular_error_deg']:.2f}°")
        print()
        print(f"  Within 15°          : {stats['pct_within_15deg']:.1f}%")
        print(f"  Within 30°          : {stats['pct_within_30deg']:.1f}%")
        print(f"  Within 45°          : {stats['pct_within_45deg']:.1f}%")
    else:
        print("  No angular errors computed (check matching / confidence settings).")
    print("=" * 50)

    # ── save JSON ──────────────────────────────────────────────────────────
    json_path = out_dir / "orientation_results.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[INFO] Results saved to {json_path}")

    # ── histogram ──────────────────────────────────────────────────────────
    hist_path = out_dir / "orientation_errors_hist.png"
    fig, ax = plt.subplots(figsize=(8, 5))

    if len(errors_arr):
        ax.hist(errors_arr, bins=18, range=(0, 90),
                color="#2196F3", edgecolor="white", alpha=0.85)
        for thresh, color, label in [
            (15, "#4CAF50", "15°"),
            (30, "#FF9800", "30°"),
            (45, "#F44336", "45°"),
        ]:
            ax.axvline(thresh, color=color, linestyle="--", linewidth=1.5,
                       label=f"≤{label}: {pct_within(thresh):.1f}%")
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)

    ax.set_xlabel("Angular Error (degrees)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Orientation Angular Error Distribution", fontsize=14)
    ax.set_xlim(0, 90)
    plt.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Histogram saved to {hist_path}")

    return stats


# ── MAIN ──────────────────────────────────────────────────────────────────────

def load_params() -> tuple[dict, str]:
    """Читає params.yaml, повертає (evaluate_params, dataset)."""
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path) as f:
        all_params = yaml.safe_load(f)
    evaluate_params = all_params.get("evaluate", {})
    dataset = all_params.get("prepare", {}).get("dataset", "pose")
    return evaluate_params, dataset


if __name__ == "__main__":
    ev, dataset = load_params()
    split_dir = PROJECT_ROOT / "data" / "split" / dataset

    weights = str(PROJECT_ROOT / ev.get("weights", "models/best.pt"))
    val_images = str(split_dir / "val" / "images")
    val_labels = str(split_dir / "val" / "labels")
    kp_conf = ev.get("kp_conf", 0.5)
    iou_thresh = ev.get("iou_thresh", 0.5)
    output_dir = str(PROJECT_ROOT / ev.get("output_dir", "evaluation") / dataset)

    evaluate(
        weights=weights,
        val_images=val_images,
        val_labels=val_labels,
        kp_conf_thresh=kp_conf,
        iou_thresh=iou_thresh,
        output_dir=output_dir,
    )
