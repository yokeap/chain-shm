"""
vertical_link_seg.py
====================
Segment vertical chain link using:
  Step 1: Wiener filter + Otsu threshold → full chain mask
  Step 2: SAM v1 + point prompt → horizontal wire mask
  Step 3: full_mask - wire_mask → vertical link only

Usage
-----
    python3 vertical_link_seg.py --image chain.png
    python3 vertical_link_seg.py --image chain.png --model sam_b.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import wiener as scipy_wiener
from scipy.ndimage import label as scipy_label

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Wiener + Otsu
# ---------------------------------------------------------------------------

def get_full_mask(gray: np.ndarray) -> np.ndarray:
    """Wiener filter + Otsu → fill small holes → binary mask."""
    filtered = scipy_wiener(gray.astype(np.float64), mysize=(5, 5))
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(
        filtered, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_small, iterations=1)

    # --- [เพิ่มระบบถมหลุมสนิม/แสงสะท้อน] ---
    # หา Contours ทั้งหมด (ทั้งขอบนอกและรูใน)
    cnts, hier = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hier = hier[0]
        for i, c in enumerate(cnts):
            # ถ้าเป็น "รู" (มี Parent) และขนาดรูเล็กกว่า 10,000 พิกเซล ให้ถมขาวทับ
            if hier[i][3] != -1 and cv2.contourArea(c) < 10000:
                cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)
    # --------------------------------------

    # Keep largest blob only (ตัวเนื้อโซ่ทั้งหมด)
    lbl, n = scipy_label(clean > 0)
    if n == 0:
        return clean
    sizes   = [(lbl == i).sum() for i in range(1, n + 1)]
    largest = np.argmax(sizes) + 1
    return ((lbl == largest) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 2: SAM v1 → wire mask
# ---------------------------------------------------------------------------

def get_wire_mask_sam(
    image     : np.ndarray,
    full_mask : np.ndarray, # <--- รับ full_mask เข้ามาใช้คำนวณจุด
    model_path: str = "sam_b.pt",
) -> np.ndarray:
    from ultralytics import SAM
    h, w = image.shape[:2]
    logger.info(f"Loading SAM: {model_path}")
    model = SAM(model_path)

    # 1. หาตำแหน่งรู (Holes) ใน full_mask เพื่อใช้เป็นจุดอ้างอิงเรขาคณิต
    cnts, hier = cv2.findContours(full_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if cnts and hier is not None:
        hier = hier[0]
        for i, c in enumerate(cnts):
            if hier[i][3] != -1 and cv2.contourArea(c) > 1000: # กรองรูจิ๋วๆ ทิ้ง
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    holes.append((cx, cy))

    holes.sort(key=lambda p: p[0]) # เรียงลำดับรูจากซ้ายไปขวา
    
    # 2. คำนวณหาจุดที่เป็น "โซ่แนวนอน (Horizontal Link)" แบบ Dynamic
    prompt_points = []
    if len(holes) >= 2:
        # จุดกึ่งกลางระหว่าง 2 รู คือ โซ่แนวนอน 100%
        for i in range(len(holes) - 1):
            mx = (holes[i][0] + holes[i+1][0]) // 2
            my = (holes[i][1] + holes[i+1][1]) // 2
            prompt_points.append([mx, my])
        
        # เผื่อโซ่แนวนอนที่อยู่ขอบซ้ายสุดและขวาสุด (Extrapolate)
        dist = np.mean([holes[i+1][0] - holes[i][0] for i in range(len(holes)-1)])
        prompt_points.append([int(holes[0][0] - dist), holes[0][1]])
        prompt_points.append([int(holes[-1][0] + dist), holes[-1][1]])
    elif len(holes) == 1:
        # กรณีซูมใกล้ เห็นรูเดียว โซ่แนวนอนจะขนาบซ้ายขวา
        cx, cy = holes[0]
        prompt_points.extend([[max(0, cx - 300), cy], [min(w - 1, cx + 300), cy]])
    else:
        prompt_points = [[w//2, h//2]] # Fallback

    # 3. กรองให้เหลือเฉพาะจุดที่ตกอยู่บนเนื้อเหล็กจริงๆ (full_mask == 255)
    valid_pts = []
    for pt in prompt_points:
        px, py = max(0, min(w-1, pt[0])), max(0, min(h-1, pt[1]))
        if full_mask[py, px] > 0:
            valid_pts.append([px, py])
    
    if not valid_pts:
        valid_pts = [[w//2, h//2]]

    logger.info(f"Dynamic SAM Prompts (Horizontal Links): {valid_pts}")

    # 4. รัน SAM ทีละจุด (ห้ามส่งรวบยอด ไม่งั้น SAM จะจับโซ่ทั้งเส้น) แล้วรวม Mask เข้าด้วยกัน
    combined_wire_mask = np.zeros((h, w), dtype=np.uint8)
    for pt in valid_pts:
        results = model(image, points=[pt], labels=[1], verbose=False)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else np.ones(len(masks))
            best  = int(np.argmax(confs))
            raw   = masks[best]
            mask  = cv2.resize((raw * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            combined_wire_mask = cv2.bitwise_or(combined_wire_mask, mask)

    return combined_wire_mask


# ---------------------------------------------------------------------------
# Step 3: Subtract wire → vertical link (topology-aware)
# ---------------------------------------------------------------------------

def get_vertical_link_mask(
    full_mask : np.ndarray,
    wire_mask : np.ndarray,
    dilate_px : int = 10,
) -> np.ndarray:
    """full_mask - wire_mask -> keep all valid vertical components"""
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    wire_d = cv2.dilate(wire_mask, k, iterations=2)

    vert_raw = cv2.bitwise_and(full_mask, cv2.bitwise_not(wire_d))

    lbl, n = scipy_label(vert_raw > 0)
    if n == 0:
        return vert_raw
        
    result = np.zeros_like(vert_raw)
    
    # --- [แก้ข้อจำกัด] ---
    # เก็บ "ทุกชิ้น" ที่ใหญ่กว่า 2,000 พิกเซล (เพื่อเอาเศษฝุ่นทิ้ง แต่เก็บ Vertical Chain ไว้ทั้งหมด)
    for i in range(1, n + 1):  
        if (lbl == i).sum() > 2000: 
            result[lbl == i] = 255
    # ----------------------

    return result


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

def draw_result(
    image    : np.ndarray,
    vert_mask: np.ndarray,
    wire_mask: np.ndarray,
) -> np.ndarray:
    vis = image.copy()

    # Vertical link overlay (green)
    ov = np.zeros_like(image)
    ov[vert_mask > 0] = (0, 200, 0)
    vis = cv2.addWeighted(vis, 0.55, ov, 0.45, 0)

    # Contour
    cnts, _ = cv2.findContours(
        vert_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 3, cv2.LINE_AA)

    # Banner
    cv2.rectangle(vis, (0, 0), (500, 44), (30, 30, 30), -1)
    cv2.putText(vis,
        f"Vertical link: {np.count_nonzero(vert_mask)//1000}k px",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return vis


# ---------------------------------------------------------------------------
# Segment export  (เพิ่ม: save แต่ละ vertical link blob แยกไฟล์ + JSON metadata)
# ---------------------------------------------------------------------------

def save_segments(
    image    : np.ndarray,
    vert_mask: np.ndarray,
    wire_mask: np.ndarray,
    full_mask: np.ndarray,
    save_dir : Path,
    stem     : str,
) -> list:
    """
    สำหรับแต่ละ vertical link blob:
      - crop image / vert_mask / wire_mask ตาม bounding box (+ padding)
      - save PNG ทั้ง 3 layer
      - บันทึก metadata JSON (bbox, wire_contact_y_range, pixel counts)

    Returns list of dicts (metadata per segment) → ใช้ใน reconstruction.py
    """
    import json

    seg_dir = save_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    lbl, n = scipy_label(vert_mask > 0)
    PAD = 20          # padding รอบ bounding box (px)
    h, w = image.shape[:2]

    segments_meta = []

    for i in range(1, n + 1):
        comp = (lbl == i).astype(np.uint8) * 255
        if comp.sum() // 255 < 2000:          # กรองเศษเล็ก
            continue

        # --- bounding box ---
        rows = np.where(comp.sum(axis=1) > 0)[0]
        cols = np.where(comp.sum(axis=0) > 0)[0]
        y1 = max(0,     int(rows[0])  - PAD)
        y2 = min(h - 1, int(rows[-1]) + PAD)
        x1 = max(0,     int(cols[0])  - PAD)
        x2 = min(w - 1, int(cols[-1]) + PAD)

        # --- wire contact y-range (ภายใน bbox นี้) ---
        wire_roi = wire_mask[y1:y2, x1:x2]
        wire_rows_in_roi = np.where(wire_roi.sum(axis=1) > 0)[0]
        if len(wire_rows_in_roi):
            wire_y_top = int(wire_rows_in_roi[0])  + y1
            wire_y_bot = int(wire_rows_in_roi[-1]) + y1
        else:
            wire_y_top = wire_y_bot = (y1 + y2) // 2

        # --- ขอบ outer (top/bottom) ของ vert_mask ที่ x กึ่งกลาง ---
        cx_col = (x1 + x2) // 2
        vert_col = vert_mask[y1:y2, cx_col]
        vert_rows_in_col = np.where(vert_col > 0)[0]
        outer_top_y = int(vert_rows_in_col[0])  + y1 if len(vert_rows_in_col) else y1
        outer_bot_y = int(vert_rows_in_col[-1]) + y1 if len(vert_rows_in_col) else y2

        # --- crop แต่ละ layer ---
        crop_img  = image    [y1:y2, x1:x2]
        crop_vert = comp     [y1:y2, x1:x2]
        crop_wire = wire_mask[y1:y2, x1:x2]
        crop_full = full_mask[y1:y2, x1:x2]

        seg_id = f"{stem}_seg{i:02d}"
        cv2.imwrite(str(seg_dir / f"{seg_id}_image.png"),     crop_img)
        cv2.imwrite(str(seg_dir / f"{seg_id}_vert_mask.png"), crop_vert)
        cv2.imwrite(str(seg_dir / f"{seg_id}_wire_mask.png"), crop_wire)
        cv2.imwrite(str(seg_dir / f"{seg_id}_full_mask.png"), crop_full)

        # --- debug overlay (vert=green, wire=red tint) ---
        dbg = crop_img.copy()
        ov_v = np.zeros_like(dbg); ov_v[crop_vert > 0] = (0, 220, 0)
        ov_w = np.zeros_like(dbg); ov_w[crop_wire > 0] = (0, 0, 200)
        dbg  = cv2.addWeighted(dbg, 0.55, ov_v, 0.30, 0)
        dbg  = cv2.addWeighted(dbg, 0.85, ov_w, 0.15, 0)

        # วาด outer boundary ของ vert_mask เส้นเขียว
        cnts_v, _ = cv2.findContours(crop_vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, cnts_v, -1, (0, 255, 0), 2, cv2.LINE_AA)
        # วาด inner boundary ของ wire_mask เส้นแดง
        cnts_w, _ = cv2.findContours(crop_wire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, cnts_w, -1, (0, 60, 255), 2, cv2.LINE_AA)

        cv2.imwrite(str(seg_dir / f"{seg_id}_overlay.png"), dbg)

        meta = {
            "seg_id"        : seg_id,
            "link_index"    : i - 1,
            # bbox ใน image พิกัดเต็ม
            "bbox"          : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            # wire ตัดผ่านที่ y เท่าไหร่ (image coords)
            "wire_y_top"    : wire_y_top,
            "wire_y_bot"    : wire_y_bot,
            # outer edge ของ vertical link ที่ x กึ่งกลาง
            "outer_top_y"   : outer_top_y,
            "outer_bot_y"   : outer_bot_y,
            "center_x"      : cx_col,
            # pixel counts
            "vert_px"       : int(comp.sum() // 255),
            "wire_px"       : int(wire_mask[y1:y2, x1:x2].sum() // 255),
            # paths (relative to save_dir)
            "files": {
                "image"    : f"segments/{seg_id}_image.png",
                "vert_mask": f"segments/{seg_id}_vert_mask.png",
                "wire_mask": f"segments/{seg_id}_wire_mask.png",
                "full_mask": f"segments/{seg_id}_full_mask.png",
                "overlay"  : f"segments/{seg_id}_overlay.png",
            }
        }
        segments_meta.append(meta)
        logger.info(
            f"  Seg {i}: bbox=({x1},{y1})-({x2},{y2})  "
            f"wire_y=[{wire_y_top},{wire_y_bot}]  "
            f"outer_top_y={outer_top_y}  vert_px={meta['vert_px']:,}"
        )

    # --- save JSON ---
    json_path = save_dir / f"segments_{stem}.json"
    with open(json_path, "w") as f:
        json.dump({"image_stem": stem, "segments": segments_meta}, f, indent=2)
    logger.info(f"Saved segment metadata → {json_path}")
    logger.info(f"Saved {len(segments_meta)} segment(s) → {seg_dir}/")

    return segments_meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    required=True)
    parser.add_argument("--model",    default="sam_b.pt")
    parser.add_argument("--save-dir", default="debug_seg")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img  = cv2.imread(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1
    logger.info("Step 1: Wiener + Otsu")
    full_mask = get_full_mask(gray)

    # Step 2
    logger.info("Step 2: SAM → wire mask")
    wire_mask = get_wire_mask_sam(img, full_mask, args.model)

    # Step 3
    logger.info("Step 3: Subtract wire → vertical link")
    vert_mask = get_vertical_link_mask(full_mask, wire_mask)

    # Save masks + overlay
    stem = Path(args.image).stem
    vis  = draw_result(img, vert_mask, wire_mask)
    cv2.imwrite(str(save_dir / f"vert_{stem}.jpg"),      vis)
    cv2.imwrite(str(save_dir / f"mask_vert_{stem}.jpg"), vert_mask)
    cv2.imwrite(str(save_dir / f"mask_wire_{stem}.jpg"), wire_mask)
    cv2.imwrite(str(save_dir / f"mask_full_{stem}.jpg"), full_mask)

    # Save per-link segments + JSON metadata  ← ใหม่
    logger.info("Saving per-link segments …")
    segs = save_segments(img, vert_mask, wire_mask, full_mask, save_dir, stem)

    print(
        f"\n{'─'*55}\n"
        f"  Full mask  : {np.count_nonzero(full_mask):,} px\n"
        f"  Wire mask  : {np.count_nonzero(wire_mask):,} px\n"
        f"  Vert mask  : {np.count_nonzero(vert_mask):,} px\n"
        f"  Segments   : {len(segs)} link(s) → {save_dir}/segments/\n"
        f"  JSON meta  : {save_dir}/segments_{stem}.json\n"
        f"{'─'*55}\n"
        f"  Next step  :\n"
        f"    python3 reconstruction.py \\\n"
        f"      --seg-dir {save_dir}/segments \\\n"
        f"      --json    {save_dir}/segments_{stem}.json\n"
        f"{'─'*55}"
    )


if __name__ == "__main__":
    main()