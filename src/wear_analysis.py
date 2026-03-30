"""
wear_analysis.py
================
1D wear measurement.

At y = midpoint of pt7/pt8:
  (7)→(8)  = d_side, horizontal (vertical chain thickness in x-direction)
  (9)→(10) = horizontal chord of green circle at same y
             (horizontal wire width in x-direction at that height)

  remaining     = |x_tool − x_orig|
  measured_depth = max(0, d_side − remaining)
  wear%         = measured_depth / d_mean × 100

Purple overlay kept for visual only.
"""

from __future__ import annotations
import logging, math
from typing import Dict, List, Tuple
import cv2, numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _collect_all_tips(hw):
    tips = []
    for seg in hw.get("segments", []):
        for tip in seg.get("tips", []):
            tips.append({**tip, "seg_index": seg["seg_index"],
                         "b_px": seg["b_px"],
                         "slope_top": seg["slope_top"],
                         "intercept_top": seg["intercept_top"],
                         "slope_bot": seg["slope_bot"],
                         "intercept_bot": seg["intercept_bot"]})
    return tips


def _find_x_on_curve_nearest(curve, y_target, x_near):
    if len(curve) < 2: return None
    cands = []
    for i in range(len(curve)-1):
        y0, y1 = curve[i,1], curve[i+1,1]
        if (y0 <= y_target <= y1) or (y1 <= y_target <= y0):
            dy = y1 - y0
            if abs(dy) < 1e-6: x = (curve[i,0]+curve[i+1,0])/2
            else: frac=(y_target-y0)/dy; x=curve[i,0]+frac*(curve[i+1,0]-curve[i,0])
            cands.append(x)
    if not cands:
        return float(curve[np.argmin(np.abs(curve[:,1]-y_target)), 0])
    cands.sort(key=lambda x: abs(x - x_near))
    return cands[0]


def _circle_chord_x(cx, cy, r, y_target):
    """Horizontal chord of circle at y_target → (x_left, x_right)."""
    dy = y_target - cy
    if abs(dy) >= r:
        return cx, cx
    dx = math.sqrt(r*r - dy*dy)
    return cx - dx, cx + dx


# ══════════════════════════════════════════════════════════════════
# Build interference mask (purple overlay — visual only)
# ══════════════════════════════════════════════════════════════════

def _build_crescent_mask(side_arc, wire_seg, shape, full_mask=None):
    h, w = shape
    s_top = wire_seg["slope_top"]; i_top = wire_seg["intercept_top"]
    s_bot = wire_seg["slope_bot"]; i_bot = wire_seg["intercept_bot"]
    outer_pts = side_arc["outer_pts"].copy()
    inner_pts = side_arc["inner_pts"].copy()
    if outer_pts[0,1] > outer_pts[-1,1]: outer_pts = outer_pts[::-1]
    if inner_pts[0,1] > inner_pts[-1,1]: inner_pts = inner_pts[::-1]
    ywt_o = s_top*outer_pts[:,0]+i_top; ywb_o = s_bot*outer_pts[:,0]+i_bot
    mo = (outer_pts[:,1] >= ywt_o-5) & (outer_pts[:,1] <= ywb_o+5); occ = outer_pts[mo]
    ywt_i = s_top*inner_pts[:,0]+i_top; ywb_i = s_bot*inner_pts[:,0]+i_bot
    mi = (inner_pts[:,1] >= ywt_i-5) & (inner_pts[:,1] <= ywb_i+5); icc = inner_pts[mi]
    if len(occ) < 2 or len(icc) < 2:
        return np.zeros((h,w), dtype=np.uint8)
    poly = np.vstack([occ, icc[::-1]]).astype(np.int32)
    mask_out = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(mask_out, [poly], 255)
    wb = np.zeros((h,w), dtype=np.uint8)
    xmn = max(0, int(min(occ[:,0].min(), icc[:,0].min()))-10)
    xmx = min(w, int(max(occ[:,0].max(), icc[:,0].max()))+10)
    for x in range(xmn, xmx):
        yt, yb = max(0,int(s_top*x+i_top)), min(h-1,int(s_bot*x+i_bot))
        if yt < yb: wb[yt:yb+1,x] = 255
    mask_out = cv2.bitwise_and(mask_out, wb)
    if full_mask is not None:
        _, bw = cv2.threshold(full_mask, 127, 255, cv2.THRESH_BINARY)
        if bw.shape[:2] != (h,w): bw = cv2.resize(bw,(w,h),interpolation=cv2.INTER_NEAREST)
        mask_out = cv2.bitwise_and(mask_out, bw)
    return mask_out


def _build_circle_mask(tip, shape):
    h, w = shape
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.circle(mask, (int(tip["center"][0]), int(tip["center"][1])),
               int(tip["radius"]), 255, -1)
    return mask


# ══════════════════════════════════════════════════════════════════
# Compute wear
# ══════════════════════════════════════════════════════════════════

def compute_wear(recon_results, hw_result, image_shape=None, vert_mask=None):
    empty = {"pairs":[], "d_mean_px":0, "b_mean_px":0,
             "wear_pct_left":0, "wear_pct_right":0}
    if not recon_results or not hw_result.get("segments"):
        return empty

    res = recon_results[0]
    cp_t, cp_b = res["cp_top"], res["cp_bot"]
    side_arcs = res.get("side_arcs", [])
    if not side_arcs: return empty
    if image_shape is None: image_shape = (1200, 1600)
    h, w = image_shape

    x_center = cp_t["x_apex"]
    d_mean = res.get("d_mean_px", 0)
    all_tips = _collect_all_tips(hw_result)

    pairs, used = [], set()
    for sa in side_arcs:
        side = "left" if sa["pt7"][0] < x_center else "right"

        ref_x = float(sa["pt7"][0])
        best, bd, bi = None, float("inf"), -1
        for i, t in enumerate(all_tips):
            if i in used: continue
            dd = abs(t["center"][0] - ref_x)
            if dd < bd: bd, best, bi = dd, t, i
        if best is None: continue
        used.add(bi)

        cx_c, cy_c, r_c = best["center"][0], best["center"][1], best["radius"]

        # (7)→(8): d_side (horizontal, vertical chain thickness)
        d_side = sa["d_side_px"]
        pt7, pt8 = sa["pt7"], sa["pt8"]

        # y_measure = circle center y (guaranteed to be within the circle)
        y_measure = cy_c

        # (9)→(10): horizontal chord of green circle at y_measure
        x9, x10 = _circle_chord_x(cx_c, cy_c, r_c, y_measure)
        pt9 = (x9, y_measure)    # left edge of chord
        pt10 = (x10, y_measure)  # right edge of chord

        # x_tool: circle edge facing link center (at y_measure)
        if side == "left":
            x_tool = x9    # right edge faces center
        else:
            x_tool = x10     # left edge faces center

        # x_orig: outer side-arc at y_measure
        x_orig = _find_x_on_curve_nearest(sa["outer_pts"], y_measure, cx_c)
        if x_orig is None: continue

        remaining = abs(x_tool - x_orig)
        measured_depth = max(0.0, d_side - remaining)
        # wear_pct = (measured_depth / d_mean * 100.0) if d_mean > 0 else 0.0

        # wear = abs((pt7, y_measure)   - (pt9, y_measure))
        # measured_depth = max(0.0, wear - d_side)
        wear_pct = (measured_depth / d_side * 100.0) 

        # Purple overlay
        wire_seg = {"slope_top": best["slope_top"], "intercept_top": best["intercept_top"],
                    "slope_bot": best["slope_bot"], "intercept_bot": best["intercept_bot"]}
        cs_mask = _build_crescent_mask(sa, wire_seg, (h,w), full_mask=vert_mask)
        cs_area = int(cs_mask.sum()//255)
        circ_mask = _build_circle_mask(best, (h,w))
        interf_mask = cv2.bitwise_and(circ_mask, cs_mask)
        interf_area = int(interf_mask.sum()//255)

        logger.info(
            f"  Wear {side}: y={y_measure:.0f}  x9={x9:.0f} x10={x10:.0f}  "
            f"x_tool={x_tool:.0f}  x_orig={x_orig:.0f}  "
            f"d_side={d_side:.0f}  rem={remaining:.0f}  "
            f"Dw={measured_depth:.1f}  wear={wear_pct:.1f}%")

        pairs.append({
            "side": side, "d_side": d_side,
            "x9": x9, "x10": x10,
            "pt9": pt9, "pt10": pt10,
            "x_tool": x_tool, "x_orig": x_orig,
            "y_measure": y_measure, "remaining": remaining,
            "measured_depth": measured_depth, "wear_pct": wear_pct,
            "tip": best, "pt7": pt7, "pt8": pt8,
            "cs_mask": cs_mask, "circ_mask": circ_mask,
            "interf_mask": interf_mask,
            "cs_area": cs_area, "interf_area": interf_area,
        })

    return {
        "pairs": pairs, "d_mean_px": d_mean,
        "b_mean_px": hw_result.get("b_mean_px", 0),
        "wear_pct_left":  next((p["wear_pct"] for p in pairs if p["side"]=="left"),  0),
        "wear_pct_right": next((p["wear_pct"] for p in pairs if p["side"]=="right"), 0),
    }


# ══════════════════════════════════════════════════════════════════
# Drawing
# ══════════════════════════════════════════════════════════════════

RED=(0,50,255); BLUE=(220,90,20); YELLOW=(0,220,255); GREEN=(50,200,50)
ORANGE=(0,165,255); WHITE=(255,255,255); MAGENTA=(255,50,255)
BLUE_FILL=(255,160,40); MAGENTA_FILL=(200,50,200); DARK_RED=(0,0,180)

def _pv(c,x): return np.polyval(c,x)
def _draw_curve(v,c,x1,x2,col,th=3):
    if c is None: return
    h=v.shape[0]; xs=np.arange(x1,x2+1,2,dtype=np.float64)
    ys=np.clip(_pv(c,xs).astype(int),0,h-1)
    p=np.column_stack([xs,ys]).astype(np.int32).reshape(-1,1,2)
    if len(p)>=2: cv2.polylines(v,[p],False,col,th,cv2.LINE_AA)
def _draw_dashed(v,pts,col,th=3,seg=14,gap=10):
    p=pts.astype(np.int32).reshape(-1,1,2); n=len(p); i=0; on=True
    while i<n:
        e=min(i+(seg if on else gap),n)
        if on and e-i>=2: cv2.polylines(v,[p[i:e]],False,col,th,cv2.LINE_AA)
        i=e; on=not on
def _dot(v,pt,col,r=8,lbl=None):
    if pt is None: return
    cx,cy=int(pt[0]),int(pt[1])
    cv2.circle(v,(cx,cy),r,col,-1,cv2.LINE_AA)
    cv2.circle(v,(cx,cy),r,WHITE,1,cv2.LINE_AA)
    if lbl: cv2.putText(v,lbl,(cx+r+2,cy-r),cv2.FONT_HERSHEY_SIMPLEX,0.5,(30,220,30),2,cv2.LINE_AA)


def draw_full_recon(image, recon_results, hw_result, wear_result, vert_mask=None):
    vis = image.copy()
    if not recon_results: return vis
    res = recon_results[0]
    cp_t, cp_b = res["cp_top"], res["cp_bot"]

    # 1. Purple overlay
    for p in wear_result.get("pairs",[]):
        im = p.get("interf_mask")
        if im is not None:
            ov=np.zeros_like(vis); ov[im>0]=MAGENTA_FILL
            vis=cv2.addWeighted(vis,1.0,ov,0.45,0)

    # 2. Outer arcs
    _draw_curve(vis,res["coef_outer_top"],res["top_x1"],res["top_x2"],RED,3)
    _draw_curve(vis,res["coef_outer_bot"],res["x1"],res["x2"],RED,3)

    # 3. Inner arcs (clipped)
    _draw_curve(vis,res["coef_inner_top"],cp_t["x2_left"],cp_t["x2_right"],BLUE,3)
    _draw_curve(vis,res["coef_inner_bot"],cp_b["x2_left"],cp_b["x2_right"],BLUE,3)

    # 4. Side arcs + (7)→(8)
    for sa in res.get("side_arcs",[]):
        _draw_dashed(vis,sa["outer_pts"],ORANGE,3)
        _draw_dashed(vis,sa["inner_pts"],YELLOW,2,seg=10,gap=8)
        _dot(vis,sa["pt7"],ORANGE,10,"7"); _dot(vis,sa["pt8"],ORANGE,10,"8")
        p7,p8=sa["pt7"],sa["pt8"]
        cv2.line(vis,(int(p7[0]),int(p7[1])),(int(p8[0]),int(p8[1])),YELLOW,3,cv2.LINE_AA)
        ds=sa["d_side_px"]; mx,my=(p7[0]+p8[0])/2,(p7[1]+p8[1])/2
        cv2.putText(vis,f"d={ds:.0f}",(int(mx)+8,int(my)+4),cv2.FONT_HERSHEY_SIMPLEX,0.6,YELLOW,2,cv2.LINE_AA)

    # 5. Key points
    for cp in (cp_t,cp_b):
        _dot(vis,cp.get("pt3"),RED,8,"3"); _dot(vis,cp.get("pt4"),BLUE,8,"4")
        for s in ("left","right"):
            _dot(vis,cp.get(f"pt1_{s}"),RED,8,"1"); _dot(vis,cp.get(f"pt2_{s}"),BLUE,8,"2")

    # 6. d at apex
    for co,ci,cp in [(res["coef_outer_top"],res["coef_inner_top"],cp_t),
                     (res["coef_outer_bot"],res["coef_inner_bot"],cp_b)]:
        if co is not None and ci is not None:
            xap=cp["x_apex"]; yo,yi=int(_pv(co,xap)),int(_pv(ci,xap)); d=abs(yi-yo)
            cv2.line(vis,(xap,yo),(xap,yi),YELLOW,3,cv2.LINE_AA)
            cv2.putText(vis,f"d={d:.0f}px",(xap+10,(yo+yi)//2),cv2.FONT_HERSHEY_SIMPLEX,0.8,YELLOW,2,cv2.LINE_AA)

    # 7. Horizontal wire edges + circles
    if hw_result:
        for seg in hw_result.get("segments",[]):
            cv2.line(vis,seg["top_line"][0],seg["top_line"][1],RED,2,cv2.LINE_AA)
            cv2.line(vis,seg["bot_line"][0],seg["bot_line"][1],RED,2,cv2.LINE_AA)
            xm=(seg["x1"]+seg["x2"])//2
            yt=int(seg["slope_top"]*xm+seg["intercept_top"])
            yb=int(seg["slope_bot"]*xm+seg["intercept_bot"])
            cv2.line(vis,(xm,yt),(xm,yb),YELLOW,2,cv2.LINE_AA)
            cv2.putText(vis,f"b={seg['b_px']:.0f}px",(xm+8,(yt+yb)//2),cv2.FONT_HERSHEY_SIMPLEX,0.7,YELLOW,2,cv2.LINE_AA)
            for tip in seg["tips"]:
                cx,cy=int(tip["center"][0]),int(tip["center"][1]); r=int(tip["radius"])
                cv2.circle(vis,(cx,cy),r,GREEN,2,cv2.LINE_AA)
                cv2.line(vis,(cx,cy-r),(cx,cy+r),YELLOW,2,cv2.LINE_AA)

    # 8. (9)→(10) HORIZONTAL + Δw + wear labels
    for p in wear_result.get("pairs",[]):
        ym = int(p["y_measure"])
        x9, x10 = int(p["x9"]), int(p["x10"])
        xt, xo = int(p["x_tool"]), int(p["x_orig"])
        md, wp, side = p["measured_depth"], p["wear_pct"], p["side"]

        # (9)→(10): HORIZONTAL red line (circle chord at y_measure)
        cv2.line(vis, (x9, ym), (x10, ym), RED, 3, cv2.LINE_AA)
        _dot(vis, (x9, ym), RED, 6, "9")
        _dot(vis, (x10, ym), RED, 6, "10")

        # Δw arrow from x_orig to x_tool at y_measure+offset for visibility
        cv2.arrowedLine(vis, (xo, ym+8), (xt, ym+8),
                        DARK_RED, 2, cv2.LINE_AA, tipLength=0.1)

        # Label
        if side == "left":
            lx = min(x9, xo) - 10
            cv2.putText(vis, f"wear={wp:.1f}%", (lx-130, ym-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MAGENTA, 2, cv2.LINE_AA)
            cv2.putText(vis, f"Dw={md:.0f}px", (lx-130, ym+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1, cv2.LINE_AA)
        else:
            lx = max(x10, xo) + 10
            cv2.putText(vis, f"wear={wp:.1f}%", (lx, ym-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MAGENTA, 2, cv2.LINE_AA)
            cv2.putText(vis, f"Dw={md:.0f}px", (lx, ym+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1, cv2.LINE_AA)

    # 9. Summary
    dm=wear_result.get("d_mean_px",0); bm=wear_result.get("b_mean_px",0)
    wl=wear_result.get("wear_pct_left",0); wr=wear_result.get("wear_pct_right",0)
    py=max(20,res.get("top_y1",40)-50)
    for i,line in enumerate([
        f"d={dm:.0f}px  b={bm:.0f}px",
        f"wear L={wl:.1f}%  R={wr:.1f}%",
    ]):
        y=py+i*24; cv2.rectangle(vis,(8,y-16),(330,y+6),(20,20,20),-1)
        cv2.putText(vis,line,(14,y),cv2.FONT_HERSHEY_SIMPLEX,0.65,WHITE,2,cv2.LINE_AA)

    return vis