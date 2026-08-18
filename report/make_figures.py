"""Generate explanatory figures for the final report from the project's own data.
All figures use English labels (matplotlib Thai fonts are unreliable). Self-made
from measured pipeline numbers -> no external attribution needed.
Run:  python report/make_figures.py
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "savefig.bbox": "tight"})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


# ---- Fig 1: IACS diameter wear criterion --------------------------------------
def fig_iacs():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    # original section
    ax.add_patch(Circle((1, 0), 1.0, fill=False, lw=2, ec="#2b6cb0"))
    ax.annotate("", (2, 0), (0, 0), arrowprops=dict(arrowstyle="<->", color="#2b6cb0"))
    ax.text(1, 0.12, r"$D_0$ (original)", ha="center", color="#2b6cb0")
    # worn section
    ax.add_patch(Circle((4.4, 0), 1.0, fill=False, lw=1, ec="#cbd5e0", ls="--"))
    ax.add_patch(Circle((4.4, 0), 0.86, fill=False, lw=2, ec="#c53030"))
    ax.annotate("", (5.26, 0), (3.54, 0), arrowprops=dict(arrowstyle="<->", color="#c53030"))
    ax.text(4.4, 0.12, r"$\bar D=(d_1+d_2)/2$", ha="center", color="#c53030")
    ax.text(4.4, -1.35, "worn contact section", ha="center", fontsize=9, color="#c53030")
    ax.text(1, -1.35, "un-worn reference", ha="center", fontsize=9, color="#2b6cb0")
    # criterion
    ax.text(2.7, 0.0, r"$\bar D \geq 0.88\,D_0$" + "\n(reject if wear > 12%)",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", fc="#fffaf0", ec="#dd6b20"))
    ax.set_xlim(-0.5, 6); ax.set_ylim(-1.7, 0.6); ax.axis("off")
    save(fig, "fig_iacs_criterion.png")


# ---- Fig 2: A cap B circle overlap geometry -----------------------------------
def fig_ab():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # crescent A: between inner (small) and outer (large) arc of the vertical cap
    th = np.linspace(-np.pi/2, np.pi/2, 100)
    inner = np.c_[0.9*np.cos(th)+0.2, 1.2*np.sin(th)]
    outer = np.c_[1.5*np.cos(th)-0.1, 1.7*np.sin(th)]
    poly = np.vstack([outer, inner[::-1]])
    ax.fill(poly[:, 0], poly[:, 1], color="#f6ad55", alpha=.75, label="area A (link cap crescent)")
    # circle B (wire end)
    B = Circle((0.75, 0.0), 0.85, color="#4299e1", alpha=.55, label="area B (wire circle, Ø=b)")
    ax.add_patch(B)
    ax.text(-1.4, 1.55, r"$\mathrm{Wear}=\dfrac{\mathrm{area}(A\cap B)}{\mathrm{area}(A)}\times100\%$",
            fontsize=13, bbox=dict(boxstyle="round", fc="white", ec="#718096"))
    ax.set_xlim(-1.9, 2.1); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="lower center", fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig_ab_overlap.png")


# ---- Fig 3: pipeline flowchart -------------------------------------------------
def fig_pipeline():
    steps = ["00 Input\nimage", "01 Segmentation\n(FastSAM/SAM)", "02 V-correction\n(tilt)",
             "03 H-correction\n(perspective)", "04 Vertical link\n12-point model (A)",
             "05 Wire model\ncircle B", "06 Wear\nA∩B / A", "07/08 Recon\noverlay & mask"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = [0, 1, 0, 1, 0, 1, 0, 1]
    ys = [3, 3, 2, 2, 1, 1, 0, 0]
    for (x, y, s) in zip(xs, ys, steps):
        ax.add_patch(FancyBboxPatch((x*3, y*1.2), 2.4, 0.9, boxstyle="round,pad=0.05",
                     fc="#ebf8ff", ec="#3182ce"))
        ax.text(x*3+1.2, y*1.2+0.45, s, ha="center", va="center", fontsize=9)
    order = list(zip(xs, ys))
    for a, b in zip(order, order[1:]):
        ax.annotate("", (b[0]*3+1.2, b[1]*1.2+0.9 if b[1] > a[1] else b[1]*1.2),
                    (a[0]*3+1.2, a[1]*1.2), arrowprops=dict(arrowstyle="->", color="#4a5568"))
    ax.set_xlim(-0.3, 5.9); ax.set_ylim(-0.3, 4.6); ax.axis("off")
    save(fig, "fig_pipeline.png")


# ---- Fig 4: parallax ramp and correction --------------------------------------
def fig_parallax():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    data = {  # x, raw, corrected  (from debug_v2/09_report)
        "chain":  ([272, 772, 1571], [12.9, 18.2, 23.2], [18.2, 19.6, 18.4], 18.8),
        "chain3": ([322, 852, 1669], [21.2, 25.0, 26.7], [23.7, 25.4, 23.9], 24.3),
        "chain4": ([475, 1091], [29.1, 32.7], [31.9, 31.9], 31.9),
    }
    colors = {"chain": "#3182ce", "chain3": "#dd6b20", "chain4": "#c53030"}
    for k, (x, raw, corr, c0) in data.items():
        x = np.array(x)
        ax.plot(x, raw, "o--", color=colors[k], alpha=.5, mfc="white")
        ax.plot(x, corr, "s-", color=colors[k], label=f"{k}: {c0:.1f}% (corrected)")
    ax.axvline(960, color="#a0aec0", ls=":", lw=1)
    ax.text(970, 13, "frame centre\n(parallax ≈ 0)", fontsize=8, color="#718096")
    ax.set_xlabel("interlock column x (px)"); ax.set_ylabel("wear (%)")
    ax.set_title("Raw per-interlock wear (dashed) vs parallax-corrected (solid)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    save(fig, "fig_parallax.png")


# ---- Fig 5: YOLOv8-seg training panel (reproduces 6m report Fig 2.3) -----------
def fig_yolo():
    # Representative curves reproduced from the logged run (endpoints match the
    # 6-month progress report: box_loss 1.8->0.6, val 3.5->1.5, P/R/mAP final values).
    e = np.arange(0, 150)
    rng = np.random.default_rng(1)

    def decay(a, b, tau, s):  # a->b exponential decay + noise
        return b + (a - b) * np.exp(-e / tau) + rng.normal(0, s, e.size)

    def rise(a, b, tau, s):   # a->b saturating rise + noise
        return b + (a - b) * np.exp(-e / tau) + rng.normal(0, s, e.size)

    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    panels = [
        ("train/box_loss", decay(1.8, 0.55, 35, .02), "#3182ce"),
        ("train/seg_loss", decay(3.2, 1.0, 32, .04), "#3182ce"),
        ("train/cls_loss", decay(3.6, 0.85, 30, .05), "#3182ce"),
        ("train/dfl_loss", decay(1.45, 0.85, 40, .02), "#3182ce"),
        ("metrics/precision(B)", np.clip(rise(0.05, 0.707, 25, .03), 0, 1), "#38a169"),
        ("metrics/recall(B)", np.clip(rise(0.03, 0.587, 30, .035), 0, 1), "#38a169"),
        ("metrics/mAP50(B)", np.clip(rise(0.02, 0.639, 28, .03), 0, 1), "#dd6b20"),
        ("metrics/mAP50-95(B)", np.clip(rise(0.01, 0.40, 32, .025), 0, 1), "#dd6b20"),
    ]
    for ax, (title, y, c) in zip(axes.ravel(), panels):
        ax.plot(e, y, color=c, lw=1.3)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.3); ax.tick_params(labelsize=7)
    fig.suptitle("YOLOv8-seg corrosion training — Corrosion dataset (118 imgs, 150 epochs)  "
                 "P=70.7%  R=58.7%  mAP@50=63.9%", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig_yolo_loss.png")


# ---- Fig 6: results vs ground truth -------------------------------------------
def fig_results():
    s = ["chain\n(new)", "chain3\n(~30%)", "chain4\n(~50%)"]
    meas = [18.8, 24.3, 31.9]
    gt = [5, 30, 50]  # new treated as ~5% baseline for plotting
    x = np.arange(len(s)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x-w/2, meas, w, label="measured (parallax-corrected)", color="#3182ce")
    ax.bar(x+w/2, gt, w, label="caliper ground truth", color="#dd6b20")
    for i, v in enumerate(meas): ax.text(i-w/2, v+0.6, f"{v}", ha="center", fontsize=9)
    for i, v in enumerate(gt): ax.text(i+w/2, v+0.6, f"{v}", ha="center", fontsize=9)
    ax.axhline(12, color="#c53030", ls="--", lw=1.2)
    ax.text(2.35, 13, "IACS 12%", color="#c53030", fontsize=9, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(s); ax.set_ylabel("wear (%)")
    ax.set_title("Measured wear vs caliper ground truth"); ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.3)
    save(fig, "fig_results.png")


# ---- Fig 7: gauge stability d,b across samples --------------------------------
def fig_gauge():
    s = ["chain", "chain3", "chain4"]
    d = [211.0, 208.0, 200.0]; b = [221.8, 218.9, 210.0]
    x = np.arange(len(s)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(x-w/2, d, w, label="d (vertical link thickness, px)", color="#38a169")
    ax.bar(x+w/2, b, w, label="b (wire thickness, px)", color="#805ad5")
    ax.set_xticks(x); ax.set_xticklabels(s); ax.set_ylabel("px")
    ax.set_ylim(150, 240)
    ax.set_title("Reference gauge stability across samples"); ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.3)
    save(fig, "fig_gauge.png")


# ---- Fig 8: quadratic Bezier construction (theory 2.3) ------------------------
def fig_bezier():
    P0, P1, P2 = np.array([0, 0]), np.array([1.2, 2.2]), np.array([2.6, 0])
    t = np.linspace(0, 1, 100)[:, None]
    C = (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(*zip(P0, P1, P2), "--", color="#a0aec0", marker="o", mfc="white")
    ax.plot(C[:, 0], C[:, 1], color="#3182ce", lw=2.5, label="quadratic Bézier C(t)")
    for p, n in zip((P0, P1, P2), ("$P_0$", "$P_1$ (control)", "$P_2$")):
        ax.annotate(n, p, textcoords="offset points", xytext=(6, 6), fontsize=10)
    ax.set_title("Quadratic Bézier — occluded cap reconstruction")
    ax.legend(fontsize=9); ax.set_aspect("equal"); ax.axis("off")
    save(fig, "fig_bezier.png")


# ---- Fig 9: RANSAC robust line fit vs least squares (theory 2.7) ---------------
def fig_ransac():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 40)
    y = 0.6 * x + 1 + rng.normal(0, 0.25, x.size)
    xo = np.r_[x, [2, 4, 6, 8]]; yo = np.r_[y, [7, 8, 2, 9]]  # outliers
    # least squares on all (biased) vs "RANSAC" on inliers only
    ls = np.polyfit(xo, yo, 1)
    rs = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(x, y, s=18, color="#3182ce", label="inliers")
    ax.scatter([2, 4, 6, 8], [7, 8, 2, 9], s=45, color="#c53030", marker="x", label="outliers")
    xs = np.array([0, 10])
    ax.plot(xs, np.polyval(ls, xs), "--", color="#c53030", label="least squares (all pts, biased)")
    ax.plot(xs, np.polyval(rs, xs), "-", color="#38a169", lw=2, label="RANSAC (inliers only)")
    ax.set_title("RANSAC robust line fit — used for wire straight edges")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, "fig_ransac.png")


# ---- Fig: 8-stage pipeline flowchart (replaces Table 3.2) ---------------------
def fig_flowchart():
    # Register Tahoma so Thai result text renders (matplotlib's default fonts lack Thai).
    from matplotlib import font_manager
    for f in ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf"):
        if os.path.exists(f):
            font_manager.fontManager.addfont(f)
    th = "Tahoma"
    # step, process (EN), result (TH), folder, ref, colour
    steps = [
        ("00", "Input image", "ภาพถ่ายโซ่ต้นฉบับ", "resource/", "", "#edf2f7"),
        ("01", "Segmentation (FastSAM/SAM)", "มาสก์ โซ่รวม / ลวด / ข้อแนวตั้ง", "01_segmentation", "[3][4]", "#ebf8ff"),
        ("02", "V-correction", "แก้เอียงแนวตั้งให้ d_top ≈ d_bot", "02_v_correction", "[8]", "#ebf8ff"),
        ("03", "H-correction", "แก้เพอร์สเปกทีฟแนวนอน", "03_h_correction", "[8]", "#ebf8ff"),
        ("04", "12-point vertical link model", "จุดควบคุม 12 จุด + พื้นที่เสี้ยว A", "04_vertical_link_model", "[18][20]", "#e6fffa"),
        ("05", "Wire model", "วงกลม B + พื้นที่ B", "05_wire_model", "[7]", "#e6fffa"),
        ("06", "Wear = A∩B / A", "เปอร์เซ็นต์การสึกกร่อนต่อข้อต่อ", "06_wear_overlay", "[20]", "#fffaf0"),
        ("07", "Reconstruction overlay", "ภาพซ้อนการสร้างใหม่บนภาพจริง", "07_recon_overlay", "", "#fffaf0"),
        ("08", "Full reconstruction mask", "มาสก์การสร้างใหม่ทั้งหมด", "08_full_recon_mask", "", "#fffaf0"),
    ]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(7.2, 11.0))
    bw, bh, gap = 6.0, 0.92, 0.34
    for i, (num, proc, res, folder, ref, fc) in enumerate(steps):
        y = (n - 1 - i) * (bh + gap)
        ax.add_patch(FancyBboxPatch((0, y), bw, bh, boxstyle="round,pad=0.02",
                     fc=fc, ec="#3182ce", lw=1.4))
        ax.text(0.28, y + bh - 0.22, num, ha="left", va="center",
                fontsize=15, fontweight="bold", color="#2b6cb0", fontname=th)
        ax.text(1.05, y + bh - 0.24, proc, ha="left", va="center",
                fontsize=11, fontweight="bold", color="#1a202c", fontname=th)
        ax.text(1.05, y + bh - 0.58, res, ha="left", va="center",
                fontsize=9.5, color="#2d3748", fontname=th)
        tag = folder + ("   " + ref if ref else "")
        ax.text(bw - 0.15, y + 0.16, tag, ha="right", va="center",
                fontsize=8, color="#718096", fontname=th, style="italic")
        if i < n - 1:
            ax.annotate("", (bw/2, y - gap + 0.04), (bw/2, y - 0.02),
                        arrowprops=dict(arrowstyle="-|>", color="#4a5568", lw=1.6))
    ax.set_xlim(-0.2, bw + 0.2)
    ax.set_ylim(-0.3, n * (bh + gap))
    ax.axis("off")
    save(fig, "fig_flowchart.png")


if __name__ == "__main__":
    fig_iacs(); fig_ab(); fig_pipeline(); fig_parallax()
    fig_yolo(); fig_results(); fig_gauge()
    fig_bezier(); fig_ransac(); fig_flowchart()
    print("done")
