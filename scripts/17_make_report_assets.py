from pathlib import Path
import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np


BG = (10, 12, 18)
PANEL = (18, 22, 30)
WHITE = (242, 244, 248)
MUTED = (170, 176, 188)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.load(open(path, "r", encoding="utf-8"))


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def resize_contain(img, out_w, out_h, bg=(0, 0, 0)):
    h, w = img.shape[:2]
    scale = min(out_w / max(w, 1), out_h / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    rs = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = rs
    return canvas


def put_text(img, text, x, y, scale=1.0, color=WHITE, thick=2):
    cv2.putText(
        img,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thick),
        lineType=cv2.LINE_AA,
    )


def fmt(x):
    return "--" if x is None else f"{x:.3f}"


def write_results_table(out_path, h1, h3, h5):
    rows = [
        ("H1", "Learned", h1["model"]),
        ("H1", "Copy-last", h1["copy_last"]),
        ("H1", "Const-vel", h1["constant_velocity"]),
        ("H3", "Learned", h3["model"]),
        ("H3", "Copy-last", h3["copy_last"]),
        ("H3", "Const-vel", h3["constant_velocity"]),
        ("H5", "Learned", h5["model"]),
        ("H5", "Copy-last", h5["copy_last"]),
        ("H5", "Const-vel", h5["constant_velocity"]),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Forecasting results across prediction horizons. The learned model is most beneficial at longer horizons, while copy-last remains very strong for one-step prediction in this mostly static indoor scene.}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Horizon & Method & L2 $\downarrow$ & Vis F1 $\uparrow$ & Edge F1 $\uparrow$ & Present Acc $\uparrow$ \\")
    lines.append(r"\midrule")

    for hz, name, r in rows:
        lines.append(
            f"{hz} & {name} & {fmt(r['l2'])} & {fmt(r['vis_f1'])} & {fmt(r['edge_f1'])} & {fmt(r['present_acc'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print("saved", out_path)


def write_ablation_table(out_path, box_h1, seg_h1, hybrid_h1, cam_h5, nocam_h5):
    rows = []

    if box_h1 is not None:
        rows.append(("Representation", "Box-only transformer", "H1", box_h1["model"]))
    if seg_h1 is not None:
        rows.append(("Representation", "Pure segmask", "H1", seg_h1["model"]))
    if hybrid_h1 is not None:
        rows.append(("Representation", "Hybrid detect+seg", "H1", hybrid_h1["model"]))
    if cam_h5 is not None:
        rows.append(("Forecasting", "Hybrid + camera motion", "H5", cam_h5["model"]))
    if nocam_h5 is not None:
        rows.append(("Forecasting", "Hybrid no camera motion", "H5", nocam_h5["model"]))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablations on representation and forecasting design choices. Hybrid detect+seg gives the best H1 representation tradeoff, while explicit camera-motion conditioning does not improve H5 performance in this limited single-sequence setting.}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Group & Variant & Horizon & L2 $\downarrow$ & Vis F1 $\uparrow$ & Edge F1 $\uparrow$ \\")
    lines.append(r"\midrule")

    for group, variant, hz, r in rows:
        lines.append(
            f"{group} & {variant} & {hz} & {fmt(r['l2'])} & {fmt(r['vis_f1'])} & {fmt(r['edge_f1'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print("saved", out_path)


def make_motion_visibility_figure(out_path, bd_h1, bd_h3, bd_h5):
    horizons = ["H1", "H3", "H5"]
    breakdowns = {"H1": bd_h1, "H3": bd_h3, "H5": bd_h5}
    methods = ["model", "copy_last", "constant_velocity"]
    labels = {
        "model": "Learned",
        "copy_last": "Copy-last",
        "constant_velocity": "Const-vel",
    }

    plt.figure(figsize=(11.5, 4.8))

    plt.subplot(1, 2, 1)
    for m in methods:
        ys = [breakdowns[h][m]["static_vs_moving"]["moving"]["l2"] for h in horizons]
        plt.plot(horizons, ys, marker="o", markersize=8, linewidth=2.5, label=labels[m])
    plt.title("Moving-object L2 vs horizon", fontsize=16)
    plt.xlabel("Horizon", fontsize=13)
    plt.ylabel("L2", fontsize=13)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=11)

    plt.subplot(1, 2, 2)
    for m in methods:
        ys = [breakdowns[h][m]["visibility_transitions"]["hid_to_vis"]["vis_acc"] for h in horizons]
        plt.plot(horizons, ys, marker="o", markersize=8, linewidth=2.5, label=labels[m])
    plt.title("Hidden→visible accuracy vs horizon", fontsize=16)
    plt.xlabel("Horizon", fontsize=13)
    plt.ylabel("Visibility accuracy", fontsize=13)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print("saved", out_path)


def make_qualitative_figure(out_path, room_only, room_hist, room_pred, overlay, triptych):
    canvas = np.full((1500, 2600, 3), BG, dtype=np.uint8)

    put_text(canvas, "Future 3D Semantic Scene Graph Prediction", 50, 75, scale=1.45, color=WHITE, thick=3)
    put_text(canvas, "Room-scale scene context first, local future graph evidence second", 52, 118, scale=0.78, color=MUTED, thick=2)

    hero = resize_contain(room_pred, 1680, 930, bg=(247, 248, 250))
    hist = resize_contain(room_hist, 760, 360, bg=(247, 248, 250))
    ov = resize_contain(overlay, 1120, 420, bg=(0, 0, 0))
    trip = resize_contain(triptych, 1120, 420, bg=(247, 247, 247))

    canvas[170:170 + hero.shape[0], 40:40 + hero.shape[1]] = hero
    canvas[170:170 + hist.shape[0], 1800:1800 + hist.shape[1]] = hist
    canvas[1060:1060 + ov.shape[0], 40:40 + ov.shape[1]] = ov
    canvas[1060:1060 + trip.shape[0], 1200:1200 + trip.shape[1]] = trip

    cv2.rectangle(canvas, (40, 170), (1720, 1100), (90, 170, 255), 2)
    cv2.rectangle(canvas, (1800, 170), (2560, 530), (70, 145, 255), 2)
    cv2.rectangle(canvas, (40, 1060), (1160, 1480), (255, 168, 90), 2)
    cv2.rectangle(canvas, (1200, 1060), (2320, 1480), (110, 220, 130), 2)

    put_text(canvas, "Forecasted future graph in room context", 50, 158, scale=0.70, color=WHITE, thick=2)
    put_text(canvas, "Recent scene graph state", 1810, 158, scale=0.62, color=WHITE, thick=2)
    put_text(canvas, "Local forecast overlay", 50, 1048, scale=0.62, color=WHITE, thick=2)
    put_text(canvas, "History / prediction / ground truth", 1210, 1048, scale=0.62, color=WHITE, thick=2)

    put_text(canvas, "Blue = recent state, orange = forecasted future graph", 1810, 580, scale=0.56, color=MUTED, thick=2)
    put_text(canvas, "The room mesh provides scene context; the prediction is object-centric scene structure.", 1810, 616, scale=0.56, color=MUTED, thick=2)

    save_rgb(out_path, canvas)
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-h1", required=True)
    ap.add_argument("--eval-h3", required=True)
    ap.add_argument("--eval-h5", required=True)
    ap.add_argument("--eval-box-h1", default="")
    ap.add_argument("--eval-seg-h1", default="")
    ap.add_argument("--eval-hybrid-h1", required=True)
    ap.add_argument("--eval-h5-nocam", default="")
    ap.add_argument("--bd-h1", required=True)
    ap.add_argument("--bd-h3", required=True)
    ap.add_argument("--bd-h5", required=True)
    ap.add_argument("--room-only", required=True)
    ap.add_argument("--room-history", required=True)
    ap.add_argument("--room-pred", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--triptych", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h1 = load_json(args.eval_h1)
    h3 = load_json(args.eval_h3)
    h5 = load_json(args.eval_h5)

    box_h1 = load_json(args.eval_box_h1) if args.eval_box_h1 else None
    seg_h1 = load_json(args.eval_seg_h1) if args.eval_seg_h1 else None
    hybrid_h1 = load_json(args.eval_hybrid_h1)
    h5_nocam = load_json(args.eval_h5_nocam) if args.eval_h5_nocam else None

    bd_h1 = load_json(args.bd_h1)
    bd_h3 = load_json(args.bd_h3)
    bd_h5 = load_json(args.bd_h5)

    write_results_table(out_dir / "results_table.tex", h1, h3, h5)
    write_ablation_table(out_dir / "ablation_table.tex", box_h1, seg_h1, hybrid_h1, h5, h5_nocam)
    make_motion_visibility_figure(out_dir / "motion_visibility_summary.png", bd_h1, bd_h3, bd_h5)

    room_only = load_rgb(args.room_only)
    room_hist = load_rgb(args.room_history)
    room_pred = load_rgb(args.room_pred)
    overlay = load_rgb(args.overlay)
    triptych = load_rgb(args.triptych)

    make_qualitative_figure(
        out_dir / "qualitative_roomscale_figure.png",
        room_only,
        room_hist,
        room_pred,
        overlay,
        triptych,
    )


if __name__ == "__main__":
    main()
