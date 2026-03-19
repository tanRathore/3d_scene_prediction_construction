from pathlib import Path
import argparse
import json

import cv2
import imageio.v2 as imageio
import numpy as np


BG = (8, 8, 14)
PANEL = (18, 18, 28)
WHITE = (235, 235, 240)
MUTED = (170, 170, 180)
ACCENTS = {
    "H1": (90, 170, 255),
    "H3": (255, 175, 90),
    "H5": (110, 215, 130),
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def crop_nonblack(img, thr=10, pad=24):
    mask = img.max(axis=2) > thr
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img.copy()

    x1 = max(0, int(xs.min()) - pad)
    x2 = min(img.shape[1], int(xs.max()) + pad + 1)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(img.shape[0], int(ys.max()) + pad + 1)
    return img[y1:y2, x1:x2].copy()


def resize_contain(img, out_w, out_h, bg=BG):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((out_h, out_w, 3), bg, dtype=np.uint8)

    scale = min(out_w / w, out_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
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


def draw_badge(img, text, x, y, w, h, color):
    x, y, w, h = map(int, [x, y, w, h])
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    put_text(img, text, x + 14, y + int(h * 0.68), scale=0.62, color=(10, 10, 14), thick=2)


def draw_panel_border(img, x, y, w, h, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def metric_triplet(eval_json):
    m = eval_json["model"]
    return [
        f"L2 {m['l2']:.3f}",
        f"Vis F1 {m['vis_f1']:.3f}",
        f"Edge F1 {m['edge_f1']:.3f}",
    ]


def make_single_hero(title, subtitle, accent, overlay, triptych, eval_json, out_path):
    W, H = 1920, 1080
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    put_text(canvas, title, 70, 90, scale=1.6, color=WHITE, thick=3)
    put_text(canvas, subtitle, 70, 130, scale=0.8, color=MUTED, thick=2)

    left_x, left_y, left_w, left_h = 60, 170, 1120, 830
    right_x, right_y, right_w, right_h = 1220, 170, 640, 830

    cv2.rectangle(canvas, (left_x, left_y), (left_x + left_w, left_y + left_h), PANEL, -1)
    cv2.rectangle(canvas, (right_x, right_y), (right_x + right_w, right_y + right_h), PANEL, -1)

    overlay_crop = crop_nonblack(overlay, thr=10, pad=28)
    overlay_big = resize_contain(overlay_crop, left_w - 40, left_h - 40, bg=(0, 0, 0))
    canvas[left_y + 20:left_y + 20 + overlay_big.shape[0], left_x + 20:left_x + 20 + overlay_big.shape[1]] = overlay_big
    draw_panel_border(canvas, left_x, left_y, left_w, left_h, accent)

    trip = resize_contain(triptych, right_w - 40, 340, bg=(245, 245, 245))
    canvas[right_y + 20:right_y + 20 + trip.shape[0], right_x + 20:right_x + 20 + trip.shape[1]] = trip

    put_text(canvas, "Model metrics", right_x + 24, right_y + 420, scale=0.9, color=WHITE, thick=2)
    badges = metric_triplet(eval_json)
    bx = right_x + 24
    by = right_y + 455
    bw = right_w - 48
    bh = 58
    for i, t in enumerate(badges):
        draw_badge(canvas, t, bx, by + i * 78, bw, bh, accent)

    put_text(canvas, "Scene-grounded future graph prediction", right_x + 24, right_y + 725, scale=0.8, color=WHITE, thick=2)
    put_text(canvas, "RGB-D + poses + hybrid mask lifting + TSDF", right_x + 24, right_y + 770, scale=0.72, color=MUTED, thick=2)
    put_text(canvas, "History / Pred / True shown in the inset", right_x + 24, right_y + 810, scale=0.72, color=MUTED, thick=2)

    save_rgb(out_path, canvas)


def make_overview(h1, h3, h5, out_path):
    W, H = 1920, 1080
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    put_text(canvas, "Future 3D Scene Graph Prediction from RGB-D Video", 60, 85, scale=1.45, color=WHITE, thick=3)
    put_text(canvas, "Hybrid object-centric lifting + local TSDF scene grounding + multi-horizon forecasting", 60, 125, scale=0.78, color=MUTED, thick=2)

    cards = [("H1", h1), ("H3", h3), ("H5", h5)]
    x_positions = [45, 660, 1275]
    card_w, card_h = 600, 860

    for (name, data), x in zip(cards, x_positions):
        accent = ACCENTS[name]
        cv2.rectangle(canvas, (x, 170), (x + card_w, 170 + card_h), PANEL, -1)
        draw_panel_border(canvas, x, 170, card_w, card_h, accent)

        put_text(canvas, f"Horizon {name[1:]}", x + 24, 220, scale=1.05, color=WHITE, thick=2)

        overlay_crop = crop_nonblack(data["overlay"], thr=10, pad=28)
        overlay_big = resize_contain(overlay_crop, card_w - 40, 420, bg=(0, 0, 0))
        canvas[245:245 + overlay_big.shape[0], x + 20:x + 20 + overlay_big.shape[1]] = overlay_big

        trip = resize_contain(data["triptych"], card_w - 40, 200, bg=(245, 245, 245))
        canvas[690:690 + trip.shape[0], x + 20:x + 20 + trip.shape[1]] = trip

        badges = metric_triplet(data["eval"])
        for i, t in enumerate(badges):
            draw_badge(canvas, t, x + 20, 905 + i * 0, card_w - 40, 46, accent)
            break

        put_text(canvas, badges[0], x + 28, 938, scale=0.62, color=(15, 15, 20), thick=2)
        draw_badge(canvas, badges[1], x + 20, 955, card_w - 40, 46, accent)
        put_text(canvas, badges[1], x + 28, 988, scale=0.62, color=(15, 15, 20), thick=2)
        draw_badge(canvas, badges[2], x + 20, 1005, card_w - 40, 46, accent)
        put_text(canvas, badges[2], x + 28, 1038, scale=0.62, color=(15, 15, 20), thick=2)

    save_rgb(out_path, canvas)


def zoom_frame(img, t):
    h, w = img.shape[:2]
    s = 1.0 + 0.06 * t
    nw = int(w / s)
    nh = int(h / s)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    crop = img[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def make_reel(hero_paths, out_path, fps=12, seconds_each=4):
    frames = []
    n = int(fps * seconds_each)

    for name, hero_path in hero_paths:
        hero = load_rgb(hero_path)
        for i in range(n):
            t = i / max(1, n - 1)
            frame = zoom_frame(hero, t)
            frames.append(frame)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps)

    for f in frames:
        writer.append_data(f)

    writer.close()


def load_pack(render_dir, eval_path):
    render_dir = Path(render_dir)
    return {
        "overlay": load_rgb(render_dir / "overlay.png"),
        "triptych": load_rgb(render_dir / "triptych.png"),
        "eval": read_json(eval_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1-dir", required=True)
    ap.add_argument("--h1-eval", required=True)
    ap.add_argument("--h3-dir", required=True)
    ap.add_argument("--h3-eval", required=True)
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--h5-eval", required=True)
    ap.add_argument("--out-dir", default="runs/final_flash")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h1 = load_pack(args.h1_dir, args.h1_eval)
    h3 = load_pack(args.h3_dir, args.h3_eval)
    h5 = load_pack(args.h5_dir, args.h5_eval)

    make_single_hero(
        "Horizon 1",
        "Hybrid mask lifting + local TSDF scene + transformer forecaster",
        ACCENTS["H1"],
        h1["overlay"],
        h1["triptych"],
        h1["eval"],
        out_dir / "hero_h1.png",
    )
    make_single_hero(
        "Horizon 3",
        "Learned model starts to beat persistence more clearly",
        ACCENTS["H3"],
        h3["overlay"],
        h3["triptych"],
        h3["eval"],
        out_dir / "hero_h3.png",
    )
    make_single_hero(
        "Horizon 5",
        "Longer-horizon future graph prediction becomes the strongest story",
        ACCENTS["H5"],
        h5["overlay"],
        h5["triptych"],
        h5["eval"],
        out_dir / "hero_h5.png",
    )

    make_overview(h1, h3, h5, out_dir / "hero_overview.png")

    make_reel(
        [
            ("H1", out_dir / "hero_h1.png"),
            ("H3", out_dir / "hero_h3.png"),
            ("H5", out_dir / "hero_h5.png"),
        ],
        out_dir / "flash_reel.mp4",
        fps=12,
        seconds_each=4,
    )

    print("saved", out_dir / "hero_h1.png")
    print("saved", out_dir / "hero_h3.png")
    print("saved", out_dir / "hero_h5.png")
    print("saved", out_dir / "hero_overview.png")
    print("saved", out_dir / "flash_reel.mp4")


if __name__ == "__main__":
    main()
