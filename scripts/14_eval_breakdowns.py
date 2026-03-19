from pathlib import Path
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from future_scene_graphs.config import load_config
from future_scene_graphs.dataset import GraphWindowDataset
from future_scene_graphs.evaluate import move_batch
from future_scene_graphs.models import SceneGraphForecaster
from future_scene_graphs.train import split_dataset
from future_scene_graphs.baselines import BASELINES


class Meter:
    def __init__(self):
        self.n = 0
        self.l2_sum = 0.0
        self.vis_correct = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, l2, pred_vis, true_vis):
        l2 = np.asarray(l2).reshape(-1)
        pred_vis = np.asarray(pred_vis).reshape(-1).astype(bool)
        true_vis = np.asarray(true_vis).reshape(-1).astype(bool)

        if l2.size == 0:
            return

        self.n += int(l2.size)
        self.l2_sum += float(l2.sum())
        self.vis_correct += int((pred_vis == true_vis).sum())
        self.tp += int(np.logical_and(pred_vis, true_vis).sum())
        self.fp += int(np.logical_and(pred_vis, ~true_vis).sum())
        self.fn += int(np.logical_and(~pred_vis, true_vis).sum())

    def as_dict(self):
        if self.n == 0:
            return {
                "count": 0,
                "l2": None,
                "vis_acc": None,
                "vis_f1": None,
            }

        prec = self.tp / (self.tp + self.fp + 1e-9)
        rec = self.tp / (self.tp + self.fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        return {
            "count": int(self.n),
            "l2": float(self.l2_sum / max(self.n, 1)),
            "vis_acc": float(self.vis_correct / max(self.n, 1)),
            "vis_f1": float(f1),
        }


def motion_bucket_name(d):
    if d < 0.05:
        return "static_lt_5cm"
    if d < 0.15:
        return "small_5_to_15cm"
    if d < 0.30:
        return "medium_15_to_30cm"
    return "large_gt_30cm"


def vis_transition_name(last_vis, true_vis):
    a = "vis" if last_vis else "hid"
    b = "vis" if true_vis else "hid"
    return f"{a}_to_{b}"


def get_outputs(method_name, model, batch):
    if method_name == "model":
        with torch.no_grad():
            return model(batch)

    fn = BASELINES[method_name]
    with torch.no_grad():
        return fn(batch)


def evaluate_breakdowns(method_name, model, loader, device):
    overall = Meter()
    static_vs_moving = {
        "static": Meter(),
        "moving": Meter(),
    }
    motion_buckets = defaultdict(Meter)
    vis_transitions = defaultdict(Meter)

    for batch in loader:
        batch = move_batch(batch, device)
        out = get_outputs(method_name, model, batch)

        target_present = (batch["y_present"] > 0.5).detach().cpu().numpy()
        target_vis = (batch["y_vis"] > 0.5).detach().cpu().numpy()
        pred_vis = (torch.sigmoid(out["vis_logits"]) > 0.5).detach().cpu().numpy()

        last_pos = batch["x"][:, -1, :, 0:3].detach().cpu().numpy()
        last_vis = (batch["x"][:, -1, :, 6] > 0.5).detach().cpu().numpy()
        true_pos = batch["y_pos"].detach().cpu().numpy()
        pred_pos = out["pos"].detach().cpu().numpy()

        l2 = np.linalg.norm(pred_pos - true_pos, axis=-1)
        disp = np.linalg.norm(true_pos - last_pos, axis=-1)

        B, N = target_present.shape
        for b in range(B):
            for j in range(N):
                if not target_present[b, j]:
                    continue

                d = float(disp[b, j])
                yv = bool(target_vis[b, j])
                pv = bool(pred_vis[b, j])
                lv = bool(last_vis[b, j])
                l2_ij = float(l2[b, j])

                overall.update([l2_ij], [pv], [yv])

                if d < 0.05:
                    static_vs_moving["static"].update([l2_ij], [pv], [yv])
                else:
                    static_vs_moving["moving"].update([l2_ij], [pv], [yv])

                motion_buckets[motion_bucket_name(d)].update([l2_ij], [pv], [yv])
                vis_transitions[vis_transition_name(lv, yv)].update([l2_ij], [pv], [yv])

    return {
        "overall": overall.as_dict(),
        "static_vs_moving": {k: v.as_dict() for k, v in static_vs_moving.items()},
        "motion_buckets": {k: motion_buckets[k].as_dict() for k in [
            "static_lt_5cm",
            "small_5_to_15cm",
            "medium_15_to_30cm",
            "large_gt_30cm",
        ]},
        "visibility_transitions": {k: vis_transitions[k].as_dict() for k in [
            "vis_to_vis",
            "vis_to_hid",
            "hid_to_vis",
            "hid_to_hid",
        ]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = GraphWindowDataset(args.windows)
    _, va_ds = split_dataset(ds, cfg["train"]["val_ratio"], cfg["seed"])
    loader = DataLoader(va_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SceneGraphForecaster(
        cfg["train"]["hidden_dim"],
        cfg["train"]["dropout"],
        cfg["train"]["use_camera_motion"],
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    rows = {
        "model": evaluate_breakdowns("model", model, loader, device),
        "copy_last": evaluate_breakdowns("copy_last", model, loader, device),
        "constant_velocity": evaluate_breakdowns("constant_velocity", model, loader, device),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
