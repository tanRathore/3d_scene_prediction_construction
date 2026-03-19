from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from .evaluate import move_batch

def _plot_edges(ax, pos, live, near_adj):
    n = pos.shape[0]
    for i in range(n):
        if not live[i]:
            continue
        for j in range(i + 1, n):
            if live[j] and near_adj[i, j]:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 2], pos[j, 2]], alpha=0.35)

def save_triptych(model, batch, out_path, device):
    batch = move_batch(batch, device)
    model.eval()
    with torch.no_grad():
        out = model(batch)

    last_pos = batch['x'][0, -1, :, 0:3].detach().cpu().numpy()
    last_vis = (batch['x'][0, -1, :, 6] > 0.5).detach().cpu().numpy()
    last_present = (batch['x'][0, -1, :, 7] > 0.5).detach().cpu().numpy()
    last_live = last_vis & last_present

    pred_pos = out['pos'][0].detach().cpu().numpy()
    pred_vis = (torch.sigmoid(out['vis_logits'][0]) > 0.5).detach().cpu().numpy()
    pred_present = (torch.sigmoid(out['present_logits'][0]) > 0.5).detach().cpu().numpy()
    pred_live = pred_vis & pred_present

    true_pos = batch['y_pos'][0].detach().cpu().numpy()
    true_vis = (batch['y_vis'][0] > 0.5).detach().cpu().numpy()
    true_present = (batch['y_present'][0] > 0.5).detach().cpu().numpy()
    true_live = true_vis & true_present

    pred_near = (torch.sigmoid(out['near_logits'][0]) > 0.5).detach().cpu().numpy()
    true_near = (batch['y_adj'][0] > 0.5).detach().cpu().numpy()
    hist_near = (batch['adj'][0] > 0.5).detach().cpu().numpy()

    track_ids = batch['track_ids'][0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        (axes[0], last_pos, last_live, hist_near, 'history'),
        (axes[1], pred_pos, pred_live, pred_near, 'pred'),
        (axes[2], true_pos, true_live, true_near, 'true'),
    ]

    for ax, pos, live, near_adj, title in panels:
        idx = np.where(live)[0]
        ax.scatter(pos[idx, 0], pos[idx, 2], s=45)

        for j in idx:
            if j < len(track_ids):
                tid = int(track_ids[j])
                if tid > 0:
                    ax.text(pos[j, 0], pos[j, 2], str(tid), fontsize=8)

        _plot_edges(ax, pos, live, near_adj)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
