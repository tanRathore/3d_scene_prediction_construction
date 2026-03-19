import numpy as np
import torch
from .baselines import BASELINES

def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

def reduce_rows(rows):
    keys = rows[0].keys() if rows else []
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}

def compute_batch_metrics(batch, out):
    target_present = batch['y_present'] > 0.5
    target_vis = batch['y_vis'] > 0.5
    pred_vis = torch.sigmoid(out['vis_logits']) > 0.5
    pred_present = torch.sigmoid(out['present_logits']) > 0.5
    l2 = torch.sqrt(((out['pos'] - batch['y_pos']) ** 2).sum(dim=-1) + 1e-9)
    l2 = (l2 * target_present).sum() / target_present.sum().clamp(min=1)
    tp = ((pred_vis & target_vis) & target_present).sum().item()
    fp = ((pred_vis & ~target_vis) & target_present).sum().item()
    fn = ((~pred_vis & target_vis) & target_present).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    near_pred = torch.sigmoid(out['near_logits']) > 0.5
    near_true = batch['y_adj'] > 0.5
    edge_mask = torch.matmul(target_present.float().unsqueeze(-1), target_present.float().unsqueeze(-2)) > 0.5
    ep = (near_pred & near_true & edge_mask).sum().item()
    efp = (near_pred & ~near_true & edge_mask).sum().item()
    efn = (~near_pred & near_true & edge_mask).sum().item()
    eprec = ep / (ep + efp + 1e-9)
    erec = ep / (ep + efn + 1e-9)
    ef1 = 2 * eprec * erec / (eprec + erec + 1e-9)
    pacc = (pred_present == target_present).float().mean().item()
    return {'l2': float(l2.item()), 'vis_f1': float(f1), 'edge_f1': float(ef1), 'present_acc': float(pacc)}

def evaluate_model(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            rows.append(compute_batch_metrics(batch, model(batch)))
    return reduce_rows(rows)

def evaluate_baseline(name, loader, device):
    fn = BASELINES[name]
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = fn(batch)
            dist = torch.cdist(out['pos'], out['pos'])
            near_logits = 1.2 - dist
            eye = torch.eye(out['pos'].shape[1], device=out['pos'].device).unsqueeze(0)
            out['near_logits'] = near_logits.masked_fill(eye.bool(), -10.0)
            rows.append(compute_batch_metrics(batch, out))
    return reduce_rows(rows)
