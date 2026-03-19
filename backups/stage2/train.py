from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, Subset
from .config import ensure_dir
from .dataset import GraphWindowDataset
from .evaluate import evaluate_baseline, evaluate_model, move_batch
from .models import SceneGraphForecaster, compute_losses

def split_dataset(ds, val_ratio, seed):
    n = len(ds)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    v = max(1, int(n * val_ratio))
    val_idx = idx[:v]
    train_idx = idx[v:] or idx[:v]
    return Subset(ds, train_idx), Subset(ds, val_idx)

def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def run_training(cfg, window_path, out_dir):
    out_dir = ensure_dir(out_dir)
    train_cfg = cfg['train']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = GraphWindowDataset(window_path)
    tr_ds, va_ds = split_dataset(ds, train_cfg['val_ratio'], cfg['seed'])
    tr_loader = make_loader(tr_ds, train_cfg['batch_size'], True, train_cfg['num_workers'])
    va_loader = make_loader(va_ds, train_cfg['batch_size'], False, train_cfg['num_workers'])
    model = SceneGraphForecaster(train_cfg['hidden_dim'], train_cfg['dropout'], train_cfg['use_camera_motion']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    best = None
    rows = []
    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        losses = []
        for batch in tr_loader:
            batch = move_batch(batch, device)
            out = model(batch)
            loss, stats = compute_losses(batch, out, train_cfg['vis_loss_weight'], train_cfg['presence_loss_weight'], train_cfg['edge_loss_weight'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            opt.step()
            losses.append(stats)
        val_metrics = evaluate_model(model, va_loader, device)
        row = {'epoch': epoch, 'train_loss': float(sum(x['loss'] for x in losses) / max(1, len(losses))), **val_metrics}
        rows.append(row)
        score = row.get('l2', 1e9) - row.get('vis_f1', 0.0) - row.get('edge_f1', 0.0)
        if best is None or score < best['score']:
            best = {'score': score, 'epoch': epoch, 'metrics': row}
            torch.save({'model': model.state_dict(), 'config': cfg, 'metrics': row}, out_dir / 'best.pt')
        print(f"ep {epoch} loss {row['train_loss']:.4f} l2 {row.get('l2', 0):.4f} vis {row.get('vis_f1', 0):.3f} edge {row.get('edge_f1', 0):.3f}")
    base_rows = {'copy_last': evaluate_baseline('copy_last', va_loader, device), 'constant_velocity': evaluate_baseline('constant_velocity', va_loader, device)}
    with open(out_dir / 'train_log.json', 'w', encoding='utf-8') as f:
        json.dump({'epochs': rows, 'best': best, 'baselines': base_rows}, f, indent=2)
    print('saved', out_dir / 'best.pt')
