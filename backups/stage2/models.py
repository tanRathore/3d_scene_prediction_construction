import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import FEATURE_DIM

class SceneGraphForecaster(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1, use_camera_motion=True, label_vocab=512):
        super().__init__()
        self.use_camera_motion = use_camera_motion
        cam_dim = 6 if use_camera_motion else 0
        self.label_emb = nn.Embedding(label_vocab, 16)
        self.in_proj = nn.Sequential(nn.Linear(FEATURE_DIM - 1 + cam_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ctx_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2 + 16 + 8, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.delta_pos = nn.Linear(hidden_dim, 3)
        self.vis_head = nn.Linear(hidden_dim, 1)
        self.presence_head = nn.Linear(hidden_dim, 1)
    def forward(self, batch):
        x = batch['x']
        adj = batch['adj']
        b, t, n, f = x.shape
        inp = x[..., :12]
        label_ids = x[:, -1, :, 11].long().clamp(min=0, max=self.label_emb.num_embeddings - 1)
        if self.use_camera_motion:
            cam = batch['camera'].unsqueeze(2).expand(-1, -1, n, -1)
            inp = torch.cat([inp, cam], dim=-1)
        h0 = self.in_proj(inp.reshape(b * t * n, -1)).reshape(b * n, t, -1)
        _, h = self.gru(h0)
        node_h = h[-1].reshape(b, n, -1)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        neigh = torch.matmul(adj, node_h) / deg
        gate = self.ctx_gate(torch.cat([node_h, neigh], dim=-1))
        ctx = gate * neigh + (1.0 - gate) * node_h
        last = x[:, -1]
        last_pos = last[..., 0:3]
        last_vel = last[..., 3:6]
        last_vis = last[..., 6:7]
        last_present = last[..., 7:8]
        fused = torch.cat([node_h, ctx, self.label_emb(label_ids), last_pos, last_vel, last_vis, last_present], dim=-1)
        z = self.head(fused)
        pos = last_pos + self.delta_pos(z)
        vis_logits = self.vis_head(z).squeeze(-1)
        presence_logits = self.presence_head(z).squeeze(-1)
        near_logits = 1.2 - torch.cdist(pos, pos)
        eye = torch.eye(n, device=pos.device).unsqueeze(0)
        near_logits = near_logits.masked_fill(eye.bool(), -10.0)
        return {'pos': pos, 'vis_logits': vis_logits, 'present_logits': presence_logits, 'near_logits': near_logits}

def compute_losses(batch, out, vis_w, presence_w, edge_w):
    target_present = batch['y_present']
    pos_mask = target_present.unsqueeze(-1)
    pos_loss = ((((out['pos'] - batch['y_pos']) ** 2) * pos_mask).sum() / pos_mask.sum().clamp(min=1.0))
    vis_loss = (F.binary_cross_entropy_with_logits(out['vis_logits'], batch['y_vis'], reduction='none') * target_present).sum() / target_present.sum().clamp(min=1.0)
    presence_loss = F.binary_cross_entropy_with_logits(out['present_logits'], target_present)
    edge_mask = torch.matmul(target_present.unsqueeze(-1), target_present.unsqueeze(-2))
    edge_loss = (F.binary_cross_entropy_with_logits(out['near_logits'], batch['y_adj'], reduction='none') * edge_mask).sum() / edge_mask.sum().clamp(min=1.0)
    loss = pos_loss + vis_w * vis_loss + presence_w * presence_loss + edge_w * edge_loss
    return loss, {'loss': float(loss.detach().cpu()), 'pos_loss': float(pos_loss.detach().cpu()), 'vis_loss': float(vis_loss.detach().cpu()), 'presence_loss': float(presence_loss.detach().cpu()), 'edge_loss': float(edge_loss.detach().cpu())}
