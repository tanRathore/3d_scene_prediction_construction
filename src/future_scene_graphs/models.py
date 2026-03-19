import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPosEnc(nn.Module):
    def __init__(self, dim, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        t = x.shape[1]
        return x + self.pe[:, :t]

class SceneGraphForecaster(nn.Module):
    def __init__(self, hidden_dim=192, dropout=0.12, use_camera_motion=True, label_vocab=512, nhead=8, num_layers=3):
        super().__init__()
        self.use_camera_motion = use_camera_motion

        base_dim = 12
        cam_dim = 6 if use_camera_motion else 0
        in_dim = base_dim + cam_dim

        self.label_emb = nn.Embedding(label_vocab, 24)

        self.node_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal_pe = TemporalPosEnc(hidden_dim, max_len=128)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.graph_msg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_ln = nn.LayerNorm(hidden_dim)

        if use_camera_motion:
            self.cam_in = nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.cam_pe = TemporalPosEnc(hidden_dim, max_len=128)
            self.cam_temporal = nn.TransformerEncoder(enc_layer, num_layers=2)
            cond_dim = hidden_dim * 2
            head_in = hidden_dim * 3 + 24 + 11
        else:
            cond_dim = hidden_dim
            head_in = hidden_dim * 2 + 24 + 11

        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.delta_pos = nn.Linear(hidden_dim, 3)
        self.delta_size = nn.Linear(hidden_dim, 3)
        self.vis_head = nn.Linear(hidden_dim, 1)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.edge_bias = nn.Parameter(torch.tensor(1.2))

    def forward(self, batch):
        x = batch['x']
        adj = batch['adj']

        b, t, n, _ = x.shape

        raw = torch.cat([x[..., :11], x[..., 12:13]], dim=-1)
        if self.use_camera_motion:
            cam = batch['camera'].unsqueeze(2).expand(-1, -1, n, -1)
            raw = torch.cat([raw, cam], dim=-1)

        seq = raw.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        seq = self.node_in(seq.reshape(b * n * t, -1)).reshape(b * n, t, -1)
        seq = self.temporal_pe(seq)
        seq = self.temporal(seq)
        node_h = seq[:, -1].reshape(b, n, -1)

        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        neigh = torch.matmul(adj / deg, node_h)
        graph_h = self.graph_ln(node_h + self.graph_msg(neigh))

        last = x[:, -1]
        last_pos = last[..., 0:3]
        last_vel = last[..., 3:6]
        last_vis = last[..., 6:7]
        last_present = last[..., 7:8]
        last_size = last[..., 8:11]
        label_ids = last[..., 11].long().clamp(min=0, max=self.label_emb.num_embeddings - 1)

        scene_h = (graph_h * last_present).sum(dim=1) / last_present.sum(dim=1).clamp(min=1.0)

        if self.use_camera_motion:
            cam_seq = self.cam_in(batch['camera'].reshape(b * t, 6)).reshape(b, t, -1)
            cam_seq = self.cam_pe(cam_seq)
            cam_seq = self.cam_temporal(cam_seq)
            cam_h = cam_seq[:, -1]
            cond = torch.cat([scene_h, cam_h], dim=-1)
            graph_h = self.graph_ln(graph_h * (1.0 + self.gamma(cond).unsqueeze(1)) + self.beta(cond).unsqueeze(1))
            fused = torch.cat([
                graph_h,
                scene_h.unsqueeze(1).expand(-1, n, -1),
                cam_h.unsqueeze(1).expand(-1, n, -1),
                self.label_emb(label_ids),
                last_pos,
                last_vel,
                last_size,
                last_vis,
                last_present,
            ], dim=-1)
        else:
            cond = scene_h
            graph_h = self.graph_ln(graph_h * (1.0 + self.gamma(cond).unsqueeze(1)) + self.beta(cond).unsqueeze(1))
            fused = torch.cat([
                graph_h,
                scene_h.unsqueeze(1).expand(-1, n, -1),
                self.label_emb(label_ids),
                last_pos,
                last_vel,
                last_size,
                last_vis,
                last_present,
            ], dim=-1)

        z = self.head(fused)

        pos = last_pos + 0.35 * last_vel + self.delta_pos(z)
        size = torch.clamp(last_size * (1.0 + 0.15 * torch.tanh(self.delta_size(z))), min=0.05)
        vis_logits = self.vis_head(z).squeeze(-1)
        present_logits = self.presence_head(z).squeeze(-1)

        pairwise = torch.cdist(pos, pos)
        span = 0.25 * (size.norm(dim=-1).unsqueeze(-1) + size.norm(dim=-1).unsqueeze(-2))
        near_logits = self.edge_bias - pairwise + span

        eye = torch.eye(n, device=pos.device).unsqueeze(0)
        near_logits = near_logits.masked_fill(eye.bool(), -10.0)

        return {
            'pos': pos,
            'size': size,
            'vis_logits': vis_logits,
            'present_logits': present_logits,
            'near_logits': near_logits,
        }

def compute_losses(batch, out, vis_w, presence_w, edge_w, size_w=0.2):
    target_present = batch['y_present']
    pos_mask = target_present.unsqueeze(-1)

    pos_loss = ((((out['pos'] - batch['y_pos']) ** 2) * pos_mask).sum() / pos_mask.sum().clamp(min=1.0))

    size_loss = (
        (F.smooth_l1_loss(out['size'], batch['y_size'], reduction='none') * pos_mask).sum()
        / pos_mask.sum().clamp(min=1.0)
    )

    vis_loss = (
        F.binary_cross_entropy_with_logits(out['vis_logits'], batch['y_vis'], reduction='none') * target_present
    ).sum() / target_present.sum().clamp(min=1.0)

    presence_loss = F.binary_cross_entropy_with_logits(out['present_logits'], target_present)

    edge_mask = torch.matmul(target_present.unsqueeze(-1), target_present.unsqueeze(-2))
    edge_loss = (
        F.binary_cross_entropy_with_logits(out['near_logits'], batch['y_adj'], reduction='none') * edge_mask
    ).sum() / edge_mask.sum().clamp(min=1.0)

    loss = pos_loss + size_w * size_loss + vis_w * vis_loss + presence_w * presence_loss + edge_w * edge_loss

    return loss, {
        'loss': float(loss.detach().cpu()),
        'pos_loss': float(pos_loss.detach().cpu()),
        'size_loss': float(size_loss.detach().cpu()),
        'vis_loss': float(vis_loss.detach().cpu()),
        'presence_loss': float(presence_loss.detach().cpu()),
        'edge_loss': float(edge_loss.detach().cpu()),
    }
