import torch

def copy_last(batch):
    last = batch['x'][:, -1]
    return {'pos': last[..., 0:3], 'vis_logits': last[..., 6] * 2.0 - 1.0, 'present_logits': last[..., 7] * 2.0 - 1.0}

def constant_velocity(batch):
    x = batch['x']
    last = x[:, -1]
    prev = x[:, -2] if x.shape[1] > 1 else x[:, -1]
    vel = last[..., 0:3] - prev[..., 0:3]
    return {'pos': last[..., 0:3] + vel, 'vis_logits': last[..., 6] * 2.0 - 1.0, 'present_logits': last[..., 7] * 2.0 - 1.0}

BASELINES = {'copy_last': copy_last, 'constant_velocity': constant_velocity}
