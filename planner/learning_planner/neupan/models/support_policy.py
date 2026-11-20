"""Trainable node encoder and masked neighbor/STOP action heads."""
import numpy as np
import torch
from torch import nn

from planner.learning_planner.neupan.inference.reachability import GoalReachability, frontier_progress, goal_mask
from planner.learning_planner.neupan.geometry.tomogram import STOP, PREDICTION_LIMIT, VST_LIMIT


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


def neighbor_mean(features, neighbors, valid):
    gathered = features[neighbors.clamp_min(0)] * valid.unsqueeze(-1)
    return gathered.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)


class SupportPolicy(nn.Module):
    def __init__(self, width=64, schema=2):
        super().__init__()
        self.schema = schema
        self.stop_logit_threshold = 0.
        self.node = mlp(4, width, width)
        self.messages = nn.ModuleList(mlp(2 * width, width, width) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(2))
        self.edge_score = mlp(2 * width + (8 if schema == 2 else 6), width, 1)
        self.stop_score = mlp(2 * width + 4, width, 1)

    def encode(self, features, neighbors):
        h = self.node(features)
        valid = neighbors >= 0
        for update, norm in zip(self.messages, self.norms):
            local = neighbor_mean(h, neighbors, valid)
            h = norm(h + update(torch.cat((h, local), dim=-1)))
        return h

    def scores(self, h, xyz, neighbors, current, goal, visited):
        next_ids = neighbors[current]
        valid = (next_ids >= 0) & ~visited
        safe = next_ids.clamp_min(0)
        source_xyz = xyz[current]
        target_xyz = xyz[safe]
        source = h[current]
        target = h[safe]
        edge = torch.cat((source[:, None].expand_as(target), target,
                          target_xyz - source_xyz[:, None], goal[:, None] - target_xyz), dim=-1)
        if self.schema == 2:
            remaining_from_source = goal - source_xyz
            remaining_from_target = goal[:, None] - target_xyz
            progress_xy = (remaining_from_source[:, None, :2].norm(dim=-1)
                           - remaining_from_target[:, :, :2].norm(dim=-1))
            progress_z = remaining_from_source[:, None, 2].abs() - remaining_from_target[:, :, 2].abs()
            edge = torch.cat((edge, progress_xy.unsqueeze(-1), progress_z.unsqueeze(-1)), dim=-1)
        neighbor_scores = self.edge_score(edge).squeeze(-1).masked_fill(~valid, -torch.inf)
        context = neighbor_mean(h, next_ids, valid)
        available = valid.float().mean(dim=1, keepdim=True)
        stop = self.stop_score(torch.cat((source, context, goal - source_xyz, available), dim=-1))
        return torch.cat((neighbor_scores, stop), dim=-1)


def features(graph, start, schema):
    scale = 1.
    if schema == 2:
        scale = PREDICTION_LIMIT if graph.tomo.prediction else VST_LIMIT / 50.
    return np.column_stack((graph.xyz - graph.xyz[start], graph.cost / scale)).astype(np.float32)


def choose_actions(scores, schema, threshold=0.):
    if schema == 1:
        return scores.argmax(dim=1)
    neighbors, stop = scores[:, :STOP], scores[:, STOP]
    stop_now = (stop >= threshold) | ~neighbors.isfinite().any(dim=1)
    return torch.where(stop_now, STOP, neighbors.argmax(dim=1))


def batch(windows, device, schema):
    all_features, edges, starts, goals, visited, targets = [], [], [], [], [], []
    base = 0
    for window in windows:
        actions = window.actions()
        if not actions:
            continue
        graph = window.graph
        current = np.array([node for node, _, _ in actions])
        # Scores touch each current node and its neighbors. The two encoder
        # layers add two hops. An induced three-hop graph is exact for these logits.
        local = np.unique(current)
        for _ in range(3):
            boundary = graph.neighbors[local].ravel()
            local = np.union1d(local, boundary[boundary >= 0])
        lookup = np.full(len(graph.xyz), -1, dtype=np.int64)
        lookup[local] = np.arange(len(local)) + base
        all_features.append(features(graph, window.chain[0], schema)[local])
        links = graph.neighbors[local]
        edges.append(np.where(links >= 0, lookup[links.clip(min=0)], -1))
        starts.extend(lookup[current])
        goals.extend([window.goal - graph.xyz[window.chain[0]]] * len(actions))
        visited.extend(mask for _, mask, _ in actions)
        targets.extend(action for _, _, action in actions)
        base += len(local)
    if not targets:
        return None
    return tuple(torch.as_tensor(np.asarray(values), dtype=dtype, device=device) for values, dtype in (
        (np.concatenate(all_features), torch.float32), (np.concatenate(edges), torch.long),
        (starts, torch.long), (goals, torch.float32), (visited, torch.bool), (targets, torch.long)))


def legal_moves(graph, here, visited, goal, field, travelled, step, horizon, max_steps):
    """Return the eligible neighbor mask, horizontal lengths, goal guard and terminal reason."""
    if goal_mask(graph.xyz[here], goal):
        return None, None, False, 'goal_reached'
    if step == max_steps or travelled >= horizon - 1e-6:
        return None, None, False, 'local_horizon'
    candidates = graph.neighbors[here]
    safe = np.maximum(candidates, 0)
    covered = (candidates >= 0) & ~visited
    if not covered.any():
        return None, None, False, 'support_blocked'
    lengths = np.linalg.norm(graph.xyz[safe, :2] - graph.xyz[here, :2], axis=1)
    legal = covered & (travelled + lengths <= horizon + 1e-6)
    if not legal.any():
        return None, None, False, 'local_horizon'
    protect_goal = False
    if field is not None:
        remaining = horizon - travelled
        preserve = field.distances(max_steps - step)
        next_dist = field.distances(max_steps - step - 1)
        if preserve is not None and np.isfinite(preserve[here]):
            protect_goal = True
            legal &= next_dist[safe] < preserve[here] - 1e-6
            if preserve[here] <= remaining + 1e-5:
                legal &= lengths + next_dist[safe] <= remaining + 1e-5
            if not legal.any():
                return None, None, False, 'local_horizon'
        elif field.distance is not None and np.isfinite(field.distance[here]):
            # The goal is connected but exceeds the caller's step budget.
            protect_goal = True
            legal &= field.distance[safe] < field.distance[here] - 1e-6
            if not legal.any():
                return None, None, False, 'local_horizon'
        else:
            legal &= frontier_progress(graph.xyz[here], graph.xyz[safe], goal)
            if not legal.any():
                return None, None, False, 'support_blocked'
    return legal, lengths, protect_goal, None


@torch.inference_mode()
def decode(model, graph, start, goal, horizon=3., max_steps=64, *, guard=True):
    if start < 0:
        return [], 'no_start_support'
    device = next(model.parameters()).device
    x = torch.as_tensor(features(graph, start, model.schema), device=device)
    edges = torch.as_tensor(graph.neighbors, dtype=torch.long, device=device)
    relative_goal = torch.as_tensor((goal - graph.xyz[start])[None], dtype=x.dtype, device=device)
    h = model.encode(x, edges)
    chain = [start]
    seen = np.zeros(len(graph.cells), dtype=bool)
    travelled = 0.
    field = GoalReachability(graph, goal, horizon, max_steps) if guard else None
    for step in range(max_steps + 1):
        here = chain[-1]
        seen[here] = True
        candidates = graph.neighbors[here]
        legal, lengths, protect_goal, reason = legal_moves(
            graph, here, seen[np.maximum(candidates, 0)], goal, field, travelled, step, horizon, max_steps)
        if reason:
            return chain, reason
        scores = model.scores(h, x[:, :3], edges, torch.tensor([here], device=device),
                              relative_goal, torch.as_tensor(~legal[None], device=device))
        if protect_goal:
            action = int(scores[0, :STOP].argmax())
        else:
            action = int(choose_actions(scores, model.schema, model.stop_logit_threshold)[0])
        if action == STOP:
            return chain, 'stop_selected'
        travelled += float(lengths[action])
        chain.append(int(candidates[action]))
    raise AssertionError('Decode exceeded its step limit')
