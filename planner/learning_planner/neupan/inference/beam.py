"""Diverse beam decoding on the same observed support graph as greedy decoding."""
from dataclasses import dataclass

import numpy as np
import torch

from planner.learning_planner.neupan.models.support_policy import choose_actions, features, legal_moves
from planner.learning_planner.neupan.inference.reachability import GoalReachability
from planner.learning_planner.neupan.geometry.tomogram import STOP


@dataclass(frozen=True)
class BeamPath:
    nodes: tuple[int, ...]
    length: float = 0.
    log_probability: float = 0.
    reason: str = ''

    @property
    def score(self):
        return self.log_probability / max(1, len(self.nodes) - 1)


def _exclusive_run(a, b, separation):
    """Longest arc of a at least separation metres from the polyline b."""
    segments = np.diff(a, axis=0)
    length = np.linalg.norm(segments, axis=1)
    count = np.maximum(1, np.ceil(length / .025).astype(int))
    pieces = length.repeat(count) / count.repeat(count)
    if len(pieces) == 0:
        return 0.
    offsets = np.r_[0, np.cumsum(count[:-1])]
    fractions = (np.arange(len(pieces)) - offsets.repeat(count) + .5) / count.repeat(count)
    points = a[:-1].repeat(count, axis=0) + fractions[:, None] * segments.repeat(count, axis=0)
    tangent = np.diff(b, axis=0)
    if len(tangent) == 0:
        distance = np.linalg.norm(points - b[0], axis=1)
    else:
        delta = points[:, None] - b[:-1]
        denominator = (tangent ** 2).sum(axis=1).clip(min=1e-12)
        t = np.clip((delta * tangent).sum(axis=2) / denominator, 0., 1.)
        distance = np.linalg.norm(delta - t[:, :, None] * tangent, axis=2).min(axis=1)
    # Distance to a polyline is 1-Lipschitz. Checking the midpoint plus half
    # the subsegment length certifies the entire subsegment.
    separated = distance >= separation + pieces / 2.
    cuts = np.flatnonzero(~separated)
    ends = np.r_[0, np.cumsum(np.where(separated, pieces, 0.))]
    return float(np.diff(ends[np.r_[0, cuts + 1, len(pieces)]]).max())


DIVERSITY_SEPARATION = .25
DIVERSITY_ARC = .30


def exclusive_arcs(first, second, separation=DIVERSITY_SEPARATION):
    """Return the separated arc length on each path, using physical XYZ."""
    return _exclusive_run(first, second, separation), _exclusive_run(second, first, separation)


def _prune(states, xyz, width):
    # Retain high scoring prefixes from different spatial histories as they
    # approach the same landing. The final test uses real 3D distances.
    first_in_bin, overflow = {}, []
    for state in sorted(states, key=lambda s: (-s.score, s.nodes)):
        length = len(state.nodes) - 1
        sample = [state.nodes[index] for index in (length // 3, 2 * length // 3, length)]
        key = tuple(np.floor(xyz[sample].ravel() / .20).astype(int))
        if key in first_in_bin:
            overflow.append(state)
        else:
            first_in_bin[key] = state
    return (list(first_in_bin.values()) + overflow)[:width]


@torch.inference_mode()
def _search(model, graph, start, goal, horizon, max_steps, width):
    if start < 0:
        return []
    device = next(model.parameters()).device
    x = torch.as_tensor(features(graph, start, model.schema), device=device)
    edges = torch.as_tensor(graph.neighbors, dtype=torch.long, device=device)
    relative_goal = torch.as_tensor((goal - graph.xyz[start])[None], dtype=x.dtype, device=device)
    h = model.encode(x, edges)
    field = GoalReachability(graph, goal, horizon, max_steps)

    beam = [BeamPath((int(start),))]
    finished = []
    for step in range(max_steps + 1):
        pending, masks, lengths, protected = [], [], [], []
        for state in beam:
            here = state.nodes[-1]
            visited = np.isin(graph.neighbors[here], state.nodes)
            allowed, distances, protect, reason = legal_moves(
                graph, here, visited, goal, field, state.length, step, horizon, max_steps)
            if reason:
                finished.append(BeamPath(state.nodes, state.length, state.log_probability, reason))
            else:
                pending.append(state)
                masks.append(allowed)
                lengths.append(distances)
                protected.append(protect)
        if not pending:
            break
        current = torch.tensor([state.nodes[-1] for state in pending], device=device)
        scores = model.scores(h, x[:, :3], edges, current, relative_goal.expand(len(pending), -1),
                              torch.as_tensor(~np.array(masks), device=device))
        selected = choose_actions(scores, model.schema, model.stop_logit_threshold).tolist()
        log_prob = torch.log_softmax(scores[:, :STOP], dim=1).cpu().numpy()
        next_states = []
        for index, state in enumerate(pending):
            if selected[index] == STOP and not protected[index]:
                finished.append(BeamPath(state.nodes, state.length, state.log_probability, 'stop_selected'))
                continue
            for slot in np.flatnonzero(masks[index]):
                node = int(graph.neighbors[state.nodes[-1], slot])
                next_states.append(BeamPath(state.nodes + (node,), state.length + float(lengths[index][slot]),
                                            state.log_probability + float(log_prob[index, slot])))
        beam = _prune(next_states, graph.xyz, width)
        if not beam:
            break

    return sorted(finished, key=lambda s: (s.reason != 'goal_reached', -s.score, s.nodes))


def decode_beam(model, graph, start, goal, horizon=3., max_steps=64, *, count=3, width=64):
    if count < 1 or width < count:
        raise ValueError('Expected positive candidate count and beam width at least that count')
    chosen, geometries = [], []
    # A repeated layer copy of the same tread cannot count as another route.
    for state in _search(model, graph, start, goal, horizon, max_steps, width):
        if len(state.nodes) < 2:
            continue
        xyz = graph.xyz[list(state.nodes)]
        if all(min(exclusive_arcs(xyz, other)) >= DIVERSITY_ARC for other in geometries):
            chosen.append(state)
            geometries.append(xyz)
            if len(chosen) == count:
                break
    return chosen


def select_local_goals(model, graph, start, goal, *, lookahead=1.2, count=3, width=64,
                       separation=.30):
    """Return short-search terminal supports that are distinct as points."""
    if (not np.isfinite([lookahead, separation]).all() or lookahead <= 0 or
            separation <= 0 or count < 1 or width < count):
        raise ValueError('Expected positive local lookahead, separation and a sufficient beam width')
    selected = []
    for state in _search(model, graph, start, goal, lookahead, 64, width):
        if len(state.nodes) < 2:
            continue
        point = graph.xyz[state.nodes[-1]]
        if all(np.linalg.norm(point - graph.xyz[p.nodes[-1]]) >= separation for p in selected):
            selected.append(state)
            if len(selected) == count:
                break
    return selected
