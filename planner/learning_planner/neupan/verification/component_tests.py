# NEUPAN 方法的独立命令行方案；请在 Pure3D_ws 根目录使用对应 Python 环境。
# Usage：python -m planner.learning_planner.neupan.verification.component_tests --help
"""Regression tests for the graph, native label indices and STOP semantics."""
import unittest

import numpy as np

from planner.learning_planner.neupan.data.expert import NativeExpert
from planner.learning_planner.neupan.data.supervision import expert_prefix, local_window
from planner.learning_planner.neupan.geometry.tomogram import STOP, SupportGraph, Tomogram


def plane(nx=10, ny=7):
    height = np.zeros((1, nx, ny), dtype=np.float32)
    return Tomogram(height.copy(), height, np.array([-4.3, -2.2]), .1)


class LocalSupportTest(unittest.TestCase):
    def test_native_xy_and_start(self):
        tomo = plane()
        first, last = np.array([0, 7, 2], dtype=np.int32), np.array([0, 2, 6], dtype=np.int32)
        chain = NativeExpert(tomo).route(first, last)
        np.testing.assert_array_equal(chain[0], first)
        np.testing.assert_array_equal(chain[-1], last)
        np.testing.assert_allclose(tomo.xyz([chain[-1]])[0], [-4.1, -1.6, 0], atol=1e-5)

    def test_no_diagonal_corner_and_exact_cost_gate(self):
        tomo = plane()
        tomo.cost[0, 2, 1] = 16.
        tomo.cost[0, 1, 2] = 15.
        graph = SupportGraph(tomo)
        start, wall, accept, diagonal = graph.lookup([[0, 1, 1], [0, 2, 1], [0, 1, 2], [0, 2, 2]])
        self.assertEqual(wall, -1)
        self.assertIn(accept, graph.neighbors[start])
        self.assertNotIn(diagonal, graph.neighbors[start])

    def test_observed_slices_never_average_overlaid_floors(self):
        cloud = np.array([[0., 0., 0., 0.], [.1, 0., 0., .8],
                          [0., 0., .85, .1], [.1, 0., .85, .1],
                          [.2, 0., .85, 0.]])
        tomo = Tomogram.observed(cloud, [0., 0., 0.])
        graph = SupportGraph(tomo)
        lower = graph.match_column([0., 0., 0.])
        upper = graph.match_column([0., 0., .85])
        self.assertGreaterEqual(lower, 0)
        self.assertGreaterEqual(upper, 0)
        self.assertEqual(graph.match_column([.1, 0., 0.]), -1)
        self.assertEqual(graph.match_column([.2, 0., 0.]), -1)
        self.assertTrue(np.all(graph.neighbors[lower] < 0))
        next_ids = graph.neighbors[upper]
        self.assertTrue(np.any(next_ids >= 0))
        np.testing.assert_allclose(graph.xyz[next_ids[next_ids >= 0], 2], .85)

    def test_relabel_block_with_observed_detour(self):
        tomo = plane()
        tomo.cost[0, 3, 3] = 50.
        graph = SupportGraph(tomo)
        route = np.array([[0, 2, 3], [0, 3, 3], [0, 4, 3], [0, 5, 3]])
        window = expert_prefix(graph, graph.lookup(route), tomo.xyz(route)[-1], 'test')
        self.assertEqual(window.end, 'goal_reached')
        self.assertGreater(len(window.chain), len(route))
        self.assertEqual(window.actions()[-1][2], STOP)
        self.assertTrue(np.any(graph.cells[window.chain, 2] != 3))

    def test_downhill_stop_and_recovery(self):
        tomo = plane(21, 7)
        tomo.ground[:, 7:10, :] = -.16
        tomo.ground[:, 10:, :] = -.32
        route = np.array([[0, i, 3] for i in range(5, 16)])
        goal = tomo.xyz(route)[-1]
        full = local_window(tomo, route, 0, goal, 'full')
        blocked = local_window(tomo, route, 0, goal, 'occluded')
        recovered = local_window(tomo, route, 0, goal, 'recovered')
        self.assertEqual(blocked.end, 'support_blocked')
        self.assertEqual(blocked.actions()[-1][2], STOP)
        self.assertEqual(full.end, 'goal_reached')
        self.assertGreater(len(full.chain), len(blocked.chain))
        self.assertGreater(len(recovered.chain), len(blocked.chain))

    def test_no_false_stop_label_at_horizon(self):
        tomo = plane(48, 7)
        route = np.array([[0, i, 3] for i in range(3, 48)])
        window = local_window(tomo, route, 0, tomo.xyz(route)[-1], 'full')
        self.assertEqual(window.end, 'local_horizon')
        self.assertTrue(all(action[2] != STOP for action in window.actions()))


class PathComponentTest(unittest.TestCase):
    def test_guard_ignores_premature_stop_only_when_reachable(self):
        import torch

        from planner.learning_planner.neupan.models.support_policy import SupportPolicy, decode

        model = SupportPolicy()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.stop_score[-1].bias.fill_(2.)
        tomo = plane(10, 5)
        graph = SupportGraph(tomo)
        start, goal = graph.lookup([[0, 0, 2], [0, 9, 2]])
        chain, reason = decode(model, graph, start, graph.xyz[goal], guard=False)
        self.assertEqual((len(chain), reason), (1, 'stop_selected'))
        chain, reason = decode(model, graph, start, graph.xyz[goal])
        self.assertEqual(reason, 'goal_reached')
        self.assertTrue(all(b in graph.neighbors[a] for a, b in zip(chain, chain[1:])))

        tomo.ground[:, 5, :] = np.nan
        graph = SupportGraph(tomo)
        start, goal = graph.lookup([[0, 0, 2], [0, 9, 2]])
        chain, reason = decode(model, graph, start, graph.xyz[goal])
        self.assertEqual((len(chain), reason), (1, 'stop_selected'))
        self.assertNotIn(goal, chain)

    def test_guard_does_not_cross_an_unconnected_floor(self):
        from planner.learning_planner.neupan.inference.reachability import GoalReachability

        tomo = plane()
        stacked = Tomogram(np.stack([tomo.cost[0]] * 2),
                            np.stack([tomo.ground[0], tomo.ground[0] + .8]),
                            tomo.origin, tomo.resolution)
        graph = SupportGraph(stacked)
        lower, upper = graph.lookup([[0, 3, 3], [1, 3, 3]])
        field = GoalReachability(graph, graph.xyz[upper], 3., 64)
        self.assertTrue(np.isinf(field.distance[lower]))
        self.assertEqual(field.distance[upper], 0.)

    def test_disconnected_downhill_stops_at_last_forward_support(self):
        import torch

        from planner.learning_planner.neupan.models.support_policy import SupportPolicy, decode

        model = SupportPolicy()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.stop_score[-1].bias.fill_(-9.)
        tomo = plane(12, 9)
        tomo.ground[:, 5:7, :] = np.nan
        tomo.ground[:, 7:, :] = -.15
        graph = SupportGraph(tomo)
        start = graph.lookup([[0, 1, 4]])[0]
        goal = tomo.xyz([[0, 10, 4]])[0]
        chain, reason = decode(model, graph, start, goal)
        self.assertEqual(reason, 'support_blocked')
        self.assertEqual(graph.cells[chain[-1], 1], 4)
        self.assertTrue(np.all(graph.cells[chain, 1] < 5))
        self.assertTrue(all(b in graph.neighbors[a] for a, b in zip(chain, chain[1:])))

    def test_component_body_offset_and_unsupported_prefix(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import torch

        from planner.learning_planner.neupan.models.support_policy import SupportPolicy
        from planner.learning_planner.neupan.inference.path_generator import PathGenerator

        torch.set_num_threads(1)
        model = SupportPolicy(schema=1)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'test.pt'
            torch.save(dict(schema=1, model=model.state_dict()), checkpoint)
            planner = PathGenerator(checkpoint)
            # A one-cell-wide observed corridor, with no alternate successors.
            cloud = np.array([[i * .1, 0., 0., .1] for i in range(7)])
            result = planner.plan(cloud, [0., 0., 0.], [.6, 0., 0.], cost_version='first')
            self.assertEqual(result.reason, 'goal_reached')
            self.assertEqual(result.cost_version, 'first')
            np.testing.assert_allclose(result.body_path()[:, :2], result.support_xyz[:, :2])
            np.testing.assert_allclose(result.body_path()[:, 2], result.support_xyz[:, 2] + .4)
            blocked = planner.plan(cloud[[0, 3, 4, 5, 6]], [0., 0., 0.], [.6, 0., 0.])
            self.assertEqual(blocked.reason, 'support_blocked')
            self.assertEqual(blocked.body_path().shape, (0, 3))


class DiverseBeamTest(unittest.TestCase):
    def test_short_differences_and_layer_copies_do_not_count(self):
        from planner.learning_planner.neupan.inference.beam import exclusive_arcs

        track = np.c_[np.linspace(0., 2., 21), np.zeros((21, 2))]
        close = track.copy()
        close[:, 1] += .1
        self.assertEqual(exclusive_arcs(track, track.copy()), (0., 0.))
        self.assertEqual(exclusive_arcs(track, close), (0., 0.))

    def test_beam_uses_goal_guard_and_returns_one_in_single_corridor(self):
        import torch

        from planner.learning_planner.neupan.inference.beam import decode_beam
        from planner.learning_planner.neupan.models.support_policy import SupportPolicy

        model = SupportPolicy()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.stop_score[-1].bias.fill_(2.)
        tomo = plane(22, 3)
        tomo.cost[:, :, (0, 2)] = 50.
        graph = SupportGraph(tomo)
        start, goal = graph.lookup([[0, 0, 1], [0, 21, 1]])
        routes = decode_beam(model, graph, start, graph.xyz[goal])
        self.assertEqual(len(routes), 1)
        route = routes[0]
        self.assertEqual(route.reason, 'goal_reached')
        self.assertLessEqual(route.length, 3.)
        self.assertTrue(all(b in graph.neighbors[a] for a, b in zip(route.nodes, route.nodes[1:])))

        tomo.ground[:, 10, :] = np.nan
        graph = SupportGraph(tomo)
        start, goal = graph.lookup([[0, 0, 1], [0, 21, 1]])
        self.assertEqual(decode_beam(model, graph, start, graph.xyz[goal]), [])


    def test_short_goals_do_not_cross_a_gap(self):
        import torch

        from planner.learning_planner.neupan.inference.beam import select_local_goals
        from planner.learning_planner.neupan.models.support_policy import SupportPolicy

        model = SupportPolicy()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.stop_score[-1].bias.fill_(-2.)
        tomo = plane(25, 5)
        tomo.ground[:, 8:10, :] = np.nan
        graph = SupportGraph(tomo)
        start, goal = graph.lookup([[0, 0, 2], [0, 22, 2]])
        routes = select_local_goals(model, graph, start, graph.xyz[goal], lookahead=1.2)
        self.assertTrue(routes)
        for path in routes:
            self.assertLessEqual(path.length, 1.2)
            self.assertLess(graph.cells[path.nodes[-1], 1], 8)
            self.assertTrue(all(b in graph.neighbors[a] for a, b in zip(path.nodes, path.nodes[1:])))


class LandingAreaTest(unittest.TestCase):
    def test_tread_center_uses_geometric_coverage(self):
        from planner.learning_planner.neupan.geometry.landing import support_patches
        from planner.learning_planner.neupan.data.landing_gpmp import checkpoints
        from planner.learning_planner.neupan.models.landing_model import center_on_support

        tomo = plane(12, 7)
        tomo.ground[0, 3:6, :] = .15
        tomo.ground[0, 6:, :] = .30
        tomo.cost[0, 5, :] = 30.
        edge, center, upper = tomo.xyz([[0, 3, 3], [0, 4, 3], [0, 6, 3]])
        points = np.array([edge, center, upper])
        patch = support_patches(tomo, points)
        self.assertGreater(patch.support_area[1], patch.area[1])
        self.assertGreater(patch.support_area[1], patch.support_area[0])
        self.assertLess(patch.support_area[0], np.pi * .15 ** 2)
        rank = np.array([3., 2., 1.])
        self.assertEqual(center_on_support(tomo, points, patch.support_area, rank, np.zeros(3, bool), 1), [1])
        self.assertEqual(center_on_support(tomo, points, patch.support_area, rank, np.array([1, 0, 0], bool), 1), [0])

        nominal = tomo.xyz([[0, 3, 3], [0, 4, 3], [0, 5, 3], [0, 6, 3]])
        result = checkpoints(tomo, (nominal, np.zeros(4, dtype=int), np.arange(4) * .10))
        self.assertEqual(result[0]['reason'], 'supported')
        np.testing.assert_allclose(result[0]['support_xyz'], center)

    def test_route_integrates_cost_over_supported_edges(self):
        from planner.learning_planner.neupan.inference.landing_route import LandingRoutes, EDGE_COST_WEIGHT

        tomo = plane(15, 9)
        tomo.prediction = True
        start, goal = tomo.xyz([[0, 2, 3], [0, 10, 3]])
        tomo.cost[0, 4:9, 3] = .74
        costs = LandingRoutes(tomo).costs(start, goal, goal[None])
        path = costs.approach_path(0)
        self.assertTrue(costs.observed_goal)
        self.assertAlmostEqual(costs.remaining_cost[0], 0.)
        self.assertGreater(costs.approach_m[0], .8)
        self.assertLess(costs.approach_m[0], 1.2)
        self.assertLess(costs.approach_cost[0], .8 + EDGE_COST_WEIGHT * .74 * .5)
        self.assertTrue(np.any(abs(path[:, 1] - start[1]) > .05))

    def test_remaining_cost_decreases_on_detour_even_when_xy_moves_away(self):
        from planner.learning_planner.neupan.inference.landing_route import LandingRoutes

        tomo = plane(17, 16)
        tomo.prediction = True
        tomo.ground[:, 7, :13] = np.nan
        start, onward, goal = tomo.xyz([[0, 4, 4], [0, 4, 10], [0, 11, 4]])
        costs = LandingRoutes(tomo).costs(start, goal, np.array([onward]))
        self.assertTrue(costs.observed_goal)
        self.assertGreater(np.linalg.norm(onward[:2] - goal[:2]), np.linalg.norm(start[:2] - goal[:2]))
        self.assertGreater(costs.progress[0], .3)
        self.assertLess(costs.approach_m[0], 1.2)

    def test_approach_accepts_supported_detour_and_rejects_gap_and_other_layer(self):
        from planner.learning_planner.neupan.data.landing_approach import approach_distances

        flat = plane(15, 7)
        topo = Tomogram(np.repeat(flat.cost, 2, axis=0),
                        np.concatenate((flat.ground, flat.ground + 1.)),
                        flat.origin, flat.resolution, True)
        start, before, beyond, upper = topo.xyz([[0, 2, 2], [0, 3, 2], [0, 7, 2], [1, 7, 2]])
        topo.cost[0, 5, :5] = .8
        d = approach_distances(topo, start, np.array([before, beyond, upper]))
        self.assertAlmostEqual(d[0], .1, places=5)
        self.assertGreater(d[1], .5)
        self.assertLess(d[1], 1.2)
        self.assertFalse(np.isfinite(d[2]))
        topo.ground[0, 5, 5:] = np.nan
        d = approach_distances(topo, start, np.array([before, beyond]))
        self.assertTrue(np.isfinite(d[0]))
        self.assertFalse(np.isfinite(d[1]))

    def test_coverage_scorer_prefers_before_gap_then_stops_without_start_support(self):
        import tempfile
        from pathlib import Path
        import torch

        from planner.learning_planner.neupan.models.landing_model import LandingModel, LandingScorer

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'always_high.pt'
            model = LandingModel(coverage_head=True)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.score[-1].bias.fill_(8.)
            torch.save(dict(type='landing_goal_v3', model=model.state_dict(), threshold=.5), checkpoint)
            scorer = LandingScorer(checkpoint)
            tomo = plane(18, 7)
            tomo.prediction = True
            tomo.ground[:, 6:8, :] = np.nan
            start, goal = tomo.xyz([[0, 2, 3], [0, 12, 3]])
            candidates, reason = scorer.candidates(tomo, start, goal)
            self.assertEqual(reason, 'candidates_ready')
            self.assertTrue(all(c.xyz[0] < tomo.origin[0] + .6 and c.approach_m <= 1.2 for c in candidates))
            tomo.ground[:, 2, 3] = np.nan
            candidates, reason = scorer.candidates(tomo, start, goal)
            self.assertEqual(candidates, [])
            self.assertEqual(reason, 'support_blocked')

    def test_disk_area_clipping_and_duplicate_layers(self):
        from planner.learning_planner.neupan.geometry.landing import support_patches

        tomo = plane(13, 13)
        xyz = tomo.xyz([[0, 5, 5]]) + [.022, -.017, 0.]
        flat = support_patches(tomo, xyz)
        stacked = Tomogram(np.repeat(tomo.cost, 2, axis=0), np.repeat(tomo.ground, 2, axis=0),
                           tomo.origin, tomo.resolution)
        self.assertAlmostEqual(float(support_patches(stacked, xyz).area[0]), np.pi * .15 ** 2, places=6)
        self.assertAlmostEqual(float(flat.area[0]), np.pi * .15 ** 2, places=6)

    def test_observation_gap_keeps_landing_and_goal_oracle_labels(self):
        from scipy.sparse.csgraph import dijkstra

        from planner.learning_planner.neupan.geometry.landing import support_patches
        from planner.learning_planner.neupan.data.landing_dataset import GoalOracle

        gt = plane(18, 7)
        start, candidate, goal = gt.xyz([[0, 1, 3], [0, 9, 3], [0, 15, 3]])
        oracle = GoalOracle(gt, gt.origin - .01, gt.origin + [1.9, .8])
        observed = plane(18, 7)
        observed.prediction = True
        gap = Tomogram(observed.cost.copy(), observed.ground.copy(), observed.origin, observed.resolution, True)
        gap.ground[:, 4:6, :] = np.nan
        a = support_patches(observed, [candidate])
        b = support_patches(gap, [candidate])
        self.assertAlmostEqual(float(a.area[0]), float(b.area[0]))
        self.assertGreater(float(b.area[0]), .04)
        incomplete = GoalOracle(gap, gt.origin - .01, gt.origin + [1.9, .8])
        source, target = incomplete.matches(np.array([start, candidate]))
        self.assertFalse(np.isfinite(dijkstra(incomplete.adjacency, indices=source)[target]))
        value, known, outgoing, incoming = oracle.labels(start, goal, np.array([candidate]), a.area)
        self.assertTrue(known[0] and outgoing[0] and incoming[0])
        self.assertGreater(float(value[0]), .5)

        # If the center itself is missing, adjacent good support must not count.
        gap.ground[:, 9, 3] = np.nan
        missing = support_patches(gap, [candidate])
        self.assertFalse(missing.center_observed[0])
        self.assertEqual(missing.area[0], 0.)


class SequenceTest(unittest.TestCase):
    def test_narrow_landing_is_legal_and_returns_continuous_segments(self):
        import tempfile
        from pathlib import Path
        import torch

        from planner.learning_planner.neupan.data.sequence_expert import Teacher
        from planner.learning_planner.neupan.models.support_sequence import CHECKPOINT_TYPE, SequenceGraph, SequenceNetwork, SequencePlanner

        tomo = plane(18, 7)
        tomo.ground[:, :, :3] = np.nan
        tomo.ground[:, :, 4:] = np.nan
        start, goal = tomo.xyz([[0, 2, 3], [0, 12, 3]])
        graph = SequenceGraph(tomo, goal)
        source = graph.source(start)
        teacher = Teacher(graph, graph, goal)
        label = teacher.label(source, visited=[start])
        self.assertTrue(label.positive.any())
        self.assertTrue(np.all(label.area[label.positive] < .04))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sequence.pt'
            model = SequenceNetwork()
            for p in model.parameters():
                p.data.zero_()
            torch.save(dict(type=CHECKPOINT_TYPE, model=model.state_dict(),
                            connect_threshold=.5, stop_threshold=.5), path)
            planner = SequencePlanner(path)
            result = planner.plan(tomo, start, goal)
            self.assertGreater(len(result.points), 0)
            self.assertLessEqual(len(result.points), 3)
            self.assertLessEqual(sum(p.approach_m for p in result.points), 3.00001)
            for p in result.points:
                self.assertLess(p.coverage_m2, .04)
                np.testing.assert_array_equal(p.xyz, p.approach_xyz[-1])
                nodes = graph.routes.point_ids(p.approach_xyz)
                self.assertTrue(np.isfinite(teacher.chain_cost(nodes)))

    def test_gate_and_teacher_respect_gap_and_height(self):
        from planner.learning_planner.neupan.data.sequence_expert import Teacher
        from planner.learning_planner.neupan.models.support_sequence import SequenceGraph

        clean = plane(16, 8)
        start, goal = clean.xyz([[0, 2, 4], [0, 13, 4]])
        tiled = Tomogram(np.repeat(clean.cost, 2, axis=0), np.r_[clean.ground, clean.ground + .8],
                         clean.origin, clean.resolution, True)
        full = SequenceGraph(tiled, goal)
        cloud = Tomogram(tiled.cost.copy(), tiled.ground.copy(), tiled.origin, tiled.resolution, True)
        cloud.ground[0, 6:8] = np.nan
        split = SequenceGraph(cloud, goal)
        teacher = Teacher(full, split, goal)
        labels = teacher.label(split.source(start), visited=[start])
        self.assertTrue(labels.positive.any())
        points = split.xyz[labels.actions.ids]
        self.assertTrue(np.all(points[:, 0] < clean.origin[0] + .6))
        self.assertTrue(np.all(points[:, 2] == 0))

    def test_teacher_checks_actual_approach_even_with_an_alternative_gt_path(self):
        from planner.learning_planner.neupan.data.sequence_expert import Teacher
        from planner.learning_planner.neupan.models.support_sequence import SequenceGraph

        observed = plane(16, 8)
        gt = plane(16, 8)
        gt.cost[0, 6, 4] = 30.
        start, goal = gt.xyz([[0, 2, 4], [0, 12, 4]])
        clean, local = SequenceGraph(gt, goal), SequenceGraph(observed, goal)
        self.assertTrue(np.isfinite(clean.remaining[clean.source(start)]))
        label = Teacher(clean, local, goal).label(local.source(start), visited=[start])
        index = int(np.flatnonzero(label.actions.ids == local.source(goal))[0])
        self.assertFalse(label.positive[index])
        self.assertTrue(np.isinf(label.costs[index]))


if __name__ == '__main__':
    unittest.main()
