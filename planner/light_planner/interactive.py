"""通过 RViz 双 InteractiveMarker 设置三维点云规划的起终点。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time

import numpy as np
from geometry_msgs.msg import Point, Pose
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from std_msgs.msg import String
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker

from .config import PlannerConfig
from .io import save_json, save_result
from .planner_wrapper import DCSurfacePlanner


MARKER_NAMESPACE = '/pure3d/planning/controls'


class InteractivePlanning:
    """持有一次建好的空间索引，ROS 回调和耗时搜索分开执行。"""

    def __init__(self, node, prediction, initial_planning):
        self.node = node
        self.output = Path(initial_planning) / 'interactive'
        self.output.mkdir(parents=True, exist_ok=True)
        config = PlannerConfig(**json.loads((initial_planning / 'config.json').read_text()))
        self.planner = DCSurfacePlanner(config)
        self.planner.load_pcd(prediction)
        queries = json.loads((initial_planning / 'queries.json').read_text())
        self.positions = self.planner.anchor([queries['start'], queries['goal']])
        self.revision = 0
        self.pending = False
        self.future = None
        # 只有一个 planner 实例，顺序处理查询，期间仍接收标记反馈。
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pointcloud_planner')
        self.status_publisher = node.create_publisher(String, '/pure3d/planning/status', 1)
        self.server = InteractiveMarkerServer(node, MARKER_NAMESPACE)
        self.menu = MenuHandler()
        self.menu.insert('Plan A* + GPMP', callback=lambda _: self.request_plan())
        for index, name in enumerate(('start_pos', 'end_pos')):
            self.make_marker(name, index)
            self.menu.apply(self.server, name)
        self.server.applyChanges()
        self.timer = node.create_timer(.1, self.poll)
        self.set_status('ready', initial_planning=str(initial_planning))
        node.get_logger().info(
            f'Interactive markers: {MARKER_NAMESPACE}. Drag START/GOAL XYZ, release to plan. '
            f'Index reused: {len(self.planner.node_indices):,} search nodes.')

    @staticmethod
    def pose(position):
        pose = Pose(position=Point(**dict(zip(('x', 'y', 'z'), map(float, position)))))
        pose.orientation.w = 1.0
        return pose

    def make_marker(self, name, index):
        marker = InteractiveMarker(name=name, scale=1.4)
        marker.header.frame_id = self.node.frame_id
        marker.description = ('START' if index == 0 else 'GOAL') + ' | drag XYZ, release to plan'
        # 标记保存输入PCD锚点，DC支撑坐标只在规划器内部生成。
        marker.pose = self.pose(self.positions[index])
        sphere = Marker(type=Marker.SPHERE)
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = .35
        sphere.color.r, sphere.color.g, sphere.color.b = (0., 1., 1.) if index == 0 else (1., .1, .8)
        sphere.color.a = 1.0
        controls = [InteractiveMarkerControl(name='menu', always_visible=True, markers=[sphere],
                    interaction_mode=InteractiveMarkerControl.MENU)]
        for axis, quaternion in [('x', (0., 0., 0., 1.)),
                                 ('y', (0., 0., 2**-.5, 2**-.5)),
                                 ('z', (0., -2**-.5, 0., 2**-.5))]:
            control = InteractiveMarkerControl(name='move_' + axis,
                interaction_mode=InteractiveMarkerControl.MOVE_AXIS,
                orientation_mode=InteractiveMarkerControl.FIXED)
            (control.orientation.x, control.orientation.y,
             control.orientation.z, control.orientation.w) = quaternion
            controls.append(control)
        marker.controls = controls
        self.server.insert(marker, feedback_callback=self.feedback)

    def feedback(self, feedback):
        """松开才提交查询，拖动过程由 InteractiveMarkerServer 同步显示。"""
        if feedback.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return
        index = ('start_pos', 'end_pos').index(feedback.marker_name)
        p = feedback.pose.position
        query = np.array([p.x, p.y, p.z])
        anchor = self.planner.anchor(query)
        distance = np.linalg.norm(anchor - query)
        changed = np.linalg.norm(anchor - self.positions[index]) >= .01
        self.positions[index] = anchor
        self.server.setPose(feedback.marker_name, self.pose(anchor))
        self.server.applyChanges()
        if changed:
            self.node.get_logger().info(
                f'{feedback.marker_name}: query={query.tolist()}, pcd_anchor={anchor.tolist()}, '
                f'distance={distance:.3f}m')
            self.request_plan()

    def set_status(self, state, **details):
        status = dict(state=state, revision=self.revision,
                      start=self.positions[0].tolist(), goal=self.positions[1].tolist(),
                      endpoint_coordinates='input PCD support anchors in map XYZ metres', **details)
        save_json(self.output / 'status.json', status)
        self.status_publisher.publish(String(data=json.dumps(status)))

    def request_plan(self):
        self.revision += 1
        self.pending = True
        self.node.clear_planning()
        self.set_status('queued' if self.future is not None else 'planning')

    def compute(self, positions):
        started = time.perf_counter()
        result = self.planner.plan(*positions)
        return result, time.perf_counter() - started

    def poll(self):
        """只发布最新端点对应的结果，搜索中继续拖动时合并为最新查询。"""
        if self.future is not None and self.future.done():
            completed, self.future = self.future, None
            try:
                result, elapsed = completed.result()
            except (RuntimeError, ValueError) as error:
                self.node.get_logger().error(str(error))
                if self.running_revision == self.revision:
                    self.set_status('failed', error=str(error))
            else:
                if self.running_revision == self.revision:
                    save_result(self.planner, result, self.output, *self.running_positions, elapsed)
                    if result['success']:
                        self.node.publish_planning(self.output)
                        self.set_status('ready', astar=result['astar']['metrics'], gpmp=result['gpmp']['metrics'])
                        self.node.get_logger().info(
                            f'Query {self.revision}: A* {result["astar"]["metrics"]["length_m"]:.2f}m, '
                            f'GPMP {result["gpmp"]["metrics"]["length_m"]:.2f}m, {elapsed:.2f}s')
                    else:
                        self.set_status('failed', astar=result['astar']['metrics'])
        if self.future is None and self.pending:
            self.pending = False
            self.running_revision = self.revision
            self.running_positions = self.positions.copy()
            self.set_status('planning')
            self.future = self.worker.submit(self.compute, self.running_positions)

    def shutdown(self):
        self.timer.cancel()
        self.server.shutdown()
        self.worker.shutdown(wait=True)
