"""Observe manual SCAN trajectories and local cost without publishing goals/control."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from common.io import atomic_json, JsonlTail


def milliseconds(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)] * 1000
    if not len(a):
        return dict(samples=0, median=None, p95=None, mean=None, maximum=None)
    return dict(samples=len(a), median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                mean=float(a.mean()), maximum=float(a.max()))


def main():
    import rclpy
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from scan_planner_msgs.msg import Bspline
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / 'monitor'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'events.jsonl').exists():
        parser.error('Monitor output already exists. Preserve it and use a new run directory.')
    rclpy.init(args=[])
    node = Node('entry_local_monitor', parameter_overrides=[Parameter('use_sim_time', value=True)])
    durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    state = dict(status=None, cost=None, body=None, replan_status=None)
    counts = Counter()
    goals, pending = [], []
    frame_latency, pipeline_latency, latest_latency, changed_latency = [], [], [], []
    recompute_latency, cached_latency, trajectory_gaps, next_cost_delays = [], [], [], []
    recent = deque()
    recent_cost = deque(maxlen=1000)
    recent_trajectory = deque(maxlen=1000)
    last_trajectory = None
    history_path = run / 'local/prediction_history.jsonl'
    # Start a fresh observation interval after model startup. Do not mix the
    # initialization backlog into the user's manual-target measurements.
    predictions = JsonlTail(history_path, start_at_end=True)
    refreshes = JsonlTail(run/'local/cache_refresh_history.jsonl', start_at_end=True)
    history_initial_offset = predictions.position
    started = time.perf_counter()
    stream = (output / 'events.jsonl').open('w', buffering=1)
    summary_pub = node.create_publisher(String, '/entry/local/monitor', durable)
    marker_pub = node.create_publisher(Marker, '/entry/local/monitor_text', durable)

    def emit(event, **value):
        stream.write(json.dumps(dict(event=event, observer_wall=time.perf_counter(),
            observer_sim=node.get_clock().now().nanoseconds * 1e-9, **value), allow_nan=False) + '\n')

    def goal(message):
        p = message.pose.position
        value = dict(index=len(goals), frame=message.header.frame_id, xyz=[p.x, p.y, p.z],
                     stamp=message.header.stamp.sec + message.header.stamp.nanosec * 1e-9)
        goals.append(value)
        emit('manual_goal', **value)

    def status(message):
        state['status'] = json.loads(message.data)

    def odometry(message):
        p = message.pose.pose.position
        state['body'] = [p.x, p.y, p.z]

    def cost(message):
        now = time.perf_counter()
        points = message.width * message.height
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        fingerprint = hashlib.blake2b(message.data, digest_size=12).hexdigest()
        previous = state['cost']
        counts['cost_messages'] += 1
        counts['nonempty_cost_messages' if points else 'empty_cost_messages'] += 1
        if previous and fingerprint != previous['fingerprint']:
            counts['cost_content_changes'] += 1
        if previous and stamp > previous['stamp']:
            counts['cost_stamp_progressions'] += 1
        state['cost'] = dict(stamp=stamp, received_wall=now, points=points, fingerprint=fingerprint)
        recent_cost.append(now)
        emit('cost_received', **state['cost'])
        for item in pending[:]:
            # A repeated/cached publication is separate from a newer input snapshot.
            if points and stamp > item['previous_cost_stamp']:
                delay = now - item['wall']
                next_cost_delays.append(delay)
                counts['trajectories_followed_by_new_cost'] += 1
                emit('trajectory_next_cost', trajectory=item['trajectory'], delay_s=delay,
                     cost_stamp=stamp, content_changed=fingerprint != item['previous_fingerprint'])
                pending.remove(item)

    def trajectory(message):
        nonlocal last_trajectory
        now = time.perf_counter()
        counts['trajectories'] += 1
        recent_trajectory.append(now)
        if last_trajectory is not None:
            trajectory_gaps.append(now - last_trajectory)
        last_trajectory = now
        current, s = state['cost'], state['status'] or {}
        emit('trajectory_received', trajectory=message.traj_id, goal_index=len(goals)-1,
             cost_stamp=None if current is None else current['stamp'],
             cost_publication_gap_s=None if current is None else now-current['received_wall'],
             status_valid=s.get('valid'), map_revision=s.get('revision'),
             cost_revision=s.get('cost_revision'), status_input_age_s=s.get('input_age_s'),
             status_result_age_s=s.get('result_age_s'), points=0 if current is None else current['points'])
        pending.append(dict(trajectory=message.traj_id, wall=now,
            previous_cost_stamp=-1 if current is None else current['stamp'],
            previous_fingerprint=None if current is None else current['fingerprint']))

    def display(message):
        counts['rviz_cost_messages'] += 1

    def read_history():
        for report in predictions.read():
            timing = report.get('latency', {})
            counts['predictions_observed'] += 1
            if not timing.get('published'):
                counts['unpublished_predictions'] += 1
                continue
            values = timing.get('frame_to_publish_s', [])
            pipeline_latency.append(report['pipeline_s'])
            latest = timing.get('latest_scan_to_publish_s')
            if latest is not None:
                latest_latency.append(latest)
            frame_latency.extend(values)
            changed_latency.extend(timing.get('changed_frame_to_publish_s', []))
            (recompute_latency if report.get('encoder_calls', 0) else cached_latency).append(report['pipeline_s'])
            recent.append((time.perf_counter(), values, report['pipeline_s']))

        for report in refreshes.read():
            timing=report['latency'];values=timing.get('frame_to_publish_s',[])
            counts['cache_publications_observed']+=1
            frame_latency.extend(values)
            changed_latency.extend(timing.get('changed_frame_to_publish_s',[]))
            latest=timing.get('latest_scan_to_publish_s')
            if latest is not None:latest_latency.append(latest)
            recent.append((time.perf_counter(),values,None))

    last_print = 0

    def tick():
        nonlocal last_print
        read_history()
        now = time.perf_counter()
        s, current = state['status'] or {}, state['cost']
        for item in pending[:]:
            if now-item['wall'] < 1:
                continue
            cache_valid = bool(current and current['points'] and s.get('valid') and
                               s.get('revision') == s.get('cost_revision'))
            reason = 'unchanged_geometry_cache_valid' if cache_valid else 'no_new_cost_within_1s'
            counts[reason] += 1
            emit('trajectory_cost_check', trajectory=item['trajectory'], result=reason,
                 status_valid=s.get('valid'), map_revision=s.get('revision'), cost_revision=s.get('cost_revision'))
            pending.remove(item)
        while recent and now-recent[0][0] > 60:
            recent.popleft()
        elapsed = now-started
        window = min(10., elapsed)
        cost_hz = sum(t >= now-window for t in recent_cost) / window if window else 0
        trajectory_hz = sum(t >= now-window for t in recent_trajectory) / window if window else 0
        full = milliseconds(frame_latency)
        summary = dict(state='observing', manual_targets=True, scan_cost_integration=False,
            elapsed_wall_s=elapsed, goals=len(goals), counts=dict(counts), body=state['body'],
            prediction_history_initial_offset=history_initial_offset,
            full_frame_to_cost_ms=full, geometry_changed_frame_to_cost_ms=milliseconds(changed_latency),
            latest_frame_to_cost_ms=milliseconds(latest_latency),
            submit_to_result_ms=milliseconds(pipeline_latency),
            encoder_recomputed_submit_to_result_ms=milliseconds(recompute_latency),
            cached_submit_to_result_ms=milliseconds(cached_latency),
            context_mode=s.get('context_mode'), context_points=s.get('context_points'),
            phase_timing_ms=s.get('phase_timing_ms'),
            recent_60s_full_frame_to_cost_ms=milliseconds([v for _, a, _ in recent for v in a]),
            trajectory_interval_ms=milliseconds(trajectory_gaps),
            trajectory_to_next_newer_cost_ms=milliseconds(next_cost_delays),
            recent_cost_hz=cost_hz, recent_trajectory_hz=trajectory_hz,
            scan_replan=state['replan_status'],
            latest_cost=None if current is None else dict(**current, publication_gap_s=now-current['received_wall']),
            local_status={key:s.get(key) for key in ('state','valid','revision','cost_revision','query_revision','context_revision','points',
                'input_age_s','result_age_s','pose_age_s','box_source','cache_refreshes','predictions','accepted_scans','scans_without_in_box_points',
                'pending_snapshots','cost_points','knn_filter','error')},
            timing_boundary='Entry registered-cloud callback -> cost publisher publish() returned',
            excludes=['raw sensor acquisition', 'Faster-LIO processing', 'RViz rendering'],
            trajectory_observation='independent observer arrival time, not optimizer execution time')
        atomic_json(output / 'summary.json', summary)
        summary_pub.publish(String(data=json.dumps(summary, allow_nan=False)))
        med = '--' if full['median'] is None else f"{full['median']:.0f}"
        p95 = '--' if full['p95'] is None else f"{full['p95']:.0f}"
        label = (f"Tiny local cost  |  valid: {s.get('valid', False)}\n"
                 f"E2E median / P95: {med} / {p95} ms\n"
                 f"Cost: {cost_hz:.1f} Hz  |  SCAN trajectory: {trajectory_hz:.1f} Hz\n"
                 f"Map / cost revision: {s.get('revision', '--')} / {s.get('cost_revision', '--')}")
        if state['body']:
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = node.get_clock().now().to_msg()
            marker.ns = 'entry_local_monitor'
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = state['body']
            marker.pose.position.z += 1.5
            marker.scale.z = .16
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = .95, .95, .95, 1.
            marker.text = label
            marker_pub.publish(marker)
        if now-last_print >= 5:
            print(label.replace('\n', ' | '), flush=True)
            last_print = now

    node.create_subscription(PoseStamped, '/scan/goal', goal, 10)
    node.create_subscription(Bspline, '/scan/planning/bspline', trajectory, 10)
    node.create_subscription(String, '/entry/local/status', status, durable)
    node.create_subscription(String, '/scan/planning/replan_status', lambda m:state.update(replan_status=json.loads(m.data)),10)
    node.create_subscription(PointCloud2, '/entry/local/cost_cloud', cost, durable)
    node.create_subscription(PointCloud2, '/entry/local/rviz/cost_cloud', display, durable)
    node.create_subscription(Odometry, '/Odometry_gazebo', odometry, sensor)
    node.create_timer(1., tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        read_history()
        emit('monitor_stopped')
        stream.close()
        path = output / 'summary.json'
        if path.exists():
            final = json.loads(path.read_text())
            final.update(state='stopped', pending_trajectory_checks=len(pending))
            atomic_json(path, final)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
