"""固定A*高度曲线的反重力GPMP，保留已验收的代价与平滑参数。"""
import time

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial import cKDTree


def gp_transition(dt):
    """状态顺序 [x,y,z,vx,vy,vz]，恒速状态转移矩阵。"""
    return np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])


def gp_covariance(dt, qc):
    """白噪声加速度积分得到的位置/速度联合协方差。"""
    return qc * np.kron(np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]), np.eye(3))


def gp_interpolation(dt, tau, qc):
    """条件 GP 插值：x(tau)=Lambda*x0+Psi*x1。"""
    cross = gp_covariance(tau, qc) @ gp_transition(dt - tau).T
    psi = np.linalg.solve(gp_covariance(dt, qc), cross.T).T
    return gp_transition(tau) - psi @ gp_transition(dt), psi


class GravityGPMP:
    def __init__(self, xyz, cost, config):
        self.config = config
        self.xyz = np.asarray(xyz, dtype=np.float32)
        self.cost = np.asarray(cost, dtype=np.float32)
        self.cost_tree = cKDTree(self.xyz)

    def query_cost(self, positions):
        """三维高斯近邻加权代价，以及归一化权重的解析 XYZ 导数。"""
        positions = np.asarray(positions, dtype=np.float64)
        distance, indices = self.cost_tree.query(positions, k=self.config.field_neighbors)
        delta = self.xyz[indices].astype(np.float64) - positions[:, None, :]
        # 减去每行最大对数权重不改变归一化权重及导数。
        logits = -(distance**2) / (2 * self.config.field_sigma**2)
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        costs = self.cost[indices].astype(np.float64)
        values = (weights * costs).sum(axis=1)
        gradient = np.einsum('nk,nkj->nj', weights * (costs - values[:, None]), delta)
        gradient /= self.config.field_sigma**2
        return values, gradient

    def optimize(self, surface_path):
        return _optimize(self, surface_path)



def _optimize(planner, path):
    """保留所有A*节点，固定平滑高度，优化水平GP曲线，再统一反重力偏移。"""
    started = time.perf_counter()
    cfg = planner.config
    gravity_up = cfg.gravity_up
    corridor_radius_m = cfg.horizontal_corridor_m
    height_smoothing_m = cfg.height_smoothing_m
    height_tolerance_m = cfg.height_tolerance_m
    path = np.asarray(path, dtype=float)
    frame = np.eye(3)
    up = np.asarray(gravity_up, dtype=float)
    up /= np.linalg.norm(up)
    axis = np.eye(3)[np.argmin(np.abs(up))]
    horizontal = axis-up*np.dot(axis, up)
    horizontal /= np.linalg.norm(horizontal)
    frame = np.column_stack((horizontal, np.cross(up, horizontal), up))
    # 只旋转坐标轴，不设置地图零点。第三轴为反重力方向。
    path = path @ frame
    normals = np.repeat(up[None, :], len(path), axis=0)
    normals = np.asarray(normals, dtype=float)
    count = len(path)
    arc = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    dt = arc[-1] / cfg.reference_speed / (count - 1)
    knot_time = np.arange(count) * dt
    initial = np.column_stack((path, np.gradient(path, dt, axis=0)))
    spacing = arc[-1]/(count-1)
    second = sparse.diags([np.ones(count-2), -2*np.ones(count-2), np.ones(count-2)],
                          [0, 1, 2], shape=(count-2, count))/spacing**2
    height_map = sparse.vstack([sparse.eye(count), height_smoothing_m**2*second], format='csr')
    target = np.r_[path[:, 2], np.zeros(count-2)]-height_map[:, [0, count-1]] @ path[[0,-1],2]
    fit = lsq_linear(height_map[:, 1:-1], target,
                     bounds=(path[1:-1,2]-height_tolerance_m, path[1:-1,2]+height_tolerance_m),
                     tol=1e-8, lsmr_tol=1e-8, max_iter=100)
    initial[1:-1, 2] = fit.x
    height_curve = CubicSpline(knot_time, initial[:, 2], bc_type=((1, 0.), (1, 0.)))
    initial[:, 5] = height_curve.derivative()(knot_time)
    initial[[0, -1], 3:] = 0.
    phi = gp_transition(dt)
    whitening = np.linalg.solve(np.linalg.cholesky(gp_covariance(dt, cfg.gp_qc)), np.eye(6))
    left = sparse.diags(np.ones(count-1), 0, shape=(count-1, count))
    right = sparse.diags(np.ones(count-1), 1, shape=(count-1, count))
    prior_full = (sparse.kron(left, -whitening @ phi)
                  + sparse.kron(right, whitening)).tocsr()
    # 固定 Z/Vz 的先验是常数。去掉这些常数残差，避免影响 ftol 停止判断。
    prior_rows = (np.arange(count-1)[:, None]*6 + np.array([0, 1, 3, 4])).ravel()
    prior = prior_full[prior_rows]
    steps = cfg.interpolation_points + 1
    coefficients = [gp_interpolation(dt, dt*k/steps, cfg.gp_qc) for k in range(steps)]
    rows, columns, values = [], [], []
    for segment in range(count-1):
        for sub, (lam, psi) in enumerate(coefficients):
            query = segment*steps + sub
            for node, coefficient in [(segment, lam), (segment+1, psi)]:
                rr, cc = np.nonzero(coefficient)
                rows.extend(6*query+rr)
                columns.extend(6*node+cc)
                values.extend(coefficient[rr, cc])
    query = (count-1)*steps
    rows.extend(6*query+np.arange(6))
    columns.extend(6*(count-1)+np.arange(6))
    values.extend(np.ones(6))
    times = np.arange(query+1) * dt/steps
    interpolator = sparse.csr_matrix((values, (rows, columns)),
                                    shape=(6*len(times), 6*count))
    position_rows = (np.arange(len(times))[:, None]*6+np.arange(3)).ravel()
    velocity_rows = (np.arange(len(times))[:, None]*6+np.arange(3, 6)).ravel()
    position_map, velocity_map = interpolator[position_rows], interpolator[velocity_rows]
    free = (np.arange(1, count-1)[:, None]*6+np.array([0, 1, 3, 4])).ravel()
    prior_jac, position_jac, velocity_jac = prior[:, free], position_map[:, free], velocity_map[:, free]
    cost_scale = np.sqrt(cfg.gp_cost_weight * dt/steps)
    speed_scale = np.sqrt(cfg.gp_speed_weight * dt/steps)
    row, column = np.repeat(np.arange(len(times)), 3), np.arange(3*len(times))

    def query_cost(positions):
        costs, gradient = planner.query_cost(positions @ frame.T)
        return costs, gradient @ frame

    def factors(variables, jacobian=False):
        flat = initial.ravel().copy()
        flat[free] = variables
        positions = (position_map @ flat).reshape(-1, 3)
        velocities = (velocity_map @ flat).reshape(-1, 3)
        costs, gradient = query_cost(positions)
        speed = np.linalg.norm(velocities, axis=1)
        if not jacobian:
            return np.r_[prior @ flat, cost_scale*costs,
                         speed_scale*np.maximum(speed-cfg.max_speed, 0)]
        cost_jac = sparse.csr_matrix((cost_scale*gradient.ravel(), (row, column)),
                                    shape=(len(times), len(column))) @ position_jac
        direction = np.zeros_like(velocities)
        active = speed > cfg.max_speed
        direction[active] = velocities[active] / speed[active, None]
        speed_jac = sparse.csr_matrix((speed_scale*direction.ravel(), (row, column)),
                                     shape=(len(times), len(column))) @ velocity_jac
        return sparse.vstack([prior_jac, cost_jac, speed_jac], format='csr')

    variables = initial.ravel()[free]
    bounds = (-np.inf, np.inf)
    # Hermite 水平修正为 h00*dp0+h01*dp1+dt*(h10*dv0+h11*dv1)。
    # h00+h01=1，|h10|+|h11|=u*(1-u)<=1/4。
    # 80% 预算用于节点位移，20% 用于段内速度弯曲，整段距离均 <= radius。
    position_bound = .8*corridor_radius_m/np.sqrt(2)
    velocity_bound = .8*corridor_radius_m/(dt*np.sqrt(2))
    extent = np.tile([position_bound, position_bound, velocity_bound, velocity_bound], count-2)
    bounds = (variables-extent, variables+extent)
    before = factors(variables)
    solution = least_squares(factors, variables, jac=lambda x: factors(x, True),
                            bounds=bounds,
                            method='trf', tr_solver='lsmr', max_nfev=cfg.max_iterations,
                            ftol=1e-5, xtol=1e-5, gtol=1e-5)
    flat = initial.ravel().copy()
    flat[free] = solution.x
    state = flat.reshape(-1, 6)
    dense = (interpolator @ flat).reshape(-1, 6)
    dense_initial = (interpolator @ initial.ravel()).reshape(-1, 6)
    dense_normals = np.repeat(up[None, :], len(times), axis=0)
    support_xyz = dense[:, :3] @ frame.T
    support_velocity = dense[:, 3:] @ frame.T
    xyz = support_xyz+cfg.endpoint_offset_m*dense_normals
    velocity = support_velocity
    costs, _ = query_cost(dense[:, :3])
    length = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    yaw = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    yaw[0], yaw[-1] = yaw[1], yaw[-2]
    endpoint_targets = path[[0, -1]] @ frame.T+cfg.endpoint_offset_m*normals[[0, -1]]
    metrics = dict(
        optimization_s=time.perf_counter()-started, converged=bool(solution.success),
        solver='scipy.optimize.least_squares (sparse TRF/LSMR)',
        prior='WNOA GP in XY, prescribed shape-preserving A* Z/Vz',
        status=int(solution.status), message=solution.message, evaluations=int(solution.nfev),
        initial_objective=float(before@before/2), objective=float(solution.cost),
        optimality=float(solution.optimality), knots=count, astar_points=count,
        trajectory_points=len(times), duration_s=float(times[-1]), length_m=float(length.sum()),
        integral_cost_m=float(np.sum(length*(costs[1:]+costs[:-1])/2)), max_cost=float(costs.max()),
        max_speed_m_s=float(np.linalg.norm(velocity, axis=1).max()),
        support_max_speed_m_s=float(np.linalg.norm(dense[:, 3:], axis=1).max()),
        endpoint_offset_m=cfg.endpoint_offset_m,
        start_error_m=float(np.linalg.norm(xyz[0]-endpoint_targets[0])),
        goal_error_m=float(np.linalg.norm(xyz[-1]-endpoint_targets[1])),
        max_knot_z_error_m=float(np.abs(state[:, 2]-path[:, 2]).max()),
        max_dense_z_profile_error_m=float(np.abs(dense[:, 2]-dense_initial[:, 2]).max()),
        max_xy_move_m=float(np.linalg.norm(state[:, :2]-path[:, :2], axis=1).max()),
        max_dense_horizontal_move_m=float(np.linalg.norm(dense[:, :2]-dense_initial[:, :2], axis=1).max()),
        horizontal_corridor_radius_m=corridor_radius_m,
        height_smoothing_m=height_smoothing_m, height_tolerance_m=height_tolerance_m,
        height_curve='C2 clamped cubic spline',
        corridor_definition='Distance to the fixed initial A* GP curve in the plane perpendicular to gravity.',
        gravity_up=up.tolist(),
        minimum_adjacent_normal_dot=float((normals[1:]*normals[:-1]).sum(1).min()),
        cost_evaluation='Optimized support XYZ, before gravity offset',
        velocity_evaluation='Support and body velocity are identical under uniform gravity translation',
        speed_factor='Support velocity',
    )
    return dict(time=times, xyz=xyz, velocity=velocity, yaw=yaw, cost=costs,
                support_xyz=support_xyz, support_velocity=support_velocity,
                normal=dense_normals, initial_xyz=dense_initial[:, :3] @ frame.T+cfg.endpoint_offset_m*dense_normals,
                reference_z=(dense_initial[:, :3] @ frame.T)[:, 2],
                initial_state=np.column_stack((initial[:, :3] @ frame.T, initial[:, 3:] @ frame.T)),
                state=np.column_stack((state[:, :3] @ frame.T, state[:, 3:] @ frame.T)), dt=dt,
                gravity_frame=frame, reference_height=dense_initial[:, 2],
                astar_surface_path=path @ frame.T,
                surface_endpoints=path[[0, -1]] @ frame.T, endpoint_normals=normals[[0, -1]],
                endpoint_targets=endpoint_targets, metrics=metrics)
