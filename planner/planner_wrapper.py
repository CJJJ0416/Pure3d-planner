from pathlib import Path
import pickle
import numpy as np

from .utils.convertion import transTrajGrid2Map

from global_planner.native_loader import load_native

a_star, ele_planner, traj_opt = load_native()


class TomogramPlanner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.use_quintic = self.cfg.planner.use_quintic
        self.max_heading_rate = self.cfg.planner.max_heading_rate
        self.cost_threshold = float(self.cfg.planner.cost_threshold)
        self.cost_barrier = float(self.cfg.planner.cost_barrier)


        self.tomogram=None
        self.resolution = None
        self.center = None
        self.n_slice = None
        self.slice_h0 = None
        self.slice_dh = None
        self.map_dim = []
        self.offset = None

        self.layer_elev_grids = None  # 存储每个layer的高度网格（elev_g）
        self.valid_ground = None

        self.start_idx = np.zeros(3, dtype=np.int32)
        self.end_idx = np.zeros(3, dtype=np.int32)

    def loadTomogram(self, tomo_file):
        # [ROS2适配] 使用 launch 提供的绝对地图路径，不再依赖脚本工作目录。
        with Path(tomo_file).open('rb') as handle:
            data_dict = pickle.load(handle)

            self.tomogram = np.asarray(data_dict['data'], dtype=np.float32)

            self.resolution = float(data_dict['resolution'])
            self.center = np.asarray(data_dict['center'], dtype=np.double)
            self.n_slice = self.tomogram.shape[1]
            self.slice_h0 = float(data_dict['slice_h0'])
            self.slice_dh = float(data_dict['slice_dh'])
            self.map_dim = [self.tomogram.shape[2], self.tomogram.shape[3]]
            self.offset = np.array([int(self.map_dim[0] / 2), int(self.map_dim[1] / 2)], dtype=np.int32)

        trav = self.tomogram[0]
        trav_gx = self.tomogram[1]
        trav_gy = self.tomogram[2]
        elev_g = self.tomogram[3]
        self.valid_ground = np.isfinite(elev_g)
        # 旧 pickle 也可能把无观测区域存成低代价；加载时屏蔽并同步代价梯度。
        if np.any(trav[~self.valid_ground] != self.cost_barrier):
            trav[~self.valid_ground] = self.cost_barrier
            trav_gx.fill(0.0)
            trav_gy.fill(0.0)
            trav_gx[:, 1:-1, :] = trav[:, 2:, :] - trav[:, :-2, :]
            trav_gy[:, :, 1:-1] = trav[:, :, 2:] - trav[:, :, :-2]
        # 保留 NaN 供 A* 显式判断地面有效性；GPMP 的缺测高度由原生入口处理。
        elev_g = np.where(self.valid_ground, elev_g, np.nan)
        elev_c = self.tomogram[4]
        elev_c = np.nan_to_num(elev_c, nan=1e6)
        # 存储所有layer的高度网格（关键：后续用于XY查高度）
        self.layer_elev_grids = elev_g
        self.initPlanner(trav, trav_gx, trav_gy, elev_g, elev_c)
      


    def _grid_cell_by_xy(self, x, y):
        """返回 tomogram 的 (x索引, y索引)；越界或非有限坐标返回 None。"""
        if not np.isfinite([x, y]).all():
            return None
        # 在整数转换前拒绝越界，避免将地图外坐标 clip 到边缘或发生整数溢出。
        cell = np.round((np.array([x, y]) - self.center) / self.resolution) + self.offset
        if not ((cell >= 0).all() and (cell < self.map_dim).all()):
            return None
        return int(cell[0]), int(cell[1])

    def get_layer_height_by_xy(self, layer_idx, x, y):
        """读取当前网格的地面高度；无观测/越界返回 NaN，不借用其他网格。"""
        cell = self._grid_cell_by_xy(x, y)
        if cell is None or not 0 <= layer_idx < self.n_slice:
            return float('nan')
        return float(self.layer_elev_grids[layer_idx, cell[0], cell[1]])
    

    def match_best_layer(self, x, y, target_height):
        """在有效且高度匹配的层中选最低代价；代价相同则选高度更近者。

        返回 layer 索引，没有有效候选时返回 None。零代价是合法通行代价。
        """
        cell = self._grid_cell_by_xy(x, y)
        if cell is None or not np.isfinite(target_height):
            return None
        grid_x, grid_y = cell
        best_key = (float('inf'), float('inf'))
        best_layer = None

        # 迭代所有layer（可优化：先粗筛再细查，减少迭代次数）
        for layer_idx in range(self.n_slice):
            # 获取该layer下XY对应的真实高度
            layer_height = float(self.layer_elev_grids[layer_idx, grid_x, grid_y])
            print("layer_idx:",layer_idx,flush=True)

            # 跳过无效层
            if not self.valid_ground[layer_idx, grid_x, grid_y]:
                continue
            # 计算高度差值（绝对值）
            diff = abs(layer_height - target_height)
            print("layer_height",layer_height,flush=True)
            print("target_height",target_height,flush=True)
            print("layer_height_diff",diff,flush=True)
            if diff > self.slice_dh:
                continue

            layer_cost_ = float(self.tomogram[0, layer_idx, grid_x, grid_y])
            print("layer_cost_",layer_cost_,flush=True)
            print("\n")

            if not np.isfinite(layer_cost_) or not 0.0 <= layer_cost_ <= self.cost_threshold:
                continue
            # 更新最佳匹配
            key = (layer_cost_, diff)
            if key < best_key:
                best_key = key
                best_layer = layer_idx
        return best_layer
        
    def initPlanner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        diff_t = trav[1:] - trav[:-1]
        diff_g = np.abs(elev_g[1:] - elev_g[:-1])

        gateway_up = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t < -8.0
        mask_g = (diff_g < 0.1) & self.valid_ground[1:] & self.valid_ground[:-1]
        gateway_up[:-1] = np.logical_and(mask_t, mask_g)

        gateway_dn = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t > 8.0
        mask_g = (diff_g < 0.1) & self.valid_ground[1:] & self.valid_ground[:-1]
        gateway_dn[1:] = np.logical_and(mask_t, mask_g)
        
        gateway = np.zeros_like(trav, dtype=np.int32)
        gateway[gateway_up] = 2
        gateway[gateway_dn] = -2

        self.planner = ele_planner.OfflineElePlanner(
            max_heading_rate=self.max_heading_rate, use_quintic=self.use_quintic
        )
        self.planner.init_map(
            self.cost_threshold, 15, self.resolution, self.n_slice, 1,
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            gateway.reshape(-1, gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
        )



    def plan(self, start_pos, end_pos):
        print("pos origin start:", start_pos,flush=True)
        print("pos origin end:", end_pos,flush=True)

        # 匹配起始点的最佳layer
        start_layer = self.match_best_layer(
            start_pos[0], start_pos[1], start_pos[2]
        )
        # 匹配目标点的最佳layer
        print("start_layer:", start_layer, flush=True)
        print("\n\n\n\n")

        end_layer= self.match_best_layer(
            end_pos[0], end_pos[1], end_pos[2]
        )
        print("end_layer:", end_layer, flush=True)
        if start_layer is None or end_layer is None:
            print("起终点选层失败：没有有效地面、匹配高度或可通行代价。", flush=True)
            return None
        

        self.start_idx[1:] = self.pos2idx(start_pos[:2])
        self.end_idx[1:] = self.pos2idx(end_pos[:2])
        self.start_idx[0] = start_layer
        self.end_idx[0] = end_layer
    

        if not self.planner.plan(self.start_idx, self.end_idx, True):
            return None
        path_finder: a_star.Astar = self.planner.get_path_finder()
        path = path_finder.get_result_matrix()
        if len(path) == 0:
            return None

        optimizer: traj_opt.GPMPOptimizer = (
            self.planner.get_trajectory_optimizer()
            if not self.use_quintic
            else self.planner.get_trajectory_optimizer_wnoj()
        )

        opt_init = optimizer.get_opt_init_value()
        init_layer = optimizer.get_opt_init_layer()
        traj_raw = optimizer.get_result_matrix()
        layers = optimizer.get_layers()
        heights = optimizer.get_heights()
        # print("heights origin:", heights)


        opt_init = np.concatenate([opt_init.transpose(1, 0), init_layer.reshape(-1, 1)], axis=-1)
        traj = np.concatenate([traj_raw, layers.reshape(-1, 1)], axis=-1)
        y_idx = (traj.shape[-1] - 1) // 2
        traj_3d = np.stack([traj[:, 0], traj[:, y_idx], heights / self.resolution], axis=1)
        traj_3d = transTrajGrid2Map(self.map_dim, self.center, self.resolution, traj_3d)

        # print(traj_raw,"traj_raw")
        return traj_3d
    

    def pos2idx(self, pos):
        pos = pos - self.center
        idx = np.round(pos / self.resolution).astype(np.int32) + self.offset
        idx = np.array([idx[1], idx[0]], dtype=np.float32)
        return idx
