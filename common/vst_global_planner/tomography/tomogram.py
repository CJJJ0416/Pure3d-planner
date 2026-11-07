import numpy as np
import cupy as cp

import cv2
from pathlib import Path


from .kernels import *


class Tomogram(object):
    def __init__(self, cfg, debug_dir):
        # [ROS2适配] 原骨架调试图片集中到地图缓存目录，不写启动终端的 cwd。
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = cfg.map.resolution
        self.slice_dh = cfg.map.slice_dh
        self.half_trav_k_size = int(cfg.trav.kernel_size / 2)
        self.interval_min = cfg.trav.interval_min
        self.interval_free = cfg.trav.interval_free
        self.step_stand = 1.2 * self.resolution * np.tan(cfg.trav.slope_max)
        self.step_cross = cfg.trav.step_max
        self.standable_th = int(cfg.trav.standable_ratio * (2 * self.half_trav_k_size + 1) ** 2) - 1
        self.cost_barrier = float(cfg.trav.cost_barrier)
        self.safe_margin = cfg.trav.safe_margin
        self.safe_margin_gamma = cfg.trav.safe_margin_gamma
        self.inflation = cfg.trav.inflation
        self.half_inf_k_size = int((self.safe_margin + self.inflation) / self.resolution)

    def compute_skeleton_and_w_local(self, cost_np, dist_np):
        n_s, h, w = cost_np.shape
        w_local_map = np.full((n_s, h, w), 0.0, dtype=np.float32)

        # 创建 3x3 十字结构元素（经典 skeleton 常用）
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for s in range(n_s):
            print("times",s)
            layer_cost = cost_np[s]
            dist_layer = dist_np[s]

            free_mask =  (layer_cost < self.cost_barrier - 0.001)
            # 地面观测可以只覆盖地图的一小部分，不能用全图占比过滤有效通道。
            # 空层直接跳过；骨架可靠性由后面的骨架点数量检查控制。
            if not np.any(free_mask):
                continue

            # 转为 OpenCV binary 图 (0/255)
            binary = (free_mask.astype(np.uint8)) * 255

            # ==================== OpenCV 形态学 Skeleton ====================
            skeleton = np.zeros(binary.shape, np.uint8)
            temp = np.zeros(binary.shape, np.uint8)
            img = binary.copy()


            cv2.imwrite(str(self.debug_dir / f"w_local_map{s}_binary.jpg"), img)

            # 停止条件：没有像素可侵蚀
            cont=0
            while cv2.countNonZero(img)>0:
                cont+=1
                # erode
                eroded = cv2.erode(img, element)
                # dilate back
                dilate = cv2.dilate(eroded, element)
                # subtract to get boundary
                temp = cv2.subtract(img, dilate)
                # add to skeleton
                skeleton = cv2.bitwise_or(skeleton, temp)
                # update img
                img = eroded.copy()

   
                # if s==23:
                #     cv2.imwrite(f"w_local_map{s}_eroded_{cont}.jpg", eroded)

                #     cv2.imwrite(f"w_local_map{s}_dilate_{cont}.jpg", dilate)

                #     cv2.imwrite(f"w_local_map{s}_skeleton_{cont}.jpg", skeleton)

                #     cv2.imwrite(f"w_local_map{s}_subtract_{cont}.jpg", temp)

                # ====================== 可视化部分（仅在第23层） ======================
                # if s == 23:   # 只可视化你关心的层（可改成其他层号）
                    
                #     vis_temp = np.full((h, w, 3), 255, dtype=np.uint8)

                #     # 2. 当前剩余区域 (img) 用浅蓝色显示（背景）
                #     blue_layer = np.zeros((h, w, 3), dtype=np.uint8)
                #     blue_layer[img > 0] = [255, 255, 255]   # 浅蓝色偏黄（容易区分）
                #     vis_temp = cv2.addWeighted(vis_temp, 0.3, blue_layer, 1, 0)

                #     # 3. 被裁减的边界层 (temp) 用**鲜艳红色**高亮 + 加粗显示
                #     if cv2.countNonZero(temp) > 0:
                #         # 先膨胀一点让边界更明显（可选但强烈推荐）
                #         temp_thick = cv2.dilate(temp, np.ones((5,5), np.uint8), iterations=1)
                #         red_layer = np.zeros((h, w, 3), dtype=np.uint8)
                #         red_layer[temp_thick > 0] = [0, 0, 255]     # 纯红色
                #         vis_temp = cv2.addWeighted(vis_temp, 1, red_layer, 1, 0)


                #     cv2.imwrite(f"w_local_map{s}_temp_{cont}.jpg", vis_temp)

                #     # ==================== skeleton 可视化（白色底 + 黑色骨架） ====================
                #     vis_skel = np.full((h, w), 255, dtype=np.uint8)
                #     vis_skel[skeleton > 0] = 0
                #     cv2.imwrite(f"w_local_map{s}_skeleton_{cont}.jpg", vis_skel)



            # skeleton 现在是 0/255，转为 bool
            sk_bool = (skeleton > 0)

            sk_points = np.argwhere(sk_bool)
            if len(sk_points) < 3:          # 太少骨架点跳过
                continue
            sk_w = dist_layer[sk_bool]                  # 骨架点对应的距离值 (一维)
            w_local_map[s][sk_bool] = sk_w              # 直接用 bool 掩码赋值

        return w_local_map

    def initKernel(self):
        # 断层扫描核（生成初始地图层）
        self.tomography_kernel = tomographyKernel(
            self.resolution, 
            self.map_dim_x, 
            self.map_dim_y,
            self.n_slice_init,
            self.slice_h0,
            self.slice_dh
        )
        # 可通行性计算核
        self.trav_kernel = travKernel(
            self.map_dim_x,
            self.map_dim_y,
            self.half_trav_k_size,
            self.interval_min,
            self.interval_free,
            self.step_cross, 
            self.step_stand, 
            self.standable_th, 
            self.cost_barrier
        )
        # 成本膨胀核
        self.inflation_kernel = inflationKernel(
            self.map_dim_x,
            self.map_dim_y,
            self.half_inf_k_size,
        )


        # 新增：距离变换 kernel（使用与 inflation 相同的 kernel size）
        self.dist_transform_kernel = distanceTransformKernel(
            self.map_dim_x, self.map_dim_y,
            self.half_inf_k_size,          # 复用 inflation 的窗口大小
            self.cost_barrier,
        )


        self.adjust_cost_kernel = adjustCostByDistanceKernel(
            self.map_dim_x, 
            self.map_dim_y,
            self.safe_margin,
            self.cost_barrier,
            self.resolution,
            p=2              # ← 可在这里改默认陡峭度，或后面动态改
        )

        # 新增：计算到骨架点的距离（w_local）
        self.skeleton_distance_kernel = skeletonDistanceKernel(
            self.map_dim_x, 
            self.map_dim_y,
            self.half_inf_k_size * 2          # 建议把窗口开大一点，保证覆盖较宽河谷
        )

        
        # 初始化膨胀表（用于后续成本膨胀计算）
        self.inf_table = cp.zeros(
            (2 * self.half_inf_k_size + 1, 2 * self.half_inf_k_size + 1), 
            dtype=cp.float32
        )
        # 全GPU版本的指数权重计算

        for i in range(self.inf_table.shape[0]):
            for j in range(self.inf_table.shape[1]):
                dist = np.sqrt(
                    (self.resolution * (i - self.half_inf_k_size)) ** 2 + \
                    (self.resolution * (j - self.half_inf_k_size)) ** 2
                )

                self.inf_table[i, j] = np.clip(
                    1 - (dist - self.inflation) / (self.safe_margin + self.resolution),
                    a_min=0.0, a_max=1.0
                )

    def initBuffers(self):
        self.layers_g = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.layers_c = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.grad_mag_sq = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.grad_mag_max = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.trav_cost = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)
        self.inflated_cost = cp.zeros((self.n_slice_init, self.map_dim_x, self.map_dim_y), dtype=cp.float32)

    def initMappingEnv(self, center, map_dim_x, map_dim_y, n_slice_init, slice_h0):
        self.center = cp.array(center, dtype=cp.float32)
        self.map_dim_x = int(map_dim_x)
        self.map_dim_y = int(map_dim_y)
        self.n_slice_init = int(n_slice_init)
        self.slice_h0 = float(slice_h0)
        
        self.initBuffers()
        self.initKernel()

    def clearMap(self):
        self.layers_g *= 0.
        self.layers_c *= 0.
        self.layers_g += -1e6
        self.layers_c += 1e6

        self.grad_mag_sq *= 0.
        self.grad_mag_max *= 0.
        self.trav_cost *= 0.
        self.inflated_cost *= 0.

    def point2map(self, points):
        points = cp.asarray(points)
        points = points[~cp.isnan(points).any(axis=1)]
        self.clearMap()

        # Tomogram
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()
        
        self.tomography_kernel(
            points, self.center, 
            self.layers_g, self.layers_c,
            size=(points.shape[0])
        )

        diff_x_sq = cp.maximum(
            (self.layers_g[:, 1:-1, :] - self.layers_g[:, :-2, :]) ** 2, 
            (self.layers_g[:, 1:-1, :] - self.layers_g[:,  2:, :]) ** 2
        )
        diff_y_sq = cp.maximum(
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :, :-2]) ** 2, 
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :,  2:]) ** 2
        )
        self.grad_mag_sq[:, 1:-1, 1:-1] = diff_x_sq[:, :, 1:-1] + diff_y_sq[:, 1:-1, :]
        self.grad_mag_max[:, 1:-1, 1:-1] = cp.maximum(diff_x_sq[:, :, 1:-1], diff_y_sq[:, 1:-1, :])
        
        interval = (self.layers_c - self.layers_g)

        end_gpu.record()
        end_gpu.synchronize()
        gpu_t_map = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

        # Traversability
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()

        self.trav_kernel(
            interval, self.grad_mag_sq, self.grad_mag_max,
            self.trav_cost,
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )

        # 无地面观测不能参与距离变换或自由空间骨架；有效平地允许零代价。
        valid_ground = cp.isfinite(self.layers_g) & (self.layers_g > -1e6)
        self.trav_cost[~valid_ground] = self.cost_barrier

        dist_to_barrier = cp.zeros_like(self.trav_cost, dtype=cp.float32)

        self.dist_transform_kernel(
            self.trav_cost, 
            dist_to_barrier,
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )

        dist_full = dist_to_barrier.get()                                      # CPU

        # 提前把简化后的 trav_cost 转 CPU（用于提取骨架）
        trav_cost_np = self.trav_cost.get() if hasattr(self, 'trav_cost') else None
        if trav_cost_np is not None:
            w_local_map_np = self.compute_skeleton_and_w_local(trav_cost_np, dist_full)
            skeleton_mask = cp.asarray(w_local_map_np)                       # 转回 GPU
    
        w_local_map = cp.zeros_like(self.trav_cost, dtype=cp.float32)

        # 计算 w_local_map，每个自由点与距离骨架点
        self.skeleton_distance_kernel(
            skeleton_mask,
            w_local_map,
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )
        # ==================== 新增：作图jetmap ====================


        # w_local_np = w_local_map.get()                    # GPU → CPU
        # skeleton_np = w_local_map_np                    # 骨架掩码

        # for s in range( w_local_np.shape[0]):      # 显示前6层
        #     vis = w_local_np[s].copy()

        #     # 【关键修改】只保留值小于 10 的区域，其余设为 0（黑色）
        #     valid_mask = vis < 100.0
        #     vis_filtered = np.where(valid_mask, vis, 0.0)

        #     # 归一化到 0-255（仅在有效区域内归一化）
        #     if vis_filtered.max() > 0:
        #         vis_norm = (vis_filtered / vis_filtered.max() * 255).astype(np.uint8)
        #     else:
        #         vis_norm = np.zeros_like(vis_filtered, dtype=np.uint8)

        #     # 使用 Jet 彩色映射
        #     vis_color = cv2.applyColorMap(vis_norm, cv2.COLORMAP_JET)

        #     # 骨架点标红（覆盖 Jet 颜色）
        #     sk_mask = (skeleton_np[s] > 0.5)
        #     vis_color[sk_mask] = [0, 0, 255]              # 纯红色

        #     # 显示窗口
        #     cv2.imwrite(f"w_local_map{s}.jpg", vis_color)

        # =====================================================================


        # 调用 kernel 计算最终 cost
        self.adjust_cost_kernel(
            self.trav_cost,          # 输入：当前 trav_cost
            dist_to_barrier,         # 输入：dist_to_barrier
            w_local_map,             # ← 新增输入：最近骨架点的 w_local
            self.trav_cost,           # 输出（in-place）
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )

        self.inflation_kernel(
            self.trav_cost,          # 输入当前代价
            self.inf_table,          # 膨胀权重表
            w_local_map,             # ← 新增：w_local 用于保护骨架
            self.inflated_cost,      # 输出
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )

        # 在层简化和梯度计算前保证最终地图仍屏蔽无地面区域。
        self.inflated_cost[~valid_ground] = self.cost_barrier

        end_gpu.record()
        end_gpu.synchronize()
        gpu_t_trav = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

        # Layer Simplification
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()

        idx_simp = [0]
        if self.layers_g.shape[0] > 1:
            l_idx, m_idx = 0, 1
            diff_h = self.layers_g[1:] - self.layers_g[:-1]
            while m_idx < self.n_slice_init - 2:
                mask_l_g = self.layers_g[m_idx] - self.layers_g[l_idx] > 0
                mask_l_t = self.inflated_cost[l_idx] > self.inflated_cost[m_idx]
                mask_u_g = diff_h[m_idx] > 0
                mask_t = self.inflated_cost[m_idx] < self.cost_barrier
                unique = (mask_l_g | mask_l_t) & mask_u_g & mask_t
                if cp.any(unique):
                    idx_simp.append(m_idx)
                    l_idx = m_idx
                m_idx += 1
            idx_simp.append(m_idx)

        trav_grad_x = (self.inflated_cost[idx_simp][:, 2:, :] - self.inflated_cost[idx_simp][:, :-2, :])
        trav_grad_y = (self.inflated_cost[idx_simp][:, :, 2:] - self.inflated_cost[idx_simp][:, :, :-2])
        
        end_gpu.record()
        end_gpu.synchronize()
        gpu_t_simp = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        
        gpu_t_all = gpu_t_map + gpu_t_trav + gpu_t_simp
        #print("CuPy GPU time (ms):", gpu_t_all)

        layers_t = self.inflated_cost[idx_simp].get()
        layers_g = cp.where(
            self.layers_g[idx_simp] > -1e6, 
            self.layers_g[idx_simp],
            cp.nan
        ).get()
        layers_c = cp.where(
            self.layers_c[idx_simp] < 1e6, 
            self.layers_c[idx_simp], 
            cp.nan
        ).get()
        trav_gx = np.zeros_like(layers_g)
        trav_gx[:, 1:-1, :] = trav_grad_x.get()
        trav_gy = np.zeros_like(layers_g)
        trav_gy[:, :, 1:-1] = trav_grad_y.get()

        t_gpu = {
            't_map': gpu_t_map, 
            't_trav': gpu_t_trav, 
            't_simp': gpu_t_simp, 
        }
        
        return layers_t, trav_gx, trav_gy, layers_g, layers_c, t_gpu
