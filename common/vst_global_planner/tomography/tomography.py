# 原 CUDA 分层建图流程；ROS 发布已分离到 plan 节点，此模块只读取点云并导出 pickle。
#!/usr/bin/python3
import os
import time
import pickle
import numpy as np
import open3d as o3d
  
import logging
from pathlib import Path

from .tomogram import Tomogram
from .config import GRID_POINTS_XYZI

logger = logging.getLogger(__name__)


class Tomography(object):
    def __init__(self, cfg, scene_cfg, pcd_file, output_file):
        self.output_file = Path(output_file)
        self.pcd_file = str(pcd_file)
        self.resolution = scene_cfg.map.resolution
        self.ground_h = scene_cfg.map.ground_h
        self.slice_dh = scene_cfg.map.slice_dh

        self.center = np.zeros(2, dtype=np.float32)
        self.tomogram = Tomogram(scene_cfg, self.output_file.parent / 'debug')
        points = self.loadPCD(self.pcd_file)

        # Process
        self.process(points)

    def loadPCD(self, pcd_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points).astype(np.float32)
        logger.info("PCD points: %d", points.shape[0])

        if points.shape[1] > 3:
            points = points[:, :3]
        self.points_max = np.max(points, axis=0)
        self.points_min = np.min(points, axis=0)           
        self.points_min[-1] = self.ground_h
        self.map_dim_x = int(np.ceil((self.points_max[0] - self.points_min[0]) / self.resolution)) + 4
        self.map_dim_y = int(np.ceil((self.points_max[1] - self.points_min[1]) / self.resolution)) + 4
        n_slice_init = int(np.ceil((self.points_max[2] - self.points_min[2]) / self.slice_dh))
        self.center = (self.points_max[:2] + self.points_min[:2]) / 2
        self.slice_h0 = self.points_min[-1] + self.slice_dh
        self.tomogram.initMappingEnv(self.center, self.map_dim_x, self.map_dim_y, n_slice_init, self.slice_h0)

        logger.info("Map center: [%.2f, %.2f]", self.center[0], self.center[1])
        logger.info("Dim_x: %d", self.map_dim_x)
        logger.info("Dim_y: %d", self.map_dim_y)
        logger.info("Num slices init: %d", n_slice_init)

        self.VISPROTO_I, self.VISPROTO_P = \
            GRID_POINTS_XYZI(self.resolution, self.map_dim_x, self.map_dim_y)

        return points
        
    def process(self, points):        
        t_map = 0.0
        t_trav = 0.0
        t_simp = 0.0
        t_all = 0.0
        n_repeat = 1

        """ 
        GPU time benchmark, where CUDA events are synchronized for correct time measurement.
        The function is repeatedly run for n_repeat times to calculate the average processing time of each modules.
        The time of the first warm-up run is excluded to reduce timing fluctuation and exclude the overhead in initial invocations.
        See https://docs.cupy.dev/en/stable/user_guide/performance.html for more details
        """
        for i in range(n_repeat + 1):
            t_start = time.time()
            layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, t_gpu = self.tomogram.point2map(points)

            if i > 0:
                t_map += t_gpu['t_map']
                t_trav += t_gpu['t_trav']
                t_simp += t_gpu['t_simp']
                t_all += (time.time() - t_start) * 1e3

        logger.info("Num slices simp: %d", layers_g.shape[0])
        logger.info("Num repeats (for benchmarking only): %d", n_repeat)
        logger.info(" -- avg t_map  (ms): %f", t_map / n_repeat)
        logger.info(" -- avg t_trav (ms): %f", t_trav / n_repeat)
        logger.info(" -- avg t_simp (ms): %f", t_simp / n_repeat)
        logger.info(" -- avg t_all  (ms): %f", t_all / n_repeat)

        self.n_slice = layers_g.shape[0]

        map_file = os.path.splitext(self.pcd_file)[0]
        self.exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)), map_file)


    def exportTomogram(self, tomogram, map_file):        
        data_dict = {
            'data': tomogram.astype(np.float16),
            'resolution': self.resolution,
            'center': self.center,
            'slice_h0': self.slice_h0,
            'slice_dh': self.slice_dh,
        }
        # [ROS2适配] 原 pickle 数据格式保持不变；原子替换避免 plan 节点读取未写完的地图。
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        file_name = str(self.output_file)
        temporary = self.output_file.with_suffix('.pickle.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        temporary.replace(self.output_file)
        logger.info("Tomogram exported: %s", file_name)

