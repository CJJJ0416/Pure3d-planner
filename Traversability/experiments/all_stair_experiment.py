"""从最新best.pt执行全类型楼梯数据增强、训练和八处楼梯回归。"""
from Traversability.experiments.wide_stair_experiment import main
from utils.pointcloud import ROOT


if __name__ == '__main__':
    main(ROOT / 'Traversability/configs/all_stairs_dr_v3.json')
